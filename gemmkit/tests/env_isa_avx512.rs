//! `GEMMKIT_REQUIRE_ISA=avx512` pin: forces the plain (widen) AVX-512 kernels the auto path never
//! takes on this VNNI/BF16 box, so their memoized-dispatch-only code lives in an isolated process.
//! Every test here wants the same `avx512` pin and funnels through [`env_isa_common::pin`], which
//! does the single `set_var` under a `Once` before any dispatch (see that module for the soundness
//! argument); the shared write overrides an inherited `GEMMKIT_REQUIRE_ISA` so the SDE/pinned CI
//! jobs still exercise these routes. The tests are otherwise independent: they touch different knobs
//! (rhs-pack threshold, deep-kc gate) and different dtypes, and each asserts a self-contained
//! parity that is invariant to the others' knob state. Every test skips gracefully when the host
//! lacks `avx512f` (the pin would otherwise assert in `select_*`)
#![cfg(all(
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared single-set_var pin helper (Once before any dispatch; all tests here pin `avx512`)
#[cfg(any(feature = "int8", feature = "half"))]
mod env_isa_common;
// Shared prepacked-i8 bit-parity check driven by the ISA pin binaries
#[cfg(feature = "int8")]
mod i8_packed_common;

#[cfg(any(feature = "int8", feature = "half"))]
use gemmkit::{MatMut, MatRef, Parallelism, tuning};
#[cfg(feature = "half")]
use gemmkit::{bf16, f16, gemm};

// `IntGemm::pack_rhs`: the widen (non-VNNI) i8 kernel's RHS packer. On this AVX-512-VNNI box the
// auto i8 path always picks the `vpdpbusd` dot kernel (which packs via `IntGemmVnni::pack_rhs`), so
// the widen packer is only reachable when the plain AVX-512 kernel is forced. Pinning `avx512`
// memoizes `select_i8` to the widen kernel, and lowering the RHS-pack threshold to `1` makes a
// small `m` still trigger packing, more robust than the `m > 2048` default since it forces the
// widen packer directly and keeps the matrix tiny/fast.

#[cfg(feature = "int8")]
#[test]
fn widen_i8_pack_rhs_under_avx512_pin() {
    env_isa_common::pin("avx512");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    // Force RHS packing for the widen kernel even at small `m` (`pack_b = m > threshold`)
    tuning::set_rhs_pack_threshold(1);

    // Driver path (k > small_k gate, m,n > small_mn gate), so RHS packing runs through
    // `IntGemm::pack_rhs`. Values stay in range so the wrapping semantics never fire
    let (m, k, n) = (96usize, 64usize, 64usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i % 13) as i32 - 6) as i8).collect();
    let c0: Vec<i32> = (0..m * n).map(|i| (i % 7) as i32 - 3).collect();
    let (alpha, beta) = (2i32, -1i32);

    let mut c = c0.clone();
    gemmkit::gemm_i8(
        alpha,
        MatRef::from_row_major(&a, m, k),
        MatRef::from_row_major(&b, k, n),
        beta,
        MatMut::from_row_major(&mut c, m, n),
        Parallelism::Serial,
    );

    // Naive wrapping-i32 reference
    let mut expect = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32).wrapping_mul(b[p * n + j] as i32));
            }
            expect[i * n + j] = alpha
                .wrapping_mul(acc)
                .wrapping_add(beta.wrapping_mul(c0[i * n + j]));
        }
    }
    assert_eq!(
        c, expect,
        "widen i8 kernel with RHS packing must match the reference"
    );
}

// Forces the **widen** AVX-512 integer kernel (plain-panel RHS prepack, `DEPTH_MULTIPLE = 1`), the
// path an auto VNNI box never takes, and checks `gemm_i8_packed_b` is bit-identical to plain
// `gemm_i8` across the shared shape/layout/alpha-beta/thread sweep

#[cfg(feature = "int8")]
#[test]
fn avx512_pin_i8_packed_matches_plain() {
    env_isa_common::pin("avx512");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    i8_packed_common::check();
}

// Deep-k narrow-twin parity under the forced widen `MixedGemm` (for both `f16` and `bf16` - the
// bf16 `vdpbf16ps` dot path is only picked when `avx512bf16` is present and unforced). Pinning
// `avx512` memoizes the widen twin `MixedGemmF32` for `bf16`, which the auto-selecting
// `tests/deep_k_narrow.rs` never reaches on an AVX-512-BF16 box. The deep-k route must be
// byte-for-byte the single depth panel for `beta in {0, 1}`, and serial == parallel

#[cfg(feature = "half")]
fn fill<N: gemmkit::NarrowFloat>(n: usize, seed: u64) -> Vec<N> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            N::narrow((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        })
        .collect()
}

/// Deep-k (gate `1`) vs single panel (gate `MAX`) must be bit-identical for `beta in {0, 1}` on a
/// full+edge-tile shape; also serial == parallel for the deep-k route
#[cfg(feature = "half")]
fn check<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(label: &str) {
    let (m, n, k) = (40usize, 50, 4096);
    let a: Vec<N> = fill(m * k, 0x1);
    let b: Vec<N> = fill(k * n, 0x2);
    let c0: Vec<N> = fill(m * n, 0x3);
    // beta == 0 and beta == 1 (accumulate arm) are the bit-identity cases
    for &beta in &[0.0f32, 1.0] {
        let run = |engage: bool, par| -> Vec<u16> {
            tuning::set_deep_kc_bytes(if engage { 1 } else { usize::MAX });
            let mut c = c0.clone();
            gemm(
                N::narrow(1.25),
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                N::narrow(beta),
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
            // SAFETY: N is f16/bf16, 2-byte transparent-over-u16
            c.iter()
                .map(|v| unsafe { core::mem::transmute_copy::<N, u16>(v) })
                .collect()
        };
        let single = run(false, Parallelism::Serial);
        let deep_ser = run(true, Parallelism::Serial);
        let deep_par = run(true, Parallelism::Rayon(0));
        assert_eq!(
            deep_ser, single,
            "{label}: deep-k != single panel (beta={beta})"
        );
        assert_eq!(
            deep_par, deep_ser,
            "{label}: deep-k serial != parallel (beta={beta})"
        );
    }
}

#[cfg(feature = "half")]
#[test]
fn deep_k_widen_under_avx512_pin() {
    env_isa_common::pin("avx512");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    let restore = tuning::deep_kc_bytes();
    check::<f16>("f16 (MixedGemm)");
    check::<bf16>("bf16 (MixedGemm widen)");
    tuning::set_deep_kc_bytes(restore);
}
