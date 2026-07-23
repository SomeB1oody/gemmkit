//! `GEMMKIT_REQUIRE_ISA=avx512f` pin: forces the plain (widen) AVX-512F kernels, which
//! auto-selection skips in favor of VNNI/BF16 on a host that reports those extensions, so this is
//! the only place their dispatch code actually runs
//!
//! Every test wants the same `avx512f` pin, so all funnel through [`env_isa_common::pin`] (a single
//! `set_var` under a `Once`, before any dispatch; see that module for why the shared write is
//! sound). The tests are otherwise independent, each touching a different knob (RHS-pack
//! threshold, deep-kc gate) and dtype, so none depends on another's knob state. Every test skips
//! when the host lacks `avx512f`, since forcing the pin there would abort in `select_*` instead of
//! testing anything
#![cfg(all(
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared GEMMKIT_REQUIRE_ISA pin helper; every test here pins `avx512f` with it
#[cfg(any(feature = "int8", feature = "half"))]
mod env_isa_common;
// Shared prepacked-vs-plain i8 bit-parity check, run under whichever ISA is pinned
#[cfg(feature = "int8")]
mod i8_packed_common;

#[cfg(any(feature = "int8", feature = "half"))]
use gemmkit::{MatMut, MatRef, Parallelism, tuning};
#[cfg(feature = "half")]
use gemmkit::{bf16, f16, gemm};

// IntGemm::pack_rhs is the widen (non-VNNI) i8 kernel's RHS packer. On a VNNI-capable host
// auto-selection always picks the vpdpbusd dot kernel instead, which packs through
// IntGemmVnni::pack_rhs, so the widen packer only runs once avx512f is forced. Also lowers the
// RHS-pack threshold from its 2048 default to 1, so a small m still triggers packing without
// needing a large (and slower) matrix

#[cfg(feature = "int8")]
#[test]
fn widen_i8_pack_rhs_under_avx512f_pin() {
    env_isa_common::pin("avx512f");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    // Below this shape's m, so pack_b (= m > threshold) is true regardless of shape size
    tuning::set_rhs_pack_threshold(1);

    // Clears both the small_k and small_mn gates, landing on the general driver where RHS
    // packing runs. Values stay well inside i8 range, so wrapping never actually triggers
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

    // Reference: naive wrapping-i32 triple loop
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

// The widen kernel's plain-panel RHS prepack (DEPTH_MULTIPLE = 1) must match plain gemm_i8

#[cfg(feature = "int8")]
#[test]
fn avx512f_pin_i8_packed_matches_plain() {
    env_isa_common::pin("avx512f");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    i8_packed_common::check();
}

// Under this pin both f16 and bf16 dispatch to the widen MixedGemm family (bf16's own
// vdpbf16ps dot path only gets picked when avx512bf16 is present and unforced), so pinning
// avx512f is what makes the widen deep-k twin MixedGemmF32<bf16> reachable at all: on an
// AVX-512 BF16 host, tests/deep_k_narrow.rs's auto-selection never lands on it

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

/// The deep-k route (gate `1`) must be byte-for-byte the single depth panel (gate `MAX`) for
/// `beta in {0, 1}` on a shape mixing full and edge tiles, and serial must equal parallel
#[cfg(feature = "half")]
fn check<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(label: &str) {
    let (m, n, k) = (40usize, 50, 4096);
    let a: Vec<N> = fill(m * k, 0x1);
    let b: Vec<N> = fill(k * n, 0x2);
    let c0: Vec<N> = fill(m * n, 0x3);
    // Only beta == 0 (overwrite) and beta == 1 (accumulate) are bit-identity cases
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
            // SAFETY: N is f16 or bf16, both 2-byte transparent-over-u16
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
fn deep_k_widen_under_avx512f_pin() {
    env_isa_common::pin("avx512f");
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    let restore = tuning::deep_kc_bytes();
    check::<f16>("f16 (MixedGemm)");
    check::<bf16>("bf16 (MixedGemm widen)");
    tuning::set_deep_kc_bytes(restore);
}
