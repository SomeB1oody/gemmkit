//! Deep-k narrow-twin parity under a forced **AVX-512** kernel (its own process, so the memoized
//! dispatch is the widen `MixedGemm` for both `f16` and `bf16` - the bf16 `vdpbf16ps` dot path is
//! only picked when `avx512bf16` is present and unforced). This pins the widen twin `MixedGemmF32`
//! for `bf16`, which the auto-selecting `tests/deep_k_narrow.rs` never reaches on an AVX-512-BF16
//! box. Asserts the deep-k route is byte-for-byte the single depth panel for `beta in {0, 1}`.
//! Skips gracefully when the host lacks `avx512f`
#![cfg(all(
    feature = "half",
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

use gemmkit::{MatMut, MatRef, Parallelism, bf16, f16, gemm, tuning};

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

#[test]
fn deep_k_widen_under_avx512_pin() {
    // Pin once, before any dispatch. SAFETY: the only test in this binary
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "avx512");
    }
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    let restore = tuning::deep_kc_bytes();
    check::<f16>("f16 (MixedGemm)");
    check::<bf16>("bf16 (MixedGemm widen)");
    tuning::set_deep_kc_bytes(restore);
}
