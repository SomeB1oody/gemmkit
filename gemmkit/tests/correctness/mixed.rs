//! Mixed-precision (f16/bf16) shapes x layouts, oracle cross-check against the
//! `gemm` crate, and parallel/serial bit-identity.

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Shapes for the mixed-precision (`f16`/`bf16`) accuracy and bit-identity tests.
#[cfg(feature = "half")]
fn mixed_dims() -> [(usize, usize, usize); 9] {
    [
        (1, 1, 1),
        (3, 4, 5),
        (16, 8, 7),
        (32, 32, 32),
        (33, 17, 19),
        (40, 33, 28),
        (64, 80, 48),
        (65, 64, 64),
        (128, 96, 112),
    ]
}

#[cfg(feature = "half")]
#[test]
fn correctness_f16_layouts() {
    for (m, k, n) in mixed_dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            for &(al, be) in &[(1.0f64, 0.0), (1.0, 1.0), (0.75, -0.5), (0.0, 2.5)] {
                run_case::<gemmkit::f16>(
                    m,
                    k,
                    n,
                    Layout::Row,
                    Layout::Col,
                    lc,
                    gemmkit::f16::from_f64(al),
                    gemmkit::f16::from_f64(be),
                    Parallelism::Serial,
                );
            }
        }
    }
}

#[cfg(feature = "half")]
#[test]
fn correctness_bf16_layouts() {
    for (m, k, n) in mixed_dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            for &(al, be) in &[(1.0f64, 0.0), (1.0, 1.0), (0.75, -0.5), (0.0, 2.5)] {
                run_case::<gemmkit::bf16>(
                    m,
                    k,
                    n,
                    Layout::Col,
                    Layout::Row,
                    lc,
                    gemmkit::bf16::from_f64(al),
                    gemmkit::bf16::from_f64(be),
                    Parallelism::Serial,
                );
            }
        }
    }
}

/// Mixed-precision (`f16`/`bf16`) serial == parallel bit-identity across thread counts —
/// same caveat as `parallel_equals_serial_bit_identical`: it holds because narrowing is a
/// pure per-position function of the f32 result *and* blocking is thread-independent, an
/// implementation property, not a promised contract. (`bf16` here is the **dot** kernel,
/// whose serial and parallel paths share one kernel + layout, so they coincide bitwise
/// even though dot ≠ widen.) Pins current behavior; relax with split-K, as noted there.
#[cfg(feature = "half")]
#[test]
fn parallel_equals_serial_mixed() {
    fn check<T: Elem>(la: Layout) {
        for (m, k, n) in [(200, 130, 175), (256, 64, 200), (384, 96, 320)] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                let a = Mat::<T>::rand(m, k, 0xF16 + m as u64);
                let b = Mat::<T>::rand(k, n, 0xBF + n as u64);
                let c0 = Mat::<T>::rand(m, n, 0xCD + k as u64);
                let (abuf, rsa, csa) = build_view(&a, la);
                let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
                let (al, be) = (T::from_f64(al), T::from_f64(be));

                let mut c_ser = cbase.clone();
                gemm(
                    al,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be,
                    MatMut::new(&mut c_ser, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                for t in [2usize, 4, 8, 16] {
                    let mut c_par = cbase.clone();
                    gemm(
                        al,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        MatRef::new(&bbuf, k, n, rsb, csb),
                        be,
                        MatMut::new(&mut c_par, m, n, rsc, csc),
                        Parallelism::Rayon(t),
                    );
                    assert!(
                        c_ser
                            .iter()
                            .zip(&c_par)
                            .all(|(a, b)| a.to_f64().to_bits() == b.to_f64().to_bits()),
                        "mixed serial != parallel({t}) for {m}x{k}x{n}"
                    );
                }
            }
        }
    }
    check::<gemmkit::f16>(Layout::Row);
    check::<gemmkit::bf16>(Layout::Col);
}

/// Cross-check `f16` against the `gemm` crate (the ecosystem oracle, which also
/// accumulates `f16` in `f32`): the two must agree to a tight `f16` tolerance.
/// `gemm`'s `f16` *is* `half::f16` *is* `gemmkit::f16`, so the comparison is direct.
/// Gated out of Miri and wasm (the `gemm` dev-dep is `cfg(all(not(miri), not(wasm)))`).
#[test]
#[cfg(all(not(miri), not(target_family = "wasm"), feature = "half"))]
fn mixed_f16_matches_gemm_crate() {
    // Includes a large-k case (k = 2048 > the f32 kc blocking ≈ 512) to exercise the
    // mixed-precision single-panel rule: because `Out` (f16) is narrower than `Acc`
    // (f32), the driver forces `kc = k` (`OUT_IS_ACC = false`), so the whole
    // contraction accumulates in f32 and rounds to f16 exactly ONCE at writeback —
    // never between kc panels. (A homogeneous f32 kernel would instead split this k.)
    // This is what keeps gemmkit matching `gemm`'s whole-k f32 accumulation.
    for (m, k, n) in [(64, 48, 40), (96, 65, 72), (33, 17, 19), (64, 2048, 64)] {
        let a = Mat::<gemmkit::f16>::rand(m, k, 0x16A + m as u64);
        let b = Mat::<gemmkit::f16>::rand(k, n, 0x16B + n as u64);
        // Column-major buffers (gemm's preferred orientation), zero beta.
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let mut c_kit = vec![gemmkit::f16::from_f64(0.0); m * n];
        let mut c_gemm = vec![gemmkit::f16::from_f64(0.0); m * n];

        gemm(
            gemmkit::f16::from_f64(1.0),
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            gemmkit::f16::from_f64(0.0),
            MatMut::from_col_major(&mut c_kit, m, n),
            Parallelism::Serial,
        );
        // gemm crate: column-major operands → (cs = leading dim, rs = 1), matching
        // the bench harness; read_dst=false (beta=0).
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c_gemm.as_mut_ptr(),
                m as isize,
                1,
                false,
                abuf.as_ptr(),
                m as isize,
                1,
                bbuf.as_ptr(),
                k as isize,
                1,
                gemmkit::f16::from_f64(0.0),
                gemmkit::f16::from_f64(1.0),
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        }
        // Both accumulate in f32 then round to f16; allow a few f16 ULPs of slack
        // (the accumulation order differs). `assert_accurate` wants a *row-major*
        // reference, so transpose the column-major `c_gemm` into one.
        let mut cref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                cref[i * n + j] = c_gemm[i + j * m].to_f64();
            }
        }
        assert_accurate(
            &c_kit,
            1,
            m as isize,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "f16 vs gemm crate",
        );
    }
}
