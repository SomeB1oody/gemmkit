//! Mixed-precision (f16/bf16) GEMM: shapes x layouts, gemv routes, an oracle
//! cross-check against the `gemm` crate, and parallel/serial bit-identity

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Shared shape list for the f16/bf16 accuracy and bit-identity tests below
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

/// f16 over the [`mixed_dims`] shapes, row-major A, col-major B, both C layouts, and an
/// alpha/beta sweep including the alpha=0 scale-only case, against the f64 reference
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

/// bf16 twin of [`correctness_f16_layouts`], with A/B layouts swapped (col-major A,
/// row-major B)
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

/// f16/bf16 serial and parallel runs land on identical bits, for every thread count:
/// the same `float.rs`'s `parallel_equals_serial_bit_identical` caveat applies (holds
/// because narrowing to f16/bf16 is a pure per-position function of the f32 result and
/// blocking is thread-independent, an implementation property, not a promised contract).
/// `bf16` here runs the dot (`vdpbf16ps`) kernel rather than f16's widen-FMA one, but
/// dispatch picks the same kernel for serial and parallel, so the 2 still coincide bitwise
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

/// f16/bf16 gemv shapes (`m == 1` or `n == 1`) through the mixed-precision gemv core,
/// checked against the f64 oracle. Row-major A pairs with a unit-stride vector to hit
/// `dot_rows_mixed`, col-major A with col-major C hits `axpy_mixed`, and `GeneralPad` A
/// falls through to `strided_rows_mixed`; `k` ranges from a sub-lane scalar tail up
/// through several SIMD widths
#[cfg(feature = "half")]
#[test]
fn correctness_mixed_gemv() {
    fn check<T: Elem>() {
        // Each shape has m == 1 or n == 1, so it dispatches as a gemv; k varies across
        // the SIMD-lane/scalar-tail boundary
        let shapes = [
            (1usize, 40usize, 1usize), // a bare dot product
            (64, 64, 1),
            (129, 200, 1),
            (1, 64, 17),
            (1, 200, 64),
            (1, 96, 129),
        ];
        for (m, k, n) in shapes {
            for &la in &[Layout::Row, Layout::Col, Layout::GeneralPad] {
                for &lc in &[Layout::Row, Layout::Col] {
                    for &(al, be) in &[(1.0f64, 0.0), (1.0, 1.0), (0.75, -0.5), (0.0, 2.5)] {
                        run_case::<T>(
                            m,
                            k,
                            n,
                            la,
                            Layout::Col,
                            lc,
                            T::from_f64(al),
                            T::from_f64(be),
                            Parallelism::Serial,
                        );
                    }
                }
            }
        }
    }
    check::<gemmkit::f16>();
    check::<gemmkit::bf16>();
}

/// The mixed-precision gemv core splits work by output row, never by `k`, so each output
/// element is always the single reduction of 1 worker: serial and parallel must land on
/// identical bits for every thread count, the mixed twin of the float gemv partition's
/// same guarantee. Covers the dot layout (row-major A) and the axpy layout (col-major A,
/// col-major C), for both `n == 1` and `m == 1`. The gemv parallel byte floor is forced to
/// 1 (RAII-restored on drop) so the row partition engages at these sizes on any machine,
/// rather than depending on the shape clearing an L3-derived floor
#[cfg(feature = "half")]
#[test]
fn parallel_equals_serial_mixed_gemv() {
    struct RestoreFloor(usize);
    impl Drop for RestoreFloor {
        fn drop(&mut self) {
            gemmkit::tuning::set_gemv_parallel_bytes(self.0);
        }
    }
    let _restore = RestoreFloor(gemmkit::tuning::gemv_parallel_bytes());
    gemmkit::tuning::set_gemv_parallel_bytes(1); // any shape below now clears the floor

    fn check<T: Elem>(la: Layout, label: &str) {
        // n == 1 (mat*vec) and m == 1 (vec*mat), large enough to split across several workers
        for (m, k, n) in [
            (4096usize, 512usize, 1usize),
            (1, 512, 4096),
            (2048, 320, 1),
        ] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                let a = Mat::<T>::rand(m, k, 0x6E7 + (m * 3 + k) as u64);
                let b = Mat::<T>::rand(k, n, 0x11D + (n * 5 + k) as u64);
                let c0 = Mat::<T>::rand(m, n, 0x03C + (m + n) as u64);
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
                        "mixed gemv {label} serial != parallel({t}) for {m}x{k}x{n}"
                    );
                }
            }
        }
    }
    check::<gemmkit::f16>(Layout::Row, "f16/dot"); // routes through dot_rows_mixed
    check::<gemmkit::f16>(Layout::Col, "f16/axpy"); // routes through axpy_mixed
    check::<gemmkit::bf16>(Layout::Row, "bf16/dot");
    check::<gemmkit::bf16>(Layout::Col, "bf16/axpy");
}

/// Cross-check f16 against the `gemm` crate, which also accumulates f16 in f32: `gemm`'s
/// f16 is `half::f16` is `gemmkit::f16`, so the 2 outputs must agree to a tight f16
/// tolerance. Gated the same as the `gemm` dev-dependency itself
/// (`cfg(all(not(miri), not(target_family = "wasm")))`)
#[test]
#[cfg(all(not(miri), not(target_family = "wasm"), feature = "half"))]
fn mixed_f16_matches_gemm_crate() {
    // k=2048 exceeds the f32 kc default (512), which would split a homogeneous f32
    // kernel's contraction across several kc panels; the mixed family is `OUT_IS_ACC
    // = false`, so the driver forces a single kc = k panel instead, accumulating the
    // whole contraction in f32 and rounding to f16 exactly once, matching how `gemm`
    // accumulates its own f16 output
    for (m, k, n) in [(64, 48, 40), (96, 65, 72), (33, 17, 19), (64, 2048, 64)] {
        let a = Mat::<gemmkit::f16>::rand(m, k, 0x16A + m as u64);
        let b = Mat::<gemmkit::f16>::rand(k, n, 0x16B + n as u64);
        // Column-major, the orientation gemm::gemm below expects; beta is 0
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
        // gemm::gemm's column-major convention: cs is the leading dimension, rs is 1;
        // read_dst = false since beta is 0
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
        // Both round f32 to f16 once; a few f16 ULPs of slack cover the differing
        // accumulation order. assert_accurate wants a row-major reference, so
        // transpose the column-major c_gemm into one
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
