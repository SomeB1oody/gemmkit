//! Correctness tests for the prepacked-LHS/RHS API surface: bit-identical (or, on
//! both-tiny/gemv shapes, accurate-and-deterministic) agreement with plain `gemm`/`gemm_i8`,
//! the raw `_unchecked` pointer entries, and the C-orientation panics each packed path enforces

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Prepacked-RHS (`prepack_rhs` + `gemm_packed_b`) must be bit-identical to plain `gemm` on
/// the same inputs, across every B layout, alpha/beta pair, and thread count: packing only
/// rearranges B's values into the driver's own micropanel layout, so the microkernel runs
/// the identical FMAs in the identical order either way. A and C stay column-major
/// throughout, the only orientation a prepacked RHS can serve
#[test]
fn prepack_equals_gemm() {
    // Every shape clears the both-tiny gate (m <= 64 && n <= 64), which the separate
    // accuracy test below covers instead
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // m=65 is just past the tiny-shortcut gate (m<=64)
        (64, 64, 65), // n=65 is just past the tiny-shortcut gate (n<=64)
        (300, 1, 256),
        (40, 200, 300),
    ] {
        for &lb in &[Layout::Col, Layout::Row] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3), (-0.5, 2.0)] {
                let a = Mat::<f32>::rand(m, k, 0x5A + (m * 7 + n) as u64);
                let b = Mat::<f32>::rand(k, n, 0x6B + (n * 3 + k) as u64);
                let c0 = Mat::<f32>::rand(m, n, 0x7C + (k + m) as u64);
                let (abuf, rsa, csa) = build_view(&a, Layout::Col);
                let (bbuf, rsb, csb) = build_view(&b, lb);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );

                let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
                assert_eq!(packed.rows(), k);
                assert_eq!(packed.cols(), n);
                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(4),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_b(
                        al as f32,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        &packed,
                        be as f32,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack != gemm for {m}x{k}x{n} lb={lb:?} a={al} b={be} par={par:?}"
                    );
                }
            }
        }
    }
}

/// The raw `_unchecked` entries (`prepack_rhs_unchecked` + `gemm_packed_b_unchecked` for the
/// RHS pack, `prepack_lhs_unchecked` + `gemm_packed_a_unchecked` for the LHS pack) must match
/// plain `gemm`, exercised directly through raw pointers and strides instead of `MatRef`/
/// `MatMut`. The checked API (`prepack_rhs`/`gemm_packed_b`, etc.) is a thin bounds-check
/// wrapper around these, so this also stands in for the FFI/adapter-facing contract
#[test]
fn packed_unchecked_matches_gemm() {
    for (m, k, n) in [(200usize, 130, 175), (65, 64, 64), (40, 200, 300)] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0x11 + (m + n) as u64);
            let b = Mat::<f32>::rand(k, n, 0x22 + (k + n) as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x33 + (m + k) as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Col);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);

            // gemm_packed_b_unchecked requires column-major-ish C
            {
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                // SAFETY: bbuf addresses (k, n) at (rsb, csb) in bounds
                let packed =
                    unsafe { gemmkit::prepack_rhs_unchecked(bbuf.as_ptr(), rsb, csb, k, n) };
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c = cbase.clone();
                    // SAFETY: abuf/c address (m, k)/(m, n) at (rsa,csa)/(rsc,csc) in bounds;
                    // c does not alias abuf
                    unsafe {
                        gemmkit::gemm_packed_b_unchecked(
                            al as f32,
                            m,
                            abuf.as_ptr(),
                            rsa,
                            csa,
                            &packed,
                            be as f32,
                            c.as_mut_ptr(),
                            rsc,
                            csc,
                            par,
                        );
                    }
                    assert_eq!(
                        c_ref, c,
                        "packed_b_unchecked != gemm {m}x{k}x{n} a={al} b={be} par={par:?}"
                    );
                }
            }

            // gemm_packed_a_unchecked requires row-major-ish C
            {
                let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                // SAFETY: abuf addresses (m, k) at (rsa, csa) in bounds
                let packed =
                    unsafe { gemmkit::prepack_lhs_unchecked(abuf.as_ptr(), rsa, csa, m, k) };
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c = cbase.clone();
                    // SAFETY: bbuf/c address (k, n)/(m, n) at (rsb,csb)/(rsc,csc) in bounds;
                    // c does not alias bbuf
                    unsafe {
                        gemmkit::gemm_packed_a_unchecked(
                            al as f32,
                            &packed,
                            n,
                            bbuf.as_ptr(),
                            rsb,
                            csb,
                            be as f32,
                            c.as_mut_ptr(),
                            rsc,
                            csc,
                            par,
                        );
                    }
                    assert_eq!(
                        c_ref, c,
                        "packed_a_unchecked != gemm {m}x{k}x{n} a={al} b={be} par={par:?}"
                    );
                }
            }
        }
    }
}

/// Mixed-precision (`f16`/`bf16`) prepacked-RHS must be bit-identical to plain `gemm` too.
/// Both `prepack_rhs` and the driver derive `kc` from `GemmScalar::OUT_IS_ACC`, `false` for
/// these narrow-output types, so both always block the whole contraction as a single depth
/// panel (`kc = k`) instead of the general cache-model `kc`: the packed buffer's geometry can
/// never disagree with what the consuming kernel expects. `(128, 1024, 96)` pushes `k` past
/// the default 512-element cache-model `kc`, where a homogeneous `f32`/`f64` family would
/// split into multiple depth slices but this single-panel rule leaves it undivided
#[cfg(feature = "half")]
#[test]
fn prepack_equals_gemm_mixed() {
    fn check<T: Elem>() {
        for (m, k, n) in [(200, 130, 175), (96, 65, 72), (128, 1024, 96)] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                let a = Mat::<T>::rand(m, k, 0x9A + (m * 3 + n) as u64);
                let b = Mat::<T>::rand(k, n, 0x9B + (n + k) as u64);
                let c0 = Mat::<T>::rand(m, n, 0x9C + (k + m) as u64);
                let (abuf, rsa, csa) = build_view(&a, Layout::Col);
                let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
                let (al, be) = (T::from_f64(al), T::from_f64(be));

                let mut c_ref = cbase.clone();
                gemm(
                    al,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_b(
                        al,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        &packed,
                        be,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert!(
                        c_ref
                            .iter()
                            .zip(&c_pk)
                            .all(|(a, b)| a.to_f64().to_bits() == b.to_f64().to_bits()),
                        "mixed prepack != gemm for {m}x{k}x{n} par={par:?}"
                    );
                }
            }
        }
    }
    check::<gemmkit::f16>();
    check::<gemmkit::bf16>();
}

/// Prepacked-i8 RHS (`prepack_rhs_i8` + `gemm_i8_packed_b`) must be bit-identical to plain
/// `gemm_i8` on the same inputs, across B's layout, alpha/beta, and thread count. Wrapping
/// i32 addition is associative, so the sum does not depend on which family, or how many
/// depth/n panels, computes it: this holds across the auto-dispatched VNNI `vpdpbusd` kernel
/// and its widen fallback too. Under multi-threading, the plain path reroutes an auto-VNNI
/// selection to the widen kernel once the problem volume (`m*n*k`) drops below a threshold
/// (VNNI's mandatory RHS pack outweighs the compute it saves there), but the prepacked path
/// always runs the buffer's own packed family regardless, and the 2 stay bit-identical either
/// way. Shapes cover k not a multiple of 4 (the VNNI depth pad), k == 1, and a small-m/long-k
/// fixed-weight inference shape
#[cfg(feature = "int8")]
#[test]
fn prepack_i8_equals_gemm_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [
        (200usize, 130, 175),
        (128, 96, 112),
        (65, 64, 64),
        (64, 65, 100), // k=65 not a multiple of 4 (the VNNI depth pad)
        (96, 129, 129),
        (33, 17, 19),
        (256, 257, 129),
        (300, 1, 256), // k == 1, below the plain path's small-k reroute
        (8, 2048, 96), // small m, long k: the fixed-weight inference shape
        (2, 1023, 12), // tiny m, k=1023 not a multiple of 4
    ] {
        let a = rand_i8(m * k, 0x51 + (m * 7 + n) as u64);
        let b_rm = rand_i8(k * n, 0x62 + (n * 3 + k) as u64); // logical k x n, row-major
        // column-major copy of the same logical B, for the lb==1 case below
        let mut b_cm = vec![0i8; k * n];
        for i in 0..k {
            for j in 0..n {
                b_cm[j * k + i] = b_rm[i * n + j];
            }
        }
        let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();

        for lb in 0..2 {
            // lb 0: row-major B (rsb=n, csb=1); lb 1: column-major B (rsb=1, csb=k)
            let (bbuf, rsb, csb): (&[i8], isize, isize) = if lb == 0 {
                (&b_rm, n as isize, 1)
            } else {
                (&b_cm, 1, k as isize)
            };
            let bview = MatRef::new(bbuf, k, n, rsb, csb);
            let packed = gemmkit::prepack_rhs_i8(bview);
            assert_eq!(packed.rows(), k);
            assert_eq!(packed.cols(), n);

            for &(alpha, beta) in &[(1i32, 0i32), (2, 3), (3, -2), (0, 5)] {
                let mut c_ref = c0.clone();
                gemmkit::gemm_i8(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    bview,
                    beta,
                    MatMut::from_col_major(&mut c_ref, m, n),
                    Parallelism::Serial,
                );
                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(4),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = c0.clone();
                    gemmkit::gemm_i8_packed_b(
                        alpha,
                        MatRef::from_col_major(&a, m, k),
                        &packed,
                        beta,
                        MatMut::from_col_major(&mut c_pk, m, n),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack_i8 != gemm_i8 for {m}x{k}x{n} lb={lb} a={alpha} b={beta} par={par:?}"
                    );
                }
            }
        }
    }
}

/// The raw `prepack_rhs_i8_unchecked` + `gemm_i8_packed_b_unchecked` entries must match plain
/// `gemm_i8`: the i8 mirror of `packed_unchecked_matches_gemm`, exercised through raw
/// pointers and strides. The checked `prepack_rhs_i8`/`gemm_i8_packed_b` are thin
/// bounds-check wrappers around these
#[cfg(feature = "int8")]
#[test]
fn packed_i8_unchecked_matches_gemm_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [(200usize, 130, 175), (65, 64, 64), (8, 512, 96)] {
        let a = rand_i8(m * k, 0x71 + (m + n) as u64);
        let b = rand_i8(k * n, 0x82 + (k + n) as u64); // B stored column-major (rsb=1, csb=k)
        let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 5) - 2).collect();
        for &(alpha, beta) in &[(1i32, 0i32), (2, 3)] {
            let mut c_ref = c0.clone();
            gemmkit::gemm_i8(
                alpha,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                beta,
                MatMut::from_col_major(&mut c_ref, m, n),
                Parallelism::Serial,
            );
            // SAFETY: b addresses (k, n) at (rsb=1, csb=k) in bounds
            let packed =
                unsafe { gemmkit::prepack_rhs_i8_unchecked(b.as_ptr(), 1, k as isize, k, n) };
            for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                let mut c = c0.clone();
                // SAFETY: a/c address (m, k)/(m, n) column-major-strided in bounds; c does
                // not alias a
                unsafe {
                    gemmkit::gemm_i8_packed_b_unchecked(
                        alpha,
                        m,
                        a.as_ptr(),
                        1,
                        m as isize,
                        &packed,
                        beta,
                        c.as_mut_ptr(),
                        1,
                        m as isize,
                        par,
                    );
                }
                assert_eq!(
                    c_ref, c,
                    "packed_i8_unchecked != gemm_i8 {m}x{k}x{n} a={alpha} b={beta} par={par:?}"
                );
            }
        }
    }
}

/// `gemm_i8_packed_b` rejects a row-major-ish C: that orientation would force an A/B swap,
/// which a prepacked B (packed as the genuine RHS) cannot support. Same panic wording as the
/// float packed path
#[cfg(feature = "int8")]
#[test]
#[should_panic(expected = "column-major-ish C")]
fn prepack_i8_row_major_c_panics() {
    use gemmkit::{MatMut, MatRef};
    let (m, k, n) = (100, 80, 120);
    let a = vec![0i8; m * k];
    let b = vec![0i8; k * n];
    let mut c = vec![0i32; m * n];
    let packed = gemmkit::prepack_rhs_i8(MatRef::from_col_major(&b, k, n));
    gemmkit::gemm_i8_packed_b(
        1,
        MatRef::from_col_major(&a, m, k),
        &packed,
        0,
        MatMut::from_row_major(&mut c, m, n), // row-major C forces the A/B swap this rejects
        Parallelism::Serial,
    );
}

/// `prepack_rhs_i8` on a `k == 0` B returns the empty-buffer sentinel (the geometry
/// arithmetic never runs), and a matching `k == 0` A through `gemm_i8_packed_b` then only
/// performs `C <- beta*C`
#[cfg(feature = "int8")]
#[test]
fn prepack_i8_empty_roundtrips() {
    use gemmkit::{MatMut, MatRef};
    let packed = gemmkit::prepack_rhs_i8(MatRef::new(&[], 0, 4, 1, 1));
    assert_eq!(packed.rows(), 0);
    assert_eq!(packed.cols(), 4);
    let mut c = vec![2i32; 3 * 4];
    gemmkit::gemm_i8_packed_b(
        1,
        MatRef::new(&[], 3, 0, 1, 1),
        &packed,
        3, // beta
        MatMut::from_col_major(&mut c, 3, 4),
        Parallelism::Serial,
    );
    assert!(c.iter().all(|&x| x == 6), "k==0 packed_b must beta-scale C");
}

/// f64 prepacked-RHS path is bit-identical too, exercising the f64 tile and the packed
/// geometry for a 2nd element type
#[test]
fn prepack_equals_gemm_f64() {
    for (m, k, n) in [(160, 96, 208), (96, 65, 65)] {
        let a = Mat::<f64>::rand(m, k, 0x1234);
        let b = Mat::<f64>::rand(k, n, 0x5678);
        let c0 = Mat::<f64>::rand(m, n, 0x9abc);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

        let mut c_ref = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
        let mut c_pk = cbase.clone();
        gemmkit::gemm_packed_b(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            &packed,
            -0.3,
            MatMut::new(&mut c_pk, m, n, rsc, csc),
            Parallelism::Rayon(8),
        );
        assert_eq!(c_ref, c_pk, "f64 prepack != gemm for {m}x{k}x{n}");
    }
}

/// Both-tiny shapes (`m <= 64 && n <= 64`) still work through `prepack_rhs`/`gemm_packed_b`,
/// but not bit-identically to plain `gemm`: the prepack constructor deliberately dodges the
/// tiny-matrix shortcut (it must serve every future `m`), so it always blocks through the
/// general cache model, while plain `gemm` takes the dedicated small-matrix path with its own
/// (differently rounded) `kc`. This checks accuracy against the f64 reference instead, and
/// that output stays bit-identical across thread counts. `(60, 600, 60)` pushes `k` past the
/// tiny-shortcut's 512-element `kc` clamp, where the 2 blocking models diverge furthest
#[test]
fn prepack_both_tiny_accurate_and_deterministic() {
    for (m, k, n) in [(48, 40, 48), (60, 600, 60), (10, 9, 12)] {
        let a = Mat::<f32>::rand(m, k, 0x11 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0x22 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0x33 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
        let cref = reference(&a, &b, &c0, 1.0, 0.5);

        let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
        let mut c_ser = cbase.clone();
        gemmkit::gemm_packed_b(
            1.0,
            MatRef::new(&abuf, m, k, rsa, csa),
            &packed,
            0.5,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        assert_accurate(
            &c_ser,
            rsc,
            csc,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "both-tiny prepack",
        );
        for threads in [2usize, 8] {
            let mut c_par = cbase.clone();
            gemmkit::gemm_packed_b(
                1.0,
                MatRef::new(&abuf, m, k, rsa, csa),
                &packed,
                0.5,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(threads),
            );
            assert_eq!(
                c_ser, c_par,
                "both-tiny prepack serial != parallel({threads}) for {m}x{k}x{n}"
            );
        }
    }
}

/// Prepacked-LHS (`prepack_lhs` + `gemm_packed_a`), the mirror of `prepack_equals_gemm`:
/// reusing a prepacked A must be bit-identical to plain `gemm`, across A's layout,
/// alpha/beta, and thread count. C stays row-major throughout, the only orientation a
/// prepacked A (packed as the transposed product's RHS) can serve; B stays column-major.
/// Packing only rearranges A's values, so the microkernel still runs the identical FMAs in
/// the identical order
#[test]
fn prepack_lhs_equals_gemm() {
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // m=65 is just past the tiny-shortcut gate (m<=64)
        (64, 64, 65), // n=65 is just past the tiny-shortcut gate (n<=64)
        (300, 1, 256),
        (40, 200, 300),
    ] {
        for &la in &[Layout::Col, Layout::Row] {
            // The pack depends only on (m, k, la), not alpha/beta: build it once per
            // layout and reuse it across the alpha/beta sweep below
            let a = Mat::<f32>::rand(m, k, 0x5A + (m * 7 + n) as u64);
            let b = Mat::<f32>::rand(k, n, 0x6B + (n * 3 + k) as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x7C + (k + m) as u64);
            let (abuf, rsa, csa) = build_view(&a, la);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            // gemm_packed_a only serves row-major-ish C
            let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
            let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
            assert_eq!(packed.rows(), m);
            assert_eq!(packed.cols(), k);

            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3), (-0.5, 2.0)] {
                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );

                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(4),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_a(
                        al as f32,
                        &packed,
                        MatRef::new(&bbuf, k, n, rsb, csb),
                        be as f32,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack_lhs != gemm for {m}x{k}x{n} la={la:?} a={al} b={be} par={par:?}"
                    );
                }
            }
        }
    }
}

/// f64 prepacked-LHS path is bit-identical too, exercising the f64 tile and the packed
/// geometry for a 2nd element type
#[test]
fn prepack_lhs_equals_gemm_f64() {
    for (m, k, n) in [(160, 96, 208), (96, 65, 65)] {
        let a = Mat::<f64>::rand(m, k, 0x1234);
        let b = Mat::<f64>::rand(k, n, 0x5678);
        let c0 = Mat::<f64>::rand(m, n, 0x9abc);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Row);

        let mut c_ref = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
        let mut c_pk = cbase.clone();
        gemmkit::gemm_packed_a(
            0.9,
            &packed,
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_pk, m, n, rsc, csc),
            Parallelism::Rayon(8),
        );
        assert_eq!(c_ref, c_pk, "f64 prepack_lhs != gemm for {m}x{k}x{n}");
    }
}

/// Mixed-precision (`f16`/`bf16`) prepacked-LHS must be bit-identical to plain `gemm` too:
/// the narrow-type twin of `prepack_lhs_equals_gemm`, run through the same LHS-as-transposed-
/// RHS pack `prepack_lhs_unchecked` delegates to. This is the only test in the file that
/// combines a narrow type, the transpose-pack path, and the `kc = k` single-panel rule
/// `prepack_equals_gemm_mixed` exercises on the RHS side. Covers both A layouts and a
/// `k > 512` case past the general cache-model `kc`
#[cfg(feature = "half")]
#[test]
fn prepack_lhs_equals_gemm_mixed() {
    fn check<T: Elem>() {
        for (m, k, n) in [(200, 130, 175), (96, 65, 72), (65, 64, 64), (96, 1024, 72)] {
            for &la in &[Layout::Col, Layout::Row] {
                let a = Mat::<T>::rand(m, k, 0x4A + (m * 7 + n) as u64);
                let b = Mat::<T>::rand(k, n, 0x4B + (n * 3 + k) as u64);
                let c0 = Mat::<T>::rand(m, n, 0x4C + (k + m) as u64);
                let (abuf, rsa, csa) = build_view(&a, la);
                let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
                // gemm_packed_a only serves row-major-ish C
                let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
                let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
                assert_eq!(packed.rows(), m);
                assert_eq!(packed.cols(), k);

                for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                    let (al, be) = (T::from_f64(al), T::from_f64(be));
                    let mut c_ref = cbase.clone();
                    gemm(
                        al,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        MatRef::new(&bbuf, k, n, rsb, csb),
                        be,
                        MatMut::new(&mut c_ref, m, n, rsc, csc),
                        Parallelism::Serial,
                    );
                    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                        let mut c_pk = cbase.clone();
                        gemmkit::gemm_packed_a(
                            al,
                            &packed,
                            MatRef::new(&bbuf, k, n, rsb, csb),
                            be,
                            MatMut::new(&mut c_pk, m, n, rsc, csc),
                            par,
                        );
                        assert!(
                            c_ref
                                .iter()
                                .zip(&c_pk)
                                .all(|(x, y)| x.to_f64().to_bits() == y.to_f64().to_bits()),
                            "mixed prepack_lhs != gemm for {m}x{k}x{n} la={la:?} par={par:?}"
                        );
                    }
                }
            }
        }
    }
    check::<gemmkit::f16>();
    check::<gemmkit::bf16>();
}

/// Both-tiny shapes (`m <= 64 && n <= 64`) through the prepacked-LHS path: like the RHS case,
/// the prepack dodges the tiny-matrix shortcut and always blocks through the general cache
/// model, so this checks accuracy against the f64 reference rather than bit-identity to plain
/// `gemm`, and that output stays bit-identical across thread counts
#[test]
fn prepack_lhs_both_tiny_accurate_and_deterministic() {
    for (m, k, n) in [(48, 40, 48), (60, 600, 60), (10, 9, 12)] {
        let a = Mat::<f32>::rand(m, k, 0x11 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0x22 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0x33 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
        let cref = reference(&a, &b, &c0, 1.0, 0.5);

        let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
        let mut c_ser = cbase.clone();
        gemmkit::gemm_packed_a(
            1.0,
            &packed,
            MatRef::new(&bbuf, k, n, rsb, csb),
            0.5,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        assert_accurate(
            &c_ser,
            rsc,
            csc,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "both-tiny prepack_lhs",
        );
        for threads in [2usize, 8] {
            let mut c_par = cbase.clone();
            gemmkit::gemm_packed_a(
                1.0,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                0.5,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(threads),
            );
            assert_eq!(
                c_ser, c_par,
                "both-tiny prepack_lhs serial != parallel({threads}) for {m}x{k}x{n}"
            );
        }
    }
}

/// gemv shapes (`n == 1` or `m == 1`) through the prepacked-LHS path. Plain `gemm` routes
/// these to a dedicated gemv kernel, but `gemm_packed_a` always drives the general
/// (transposed) prepacked kernel instead, so this checks accuracy against the f64 reference
/// rather than bit-identity, and that the row-major-ish C contract (`|csc| <= |rsc|`) still
/// admits the natural vector layout on either axis
#[test]
fn prepack_lhs_gemv_accurate() {
    // (m, k, n, rsc, csc): n==1 column-vector C (rsc=1,csc=1) and m==1 row-vector C
    // (rsc=n,csc=1), both row-major-ish so gemm_packed_a's C-orientation guard accepts them
    for &(m, k, n, rsc, csc) in &[
        (64usize, 40, 1usize, 1isize, 1isize),
        (255, 129, 1, 1, 1),
        (1, 40, 64, 64, 1),
        (1, 100, 255, 255, 1),
    ] {
        let a = Mat::<f32>::rand(m, k, 0xAA + (m * 5 + k) as u64);
        let b = Mat::<f32>::rand(k, n, 0xBB + (n * 7 + k) as u64);
        let c0 = Mat::<f32>::rand(m, n, 0xCC + (m + n) as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let mut cbuf = c0.v.clone(); // already laid out for either (rsc, csc) above, since one of m,n is 1
        let cref = reference(&a, &b, &c0, 1.3, -0.4);

        for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
            cbuf.copy_from_slice(&c0.v);
            let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
            gemmkit::gemm_packed_a(
                1.3,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                -0.4,
                MatMut::new(&mut cbuf, m, n, rsc, csc),
                par,
            );
            assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, "prepack_lhs gemv");
        }
    }
}

/// `gemm_packed_a` rejects a column-major-ish C: that orientation would keep A in the
/// genuine LHS role, which a prepacked A (laid out as the transposed product's RHS) cannot
/// serve
#[test]
#[should_panic(expected = "row-major-ish C")]
fn prepack_lhs_col_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = gemmkit::prepack_lhs(MatRef::from_col_major(&a, m, k));
    gemmkit::gemm_packed_a(
        1.0,
        &packed,
        MatRef::from_col_major(&b, k, n),
        0.0,
        MatMut::from_col_major(&mut c, m, n), // column-major C keeps A in the LHS role
        Parallelism::Serial,
    );
}

/// `gemm_packed_b` rejects a row-major-ish C: that orientation would force an A/B swap,
/// which a prepacked B (packed as the genuine RHS) cannot support
#[test]
#[should_panic(expected = "column-major-ish C")]
fn prepack_row_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = gemmkit::prepack_rhs(MatRef::from_col_major(&b, k, n));
    gemmkit::gemm_packed_b(
        1.0,
        MatRef::from_col_major(&a, m, k),
        &packed,
        0.0,
        MatMut::from_row_major(&mut c, m, n), // row-major C forces the A/B swap this rejects
        Parallelism::Serial,
    );
}
