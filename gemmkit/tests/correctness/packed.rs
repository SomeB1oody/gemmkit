//! Prepacked-LHS/RHS paths vs plain gemm: bit-identity, both-tiny accuracy,
//! and the packed-orientation panic guarantees

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Prepacked-RHS must be **bit-identical** to a plain `gemm()` on the same
/// inputs, for any thread count and any B layout (C column-major = the supported
/// no-swap orientation). This is the determinism gate for the reuse path: packing
/// only rearranges B's values, so the microkernel does the identical fused FMAs in
/// the identical order
#[test]
fn prepack_equals_gemm() {
    // All shapes are non-both-tiny (not m<=64 && n<=64), the supported regime
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // just above the tiny shortcut on m
        (64, 64, 65), // just above on n
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

/// The raw packed entries (`prepack_rhs_unchecked` + `gemm_packed_b_unchecked`, and the LHS pair
/// `prepack_lhs_unchecked` + `gemm_packed_a_unchecked`) must equal a plain `gemm()`. These are the
/// adapter/FFI-facing signatures, exercised directly through raw pointers + strides (the safe
/// packed path forwards through them, so the result is bit-identical to `gemm`)
#[test]
fn packed_unchecked_matches_gemm() {
    for (m, k, n) in [(200usize, 130, 175), (65, 64, 64), (40, 200, 300)] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0x11 + (m + n) as u64);
            let b = Mat::<f32>::rand(k, n, 0x22 + (k + n) as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x33 + (m + k) as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Col);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);

            // RHS-packed: requires column-major-ish C
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
                // SAFETY: views are in bounds; B is a distinct buffer read through its strides
                let packed =
                    unsafe { gemmkit::prepack_rhs_unchecked(bbuf.as_ptr(), rsb, csb, k, n) };
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c = cbase.clone();
                    // SAFETY: A/C in bounds, C column-major (packed_b orientation), distinct buffers
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

            // LHS-packed: requires row-major-ish C
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
                // SAFETY: views are in bounds; A is a distinct buffer read through its strides
                let packed =
                    unsafe { gemmkit::prepack_lhs_unchecked(abuf.as_ptr(), rsa, csa, m, k) };
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c = cbase.clone();
                    // SAFETY: B/C in bounds, C row-major (packed_a orientation), distinct buffers
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

/// Mixed-precision prepacked-RHS must be **bit-identical** to plain `gemm()` for
/// the narrow types too: the prepack blocks with the packed-input (`Lhs`) size and
/// the same `kc = k` the driver uses, so packed and unpacked never diverge. Includes
/// a `k > 512` cross-panel case
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

/// Prepacked-**i8** RHS must be **bit-identical** to a plain `gemm_i8()` on the same inputs, for
/// any thread count and any B layout (C column-major = the supported no-swap orientation).
/// Integer accumulation is exact (wrapping i32, associative), so this is a hard equality across
/// the auto-dispatched integer kernel (the VNNI `vpdpbusd` dot kernel on a VNNI box, the widen
/// kernel elsewhere) - including the auto-VNNI *small parallel* problems the plain path hands to
/// the widen fallback (the prepacked path always runs the buffer's own family). Covers `k` not a
/// multiple of 4 (the VNNI depth pad), `n` not a multiple of the panel width, `k == 1`, and the
/// small-`m` fixed-weight inference shape
#[cfg(feature = "int8")]
#[test]
fn prepack_i8_equals_gemm_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [
        (200usize, 130, 175),
        (128, 96, 112),
        (65, 64, 64),
        (64, 65, 100), // k not a multiple of 4, n not a multiple of nr
        (96, 129, 129),
        (33, 17, 19),
        (256, 257, 129),
        (300, 1, 256), // k == 1 (below the small-k gate on the plain path)
        (8, 2048, 96), // small m, long k: the fixed-weight inference shape
        (2, 1023, 12), // tiny m, k not a multiple of 4, n == nr
    ] {
        let a = rand_i8(m * k, 0x51 + (m * 7 + n) as u64);
        let b_rm = rand_i8(k * n, 0x62 + (n * 3 + k) as u64); // logical k x n stored row-major
        // column-major copy of the same logical B (so both layouts describe the same matrix)
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

/// The raw i8 packed entries (`prepack_rhs_i8_unchecked` + `gemm_i8_packed_b_unchecked`) must equal
/// a plain `gemm_i8()`: the adapter/FFI-facing signatures, exercised directly through raw pointers
/// + strides (the safe packed path forwards through them)
#[cfg(feature = "int8")]
#[test]
fn packed_i8_unchecked_matches_gemm_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [(200usize, 130, 175), (65, 64, 64), (8, 512, 96)] {
        let a = rand_i8(m * k, 0x71 + (m + n) as u64);
        let b = rand_i8(k * n, 0x82 + (k + n) as u64); // column-major k x n
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
            // SAFETY: views are in bounds; B is a distinct buffer read through its (col-major) strides
            let packed =
                unsafe { gemmkit::prepack_rhs_i8_unchecked(b.as_ptr(), 1, k as isize, k, n) };
            for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                let mut c = c0.clone();
                // SAFETY: A/C in bounds, C column-major (packed_b orientation), distinct buffers
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

/// A row-major-ish C is unsupported by the i8 prepacked path (it would swap A/B); `gemm_i8_packed_b`
/// must reject it with the same wording as the float packed path
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
        MatMut::from_row_major(&mut c, m, n), // row-major C -> swap -> reject
        Parallelism::Serial,
    );
}

/// An empty i8 operand must prepack and round-trip through the consume call as `C <- beta*C`
/// without running the pack-geometry arithmetic
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

/// f64 prepacked path is bit-identical too (exercises the f64 tile + the packed
/// geometry for a 2nd element type)
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

/// Both-tiny products (`m <= 64 && n <= 64`) still work via the prepacked path: it
/// uses the buffer's own blocking (which may round differently from plain gemm's
/// small-matrix shortcut), so it checks *accuracy* against the f64 reference rather
/// than bit-identity to plain gemm, and the output must stay bit-identical across
/// thread counts. `(60, 600, 60)` exercises the `k > 512` case where the
/// general/tiny blocking diverges
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

/// Prepacked-LHS (the mirror of `prepack_equals_gemm`): reusing a prepacked `A`
/// must be **bit-identical** to a plain `gemm()` for any thread count and any A
/// layout, with a **row-major-ish C** (the supported no-extra-swap orientation:
/// the engine drives the prepacked-A product transposed). Packing only rearranges
/// A's values, so the microkernel does the identical fused FMAs in the identical
/// order
#[test]
fn prepack_lhs_equals_gemm() {
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // just above the tiny shortcut on m
        (64, 64, 65), // just above on n
        (300, 1, 256),
        (40, 200, 300),
    ] {
        for &la in &[Layout::Col, Layout::Row] {
            // A and its packed buffer depend only on the (shape, layout): hoist the
            // pack above the alpha/beta loop so it happens once, not per combo
            let a = Mat::<f32>::rand(m, k, 0x5A + (m * 7 + n) as u64);
            let b = Mat::<f32>::rand(k, n, 0x6B + (n * 3 + k) as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x7C + (k + m) as u64);
            let (abuf, rsa, csa) = build_view(&a, la);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            // Row-major C is the supported orientation for the packed-LHS path
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

/// f64 prepacked-LHS path is bit-identical too (exercises the f64 tile + the packed
/// geometry for a 2nd element type)
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

/// Mixed-precision prepacked-**LHS** must be bit-identical to plain `gemm()` for the
/// narrow types: the LHS transpose-pack path at `Acc != Lhs` sizes. This is the only
/// path that simultaneously hits a narrow type, the transpose/packing framework, and
/// the `kc = k` single-panel rule (`prepack_equals_gemm_mixed` covers the RHS pack;
/// `prepack_lhs_equals_gemm` covers the transpose at `Acc == Lhs`). Mirrors the RHS
/// mixed test on the `gemm_packed_a` side, with both A layouts and a `k > 512` case
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
                // Row-major C is the supported orientation for the packed-LHS path
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

/// Both-tiny products (`m <= 64 && n <= 64`) via the prepacked-LHS path: like the
/// RHS case it uses the buffer's own blocking, so check *accuracy* against the f64
/// reference rather than bit-identity to plain gemm, and the output must stay
/// bit-identical across thread counts
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

/// gemv shapes (`n == 1` and `m == 1`) through the prepacked-LHS path. Plain `gemm`
/// routes these to the dedicated gemv kernel, but `gemm_packed_a` runs them through
/// the general driver (the transpose maps a unit dimension onto a unit *driver*
/// dimension), so this checks **accuracy** against the f64 reference, and that the
/// row-major-ish C contract admits the natural vector layouts (a unit column/row is
/// addressed with `|csc| <= |rsc|`)
#[test]
fn prepack_lhs_gemv_accurate() {
    // (m, k, n, rsc, csc): n==1 column-vector C (rsc=1,csc=1) and m==1 row-vector C
    // (rsc=n,csc=1), both row-major-ish so the packed-LHS guard accepts them
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
        let mut cbuf = c0.v.clone(); // row-major m*n vector (rs=csc-major chosen above)
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

/// A column-major-ish C is unsupported by the prepacked-LHS path (it would keep A
/// in the genuine LHS role); `gemm_packed_a` must reject it
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
        MatMut::from_col_major(&mut c, m, n), // column-major C -> reject
        Parallelism::Serial,
    );
}

/// A row-major-ish C is unsupported by the prepacked path (it would swap A/B);
/// `gemm_packed_b` must reject it instead of silently computing the wrong thing
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
        MatMut::from_row_major(&mut c, m, n), // row-major C -> swap -> reject
        Parallelism::Serial,
    );
}
