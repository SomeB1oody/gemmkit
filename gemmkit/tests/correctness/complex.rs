//! Complex GEMM (c32/c64, conj variants)

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace};

/// Complex counterpart of [`beta_zero_does_not_read_c`]: the SoA complex kernel's
/// `Zero` branch must overwrite C without reading it. `ref_cplx` has no beta==0 guard
/// (it always multiplies `beta * C0`), so the reference is built from a *zeroed* C
/// while the kernel is fed a NaN-seeded C: a spurious read would surface as a
/// non-finite output
#[cfg(feature = "complex")]
#[test]
fn beta_zero_does_not_read_c_complex() {
    fn check<T: CElem>() {
        for (m, k, n) in [(40usize, 33, 28), (64, 16, 96)] {
            let a = rand_cplx::<T>(m * k, 0x5E + m as u64);
            let b = rand_cplx::<T>(k * n, 0x5F + n as u64);
            let zero_c0 = vec![T::of(0.0, 0.0); m * n];
            let (alpha, beta) = (T::of(1.0, 0.0), T::of(0.0, 0.0));
            let cref = ref_cplx(&a, &b, &zero_c0, m, k, n, alpha, beta, false, false);
            let mut c = vec![T::of(f64::NAN, f64::NAN); m * n];
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                false,
                MatRef::from_col_major(&b, k, n),
                false,
                beta,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Serial,
            );
            assert_cplx_accurate(
                &c,
                m,
                n,
                &cref,
                k,
                &format!("cplx beta=0 NaN C {m}x{k}x{n}"),
            );
        }
    }
    check::<gemmkit::c32>();
    check::<gemmkit::c64>();
}

/// Complex counterpart: `gemm_cplx_with` (the only complex workspace-reuse entry) had
/// no test at all. Must match `gemm_cplx` bit-for-bit, including conjugated cases
#[cfg(feature = "complex")]
#[test]
fn workspace_reuse_matches_allocating_complex() {
    fn check<T: CElem>() {
        let (m, k, n) = (96, 65, 72);
        let a = rand_cplx::<T>(m * k, 0x8A + m as u64);
        let b = rand_cplx::<T>(k * n, 0x8B + n as u64);
        let c0 = rand_cplx::<T>(m * n, 0x8C + k as u64);
        let (alpha, beta) = (T::of(1.1, -0.3), T::of(0.5, 0.7));
        for &(ca, cb) in &[(false, false), (true, false), (true, true)] {
            let mut c_ref = c0.clone();
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                beta,
                MatMut::from_col_major(&mut c_ref, m, n),
                Parallelism::Serial,
            );
            let mut ws = Workspace::new();
            for _ in 0..2 {
                let mut c_ws = c0.clone();
                gemmkit::gemm_cplx_with(
                    &mut ws,
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    ca,
                    MatRef::from_col_major(&b, k, n),
                    cb,
                    beta,
                    MatMut::from_col_major(&mut c_ws, m, n),
                    Parallelism::Serial,
                );
                for idx in 0..m * n {
                    let (rr, ri) = c_ref[idx].parts();
                    let (wr, wi) = c_ws[idx].parts();
                    assert!(
                        rr.to_bits() == wr.to_bits() && ri.to_bits() == wi.to_bits(),
                        "gemm_cplx_with != gemm_cplx at {idx} ca={ca} cb={cb}"
                    );
                }
            }
        }
    }
    check::<gemmkit::c32>();
    check::<gemmkit::c64>();
}

#[cfg(feature = "complex")]
#[test]
fn correctness_complex() {
    fn check<T: CElem>() {
        for (m, k, n) in [
            (1, 1, 1),
            (3, 4, 5),
            (32, 32, 32),
            (33, 17, 19),
            (64, 80, 48),
        ] {
            for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
                let a = rand_cplx::<T>(m * k, 0xC0 + (m * 7 + k) as u64);
                let b = rand_cplx::<T>(k * n, 0xC1 + (n * 3 + k) as u64);
                let c0 = rand_cplx::<T>(m * n, 0xC2 + (m + n) as u64);
                let (alpha, beta) = (T::of(1.1, -0.3), T::of(0.5, 0.7));
                let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, ca, cb);
                let mut c = c0.clone();
                // All column-major
                gemmkit::gemm_cplx(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    ca,
                    MatRef::from_col_major(&b, k, n),
                    cb,
                    beta,
                    MatMut::from_col_major(&mut c, m, n),
                    Parallelism::Serial,
                );
                assert_cplx_accurate(&c, m, n, &cref, k, &format!("{m}x{k}x{n} ca={ca} cb={cb}"));
            }
        }
    }
    check::<gemmkit::c32>();
    check::<gemmkit::c64>();
}

/// `alpha == 0` is the scale-only path: `C <- beta*C0`, with `A*B` (and any conj)
/// skipped. It is the only way to reach the complex `scale_c_float` monomorphization,
/// previously unexercised. `conj_a = true` is set on purpose and must be ignored
#[cfg(feature = "complex")]
#[test]
fn correctness_complex_alpha_zero() {
    fn check<T: CElem>() {
        for (m, k, n) in [(3, 4, 5), (40, 33, 28), (64, 80, 48)] {
            let a = rand_cplx::<T>(m * k, 0xA0 + (m * 7 + k) as u64);
            let b = rand_cplx::<T>(k * n, 0xA1 + (n * 3 + k) as u64);
            let c0 = rand_cplx::<T>(m * n, 0xA2 + (m + n) as u64);
            let (alpha, beta) = (T::of(0.0, 0.0), T::of(0.5, 0.7));
            let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, false, false);
            let mut c = c0.clone();
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                true, // conj A on purpose: alpha==0 skips A*B, so it must have no effect
                MatRef::from_col_major(&b, k, n),
                false,
                beta,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Serial,
            );
            assert_cplx_accurate(&c, m, n, &cref, k, &format!("cplx alpha=0 {m}x{k}x{n}"));
        }
    }
    check::<gemmkit::c32>();
    check::<gemmkit::c64>();
}

/// Complex serial == parallel bit-identity across thread counts: complex add isn't
/// associative either, so the same thread-independent-blocking caveat as
/// `parallel_equals_serial_bit_identical` applies
#[cfg(feature = "complex")]
#[test]
fn parallel_equals_serial_complex() {
    for (m, k, n) in [(200, 130, 175), (256, 64, 200)] {
        for &(ca, cb) in &[(false, false), (true, true)] {
            let a = rand_cplx::<gemmkit::c32>(m * k, 0xD0 + m as u64);
            let b = rand_cplx::<gemmkit::c32>(k * n, 0xD1 + n as u64);
            let c0 = rand_cplx::<gemmkit::c32>(m * n, 0xD2 + k as u64);
            let (alpha, beta) = (
                gemmkit::Complex::new(0.7f32, 0.2),
                gemmkit::Complex::new(1.3f32, -0.4),
            );
            let mut c_ser = c0.clone();
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                beta,
                MatMut::from_col_major(&mut c_ser, m, n),
                Parallelism::Serial,
            );
            for t in [2usize, 4, 8, 16] {
                let mut c_par = c0.clone();
                gemmkit::gemm_cplx(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    ca,
                    MatRef::from_col_major(&b, k, n),
                    cb,
                    beta,
                    MatMut::from_col_major(&mut c_par, m, n),
                    Parallelism::Rayon(t),
                );
                let bits = |v: &[gemmkit::c32]| {
                    v.iter()
                        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&c_ser), bits(&c_par), "complex serial != par({t})");
            }
        }
    }
}

/// The raw `gemm_cplx_unchecked` entry (added for the ndarray adapter): exercise it
/// with a **negative-row-stride** A view + conj, against the row-reversed reference.
/// The safe `MatRef` surface can't express reversed strides, so this raw path is what
/// arbitrary-stride callers (the ndarray adapter) rely on
#[cfg(feature = "complex")]
#[test]
fn cplx_unchecked_negative_strides() {
    let (m, k, n) = (12, 9, 7);
    for &(ca, cb) in &[(false, false), (true, false), (true, true)] {
        let a = rand_cplx::<gemmkit::c32>(m * k, 0xF0 + (m + k) as u64);
        let b = rand_cplx::<gemmkit::c32>(k * n, 0xF1 + (n + k) as u64);
        let c0 = rand_cplx::<gemmkit::c32>(m * n, 0xF2 + (m + n) as u64);
        let (alpha, beta) = (
            gemmkit::Complex::new(1.1f32, -0.3),
            gemmkit::Complex::new(0.5f32, 0.7),
        );
        // Row-reversed copy of the (column-major) A, for the reference
        let mut a_rev = a.clone();
        for p in 0..k {
            for i in 0..m {
                a_rev[p * m + i] = a[p * m + (m - 1 - i)];
            }
        }
        let cref = ref_cplx(&a_rev, &b, &c0, m, k, n, alpha, beta, ca, cb);
        let mut c = c0.clone();
        // A: base at physical row m-1, row stride -1 (col-major col stride m); B/C col-major
        unsafe {
            gemmkit::gemm_cplx_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr().add(m - 1),
                -1,
                m as isize,
                ca,
                b.as_ptr(),
                1,
                k as isize,
                cb,
                beta,
                c.as_mut_ptr(),
                1,
                m as isize,
                Parallelism::Serial,
            );
        }
        assert_cplx_accurate(
            &c,
            m,
            n,
            &cref,
            k,
            &format!("unchecked neg-stride ca={ca} cb={cb}"),
        );
    }
}

/// Cross-check complex (c32) against the `gemm` crate (which has native c32 and
/// `conj_lhs`/`conj_rhs` flags); `gemm::c32 == num_complex::Complex32 == gemmkit::c32`,
/// so the comparison is direct. Gated out of Miri and wasm (the `gemm` dev-dep is
/// `cfg(all(not(miri), not(wasm)))`)
#[test]
#[cfg(all(not(miri), not(target_family = "wasm"), feature = "complex"))]
fn complex_matches_gemm_crate() {
    for (m, k, n) in [(64, 48, 40), (96, 65, 72), (33, 17, 19)] {
        for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
            let a = rand_cplx::<gemmkit::c32>(m * k, 0xE0 + (m + k) as u64);
            let b = rand_cplx::<gemmkit::c32>(k * n, 0xE1 + (n + k) as u64);
            let mut c_kit = vec![gemmkit::Complex::new(0.0f32, 0.0); m * n];
            let mut c_gemm = c_kit.clone();

            gemmkit::gemm_cplx(
                gemmkit::Complex::new(1.0f32, 0.0),
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                gemmkit::Complex::new(0.0f32, 0.0),
                MatMut::from_col_major(&mut c_kit, m, n),
                Parallelism::Serial,
            );
            // gemm crate: dst = alpha*dst + beta*op(lhs)*op(rhs); alpha=0, beta=1
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    c_gemm.as_mut_ptr(),
                    m as isize,
                    1,
                    false,
                    a.as_ptr(),
                    m as isize,
                    1,
                    b.as_ptr(),
                    k as isize,
                    1,
                    gemmkit::Complex::new(0.0f32, 0.0),
                    gemmkit::Complex::new(1.0f32, 0.0),
                    false, // conj_dst
                    ca,    // conj_lhs
                    cb,    // conj_rhs
                    gemm::Parallelism::None,
                );
            }
            // Both column-major; build a row-major (f64,f64) reference from c_gemm
            let mut cref = vec![(0.0f64, 0.0f64); m * n];
            for i in 0..m {
                for j in 0..n {
                    let z = c_gemm[j * m + i];
                    cref[i * n + j] = (z.re as f64, z.im as f64);
                }
            }
            assert_cplx_accurate(
                &c_kit,
                m,
                n,
                &cref,
                k,
                &format!("c32 vs gemm {m}x{k}x{n} ca={ca} cb={cb}"),
            );
        }
    }
}

/// Exact conjugation check. On small-integer inputs every product and sum is exactly
/// representable in `f32`, so the FMA kernel and a scalar `num_complex` reference must
/// agree *exactly* (value equality, not the L2 tolerance the other complex tests use)
#[cfg(feature = "complex")]
#[test]
fn correctness_complex_conj_bit_exact() {
    use gemmkit::Complex;
    // Deterministic small integers in [-3, 3] (exactly representable; exact arithmetic)
    let cval = |seed: u64, i: usize| -> Complex<f32> {
        let r = (seed.wrapping_mul(2654435761).wrapping_add(i as u64) % 7) as i64 - 3;
        let m = (seed.wrapping_mul(40503).wrapping_add(i as u64 * 3) % 7) as i64 - 3;
        Complex::new(r as f32, m as f32)
    };
    let conj = |z: Complex<f32>, c: bool| if c { z.conj() } else { z };
    for (m, k, n) in [(2usize, 3usize, 2usize), (4, 5, 3)] {
        let a: Vec<Complex<f32>> = (0..m * k).map(|i| cval(0x11, i)).collect();
        let b: Vec<Complex<f32>> = (0..k * n).map(|i| cval(0x22, i)).collect();
        let c0: Vec<Complex<f32>> = (0..m * n).map(|i| cval(0x33, i)).collect();
        let (alpha, beta) = (Complex::new(2.0f32, 1.0), Complex::new(1.0f32, -1.0));
        for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
            // Column-major scalar reference, exact in f32
            let mut expect = c0.clone();
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex::new(0.0f32, 0.0);
                    for p in 0..k {
                        acc += conj(a[p * m + i], ca) * conj(b[j * k + p], cb);
                    }
                    expect[j * m + i] = beta * c0[j * m + i] + alpha * acc;
                }
            }
            let mut c = c0.clone();
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                beta,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Serial,
            );
            for idx in 0..m * n {
                assert_eq!(
                    c[idx], expect[idx],
                    "conj mismatch at {idx} ca={ca} cb={cb} ({m}x{k}x{n})"
                );
            }
        }
    }
}
