//! The workspace-reusing `_with` adapters: `gemm_with`, `gemm_batched_with`,
//! `gemm_packed_b_with`, `gemm_packed_a_with`, `gemm_i8_with`, and `gemm_cplx_with` must each
//! produce the same result as their allocating counterpart (already checked in `adapter.rs`),
//! reusing one caller-owned [`Workspace`] across calls. This also drives the `Some(ws)` match arm
//! of both `_common` helpers and, transitively, `gemmkit::gemm_unchecked_with`.

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, Axis, ShapeBuilder};

use gemmkit::{Parallelism, Workspace};
use gemmkit_ndarray::{
    dot, gemm, gemm_batched, gemm_batched_with, gemm_packed_a, gemm_packed_a_with, gemm_packed_b,
    gemm_packed_b_with, gemm_with, prepack_lhs, prepack_rhs,
};

fn rand2(r: usize, c: usize, seed: u64) -> Array2<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Array2::from_shape_fn((r, c), |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

fn rand3(b: usize, r: usize, c: usize, seed: u64) -> Array3<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Array3::from_shape_fn((b, r, c), |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// `gemm_with` reusing one workspace across several shapes must equal the allocating `gemm`.
#[test]
fn gemm_with_matches_gemm() {
    let mut ws = Workspace::new();
    for &(m, k, n) in &[(8usize, 6, 5), (64, 64, 64), (100, 1, 80)] {
        let a = rand2(m, k, 1 + m as u64);
        let b = rand2(k, n, 2 + n as u64);
        let c0 = rand2(m, n, 3 + k as u64);
        let (alpha, beta) = (1.4f64, -0.6);
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            let mut c_with = c0.clone();
            gemm_with(&mut ws, alpha, &a, &b, beta, &mut c_with, par);
            let mut c_ref = c0.clone();
            gemm(alpha, &a, &b, beta, &mut c_ref, par);
            assert_relative_eq!(c_with, c_ref, max_relative = 1e-12);
        }
    }
}

/// `gemm_batched_with` on a `(4, 8, 8, 8)` stack, reusing one workspace, must equal `gemm_batched`.
#[test]
fn gemm_batched_with_matches_gemm_batched() {
    let (batch, m, k, n) = (4usize, 8, 8, 8);
    let a = rand3(batch, m, k, 11);
    let b = rand3(batch, k, n, 22);
    let c0 = rand3(batch, m, n, 33);
    let (alpha, beta) = (0.9f64, 1.2);
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = c0.clone();
        gemm_batched_with(&mut ws, alpha, &a, &b, beta, &mut c_with, par);
        let mut c_ref = c0.clone();
        gemm_batched(alpha, &a, &b, beta, &mut c_ref, par);
        for e in 0..batch {
            assert_relative_eq!(
                c_with.index_axis(Axis(0), e).to_owned(),
                c_ref.index_axis(Axis(0), e).to_owned(),
                max_relative = 1e-12
            );
        }
    }
}

/// `prepack_rhs` + `gemm_packed_b_with` (column-major C) must equal `gemm_packed_b` and `dot`.
#[test]
fn gemm_packed_b_with_matches() {
    let (m, k, n) = (100usize, 64, 80);
    let a = rand2(m, k, 51);
    let b = rand2(k, n, 52);
    let packed = prepack_rhs(&b);
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = Array2::<f64>::zeros((m, n).f()); // column-major (packed_b orientation)
        gemm_packed_b_with(&mut ws, 1.0, &a, &packed, 0.0, &mut c_with, par);

        let mut c_ref = Array2::<f64>::zeros((m, n).f());
        gemm_packed_b(1.0, &a, &packed, 0.0, &mut c_ref, par);

        assert_relative_eq!(c_with, c_ref, max_relative = 1e-12);
        assert_relative_eq!(c_with, a.dot(&b), max_relative = 1e-10);
    }
}

/// `prepack_lhs` + `gemm_packed_a_with` (row-major C) must equal `gemm_packed_a` and `dot`.
#[test]
fn gemm_packed_a_with_matches() {
    let (m, k, n) = (96usize, 50, 72);
    let a = rand2(m, k, 61);
    let b = rand2(k, n, 62);
    let packed = prepack_lhs(&a);
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = Array2::<f64>::zeros((m, n)); // row-major (packed_a orientation)
        gemm_packed_a_with(&mut ws, 1.0, &packed, &b, 0.0, &mut c_with, par);

        let mut c_ref = Array2::<f64>::zeros((m, n));
        gemm_packed_a(1.0, &packed, &b, 0.0, &mut c_ref, par);

        assert_relative_eq!(c_with, c_ref, max_relative = 1e-12);
        assert_relative_eq!(c_with, a.dot(&b), max_relative = 1e-10);
    }
}

/// The `dot` convenience keeps working — a smoke check the shared imports resolve.
#[test]
fn dot_still_matches_ndarray() {
    let a = rand2(12, 9, 71);
    let b = rand2(9, 7, 72);
    assert_relative_eq!(dot(&a, &b), a.dot(&b), max_relative = 1e-10);
}

/// `gemm_i8_with` (i8 -> i32) reusing a workspace must equal the allocating `gemm_i8`.
#[cfg(feature = "int8")]
#[test]
fn gemm_i8_with_matches_gemm_i8() {
    use gemmkit_ndarray::{gemm_i8, gemm_i8_with};

    let randi8 = |r: usize, c: usize, seed: u64| -> Array2<i8> {
        let mut s = seed | 1;
        Array2::from_shape_fn((r, c), |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 17 - 8) as i8
        })
    };
    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);
    let c0 = Array2::<i32>::from_shape_fn((m, n), |(i, j)| (i * n + j) as i32 - 5);
    let (alpha, beta) = (3i32, -2i32);

    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = c0.clone();
        gemm_i8_with(&mut ws, alpha, &a, &b, beta, &mut c_with, par);
        let mut c_ref = c0.clone();
        gemm_i8(alpha, &a, &b, beta, &mut c_ref, par);
        assert_eq!(c_with, c_ref);
    }
}

/// `gemm_cplx_with` reusing a workspace must equal `gemm_cplx` for both conjugation flags.
#[cfg(feature = "complex")]
#[test]
fn gemm_cplx_with_matches_gemm_cplx() {
    use gemmkit::Complex;
    use gemmkit_ndarray::{gemm_cplx, gemm_cplx_with};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> Array2<C> {
        let re = rand2(r, c, s);
        let im = rand2(r, c, s ^ 0xABCD);
        Array2::from_shape_fn((r, c), |(i, j)| Complex::new(re[(i, j)], im[(i, j)]))
    };

    let (m, k, n) = (12usize, 9, 7);
    let a = crand(m, k, 1);
    let b = crand(k, n, 2);
    let c0 = crand(m, n, 4);
    let alpha = Complex::new(1.3, -0.4);
    let beta = Complex::new(0.5, 0.7);

    let mut ws = Workspace::new();
    for &(conj_a, conj_b) in &[(false, false), (true, false), (false, true), (true, true)] {
        let mut c_with = c0.clone();
        gemm_cplx_with(
            &mut ws,
            alpha,
            &a,
            conj_a,
            &b,
            conj_b,
            beta,
            &mut c_with,
            Parallelism::Serial,
        );
        let mut c_ref = c0.clone();
        gemm_cplx(
            alpha,
            &a,
            conj_a,
            &b,
            conj_b,
            beta,
            &mut c_ref,
            Parallelism::Serial,
        );
        for ((i, j), &g) in c_with.indexed_iter() {
            let d = g - c_ref[(i, j)];
            assert!(
                d.re.hypot(d.im) < 1e-9,
                "gemm_cplx_with conj=({conj_a},{conj_b}) mismatch at ({i},{j})"
            );
        }
    }
}
