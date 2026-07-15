//! The workspace-reusing `_with` adapters: `gemm_with`, `gemm_packed_b_with`,
//! `gemm_packed_a_with`, `gemm_i8_with`, and `gemm_cplx_with` must each produce the same result as
//! their allocating counterpart (already checked in `adapter.rs`), reusing one caller-owned
//! [`Workspace`] across calls. This also drives the `Some(ws)` match arm of every `_common` helper
//! and, transitively, `gemmkit::gemm_unchecked_with`.

use faer::{Mat, MatMut, MatRef};
use gemmkit::{Parallelism, Workspace};

use gemmkit_faer::{
    dot, gemm, gemm_packed_a, gemm_packed_a_with, gemm_packed_b, gemm_packed_b_with, gemm_with,
    prepack_lhs, prepack_rhs,
};

fn rand_mat(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Mat::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

fn naive_ref(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    Mat::from_fn(m, n, |i, j| (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum())
}

fn assert_close(got: MatRef<'_, f64>, exp: MatRef<'_, f64>, tol: f64) {
    assert_eq!(
        (got.nrows(), got.ncols()),
        (exp.nrows(), exp.ncols()),
        "shape mismatch"
    );
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            let (g, e) = (got[(i, j)], exp[(i, j)]);
            assert!(
                (g - e).abs() <= tol + tol * e.abs(),
                "mismatch at ({i},{j}): {g} vs {e}"
            );
        }
    }
}

/// `gemm_with` reusing one workspace across several shapes must equal the allocating `gemm`.
#[test]
fn gemm_with_matches_gemm() {
    let mut ws = Workspace::new();
    for &(m, k, n) in &[(8usize, 6, 5), (64, 64, 64), (100, 1, 80)] {
        let a = rand_mat(m, k, 1 + m as u64);
        let b = rand_mat(k, n, 2 + n as u64);
        let c0 = rand_mat(m, n, 3 + k as u64);
        let (alpha, beta) = (1.4f64, -0.6);
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            let mut c_with = c0.clone();
            gemm_with(
                &mut ws,
                alpha,
                a.as_dyn_stride(),
                b.as_dyn_stride(),
                beta,
                c_with.as_dyn_stride_mut(),
                par,
            );
            let mut c_ref = c0.clone();
            gemm(
                alpha,
                a.as_dyn_stride(),
                b.as_dyn_stride(),
                beta,
                c_ref.as_dyn_stride_mut(),
                par,
            );
            assert_close(c_with.as_dyn_stride(), c_ref.as_dyn_stride(), 1e-12);
        }
    }
}

/// `prepack_rhs` + `gemm_packed_b_with` (column-major C) must equal `gemm_packed_b` and the naive
/// reference.
#[test]
fn gemm_packed_b_with_matches() {
    let (m, k, n) = (100usize, 64, 80);
    let a = rand_mat(m, k, 51);
    let b = rand_mat(k, n, 52);
    let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let packed = prepack_rhs(b.as_dyn_stride());
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = Mat::<f64>::from_fn(m, n, |_, _| 0.0); // column-major
        gemm_packed_b_with(
            &mut ws,
            1.0,
            a.as_dyn_stride(),
            &packed,
            0.0,
            c_with.as_dyn_stride_mut(),
            par,
        );

        let mut c_ref = Mat::<f64>::from_fn(m, n, |_, _| 0.0);
        gemm_packed_b(
            1.0,
            a.as_dyn_stride(),
            &packed,
            0.0,
            c_ref.as_dyn_stride_mut(),
            par,
        );

        assert_close(c_with.as_dyn_stride(), c_ref.as_dyn_stride(), 1e-12);
        assert_close(c_with.as_dyn_stride(), exp.as_dyn_stride(), 1e-10);
    }
}

/// `prepack_lhs` + `gemm_packed_a_with` (row-major C) must equal `gemm_packed_a` and the reference.
#[test]
fn gemm_packed_a_with_matches() {
    let (m, k, n) = (96usize, 50, 72);
    let a = rand_mat(m, k, 61);
    let b = rand_mat(k, n, 62);
    let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let packed = prepack_lhs(a.as_dyn_stride());
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut d_with = vec![0.0f64; m * n];
        {
            let c = MatMut::from_row_major_slice_mut(&mut d_with, m, n);
            gemm_packed_a_with(&mut ws, 1.0, &packed, b.as_dyn_stride(), 0.0, c, par);
        }

        let mut d_ref = vec![0.0f64; m * n];
        {
            let c = MatMut::from_row_major_slice_mut(&mut d_ref, m, n);
            gemm_packed_a(1.0, &packed, b.as_dyn_stride(), 0.0, c, par);
        }

        let got = MatRef::from_row_major_slice(&d_with, m, n);
        let refv = MatRef::from_row_major_slice(&d_ref, m, n);
        assert_close(got, exp.as_dyn_stride(), 1e-10);
        assert_close(got, refv, 1e-12);
    }
}

/// The `dot` convenience keeps working — a smoke check the shared imports resolve.
#[test]
fn dot_still_matches_naive() {
    let a = rand_mat(12, 9, 71);
    let b = rand_mat(9, 7, 72);
    assert_close(
        dot(a.as_dyn_stride(), b.as_dyn_stride()).as_dyn_stride(),
        naive_ref(a.as_dyn_stride(), b.as_dyn_stride()).as_dyn_stride(),
        1e-10,
    );
}

/// `gemm_i8_with` (i8 -> i32) reusing a workspace must equal the allocating `gemm_i8`.
#[cfg(feature = "int8")]
#[test]
fn gemm_i8_with_matches_gemm_i8() {
    use gemmkit_faer::{gemm_i8, gemm_i8_with};

    let randi8 = |r: usize, c: usize, seed: u64| -> Mat<i8> {
        let mut s = seed | 1;
        Mat::from_fn(r, c, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 17 - 8) as i8
        })
    };
    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);
    let c0 = Mat::<i32>::from_fn(m, n, |i, j| (i * n + j) as i32 - 5);
    let (alpha, beta) = (3i32, -2i32);

    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = c0.clone();
        gemm_i8_with(
            &mut ws,
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_with.as_dyn_stride_mut(),
            par,
        );
        let mut c_ref = c0.clone();
        gemm_i8(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_ref.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(c_with[(i, j)], c_ref[(i, j)]);
            }
        }
    }
}

/// `gemm_cplx_with` reusing a workspace must equal `gemm_cplx` for every conjugation combination.
#[cfg(feature = "complex")]
#[test]
fn gemm_cplx_with_matches_gemm_cplx() {
    use gemmkit::Complex;
    use gemmkit_faer::{gemm_cplx, gemm_cplx_with};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> Mat<C> {
        let re = rand_mat(r, c, s);
        let im = rand_mat(r, c, s ^ 0xABCD);
        Mat::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
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
            a.as_dyn_stride(),
            conj_a,
            b.as_dyn_stride(),
            conj_b,
            beta,
            c_with.as_dyn_stride_mut(),
            Parallelism::Serial,
        );
        let mut c_ref = c0.clone();
        gemm_cplx(
            alpha,
            a.as_dyn_stride(),
            conj_a,
            b.as_dyn_stride(),
            conj_b,
            beta,
            c_ref.as_dyn_stride_mut(),
            Parallelism::Serial,
        );
        for i in 0..m {
            for j in 0..n {
                let d = c_with[(i, j)] - c_ref[(i, j)];
                assert!(
                    d.re.hypot(d.im) < 1e-9,
                    "gemm_cplx_with conj=({conj_a},{conj_b}) mismatch at ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_fused_with` reusing a workspace must equal the allocating `gemm_fused` bit-for-bit.
#[cfg(feature = "epilogue")]
#[test]
fn gemm_fused_with_matches_gemm_fused() {
    use gemmkit_faer::{Activation, Bias, gemm_fused, gemm_fused_with};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_mat(m, k, 401);
    let b = rand_mat(k, n, 402);
    let c0 = rand_mat(m, n, 403);
    let bias: Vec<f64> = (0..n).map(|j| 0.25 * j as f64 - 1.0).collect();
    let (alpha, beta) = (1.1f64, -0.5);
    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_with = c0.clone();
        gemm_fused_with(
            &mut ws,
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_with.as_dyn_stride_mut(),
            Some(Bias::PerCol(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        let mut c_ref = c0.clone();
        gemm_fused(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_ref.as_dyn_stride_mut(),
            Some(Bias::PerCol(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c_with[(i, j)].to_bits(),
                    c_ref[(i, j)].to_bits(),
                    "fused_with ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_i8_requant_with` / `gemm_i8_requant_u8_with` reusing a workspace must equal the allocating
/// entries.
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[test]
fn gemm_i8_requant_with_matches() {
    use gemmkit_faer::{
        Requantize, gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_with,
        gemm_i8_requant_with,
    };

    let randi8 = |r: usize, c: usize, seed: u64| -> Mat<i8> {
        let mut s = seed | 1;
        Mat::from_fn(r, c, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 51 - 25) as i8
        })
    };
    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);
    let bias: Vec<i32> = (0..m).map(|i| 30 * i as i32 - 100).collect();
    let (scale, zp_i8, zp_u8) = (0.05f32, -7i32, 30i32);

    let mut ws = Workspace::new();
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mk = |zp: i32| Requantize {
            scale,
            zero_point: zp,
            bias: Some(&bias),
        };
        let mut c_with = Mat::<i8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant_with(
            &mut ws,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            mk(zp_i8),
            c_with.as_dyn_stride_mut(),
            par,
        );
        let mut c_ref = Mat::<i8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            mk(zp_i8),
            c_ref.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(c_with[(i, j)], c_ref[(i, j)], "requant_with i8 ({i},{j})");
            }
        }

        let mut cu_with = Mat::<u8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant_u8_with(
            &mut ws,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            mk(zp_u8),
            cu_with.as_dyn_stride_mut(),
            par,
        );
        let mut cu_ref = Mat::<u8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant_u8(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            mk(zp_u8),
            cu_ref.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(cu_with[(i, j)], cu_ref[(i, j)], "requant_with u8 ({i},{j})");
            }
        }
    }
}

/// `gemm_cplx_fused_with` reusing a workspace must equal the allocating `gemm_cplx_fused`.
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[test]
fn gemm_cplx_fused_with_matches() {
    use gemmkit::Complex;
    use gemmkit_faer::{Bias, gemm_cplx_fused, gemm_cplx_fused_with};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> Mat<C> {
        let re = rand_mat(r, c, s);
        let im = rand_mat(r, c, s ^ 0xABCD);
        Mat::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
    };
    let (m, k, n) = (12usize, 9, 7);
    let a = crand(m, k, 421);
    let b = crand(k, n, 422);
    let c0 = crand(m, n, 423);
    let bias: Vec<C> = (0..n).map(|j| Complex::new(0.3 * j as f64, -0.6)).collect();
    let alpha = Complex::new(1.3, -0.4);
    let beta = Complex::new(0.5, 0.7);

    let mut ws = Workspace::new();
    let mut c_with = c0.clone();
    gemm_cplx_fused_with(
        &mut ws,
        alpha,
        a.as_dyn_stride(),
        true,
        b.as_dyn_stride(),
        false,
        beta,
        c_with.as_dyn_stride_mut(),
        Some(Bias::PerCol(&bias)),
        Parallelism::Serial,
    );
    let mut c_ref = c0.clone();
    gemm_cplx_fused(
        alpha,
        a.as_dyn_stride(),
        true,
        b.as_dyn_stride(),
        false,
        beta,
        c_ref.as_dyn_stride_mut(),
        Some(Bias::PerCol(&bias)),
        Parallelism::Serial,
    );
    for i in 0..m {
        for j in 0..n {
            assert_eq!(c_with[(i, j)].re.to_bits(), c_ref[(i, j)].re.to_bits());
            assert_eq!(c_with[(i, j)].im.to_bits(), c_ref[(i, j)].im.to_bits());
        }
    }
}
