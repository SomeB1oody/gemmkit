//! Correctness tests for the gemmkit-nalgebra adapter: `dot`/`gemm`, the packed (`prepack_rhs`/
//! `prepack_lhs` + `gemm_packed_b`/`gemm_packed_a`), i8, f16, complex, fused, map, and requantize
//! entries, checked against naive scalar references across owned `DMatrix`, static `SMatrix`,
//! contiguous views, row-major (strided) views, and non-contiguous stepped views. Dimension
//! mismatches and a prepacked operand paired with the wrong C orientation panic

use approx::assert_relative_eq;
use gemmkit::Parallelism;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, Dim, Matrix, RawStorage, SMatrix};

use gemmkit_nalgebra::{dot, gemm, gemm_packed_a, gemm_packed_b, prepack_lhs, prepack_rhs};

/// Xorshift64 fill for a column-major `DMatrix<f64>`, values uniform in `[-0.5, 0.5)`
fn rand2(r: usize, c: usize, seed: u64) -> DMatrix<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    DMatrix::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// Reference `A*B` computed via nalgebra's index operator, which honors whatever strides a view
/// carries, independent of nalgebra's own matmul implementation
fn naive_ref<R1, C1, S1, R2, C2, S2>(
    a: &Matrix<f64, R1, C1, S1>,
    b: &Matrix<f64, R2, C2, S2>,
) -> DMatrix<f64>
where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<f64, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<f64, R2, C2>,
{
    let (m, k) = a.shape();
    let (_, n) = b.shape();
    DMatrix::from_fn(m, n, |i, j| (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum())
}

/// Asserts `got` equals `exp` element-wise within `tol` relative error, after checking shapes match
fn assert_close<R, C, S>(got: &Matrix<f64, R, C, S>, exp: &DMatrix<f64>, tol: f64)
where
    R: Dim,
    C: Dim,
    S: RawStorage<f64, R, C>,
{
    assert_eq!(got.shape(), exp.shape(), "shape mismatch");
    let (m, n) = exp.shape();
    for i in 0..m {
        for j in 0..n {
            assert_relative_eq!(got[(i, j)], exp[(i, j)], max_relative = tol);
        }
    }
}

#[test]
fn dot_matches_naive() {
    for &(m, k, n) in &[
        (2, 2, 2),
        (7, 9, 5),
        (32, 40, 24),
        (64, 64, 64),
        (100, 1, 80),
    ] {
        let a = rand2(m, k, 1 + m as u64);
        let b = rand2(k, n, 2 + n as u64);
        assert_close(&dot(&a, &b), &naive_ref(&a, &b), 1e-10);
    }
}

#[test]
fn dot_small_exact() {
    // Same numbers as the crate's doc example: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let c = dot(&a, &b);
    assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
}

#[test]
fn accepts_view_and_owned() {
    let a = rand2(8, 6, 3);
    let b = rand2(6, 5, 4);
    let c1 = dot(&a, &b);
    // views spanning the whole matrix must give the same result as the owned matrices
    let av = a.view((0, 0), (8, 6));
    let bv = b.view((0, 0), (6, 5));
    let c2 = dot(&av, &bv);
    assert_close(&c1, &c2, 1e-12);
}

#[test]
fn row_major_strided_view() {
    let (m, k, n) = (16usize, 12, 10);
    let b = rand2(k, n, 6);

    // A stored row-major (nalgebra's non-natural layout) and viewed with strides (k, 1), so
    // |row stride| > |col stride|: the view reads straight through the buffer, no copy
    let mut s = 7u64.wrapping_add(0x9E3779B97F4A7C15);
    let data: Vec<f64> = (0..m * k)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    let a = DMatrixView::from_slice_with_strides(&data, m, k, k, 1);
    assert_eq!(a.strides(), (k, 1));

    assert_close(&dot(&a, &b), &naive_ref(&a, &b), 1e-10);

    // gemm() accumulates (alpha=1.5, beta=2) into a row-major C this time
    let exp = {
        let mut c0 = rand2(m, n, 8);
        let prod = naive_ref(&a, &b);
        for i in 0..m {
            for j in 0..n {
                c0[(i, j)] = 2.0 * c0[(i, j)] + 1.5 * prod[(i, j)];
            }
        }
        c0
    };
    let c0 = rand2(m, n, 8);
    let mut cdata: Vec<f64> = (0..m * n).map(|idx| c0[(idx / n, idx % n)]).collect();
    let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut cdata, m, n, n, 1);
    gemm(1.5, &a, &b, 2.0, &mut c, Parallelism::Rayon(0));
    assert_close(&c, &exp, 1e-10);
}

#[test]
fn non_contiguous_stepped_view() {
    let (m, k, n) = (10usize, 8, 6);
    // big has 2x the rows; a row step of 1 selects every other row (row stride 2, non-contiguous)
    // and the (1, 0) start offsets into row 1
    let big = rand2(2 * m, k, 9);
    let a = big.view_with_steps((1, 0), (m, k), (1, 0));
    assert_eq!(a.strides().0, 2, "row stride should be non-contiguous");
    let b = rand2(k, n, 10);
    assert_close(&dot(&a, &b), &naive_ref(&a, &b), 1e-10);
}

#[test]
fn static_smatrix_inputs() {
    // static A times static B
    let a = SMatrix::<f64, 3, 4>::from_fn(|i, j| (i as f64) - 0.5 * (j as f64) + 1.0);
    let b = SMatrix::<f64, 4, 2>::from_fn(|i, j| 0.25 * (i as f64) * (i as f64) - (j as f64));
    assert_close(&dot(&a, &b), &naive_ref(&a, &b), 1e-12);

    // static A times dynamic B: dot's independent R/C generics on each operand allow mixing them
    let bd = rand2(4, 5, 12);
    assert_close(&dot(&a, &bd), &naive_ref(&a, &bd), 1e-10);
}

#[test]
fn accumulate_with_beta() {
    let a = rand2(12, 9, 11);
    let b = rand2(9, 7, 12);
    let mut c = rand2(12, 7, 13);
    let prod = naive_ref(&a, &b);
    let mut exp = c.clone();
    for i in 0..12 {
        for j in 0..7 {
            exp[(i, j)] = 2.0 * exp[(i, j)] + 1.5 * prod[(i, j)];
        }
    }
    gemm(1.5, &a, &b, 2.0, &mut c, Parallelism::Serial);
    assert_close(&c, &exp, 1e-10);
}

#[test]
#[should_panic(expected = "A.cols")]
fn inner_dim_mismatch_panics() {
    let a = rand2(3, 4, 1);
    let b = rand2(5, 2, 2); // A.cols=4 != B.rows=5
    let mut c = rand2(3, 2, 3);
    gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Serial);
}

#[test]
#[should_panic(expected = "C.rows")]
fn output_rows_mismatch_panics() {
    let a = rand2(3, 4, 1);
    let b = rand2(4, 2, 2);
    let mut c = rand2(5, 2, 3); // C.rows=5 != A.rows=3
    gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Serial);
}

/// `prepack_rhs` + `gemm_packed_b`, serial and parallel, matches the naive reference (C stays
/// column-major, `gemm_packed_b`'s required orientation)
#[test]
fn packed_b_matches_dot() {
    let (m, k, n) = (100usize, 64, 80);
    let a = rand2(m, k, 51);
    let b = rand2(k, n, 52);
    let exp = naive_ref(&a, &b);
    let packed = prepack_rhs(&b);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = DMatrix::<f64>::zeros(m, n); // column-major: the orientation gemm_packed_b requires
        gemm_packed_b(1.0, &a, &packed, 0.0, &mut c, par);
        assert_close(&c, &exp, 1e-10);
    }
}

/// `prepack_lhs` + `gemm_packed_a`, serial and parallel, matches the naive reference (C is
/// row-major, `gemm_packed_a`'s required orientation)
#[test]
fn packed_a_matches_dot() {
    let (m, k, n) = (96usize, 50, 72);
    let a = rand2(m, k, 61);
    let b = rand2(k, n, 62);
    let exp = naive_ref(&a, &b);
    let packed = prepack_lhs(&a);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut data = vec![0.0f64; m * n];
        // row-major, strides (n, 1): the orientation gemm_packed_a requires
        let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut data, m, n, n, 1);
        gemm_packed_a(1.0, &packed, &b, 0.0, &mut c, par);
        assert_close(&c, &exp, 1e-10);
    }
}

/// gemm_packed_b panics when C is row-major: that orientation would swap A and B, invalidating the
/// prepacked RHS
#[test]
#[should_panic]
fn packed_b_row_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand2(m, k, 71);
    let b = rand2(k, n, 72);
    let packed = prepack_rhs(&b);
    let mut data = vec![0.0f64; m * n];
    let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut data, m, n, n, 1); // row-major: rejected orientation
    gemm_packed_b(1.0, &a, &packed, 0.0, &mut c, Parallelism::Serial);
}

/// gemm_packed_a panics when C is column-major: that orientation would keep A as the LHS,
/// invalidating the prepacked LHS
#[test]
#[should_panic]
fn packed_a_col_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand2(m, k, 81);
    let b = rand2(k, n, 82);
    let packed = prepack_lhs(&a);
    let mut c = DMatrix::<f64>::zeros(m, n); // column-major: rejected orientation
    gemm_packed_a(1.0, &packed, &b, 0.0, &mut c, Parallelism::Serial);
}

/// gemm_i8/dot_i8 accumulate i8 into i32 exactly like a naive i32 reference, including alpha/beta
/// and a strided (row-major) A view. Operand magnitudes stay small enough that the wrapping
/// (standard integer-GEMM) semantics never actually fire, so the comparison is exact
#[cfg(feature = "int8")]
#[test]
fn i8_matches_reference() {
    use gemmkit_nalgebra::{dot_i8, gemm_i8};
    use nalgebra::DMatrixView;

    let randi8 = |r: usize, c: usize, seed: u64| -> DMatrix<i8> {
        let mut s = seed | 1;
        DMatrix::from_fn(r, c, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 17 - 8) as i8
        })
    };
    let refmul = |a: &DMatrix<i8>, b: &DMatrix<i8>| -> DMatrix<i32> {
        let (m, k) = a.shape();
        let (_, n) = b.shape();
        DMatrix::from_fn(m, n, |i, j| {
            (0..k).map(|p| a[(i, p)] as i32 * b[(p, j)] as i32).sum()
        })
    };

    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);

    // dot_i8 matches the naive i32 product
    assert_eq!(dot_i8(&a, &b), refmul(&a, &b));

    // gemm_i8 with alpha/beta accumulate, over a row-major (strided) A view
    let arm = randi8(m, k, 0x3);
    let adata: Vec<i8> = (0..m * k).map(|idx| arm[(idx / k, idx % k)]).collect();
    let a_view = DMatrixView::from_slice_with_strides(&adata, m, k, k, 1);
    let mut c = DMatrix::<i32>::from_fn(m, n, |i, j| (i * n + j) as i32 - 5);
    let c0 = c.clone();
    let (alpha, beta) = (3i32, -2i32);
    gemm_i8(alpha, &a_view, &b, beta, &mut c, Parallelism::Serial);
    let prod = {
        let ao = DMatrix::from_fn(m, k, |i, j| a_view[(i, j)]);
        refmul(&ao, &b)
    };
    let exp = DMatrix::from_fn(m, n, |i, j| alpha * prod[(i, j)] + beta * c0[(i, j)]);
    assert_eq!(c, exp);
}

/// f16 implements `GemmScalar` like f32/f64, so it flows through the same generic `dot` with no
/// adapter-specific code; checked against the f64 reference within f16's precision
#[cfg(feature = "half")]
#[test]
fn dot_f16_matches_reference() {
    use gemmkit::f16;
    let (m, k, n) = (16, 12, 10);
    let af = rand2(m, k, 21);
    let bf = rand2(k, n, 22);
    let a = af.map(f16::from_f64);
    let b = bf.map(f16::from_f64);
    let got = dot(&a, &b); // accumulates in f32 (f16's Acc), rounds back to f16 on store
    let exp = naive_ref(&af, &bf); // f64 reference, computed before rounding to f16
    for i in 0..m {
        for j in 0..n {
            assert!(
                (got[(i, j)].to_f64() - exp[(i, j)]).abs() < 1e-2,
                "f16 dot ({i},{j}): {} vs {}",
                got[(i, j)].to_f64(),
                exp[(i, j)]
            );
        }
    }
}

/// `dot_cplx` (plain `A*B`) and `gemm_cplx` (conjugation plus alpha/beta accumulate) match a naive
/// complex reference, including a conjugated, row-major (strided) A view
#[cfg(feature = "complex")]
#[test]
fn cplx_dot_and_conj_matches_reference() {
    use gemmkit::Complex;
    use gemmkit_nalgebra::{dot_cplx, gemm_cplx};
    use nalgebra::DMatrixView;

    type C = Complex<f64>;
    // Complex::norm needs `sqrt`, unavailable through the no_std num-complex re-export; use hypot
    let cabs = |z: C| z.re.hypot(z.im);
    let crand = |r: usize, c: usize, s: u64| -> DMatrix<C> {
        let re = rand2(r, c, s);
        let im = rand2(r, c, s ^ 0xABCD);
        DMatrix::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
    };
    // C = alpha*op(A)*op(B) + beta*C0, where op(X) = conj(X) when the matching flag is set
    let refgemm = |a: &DMatrix<C>,
                   ca: bool,
                   b: &DMatrix<C>,
                   cb: bool,
                   alpha: C,
                   beta: C,
                   c0: &DMatrix<C>|
     -> DMatrix<C> {
        let (m, k) = a.shape();
        let (_, n) = b.shape();
        DMatrix::from_fn(m, n, |i, j| {
            let mut acc = Complex::new(0.0, 0.0);
            for p in 0..k {
                let av = if ca { a[(i, p)].conj() } else { a[(i, p)] };
                let bv = if cb { b[(p, j)].conj() } else { b[(p, j)] };
                acc += av * bv;
            }
            beta * c0[(i, j)] + alpha * acc
        })
    };

    let (m, k, n) = (12, 9, 7);
    let a = crand(m, k, 1);
    let b = crand(k, n, 2);

    // dot_cplx matches plain A*B (alpha=1, beta=0, no conjugation)
    let got = dot_cplx(&a, &b);
    let zero = DMatrix::from_element(m, n, Complex::new(0.0, 0.0));
    let exp = refgemm(
        &a,
        false,
        &b,
        false,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        &zero,
    );
    for i in 0..m {
        for j in 0..n {
            assert!(
                cabs(got[(i, j)] - exp[(i, j)]) < 1e-10,
                "dot_cplx ({i},{j})"
            );
        }
    }

    // gemm_cplx with conj_a=true on a row-major (strided) A view, alpha/beta accumulate
    let arm = crand(m, k, 3);
    let adata: Vec<C> = (0..m * k).map(|idx| arm[(idx / k, idx % k)]).collect();
    let a_view = DMatrixView::from_slice_with_strides(&adata, m, k, k, 1);
    let a_dense = DMatrix::from_fn(m, k, |i, j| a_view[(i, j)]);
    let alpha = Complex::new(1.3, -0.4);
    let beta = Complex::new(0.5, 0.7);
    let mut c = crand(m, n, 4);
    let c0 = c.clone();
    gemm_cplx(
        alpha,
        &a_view,
        true,
        &b,
        false,
        beta,
        &mut c,
        Parallelism::Serial,
    );
    let exp2 = refgemm(&a_dense, true, &b, false, alpha, beta, &c0);
    for i in 0..m {
        for j in 0..n {
            assert!(
                cabs(c[(i, j)] - exp2[(i, j)]) < 1e-9,
                "gemm_cplx conjA ({i},{j})"
            );
        }
    }
}

/// `gemm_fused` with a `PerRow` bias + `ReLU` is bit-identical to plain `gemm` followed by the same
/// scalar map; `None`/`None` is bit-identical to `gemm` alone. Runs both Serial and Rayon(0)
#[cfg(feature = "epilogue")]
#[test]
fn fused_matches_plain_then_map() {
    use gemmkit_nalgebra::{Activation, Bias, gemm_fused};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand2(m, k, 101);
    let b = rand2(k, n, 102);
    let c0 = rand2(m, n, 103);
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        // bias + ReLU case
        let mut c_fused = c0.clone();
        gemm_fused(
            alpha,
            &a,
            &b,
            beta,
            &mut c_fused,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_ref = c0.clone();
        gemm(alpha, &a, &b, beta, &mut c_ref, par);
        for i in 0..m {
            for j in 0..n {
                let v = c_ref[(i, j)] + bias[i];
                let want = if v > 0.0 { v } else { 0.0 };
                assert_eq!(
                    c_fused[(i, j)].to_bits(),
                    want.to_bits(),
                    "fused PerRow+ReLU ({i},{j})"
                );
            }
        }
        // None/None case: must equal plain gemm bit-for-bit
        let mut c_id = c0.clone();
        gemm_fused(alpha, &a, &b, beta, &mut c_id, None, None, par);
        let mut c_plain = c0.clone();
        gemm(alpha, &a, &b, beta, &mut c_plain, par);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c_id[(i, j)].to_bits(),
                    c_plain[(i, j)].to_bits(),
                    "fused None/None ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_map` and its `_with` twin are bit-identical to plain `gemm` followed by mapping each
/// output through the same closure `f(value, row, col)`. `f` is asymmetric in `(row, col)` and
/// captures a lookup table by reference, exercising both the user-frame coordinates and the
/// adapter's environment capture
#[cfg(feature = "epilogue")]
#[test]
fn map_matches_plain_then_map() {
    use gemmkit::Workspace;
    use gemmkit_nalgebra::{gemm_map, gemm_map_with};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand2(m, k, 111);
    let b = rand2(k, n, 112);
    let c0 = rand2(m, n, 113);
    let lut: Vec<f64> = (0..5).map(|i| 0.5 + 0.3 * i as f64).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    let f = |v: f64, r: usize, c: usize| v.mul_add(lut[r % lut.len()], r as f64 - 0.25 * c as f64);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_map = c0.clone();
        gemm_map(alpha, &a, &b, beta, &mut c_map, &f, par);
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_map_with(&mut ws, alpha, &a, &b, beta, &mut c_with, &f, par);
        let mut c_ref = c0.clone();
        gemm(alpha, &a, &b, beta, &mut c_ref, par);
        for i in 0..m {
            for j in 0..n {
                let want = f(c_ref[(i, j)], i, j);
                assert_eq!(c_map[(i, j)].to_bits(), want.to_bits(), "map ({i},{j})");
                assert_eq!(
                    c_with[(i, j)].to_bits(),
                    want.to_bits(),
                    "map_with ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_i8_requant` / `gemm_i8_requant_u8`, per-tensor and per-row scale, are bit-exact against an
/// independent round-half-to-even-then-clamp scalar model applied to the exact `i32` accumulator
/// from `dot_i8`
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[test]
fn requant_matches_scalar_model() {
    use gemmkit_nalgebra::{RequantScale, Requantize, dot_i8, gemm_i8_requant, gemm_i8_requant_u8};

    let randi8 = |r: usize, c: usize, seed: u64| -> DMatrix<i8> {
        let mut s = seed | 1;
        DMatrix::from_fn(r, c, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 51 - 25) as i8
        })
    };
    let ref_i8 = |acc: i32, bias: i32, scale: f32, zp: i32| -> i8 {
        let scaled = (f64::from(acc.wrapping_add(bias)) * f64::from(scale)).round_ties_even();
        (scaled as i64).saturating_add(zp as i64).clamp(-128, 127) as i8
    };
    let ref_u8 = |acc: i32, bias: i32, scale: f32, zp: i32| -> u8 {
        let scaled = (f64::from(acc.wrapping_add(bias)) * f64::from(scale)).round_ties_even();
        (scaled as i64).saturating_add(zp as i64).clamp(0, 255) as u8
    };

    let (m, k, n) = (17usize, 20, 13);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);
    let acc = dot_i8(&a, &b);
    let bias: Vec<i32> = (0..m).map(|i| 40 * i as i32 - 200).collect();
    let (scale, zp_i8, zp_u8) = (0.05f32, -7i32, 30i32);
    // per-row (per-channel) scales: must be finite and > 0
    let scales: Vec<f32> = (0..m).map(|i| 0.01 * (1 + i % 5) as f32).collect();

    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = DMatrix::from_element(m, n, 0i8);
        let req = Requantize {
            scale: RequantScale::PerTensor(scale),
            zero_point: zp_i8,
            bias: Some(&bias),
        };
        gemm_i8_requant(&a, &b, req, &mut c, par);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c[(i, j)],
                    ref_i8(acc[(i, j)], bias[i], scale, zp_i8),
                    "requant i8 ({i},{j})"
                );
            }
        }
        let mut cu = DMatrix::from_element(m, n, 0u8);
        let requ = Requantize {
            scale: RequantScale::PerTensor(scale),
            zero_point: zp_u8,
            bias: None,
        };
        gemm_i8_requant_u8(&a, &b, requ, &mut cu, par);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    cu[(i, j)],
                    ref_u8(acc[(i, j)], 0, scale, zp_u8),
                    "requant u8 ({i},{j})"
                );
            }
        }
        // per-row scales: same model but with scales[i] in place of the per-tensor scale
        let mut cr = DMatrix::from_element(m, n, 0i8);
        gemm_i8_requant(
            &a,
            &b,
            Requantize {
                scale: RequantScale::PerRow(&scales),
                zero_point: zp_i8,
                bias: Some(&bias),
            },
            &mut cr,
            par,
        );
        let mut cru = DMatrix::from_element(m, n, 0u8);
        gemm_i8_requant_u8(
            &a,
            &b,
            Requantize {
                scale: RequantScale::PerRow(&scales),
                zero_point: zp_u8,
                bias: None,
            },
            &mut cru,
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    cr[(i, j)],
                    ref_i8(acc[(i, j)], bias[i], scales[i], zp_i8),
                    "requant per-row i8 ({i},{j})"
                );
                assert_eq!(
                    cru[(i, j)],
                    ref_u8(acc[(i, j)], 0, scales[i], zp_u8),
                    "requant per-row u8 ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_cplx_fused` with a `PerRow` complex bias, across 3 conjugation combinations, is
/// bit-identical to [`gemm_cplx`] followed by the same element-wise bias add
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[test]
fn cplx_fused_matches_gemm_cplx_then_add() {
    use gemmkit::Complex;
    use gemmkit_nalgebra::{Bias, gemm_cplx, gemm_cplx_fused};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> DMatrix<C> {
        let re = rand2(r, c, s);
        let im = rand2(r, c, s ^ 0xABCD);
        DMatrix::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
    };

    let (m, k, n) = (12usize, 9, 7);
    let a = crand(m, k, 301);
    let b = crand(k, n, 302);
    let c0 = crand(m, n, 303);
    let bias: Vec<C> = (0..m)
        .map(|i| Complex::new(0.4 * i as f64 - 1.0, 0.7))
        .collect();
    let alpha = Complex::new(1.3, -0.4);
    let beta = Complex::new(0.5, 0.7);

    for &(conj_a, conj_b) in &[(false, false), (true, false), (false, true)] {
        let mut c_fused = c0.clone();
        gemm_cplx_fused(
            alpha,
            &a,
            conj_a,
            &b,
            conj_b,
            beta,
            &mut c_fused,
            Some(Bias::PerRow(&bias)),
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
        for i in 0..m {
            for j in 0..n {
                let want = c_ref[(i, j)] + bias[i];
                assert_eq!(
                    c_fused[(i, j)].re.to_bits(),
                    want.re.to_bits(),
                    "cplx_fused re conj=({conj_a},{conj_b}) ({i},{j})"
                );
                assert_eq!(
                    c_fused[(i, j)].im.to_bits(),
                    want.im.to_bits(),
                    "cplx_fused im conj=({conj_a},{conj_b}) ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_packed_b_fused` and its `_with` twin, with a `PerRow` bias + `ReLU`, are bit-identical to
/// plain `gemm_packed_b` off the same prepacked handle followed by the same scalar map, into a
/// column-major C
#[cfg(feature = "epilogue")]
#[test]
fn packed_b_fused_matches_packed_then_map() {
    use gemmkit::Workspace;
    use gemmkit_nalgebra::{
        Activation, Bias, gemm_packed_b, gemm_packed_b_fused, gemm_packed_b_fused_with, prepack_rhs,
    };
    let (m, k, n) = (100usize, 64, 80);
    let a = rand2(m, k, 251);
    let b = rand2(k, n, 252);
    let c0 = rand2(m, n, 253); // column-major: DMatrix's default layout, and gemm_packed_b's required orientation
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    let packed = prepack_rhs(&b);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_fused = c0.clone();
        gemm_packed_b_fused(
            alpha,
            &a,
            &packed,
            beta,
            &mut c_fused,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_packed_b_fused_with(
            &mut ws,
            alpha,
            &a,
            &packed,
            beta,
            &mut c_with,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_ref = c0.clone();
        gemm_packed_b(alpha, &a, &packed, beta, &mut c_ref, par);
        for i in 0..m {
            for j in 0..n {
                let v = c_ref[(i, j)] + bias[i];
                let want = if v > 0.0 { v } else { 0.0 };
                assert_eq!(
                    c_fused[(i, j)].to_bits(),
                    want.to_bits(),
                    "b_fused ({i},{j})"
                );
                assert_eq!(
                    c_with[(i, j)].to_bits(),
                    want.to_bits(),
                    "b_fused_with ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_packed_a_fused` and its `_with` (caller-owned `Workspace`) twin, with a `PerRow` bias +
/// `ReLU`, are bit-identical to plain `gemm_packed_a` off the same prepacked handle followed by the
/// same scalar map, into a row-major C (`gemm_packed_a`'s required orientation)
#[cfg(feature = "epilogue")]
#[test]
fn packed_a_fused_matches_packed_then_map() {
    use gemmkit::Workspace;
    use gemmkit_nalgebra::{
        Activation, Bias, gemm_packed_a, gemm_packed_a_fused, gemm_packed_a_fused_with, prepack_lhs,
    };
    use nalgebra::DMatrixViewMut;
    let (m, k, n) = (96usize, 50, 72);
    let a = rand2(m, k, 261);
    let b = rand2(k, n, 262);
    let c0 = rand2(m, n, 263);
    // row-major layout: element (i, j) lives at i*n + j
    let mut base = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            base[i * n + j] = c0[(i, j)];
        }
    }
    let bias: Vec<f64> = (0..m).map(|i| 0.3 * i as f64 - 1.5).collect();
    let (alpha, beta) = (0.9f64, 0.7);
    let packed = prepack_lhs(&a);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut data_f = base.clone();
        {
            let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut data_f, m, n, n, 1);
            gemm_packed_a_fused(
                alpha,
                &packed,
                &b,
                beta,
                &mut c,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
        }
        // same args, through the _with (caller-owned Workspace) entry
        let mut data_w = base.clone();
        {
            let mut ws = Workspace::new();
            let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut data_w, m, n, n, 1);
            gemm_packed_a_fused_with(
                &mut ws,
                alpha,
                &packed,
                &b,
                beta,
                &mut c,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
        }
        let mut data_r = base.clone();
        {
            let mut c = DMatrixViewMut::from_slice_with_strides_mut(&mut data_r, m, n, n, 1);
            gemm_packed_a(alpha, &packed, &b, beta, &mut c, par);
        }
        for i in 0..m {
            for j in 0..n {
                let v = data_r[i * n + j] + bias[i];
                let want = if v > 0.0 { v } else { 0.0 };
                assert_eq!(
                    data_f[i * n + j].to_bits(),
                    want.to_bits(),
                    "a_fused ({i},{j})"
                );
                assert_eq!(
                    data_w[i * n + j].to_bits(),
                    want.to_bits(),
                    "a_fused_with ({i},{j})"
                );
            }
        }
    }
}
