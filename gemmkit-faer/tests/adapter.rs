//! faer adapter correctness: `dot`/`gemm` against a naive reference over faer's native
//! column-major layout plus transposed, reversed (negative-stride), and offset sub-matrix views;
//! the prepacked `gemm_packed_b`/`gemm_packed_a` entries against the same reference and their
//! required C orientation; dimension-mismatch panics; and, gated by feature, the i8, f16, complex,
//! fused-epilogue, map-epilogue, and i8-requantize entries, each checked bit-exact or bit-identical
//! against a scalar reference or the plain entry followed by the same epilogue

use faer::{Mat, MatMut, MatRef};
use gemmkit::Parallelism;

use gemmkit_faer::{dot, gemm, gemm_packed_a, gemm_packed_b, prepack_lhs, prepack_rhs};

/// Column-major `Mat<f64>` fill from a `seed`-derived xorshift64 stream, values in `[-0.5, 0.5)`
fn rand_mat(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Mat::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// Triple-loop `A*B` reference that never calls into faer's or gemmkit's matmul; indexes both
/// operands through faer's `MatRef`, so it honours whatever strides a view carries
fn naive_ref(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    Mat::from_fn(m, n, |i, j| (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum())
}

/// Element-wise comparison of 2 views: fails unless `abs(got - exp) <= tol + tol * abs(exp)`
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

#[test]
fn dot_matches_naive() {
    for &(m, k, n) in &[
        (2, 2, 2),
        (7, 9, 5),
        (32, 40, 24),
        (64, 64, 64),
        (100, 1, 80),
    ] {
        let a = rand_mat(m, k, 1 + m as u64);
        let b = rand_mat(k, n, 2 + n as u64);
        let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
        assert_close(
            dot(a.as_dyn_stride(), b.as_dyn_stride()).as_dyn_stride(),
            exp.as_dyn_stride(),
            1e-10,
        );
    }
}

#[test]
fn dot_small_exact() {
    // Same inputs as the crate-level doctest; worked by hand: [[1,2],[3,4]]*[[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
    let c = dot(a.as_dyn_stride(), b.as_dyn_stride());
    let exp = [[19.0, 22.0], [43.0, 50.0]];
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(c[(i, j)], exp[i][j]);
        }
    }
}

/// `.transpose()` on a column-major view flips the row/col strides without moving data, so the
/// resulting non-unit-row-stride A view still reads correctly through `dot`
#[test]
fn transposed_view() {
    let (m, k, n) = (9usize, 7, 5);
    // `at` is k x m column-major; transposing yields the m x k "row-major A" case: row stride
    // equals `at`'s column stride (k), not 1
    let at = rand_mat(k, m, 30);
    let a = at.as_dyn_stride().transpose();
    assert_eq!((a.nrows(), a.ncols()), (m, k));
    assert_ne!(
        a.row_stride(),
        1,
        "transposed A should have non-unit row stride"
    );
    let b = rand_mat(k, n, 31);
    let exp = naive_ref(a, b.as_dyn_stride());
    assert_close(
        dot(a, b.as_dyn_stride()).as_dyn_stride(),
        exp.as_dyn_stride(),
        1e-10,
    );
}

/// `reverse_rows()` yields a negative row stride; `gemm` must read it correctly while also
/// accumulating into C through a nonzero `beta`
#[test]
fn reversed_negative_stride_view() {
    let (m, k, n) = (8usize, 6, 5);
    let src = rand_mat(m, k, 50);
    let a = src.as_dyn_stride().reverse_rows();
    assert!(
        a.row_stride() < 0,
        "reversed view should have negative row stride"
    );
    let b = rand_mat(k, n, 51);
    let prod = naive_ref(a, b.as_dyn_stride());

    let mut c = rand_mat(m, n, 52);
    let c0 = c.clone();
    gemm(
        1.5,
        a,
        b.as_dyn_stride(),
        2.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
    for i in 0..m {
        for j in 0..n {
            let e = 2.0 * c0[(i, j)] + 1.5 * prod[(i, j)];
            assert!((c[(i, j)] - e).abs() < 1e-9, "reversed gemm ({i},{j})");
        }
    }
}

/// A row-offset `submatrix()` view shifts the base pointer while its column stride still reflects
/// the wider parent, so `dot` must follow the stride rather than assume tight packing
#[test]
fn submatrix_offset_view() {
    let (m, k, n) = (10usize, 8, 6);
    // Parent has 2m rows; rows 1..1+m are offset by 1 element, column stride stays the parent's 2m
    let big = rand_mat(2 * m, k, 40);
    let a = big.as_dyn_stride().submatrix(1, 0, m, k);
    assert_eq!((a.nrows(), a.ncols()), (m, k));
    let b = rand_mat(k, n, 41);
    let exp = naive_ref(a, b.as_dyn_stride());
    assert_close(
        dot(a, b.as_dyn_stride()).as_dyn_stride(),
        exp.as_dyn_stride(),
        1e-10,
    );
}

#[test]
fn accumulate_with_beta() {
    let a = rand_mat(12, 9, 11);
    let b = rand_mat(9, 7, 12);
    let mut c = rand_mat(12, 7, 13);
    let prod = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let c0 = c.clone();
    gemm(
        1.5,
        a.as_dyn_stride(),
        b.as_dyn_stride(),
        2.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
    for i in 0..12 {
        for j in 0..7 {
            let e = 2.0 * c0[(i, j)] + 1.5 * prod[(i, j)];
            assert!((c[(i, j)] - e).abs() < 1e-9, "beta accumulate ({i},{j})");
        }
    }
}

#[test]
#[should_panic(expected = "A.cols")]
fn inner_dim_mismatch_panics() {
    let a = rand_mat(3, 4, 1);
    let b = rand_mat(5, 2, 2); // A.cols 4 != B.rows 5
    let mut c = rand_mat(3, 2, 3);
    gemm(
        1.0,
        a.as_dyn_stride(),
        b.as_dyn_stride(),
        0.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "C.rows")]
fn output_rows_mismatch_panics() {
    let a = rand_mat(3, 4, 1);
    let b = rand_mat(4, 2, 2);
    let mut c = rand_mat(5, 2, 3); // C.rows 5 != A.rows 3
    gemm(
        1.0,
        a.as_dyn_stride(),
        b.as_dyn_stride(),
        0.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
}

/// `prepack_rhs` followed by `gemm_packed_b` into a column-major C, under both Serial and Rayon,
/// matches the naive reference
#[test]
fn packed_b_matches_dot() {
    let (m, k, n) = (100usize, 64, 80);
    let a = rand_mat(m, k, 51);
    let b = rand_mat(k, n, 52);
    let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let packed = prepack_rhs(b.as_dyn_stride());
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = Mat::<f64>::from_fn(m, n, |_, _| 0.0); // column-major, as gemm_packed_b requires
        gemm_packed_b(
            1.0,
            a.as_dyn_stride(),
            &packed,
            0.0,
            c.as_dyn_stride_mut(),
            par,
        );
        assert_close(c.as_dyn_stride(), exp.as_dyn_stride(), 1e-10);
    }
}

/// `prepack_lhs` followed by `gemm_packed_a` into a row-major C, under both Serial and Rayon,
/// matches the naive reference
#[test]
fn packed_a_matches_dot() {
    let (m, k, n) = (96usize, 50, 72);
    let a = rand_mat(m, k, 61);
    let b = rand_mat(k, n, 62);
    let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let packed = prepack_lhs(a.as_dyn_stride());
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut data = vec![0.0f64; m * n];
        {
            // row-major C, as gemm_packed_a requires: strides (n, 1)
            let c = MatMut::from_row_major_slice_mut(&mut data, m, n);
            gemm_packed_a(1.0, &packed, b.as_dyn_stride(), 0.0, c, par);
        }
        let got = MatRef::from_row_major_slice(&data, m, n);
        assert_close(got, exp.as_dyn_stride(), 1e-10);
    }
}

/// `gemm_packed_b` requires a column-major-ish C; a row-major C would swap A/B and invalidate the
/// prepacked RHS, so gemmkit rejects it
#[test]
#[should_panic]
fn packed_b_row_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand_mat(m, k, 71);
    let b = rand_mat(k, n, 72);
    let packed = prepack_rhs(b.as_dyn_stride());
    let mut data = vec![0.0f64; m * n];
    let c = MatMut::from_row_major_slice_mut(&mut data, m, n); // the rejected row-major layout
    gemm_packed_b(1.0, a.as_dyn_stride(), &packed, 0.0, c, Parallelism::Serial);
}

/// `gemm_packed_a` requires a row-major-ish C; a column-major C would keep A in the LHS role and
/// invalidate the prepacked LHS, so gemmkit rejects it
#[test]
#[should_panic]
fn packed_a_col_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand_mat(m, k, 81);
    let b = rand_mat(k, n, 82);
    let packed = prepack_lhs(a.as_dyn_stride());
    let mut c = Mat::<f64>::from_fn(m, n, |_, _| 0.0); // the rejected column-major layout
    gemm_packed_a(
        1.0,
        &packed,
        b.as_dyn_stride(),
        0.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
}

/// `dot_i8`/`gemm_i8` (`i8` inputs, `i32` accumulator) checked against a naive `i32` reference,
/// including a transposed (non-unit-stride) A view. Inputs stay in `[-8, 8]`, so the products and
/// their alpha/beta-scaled sums stay well inside `i32` range and the `i32` wraparound gemmkit's
/// contract allows never actually fires, making the comparison exact
#[cfg(feature = "int8")]
#[test]
fn i8_matches_reference() {
    use gemmkit_faer::{dot_i8, gemm_i8};

    let randi8 = |r: usize, c: usize, seed: u64| -> Mat<i8> {
        let mut s = seed | 1;
        Mat::from_fn(r, c, |_, _| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 17 - 8) as i8
        })
    };
    let refmul = |a: MatRef<'_, i8>, b: MatRef<'_, i8>| -> Mat<i32> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        Mat::from_fn(m, n, |i, j| {
            (0..k).map(|p| a[(i, p)] as i32 * b[(p, j)] as i32).sum()
        })
    };

    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);

    // dot_i8 against the naive i32 product
    let got = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());
    let exp = refmul(a.as_dyn_stride(), b.as_dyn_stride());
    for i in 0..m {
        for j in 0..n {
            assert_eq!(got[(i, j)], exp[(i, j)]);
        }
    }

    // gemm_i8 with an alpha/beta accumulate, A read through a transposed non-unit-stride view
    let at = randi8(k, m, 0x3);
    let a_view = at.as_dyn_stride().transpose(); // m x k, non-unit row stride
    let mut c = Mat::<i32>::from_fn(m, n, |i, j| (i * n + j) as i32 - 5);
    let c0 = c.clone();
    let (alpha, beta) = (3i32, -2i32);
    gemm_i8(
        alpha,
        a_view,
        b.as_dyn_stride(),
        beta,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
    let prod = refmul(a_view, b.as_dyn_stride());
    for i in 0..m {
        for j in 0..n {
            let e = alpha * prod[(i, j)] + beta * c0[(i, j)];
            assert_eq!(c[(i, j)], e);
        }
    }
}

/// `f16` (a `GemmScalar` under the `half` feature) flows through the same generic `dot` used for
/// `f64`, with no adapter-specific code; checked against an `f64` reference at a loose tolerance
/// appropriate for `f16`'s precision
#[cfg(feature = "half")]
#[test]
fn dot_f16_matches_reference() {
    use gemmkit::f16;
    let (m, k, n) = (16usize, 12, 10);
    let af = rand_mat(m, k, 21);
    let bf = rand_mat(k, n, 22);
    let a = Mat::from_fn(m, k, |i, j| f16::from_f64(af[(i, j)]));
    let b = Mat::from_fn(k, n, |i, j| f16::from_f64(bf[(i, j)]));
    let got = dot(a.as_dyn_stride(), b.as_dyn_stride()); // accumulates in f32, narrows to f16 once
    let exp = naive_ref(af.as_dyn_stride(), bf.as_dyn_stride());
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

/// `dot_cplx` (plain `A*B`) and `gemm_cplx` (conjugation plus a beta accumulate), checked against a
/// naive complex reference; `gemm_cplx` is also exercised with `conj_a` over a transposed A view,
/// the combination its conjugation flags exist for
#[cfg(feature = "complex")]
#[test]
fn cplx_dot_and_conj_matches_reference() {
    use gemmkit::Complex;
    use gemmkit_faer::{dot_cplx, gemm_cplx};

    type C = Complex<f64>;
    // `Complex::norm` is gated behind num-complex's std/libm feature, which gemmkit's re-export
    // does not enable; hypot on the f64 fields gives the same magnitude directly
    let cabs = |z: C| z.re.hypot(z.im);
    let crand = |r: usize, c: usize, s: u64| -> Mat<C> {
        let re = rand_mat(r, c, s);
        let im = rand_mat(r, c, s ^ 0xABCD);
        Mat::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
    };
    // C = alpha*op(A)*op(B) + beta*C0, op = conj when the matching ca/cb flag is set
    let refgemm = |a: MatRef<'_, C>,
                   ca: bool,
                   b: MatRef<'_, C>,
                   cb: bool,
                   alpha: C,
                   beta: C,
                   c0: MatRef<'_, C>|
     -> Mat<C> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        Mat::from_fn(m, n, |i, j| {
            let mut acc = Complex::new(0.0, 0.0);
            for p in 0..k {
                let av = if ca { a[(i, p)].conj() } else { a[(i, p)] };
                let bv = if cb { b[(p, j)].conj() } else { b[(p, j)] };
                acc += av * bv;
            }
            beta * c0[(i, j)] + alpha * acc
        })
    };

    let (m, k, n) = (12usize, 9, 7);
    let a = crand(m, k, 1);
    let b = crand(k, n, 2);

    // dot_cplx against the plain (non-conjugated) product
    let got = dot_cplx(a.as_dyn_stride(), b.as_dyn_stride());
    let zero = Mat::from_fn(m, n, |_, _| Complex::new(0.0, 0.0));
    let exp = refgemm(
        a.as_dyn_stride(),
        false,
        b.as_dyn_stride(),
        false,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        zero.as_dyn_stride(),
    );
    for i in 0..m {
        for j in 0..n {
            assert!(
                cabs(got[(i, j)] - exp[(i, j)]) < 1e-10,
                "dot_cplx ({i},{j})"
            );
        }
    }

    // gemm_cplx with conj_a set, A read through a transposed view, plus a beta accumulate
    let at = crand(k, m, 3);
    let a_view = at.as_dyn_stride().transpose(); // m x k
    let alpha = Complex::new(1.3, -0.4);
    let beta = Complex::new(0.5, 0.7);
    let mut c = crand(m, n, 4);
    let c0 = c.clone();
    gemm_cplx(
        alpha,
        a_view,
        true,
        b.as_dyn_stride(),
        false,
        beta,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
    let exp2 = refgemm(
        a_view,
        true,
        b.as_dyn_stride(),
        false,
        alpha,
        beta,
        c0.as_dyn_stride(),
    );
    for i in 0..m {
        for j in 0..n {
            assert!(
                cabs(c[(i, j)] - exp2[(i, j)]) < 1e-9,
                "gemm_cplx conjA ({i},{j})"
            );
        }
    }
}

/// `gemm_fused` matches gemmkit's fused contract bit-for-bit: a `PerRow` bias plus `ReLU` equals
/// plain `gemm` followed by the same bias-add-then-clamp, and `None`/`None` equals plain `gemm`
/// with no epilogue at all
#[cfg(feature = "epilogue")]
#[test]
fn fused_matches_plain_then_map() {
    use gemmkit_faer::{Activation, Bias, gemm_fused};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_mat(m, k, 101);
    let b = rand_mat(k, n, 102);
    let c0 = rand_mat(m, n, 103);
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        // PerRow bias followed by ReLU
        let mut c_fused = c0.clone();
        gemm_fused(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_fused.as_dyn_stride_mut(),
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
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
        // No bias, no activation: must equal plain gemm bit-for-bit
        let mut c_id = c0.clone();
        gemm_fused(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_id.as_dyn_stride_mut(),
            None,
            None,
            par,
        );
        let mut c_plain = c0.clone();
        gemm(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_plain.as_dyn_stride_mut(),
            par,
        );
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

/// `gemm_map` and `gemm_map_with` both equal plain `gemm` followed by the same per-element closure,
/// bit-for-bit. The closure treats row and column differently and captures a lookup table by
/// reference, so the test also checks that the adapter delivers the correct `(row, col)` and that
/// closure environment capture survives the call
#[cfg(feature = "epilogue")]
#[test]
fn map_matches_plain_then_map() {
    use gemmkit::Workspace;
    use gemmkit_faer::{gemm_map, gemm_map_with};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_mat(m, k, 111);
    let b = rand_mat(k, n, 112);
    let c0 = rand_mat(m, n, 113);
    let lut: Vec<f64> = (0..5).map(|i| 0.5 + 0.3 * i as f64).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    let f = |v: f64, r: usize, c: usize| v.mul_add(lut[r % lut.len()], r as f64 - 0.25 * c as f64);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_map = c0.clone();
        gemm_map(
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_map.as_dyn_stride_mut(),
            &f,
            par,
        );
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_map_with(
            &mut ws,
            alpha,
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            beta,
            c_with.as_dyn_stride_mut(),
            &f,
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

/// `gemm_fused` accepts a reversed (negative-stride) A view just like plain `gemm`: no special
/// rejection for it, so the result equals `gemm` on the same view then the same bias/ReLU, bit-for-bit
#[cfg(feature = "epilogue")]
#[test]
fn fused_reversed_view_matches_plain_then_map() {
    use gemmkit_faer::{Activation, Bias, gemm_fused};
    let (m, k, n) = (10usize, 8, 6);
    let asrc = rand_mat(m, k, 401);
    let a = asrc.as_dyn_stride().reverse_rows(); // negative row stride
    assert!(
        a.row_stride() < 0,
        "reversed view should have a negative row stride"
    );
    let b = rand_mat(k, n, 402); // fused_matches_plain_then_map covers the non-reversed case
    let c0 = rand_mat(m, n, 403);
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_fused = c0.clone();
        gemm_fused(
            alpha,
            a,
            b.as_dyn_stride(),
            beta,
            c_fused.as_dyn_stride_mut(),
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_ref = c0.clone();
        gemm(
            alpha,
            a,
            b.as_dyn_stride(),
            beta,
            c_ref.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                let v = c_ref[(i, j)] + bias[i];
                let want = if v > 0.0 { v } else { 0.0 };
                assert_eq!(
                    c_fused[(i, j)].to_bits(),
                    want.to_bits(),
                    "fused reversed ({i},{j})"
                );
            }
        }
    }
}

/// `gemm_i8_requant`/`gemm_i8_requant_u8` match, bit-exact, an independent scalar model (bias add,
/// scale, round-half-to-even, clamp to the output range) applied to the exact `i32` accumulator
/// `dot_i8` produces for the same `A`/`B`; covers per-tensor and per-row scale, with and without bias
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[test]
fn requant_matches_scalar_model() {
    use gemmkit_faer::{RequantScale, Requantize, dot_i8, gemm_i8_requant, gemm_i8_requant_u8};

    let randi8 = |r: usize, c: usize, seed: u64| -> Mat<i8> {
        let mut s = seed | 1;
        Mat::from_fn(r, c, |_, _| {
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
    let acc = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());
    let bias: Vec<i32> = (0..m).map(|i| 40 * i as i32 - 200).collect();
    let (scale, zp_i8, zp_u8) = (0.05f32, -7i32, 30i32);
    // per-row (per-channel) scales; requant_scale requires every element finite and > 0
    let scales: Vec<f32> = (0..m).map(|i| 0.01 * (1 + i % 5) as f32).collect();

    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = Mat::<i8>::from_fn(m, n, |_, _| 0);
        let req = Requantize {
            scale: RequantScale::PerTensor(scale),
            zero_point: zp_i8,
            bias: Some(&bias),
        };
        gemm_i8_requant(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            req,
            c.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c[(i, j)],
                    ref_i8(acc[(i, j)], bias[i], scale, zp_i8),
                    "requant i8 ({i},{j})"
                );
            }
        }
        let mut cu = Mat::<u8>::from_fn(m, n, |_, _| 0);
        let requ = Requantize {
            scale: RequantScale::PerTensor(scale),
            zero_point: zp_u8,
            bias: None,
        };
        gemm_i8_requant_u8(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            requ,
            cu.as_dyn_stride_mut(),
            par,
        );
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    cu[(i, j)],
                    ref_u8(acc[(i, j)], 0, scale, zp_u8),
                    "requant u8 ({i},{j})"
                );
            }
        }
        // per-row scales, against the same scalar model with scales[i] in place of scale (i8 case
        // keeps the bias, u8 case drops it, mirroring the per-tensor checks above)
        let mut cr = Mat::<i8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            Requantize {
                scale: RequantScale::PerRow(&scales),
                zero_point: zp_i8,
                bias: Some(&bias),
            },
            cr.as_dyn_stride_mut(),
            par,
        );
        let mut cru = Mat::<u8>::from_fn(m, n, |_, _| 0);
        gemm_i8_requant_u8(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            Requantize {
                scale: RequantScale::PerRow(&scales),
                zero_point: zp_u8,
                bias: None,
            },
            cru.as_dyn_stride_mut(),
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

/// `gemm_cplx_fused` with a `PerRow` complex bias matches `gemm_cplx` followed by the same
/// element-wise bias add, bit-for-bit, for every conjugation combination the loop covers
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[test]
fn cplx_fused_matches_gemm_cplx_then_add() {
    use gemmkit::Complex;
    use gemmkit_faer::{Bias, gemm_cplx, gemm_cplx_fused};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> Mat<C> {
        let re = rand_mat(r, c, s);
        let im = rand_mat(r, c, s ^ 0xABCD);
        Mat::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
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
            a.as_dyn_stride(),
            conj_a,
            b.as_dyn_stride(),
            conj_b,
            beta,
            c_fused.as_dyn_stride_mut(),
            Some(Bias::PerRow(&bias)),
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

/// `gemm_packed_b_fused` and `gemm_packed_b_fused_with`, given a `PerRow` bias plus `ReLU`, both
/// equal plain `gemm_packed_b` off the same prepacked handle followed by the same bias-add-then-clamp,
/// bit-for-bit, into a column-major C
#[cfg(feature = "epilogue")]
#[test]
fn packed_b_fused_matches_packed_then_map() {
    use gemmkit::Workspace;
    use gemmkit_faer::{
        Activation, Bias, gemm_packed_b, gemm_packed_b_fused, gemm_packed_b_fused_with, prepack_rhs,
    };
    let (m, k, n) = (100usize, 64, 80);
    let a = rand_mat(m, k, 351);
    let b = rand_mat(k, n, 352);
    let c0 = rand_mat(m, n, 353); // column-major, as gemm_packed_b_fused requires
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    let packed = prepack_rhs(b.as_dyn_stride());
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_fused = c0.clone();
        gemm_packed_b_fused(
            alpha,
            a.as_dyn_stride(),
            &packed,
            beta,
            c_fused.as_dyn_stride_mut(),
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_packed_b_fused_with(
            &mut ws,
            alpha,
            a.as_dyn_stride(),
            &packed,
            beta,
            c_with.as_dyn_stride_mut(),
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_ref = c0.clone();
        gemm_packed_b(
            alpha,
            a.as_dyn_stride(),
            &packed,
            beta,
            c_ref.as_dyn_stride_mut(),
            par,
        );
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

/// `gemm_packed_a_fused` and `gemm_packed_a_fused_with` (the latter with a caller-owned
/// `Workspace`), given a `PerRow` bias plus `ReLU`, both equal plain `gemm_packed_a` off the same
/// prepacked handle followed by the same bias-add-then-clamp, bit-for-bit, into a row-major C
#[cfg(feature = "epilogue")]
#[test]
fn packed_a_fused_matches_packed_then_map() {
    use gemmkit::Workspace;
    use gemmkit_faer::{
        Activation, Bias, gemm_packed_a, gemm_packed_a_fused, gemm_packed_a_fused_with, prepack_lhs,
    };
    let (m, k, n) = (96usize, 50, 72);
    let a = rand_mat(m, k, 361);
    let b = rand_mat(k, n, 362);
    let c0 = rand_mat(m, n, 363);
    // row-major base data, as gemm_packed_a_fused requires: element (i,j) lives at i*n + j
    let mut base = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            base[i * n + j] = c0[(i, j)];
        }
    }
    let bias: Vec<f64> = (0..m).map(|i| 0.3 * i as f64 - 1.5).collect();
    let (alpha, beta) = (0.9f64, 0.7);
    let packed = prepack_lhs(a.as_dyn_stride());
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut data_f = base.clone();
        {
            let c = MatMut::from_row_major_slice_mut(&mut data_f, m, n);
            gemm_packed_a_fused(
                alpha,
                &packed,
                b.as_dyn_stride(),
                beta,
                c,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
        }
        // gemm_packed_a_fused_with, same arguments plus an explicit Workspace
        let mut data_w = base.clone();
        {
            let mut ws = Workspace::new();
            let c = MatMut::from_row_major_slice_mut(&mut data_w, m, n);
            gemm_packed_a_fused_with(
                &mut ws,
                alpha,
                &packed,
                b.as_dyn_stride(),
                beta,
                c,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
        }
        let mut data_r = base.clone();
        {
            let c = MatMut::from_row_major_slice_mut(&mut data_r, m, n);
            gemm_packed_a(alpha, &packed, b.as_dyn_stride(), beta, c, par);
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
