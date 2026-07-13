//! faer adapter correctness: accepts owned `Mat` (faer's native column-major layout), a
//! transposed view, a reversed (negative-stride) view, and an offset sub-matrix; `dot`/`gemm`
//! match a naive reference independent of faer's own matmul; dimension mismatches and wrong
//! prepacked-C orientations panic.

use faer::{Mat, MatMut, MatRef};
use gemmkit::Parallelism;

use gemmkit_faer::{dot, gemm, gemm_packed_a, gemm_packed_b, prepack_lhs, prepack_rhs};

/// Deterministic column-major `Mat<f64>` fill (xorshift), values in `[-0.5, 0.5)`.
fn rand_mat(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Mat::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// Naive triple-loop `A·B` reference, independent of faer's own matmul. Reads both operands
/// through faer indexing, so it honours whatever strides a view carries.
fn naive_ref(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    Mat::from_fn(m, n, |i, j| (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum())
}

/// Element-wise (relative + absolute) comparison of two views.
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
    // The doc example, by hand: [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]].
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

/// A transposed view (`.transpose()` on a column-major matrix yields non-unit row stride) reads
/// straight through with no copy.
#[test]
fn transposed_view() {
    let (m, k, n) = (9usize, 7, 5);
    // `at` is k×m column-major; transposing gives an m×k view whose row stride is `at`'s column
    // stride (non-unit) — the "row-major A" case.
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

/// A reversed view carries a negative row stride; gemm must honour it (and accumulate with beta).
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

/// An offset sub-matrix moves the base pointer and keeps a non-contiguous column stride.
#[test]
fn submatrix_offset_view() {
    let (m, k, n) = (10usize, 8, 6);
    // A source with twice the rows; take rows 1..1+m → base pointer offset by 1, column stride 2m.
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
    let b = rand_mat(5, 2, 2); // 4 != 5
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

/// Prepacked RHS (`prepack_rhs` + `gemm_packed_b`) into a column-major C matches the reference.
#[test]
fn packed_b_matches_dot() {
    let (m, k, n) = (100usize, 64, 80);
    let a = rand_mat(m, k, 51);
    let b = rand_mat(k, n, 52);
    let exp = naive_ref(a.as_dyn_stride(), b.as_dyn_stride());
    let packed = prepack_rhs(b.as_dyn_stride());
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = Mat::<f64>::from_fn(m, n, |_, _| 0.0); // column-major (packed_b orientation)
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

/// Prepacked LHS (`prepack_lhs` + `gemm_packed_a`) into a row-major C matches the reference.
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
            // row-major C (packed_a orientation): strides (n, 1).
            let c = MatMut::from_row_major_slice_mut(&mut data, m, n);
            gemm_packed_a(1.0, &packed, b.as_dyn_stride(), 0.0, c, par);
        }
        let got = MatRef::from_row_major_slice(&data, m, n);
        assert_close(got, exp.as_dyn_stride(), 1e-10);
    }
}

/// A prepacked RHS cannot serve a row-major C — gemmkit rejects the swapped orientation.
#[test]
#[should_panic]
fn packed_b_row_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand_mat(m, k, 71);
    let b = rand_mat(k, n, 72);
    let packed = prepack_rhs(b.as_dyn_stride());
    let mut data = vec![0.0f64; m * n];
    let c = MatMut::from_row_major_slice_mut(&mut data, m, n); // row-major
    gemm_packed_b(1.0, a.as_dyn_stride(), &packed, 0.0, c, Parallelism::Serial);
}

/// A prepacked LHS cannot serve a column-major C — gemmkit rejects the orientation.
#[test]
#[should_panic]
fn packed_a_col_major_c_panics() {
    let (m, k, n) = (8usize, 6, 5);
    let a = rand_mat(m, k, 81);
    let b = rand_mat(k, n, 82);
    let packed = prepack_lhs(a.as_dyn_stride());
    let mut c = Mat::<f64>::from_fn(m, n, |_, _| 0.0); // column-major
    gemm_packed_a(
        1.0,
        &packed,
        b.as_dyn_stride(),
        0.0,
        c.as_dyn_stride_mut(),
        Parallelism::Serial,
    );
}

/// The i8 adapter (`gemm_i8`/`dot_i8`) accumulates `i8` inputs into an `i32` output; checked
/// against a naive `i32` reference including a transposed (non-unit-stride) A view. Values stay in
/// range so the wrapping semantics never fire and the comparison is exact.
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

    // dot_i8 == naive product.
    let got = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());
    let exp = refmul(a.as_dyn_stride(), b.as_dyn_stride());
    for i in 0..m {
        for j in 0..n {
            assert_eq!(got[(i, j)], exp[(i, j)]);
        }
    }

    // gemm_i8 with alpha/beta accumulate over a transposed (non-unit-stride) A view.
    let at = randi8(k, m, 0x3);
    let a_view = at.as_dyn_stride().transpose(); // m×k, non-unit row stride
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

/// `f16` (a `GemmScalar`) flows through the same generic `dot` with no adapter-specific code;
/// checked against the `f64` reference at 16-bit tolerance.
#[cfg(feature = "half")]
#[test]
fn dot_f16_matches_reference() {
    use gemmkit::f16;
    let (m, k, n) = (16usize, 12, 10);
    let af = rand_mat(m, k, 21);
    let bf = rand_mat(k, n, 22);
    let a = Mat::from_fn(m, k, |i, j| f16::from_f64(af[(i, j)]));
    let b = Mat::from_fn(k, n, |i, j| f16::from_f64(bf[(i, j)]));
    let got = dot(a.as_dyn_stride(), b.as_dyn_stride()); // f32-accumulated then rounded to f16
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

/// Complex adapters `dot_cplx` (plain `A·B`) and `gemm_cplx` (conj + accumulate), checked against a
/// naive reference — including a conjugated, transposed A view, the case the raw `gemm_cplx`
/// path exists for.
#[cfg(feature = "complex")]
#[test]
fn cplx_dot_and_conj_matches_reference() {
    use gemmkit::Complex;
    use gemmkit_faer::{dot_cplx, gemm_cplx};

    type C = Complex<f64>;
    // `Complex::norm` needs `sqrt`, unavailable via the `no_std` num-complex re-export; use `hypot`.
    let cabs = |z: C| z.re.hypot(z.im);
    let crand = |r: usize, c: usize, s: u64| -> Mat<C> {
        let re = rand_mat(r, c, s);
        let im = rand_mat(r, c, s ^ 0xABCD);
        Mat::from_fn(r, c, |i, j| Complex::new(re[(i, j)], im[(i, j)]))
    };
    // Naive reference: C = alpha·op(A)·op(B) + beta·C0.
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

    // dot_cplx == plain A·B.
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

    // gemm_cplx with conj-A on a transposed A view + beta accumulate.
    let at = crand(k, m, 3);
    let a_view = at.as_dyn_stride().transpose(); // m×k
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
