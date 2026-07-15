//! ndarray adapter correctness: accepts both `&Array2` and `ArrayView2`, handles
//! C-order / F-order / transposed / reversed views, and `dot` matches ndarray's
//! own `.dot()`.

use approx::assert_relative_eq;
use ndarray::{Array2, Array3, Axis};

use gemmkit::Parallelism;
use gemmkit_ndarray::{
    dot, dot_batched, gemm, gemm_batched, gemm_packed_a, gemm_packed_b, prepack_lhs, prepack_rhs,
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

#[test]
fn dot_matches_ndarray() {
    for &(m, k, n) in &[
        (2, 2, 2),
        (7, 9, 5),
        (32, 40, 24),
        (64, 64, 64),
        (100, 1, 80),
    ] {
        let a = rand2(m, k, 1 + m as u64);
        let b = rand2(k, n, 2 + n as u64);
        let got = dot(&a, &b);
        let exp = a.dot(&b);
        assert_relative_eq!(got, exp, max_relative = 1e-10);
    }
}

#[test]
fn accepts_view_and_owned() {
    let a = rand2(8, 6, 3);
    let b = rand2(6, 5, 4);
    // &Array2
    let c1 = dot(&a, &b);
    // ArrayView2 via .view()
    let c2 = dot(&a.view(), &b.view());
    assert_relative_eq!(c1, c2, max_relative = 1e-12);
}

#[test]
fn transposed_and_fortran_layouts() {
    let m = 16;
    let k = 12;
    let n = 10;
    let a = rand2(m, k, 5);
    let b = rand2(k, n, 6);

    // A transposed: build a (k,m) array and transpose to a (m,k) view (col-major).
    let at = rand2(k, m, 7);
    let a_view = at.t(); // shape (m,k), F-order strides
    let exp = a_view.dot(&b);
    let got = dot(&a_view, &b);
    assert_relative_eq!(got, exp, max_relative = 1e-10);

    // F-order C output via gemm into a column-major array.
    use ndarray::ShapeBuilder;
    let mut c = Array2::<f64>::zeros((m, n).f());
    gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Rayon(0));
    assert_relative_eq!(c, a.dot(&b), max_relative = 1e-10);
}

#[test]
fn reversed_view_negative_strides() {
    let a = rand2(10, 8, 9);
    let b = rand2(8, 6, 10);
    // Reverse A along rows → negative row stride.
    let a_rev = a.slice(ndarray::s![..;-1, ..]);
    let exp = a_rev.dot(&b);
    let got = dot(&a_rev, &b);
    assert_relative_eq!(got, exp, max_relative = 1e-10);
}

#[test]
fn accumulate_with_beta() {
    let a = rand2(12, 9, 11);
    let b = rand2(9, 7, 12);
    let mut c = rand2(12, 7, 13);
    let exp = 2.0 * &c + 1.5 * a.dot(&b);
    gemm(1.5, &a, &b, 2.0, &mut c, Parallelism::Serial);
    assert_relative_eq!(c, exp, max_relative = 1e-10);
}

#[test]
fn dot_batched_matches_per_element_dot() {
    let (batch, m, k, n) = (5usize, 7, 9, 6);
    let a = rand3(batch, m, k, 11);
    let b = rand3(batch, k, n, 22);
    let got = dot_batched(&a, &b);
    for e in 0..batch {
        let ae = a.index_axis(Axis(0), e);
        let be = b.index_axis(Axis(0), e);
        assert_relative_eq!(
            got.index_axis(Axis(0), e).to_owned(),
            ae.dot(&be),
            max_relative = 1e-10
        );
    }
}

#[test]
fn gemm_batched_matches_per_element_loop() {
    let (batch, m, k, n) = (4usize, 8, 5, 6);
    let a = rand3(batch, m, k, 31);
    let b = rand3(batch, k, n, 32);
    let c0 = rand3(batch, m, n, 33);
    let (alpha, beta) = (0.7f64, 1.3);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = c0.clone();
        gemm_batched(alpha, &a, &b, beta, &mut c, par);
        for e in 0..batch {
            let ae = a.index_axis(Axis(0), e);
            let be = b.index_axis(Axis(0), e);
            let exp = alpha * &ae.dot(&be) + beta * &c0.index_axis(Axis(0), e).to_owned();
            assert_relative_eq!(
                c.index_axis(Axis(0), e).to_owned(),
                exp,
                max_relative = 1e-10
            );
        }
    }
}

/// Batched over a permuted-axes (non-contiguous) 3-D view: strides read straight through, no copy.
#[test]
fn gemm_batched_permuted_axes_view() {
    let (batch, m, k, n) = (3usize, 6, 4, 5);
    let araw = rand3(batch, k, m, 41); // (batch, k, m)
    let a = araw.view().permuted_axes([0, 2, 1]); // (batch, m, k), strided view
    let b = rand3(batch, k, n, 42);
    let got = dot_batched(&a, &b);
    for e in 0..batch {
        let ae = a.index_axis(Axis(0), e);
        let be = b.index_axis(Axis(0), e);
        assert_relative_eq!(
            got.index_axis(Axis(0), e).to_owned(),
            ae.dot(&be),
            max_relative = 1e-10
        );
    }
}

/// Prepacked RHS (`prepack_rhs` + `gemm_packed_b`) into a column-major-ish C matches `dot`.
#[test]
fn packed_b_matches_dot() {
    use ndarray::ShapeBuilder;
    let (m, k, n) = (100usize, 64, 80);
    let a = rand2(m, k, 51);
    let b = rand2(k, n, 52);
    let packed = prepack_rhs(&b);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = Array2::<f64>::zeros((m, n).f()); // column-major (packed_b orientation)
        gemm_packed_b(1.0, &a, &packed, 0.0, &mut c, par);
        assert_relative_eq!(c, a.dot(&b), max_relative = 1e-10);
    }
}

/// Prepacked LHS (`prepack_lhs` + `gemm_packed_a`) into a row-major-ish C matches `dot`.
#[test]
fn packed_a_matches_dot() {
    let (m, k, n) = (96usize, 50, 72);
    let a = rand2(m, k, 61);
    let b = rand2(k, n, 62);
    let packed = prepack_lhs(&a);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = Array2::<f64>::zeros((m, n)); // row-major (packed_a orientation)
        gemm_packed_a(1.0, &packed, &b, 0.0, &mut c, par);
        assert_relative_eq!(c, a.dot(&b), max_relative = 1e-10);
    }
}

/// The i8 adapter (`gemm_i8`/`dot_i8`) accumulates `i8` inputs into an `i32` output; checked
/// against a naive `i32` reference, including a transposed (F-order) A view. Values stay in range
/// so the wrapping semantics never fire and the comparison is exact.
#[cfg(feature = "int8")]
#[test]
fn i8_matches_reference() {
    use gemmkit_ndarray::{dot_i8, gemm_i8};

    let randi8 = |r: usize, c: usize, seed: u64| -> Array2<i8> {
        let mut s = seed | 1;
        Array2::from_shape_fn((r, c), |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 17 - 8) as i8
        })
    };
    let refmul = |a: &Array2<i8>, b: &Array2<i8>| -> Array2<i32> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        Array2::from_shape_fn((m, n), |(i, j)| {
            (0..k).map(|p| a[(i, p)] as i32 * b[(p, j)] as i32).sum()
        })
    };

    let (m, k, n) = (16usize, 12, 10);
    let a = randi8(m, k, 0x1);
    let b = randi8(k, n, 0x2);

    // dot_i8 == naive product.
    assert_eq!(dot_i8(&a, &b), refmul(&a, &b));

    // gemm_i8 with alpha/beta accumulate over a transposed (F-order) A view.
    let at = randi8(k, m, 0x3);
    let a_view = at.t(); // (m, k), F-order strides
    let mut c = Array2::<i32>::from_shape_fn((m, n), |(i, j)| (i * n + j) as i32 - 5);
    let c0 = c.clone();
    let (alpha, beta) = (3i32, -2i32);
    gemm_i8(alpha, &a_view, &b, beta, &mut c, Parallelism::Serial);
    let prod = refmul(&a_view.to_owned(), &b);
    let exp = Array2::from_shape_fn((m, n), |(i, j)| alpha * prod[(i, j)] + beta * c0[(i, j)]);
    assert_eq!(c, exp);
}

/// `f16` (a `GemmScalar`) flows through the same generic `dot`/`gemm` with no
/// adapter-specific code; checked against the `f64` reference at 16-bit tolerance.
#[cfg(feature = "half")]
#[test]
fn dot_f16_matches_reference() {
    use gemmkit::f16;
    let (m, k, n) = (16, 12, 10);
    let af = rand2(m, k, 21);
    let bf = rand2(k, n, 22);
    let a = af.mapv(f16::from_f64);
    let b = bf.mapv(f16::from_f64);
    let got = dot(&a, &b); // Array2<f16>, f32-accumulated then rounded to f16
    let exp = af.dot(&bf); // f64 reference
    for ((i, j), &g) in got.indexed_iter() {
        assert!(
            (g.to_f64() - exp[(i, j)]).abs() < 1e-2,
            "f16 dot ({i},{j}): {} vs {}",
            g.to_f64(),
            exp[(i, j)]
        );
    }
}

/// Complex adapters `dot_cplx` (plain `A·B`) and `gemm_cplx` (conj + accumulate),
/// checked against a naive reference — including a transposed (F-order) conjugated
/// A view, the case the raw `gemm_cplx_unchecked` path exists for.
#[cfg(feature = "complex")]
#[test]
fn cplx_dot_and_conj_matches_reference() {
    use gemmkit::Complex;
    use gemmkit_ndarray::{dot_cplx, gemm_cplx};
    use ndarray::Array2;

    type C = Complex<f64>;
    // `Complex::norm` needs `sqrt`, unavailable via the `no_std` num-complex re-export;
    // compute the magnitude with `f64::hypot` instead.
    let cabs = |z: C| z.re.hypot(z.im);
    let crand = |r: usize, c: usize, s: u64| -> Array2<C> {
        let re = rand2(r, c, s);
        let im = rand2(r, c, s ^ 0xABCD);
        Array2::from_shape_fn((r, c), |(i, j)| Complex::new(re[(i, j)], im[(i, j)]))
    };
    // Naive reference: C = alpha·op(A)·op(B) + beta·C0.
    let refgemm = |a: &Array2<C>,
                   ca: bool,
                   b: &Array2<C>,
                   cb: bool,
                   alpha: C,
                   beta: C,
                   c0: &Array2<C>|
     -> Array2<C> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        Array2::from_shape_fn((m, n), |(i, j)| {
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

    // dot_cplx == plain A·B.
    let got = dot_cplx(&a, &b);
    let zero = Array2::from_elem((m, n), Complex::new(0.0, 0.0));
    let exp = refgemm(
        &a,
        false,
        &b,
        false,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        &zero,
    );
    for ((i, j), &g) in got.indexed_iter() {
        assert!(cabs(g - exp[(i, j)]) < 1e-10, "dot_cplx ({i},{j})");
    }

    // gemm_cplx with conj-A on a transposed (F-order) A view + beta accumulate.
    let at = crand(k, m, 3); // (k,m); transpose to an (m,k) F-order view
    let a_view = at.t();
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
    let exp2 = refgemm(&a_view.to_owned(), true, &b, false, alpha, beta, &c0);
    for ((i, j), &g) in c.indexed_iter() {
        assert!(cabs(g - exp2[(i, j)]) < 1e-9, "gemm_cplx conjA ({i},{j})");
    }
}

/// `gemm_fused` (a `PerRow` bias + `ReLU`, and the identity `None`/`None` case) is **bit-identical**
/// to plain `gemm` followed by the same scalar map — gemmkit's `f32`/`f64` fused contract, mirrored
/// through the adapter over a general-stride C.
#[cfg(feature = "epilogue")]
#[test]
fn fused_matches_plain_then_map() {
    use gemmkit_ndarray::{Activation, Bias, gemm_fused};
    let (m, k, n) = (12usize, 9, 7);
    let a = rand2(m, k, 101);
    let b = rand2(k, n, 102);
    let c0 = rand2(m, n, 103);
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        // PerRow bias + ReLU.
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
        // None/None ≡ plain gemm, bit-for-bit.
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

/// A reversed (negative-stride) A view through `gemm_fused` must equal plain `gemm` on the **same**
/// reversed view then the same scalar map, **bit-for-bit**: the fused entry now forwards to gemmkit's
/// raw engine (no reversed-view rejection), exactly like the plain entry.
#[cfg(feature = "epilogue")]
#[test]
fn fused_reversed_view_matches_plain_then_map() {
    use gemmkit_ndarray::{Activation, Bias, gemm_fused};
    let (m, k, n) = (10usize, 8, 6);
    let a = rand2(m, k, 401);
    let b = rand2(k, n, 402);
    let c0 = rand2(m, n, 403);
    let a_rev = a.slice(ndarray::s![..;-1, ..]); // negative row stride
    assert!(
        a_rev.strides()[0] < 0,
        "reversed view should have a negative row stride"
    );
    let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
    let (alpha, beta) = (1.3f64, -0.7);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_fused = c0.clone();
        gemm_fused(
            alpha,
            &a_rev,
            &b,
            beta,
            &mut c_fused,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        let mut c_ref = c0.clone();
        gemm(alpha, &a_rev, &b, beta, &mut c_ref, par);
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

/// `gemm_batched_fused` (one shared `PerRow` bias + `ReLU`) is **bit-identical** to a loop of
/// [`gemm_fused`] calls, one per batch element — gemmkit's batched-fused headline property.
#[cfg(feature = "epilogue")]
#[test]
fn batched_fused_matches_loop_of_gemm_fused() {
    use gemmkit_ndarray::{Activation, Bias, gemm_batched_fused, gemm_fused};
    let (batch, m, k, n) = (4usize, 8, 5, 6);
    let a = rand3(batch, m, k, 201);
    let b = rand3(batch, k, n, 202);
    let c0 = rand3(batch, m, n, 203);
    let bias: Vec<f64> = (0..m).map(|i| 0.3 * i as f64 - 1.0).collect();
    let (alpha, beta) = (0.9f64, 1.1);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_b = c0.clone();
        gemm_batched_fused(
            alpha,
            &a,
            &b,
            beta,
            &mut c_b,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        for e in 0..batch {
            let ae = a.index_axis(Axis(0), e);
            let be = b.index_axis(Axis(0), e);
            let mut ce = c0.index_axis(Axis(0), e).to_owned();
            gemm_fused(
                alpha,
                &ae,
                &be,
                beta,
                &mut ce,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        c_b[(e, i, j)].to_bits(),
                        ce[(i, j)].to_bits(),
                        "batched_fused ({e},{i},{j})"
                    );
                }
            }
        }
    }
}

/// `gemm_i8_requant` / `gemm_i8_requant_u8` are **bit-exact** against an independent scalar model
/// (round-half-to-even, clamp) applied to the exact `i32` accumulator from `dot_i8`.
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[test]
fn requant_matches_scalar_model() {
    use gemmkit_ndarray::{Requantize, dot_i8, gemm_i8_requant, gemm_i8_requant_u8};

    let randi8 = |r: usize, c: usize, seed: u64| -> Array2<i8> {
        let mut s = seed | 1;
        Array2::from_shape_fn((r, c), |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 % 51 - 25) as i8
        })
    };
    // Independent contract: round-ties-even scale, integer zero-point join, saturating clamp.
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

    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        // i8 output, with bias.
        let mut c = Array2::<i8>::zeros((m, n));
        let req = Requantize {
            scale,
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
        // u8 output, no bias.
        let mut cu = Array2::<u8>::zeros((m, n));
        let requ = Requantize {
            scale,
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
    }
}

/// `gemm_cplx_fused` (a `PerRow` complex bias, with conjugation) is **bit-identical** to
/// [`gemm_cplx`] followed by the same element-wise bias add.
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[test]
fn cplx_fused_matches_gemm_cplx_then_add() {
    use gemmkit::Complex;
    use gemmkit_ndarray::{Bias, gemm_cplx, gemm_cplx_fused};

    type C = Complex<f64>;
    let crand = |r: usize, c: usize, s: u64| -> Array2<C> {
        let re = rand2(r, c, s);
        let im = rand2(r, c, s ^ 0xABCD);
        Array2::from_shape_fn((r, c), |(i, j)| Complex::new(re[(i, j)], im[(i, j)]))
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
