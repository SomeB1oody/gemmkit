//! ndarray adapter correctness: accepts both `&Array2` and `ArrayView2`, handles
//! C-order / F-order / transposed / reversed views, and `dot` matches ndarray's
//! own `.dot()`.

use approx::assert_relative_eq;
use ndarray::Array2;

use gemmkit::Parallelism;
use gemmkit_ndarray::{dot, gemm};

fn rand2(r: usize, c: usize, seed: u64) -> Array2<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Array2::from_shape_fn((r, c), |_| {
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

/// The `half` feature forwards `gemmkit/half`, so the *same generic* `dot`/`gemm`
/// serve `f16` (a `GemmScalar`) with no adapter-specific code — proven here against
/// the `f64` reference at a 16-bit tolerance.
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

/// The `complex` feature's dedicated adapters: `dot_cplx` (plain `A·B`) and `gemm_cplx`
/// (with conj + accumulate), checked against a naive complex reference — including a
/// transposed (F-order) conjugated A view, the case the raw `gemm_cplx_unchecked` path
/// exists for.
#[cfg(feature = "complex")]
#[test]
fn cplx_dot_and_conj_matches_reference() {
    use gemmkit::Complex;
    use gemmkit_ndarray::{dot_cplx, gemm_cplx};
    use ndarray::Array2;

    type C = Complex<f64>;
    // gemmkit pulls num-complex `no_std`, so `Complex::norm` (needs `sqrt`) isn't in
    // scope on its re-export; compute the magnitude with `f64::hypot` instead.
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
