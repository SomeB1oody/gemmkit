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
