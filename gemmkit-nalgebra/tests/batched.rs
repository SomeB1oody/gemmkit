//! Correctness tests for `gemm_batched`: 1 call over a slice of per-element `(&A, &B)` inputs
//! paired with a slice of `&mut C` outputs must reproduce a per-element loop of `gemm` bit-for-bit,
//! across a heterogeneous batch of shapes and mixed column-major (F-order) / row-major (C-order)
//! layouts, and stay bit-identical between serial and parallel. Count and inner-dimension
//! mismatches panic; an empty batch is a no-op

use gemmkit::Parallelism;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut};

use gemmkit_nalgebra::{gemm, gemm_batched};

/// Xorshift64 fill in `[-0.5, 0.5)`
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// The batch's 4 elements as `(m, k, n, A row-major, B row-major, C row-major)`; `false` selects
/// the natural column-major (F-order) layout, `true` a row-major (C-order) strided view
const ELEMS: [(usize, usize, usize, bool, bool, bool); 4] = [
    (5, 7, 3, false, false, false),
    (8, 4, 6, true, false, true),
    (2, 9, 4, false, true, false),
    (16, 5, 10, true, true, true),
];

/// Strides for an `r x c` matrix in the given layout: column-major `(1, r)` or row-major `(c, 1)`
fn strides(r: usize, c: usize, row_major: bool) -> (usize, usize) {
    if row_major { (c, 1) } else { (1, r) }
}

/// Runs the batch under `par`, asserts it matches a per-element `gemm(par)` loop bit-for-bit, and
/// returns the batched outputs so the caller can also compare across parallelism settings
fn run(par: Parallelism) -> Vec<Vec<f64>> {
    let (alpha, beta) = (1.3f64, -0.7f64);

    let a_data: Vec<Vec<f64>> = ELEMS
        .iter()
        .enumerate()
        .map(|(i, &(m, k, _, _, _, _))| fill(m * k, 1 + i as u64))
        .collect();
    let b_data: Vec<Vec<f64>> = ELEMS
        .iter()
        .enumerate()
        .map(|(i, &(_, k, n, _, _, _))| fill(k * n, 100 + i as u64))
        .collect();
    let c_init: Vec<Vec<f64>> = ELEMS
        .iter()
        .enumerate()
        .map(|(i, &(m, _, n, _, _, _))| fill(m * n, 200 + i as u64))
        .collect();

    // from_slice_with_strides only exists on the fully-dynamic-stride view (DMatrixView's own
    // default assumes a unit row stride), so Vec<_> lets inference name the real return type
    let a_views: Vec<_> = ELEMS
        .iter()
        .enumerate()
        .map(|(i, &(m, k, _, a_rm, _, _))| {
            let (rs, cs) = strides(m, k, a_rm);
            DMatrixView::from_slice_with_strides(&a_data[i], m, k, rs, cs)
        })
        .collect();
    let b_views: Vec<_> = ELEMS
        .iter()
        .enumerate()
        .map(|(i, &(_, k, n, _, b_rm, _))| {
            let (rs, cs) = strides(k, n, b_rm);
            DMatrixView::from_slice_with_strides(&b_data[i], k, n, rs, cs)
        })
        .collect();

    // batched: 1 call over the slice of (&A, &B) inputs and &mut C outputs
    let mut c_bat = c_init.clone();
    {
        let mut c_views: Vec<_> = c_bat
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| {
                let (m, _, n, _, _, c_rm) = ELEMS[i];
                let (rs, cs) = strides(m, n, c_rm);
                DMatrixViewMut::from_slice_with_strides_mut(buf, m, n, rs, cs)
            })
            .collect();
        let ab: Vec<_> = a_views.iter().zip(&b_views).collect();
        gemm_batched(alpha, &ab, beta, &mut c_views, par);
    }

    // reference: a loop of single gemm(par) calls, same per-element layout
    let mut c_ref = c_init.clone();
    for (i, &(m, _, n, _, _, c_rm)) in ELEMS.iter().enumerate() {
        let (rs, cs) = strides(m, n, c_rm);
        let mut cv = DMatrixViewMut::from_slice_with_strides_mut(&mut c_ref[i], m, n, rs, cs);
        gemm(alpha, &a_views[i], &b_views[i], beta, &mut cv, par);
    }

    assert_eq!(c_bat, c_ref, "gemm_batched != gemm() loop (par={par:?})");
    c_bat
}

#[test]
fn gemm_batched_matches_gemm_loop_and_par_reproducible() {
    let serial = run(Parallelism::Serial);
    let parallel = run(Parallelism::Rayon(0));
    assert_eq!(
        serial, parallel,
        "batched serial must equal parallel bit-for-bit"
    );
}

#[test]
#[should_panic(expected = "count")]
fn gemm_batched_count_mismatch_panics() {
    let a = DMatrix::from_element(2, 2, 1.0f64);
    let b = DMatrix::from_element(2, 2, 1.0f64);
    let ab = [(&a, &b)];
    let mut c: Vec<DMatrix<f64>> = Vec::new(); // 1 A/B pair, 0 C outputs: count mismatch
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
}

#[test]
#[should_panic(expected = "A.cols")]
fn gemm_batched_inner_dim_mismatch_panics() {
    let a = DMatrix::from_element(3, 4, 1.0f64);
    let b = DMatrix::from_element(5, 2, 1.0f64); // A.cols=4 != B.rows=5
    let ab = [(&a, &b)];
    let mut c = vec![DMatrix::<f64>::zeros(3, 2)];
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
}

/// An empty batch, no A/B pairs and no C outputs, does not panic and leaves `c` empty
#[test]
fn gemm_batched_empty_is_noop() {
    let ab: [(&DMatrix<f64>, &DMatrix<f64>); 0] = [];
    let mut c: Vec<DMatrix<f64>> = Vec::new();
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Rayon(0));
    assert!(c.is_empty());
}
