//! `gemm_batched`: a slice of per-element `(A, B)` `MatRef` inputs paired with a slice of `&mut C`
//! `MatMut` outputs must match, bit-for-bit, a loop of individual `gemm` calls over the same
//! elements, across heterogeneous shapes and mixed layouts (natural column-major, transposed
//! non-unit-stride, and reversed negative-stride views), and must give the same result under Serial
//! and Rayon. Batch/output count mismatches and per-element inner-dimension mismatches panic; an
//! empty batch is a no-op

use faer::{Mat, MatMut, MatRef};
use gemmkit::Parallelism;

use gemmkit_faer::{gemm, gemm_batched};

/// Column-major `Mat<f64>` fill from a `seed`-derived xorshift64 stream, values in `[-0.5, 0.5)`
fn fill_mat(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    Mat::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// Runs a 4-element batch through `gemm_batched(par)`, asserts it equals a per-element `gemm(par)`
/// loop bit-for-bit, and returns the flattened batched output so the caller can also compare across
/// `par` values. The 4 elements are heterogeneous in shape and layout: natural column-major,
/// transposed (non-unit positive stride), and reversed (negative stride) views appear across `A`,
/// `B`, and the output `C`
fn run(par: Parallelism) -> Vec<f64> {
    let (alpha, beta) = (1.3f64, -0.7f64);

    // A/B storage, read-only and shared between the batched call and the reference loop below
    let a0 = fill_mat(5, 7, 1);
    let b0 = fill_mat(7, 3, 2);
    let a1t = fill_mat(4, 8, 3); // transposed to an 8x4 A1
    let b1 = fill_mat(4, 6, 4);
    let a2 = fill_mat(6, 9, 5);
    let b2 = fill_mat(9, 5, 6);
    let a3 = fill_mat(16, 5, 7);
    let b3t = fill_mat(10, 5, 8); // transposed to a 5x10 B3

    let a0v = a0.as_dyn_stride();
    let a1v = a1t.as_dyn_stride().transpose(); // 8x4, non-unit row stride
    let a2v = a2.as_dyn_stride();
    let a3v = a3.as_dyn_stride().reverse_rows(); // 16x5, negative row stride
    let b0v = b0.as_dyn_stride();
    let b1v = b1.as_dyn_stride();
    let b2v = b2.as_dyn_stride().reverse_rows(); // 9x5, negative row stride
    let b3v = b3t.as_dyn_stride().transpose(); // 5x10, non-unit row stride

    let ab = [(a0v, b0v), (a1v, b1v), (a2v, b2v), (a3v, b3v)];

    // Initial C values, cloned below into both the batched and the reference buffers
    let ci0 = fill_mat(5, 3, 10);
    let ci1 = fill_mat(8, 6, 11);
    let ci2 = fill_mat(6, 5, 12);
    let ci3 = fill_mat(16, 10, 13);

    // 1 gemm_batched call over the slice of (A, B) inputs and &mut C outputs; element 3's output
    // is a reversed (negative-stride) view
    let mut cb0 = ci0.clone();
    let mut cb1 = ci1.clone();
    let mut cb2 = ci2.clone();
    let mut cb3 = ci3.clone();
    {
        let mut cbat = [
            cb0.as_dyn_stride_mut(),
            cb1.as_dyn_stride_mut(),
            cb2.as_dyn_stride_mut(),
            cb3.as_dyn_stride_mut().reverse_rows_mut(),
        ];
        gemm_batched(alpha, &ab, beta, &mut cbat, par);
    }

    // Reference: 1 gemm(par) call per element, same views and the same reversed output for element 3
    let mut cr0 = ci0.clone();
    let mut cr1 = ci1.clone();
    let mut cr2 = ci2.clone();
    let mut cr3 = ci3.clone();
    gemm(alpha, a0v, b0v, beta, cr0.as_dyn_stride_mut(), par);
    gemm(alpha, a1v, b1v, beta, cr1.as_dyn_stride_mut(), par);
    gemm(alpha, a2v, b2v, beta, cr2.as_dyn_stride_mut(), par);
    gemm(
        alpha,
        a3v,
        b3v,
        beta,
        cr3.as_dyn_stride_mut().reverse_rows_mut(),
        par,
    );

    let mut out = Vec::new();
    for (cb, cr) in [(&cb0, &cr0), (&cb1, &cr1), (&cb2, &cr2), (&cb3, &cr3)] {
        assert_eq!((cb.nrows(), cb.ncols()), (cr.nrows(), cr.ncols()));
        for i in 0..cb.nrows() {
            for j in 0..cb.ncols() {
                assert_eq!(
                    cb[(i, j)].to_bits(),
                    cr[(i, j)].to_bits(),
                    "gemm_batched != gemm() loop (par={par:?})"
                );
                out.push(cb[(i, j)]);
            }
        }
    }
    out
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
    let a = fill_mat(2, 2, 1);
    let b = fill_mat(2, 2, 2);
    let ab = [(a.as_dyn_stride(), b.as_dyn_stride())];
    let mut c: Vec<MatMut<'_, f64>> = Vec::new(); // 1 input pair, 0 outputs: count mismatch
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
}

#[test]
#[should_panic(expected = "A.cols")]
fn gemm_batched_inner_dim_mismatch_panics() {
    let a = fill_mat(3, 4, 1);
    let b = fill_mat(5, 2, 2); // A.cols 4 != B.rows 5
    let ab = [(a.as_dyn_stride(), b.as_dyn_stride())];
    let mut cc = Mat::<f64>::zeros(3, 2);
    let mut c = [cc.as_dyn_stride_mut()];
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
}

/// An empty batch (no inputs, no outputs) does not panic and leaves the output slice empty
#[test]
fn gemm_batched_empty_is_noop() {
    let ab: [(MatRef<'_, f64>, MatRef<'_, f64>); 0] = [];
    let mut c: Vec<MatMut<'_, f64>> = Vec::new();
    gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Rayon(0));
    assert!(c.is_empty());
}
