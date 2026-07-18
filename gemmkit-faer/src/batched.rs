//! Pointer-array-batched GEMM over a slice of per-element view triples
use super::*;
use crate::common::ref_parts;

/// Batched `C_e <- alpha*A_e*B_e + beta*C_e` for each element `e`, 1 call, parallelized **across
/// the batch** (whole GEMMs assigned to workers, each run serially and cache-hot). faer has no
/// rank-3 type, so the batch is a slice of per-element `(A, B)` [`MatRef`] inputs paired
/// positionally with a slice of `&mut C` [`MatMut`] outputs; `alpha`/`beta`/`par` are shared by
/// every element. Element shapes may differ (heterogeneous batch), as long as
/// `A_e.ncols == B_e.nrows`, `A_e.nrows == C_e.nrows`, and `B_e.ncols == C_e.ncols`. Each `A`/`B`/`C`
/// is read straight through its pointer/strides, so faer's natural column-major layout, transposed
/// views, sub-matrices, and reversed (negative-stride) views all work without copying, exactly like
/// [`gemm`]
///
/// The `ab.len()` inputs and `c.len()` outputs must agree (the batch size). Each element
/// re-dispatches through the full engine, so the result **reproduces** a loop of [`gemm`] calls and
/// is **deterministic** across thread counts; each element runs wholly on one worker, so serial and
/// parallel are additionally bit-identical
///
/// ```
/// use faer::Mat;
/// use gemmkit::Parallelism;
/// use gemmkit_faer::gemm_batched;
/// let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
/// let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
/// let mut c0 = Mat::<f64>::zeros(2, 2);
/// let mut c1 = Mat::<f64>::zeros(2, 2);
/// let ab = [
///     (a.as_dyn_stride(), b.as_dyn_stride()),
///     (a.as_dyn_stride(), b.as_dyn_stride()),
/// ];
/// {
///     let mut c = [c0.as_dyn_stride_mut(), c1.as_dyn_stride_mut()];
///     gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
/// }
/// assert_eq!(c0[(0, 0)], 19.0);
/// assert_eq!(c0[(1, 1)], 50.0);
/// assert_eq!(c0, c1);
/// ```
///
/// # Panics
/// If the input and output counts disagree (`ab.len() != c.len()`), or if any element's dimensions
/// disagree (`A.ncols != B.nrows`, `A.nrows != C.nrows`, `B.ncols != C.ncols`)
pub fn gemm_batched<T: GemmScalar>(
    alpha: T,
    ab: &[(MatRef<'_, T>, MatRef<'_, T>)],
    beta: T,
    c: &mut [MatMut<'_, T>],
    par: Parallelism,
) {
    assert_eq!(
        ab.len(),
        c.len(),
        "gemmkit-faer: batch A/B count ({}) != C count ({})",
        ab.len(),
        c.len()
    );
    let problems: Vec<GemmProblem<T>> = ab
        .iter()
        .zip(c.iter_mut())
        .enumerate()
        .map(|(i, (&(a, b), ci))| {
            let (m, k, rsa, csa, ap) = ref_parts(a);
            let (kb, n, rsb, csb, bp) = ref_parts(b);
            let (cm, cn) = (ci.nrows(), ci.ncols());
            assert_eq!(
                k, kb,
                "gemmkit-faer: batch element {i} A.cols ({k}) != B.rows ({kb})"
            );
            assert_eq!(
                m, cm,
                "gemmkit-faer: batch element {i} A.rows ({m}) != C.rows ({cm})"
            );
            assert_eq!(
                n, cn,
                "gemmkit-faer: batch element {i} B.cols ({n}) != C.cols ({cn})"
            );
            GemmProblem {
                m,
                k,
                n,
                alpha,
                a: ap,
                rsa,
                csa,
                b: bp,
                rsb,
                csb,
                beta,
                c: ci.as_ptr_mut(),
                rsc: ci.row_stride(),
                csc: ci.col_stride(),
            }
        })
        .collect();

    // SAFETY: each element's dims are validated above and faer's `MatRef`/`MatMut` guarantee the
    // pointer + element-unit `isize` strides describe a valid in-bounds layout (possibly negative
    // for a reversed view, which gemmkit's unchecked path handles) that addresses each (i,j)
    // uniquely. The outputs are a `&mut [MatMut]` of distinct exclusive borrows, so the batch's `C`
    // regions are pairwise disjoint and none aliases any `A`/`B` input (a `MatMut` and a `MatRef`
    // over the same storage cannot coexist) - the disjointness `gemm_batched_ptr_unchecked` requires
    // holds by construction
    unsafe {
        gemm_batched_ptr_unchecked(&problems, par);
    }
}
