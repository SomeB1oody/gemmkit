//! Batched GEMM over a slice of independently shaped `(&A, &B) -> &mut C` triples, dispatched
//! through gemmkit's pointer-array batched engine
use super::*;
use crate::common::dims_strides;

/// Runs `C_e <- alpha*A_e*B_e + beta*C_e` for every element `e` of a batch in 1 call, parallelized
/// **across the batch**: whole per-element GEMMs are handed to workers, each running serially and
/// staying cache-hot. nalgebra has no rank-3 array type, so the batch is expressed as a slice of
/// per-element `(&A, &B)` inputs paired positionally with a slice of `&mut C` outputs, with
/// `alpha`/`beta`/`par` shared by every element. Element shapes may differ (a heterogeneous batch
/// is fine) as long as each obeys `A_e.cols == B_e.rows`, `A_e.rows == C_e.rows`, and
/// `B_e.cols == C_e.cols`; each `A`/`B`/`C` is read through its own pointer/strides, so
/// column-major, row-major, and general-stride views all work without copying, exactly as in
/// [`gemm`]
///
/// `ab` and `c` must have the same length (the batch size). Because every element re-dispatches
/// through the same engine [`gemm`] uses, the batched result **reproduces** a loop of [`gemm`]
/// calls and is **deterministic** across thread counts; since each element also runs wholly on 1
/// worker, serial and parallel runs are additionally bit-identical
///
/// ```
/// use nalgebra::DMatrix;
/// use gemmkit::Parallelism;
/// use gemmkit_nalgebra::gemm_batched;
/// let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
/// let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);
/// let mut c = vec![DMatrix::<f32>::zeros(2, 2), DMatrix::<f32>::zeros(2, 2)];
/// let ab = [(&a, &b), (&a, &b)];
/// gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
/// assert_eq!(c[0], DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
/// assert_eq!(c[1], c[0]);
/// ```
///
/// # Panics
/// If the input and output counts disagree (`ab.len() != c.len()`), or if any element's dimensions
/// disagree (`A.cols != B.rows`, `A.rows != C.rows`, `B.cols != C.cols`)
// `(&A, &B)` already carries both operands' full storage-generic lists; a type alias would just
// rename that same complexity, not reduce it
#[allow(clippy::type_complexity)]
pub fn gemm_batched<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    ab: &[(&Matrix<T, R1, C1, S1>, &Matrix<T, R2, C2, S2>)],
    beta: T,
    c: &mut [Matrix<T, RC, CC, SC>],
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    assert_eq!(
        ab.len(),
        c.len(),
        "gemmkit-nalgebra: batch A/B count ({}) != C count ({})",
        ab.len(),
        c.len()
    );
    let problems: Vec<GemmProblem<T>> = ab
        .iter()
        .zip(c.iter_mut())
        .enumerate()
        .map(|(i, (&(a, b), ci))| {
            let (m, k, rsa, csa) = dims_strides(a);
            let (kb, n, rsb, csb) = dims_strides(b);
            let (cm, cn) = ci.shape();
            assert_eq!(
                k, kb,
                "gemmkit-nalgebra: batch element {i} A.cols ({k}) != B.rows ({kb})"
            );
            assert_eq!(
                m, cm,
                "gemmkit-nalgebra: batch element {i} A.rows ({m}) != C.rows ({cm})"
            );
            assert_eq!(
                n, cn,
                "gemmkit-nalgebra: batch element {i} B.cols ({n}) != C.cols ({cn})"
            );
            let cs = ci.strides();
            GemmProblem {
                m,
                k,
                n,
                alpha,
                a: a.as_ptr(),
                rsa,
                csa,
                b: b.as_ptr(),
                rsb,
                csb,
                beta,
                c: ci.as_mut_ptr(),
                rsc: cs.0 as isize,
                csc: cs.1 as isize,
            }
        })
        .collect();

    // SAFETY: each element's dims are checked above, and nalgebra guarantees its pointer/strides
    // describe a valid in-bounds layout addressing each (i,j) uniquely. `c.iter_mut()` hands out
    // distinct exclusive borrows, so the batch's C regions are pairwise disjoint and none can alias
    // an `&` A/B input (a shared and an exclusive borrow of the same storage can't coexist) - the
    // disjointness gemm_batched_ptr_unchecked requires holds by construction, not by a runtime check
    unsafe {
        gemm_batched_ptr_unchecked(&problems, par);
    }
}
