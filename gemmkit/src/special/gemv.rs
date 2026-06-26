//! gemv: matrix·vector (`n == 1` or `m == 1`).
//!
//! gemv is memory-bound, so register tiling buys nothing; an axpy / dot sweep is
//! the right shape. Both `m == 1` and `n == 1` reduce to one core routine by
//! viewing the matrix (transposed for `m == 1`) as `rows × k` times a `k`-vector.
//! Correct for every layout; vectorized for the contiguous ones. Single-threaded
//! (gemv rarely benefits from threads at these sizes), so there is no
//! serial/parallel reproducibility concern.

use crate::scalar::Float;
use crate::simd::SimdOps;

/// Dispatch a gemv shape to the core routine.
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must
/// not alias `a`/`b`. Must be called only when the CPU supports `S`'s features.
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_typed<T, S>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        simd.vectorize(|| {
            if n == 1 {
                // C (m×1) = beta·C + alpha·A·b, A = m×k, b = k-vector.
                core::<T, S>(simd, m, k, alpha, a, rsa, csa, b, rsb, beta, c, rsc);
            } else {
                // C (1×n) = beta·C + alpha·a·B. View Bᵀ (n×k) times a (k-vector):
                // Bᵀ[j,k] = B[k,j] → row stride csb, col stride rsb; out stride csc.
                core::<T, S>(simd, n, k, alpha, b, csb, rsb, a, csa, beta, c, csc);
            }
        });
    }
}

/// `out[i] = beta·out[i] + alpha · Σ_k mat[i,k]·vec[k]` for `i in 0..rows`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn core<T, S>(
    simd: S,
    rows: usize,
    k: usize,
    alpha: T,
    mat: *const T,
    mat_rs: isize,
    mat_cs: isize,
    vec: *const T,
    vec_s: isize,
    beta: T,
    out: *mut T,
    out_s: isize,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;

        // Scale C first (β == 0 must not read C).
        for i in 0..rows {
            let op = out.offset(i as isize * out_s);
            if beta == T::ZERO {
                *op = T::ZERO;
            } else if beta != T::ONE {
                *op = beta * *op;
            }
        }

        if mat_rs == 1 && out_s == 1 {
            // axpy form: for each column k, out += (alpha·vec[k]) · mat[:,k].
            for kk in 0..k {
                let s = alpha * *vec.offset(kk as isize * vec_s);
                let col = mat.offset(kk as isize * mat_cs);
                let sv = simd.splat(s);
                let mut i = 0;
                while i + lanes <= rows {
                    let mv = simd.loadu(col.add(i));
                    let ov = simd.loadu(out.add(i));
                    simd.storeu(out.add(i), simd.mul_add(mv, sv, ov));
                    i += lanes;
                }
                while i < rows {
                    let op = out.add(i);
                    *op = s.mul_add(*col.add(i), *op);
                    i += 1;
                }
            }
        } else if mat_cs == 1 && vec_s == 1 {
            // dot form: out[i] += alpha · ⟨mat[i,:], vec⟩.
            for i in 0..rows {
                let row = mat.offset(i as isize * mat_rs);
                let mut acc = simd.zero();
                let mut kk = 0;
                while kk + lanes <= k {
                    acc = simd.mul_add(simd.loadu(row.add(kk)), simd.loadu(vec.add(kk)), acc);
                    kk += lanes;
                }
                let mut s = simd.reduce_sum(acc);
                while kk < k {
                    s = (*row.add(kk)).mul_add(*vec.add(kk), s);
                    kk += 1;
                }
                let op = out.offset(i as isize * out_s);
                *op = alpha.mul_add(s, *op);
            }
        } else {
            // Fully general strided fallback.
            for i in 0..rows {
                let mut s = T::ZERO;
                for kk in 0..k {
                    s = (*mat.offset(i as isize * mat_rs + kk as isize * mat_cs))
                        .mul_add(*vec.offset(kk as isize * vec_s), s);
                }
                let op = out.offset(i as isize * out_s);
                *op = alpha.mul_add(s, *op);
            }
        }
    }
}
