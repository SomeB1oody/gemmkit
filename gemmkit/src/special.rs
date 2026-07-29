//! Special-case paths (layer L6): shape-specific routes around the register-tiling driver
//!
//! The driver's packing and cache-blocking machinery pays off only when there is enough
//! reuse per packed element to amortize it. When a shape never reaches that reuse, these
//! modules compute the product a different way. [`gemv`] handles a matrix times a vector.
//! [`small_k`] handles a skinny or low-depth GEMM, such as gevv, rank-`k`, or tall-skinny.
//! [`small_mn`] handles small `m` and `n` with a long `k`, as a grid of horizontal inner
//! products. [`batched`] is not a new compute strategy. It is an orchestration layer that
//! fans many independent products across workers, each one re-entering the normal
//! single-GEMM engine

// Batched GEMM: many independent products, scheduled whole-GEMM-per-worker
pub mod batched;
// gemv: matrix*vector product, computed as output-row-partitioned dot/axpy sweeps
pub mod gemv;
// Skinny/low-depth GEMM: one unpacked, in-place depth panel through the family microkernel
pub mod small_k;
// Small-`m,n`, long-`k` GEMM: a grid of horizontal (inner-product) dots
pub mod small_mn;

use crate::scalar::Float;
use crate::simd::SimdOps;

/// Horizontal dot of 2 unit-stride length-`k` vectors, `sum_k(x[k]*y[k])`. A SIMD `mul_add`
/// sweep accumulates the products, then `reduce_sum` folds the lanes in a fixed order. An
/// ascending scalar loop handles the `k % LANES` remainder. Every same-precision dot path in
/// this module calls this routine, so every caller rounds the same way. [`gemv`]'s row-vector
/// sweep and [`small_mn`]'s edge-tile cell both do. The mixed-precision gemv path widens to
/// `f32` and uses its own twin instead
///
/// # Safety
/// `x` and `y` must be valid for `k` contiguous reads. The call must run inside `S`'s
/// [`crate::simd::Simd::vectorize`]
#[inline(always)]
pub(crate) unsafe fn dot_contiguous<T, S>(simd: S, k: usize, x: *const T, y: *const T) -> T
where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let mut acc = simd.zero();
        let mut kk = 0;
        while kk + lanes <= k {
            acc = simd.mul_add(simd.loadu(x.add(kk)), simd.loadu(y.add(kk)), acc);
            kk += lanes;
        }
        let mut dot = simd.reduce_sum(acc);
        while kk < k {
            dot = (*x.add(kk)).mul_add(*y.add(kk), dot);
            kk += 1;
        }
        dot
    }
}
