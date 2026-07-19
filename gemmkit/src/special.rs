//! Special-case paths (layer L6): shape-specific routes around the register-tiling driver
//!
//! The driver's packing and cache-blocking machinery pays off only once there is enough
//! reuse per packed element to amortize it. For shapes where that never holds, these
//! modules compute the product a different way: [`gemv`] (matrix*vector), [`small_k`]
//! (skinny / low-depth GEMM: gevv, rank-`k`, tall-skinny), and [`small_mn`] (small `m,n`,
//! long `k`, computed as a grid of horizontal inner products). [`batched`] is not a new
//! compute strategy but an orchestration layer that fans many independent products out
//! across workers, each one re-entering the normal single-GEMM engine

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

/// Horizontal dot of 2 unit-stride length-`k` vectors, `sum_k(x[k]*y[k])`: a SIMD `mul_add`
/// sweep reduced by `reduce_sum` (in its fixed lane order), followed by an ascending scalar
/// tail for the `k % LANES` remainder. This is the one fixed-order reduction every
/// bandwidth-bound dot path in this module shares ([`gemv`]'s row*vector sweep and
/// [`small_mn`]'s edge-tile cell both call it), which is what lets them round identically and
/// keeps the determinism contract those paths rely on
///
/// # Safety
/// `x`/`y` valid for `k` contiguous reads; run inside `S`'s [`crate::simd::Simd::vectorize`]
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
