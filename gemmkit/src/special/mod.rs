//! Special-case paths (layer L6).
//!
//! These bypass the register-tiling driver for shapes where it is the wrong tool: [`gemv`]
//! (matrix·vector), [`small_k`] (skinny / low-depth GEMM — gevv, rank-`k`, tall-skinny), and
//! [`small_mn`] (small `m,n`, long `k` — the horizontal inner-product kernel). Batched GEMM is
//! an orchestration layer over the single-GEMM engine.

pub mod batched;
pub mod gemv;
pub mod small_k;
pub mod small_mn;

use crate::scalar::Float;
use crate::simd::SimdOps;

/// Horizontal dot of two unit-stride length-`k` vectors: `Σ_k x[k]·y[k]`, a SIMD `mul_add` sweep
/// reduced by `reduce_sum` (fixed lane order) then an ascending scalar tail. This is the single
/// fixed-order reduction the bandwidth-bound dot paths share ([`gemv`]'s row·vector and
/// [`small_mn`]'s edge-tile cell), so they round identically — the determinism contract those
/// paths rely on.
///
/// # Safety
/// `x`/`y` valid for `k` contiguous reads; run inside `S`'s [`crate::simd::Simd::vectorize`].
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
