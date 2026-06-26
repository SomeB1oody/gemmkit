//! AArch64 NEON ISA token.
//!
//! `f32` → 128-bit registers (4 lanes), `f64` → 128-bit (2 lanes). NEON is
//! baseline / mandatory on AArch64, so [`Simd::vectorize`] is effectively a
//! no-op; the structure still mirrors the x86 tokens (a `#[target_feature]`
//! trampoline plus thin `#[inline(always)]` intrinsic wrappers) for uniformity.
//!
//! AArch64 exposes **32** 128-bit vector registers, so the microkernel tile can
//! be wider than the AVX2 one. `vld1q`/`vst1q` make no aligned/unaligned
//! distinction, so `load == loadu` and `store == storeu`. `mul_add` maps to the
//! true fused `vfmaq_*`; mind the operand order — `vfmaq_f32(c, a, b)` computes
//! `a*b + c`, which is exactly our `mul_add(a, b, c)`.

use core::arch::aarch64::*;

use super::{Simd, SimdOps};

/// AArch64 NEON ISA token.
#[derive(Copy, Clone, Default)]
pub struct Neon;

impl Simd for Neon {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "neon")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: NEON is mandatory on aarch64, so the feature is always present;
        // `inner` establishes the codegen context and `f` inlines into it.
        unsafe { inner(f) }
    }
}

impl SimdOps<f32> for Neon {
    type Reg = float32x4_t;
    const LANES: usize = 4;
    const ALIGN: usize = 16;
    const LANE_FMA: bool = true;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { vdupq_n_f32(0.0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        unsafe { vdupq_n_f32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f32) -> Self::Reg {
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        // NEON `vld1q` has no aligned/unaligned distinction.
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32, v: Self::Reg) {
        unsafe { vst1q_f32(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        // NEON `vst1q` has no aligned/unaligned distinction.
        unsafe { vst1q_f32(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmaq_f32(c, a, b)` == `a*b + c` == our `mul_add(a, b, c)`.
        unsafe { vfmaq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        unsafe { vaddvq_f32(v) }
    }
    #[inline(always)]
    unsafe fn fma_bvec<const MR_REG: usize>(
        self,
        a_regs: &[Self::Reg; MR_REG],
        bvec: Self::Reg,
        acc: &mut [[Self::Reg; MR_REG]],
    ) {
        // 4 lanes, each a compile-time immediate; `vfmaq_laneq_f32::<L>(c, a, v)`
        // == `a * v[L] + c`. One loaded `bvec` feeds all four columns with no
        // per-column broadcast load.
        debug_assert_eq!(acc.len(), 4);
        unsafe {
            for i in 0..MR_REG {
                acc[0][i] = vfmaq_laneq_f32::<0>(acc[0][i], a_regs[i], bvec);
            }
            for i in 0..MR_REG {
                acc[1][i] = vfmaq_laneq_f32::<1>(acc[1][i], a_regs[i], bvec);
            }
            for i in 0..MR_REG {
                acc[2][i] = vfmaq_laneq_f32::<2>(acc[2][i], a_regs[i], bvec);
            }
            for i in 0..MR_REG {
                acc[3][i] = vfmaq_laneq_f32::<3>(acc[3][i], a_regs[i], bvec);
            }
        }
    }
}

impl SimdOps<f64> for Neon {
    type Reg = float64x2_t;
    const LANES: usize = 2;
    const ALIGN: usize = 16;
    const LANE_FMA: bool = true;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { vdupq_n_f64(0.0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        unsafe { vdupq_n_f64(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f64) -> Self::Reg {
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        // NEON `vld1q` has no aligned/unaligned distinction.
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64, v: Self::Reg) {
        unsafe { vst1q_f64(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        // NEON `vst1q` has no aligned/unaligned distinction.
        unsafe { vst1q_f64(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vmulq_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vaddq_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmaq_f64(c, a, b)` == `a*b + c` == our `mul_add(a, b, c)`.
        unsafe { vfmaq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        unsafe { vaddvq_f64(v) }
    }
    #[inline(always)]
    unsafe fn fma_bvec<const MR_REG: usize>(
        self,
        a_regs: &[Self::Reg; MR_REG],
        bvec: Self::Reg,
        acc: &mut [[Self::Reg; MR_REG]],
    ) {
        // 2 lanes; `vfmaq_laneq_f64::<L>(c, a, v)` == `a * v[L] + c`.
        debug_assert_eq!(acc.len(), 2);
        unsafe {
            for i in 0..MR_REG {
                acc[0][i] = vfmaq_laneq_f64::<0>(acc[0][i], a_regs[i], bvec);
            }
            for i in 0..MR_REG {
                acc[1][i] = vfmaq_laneq_f64::<1>(acc[1][i], a_regs[i], bvec);
            }
        }
    }
}
