//! AArch64 NEON ISA token.
//!
//! `f32` → 128-bit registers (4 lanes), `f64` → 128-bit (2 lanes). NEON is
//! baseline / mandatory on AArch64, so [`Simd::vectorize`] is effectively a
//! no-op; the structure still mirrors the x86 tokens (a `#[target_feature]`
//! trampoline plus thin `#[inline(always)]` intrinsic wrappers) for uniformity.
//!
//! AArch64 exposes **32** 128-bit vector registers, so the microkernel tile can
//! be wider than the AVX2 one. `vld1q`/`vst1q` make no aligned/unaligned
//! distinction. `mul_add` maps to the true fused `vfmaq_*`; mind the operand
//! order — `vfmaq_f32(c, a, b)` computes `a*b + c`, which is exactly our
//! `mul_add(a, b, c)`.

use core::arch::aarch64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

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
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        // NEON `vld1q` has no aligned/unaligned distinction.
        unsafe { vld1q_f32(p) }
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
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmsq_f32(c, a, b)` == `c - a*b` == our `fnma(a, b, c)`.
        unsafe { vfmsq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM (not FMAX): returns the non-NaN operand on an unordered compare, so a
        // `NaN` `a` returns `b` — the `max`/`min` NaN-in-`a` contract. FMAX would
        // propagate the NaN and break the fast-vs-scalar epilogue agreement.
        unsafe { vmaxnmq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vminnmq_f32(a, b) }
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
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        // NEON `vld1q` has no aligned/unaligned distinction.
        unsafe { vld1q_f64(p) }
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
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmsq_f64(c, a, b)` == `c - a*b` == our `fnma(a, b, c)`.
        unsafe { vfmsq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM: non-NaN operand on unordered compare (NaN `a` -> `b`).
        unsafe { vmaxnmq_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { vminnmq_f64(a, b) }
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

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (4-wide float32x4_t) ----
//
// Both stay per-lane scalar widen/narrow, for two separate reasons:
//
// * `bf16`: a vectorized widen is *possible* (bf16 is the top half of an f32, so
//   the widen is an exact `vshll` left-shift by 16, bit-identical to `to_f32`) but
//   not *worth it*: measured against this scalar form it showed no throughput gain —
//   the out-of-order core already hides the per-lane widen among the FMAs, as the
//   `SimdOps::accumulate_tile` doc notes. "Keep only measured wins", so it was
//   dropped in favor of the simpler scalar code.
// * `f16`: the native conversion (`vcvt_f32_f16` over `float16x4_t`) needs the
//   primitive `f16` type and the `stdarch_neon_f16` intrinsics, both still unstable
//   (rust-lang/rust#116909, #136306) and so unavailable on stable Rust. A
//   hand-rolled integer f16->f32 path is not worth the risk for a widen the OoO
//   core already hides (see bf16 above). Revisit once those stabilize.
//
// A bf16 *dot* kernel — `BFDOT` (`vbfdotq_f32`) — would reuse `Bf16DotGemm` as-is
// (`Q = 2`, identity pack, `kc = k`), adding only a `bf16` token whose conversions
// delegate to `Neon`. DEFERRED on a harder wall than f16: the NEON bf16 vector type
// (`bfloat16x8_t`) and `vbfdotq_f32` are unimplemented in Rust `core::arch` on every
// channel (stable and nightly alike), with no `stdarch_*` feature gate or tracking issue
// to even enable them. The matrix sibling `BFMMLA` (`vbfmmlaq_f32`) is likewise absent;
// revisit once stdarch grows NEON bf16 support.

/// `f16` mixed precision (scalar fallback): widens/narrows 4 lanes one at a time
/// through [`NarrowFloat`], matching the scalar engine's `f16` path. (See the note
/// above on why the native NEON fp16 path is unavailable on stable Rust.)
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Neon {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> float32x4_t {
        unsafe {
            let a = [
                (*p).widen(),
                (*p.add(1)).widen(),
                (*p.add(2)).widen(),
                (*p.add(3)).widen(),
            ];
            vld1q_f32(a.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> float32x4_t {
        unsafe { vdupq_n_f32(v.widen()) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> float32x4_t {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: float32x4_t) {
        unsafe {
            let mut t = [0.0f32; 4];
            vst1q_f32(t.as_mut_ptr(), v);
            for (i, &x) in t.iter().enumerate() {
                *p.add(i) = f16::narrow(x);
            }
        }
    }
}

/// `bf16` mixed precision (scalar fallback), mirror of the `f16` impl. (A vectorized
/// `vshll` widen showed no measured gain over this — see the note above.)
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Neon {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> float32x4_t {
        unsafe {
            let a = [
                (*p).widen(),
                (*p.add(1)).widen(),
                (*p.add(2)).widen(),
                (*p.add(3)).widen(),
            ];
            vld1q_f32(a.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> float32x4_t {
        unsafe { vdupq_n_f32(v.widen()) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> float32x4_t {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: float32x4_t) {
        unsafe {
            let mut t = [0.0f32; 4];
            vst1q_f32(t.as_mut_ptr(), v);
            for (i, &x) in t.iter().enumerate() {
                *p.add(i) = bf16::narrow(x);
            }
        }
    }
}

// ---- integer: i8 inputs, i32 accumulator (4-wide int32x4_t) ----
//
// The `i32` accumulator ops are native NEON; the `i8 -> i32` widen-load uses a
// per-lane scalar fallback to avoid loading bytes past a 4-wide panel slot.
// TODO(neon): vectorize the widen with `vmovl_s8`/`vmovl_s16` over the full `mr`
// row block at once (where the 8-byte read stays in bounds).
//
// A hardware `i8` dot kernel — `SDOT` (`vdotq_s32`) — is the NEON analogue of x86 VNNI
// but cleaner: signed×signed `i8·i8 -> i32` is GEMM's native op, so no `+128` bias or
// per-column-sum correction. The arch-neutral dot seams already exist
// (`KernelFamily::DEPTH_MULTIPLE`, `KernelSimd::dot_accumulate`, `pack_kgroup_panels`),
// so it would add only a `dotprod` token and an identity-pack `IntDotGemm` family
// (sibling to `IntGemmVnni`, which bakes the `+128` xform into its pack), `Q = 4`,
// bit-exact to this widen path. DEFERRED like the native f16 path above: `vdotq_s32` is
// gated behind the unstable `stdarch_neon_dotprod` (rust-lang/rust#117224), stable only
// in Rust 1.98 — below the crate's stable/MSRV. (`USDOT`, `vusdotq_s32`, would map onto
// VNNI and reuse `IntGemmVnni`, but `stdarch_neon_i8mm` (#117223) is unstable even on
// nightly.) Revisit once `stdarch_neon_dotprod` is stable.

#[cfg(feature = "int8")]
impl SimdOps<i32> for Neon {
    type Reg = int32x4_t;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn zero(self) -> int32x4_t {
        unsafe { vdupq_n_s32(0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> int32x4_t {
        unsafe { vdupq_n_s32(v) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: int32x4_t) {
        unsafe { vst1q_s32(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vmulq_s32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vaddq_s32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
        // `vmlaq_s32(c, a, b)` == a*b + c.
        unsafe { vmlaq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
        // `vmlsq_s32(c, a, b)` == c - a*b (wrapping i32). Present only to satisfy the
        // trait; the integer kernel never calls it.
        unsafe { vmlsq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: int32x4_t) -> i32 {
        unsafe { vaddvq_s32(v) }
    }
}

// A NEON `requant_store` override (the `REQUANT_VECTOR` seam) is deliberately **deferred**:
// this dev box cannot execute aarch64, and project policy is to validate an ISA kernel on the
// device before shipping it. `Neon` therefore keeps the trait default (`REQUANT_VECTOR = false`),
// so the requantizing epilogue routes every element through the scalar map on aarch64 — correct,
// just not yet vectorized.
#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Neon {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> int32x4_t {
        unsafe {
            let a = [
                *p as i32,
                *p.add(1) as i32,
                *p.add(2) as i32,
                *p.add(3) as i32,
            ];
            vld1q_s32(a.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> int32x4_t {
        unsafe { vdupq_n_s32(v as i32) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: int32x4_t) {
        unsafe { vst1q_s32(p, v) }
    }
}

// Complex (NEON): real `Reg`; `LANES` is the real lane count (4 / 2). Complex GEMM
// routes through the shared SoA `soa_microkernel`, so the **inner loop is already
// vectorized** for free — `mul_add` lowers to `vfmaq_f32`/`vfmaq_f64` and `fnma` to
// `vfmsq_f32`/`vfmsq_f64` through the real `SimdOps<f32>`/`<f64>` above. The tile is
// MR_REG=2, NR=5 (see `dispatch.rs` for the register-budget rationale). The de-interleave
// pack (`pack_planar`) and C re-interleave epilogue stay **scalar**: they are a small
// fraction of runtime (the inner loop dominates), so a `vld2q`/`vst2q` seam does not pay
// and the generic scalar path is the floor.
//
// Do NOT use ARMv8.3 `FCMLA`/`FCADD`: they are nightly-gated on stable Rust (the very
// reason this SoA path exists), and they fold the complex cross-terms into a single
// rounding step — a different accumulation structure than the four separate real FMAs
// here — so they cannot interleave with this kernel without breaking the full-vs-edge
// rounding identity (the analogue of why `sdot`/`bfmmla` are out of scope).
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f32, float32x4_t, 4);
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f64, float64x2_t, 2);
