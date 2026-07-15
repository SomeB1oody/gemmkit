//! AArch64 NEON ISA token
//!
//! `f32` -> 128-bit registers (4 lanes), `f64` -> 128-bit (2 lanes). NEON is
//! baseline / mandatory on AArch64, so [`Simd::vectorize`] is effectively a
//! no-op; the structure still mirrors the x86 tokens (a `#[target_feature]`
//! trampoline plus thin `#[inline(always)]` intrinsic wrappers) for uniformity
//!
//! AArch64 exposes **32** 128-bit vector registers, so the microkernel tile can
//! be wider than the AVX2 one. `vld1q`/`vst1q` make no aligned/unaligned
//! distinction. `mul_add` maps to the true fused `vfmaq_*`; mind the operand
//! order: `vfmaq_f32(c, a, b)` computes `a*b + c`, which is exactly this
//! trait's `mul_add(a, b, c)`

use core::arch::aarch64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

/// AArch64 NEON ISA token
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
        // `inner` establishes the codegen context and `f` inlines into it
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
        // NEON `vld1q` has no aligned/unaligned distinction
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        // NEON `vst1q` has no aligned/unaligned distinction
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
        // `vfmaq_f32(c, a, b)` == `a*b + c` == this trait's `mul_add(a, b, c)`
        unsafe { vfmaq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmsq_f32(c, a, b)` == `c - a*b` == this trait's `fnma(a, b, c)`
        unsafe { vfmsq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM (not FMAX): returns the non-NaN operand on an unordered compare, so a
        // `NaN` `a` returns `b`: the `max`/`min` NaN-in-`a` contract. FMAX would
        // propagate the NaN and break the fast-vs-scalar epilogue agreement
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
        // == `a * v[L] + c`. One loaded `bvec` feeds all 4 columns with no
        // per-column broadcast load
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
        // NEON `vld1q` has no aligned/unaligned distinction
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        // NEON `vst1q` has no aligned/unaligned distinction
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
        // `vfmaq_f64(c, a, b)` == `a*b + c` == this trait's `mul_add(a, b, c)`
        unsafe { vfmaq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `vfmsq_f64(c, a, b)` == `c - a*b` == this trait's `fnma(a, b, c)`
        unsafe { vfmsq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM: non-NaN operand on unordered compare (NaN `a` -> `b`)
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
        // 2 lanes; `vfmaq_laneq_f64::<L>(c, a, v)` == `a * v[L] + c`
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

// Mixed precision: f16 / bf16 inputs, f32 accumulator (4-wide float32x4_t)
//
// Both stay per-lane scalar widen/narrow, for 2 separate reasons:
//
// * `bf16`: a vectorized widen is *possible* (bf16 is the top half of an f32, so
//   the widen is an exact `vshll` left-shift by 16, bit-identical to `to_f32`) but
//   not *worth it*: measured against this scalar form it showed no throughput gain:
//   the out-of-order core already hides the per-lane widen among the FMAs, as the
//   `SimdOps::accumulate_tile` doc notes. "Keep only measured wins", so it was
//   dropped in favor of the simpler scalar code
// * `f16`: the native conversion (`vcvt_f32_f16` over `float16x4_t`) needs the
//   primitive `f16` type and the `stdarch_neon_f16` intrinsics, both still unstable
//   (rust-lang/rust#116909, #136306) and so unavailable on stable Rust. A
//   hand-rolled integer f16->f32 path is not worth the risk for a widen the OoO
//   core already hides (see bf16 above). Revisit once those stabilize
//
// A bf16 *dot* kernel, `BFDOT` (`vbfdotq_f32`), would reuse `Bf16DotGemm` as-is
// (`Q = 2`, identity pack, `kc = k`), adding only a `bf16` token whose conversions
// delegate to `Neon`. DEFERRED on a harder wall than f16: the NEON bf16 vector type
// (`bfloat16x8_t`) and `vbfdotq_f32` are unimplemented in Rust `core::arch` on every
// channel (stable and nightly alike), with no `stdarch_*` feature gate or tracking issue
// to even enable them. The matrix sibling `BFMMLA` (`vbfmmlaq_f32`) is likewise absent;
// revisit once stdarch grows NEON bf16 support

/// `f16` mixed precision (scalar fallback): widens/narrows 4 lanes one at a time
/// through [`NarrowFloat`], matching the scalar engine's `f16` path. (See the note
/// above on why the native NEON fp16 path is unavailable on stable Rust)
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
/// `vshll` widen showed no measured gain over this: see the note above)
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

// Integer: i8 inputs, i32 accumulator (4-wide int32x4_t)
//
// The `i32` accumulator ops are native NEON; the `i8 -> i32` widen-load uses a
// per-lane scalar fallback to avoid loading bytes past a 4-wide panel slot
// TODO(neon): vectorize the widen with `vmovl_s8`/`vmovl_s16` over the full `mr`
// row block at once (where the 8-byte read stays in bounds)
//
// A hardware `i8` dot kernel, `SDOT` (`vdotq_s32`), is the NEON analogue of x86 VNNI
// but cleaner: signed*signed `i8*i8 -> i32` is GEMM's native op, so no `+128` bias or
// per-column-sum correction. The arch-neutral dot seams already exist
// (`KernelFamily::DEPTH_MULTIPLE`, `KernelSimd::dot_accumulate`, `pack_kgroup_panels`),
// so it would add only a `dotprod` token and an identity-pack `IntDotGemm` family
// (sibling to `IntGemmVnni`, which bakes the `+128` xform into its pack), `Q = 4`,
// bit-exact to this widen path. DEFERRED like the native f16 path above: `vdotq_s32` is
// gated behind the unstable `stdarch_neon_dotprod` (rust-lang/rust#117224), stable only
// in Rust 1.98, below the crate's stable/MSRV. (`USDOT`, `vusdotq_s32`, would map onto
// VNNI and reuse `IntGemmVnni`, but `stdarch_neon_i8mm` (#117223) is unstable even on
// nightly.) Revisit once `stdarch_neon_dotprod` is stable

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
        // `vmlaq_s32(c, a, b)` == a*b + c
        unsafe { vmlaq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
        // `vmlsq_s32(c, a, b)` == c - a*b (wrapping i32). Present only to satisfy the
        // trait; the integer kernel never calls it
        unsafe { vmlsq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: int32x4_t) -> i32 {
        unsafe { vaddvq_s32(v) }
    }
}

/// Requantize one pair (2 `i32` lanes sign-extended to `int64x2_t`) of an `int32x4_t` accumulator
/// to 2 integral `i64` in `[lo, hi]`, following the exact scalar map: widen `i64 -> f64` (exact,
/// every `i32` is representable in `f64`), multiply by `scale` (one IEEE multiply), round-to-
/// nearest-even in hardware, add `zp`, clamp `[lo, hi]`, convert back toward zero to `i64` (exact:
/// the clamped value is already integral in `[lo, hi]`). `#[inline(always)]`, so the intrinsics
/// fold into the caller's `#[target_feature]` context
///
/// `vrndnq_f64` is FRINTN, round-to-nearest ties-to-even, matching the x86 `vroundpd` reference and
/// the scalar `round_ne_f64`. `vmaxq_f64` / `vminq_f64` (FMAX/FMIN) are the numeric clamp: no NaN
/// can reach them (the API validates `scale` finite and `> 0` and `v` is a finite `i32`), so the
/// FMAX-vs-FMAXNM NaN distinction is immaterial here and the plain min/max mirror the x86
/// `max_pd`/`min_pd`
///
/// # Safety
/// Run inside [`Neon`]'s `neon` [`Simd::vectorize`] context
#[cfg(feature = "int8")]
#[inline(always)]
unsafe fn requant_pair_neon(
    x: int64x2_t,
    scale_v: float64x2_t,
    zp_v: float64x2_t,
    lo_v: float64x2_t,
    hi_v: float64x2_t,
) -> int64x2_t {
    unsafe {
        let t = vcvtq_f64_s64(x);
        let t = vmulq_f64(t, scale_v);
        let t = vrndnq_f64(t);
        let u = vaddq_f64(t, zp_v);
        let u = vmaxq_f64(u, lo_v);
        let u = vminq_f64(u, hi_v);
        vcvtq_s64_f64(u)
    }
}

/// Vectorized `i32 -> i8` requantize store for [`Neon`] (see the [`KernelSimd::requant_store`]
/// contract for the bit-for-bit equivalence with the scalar map): sign-extend the low/high `i32`
/// pairs to 2 `int64x2_t` (exact), requantize each in `f64` ([`requant_pair_neon`]), narrow both
/// integral pairs back to one `int32x4_t` (truncating `vmovn_s64`), then gather the **low byte** of
/// each of the 4 integral, pre-clamped lanes into 4 contiguous output bytes with a byte-table
/// lookup (`vqtbl1q_u8`, source indices {0, 4, 8, 12}: the direct analogue of the x86 `pshufb`).
/// This is a TRUNCATION, NOT a saturating `vqmovn`/`vqmovun`: the lanes are already in `[lo, hi]`,
/// so only the byte gather matters, and a saturating narrow would be wrong for the `u8` / `[0, 255]`
/// phase
///
/// # Safety
/// `dst` valid for 4 byte writes; run inside [`Neon`]'s `neon` [`Simd::vectorize`] context
#[cfg(feature = "int8")]
#[inline(always)]
unsafe fn requant_store_neon(dst: *mut i8, v: int32x4_t, scale: f64, zp: i32, lo: i32, hi: i32) {
    unsafe {
        let scale_v = vdupq_n_f64(scale);
        let zp_v = vdupq_n_f64(zp as f64);
        let lo_v = vdupq_n_f64(lo as f64);
        let hi_v = vdupq_n_f64(hi as f64);
        // Sign-extend the low (lanes 0, 1) and high (lanes 2, 3) i32 pairs to i64, requant each pair
        let i_lo = requant_pair_neon(vmovl_s32(vget_low_s32(v)), scale_v, zp_v, lo_v, hi_v);
        let i_hi = requant_pair_neon(vmovl_s32(vget_high_s32(v)), scale_v, zp_v, lo_v, hi_v);
        // Narrow both integral pairs back to the 4 i32 lanes (truncating; each value is in `[lo, hi]`
        // so its low 32 bits are the value), lane order 0, 1, 2, 3 preserved
        let i32_all = vcombine_s32(vmovn_s64(i_lo), vmovn_s64(i_hi));
        // Byte-table gather of the low byte of each i32 lane {0, 4, 8, 12} into the low 4 output
        // bytes (the x86 `pshufb` analogue); the high 12 index slots are out of range so read as 0
        let idx: [u8; 16] = [
            0, 4, 8, 12, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let gathered = vqtbl1q_u8(vreinterpretq_u8_s32(i32_all), vld1q_u8(idx.as_ptr()));
        // AArch64 is little-endian, so byte `l` of the u32 is output lane `l`
        let packed = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(gathered));
        core::ptr::write_unaligned(dst as *mut u32, packed);
    }
}

// The NEON `requant_store` override (the `REQUANT_VECTOR` seam)
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

    const REQUANT_VECTOR: bool = true;
    #[inline(always)]
    unsafe fn requant_store(
        self,
        dst: *mut i8,
        v: int32x4_t,
        scale: f64,
        zp: i32,
        lo: i32,
        hi: i32,
    ) {
        unsafe { requant_store_neon(dst, v, scale, zp, lo, hi) }
    }
}

// Complex (NEON): real `Reg`; `LANES` is the real lane count (4 / 2). Complex GEMM
// routes through the shared SoA `soa_microkernel`, so the **inner loop is already
// vectorized** for free: `mul_add` lowers to `vfmaq_f32`/`vfmaq_f64` and `fnma` to
// `vfmsq_f32`/`vfmsq_f64` through the real `SimdOps<f32>`/`<f64>` above. The tile is
// MR_REG=2, NR=5 (see `dispatch.rs` for the register-budget rationale). The de-interleave
// pack (`pack_planar`) and C re-interleave epilogue stay **scalar**: they are a small
// fraction of runtime (the inner loop dominates), so a `vld2q`/`vst2q` seam does not pay
// and the generic scalar path is the floor
//
// Do NOT use ARMv8.3 `FCMLA`/`FCADD`: they are nightly-gated on stable Rust (the very
// reason this SoA path exists), and they fold the complex cross-terms into a single
// rounding step, a different accumulation structure than the 4 separate real FMAs
// here, so they cannot interleave with this kernel without breaking the full-vs-edge
// rounding identity (the analogue of why `sdot`/`bfmmla` are out of scope)
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f32, float32x4_t, 4);
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f64, float64x2_t, 2);
