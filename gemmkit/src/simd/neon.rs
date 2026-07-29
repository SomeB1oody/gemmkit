//! AArch64 NEON ISA token
//!
//! `f32` uses 128-bit registers with 4 lanes. `f64` uses 128-bit registers with 2 lanes.
//! NEON is baseline on AArch64, so [`Simd::vectorize`] enables no feature beyond what is
//! already present. The token still follows the same trampoline-plus-thin-wrapper shape as
//! the x86 tokens, for consistency across ISAs
//!
//! AArch64 has 32 128-bit vector registers, twice the 16 YMM registers the AVX2 token uses.
//! The microkernel tile here spends that extra headroom on slack for the out-of-order
//! scheduler, not on a wider tile. `vld1q`/`vst1q` draw no aligned/unaligned distinction.
//! Watch the operand order on `mul_add`. `vfmaq_f32(c, a, b)` computes `a*b + c`, which
//! matches this trait's `mul_add(a, b, c)` signature but not its argument order

use core::arch::aarch64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

/// AArch64 NEON ISA token: 128-bit vector registers, baseline on the arch
#[derive(Copy, Clone, Default)]
pub struct Neon;

impl Simd for Neon {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "neon")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: NEON is mandatory on aarch64, so it is always present. inner sets up the
        // codegen context, and f inlines into it
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
        // vld1q draws no aligned/unaligned distinction on NEON
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        // vst1q draws no aligned/unaligned distinction on NEON
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
        // vfmaq_f32(c, a, b) computes a*b + c, matching this trait's mul_add(a, b, c) order
        unsafe { vfmaq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // vfmsq_f32(c, a, b) computes c - a*b, matching this trait's fnma(a, b, c) order
        unsafe { vfmsq_f32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM, not FMAX, returns the non-NaN operand on an unordered compare, so a
        // NaN `a` returns `b`, matching the trait's contract. Plain FMAX would propagate
        // the NaN and break the match with the scalar edge path
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
        // vfmaq_laneq_f32::<L>(c, a, v) computes a*v[L] + c for a compile-time lane index
        // L. One loaded bvec feeds all 4 output columns with no per-column broadcast load
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
        // vld1q draws no aligned/unaligned distinction on NEON
        unsafe { vld1q_f64(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        // vst1q draws no aligned/unaligned distinction on NEON
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
        // vfmaq_f64(c, a, b) computes a*b + c, matching this trait's mul_add(a, b, c) order
        unsafe { vfmaq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // vfmsq_f64(c, a, b) computes c - a*b, matching this trait's fnma(a, b, c) order
        unsafe { vfmsq_f64(c, a, b) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // FMAXNM: non-NaN operand wins on an unordered compare, so NaN `a` -> `b`
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
        // 2 lanes: vfmaq_laneq_f64::<L>(c, a, v) computes a*v[L] + c
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

// Mixed precision: f16/bf16 inputs, f32 accumulator, 4-wide float32x4_t
//
// Both widen and narrow one lane at a time in scalar code. NEON has a native f16
// conversion, vcvt_f32_f16 over float16x4_t, but it stabilizes only above this crate's
// minimum supported Rust version. Loading raw f16 bytes into that register also needs
// vld1_f16/vst1_f16, which take the unstable primitive f16 type and stay gated behind
// stdarch_neon_f16 regardless. bf16 also takes the same scalar path here, mirroring the
// f16 fallback above. Revisit once stdarch_neon_f16 stabilizes
//
// A bf16 dot kernel via BFDOT (vbfdotq_f32) would slot into Bf16DotGemm (Q = 2) with an
// identity pack. It would need only a bf16 NEON token whose conversions delegate to
// Neon. This is deferred on a harder wall than f16. The NEON bf16 vector type,
// bfloat16x8_t, and vbfdotq_f32 are not implemented in core::arch on any Rust channel,
// stable or nightly. The matrix instruction BFMMLA (vbfmmlaq_f32) is likewise absent
// Revisit once stdarch adds NEON bf16 support

/// f16 mixed precision, scalar-widen fallback: converts all 4 lanes one at a time
/// through [`NarrowFloat`], matching the scalar token's f16 path bit-for-bit
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
        // Qualified: the f32-output twin adds a 2nd KernelSimd<f16, .., f32, ..> impl,
        // making the bare method name ambiguous
        unsafe { <Self as KernelSimd<f16, f16, f32, f16>>::load_lhs(self, p) }
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

/// bf16 mixed precision, scalar-widen fallback: a mirror of the f16 impl above
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
        // Qualified: the f32-output twin adds a 2nd KernelSimd<bf16, .., f32, ..> impl,
        // making the bare method name ambiguous
        unsafe { <Self as KernelSimd<bf16, bf16, f32, bf16>>::load_lhs(self, p) }
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

// Integer: i8 inputs, i32 accumulator, 4-wide int32x4_t
//
// The i32 accumulator ops below are native NEON. The i8 -> i32 widen-load (in the
// KernelSimd impl further down) looks like a per-byte scalar loop. rustc already lowers
// it to the optimal sequence: 1 4-byte ldr into a lane plus 2 sshll widens. An explicit
// vmovl_s8/vmovl_s16 rewrite compiles to the same code, so the plain loop stays as the
// source form. A full 8-byte vld1_s8 remains out of bounds regardless. pack_panels sizes
// the destination exactly, with no trailing slack. An 8-byte read at the last 4-wide slot
// of the last panel would overrun it
//
// A hardware i8 dot kernel via SDOT (vdotq_s32) is the NEON analogue of x86 VNNI, and
// cleaner. Signed*signed i8*i8 -> i32 is already GEMM's native op, so it needs no +128
// bias or column-sum correction. The arch-neutral dot seams already exist:
// KernelFamily::DEPTH_MULTIPLE, KernelSimd::dot_accumulate, and pack_kgroup_panels
// Adding SDOT support would mean only a dotprod token plus an identity-pack IntDotGemm
// family, Q = 4. That family would be a sibling to IntGemmVnni, which instead bakes the
// +128 transform into its pack. The result would be bit-exact to this widen path
// This is deferred for the same reason as the f16 path above: vdotq_s32 is gated behind
// the unstable stdarch_neon_dotprod on stable Rust. USDOT (vusdotq_s32) would map onto
// VNNI and reuse IntGemmVnni, but its gate, stdarch_neon_i8mm, is unstable even on
// nightly. Revisit once stdarch_neon_dotprod stabilizes

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
        // vmlaq_s32(c, a, b) computes a*b + c
        unsafe { vmlaq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
        // vmlsq_s32(c, a, b) computes c - a*b (wrapping i32). This satisfies the trait
        // The integer kernel never calls it
        unsafe { vmlsq_s32(c, a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: int32x4_t) -> i32 {
        unsafe { vaddvq_s32(v) }
    }
}

/// Requantize one pair of `int32x4_t` lanes (2 `i32` values, sign-extended to
/// `int64x2_t`) to 2 integral `i64` values in `[lo, hi]`, following the scalar map
/// exactly:
///
/// 1. Widen `i64` to `f64`. This is exact because every `i32` fits in an `f64`
/// 2. Multiply by `scale`, one IEEE multiply
/// 3. Round to nearest-even in hardware
/// 4. Add `zp`, then clamp to `[lo, hi]`
/// 5. Convert back to `i64`. This is exact because the clamped value is already integral
///
/// `#[inline(always)]` lets the intrinsics fold straight into the caller's
/// `#[target_feature]` context
///
/// `vrndnq_f64` is FRINTN, round-to-nearest ties-to-even, the same rounding as the x86
/// `vroundpd` reference and the scalar `round_ne_f64`. `vmaxq_f64`/`vminq_f64` (FMAX/FMIN)
/// do the clamp. No NaN can reach them because the API validates `scale` as finite and
/// positive, and `v` as a finite `i32`. So the FMAX-vs-FMAXNM NaN distinction does not
/// matter here, and the plain min/max mirror x86's `max_pd`/`min_pd`
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

/// Vectorized `i32 -> i8` requantize store for [`Neon`] (see [`KernelSimd::requant_store`]
/// for the bit-for-bit-with-scalar contract). It sign-extends the low and high `i32`
/// pairs to 2 `int64x2_t` values, which is exact, and requantizes each in `f64` through
/// [`requant_pair_neon`]. It then narrows both integral pairs back to one `int32x4_t`,
/// truncating with `vmovn_s64`
///
/// It gathers the low byte of each of the 4 integral, pre-clamped lanes into 4 contiguous
/// output bytes. The lookup uses `vqtbl1q_u8` with source indices `{0, 4, 8, 12}`, the
/// direct analogue of x86's `pshufb`. This is a truncating byte gather, not a saturating
/// `vqmovn`/`vqmovun`. The lanes are already clamped into `[lo, hi]`, so a saturating
/// narrow would double-clamp and give the wrong answer for the `u8`/`[0, 255]` phase
///
/// # Safety
/// `dst` is valid for 4 byte writes. Run inside [`Neon`]'s `neon` [`Simd::vectorize`]
/// context
#[cfg(feature = "int8")]
#[inline(always)]
unsafe fn requant_store_neon(dst: *mut i8, v: int32x4_t, scale: f64, zp: i32, lo: i32, hi: i32) {
    unsafe {
        let scale_v = vdupq_n_f64(scale);
        let zp_v = vdupq_n_f64(zp as f64);
        let lo_v = vdupq_n_f64(lo as f64);
        let hi_v = vdupq_n_f64(hi as f64);
        // Sign-extend the low (lanes 0, 1) and high (lanes 2, 3) i32 pairs to i64 and
        // requantize each pair separately
        let i_lo = requant_pair_neon(vmovl_s32(vget_low_s32(v)), scale_v, zp_v, lo_v, hi_v);
        let i_hi = requant_pair_neon(vmovl_s32(vget_high_s32(v)), scale_v, zp_v, lo_v, hi_v);
        // Narrow both integral pairs back to 4 i32 lanes (truncating: each value is already
        // in [lo, hi], so its low 32 bits are the value). This keeps lane order 0, 1, 2, 3
        let i32_all = vcombine_s32(vmovn_s64(i_lo), vmovn_s64(i_hi));
        // Byte-table gather of the low byte of each i32 lane {0, 4, 8, 12} into the low 4
        // output bytes (the x86 pshufb analogue). The other 12 index slots are out of
        // range, so vqtbl1q_u8 reads them as 0
        let idx: [u8; 16] = [
            0, 4, 8, 12, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let gathered = vqtbl1q_u8(vreinterpretq_u8_s32(i32_all), vld1q_u8(idx.as_ptr()));
        // AArch64 is little-endian, so byte l of the u32 is output lane l
        let packed = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(gathered));
        core::ptr::write_unaligned(dst as *mut u32, packed);
    }
}

// The i8 -> i32 widen seam plus the REQUANT_VECTOR override (requant_store_neon above)
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

// Complex (NEON): the real Reg is the plain float32x4_t/float64x2_t register, and LANES
// is the real lane count, 4 or 2. Complex GEMM routes through the shared SoA
// soa_microkernel, so its inner loop is already vectorized for free. mul_add lowers to
// vfmaq_f32/vfmaq_f64, and fnma lowers to vfmsq_f32/vfmsq_f64, through the
// SimdOps<f32>/<f64> impls above. The complex tile is MR_REG=2, NR=5 (see
// dispatch/complex.rs for the register-budget rationale). The de-interleaving pack
// (pack_planar) and the C re-interleave in the epilogue stay scalar. Both run once per
// output element, not once per depth step, so they fall outside the inner FMA loop. The
// generic scalar path is the floor for both
//
// ARMv8.3 FCMLA/FCADD are deliberately unused. They are nightly-gated on stable Rust,
// which is the reason this SoA path exists at all. They also fold the complex cross-terms
// into a single rounding step, a different accumulation structure from the 4 separate
// real FMAs used here. They cannot interleave with this kernel without breaking the
// full-vs-edge rounding identity, the same reason SDOT/BFMMLA are out of scope above
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f32, float32x4_t, 4);
#[cfg(feature = "complex")]
impl_complex_simd!(Neon, f64, float64x2_t, 2);
