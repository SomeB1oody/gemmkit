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

#[cfg(feature = "half")]
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;

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

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (4-wide float32x4_t) ----
//
// Both stay per-lane scalar widen/narrow, for two separate reasons:
//
// * `bf16`: a vectorized widen is *possible* (bf16 is the top half of an f32, so
//   the widen is an exact `vshll` left-shift by 16, bit-identical to `to_f32`) but
//   not *worth it*: implemented and benchmarked against this scalar form, it showed
//   no throughput gain (~36/51/49 GFLOP/s either way at n=256/512/1024) — the wide
//   OoO core already hides the per-lane widen among the FMAs, exactly as the
//   `SimdOps::accumulate_tile` doc notes. "Keep only measured wins", so it was
//   dropped in favor of the simpler scalar code.
// * `f16`: the native conversion (`vcvt_f32_f16` over `float16x4_t`) needs the
//   primitive `f16` type and the `stdarch_neon_f16` intrinsics, both still unstable
//   (rust-lang/rust#116909, #136306) and so unavailable on stable Rust. A
//   hand-rolled integer f16->f32 path is not worth the risk for a widen the OoO
//   core already hides (see bf16 above). Revisit once those stabilize.

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

/// `bf16` mixed precision (scalar fallback), mirror of the `f16` impl. (A
/// vectorized `vshll` widen was implemented and measured but showed no throughput
/// gain over this — the wide OoO core already hides the per-lane widen among the
/// FMAs — so the simpler scalar form is kept; see the note above.)
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
// row block at once (where the 8-byte read stays in bounds), and a hardware
// `i8` dot (`SDOT`) where the `dotprod` extension is present — the NEON analogue of
// VNNI.

#[cfg(feature = "int8")]
impl SimdOps<i32> for Neon {
    type Reg = int32x4_t;
    const LANES: usize = 4;
    const ALIGN: usize = 16;

    #[inline(always)]
    unsafe fn zero(self) -> int32x4_t {
        unsafe { vdupq_n_s32(0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> int32x4_t {
        unsafe { vdupq_n_s32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> int32x4_t {
        unsafe { vld1q_s32(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: int32x4_t) {
        unsafe { vst1q_s32(p, v) }
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
    unsafe fn reduce_sum(self, v: int32x4_t) -> i32 {
        unsafe { vaddvq_s32(v) }
    }
}

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

// ---- complex (vectorized, bit-identical to the scalar fallback) ----
//
// The multiply runs in-register on the interleaved `[re, im, ...]` layout with
// plain `vmul`/`vsub`/`vadd` + a lane blend (helpers below), replacing the old
// stack-spill → `num_complex` → reload stub. Each partial product is its own
// rounded `vmul`, the real lane a `vsub` and the imag lane a `vadd` — exactly the
// rounding `num_complex`'s `re*re - im*im` / `re*im + im*re` performs, so the
// result is bit-identical to the scalar path (cross-checked by
// `neon_complex_bit_identical`).
//
// The ARMv8.3 `FCMLA`/`FCADD` (fcma) complex MACs are deliberately *not* used:
// they fold the multiply and add into a single fused rounding step, which would
// change the result bits versus the unfused scalar reference and break the
// determinism contract (the float analogue of why `sdot`/`bfmmla` are out of
// scope — see the `SimdOps::accumulate_tile` doc).

/// Complex `a * b` for the c32 interleaved layout `[re0, im0, re1, im1]` (two
/// complex per register), bit-identical to `num_complex`'s `Mul`.
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe {
        let a_re = vtrn1q_f32(a, a); // [re0, re0, re1, re1]
        let a_im = vtrn2q_f32(a, a); // [im0, im0, im1, im1]
        let b_sw = vrev64q_f32(b); // [im, re, im, re]
        let p1 = vmulq_f32(a_re, b); // [re*re, re*im, ...]
        let p2 = vmulq_f32(a_im, b_sw); // [im*im, im*re, ...]
        let re = vsubq_f32(p1, p2); // real lanes: re*re - im*im
        let im = vaddq_f32(p1, p2); // imag lanes: re*im + im*re
        // Blend: real from the even lanes of `re`, imag from the odd lanes of `im`.
        let mask = vld1q_u32([0u32, u32::MAX, 0, u32::MAX].as_ptr());
        vbslq_f32(mask, im, re)
    }
}

/// Complex `a * b` for the c64 layout `[re, im]` (one complex per register),
/// bit-identical to `num_complex`'s `Mul`.
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    unsafe {
        let a_re = vtrn1q_f64(a, a); // [re, re]
        let a_im = vtrn2q_f64(a, a); // [im, im]
        let b_sw = vextq_f64::<1>(b, b); // [im, re]
        let p1 = vmulq_f64(a_re, b);
        let p2 = vmulq_f64(a_im, b_sw);
        let re = vsubq_f64(p1, p2);
        let im = vaddq_f64(p1, p2);
        let mask = vld1q_u64([0u64, u64::MAX].as_ptr());
        vbslq_f64(mask, im, re)
    }
}

#[cfg(feature = "complex")]
impl SimdOps<Complex<f32>> for Neon {
    type Reg = float32x4_t; // 2 complex = 4 f32, interleaved
    const LANES: usize = 2;
    const ALIGN: usize = 16;

    #[inline(always)]
    unsafe fn zero(self) -> float32x4_t {
        unsafe { vdupq_n_f32(0.0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f32>) -> float32x4_t {
        unsafe {
            let t = [v.re, v.im, v.re, v.im];
            vld1q_f32(t.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f32>) -> float32x4_t {
        unsafe { vld1q_f32(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f32>) -> float32x4_t {
        unsafe { vld1q_f32(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f32>, v: float32x4_t) {
        unsafe { vst1q_f32(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f32>, v: float32x4_t) {
        unsafe { vst1q_f32(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { cmul_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        // `a*b` (unfused complex multiply) then a lane-wise `+ c`, matching
        // `num_complex`'s `a * b + c` rounding exactly. A free `cmul_f32` (not
        // `self.mul`) avoids the ambiguity with the real `SimdOps<f32>::mul`.
        unsafe { vaddq_f32(cmul_f32(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: float32x4_t) -> Complex<f32> {
        unsafe {
            let mut t = [0.0f32; 4];
            vst1q_f32(t.as_mut_ptr(), v);
            Complex::new(t[0] + t[2], t[1] + t[3])
        }
    }
}

#[cfg(feature = "complex")]
impl SimdOps<Complex<f64>> for Neon {
    type Reg = float64x2_t; // 1 complex = 2 f64
    const LANES: usize = 1;
    const ALIGN: usize = 16;

    #[inline(always)]
    unsafe fn zero(self) -> float64x2_t {
        unsafe { vdupq_n_f64(0.0) }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f64>) -> float64x2_t {
        unsafe {
            let t = [v.re, v.im];
            vld1q_f64(t.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f64>) -> float64x2_t {
        unsafe { vld1q_f64(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f64>) -> float64x2_t {
        unsafe { vld1q_f64(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f64>, v: float64x2_t) {
        unsafe { vst1q_f64(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f64>, v: float64x2_t) {
        unsafe { vst1q_f64(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { cmul_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vaddq_f64(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        // `a*b` (unfused) then lane-wise `+ c`, matching `num_complex` exactly.
        unsafe { vaddq_f64(cmul_f64(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: float64x2_t) -> Complex<f64> {
        unsafe {
            let mut t = [0.0f64; 2];
            vst1q_f64(t.as_mut_ptr(), v);
            Complex::new(t[0], t[1])
        }
    }
}

// Cross-check that the vectorized NEON complex path is **bit-identical to the
// scalar fallback** (the Part 2 acceptance gate): the c32/c64 multiply and FMA
// against `num_complex`'s unfused arithmetic.
#[cfg(all(test, feature = "complex"))]
mod tests {
    use super::Neon;
    use core::arch::aarch64::*;

    /// Deterministic SplitMix64 — reproducible, no dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A normal (non-zero, finite, non-overflowing) `f32`: any sign/mantissa,
        /// exponent in `2^[-63, 56]` so products of two stay finite.
        fn norm_f32(&mut self) -> f32 {
            let u = self.next();
            let sign = (u & 1) << 31;
            let exp = ((u >> 1) % 120 + 64) << 23;
            let mant = (u >> 16) & 0x007F_FFFF;
            f32::from_bits((sign | exp | mant) as u32)
        }
        /// `f32` test value: occasionally a tricky edge (incl. signed zero), else normal.
        fn f32(&mut self) -> f32 {
            const EDGE: [f32; 10] = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.25, -7.5, 123.0];
            let u = self.next();
            if u & 7 == 0 {
                EDGE[(u >> 3) as usize % EDGE.len()]
            } else {
                self.norm_f32()
            }
        }
        fn norm_f64(&mut self) -> f64 {
            let u = self.next();
            let m = self.next();
            let sign = u & 0x8000_0000_0000_0000;
            let exp = ((u >> 1) % 384 + 768) << 52;
            let mant = m & 0x000F_FFFF_FFFF_FFFF;
            f64::from_bits(sign | exp | mant)
        }
        fn f64(&mut self) -> f64 {
            const EDGE: [f64; 10] = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.25, -7.5, 123.0];
            let u = self.next();
            if u & 7 == 0 {
                EDGE[(u >> 3) as usize % EDGE.len()]
            } else {
                self.norm_f64()
            }
        }
    }

    /// The vectorized c32/c64 `mul` and `mul_add` must reproduce `num_complex`'s
    /// (unfused) `a*b` / `a*b + c` bit-for-bit — the determinism contract that lets
    /// a future FCMLA-enabled CPU and the scalar fallback agree to the bit.
    #[test]
    fn neon_complex_bit_identical() {
        use crate::simd::SimdOps;
        use num_complex::Complex;

        let mut r = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..30_000 {
            // ---- c32: two complex per register ----
            let a = [
                Complex::new(r.f32(), r.f32()),
                Complex::new(r.f32(), r.f32()),
            ];
            let b = [
                Complex::new(r.f32(), r.f32()),
                Complex::new(r.f32(), r.f32()),
            ];
            let c = [
                Complex::new(r.f32(), r.f32()),
                Complex::new(r.f32(), r.f32()),
            ];
            unsafe {
                let ar = vld1q_f32(a.as_ptr() as *const f32);
                let br = vld1q_f32(b.as_ptr() as *const f32);
                let cr = vld1q_f32(c.as_ptr() as *const f32);

                let mut got = [Complex::new(0.0f32, 0.0); 2];
                vst1q_f32(
                    got.as_mut_ptr() as *mut f32,
                    <Neon as SimdOps<Complex<f32>>>::mul(Neon, ar, br),
                );
                for i in 0..2 {
                    let want = a[i] * b[i];
                    assert_eq!(
                        (got[i].re.to_bits(), got[i].im.to_bits()),
                        (want.re.to_bits(), want.im.to_bits()),
                        "c32 mul lane {i}: a={:?} b={:?}",
                        a[i],
                        b[i]
                    );
                }
                vst1q_f32(
                    got.as_mut_ptr() as *mut f32,
                    <Neon as SimdOps<Complex<f32>>>::mul_add(Neon, ar, br, cr),
                );
                for i in 0..2 {
                    let want = a[i] * b[i] + c[i];
                    assert_eq!(
                        (got[i].re.to_bits(), got[i].im.to_bits()),
                        (want.re.to_bits(), want.im.to_bits()),
                        "c32 mul_add lane {i}: a={:?} b={:?} c={:?}",
                        a[i],
                        b[i],
                        c[i]
                    );
                }
            }

            // ---- c64: one complex per register ----
            let a = Complex::new(r.f64(), r.f64());
            let b = Complex::new(r.f64(), r.f64());
            let c = Complex::new(r.f64(), r.f64());
            unsafe {
                let ar = vld1q_f64(&a as *const Complex<f64> as *const f64);
                let br = vld1q_f64(&b as *const Complex<f64> as *const f64);
                let cr = vld1q_f64(&c as *const Complex<f64> as *const f64);

                let mut got = Complex::new(0.0f64, 0.0);
                vst1q_f64(
                    &mut got as *mut Complex<f64> as *mut f64,
                    <Neon as SimdOps<Complex<f64>>>::mul(Neon, ar, br),
                );
                let want = a * b;
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "c64 mul: a={a:?} b={b:?}"
                );
                vst1q_f64(
                    &mut got as *mut Complex<f64> as *mut f64,
                    <Neon as SimdOps<Complex<f64>>>::mul_add(Neon, ar, br, cr),
                );
                let want = a * b + c;
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "c64 mul_add: a={a:?} b={b:?} c={c:?}"
                );
            }
        }
    }
}
