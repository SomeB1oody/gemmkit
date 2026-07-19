//! AVX2 + FMA ISA token (x86 / x86-64)
//!
//! `f32` uses 256-bit registers (8 lanes), `f64` 256-bit (4 lanes). The token's job is
//! [`Simd::vectorize`], a `#[target_feature(enable = "avx2,fma,f16c")]` trampoline, plus
//! thin `#[inline(always)]` wrappers around the AVX2/FMA intrinsics: that is the entire
//! per-ISA surface a new instruction set has to fill in

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};

/// AVX2 + FMA ISA token: 256-bit vector registers
#[derive(Copy, Clone, Default)]
pub struct Fma;

impl Simd for Fma {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // f16c enables unconditionally alongside avx2,fma: every AVX2+FMA CPU
        // (Haswell and later) also has F16C, which the f16 path needs for
        // vcvtph2ps/vcvtps2ph. The f32/f64/bf16 paths never emit an F16C
        // instruction, so enabling it is a no-op for them even on a hypothetical
        // AVX2-without-F16C part
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: the dispatcher guarantees avx2+fma(+f16c) support before calling
        // vectorize; inner establishes the codegen context and f inlines into it
        unsafe { inner(f) }
    }
}

impl SimdOps<f32> for Fma {
    type Reg = __m256;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm256_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        unsafe { _mm256_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        unsafe { _mm256_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm256_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // vfnmadd213ps computes c - a*b
        unsafe { _mm256_fnmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // MAXPS returns the 2nd operand on an unordered compare, so a NaN `a` returns
        // `b`, matching the trait's NaN-in-`a` contract
        unsafe { _mm256_max_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_min_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        unsafe {
            // Fold the 8 f32 lanes to 1 in log2(8) = 3 adds: 256 -> 128-bit halves, then
            // pairwise within the 128-bit lane
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let s = _mm_add_ps(lo, hi); // 4 partial sums
            let shuf = _mm_movehdup_ps(s); // duplicate odd lanes: [1, 1, 3, 3]
            let sums = _mm_add_ps(s, shuf); // lane 0 = 0+1, lane 2 = 2+3
            let hi2 = _mm_movehl_ps(shuf, sums); // bring lane 2 (2+3) down to lane 0
            let r = _mm_add_ss(sums, hi2);
            _mm_cvtss_f32(r)
        }
    }
}

impl SimdOps<f64> for Fma {
    type Reg = __m256d;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm256_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        unsafe { _mm256_set1_pd(v) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        unsafe { _mm256_loadu_pd(p) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm256_storeu_pd(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_mul_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // vfnmadd213pd computes c - a*b
        unsafe { _mm256_fnmadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // MAXPD returns the 2nd operand when unordered: NaN `a` -> `b`
        unsafe { _mm256_max_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_min_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        unsafe {
            // Fold the 4 f64 lanes to 1: 256 -> 128-bit halves, then the 2 lanes within
            let hi = _mm256_extractf128_pd(v, 1);
            let lo = _mm256_castpd256_pd128(v);
            let s = _mm_add_pd(lo, hi); // 2 partial sums
            let sh = _mm_unpackhi_pd(s, s); // lane 1 duplicated down to lane 0
            let r = _mm_add_sd(s, sh);
            _mm_cvtsd_f64(r)
        }
    }
}

// Mixed precision: f16/bf16 inputs, f32 accumulator, 8-wide __m256

/// f16 via F16C: widen all 8 lanes with `vcvtph2ps`, narrow with `vcvtps2ph`
/// (round-to-nearest-even, matching `half::f16::from_f32`)
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Fma {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> __m256 {
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> __m256 {
        unsafe {
            let lo = _mm_cvtph_ps(_mm_cvtsi32_si128(v.to_bits() as i32)); // lane 0 = f32(v)
            _mm256_broadcastss_ps(lo)
        }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> __m256 {
        // Qualified because the f32-output twin adds a 2nd KernelSimd<f16, .., f32, ..>
        // impl, so the plain method name would be ambiguous here
        unsafe { <Self as KernelSimd<f16, f16, f32, f16>>::load_lhs(self, p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: __m256) {
        unsafe {
            // _MM_FROUND_TO_NEAREST_INT (0) is round-to-nearest-even, matching
            // half::f16::from_f32. _MM_FROUND_NO_EXC is deliberately not OR'd in: the
            // 256-bit F16C vcvtps2ph has no suppress-all-exceptions field (that's
            // EVEX-only), and the intrinsic rejects the bit at compile time
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(p as *mut __m128i, h);
        }
    }
}

/// bf16 via plain integer ops (no dedicated bf16 hardware needed): widening is a 16-bit
/// left shift into the top of an f32; narrowing is the round-to-nearest-even bias trick
/// (add `((bits>>16)&1) + 0x7FFF`, then shift right 16). The narrowing side is
/// bit-identical to `half::bf16::from_f32`, NaN included (forced to `(bits>>16) | 0x0040`),
/// so this conversion matches the scalar path exactly; that is what keeps full and edge
/// tiles of the same matrix consistent, even though the `vdpbf16ps` dot kernel's fused
/// 2-term MAC rounds differently from this widen-and-FMA path
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Fma {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> __m256 {
        unsafe {
            let w = _mm_loadu_si128(p as *const __m128i); // 8 x u16
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w)))
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> __m256 {
        unsafe { _mm256_set1_ps(f32::from_bits((v.to_bits() as u32) << 16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> __m256 {
        // Qualified for the same reason as the f16 twin: a 2nd KernelSimd<bf16, ..> impl
        // makes the plain method name ambiguous
        unsafe { <Self as KernelSimd<bf16, bf16, f32, bf16>>::load_lhs(self, p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: __m256) {
        unsafe {
            let bits = _mm256_castps_si256(v);
            // Round-to-nearest-even bias trick for finite values
            let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));
            let bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7FFF));
            let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
            // NaN lanes (|bits| > 0x7F80_0000) bypass rounding: half forces (bits>>16) | 0x0040
            let abs = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            let is_nan = _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x7F80_0000));
            let nan_out = _mm256_or_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(0x0040));
            let out = _mm256_blendv_epi8(rounded, nan_out, is_nan);
            // Each lane is now < 0x10000: pack the 8 u32 lanes into 8 contiguous u16
            let lo = _mm256_castsi256_si128(out);
            let hi = _mm256_extracti128_si256::<1>(out);
            _mm_storeu_si128(p as *mut __m128i, _mm_packus_epi32(lo, hi));
        }
    }
}

// Integer: i8 inputs, i32 accumulator, 8-wide __m256i via plain AVX2 integer ops

#[cfg(feature = "int8")]
impl SimdOps<i32> for Fma {
    type Reg = __m256i;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn zero(self) -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> __m256i {
        unsafe { _mm256_set1_epi32(v) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: __m256i) {
        unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_mullo_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(_mm256_mullo_epi32(a, b), c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: __m256i, b: __m256i, c: __m256i) -> __m256i {
        // c - a*b, wrapping i32. Satisfies the trait; the integer kernel never calls it
        unsafe { _mm256_sub_epi32(c, _mm256_mullo_epi32(a, b)) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m256i) -> i32 {
        unsafe {
            // Fold the 8 i32 lanes to 1 in log2(8) = 3 adds
            let hi = _mm256_extracti128_si256::<1>(v);
            let lo = _mm256_castsi256_si128(v);
            let s = _mm_add_epi32(lo, hi); // 4 partials
            let sh = _mm_shuffle_epi32::<0b01_00_11_10>(s); // swap the 2 halves
            let s = _mm_add_epi32(s, sh); // 2 partials, replicated
            let sh = _mm_shuffle_epi32::<0b00_00_00_01>(s); // lane 1 down to lane 0
            _mm_cvtsi128_si32(_mm_add_epi32(s, sh))
        }
    }
}

/// Requantize one quad (4 `i32` lanes, as `__m128i`) of a `__m256i` accumulator to 4
/// integral `i32` in `[lo, hi]`, following the scalar map exactly: widen `i32 -> f64`,
/// multiply by `scale` (both exact), round to nearest-even in hardware, add `zp`, clamp to
/// `[lo, hi]`, convert back to `i32` (exact, since the clamped value is already integral).
/// `#[inline(always)]` so the intrinsics fold straight into the caller's
/// `#[target_feature]` context
///
/// `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC` on `_mm256_round_pd` is round-to-
/// nearest-even with the precision exception suppressed; unlike the F16C `vcvtps2ph` used
/// above, the 256-bit VEX `vroundpd` accepts the suppress-exception bit
#[cfg(feature = "int8")]
#[inline(always)]
unsafe fn requant_quad_fma(
    x: __m128i,
    scale_v: __m256d,
    zp_v: __m256d,
    lo_v: __m256d,
    hi_v: __m256d,
) -> __m128i {
    unsafe {
        let t = _mm256_cvtepi32_pd(x);
        let t = _mm256_mul_pd(t, scale_v);
        let t = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(t);
        let u = _mm256_add_pd(t, zp_v);
        let u = _mm256_max_pd(u, lo_v);
        let u = _mm256_min_pd(u, hi_v);
        _mm256_cvtpd_epi32(u)
    }
}

/// Vectorized `i32 -> i8` requantize store for [`Fma`] (see [`KernelSimd::requant_store`]
/// for the bit-for-bit-with-scalar contract): split the 8 `i32` lanes into 2 `__m128i`
/// quads, requantize each in `f64` ([`requant_quad_fma`]), then gather the **low byte** of
/// each of the 8 integral, pre-clamped lanes into 8 contiguous output bytes. This is a
/// TRUNCATING byte gather, not a saturating `packs`/`packus`: the lanes are already
/// clamped into `[lo, hi]`, so a saturating pack would double-clamp and give the wrong
/// answer for the `u8`/`[0, 255]` phase
///
/// # Safety
/// `dst` valid for 8 byte writes; run inside [`Fma`]'s `avx2,fma` [`Simd::vectorize`] context
#[cfg(feature = "int8")]
#[inline(always)]
unsafe fn requant_store_fma(dst: *mut i8, v: __m256i, scale: f64, zp: i32, lo: i32, hi: i32) {
    unsafe {
        let scale_v = _mm256_set1_pd(scale);
        let zp_v = _mm256_set1_pd(zp as f64);
        let lo_v = _mm256_set1_pd(lo as f64);
        let hi_v = _mm256_set1_pd(hi as f64);
        let i_lo = requant_quad_fma(_mm256_castsi256_si128(v), scale_v, zp_v, lo_v, hi_v);
        let i_hi = requant_quad_fma(_mm256_extracti128_si256::<1>(v), scale_v, zp_v, lo_v, hi_v);
        // pshufb mask: gather source bytes {0, 4, 8, 12} (the low byte of each i32 lane)
        // into the low 4 output bytes; the high 12 mask bytes have the sign bit set, which
        // zeros those output lanes
        let mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
        let lo_u32 = _mm_cvtsi128_si32(_mm_shuffle_epi8(i_lo, mask)) as u32;
        let hi_u32 = _mm_cvtsi128_si32(_mm_shuffle_epi8(i_hi, mask)) as u32;
        // x86 is little-endian, so byte l of packed is output lane l: lanes 0..4 from the
        // low quad, lanes 4..8 from the high quad
        let packed = (lo_u32 as u64) | ((hi_u32 as u64) << 32);
        core::ptr::write_unaligned(dst as *mut u64, packed);
    }
}

/// i8 -> i32 widen kernel: sign-extend 8 LHS bytes on load, broadcast a sign-extended
/// RHS byte. `Out == Acc == i32` here, so `load_out`/`store_out` are plain load/store
#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Fma {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> __m256i {
        unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> __m256i {
        unsafe { _mm256_set1_epi32(v as i32) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: __m256i) {
        unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
    }

    const REQUANT_VECTOR: bool = true;
    #[inline(always)]
    unsafe fn requant_store(self, dst: *mut i8, v: __m256i, scale: f64, zp: i32, lo: i32, hi: i32) {
        unsafe { requant_store_fma(dst, v, scale, zp, lo, hi) }
    }
}

// Complex (AVX2): the real Reg is the plain f32/f64 register, LANES the real lane count
// (8 / 4), and complex GEMM routes through the shared SoA soa_microkernel
#[cfg(feature = "complex")]
impl_complex_simd!(Fma, f32, __m256, 8);
#[cfg(feature = "complex")]
impl_complex_simd!(Fma, f64, __m256d, 4);
