//! AVX-512 ISA token (x86 / x86-64).
//!
//! `f32` → 512-bit registers (16 lanes), `f64` → 512-bit (8 lanes). Requires the
//! stable AVX-512 intrinsics (Rust 1.89+). Adding this entire instruction set on
//! top of the FMA token cost exactly this file plus one dispatch line — the
//! concrete demonstration of the `SimdOps` abstraction's leverage.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use half::{bf16, f16};

use super::{KernelSimd, Simd, SimdOps};

/// AVX-512 (foundation) ISA token.
#[derive(Copy, Clone, Default)]
pub struct Avx512;

impl Simd for Avx512 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: dispatcher guarantees avx512f support; `inner` sets the
        // codegen context and `f` inlines into it.
        unsafe { inner(f) }
    }
}

impl SimdOps<f32> for Avx512 {
    type Reg = __m512;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        unsafe { _mm512_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f32) -> Self::Reg {
        unsafe { _mm512_load_ps(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        unsafe { _mm512_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm512_store_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm512_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        unsafe { _mm512_reduce_add_ps(v) }
    }
}

impl SimdOps<f64> for Avx512 {
    type Reg = __m512d;
    const LANES: usize = 8;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm512_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        unsafe { _mm512_set1_pd(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f64) -> Self::Reg {
        unsafe { _mm512_load_pd(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        unsafe { _mm512_loadu_pd(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm512_store_pd(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm512_storeu_pd(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_mul_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        unsafe { _mm512_reduce_add_pd(v) }
    }
}

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (16-wide __m512) ----

/// `f16` via AVX-512 `vcvtph2ps` / `vcvtps2ph` (round-to-nearest-even on store,
/// matching `half::f16::from_f32`).
impl KernelSimd<f16, f16, f32, f16> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> __m512 {
        unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> __m512 {
        // Broadcast the f16 bits to all 16 lanes, then widen with the AVX-512
        // `vcvtph2ps` (zmm) — pure AVX-512F, so this path needs no separate F16C
        // feature (unlike the FMA token's 256-bit conversion).
        unsafe { _mm512_cvtph_ps(_mm256_set1_epi16(v.to_bits() as i16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> __m512 {
        // C (== Out == Lhs == f16) is widened exactly like an A panel.
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: __m512) {
        unsafe {
            let h = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
            _mm256_storeu_si256(p as *mut __m256i, h);
        }
    }
}

/// `bf16` via integer ops (all AVX-512F): widen = 16-bit left shift into `f32`;
/// narrow = round-to-nearest-even bias trick then truncate. **Bit-identical to
/// `half::bf16::from_f32`** including NaN (which half maps to `(bits>>16) | 0x0040`,
/// not the bias trick's garbage), so the vector and scalar paths never diverge.
impl KernelSimd<bf16, bf16, f32, bf16> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> __m512 {
        unsafe {
            let w = _mm256_loadu_si256(p as *const __m256i); // 16 × u16
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w)))
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> __m512 {
        unsafe { _mm512_set1_ps(f32::from_bits((v.to_bits() as u32) << 16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> __m512 {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: __m512) {
        unsafe {
            let bits = _mm512_castps_si512(v);
            // RNE round-and-truncate for finite values.
            let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(bits), _mm512_set1_epi32(1));
            let bias = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF));
            let rounded = _mm512_srli_epi32::<16>(_mm512_add_epi32(bits, bias));
            // NaN lanes (|bits| > 0x7F80_0000): half forces `(bits>>16) | 0x0040`.
            let abs = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
            let nan = _mm512_cmpgt_epi32_mask(abs, _mm512_set1_epi32(0x7F80_0000));
            let nan_out = _mm512_or_si512(_mm512_srli_epi32::<16>(bits), _mm512_set1_epi32(0x0040));
            let out = _mm512_mask_blend_epi32(nan, rounded, nan_out);
            // Truncate each 32-bit lane to its low 16 bits → 16 contiguous u16.
            _mm256_storeu_si256(p as *mut __m256i, _mm512_cvtepi32_epi16(out));
        }
    }
}
