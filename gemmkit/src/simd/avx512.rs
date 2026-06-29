//! AVX-512 ISA token (x86 / x86-64).
//!
//! `f32` → 512-bit registers (16 lanes), `f64` → 512-bit (8 lanes). Requires the
//! stable AVX-512 intrinsics (Rust 1.89+).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};

/// Complex multiply of 16 interleaved `f32` complex (8 complex per 512-bit reg).
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_ps_512(a: __m512, b: __m512) -> __m512 {
    unsafe {
        let b_re = _mm512_moveldup_ps(b);
        let b_im = _mm512_movehdup_ps(b);
        let a_sw = _mm512_permute_ps::<0xB1>(a);
        let t = _mm512_mul_ps(a_sw, b_im);
        _mm512_fmaddsub_ps(a, b_re, t)
    }
}

/// Complex multiply of 4 interleaved `f64` complex.
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_pd_512(a: __m512d, b: __m512d) -> __m512d {
    unsafe {
        let b_re = _mm512_movedup_pd(b);
        let b_im = _mm512_permute_pd::<0b11111111>(b);
        let a_sw = _mm512_permute_pd::<0b01010101>(a);
        let t = _mm512_mul_pd(a_sw, b_im);
        _mm512_fmaddsub_pd(a, b_re, t)
    }
}

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
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> __m512 {
        unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> __m512 {
        // Broadcast the f16 bits to all 16 lanes, then widen via `vcvtph2ps`
        // (zmm) — pure AVX-512F, so no separate F16C feature is needed.
        unsafe { _mm512_cvtph_ps(_mm256_set1_epi16(v.to_bits() as i16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> __m512 {
        // C (== Out == Lhs == f16) widens exactly like an A panel.
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
/// narrow = round-to-nearest-even bias trick then truncate. Bit-identical to
/// `half::bf16::from_f32`, including NaN (which half forces to `(bits>>16) | 0x0040`
/// rather than the bias trick's result), so vector and scalar paths never diverge.
#[cfg(feature = "half")]
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

// ---- integer: i8 inputs, i32 accumulator (16-wide __m512i, AVX-512F integer) ----

#[cfg(feature = "int8")]
impl SimdOps<i32> for Avx512 {
    type Reg = __m512i;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> __m512i {
        unsafe { _mm512_load_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_store_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(_mm512_mullo_epi32(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512i) -> i32 {
        unsafe { _mm512_reduce_add_epi32(v) }
    }
}

/// `i8 -> i32`: sign-extend 16 LHS bytes on load, broadcast a sign-extended RHS
/// byte; `Out == Acc == i32`, so `load_out`/`store_out` are plain `i32` load/store.
#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> __m512i {
        unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> __m512i {
        unsafe { _mm512_set1_epi32(v as i32) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }
}

// ---- complex: interleaved (re,im), shuffle + fmaddsub complex multiply ----

#[cfg(feature = "complex")]
impl SimdOps<Complex<f32>> for Avx512 {
    type Reg = __m512; // 8 complex = 16 f32
    const LANES: usize = 8;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512 {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f32>) -> __m512 {
        // Build (re,im) in a 128-bit lane and broadcast it to all 4 lanes (no
        // misaligned `[f32;2]`-as-f64 load, which would be UB).
        unsafe { _mm512_broadcast_f32x4(_mm_set_ps(v.im, v.re, v.im, v.re)) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f32>) -> __m512 {
        unsafe { _mm512_load_ps(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f32>) -> __m512 {
        unsafe { _mm512_loadu_ps(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f32>, v: __m512) {
        unsafe { _mm512_store_ps(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f32>, v: __m512) {
        unsafe { _mm512_storeu_ps(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512, b: __m512) -> __m512 {
        unsafe { cmul_ps_512(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_add_ps(cmul_ps_512(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512) -> Complex<f32> {
        unsafe {
            let mut t = [0.0f32; 16];
            _mm512_storeu_ps(t.as_mut_ptr(), v);
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for c in 0..8 {
                re += t[2 * c];
                im += t[2 * c + 1];
            }
            Complex::new(re, im)
        }
    }
}

#[cfg(feature = "complex")]
impl SimdOps<Complex<f64>> for Avx512 {
    type Reg = __m512d; // 4 complex = 8 f64
    const LANES: usize = 4;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512d {
        unsafe { _mm512_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f64>) -> __m512d {
        // Broadcast the (re,im) pair to all 4 complex lanes (AVX-512F `setr`).
        unsafe { _mm512_setr_pd(v.re, v.im, v.re, v.im, v.re, v.im, v.re, v.im) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f64>) -> __m512d {
        unsafe { _mm512_load_pd(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f64>) -> __m512d {
        unsafe { _mm512_loadu_pd(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f64>, v: __m512d) {
        unsafe { _mm512_store_pd(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f64>, v: __m512d) {
        unsafe { _mm512_storeu_pd(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { cmul_pd_512(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { _mm512_add_pd(cmul_pd_512(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512d) -> Complex<f64> {
        unsafe {
            let mut t = [0.0f64; 8];
            _mm512_storeu_pd(t.as_mut_ptr(), v);
            Complex::new(t[0] + t[2] + t[4] + t[6], t[1] + t[3] + t[5] + t[7])
        }
    }
}
