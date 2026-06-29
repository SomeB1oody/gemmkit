//! AVX2 + FMA ISA token (x86 / x86-64).
//!
//! `f32` → 256-bit registers (8 lanes), `f64` → 256-bit (4 lanes). The whole
//! file is one `#[target_feature(enable = "avx2,fma")]` trampoline plus thin
//! `#[inline(always)]` intrinsic wrappers — that is the *entire* per-ISA cost of
//! a new instruction set in this design.

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

/// AVX2 + FMA ISA token.
#[derive(Copy, Clone, Default)]
pub struct Fma;

impl Simd for Fma {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // `f16c` is enabled alongside `avx2,fma`: every AVX2+FMA CPU (Haswell and
        // later) also has F16C, which the mixed-precision `f16` path needs for its
        // `vcvtph2ps`/`vcvtps2ph` conversions. The dispatch ladder still checks
        // `f16c` before selecting this token for an `f16` GEMM (defensive); the
        // `f32`/`f64`/`bf16` paths emit no F16C instructions, so enabling it is
        // harmless even on the hypothetical AVX2-without-F16C part.
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: the caller of `vectorize` (the runtime dispatcher) guarantees
        // the CPU supports avx2+fma(+f16c); `inner` then establishes the codegen
        // context, and `f` inlines into it.
        unsafe { inner(f) }
    }
}

impl SimdOps<f32> for Fma {
    type Reg = __m256;
    const LANES: usize = 8;
    const ALIGN: usize = 32;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm256_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        unsafe { _mm256_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f32) -> Self::Reg {
        unsafe { _mm256_load_ps(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        unsafe { _mm256_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm256_store_ps(p, v) }
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
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let s = _mm_add_ps(lo, hi); // 4 partial sums
            let shuf = _mm_movehdup_ps(s); // [1,1,3,3]
            let sums = _mm_add_ps(s, shuf); // [0+1, _, 2+3, _]
            let hi2 = _mm_movehl_ps(shuf, sums); // bring 2+3 down
            let r = _mm_add_ss(sums, hi2);
            _mm_cvtss_f32(r)
        }
    }
}

impl SimdOps<f64> for Fma {
    type Reg = __m256d;
    const LANES: usize = 4;
    const ALIGN: usize = 32;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm256_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        unsafe { _mm256_set1_pd(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f64) -> Self::Reg {
        unsafe { _mm256_load_pd(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        unsafe { _mm256_loadu_pd(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm256_store_pd(p, v) }
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
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(v, 1);
            let lo = _mm256_castpd256_pd128(v);
            let s = _mm_add_pd(lo, hi); // 2 partial sums
            let sh = _mm_unpackhi_pd(s, s);
            let r = _mm_add_sd(s, sh);
            _mm_cvtsd_f64(r)
        }
    }
}

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (8-wide __m256) ----

/// `f16` via F16C: 8 lanes widen with `vcvtph2ps`, narrow with `vcvtps2ph`
/// (round-to-nearest-even, matching `half::f16::from_f32`).
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Fma {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> __m256 {
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> __m256 {
        unsafe {
            let lo = _mm_cvtph_ps(_mm_cvtsi32_si128(v.to_bits() as i32)); // lane0 = f32(v)
            _mm256_broadcastss_ps(lo)
        }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> __m256 {
        // C (== Out == Lhs == f16) is widened exactly like an A panel.
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: __m256) {
        unsafe {
            // F16C `vcvtps2ph` takes a 3-bit rounding immediate; round-to-nearest-
            // even is `_MM_FROUND_TO_NEAREST_INT` (0), matching `half::f16::from_f32`.
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(p as *mut __m128i, h);
        }
    }
}

/// `bf16` via integer ops: widen is a 16-bit left shift into `f32`; narrow is the
/// round-to-nearest-even bias trick (`+ ((bits>>16)&1) + 0x7FFF`, then `>>16`).
/// **Bit-identical to `half::bf16::from_f32`** including NaN (mapped to
/// `(bits>>16) | 0x0040`), so the vector and scalar paths never diverge.
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Fma {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> __m256 {
        unsafe {
            let w = _mm_loadu_si128(p as *const __m128i); // 8 × u16
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w)))
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> __m256 {
        unsafe { _mm256_set1_ps(f32::from_bits((v.to_bits() as u32) << 16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> __m256 {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: __m256) {
        unsafe {
            let bits = _mm256_castps_si256(v);
            // RNE round-and-truncate for finite values.
            let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));
            let bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7FFF));
            let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
            // NaN lanes (|bits| > 0x7F80_0000): half forces `(bits>>16) | 0x0040`.
            let abs = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            let is_nan = _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(0x7F80_0000));
            let nan_out = _mm256_or_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(0x0040));
            let out = _mm256_blendv_epi8(rounded, nan_out, is_nan);
            // Pack 8 × u32 (each < 0x10000) into 8 contiguous u16 (order preserved).
            let lo = _mm256_castsi256_si128(out);
            let hi = _mm256_extracti128_si256::<1>(out);
            _mm_storeu_si128(p as *mut __m128i, _mm_packus_epi32(lo, hi));
        }
    }
}

// ---- integer: i8 inputs, i32 accumulator (8-wide __m256i, AVX2 integer ops) ----

#[cfg(feature = "int8")]
impl SimdOps<i32> for Fma {
    type Reg = __m256i;
    const LANES: usize = 8;
    const ALIGN: usize = 32;

    #[inline(always)]
    unsafe fn zero(self) -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> __m256i {
        unsafe { _mm256_set1_epi32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> __m256i {
        unsafe { _mm256_load_si256(p as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: __m256i) {
        unsafe { _mm256_store_si256(p as *mut __m256i, v) }
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
    unsafe fn reduce_sum(self, v: __m256i) -> i32 {
        unsafe {
            let hi = _mm256_extracti128_si256::<1>(v);
            let lo = _mm256_castsi256_si128(v);
            let s = _mm_add_epi32(lo, hi); // 4 partials
            let sh = _mm_shuffle_epi32::<0b01_00_11_10>(s);
            let s = _mm_add_epi32(s, sh);
            let sh = _mm_shuffle_epi32::<0b00_00_00_01>(s);
            _mm_cvtsi128_si32(_mm_add_epi32(s, sh))
        }
    }
}

/// `i8 -> i32`: sign-extend 8 LHS bytes on load, broadcast a sign-extended RHS byte;
/// `Out == Acc == i32`, so `load_out`/`store_out` are plain `i32` load/store.
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
}

// ---- complex: interleaved (re,im), shuffle + fmaddsub complex multiply ----
//
// `Reg` is `__m256`/`__m256d` — the same as `SimdOps<f32>`/`<f64>` — so internal
// complex multiplies go through these free helpers (a `self.mul` call would be
// ambiguous between the real and complex `SimdOps`).

/// Complex multiply of 4 interleaved `f32` complex: `(ar*br - ai*bi, ar*bi + ai*br)`.
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_ps(a: __m256, b: __m256) -> __m256 {
    unsafe {
        let b_re = _mm256_moveldup_ps(b); // [br br ...] (dup real)
        let b_im = _mm256_movehdup_ps(b); // [bi bi ...] (dup imag)
        let a_sw = _mm256_permute_ps::<0xB1>(a); // [ai ar ...] (swap each pair)
        let t = _mm256_mul_ps(a_sw, b_im); // [ai*bi  ar*bi ...]
        _mm256_fmaddsub_ps(a, b_re, t) // even: a*b_re - t, odd: + t
    }
}

/// Complex multiply of 2 interleaved `f64` complex.
#[cfg(feature = "complex")]
#[inline(always)]
unsafe fn cmul_pd(a: __m256d, b: __m256d) -> __m256d {
    unsafe {
        let b_re = _mm256_movedup_pd(b);
        let b_im = _mm256_permute_pd::<0b1111>(b);
        let a_sw = _mm256_permute_pd::<0b0101>(a);
        let t = _mm256_mul_pd(a_sw, b_im);
        _mm256_fmaddsub_pd(a, b_re, t)
    }
}

#[cfg(feature = "complex")]
impl SimdOps<Complex<f32>> for Fma {
    type Reg = __m256; // 4 complex = 8 f32, interleaved [r0 i0 r1 i1 r2 i2 r3 i3]
    const LANES: usize = 4;
    const ALIGN: usize = 32;

    #[inline(always)]
    unsafe fn zero(self) -> __m256 {
        unsafe { _mm256_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f32>) -> __m256 {
        // Build the (re,im) pair in a 128-bit lane and broadcast it to both halves —
        // no `[f32;2]`-reinterpreted-as-f64 misaligned load (which would be UB).
        unsafe {
            let m = _mm_set_ps(v.im, v.re, v.im, v.re); // [re im re im]
            _mm256_broadcast_ps(&m)
        }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f32>) -> __m256 {
        unsafe { _mm256_load_ps(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f32>) -> __m256 {
        unsafe { _mm256_loadu_ps(p as *const f32) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f32>, v: __m256) {
        unsafe { _mm256_store_ps(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f32>, v: __m256) {
        unsafe { _mm256_storeu_ps(p as *mut f32, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m256, b: __m256) -> __m256 {
        unsafe { cmul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { _mm256_add_ps(cmul_ps(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m256) -> Complex<f32> {
        // gemv-only and perf-noncritical: sum even (re) / odd (im) lanes via a store.
        unsafe {
            let mut t = [0.0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), v);
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for c in 0..4 {
                re += t[2 * c];
                im += t[2 * c + 1];
            }
            Complex::new(re, im)
        }
    }
}

#[cfg(feature = "complex")]
impl SimdOps<Complex<f64>> for Fma {
    type Reg = __m256d; // 2 complex = 4 f64
    const LANES: usize = 2;
    const ALIGN: usize = 32;

    #[inline(always)]
    unsafe fn zero(self) -> __m256d {
        unsafe { _mm256_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: Complex<f64>) -> __m256d {
        // Build (re,im) in a 128-bit lane and broadcast to both complex lanes (no
        // misaligned `[f64;2]`-as-`__m128d` load).
        unsafe {
            let m = _mm_set_pd(v.im, v.re); // [re im]
            _mm256_broadcast_pd(&m)
        }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const Complex<f64>) -> __m256d {
        unsafe { _mm256_load_pd(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const Complex<f64>) -> __m256d {
        unsafe { _mm256_loadu_pd(p as *const f64) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut Complex<f64>, v: __m256d) {
        unsafe { _mm256_store_pd(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut Complex<f64>, v: __m256d) {
        unsafe { _mm256_storeu_pd(p as *mut f64, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { cmul_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { _mm256_add_pd(cmul_pd(a, b), c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m256d) -> Complex<f64> {
        unsafe {
            let mut t = [0.0f64; 4];
            _mm256_storeu_pd(t.as_mut_ptr(), v);
            Complex::new(t[0] + t[2], t[1] + t[3])
        }
    }
}
