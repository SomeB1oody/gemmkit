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

use super::{Simd, SimdOps};

/// AVX2 + FMA ISA token.
#[derive(Copy, Clone, Default)]
pub struct Fma;

impl Simd for Fma {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: the caller of `vectorize` (the runtime dispatcher) guarantees
        // the CPU supports avx2+fma; `inner` then establishes the codegen
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
