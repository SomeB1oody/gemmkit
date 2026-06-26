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

use super::{Simd, SimdOps};

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
