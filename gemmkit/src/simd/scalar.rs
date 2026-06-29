//! Scalar fallback ISA token: `LANES == 1`, no intrinsics, works everywhere.
//!
//! This is the portability floor and the Miri-checkable reference path. It is
//! also what makes the `SimdOps` abstraction honest: the exact same generic
//! kernel that drives AVX-512 drives this token with a one-element "register".

use half::{bf16, f16};
use num_complex::Complex;

use super::{KernelSimd, Simd, SimdOps};
use crate::scalar::NarrowFloat;

// Complex (scalar fallback): the `Reg` is one `Complex`, arithmetic is num-complex's
// (the complex multiply / FMA the SIMD tokens vectorize). The Miri-checked reference
// for the complex path.
macro_rules! impl_scalar_complex {
    ($t:ty) => {
        impl SimdOps<Complex<$t>> for ScalarTok {
            type Reg = Complex<$t>;
            const LANES: usize = 1;
            const ALIGN: usize = core::mem::align_of::<Complex<$t>>();

            #[inline(always)]
            unsafe fn zero(self) -> Self::Reg {
                Complex::new(0.0, 0.0)
            }
            #[inline(always)]
            unsafe fn splat(self, v: Complex<$t>) -> Self::Reg {
                v
            }
            #[inline(always)]
            unsafe fn load(self, p: *const Complex<$t>) -> Self::Reg {
                unsafe { *p }
            }
            #[inline(always)]
            unsafe fn loadu(self, p: *const Complex<$t>) -> Self::Reg {
                unsafe { *p }
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut Complex<$t>, v: Self::Reg) {
                unsafe { *p = v }
            }
            #[inline(always)]
            unsafe fn storeu(self, p: *mut Complex<$t>, v: Self::Reg) {
                unsafe { *p = v }
            }
            #[inline(always)]
            unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                a * b
            }
            #[inline(always)]
            unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                a + b
            }
            #[inline(always)]
            unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
                a * b + c
            }
            #[inline(always)]
            unsafe fn reduce_sum(self, v: Self::Reg) -> Complex<$t> {
                v
            }
        }
    };
}
impl_scalar_complex!(f32);
impl_scalar_complex!(f64);

/// The scalar (1-lane) ISA token. Always available.
#[derive(Copy, Clone, Default)]
pub struct ScalarTok;

impl Simd for ScalarTok {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // No target feature needed; nothing to enable.
        f()
    }
}

macro_rules! impl_scalar_ops {
    ($t:ty) => {
        impl SimdOps<$t> for ScalarTok {
            type Reg = $t;
            const LANES: usize = 1;
            const ALIGN: usize = core::mem::align_of::<$t>();

            #[inline(always)]
            unsafe fn zero(self) -> Self::Reg {
                0.0
            }
            #[inline(always)]
            unsafe fn splat(self, v: $t) -> Self::Reg {
                v
            }
            #[inline(always)]
            unsafe fn load(self, p: *const $t) -> Self::Reg {
                unsafe { *p }
            }
            #[inline(always)]
            unsafe fn loadu(self, p: *const $t) -> Self::Reg {
                unsafe { *p }
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut $t, v: Self::Reg) {
                unsafe { *p = v }
            }
            #[inline(always)]
            unsafe fn storeu(self, p: *mut $t, v: Self::Reg) {
                unsafe { *p = v }
            }
            #[inline(always)]
            unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                a * b
            }
            #[inline(always)]
            unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                a + b
            }
            #[inline(always)]
            unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
                // Plain `a*b + c` keeps the scalar path bit-reproducible and in
                // agreement with `Float::mul_add` used by the epilogue.
                a * b + c
            }
            #[inline(always)]
            unsafe fn reduce_sum(self, v: Self::Reg) -> $t {
                v
            }
        }
    };
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);

// Mixed-precision (scalar fallback): `f16`/`bf16` widen to `f32` on load and round
// back on store, one element at a time (LANES == 1, so the `Reg` is a bare `f32`).
// This is the portability floor and the Miri-checked reference for the narrow types.
impl KernelSimd<f16, f16, f32, f16> for ScalarTok {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> f32 {
        unsafe { (*p).widen() }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> f32 {
        v.widen()
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> f32 {
        unsafe { (*p).widen() }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: f32) {
        unsafe { *p = f16::narrow(v) }
    }
}

// Integer GEMM (scalar fallback): `i32` accumulator ops and the `i8 -> i32`
// widen-load, one element at a time. Wrapping arithmetic = conventional integer
// overflow semantics. This is the Miri-checked reference for the integer path.
impl SimdOps<i32> for ScalarTok {
    type Reg = i32;
    const LANES: usize = 1;
    const ALIGN: usize = core::mem::align_of::<i32>();

    #[inline(always)]
    unsafe fn zero(self) -> i32 {
        0
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> i32 {
        v
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> i32 {
        unsafe { *p }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> i32 {
        unsafe { *p }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: i32) {
        unsafe { *p = v }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: i32) {
        unsafe { *p = v }
    }
    #[inline(always)]
    unsafe fn mul(self, a: i32, b: i32) -> i32 {
        a.wrapping_mul(b)
    }
    #[inline(always)]
    unsafe fn add(self, a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: i32, b: i32, c: i32) -> i32 {
        a.wrapping_mul(b).wrapping_add(c)
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: i32) -> i32 {
        v
    }
}

impl KernelSimd<i8, i8, i32, i32> for ScalarTok {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> i32 {
        unsafe { *p as i32 }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> i32 {
        v as i32
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> i32 {
        unsafe { *p }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: i32) {
        unsafe { *p = v }
    }
}

impl KernelSimd<bf16, bf16, f32, bf16> for ScalarTok {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> f32 {
        unsafe { (*p).widen() }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> f32 {
        v.widen()
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> f32 {
        unsafe { (*p).widen() }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: f32) {
        unsafe { *p = bf16::narrow(v) }
    }
}
