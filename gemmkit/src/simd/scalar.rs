//! Scalar fallback ISA token: `LANES == 1`, no intrinsics, works everywhere.
//!
//! This is the portability floor and the Miri-checkable reference path. It is
//! also what makes the `SimdOps` abstraction honest: the exact same generic
//! kernel that drives AVX-512 drives this token with a one-element "register".

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

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

            #[inline(always)]
            unsafe fn zero(self) -> Self::Reg {
                0.0
            }
            #[inline(always)]
            unsafe fn splat(self, v: $t) -> Self::Reg {
                v
            }
            #[inline(always)]
            unsafe fn loadu(self, p: *const $t) -> Self::Reg {
                unsafe { *p }
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
                // Plain `a*b + c` keeps the scalar path reproducible and in
                // agreement with `Float::mul_add` used by the epilogue.
                a * b + c
            }
            #[inline(always)]
            unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
                // Plain `c - a*b` (the scalar reference for the SoA complex kernel's
                // `acc_re -= a_im·b_im` step).
                c - a * b
            }
            #[inline(always)]
            unsafe fn reduce_sum(self, v: Self::Reg) -> $t {
                v
            }
            #[inline(always)]
            unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                // `NaN > b` is false, so a `NaN` `a` returns `b` (the contract).
                if a > b { a } else { b }
            }
            #[inline(always)]
            unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
                if a < b { a } else { b }
            }
        }
    };
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);

// Mixed-precision (scalar fallback): `f16`/`bf16` widen to `f32` on load and round
// back on store, one element at a time (`Reg` is a bare `f32`). Miri-checked
// reference for the narrow types.
#[cfg(feature = "half")]
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
// widen-load, one element at a time. Wrapping arithmetic gives conventional
// integer overflow semantics. Miri-checked reference for the integer path.
#[cfg(feature = "int8")]
impl SimdOps<i32> for ScalarTok {
    type Reg = i32;
    const LANES: usize = 1;

    #[inline(always)]
    unsafe fn zero(self) -> i32 {
        0
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> i32 {
        v
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> i32 {
        unsafe { *p }
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
    unsafe fn fnma(self, a: i32, b: i32, c: i32) -> i32 {
        c.wrapping_sub(a.wrapping_mul(b))
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: i32) -> i32 {
        v
    }
}

#[cfg(feature = "int8")]
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

#[cfg(feature = "half")]
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

// Complex (scalar fallback): the Miri-checked SoA reference. `LANES = 1`, the real `Reg`
// is the scalar itself; complex GEMM routes through the shared `soa_microkernel`.
#[cfg(feature = "complex")]
impl_complex_simd!(ScalarTok, f32, f32, 1);
#[cfg(feature = "complex")]
impl_complex_simd!(ScalarTok, f64, f64, 1);
