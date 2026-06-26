//! Scalar fallback ISA token: `LANES == 1`, no intrinsics, works everywhere.
//!
//! This is the portability floor and the Miri-checkable reference path. It is
//! also what makes the `SimdOps` abstraction honest: the exact same generic
//! kernel that drives AVX-512 drives this token with a one-element "register".

use super::{Simd, SimdOps};

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
