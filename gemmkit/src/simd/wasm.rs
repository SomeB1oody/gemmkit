//! WebAssembly `simd128` ISA token (128-bit): `f32` (4 lanes) and `f64` (2 lanes).
//!
//! This is gemmkit's portability/deployment backend, not a peak-FLOPs target.
//! `simd128` is a **compile-time** target feature (wasm has no runtime feature
//! detection), so — like the AArch64 NEON arm — this token exists only under
//! `cfg(target_arch = "wasm32")` and is selected by `cfg`, never by a runtime probe.
//!
//! There is **no hardware FMA** in core simd128, so [`SimdOps::mul_add`] is the
//! two-rounding `add(mul(a, b), c)` — exactly what `ScalarTok`/`Float::mul_add`
//! compute, so the simd128 path stays *reproducible* against the scalar reference
//! and the serial/parallel runs agree (the crate's determinism contract). For the
//! same reason [`SimdOps::LANE_FMA`] is left `false` and [`SimdOps::accumulate_tile`]
//! is **not** overridden, and the spec-nondeterministic relaxed-SIMD ops
//! (`f32x4_relaxed_madd` / `f64x2_relaxed_madd`) are deliberately never used.
//!
//! `core::arch::wasm32::v128` is an untyped 128-bit register shared by both widths;
//! the lane width comes only from the intrinsic family (`f32x4_*` vs `f64x2_*`), so
//! the type checker will *not* catch a `f32x4_*` op used in the `f64` impl — mind the
//! family in each method. wasm loads/stores are alignment-agnostic (as on NEON).

use core::arch::wasm32::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

/// WebAssembly `simd128` ISA token.
#[derive(Copy, Clone, Default)]
pub struct Simd128;

impl Simd for Simd128 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // `simd128` is a compile-time feature: when the build enables it
        // (`-C target-feature=+simd128`), this just establishes the codegen
        // context the `#[inline(always)]` intrinsic wrappers fold into (mirror of
        // the NEON trampoline). When it is *not* enabled the selector never picks
        // this token, so this is unreachable at runtime.
        #[target_feature(enable = "simd128")]
        fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        inner(f)
    }
}

impl SimdOps<f32> for Simd128 {
    type Reg = v128;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        f32x4_splat(v)
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        // wasm `v128_load` makes no aligned/unaligned distinction.
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        // wasm `v128_store` makes no aligned/unaligned distinction.
        unsafe { v128_store(p as *mut v128, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        f32x4_mul(a, b)
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        f32x4_add(a, b)
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // No hardware FMA: two roundings `(a*b) + c`, matching `Float::mul_add` /
        // `ScalarTok`. NOT fused, NOT relaxed-SIMD (which would be nondeterministic).
        f32x4_add(f32x4_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `c - a*b`, two roundings (the subtractive partner of `mul_add`). Required by
        // the trait for the SoA complex path; plain f32/f64 GEMM never calls it.
        f32x4_sub(c, f32x4_mul(a, b))
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // `f32x4_pmax(a, b)` is `b < a ? a : b`: a `NaN` `a` makes `b < a` false, so it
        // returns `b` — the `max`/`min` NaN-in-`a` contract. NOT `f32x4_max`, which
        // propagates NaN and would break the fast-vs-scalar epilogue agreement.
        f32x4_pmax(a, b)
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // `f32x4_pmin(a, b)` is `a < b ? a : b`: NaN `a` -> `b`.
        f32x4_pmin(a, b)
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        // Fixed lane order (0,1,2,3) → reproducible. Used only by the gemv path.
        f32x4_extract_lane::<0>(v)
            + f32x4_extract_lane::<1>(v)
            + f32x4_extract_lane::<2>(v)
            + f32x4_extract_lane::<3>(v)
    }
}

impl SimdOps<f64> for Simd128 {
    type Reg = v128;
    const LANES: usize = 2;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        f64x2_splat(0.0)
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        f64x2_splat(v)
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        unsafe { v128_store(p as *mut v128, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        f64x2_mul(a, b)
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        f64x2_add(a, b)
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // Two roundings `(a*b) + c`; see the `f32` impl.
        f64x2_add(f64x2_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        f64x2_sub(c, f64x2_mul(a, b))
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // `f64x2_pmax(a, b)` is `b < a ? a : b` (NaN `a` -> `b`); see the `f32` impl.
        f64x2_pmax(a, b)
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        f64x2_pmin(a, b)
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v)
    }
}

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (4-wide f32x4 v128) ----
//
// wasm `simd128` has no half-precision type or conversion intrinsic, so (like NEON) the widen
// load / narrow store are per-lane scalar through [`NarrowFloat`], assembling one `f32x4`. The
// accumulator is the real [`SimdOps<f32>`] above, so the inner loop is the ordinary two-rounding
// `mul_add`, reproducible against the scalar reference.

/// `f16` mixed precision: per-lane scalar widen/narrow through [`NarrowFloat`] (no native
/// wasm fp16).
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Simd128 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> v128 {
        unsafe {
            let a = [
                (*p).widen(),
                (*p.add(1)).widen(),
                (*p.add(2)).widen(),
                (*p.add(3)).widen(),
            ];
            v128_load(a.as_ptr() as *const v128)
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> v128 {
        f32x4_splat(v.widen())
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> v128 {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: v128) {
        unsafe {
            let mut t = [0.0f32; 4];
            v128_store(t.as_mut_ptr() as *mut v128, v);
            for (i, &x) in t.iter().enumerate() {
                *p.add(i) = f16::narrow(x);
            }
        }
    }
}

/// `bf16` mixed precision (scalar widen/narrow), mirror of the `f16` impl above.
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Simd128 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> v128 {
        unsafe {
            let a = [
                (*p).widen(),
                (*p.add(1)).widen(),
                (*p.add(2)).widen(),
                (*p.add(3)).widen(),
            ];
            v128_load(a.as_ptr() as *const v128)
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> v128 {
        f32x4_splat(v.widen())
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> v128 {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: v128) {
        unsafe {
            let mut t = [0.0f32; 4];
            v128_store(t.as_mut_ptr() as *mut v128, v);
            for (i, &x) in t.iter().enumerate() {
                *p.add(i) = bf16::narrow(x);
            }
        }
    }
}

// ---- integer: i8 inputs, i32 accumulator (4-wide i32x4 v128) ----
//
// The `i32` ops are native `i32x4_*` (wrapping two's complement, bit-exact with the scalar
// widen path); the `i8 -> i32` widen-load is per-lane scalar (reads exactly the 4 panel-slot
// bytes). wasm has no `vpdpbusd`/`SDOT` analogue, so `i8` rides the widen-and-multiply
// [`crate::kernel::IntGemm`] family, never the `dot_accumulate` seam.

#[cfg(feature = "int8")]
impl SimdOps<i32> for Simd128 {
    type Reg = v128;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn zero(self) -> v128 {
        i32x4_splat(0)
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> v128 {
        i32x4_splat(v)
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> v128 {
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: v128) {
        unsafe { v128_store(p as *mut v128, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }
    #[inline(always)]
    unsafe fn add(self, a: v128, b: v128) -> v128 {
        i32x4_add(a, b)
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {
        // Wrapping i32 `a*b + c` (no integer FMA; both ops wrap two's complement).
        i32x4_add(i32x4_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: v128, b: v128, c: v128) -> v128 {
        // `c - a*b` (wrapping i32). Present only to satisfy the trait; the integer
        // kernel never calls it.
        i32x4_sub(c, i32x4_mul(a, b))
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: v128) -> i32 {
        // Fixed lane order (wrapping i32 add). Used only by a gemv epilogue.
        i32x4_extract_lane::<0>(v)
            .wrapping_add(i32x4_extract_lane::<1>(v))
            .wrapping_add(i32x4_extract_lane::<2>(v))
            .wrapping_add(i32x4_extract_lane::<3>(v))
    }
}

#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Simd128 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> v128 {
        unsafe {
            let a = [
                *p as i32,
                *p.add(1) as i32,
                *p.add(2) as i32,
                *p.add(3) as i32,
            ];
            v128_load(a.as_ptr() as *const v128)
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> v128 {
        i32x4_splat(v as i32)
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> v128 {
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: v128) {
        unsafe { v128_store(p as *mut v128, v) }
    }
}

// Complex (simd128): real `Reg` = `v128`, `LANES` the real lane count (4 / 2). Complex GEMM
// routes through the shared SoA `soa_microkernel`, so it's vectorized for free — `mul_add`/`fnma`
// lower to the two-rounding `f32x4`/`f64x2` ops above (no FMA, reproducible). The relaxed-SIMD
// `*_relaxed_madd` ops are never used: they would fuse the cross-terms into one rounding and
// break the full-vs-edge identity (the same reason NEON avoids `FCMLA`).
#[cfg(feature = "complex")]
impl_complex_simd!(Simd128, f32, v128, 4);
#[cfg(feature = "complex")]
impl_complex_simd!(Simd128, f64, v128, 2);
