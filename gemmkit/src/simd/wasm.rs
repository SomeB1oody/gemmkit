//! WebAssembly `simd128` ISA token: `f32` (4 lanes) and `f64` (2 lanes) over the same
//! 128-bit `v128` register
//!
//! This module is compiled on any `wasm32` build (see the `cfg` on the `mod wasm`
//! declaration), but `simd128` itself is a **compile-time** target feature: a wasm
//! module either was built with `-C target-feature=+simd128` or was not, and there is
//! no CPUID-style runtime probe to fall back on. The dispatch ladder therefore only
//! ever reaches [`Simd128`] under a matching `cfg(target_feature = "simd128")`, the
//! same compile-time-only story as the AArch64 NEON token
//!
//! Core `simd128` has **no hardware FMA**, so [`SimdOps::mul_add`] is the 2-rounding
//! `add(mul(a, b), c)`, matching `ScalarTok` and `Float::mul_add` exactly and keeping this
//! path reproducible against the scalar reference (the crate's determinism contract, not
//! bitwise but reproducible under a fixed config). For the same reason
//! [`SimdOps::LANE_FMA`] stays at its default `false` and [`SimdOps::accumulate_tile`]
//! is not overridden here. The `relaxed-simd` proposal's fused ops
//! (`f32x4_relaxed_madd` / `f64x2_relaxed_madd`, a separate target feature) are
//! explicitly allowed by spec to round with either 1 or 2 roundings depending on the
//! engine, so they are never used: that nondeterminism would break the reproducibility
//! this module is built to preserve
//!
//! `core::arch::wasm32::v128` is untyped: `SimdOps<f32>` and `SimdOps<f64>` both set
//! `Reg = v128`, and only the intrinsic family (`f32x4_*` vs `f64x2_*`) picks the lane
//! width, so a `f32x4_*` op called from the `f64` impl still type-checks; mind the
//! family in every method. `v128_load`/`v128_store` make no aligned/unaligned
//! distinction, as on NEON

use core::arch::wasm32::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;

/// WebAssembly `simd128` ISA token
#[derive(Copy, Clone, Default)]
pub struct Simd128;

impl Simd for Simd128 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // Establishes the `simd128`-enabled codegen context that the `#[inline(always)]`
        // intrinsic wrappers below fold into, the same trampoline shape every other
        // vector ISA token uses. Dispatch only ever calls this where the build already has
        // `target_feature = "simd128"`, so the `enable` here does not gate reachability
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
        unsafe { v128_load(p as *const v128) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
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
        // No hardware FMA on this target: 2 separate roundings, matching the scalar
        // `Float::mul_add` reference bit-for-bit
        f32x4_add(f32x4_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `c - a*b`, 2 roundings. Only the SoA complex path calls this; plain f32/f64
        // GEMM never does
        f32x4_sub(c, f32x4_mul(a, b))
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // f32x4_pmax(a, b) is `a < b ? b : a`, and an unordered compare (NaN in either
        // side) makes `a < b` false, so it takes the `a` branch: a NaN `a` returns NaN
        // here, not `b`. NOT f32x4_max, which propagates NaN from either operand
        f32x4_pmax(a, b)
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // f32x4_pmin(a, b) is `b < a ? b : a`; same unordered-takes-`a` shape as pmax
        f32x4_pmin(a, b)
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        // Fixed lane order keeps this reproducible; used by the gemv and small_mn dot paths
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
        // 2 roundings, no hardware FMA; see the f32 impl
        f64x2_add(f64x2_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // Only the SoA complex path calls this; see the f32 impl
        f64x2_sub(c, f64x2_mul(a, b))
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // f64x2_pmax(a, b) is `a < b ? b : a`: a NaN `a` returns NaN, not `b`; see the
        // f32 impl
        f64x2_pmax(a, b)
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // f64x2_pmin(a, b) is `b < a ? b : a`; see the f32 impl
        f64x2_pmin(a, b)
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        // Fixed lane order keeps this reproducible
        f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v)
    }
}

// Mixed precision: f16 / bf16 inputs, f32 accumulator (4-wide f32x4 v128)
//
// simd128 has no half-precision register type or widen/narrow instruction, so, as on
// NEON, the widen on load and narrow on store go through NarrowFloat one lane at a time
// into a plain [f32; 4] buffer, then a single v128_load/v128_store moves the whole
// buffer. Everything past that point is the ordinary SimdOps<f32> impl above (the same
// 2-rounding mul_add), so accuracy tracks the scalar reference

/// `f16` mixed precision: per-lane scalar widen/narrow through [`NarrowFloat`] since
/// simd128 has no fp16 conversion instruction
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
        // Qualified: the f32-output twin (simd.rs) adds a 2nd KernelSimd<f16, .., f32, ..>
        // impl for this token, so the plain method name alone would be ambiguous
        unsafe { <Self as KernelSimd<f16, f16, f32, f16>>::load_lhs(self, p) }
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

/// `bf16` mixed precision, mirror of the `f16` impl above (same per-lane scalar
/// widen/narrow through [`NarrowFloat`])
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
        // Qualified: the f32-output twin (simd.rs) adds a 2nd KernelSimd<bf16, .., f32, ..>
        // impl for this token, so the plain method name alone would be ambiguous
        unsafe { <Self as KernelSimd<bf16, bf16, f32, bf16>>::load_lhs(self, p) }
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

// Integer: i8 inputs, i32 accumulator (4-wide i32x4 v128)
//
// The i32 ops below are native i32x4_* (wrapping two's complement, matching the scalar
// widen path bit-for-bit); the i8 -> i32 widen on load is per-lane scalar, 1 byte per
// LANES slot. simd128 has no packed dot-product instruction (no vpdpbusd/SDOT
// analogue), so i8 GEMM runs through the widen-and-multiply crate::kernel::IntGemm
// family and never reaches the dot_accumulate seam

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
        // i32 mul and add each wrap on overflow; no integer FMA to fuse them into
        i32x4_add(i32x4_mul(a, b), c)
    }
    #[inline(always)]
    unsafe fn fnma(self, a: v128, b: v128, c: v128) -> v128 {
        // Present only to satisfy the trait; the i8 kernel never calls this
        i32x4_sub(c, i32x4_mul(a, b))
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: v128) -> i32 {
        // Fixed lane order keeps this reproducible; used only by small_mn's int path
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

// Complex (simd128): impl_complex_simd! reuses Reg = v128 and LANES = the real lane
// count (4 for f32, 2 for f64) from the SimdOps<f32>/<f64> impls above. Complex GEMM
// routes entirely through the shared SoA soa_microkernel, which calls back into this
// token's own mul_add/fnma, so the complex path is already vectorized without any code
// here: no separate complex-multiply instruction to reach for or avoid, unlike NEON's
// FCMLA, which is nightly-gated on stable Rust and would also fuse the cross-terms into
// a single rounding, breaking the full-vs-edge rounding identity this kernel relies on
#[cfg(feature = "complex")]
impl_complex_simd!(Simd128, f32, v128, 4);
#[cfg(feature = "complex")]
impl_complex_simd!(Simd128, f64, v128, 2);
