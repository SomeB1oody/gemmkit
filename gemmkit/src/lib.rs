//! # gemmkit
//!
//! A clean, extensible, high-performance GEMM (general matrix multiply) engine
//! with **zero ndarray dependency**. It computes `C <- alpha·A·B + beta·C` over
//! data-type-agnostic `&[T]` + stride views (or raw pointers), choosing the best
//! available x86 instruction set at runtime.
//!
//! ## Quick start
//!
//! ```
//! use gemmkit::{gemm, MatRef, MatMut, Parallelism};
//!
//! // 2x3 · 3x2 = 2x2, all row-major.
//! let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
//! let mut c = [0.0_f32; 4];
//! gemm(
//!     1.0,
//!     MatRef::from_row_major(&a, 2, 3),
//!     MatRef::from_row_major(&b, 3, 2),
//!     0.0,
//!     MatMut::from_row_major(&mut c, 2, 2),
//!     Parallelism::Serial,
//! );
//! assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
//! ```
//!
//! ## Architecture in one paragraph
//!
//! The variation points are isolated to traits: ISA → [`simd::SimdOps`], element
//! type → [`scalar::Scalar`], operation family → [`kernel::KernelFamily`]. One
//! generic five-loop [`driver`] and one generic float microkernel cover every
//! `(type, ISA, tile)` combination with no macros and no `transmute`. See
//! `ARCHITECTURE.md` for the full tour.
//!
//! ## Features
//!
//! * `std` (default) — runtime cache/CPU-feature detection, the
//!   `GEMMKIT_REQUIRE_ISA` and tuning env knobs, and the thread-local workspace
//!   pool. **With `std` off the crate is `#![no_std]`**, needing only `core` + `alloc`
//! * `parallel` (default, implies `std`) — rayon multithreading. With it off
//!   everything still compiles and runs single-threaded.
//! * `complex` — complex GEMM over `c32`/`c64` with optional conjugation
//!   ([`gemm_cplx`]); pulls in `num-complex`.
//! * `half` — `f16`/`bf16` mixed-precision GEMM (`f32` accumulate); pulls in `half`.
//! * `int8` — `i8 -> i32` integer GEMM ([`gemm_i8`]); no extra dependency.
//! * `epilogue` — fused epilogues: bias/activation ([`gemm_fused`],
//!   `gemm_batched_fused*`, and, with `complex`, `gemm_cplx_fused*`) and, with
//!   `int8`, requantized `i8`/`u8` output (`gemm_i8_requant*`); no extra dependency.
//!
//! These three element-type families are off by default, so a plain `f32`/`f64`
//! build pays for none of their codegen or dependencies. The `epilogue` capability
//! is likewise off by default, so a plain-GEMM build pays for none of its codegen.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)] // safety documented at the module / contract level

// Heap-backed packing needs `alloc` in both builds; keep it unconditional so
// downstream `alloc::` imports resolve under `std` too, with no cfg noise.
extern crate alloc;

pub mod cache;
pub mod driver;
pub mod kernel;
pub mod scalar;
pub mod simd;
pub mod tuning;

mod api;
mod dispatch;
mod pack;
mod parallel;
mod special;
mod workspace;

#[cfg(feature = "epilogue")]
pub use api::{
    Activation, Bias, gemm_batched_fused, gemm_batched_fused_with, gemm_fused,
    gemm_fused_unchecked, gemm_fused_with,
};
pub use api::{
    BatchProblem, MatMut, MatRef, PackedLhs, PackedRhs, gemm, gemm_batched,
    gemm_batched_ptr_unchecked, gemm_batched_slice, gemm_batched_unchecked,
    gemm_batched_unchecked_with, gemm_batched_with, gemm_packed_a, gemm_packed_a_unchecked,
    gemm_packed_a_unchecked_with, gemm_packed_a_with, gemm_packed_b, gemm_packed_b_unchecked,
    gemm_packed_b_unchecked_with, gemm_packed_b_with, gemm_unchecked, gemm_unchecked_with,
    gemm_with, prepack_lhs, prepack_lhs_unchecked, prepack_rhs, prepack_rhs_unchecked,
};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use api::{
    Requantize, gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_unchecked,
    gemm_i8_requant_u8_with, gemm_i8_requant_unchecked, gemm_i8_requant_with,
};
#[cfg(feature = "complex")]
pub use api::{gemm_cplx, gemm_cplx_unchecked, gemm_cplx_unchecked_with, gemm_cplx_with};
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub use api::{gemm_cplx_fused, gemm_cplx_fused_with};
#[cfg(feature = "int8")]
pub use api::{gemm_i8, gemm_i8_unchecked, gemm_i8_unchecked_with, gemm_i8_with};
#[cfg(feature = "complex")]
pub use dispatch::ComplexScalar;
#[cfg(feature = "epilogue")]
pub use dispatch::FusedScalar;
pub use dispatch::GemmProblem;
pub use dispatch::GemmScalar;
#[cfg(feature = "epilogue")]
pub use kernel::epilogue::BiasDim;
pub use parallel::Parallelism;
#[cfg(feature = "complex")]
pub use scalar::ComplexFloat;
#[cfg(feature = "half")]
pub use scalar::NarrowFloat;
pub use scalar::{Float, Scalar};
pub use workspace::Workspace;

#[doc(no_inline)]
pub use cache::{CacheTopology, Machine, topology};

/// Re-exported [`half`] narrow float types (`f16`, `bf16`), so callers need not
/// depend on `half` directly. They accumulate in `f32` (their [`Scalar::Acc`]).
#[cfg(feature = "half")]
#[doc(no_inline)]
pub use half::{bf16, f16};

/// Re-exported [`num_complex::Complex`] (and the `c32`/`c64` aliases), so callers
/// need not depend on `num-complex` directly. Used by [`gemm_cplx`].
#[cfg(feature = "complex")]
#[doc(no_inline)]
pub use num_complex::Complex;
/// `Complex<f32>` — the single-precision complex element type.
#[cfg(feature = "complex")]
#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex<f32>;
/// `Complex<f64>` — the double-precision complex element type.
#[cfg(feature = "complex")]
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex<f64>;
