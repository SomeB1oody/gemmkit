//! # gemmkit
//!
//! A GEMM (general matrix multiply) engine with no ndarray dependency. It computes
//! `C <- alpha*A*B + beta*C` over data-type-agnostic `&[T]` + stride views (or raw
//! pointers), picking the fastest instruction set the running CPU supports
//!
//! ## Quick start
//!
//! ```
//! use gemmkit::{gemm, MatRef, MatMut, Parallelism};
//!
//! // 2x3 * 3x2 = 2x2, row-major
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
//! ## Architecture in brief
//!
//! Every axis of variation lives behind a trait: the instruction set is
//! [`simd::SimdOps`], the element type is [`scalar::Scalar`], the operation family
//! is [`kernel::KernelFamily`]. One generic 5-loop [`driver`] and one generic
//! microkernel cover every `(type, ISA, tile)` combination with no macros and no
//! `transmute`. See `ARCHITECTURE.md` for the full tour
//!
//! ## Features
//!
//! * `std` (default): runtime cache/CPU-feature detection, the `GEMMKIT_REQUIRE_ISA`
//!   and tuning env knobs, and the thread-local workspace pool. With `std` off the
//!   crate is `#![no_std]`, needing only `core` and `alloc`
//! * `parallel` (default, implies `std`): rayon multithreading. With it off the
//!   crate still compiles and runs, single-threaded
//! * `complex`: complex GEMM over `c32`/`c64` with optional conjugation
//!   ([`gemm_cplx`]); pulls in `num-complex`
//! * `half`: `f16`/`bf16` mixed-precision GEMM, accumulating in `f32`; pulls in `half`
//! * `int8`: `i8 -> i32` integer GEMM ([`gemm_i8`]); no extra dependency
//! * `epilogue`: fused epilogues, bias/activation ([`gemm_fused`], `gemm_batched_fused*`,
//!   the prepacked `gemm_packed_a_fused*` / `gemm_packed_b_fused*`, and, with `complex`,
//!   `gemm_cplx_fused*`); a user-defined per-element closure ([`gemm_map`], `f32`/`f64`
//!   only); and, with `int8`, requantized `i8`/`u8` output (`gemm_i8_requant*`); no
//!   extra dependency
//!
//! `complex`, `half`, and `int8` are off by default, so a plain `f32`/`f64` build pays
//! for none of their codegen or dependencies. `epilogue` is off by default too, so a
//! plain-GEMM build pays for none of its codegen

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)] // documented at the module/contract level, not per fn

// Some modules (batched/packed API buffers) need heap `Vec`s even in a no_std
// build, so pull in `alloc` unconditionally rather than gating this on `std`
extern crate alloc;

// Cache topology detection and BLIS-model analytical blocking (layer L3)
pub mod cache;
// Generic 5-loop GEMM driver, generic over the kernel family and the ISA (layer L5)
pub mod driver;
// Kernel families: the operation-family seam (layer L4)
pub mod kernel;
// Numeric scalar abstraction: identity constants and the accumulator type (layer L0)
pub mod scalar;
// SIMD abstraction: ISA tokens and per-type vector ops (layer L0)
pub mod simd;
// Tuning surface: heuristic thresholds with env-var and setter overrides
pub mod tuning;

// Public API: safe slice/stride entry points plus the raw unchecked engine (layer L8a)
mod api;
// Shared validation/lowering surface for the view adapters (support for L8a, not a layer of
// its own): the pointer-level bias/requant checks the raw-pointer adapters and the checked
// core entries both consume. doc(hidden) = not part of the documented API, versioned in
// lockstep with the adapters
#[doc(hidden)]
pub mod adapter;
// Runtime ISA dispatch, memoized per element type (layer L7)
mod dispatch;
// Packing primitives shared by every kernel family (layer L1)
mod pack;
// Parallelism control and job splitting across workers (layer L2)
mod parallel;
// Special-case paths that bypass the driver for shapes it fits poorly (layer L6)
mod special;
// Packing-buffer workspace: a thread-local pool plus explicit reuse
mod workspace;

#[cfg(feature = "epilogue")]
pub use api::{
    Activation, Bias, gemm_batched_fused, gemm_batched_fused_unchecked,
    gemm_batched_fused_unchecked_with, gemm_batched_fused_with, gemm_fused, gemm_fused_unchecked,
    gemm_fused_unchecked_with, gemm_fused_with, gemm_map, gemm_map_unchecked,
    gemm_map_unchecked_with, gemm_map_with, gemm_packed_a_fused, gemm_packed_a_fused_unchecked,
    gemm_packed_a_fused_unchecked_with, gemm_packed_a_fused_with, gemm_packed_b_fused,
    gemm_packed_b_fused_unchecked, gemm_packed_b_fused_unchecked_with, gemm_packed_b_fused_with,
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
    RequantScale, Requantize, gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_unchecked,
    gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_u8_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with, gemm_i8_requant_with,
};
#[cfg(feature = "complex")]
pub use api::{gemm_cplx, gemm_cplx_unchecked, gemm_cplx_unchecked_with, gemm_cplx_with};
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub use api::{
    gemm_cplx_fused, gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with,
    gemm_cplx_fused_with,
};
#[cfg(feature = "int8")]
pub use api::{
    gemm_i8, gemm_i8_packed_b, gemm_i8_packed_b_unchecked, gemm_i8_packed_b_unchecked_with,
    gemm_i8_packed_b_with, gemm_i8_unchecked, gemm_i8_unchecked_with, gemm_i8_with, prepack_rhs_i8,
    prepack_rhs_i8_unchecked,
};
#[cfg(feature = "complex")]
pub use dispatch::ComplexScalar;
pub use dispatch::GemmProblem;
pub use dispatch::GemmScalar;
#[cfg(feature = "epilogue")]
pub use dispatch::{FusedScalar, MapScalar};
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

/// Re-exported [`half`] narrow float types (`f16`, `bf16`), so you do not need to
/// depend on `half` directly to call the `half`-gated GEMM entry points. Both
/// accumulate in `f32` (their [`Scalar::Acc`])
#[cfg(feature = "half")]
#[doc(no_inline)]
pub use half::{bf16, f16};

/// Re-exported [`num_complex::Complex`] (and the `c32`/`c64` aliases below), so you
/// do not need to depend on `num-complex` directly. The element type for [`gemm_cplx`]
#[cfg(feature = "complex")]
#[doc(no_inline)]
pub use num_complex::Complex;
/// `Complex<f32>`: the single-precision complex element type
#[cfg(feature = "complex")]
#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex<f32>;
/// `Complex<f64>`: the double-precision complex element type
#[cfg(feature = "complex")]
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex<f64>;
