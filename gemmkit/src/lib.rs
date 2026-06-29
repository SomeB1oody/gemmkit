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
//! * `std` (default) — runtime cache/feature detection and the thread-local
//!   workspace pool.
//! * `parallel` (default) — rayon multithreading. With it off everything still
//!   compiles and runs single-threaded.
//! * `complex` — complex GEMM over `c32`/`c64` with optional conjugation
//!   ([`gemm_cplx`]); pulls in `num-complex`.
//! * `half` — `f16`/`bf16` mixed-precision GEMM (`f32` accumulate); pulls in `half`.
//! * `int8` — `i8 -> i32` integer GEMM ([`gemm_i8`]); no extra dependency.
//!
//! The three element-type families are **off by default** — a plain `f32`/`f64`
//! build pays for none of their codegen or dependencies. Enable the ones you need.

#![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)] // safety documented at the module / contract level

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

pub use api::{
    MatMut, MatRef, PackedLhs, PackedRhs, gemm, gemm_packed_a, gemm_packed_a_with, gemm_packed_b,
    gemm_packed_b_with, gemm_unchecked, gemm_unchecked_with, gemm_with, prepack_lhs, prepack_rhs,
};
#[cfg(feature = "complex")]
pub use api::{gemm_cplx, gemm_cplx_unchecked, gemm_cplx_unchecked_with, gemm_cplx_with};
#[cfg(feature = "int8")]
pub use api::{gemm_i8, gemm_i8_unchecked, gemm_i8_with};
#[cfg(feature = "complex")]
pub use dispatch::ComplexScalar;
pub use dispatch::GemmScalar;
pub use parallel::Parallelism;
#[cfg(feature = "complex")]
pub use scalar::Conjugate;
#[cfg(feature = "half")]
pub use scalar::NarrowFloat;
pub use scalar::{Float, Scalar};
pub use workspace::Workspace;

#[doc(no_inline)]
pub use cache::{CacheTopology, Machine, topology};

/// Re-exported [`half`] narrow float types (`f16`, `bf16`), so callers can run a
/// mixed-precision GEMM without depending on `half` directly. They accumulate in
/// `f32` (their [`Scalar::Acc`]). Requires the `half` feature.
#[cfg(feature = "half")]
#[doc(no_inline)]
pub use half::{bf16, f16};

/// Re-exported [`num_complex::Complex`] (and the `c32`/`c64` aliases), so callers can
/// run a complex GEMM (via [`gemm_cplx`]) without depending on `num-complex` directly.
/// Requires the `complex` feature.
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
