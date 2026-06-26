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

pub use api::{MatMut, MatRef, gemm, gemm_unchecked, gemm_unchecked_with, gemm_with};
pub use dispatch::GemmScalar;
pub use parallel::Parallelism;
pub use scalar::{Float, Scalar};
pub use workspace::Workspace;

#[doc(no_inline)]
pub use cache::{CacheTopology, topology};
