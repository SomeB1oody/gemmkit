//! # gemmkit-faer
//!
//! A thin [`faer`] adapter over the [`gemmkit`] GEMM engine. Every entry point takes faer's view
//! types ([`MatRef<'_, T>`](faer::MatRef) for inputs, [`MatMut<'_, T>`](faer::MatMut) for the
//! output). Each entry pulls the pointer and the element-unit `isize` row and column strides from
//! the view, then forwards them to gemmkit's raw engine. This avoids assuming a packed
//! column-major layout, so faer's transposed views, sub-matrices, and reversed (negative-stride)
//! views all work without copying
//!
//! ```
//! use faer::Mat;
//! let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
//! let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
//! let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
//! assert_eq!(c[(0, 0)], 19.0);
//! assert_eq!(c[(1, 1)], 50.0);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`GemmScalar`]: `f32`/`f64` unconditionally,
//! plus `f16`/`bf16` under the `half` feature. [`prepack_rhs`]/[`prepack_lhs`] pack the reused
//! operand once for a fixed-weight loop, consumed by repeated
//! [`gemm_packed_b`]/[`gemm_packed_a`] calls
//!
//! Complex products (`Complex<f32>`/`Complex<f64>`, that is faer's `c32`/`c64`, with optional
//! conjugation) need the separate [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] entries under the
//! `complex` feature. The conjugation flags do not fit the homogeneous signature. The integer
//! path (`i8` inputs into an `i32` output) likewise has its own
//! [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature
//!
//! Under the `epilogue` feature, [`gemm_fused`]/[`gemm_fused_with`] fuse an optional [`Bias`] and
//! [`Activation`] into the store in 1 pass (`C <- act(alpha*A*B + beta*C + bias)`). The
//! prepacked-operand twins [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] reuse the same [`PackedRhs`]/[`PackedLhs`]
//! handle as the plain packed entries. `f16`/`bf16` use the same generic entries when `half` is
//! also on. [`gemm_map`]/[`gemm_map_with`] instead run an arbitrary `f32`/`f64` per-element
//! closure, for transforms a bias or activation cannot express
//!
//! Combining `int8` with `epilogue` adds a requantized output.
//! [`gemm_i8_requant`]/[`gemm_i8_requant_with`] and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`] take a [`Requantize`] and fuse the requantize
//! into a quantized `i8` or `u8` output. Combining `complex` with `epilogue` adds the bias-only
//! [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`]. It takes no activation parameter, because ordering
//! is undefined on complex numbers. Every one of these still reads raw parts out of the view and
//! forwards to gemmkit's raw engine. Transposed, sub-matrix, and reversed views keep working
//! without copying
//!
//! faer has no rank-3 array or batch type. [`gemm_batched`] instead takes the batch as a slice of
//! per-element `(A, B)` [`MatRef`] input pairs, matched positionally with a slice of `&mut C`
//! [`MatMut`] outputs. It runs over gemmkit's pointer-array
//! [`gemmkit::gemm_batched_ptr_unchecked`] engine, rather than the 3-D strided form the ndarray
//! adapter uses. gemmkit's core also has a shared bias/activation `gemm_batched_fused` for its own
//! strided batch type. No pointer-array version of it exists, so this crate has no fused entry for
//! batched GEMM

#![cfg_attr(docsrs, feature(doc_cfg))]

// `MatRef`/`MatMut` unqualified in this crate are always faer's types, imported here. gemmkit
// defines its own type of the same name for its checked slice API, but never imports it here
use faer::{Mat, MatMut, MatRef};
/// The element-type bound of the complex entries. Re-exported so a wrapper generic over
/// [`gemm_cplx`] can name the bound without depending on `gemmkit` directly
#[cfg(feature = "complex")]
pub use gemmkit::ComplexScalar;
/// The element-type bound of the plain real entries. Re-exported so a wrapper generic over
/// [`gemm`]/[`dot`]/[`prepack_rhs`] can name the bound without depending on `gemmkit` directly
pub use gemmkit::GemmScalar;
/// gemmkit's heuristic thresholds: the `tuning::set_*` setters, their getters, and the compiled
/// defaults. Re-exported so callers need not depend on `gemmkit` directly. The knobs are
/// process-global atomics, so setting them through a separately resolved 2nd `gemmkit` sets a
/// copy this adapter never reads
#[doc(no_inline)]
pub use gemmkit::tuning;
/// The bias and activation selectors taken by [`gemm_fused`] and its packed twins. The complex
/// twin takes only `Bias`. Re-exported so callers need not depend on `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
/// The complex element type of [`gemm_cplx`] and its `c32`/`c64` aliases. Re-exported so a caller
/// need not reach for a 2nd source of them. They are the same `num_complex` types as faer's own
/// `c32`/`c64`
#[cfg(feature = "complex")]
#[doc(no_inline)]
pub use gemmkit::{Complex, c32, c64};
/// The element-type bounds of the fused entries: [`FusedScalar`] for the bias/activation form,
/// [`MapScalar`] for [`gemm_map`]. Re-exported so a wrapper generic over them can name the bound
/// without depending on `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{FusedScalar, MapScalar};
use gemmkit::{
    GemmProblem, gemm_batched_ptr_unchecked, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_b_unchecked, gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with,
    prepack_lhs_unchecked, prepack_rhs_unchecked,
};
/// The handles produced by [`prepack_rhs`]/[`prepack_lhs`] and consumed by the `gemm_packed_*`
/// entries. Re-exported so callers need not depend on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
/// The [`Parallelism`] selector taken by every entry, and the reusable [`Workspace`] taken by the
/// `_with` variants. Re-exported so callers need not depend on `gemmkit` directly
pub use gemmkit::{Parallelism, Workspace};
/// The requantization parameters [`Requantize`] and its per-tensor or per-row output scale
/// [`RequantScale`], taken by [`gemm_i8_requant`] and [`gemm_i8_requant_u8`]. Re-exported so
/// callers need not depend on `gemmkit` directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
/// The narrow float element types of the `half`-gated entries. Re-exported so callers need not
/// depend on `half` directly, since [`faer`] exposes neither. Both accumulate in `f32`
#[cfg(feature = "half")]
#[doc(no_inline)]
pub use gemmkit::{bf16, f16};
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
#[cfg(feature = "complex")]
use gemmkit::{gemm_cplx_unchecked, gemm_cplx_unchecked_with};
#[cfg(feature = "epilogue")]
use gemmkit::{
    gemm_fused_unchecked, gemm_fused_unchecked_with, gemm_map_unchecked, gemm_map_unchecked_with,
    gemm_packed_a_fused_unchecked, gemm_packed_a_fused_unchecked_with,
    gemm_packed_b_fused_unchecked, gemm_packed_b_fused_unchecked_with,
};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::{
    gemm_i8_requant_u8_unchecked, gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with,
};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};

// batched GEMM over a slice of per-element (A, B) / C view triples
mod batched;
// shared view-adapter helpers used by every other module in this crate
mod common;
// complex GEMM entries with optional per-operand conjugation
#[cfg(feature = "complex")]
mod cplx;
// homogeneous GEMM entries (f32/f64, plus f16/bf16 under half)
mod float;
// fused bias/activation GEMM entries
#[cfg(feature = "epilogue")]
mod fused;
// i8-input GEMM entries, plain (i32 output) and requantizing (i8/u8 output)
#[cfg(feature = "int8")]
mod int8;
// user-defined per-element map-epilogue GEMM entries
#[cfg(feature = "epilogue")]
mod map;
// prepacked-operand (PackedLhs/PackedRhs) GEMM entries
mod packed;

pub use batched::gemm_batched;
#[cfg(feature = "complex")]
pub use cplx::{dot_cplx, gemm_cplx, gemm_cplx_with};
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub use cplx::{gemm_cplx_fused, gemm_cplx_fused_with};
pub use float::{dot, gemm, gemm_with};
#[cfg(feature = "epilogue")]
pub use fused::{gemm_fused, gemm_fused_with};
#[cfg(feature = "int8")]
pub use int8::{dot_i8, gemm_i8, gemm_i8_with};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use int8::{
    gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_with, gemm_i8_requant_with,
};
#[cfg(feature = "epilogue")]
pub use map::{gemm_map, gemm_map_with};
pub use packed::{
    gemm_packed_a, gemm_packed_a_with, gemm_packed_b, gemm_packed_b_with, prepack_lhs, prepack_rhs,
};
#[cfg(feature = "epilogue")]
pub use packed::{
    gemm_packed_a_fused, gemm_packed_a_fused_with, gemm_packed_b_fused, gemm_packed_b_fused_with,
};
