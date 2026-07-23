//! # gemmkit-faer
//!
//! A thin [`faer`] adapter over the [`gemmkit`] GEMM engine. Every entry point takes faer's view
//! types ([`MatRef<'_, T>`](faer::MatRef) for inputs, [`MatMut<'_, T>`](faer::MatMut) for the
//! output), pulls the pointer and the element-unit `isize` row/column strides straight out of the
//! view, and forwards them to gemmkit's raw engine. Reading the view this way, instead of assuming a
//! packed column-major layout, means faer's transposed views, sub-matrices, and reversed
//! (negative-stride) views all work without copying
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
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! unconditionally, plus `f16`/`bf16` under the `half` feature. [`prepack_rhs`]/[`prepack_lhs`] pack
//! the reused operand once for a fixed-weight loop, consumed by repeated
//! [`gemm_packed_b`]/[`gemm_packed_a`] calls. Complex products (`Complex<f32>`/`Complex<f64>`,
//! i.e. faer's `c32`/`c64`, with optional conjugation) need the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since the conj flags
//! don't fit the homogeneous signature. The integer path (`i8` inputs into an `i32` output) is
//! likewise its own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature
//!
//! Under the `epilogue` feature, [`gemm_fused`]/[`gemm_fused_with`] fuse an optional [`Bias`] and/or
//! [`Activation`] into the store in 1 pass (`C <- act(alpha*A*B + beta*C + bias)`), with
//! prepacked-operand twins [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] that reuse the same [`PackedRhs`]/[`PackedLhs`]
//! handle the plain packed entries use. `f16`/`bf16` ride the same generic entries when `half` is
//! also on. [`gemm_map`]/[`gemm_map_with`] instead run an arbitrary `f32`/`f64` per-element closure,
//! for transforms a bias/activation can't express. Combining `int8` with `epilogue` adds requantized
//! output: [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse the requantize
//! into a quantized `i8` (resp. `u8`) output. Combining `complex` with `epilogue` adds the bias-only
//! [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`]: there is no activation parameter, since an ordering
//! activation is undefined on complex numbers. Every one of these still reads raw parts out of the
//! view and forwards to gemmkit's raw engine, so transposed, sub-matrix, and reversed views keep
//! working without copying
//!
//! faer has no rank-3 array / batch type, so [`gemm_batched`] takes the batch as a slice of
//! per-element `(A, B)` [`MatRef`] input pairs matched positionally with a slice of `&mut C`
//! [`MatMut`] outputs, over gemmkit's pointer-array [`gemmkit::gemm_batched_ptr_unchecked`] engine,
//! rather than the 3-D strided form the ndarray adapter uses. gemmkit's core has a shared
//! bias/activation `gemm_batched_fused` for its own strided batch type, but no pointer-array
//! analogue of it, so this crate has nothing to mirror it with

#![cfg_attr(docsrs, feature(doc_cfg))]

use faer::{Mat, MatMut, MatRef};
/// The bias and activation selectors accepted by [`gemm_fused`] and its packed twins (`Bias`
/// only for its complex twin), re-exported so callers need not depend on `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmProblem, GemmScalar, Parallelism, Workspace, gemm_batched_ptr_unchecked,
    gemm_packed_a_unchecked, gemm_packed_a_unchecked_with, gemm_packed_b_unchecked,
    gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with, prepack_lhs_unchecked,
    prepack_rhs_unchecked,
};
/// The handles produced by [`prepack_rhs`]/[`prepack_lhs`] and consumed by the `gemm_packed_*`
/// entries, re-exported so callers need not depend on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
/// The requantization parameters ([`Requantize`]) and its per-tensor / per-row output scale
/// ([`RequantScale`]), taken by [`gemm_i8_requant`] and [`gemm_i8_requant_u8`], re-exported so
/// callers need not depend on `gemmkit` directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
// `MatRef`/`MatMut` unqualified in this crate are always faer's types (imported above); gemmkit
// defines its own type of the same name for its checked slice API, but it is never imported here
#[cfg(feature = "epilogue")]
use gemmkit::{
    FusedScalar, MapScalar, gemm_fused_unchecked, gemm_fused_unchecked_with, gemm_map_unchecked,
    gemm_map_unchecked_with, gemm_packed_a_fused_unchecked, gemm_packed_a_fused_unchecked_with,
    gemm_packed_b_fused_unchecked, gemm_packed_b_fused_unchecked_with,
};
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
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
