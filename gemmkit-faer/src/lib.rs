//! # gemmkit-faer
//!
//! A thin [`faer`] adapter over the [`gemmkit`] GEMM engine: it takes faer's view types
//! ([`MatRef<'_, T>`](faer::MatRef) for inputs, [`MatMut<'_, T>`](faer::MatMut) for the output),
//! pulls the data pointer and the element-unit `isize` row/column strides straight out of the view,
//! and forwards to gemmkit's raw engine. faer's natural column-major layout, transposed views,
//! sub-matrices, and reversed (negative-stride) views therefore all work without copying
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
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64` always, plus
//! `f16`/`bf16` under the `half` feature. [`prepack_rhs`]/[`prepack_lhs`] (with their
//! [`gemm_packed_b`]/[`gemm_packed_a`] consumers) pre-pack the reused operand for the fixed-weight
//! loop. Complex (`Complex<f32>`/`Complex<f64>`, i.e. faer's `c32`/`c64`, with optional conjugation)
//! needs the separate [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since
//! the conj flags don't fit the homogeneous surface. The integer (`i8 -> i32`) path likewise gets its
//! own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused-epilogue entries mirror gemmkit's own:
//! [`gemm_fused`]/[`gemm_fused_with`] (`C <- act(alpha*A*B + beta*C + bias)` in 1 pass, an optional
//! [`Bias`] plus an optional [`Activation`]) and the prepacked-operand twins
//! [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] (the same reused [`PackedRhs`]/[`PackedLhs`]
//! handle plus a fused bias/activation). `f16`/`bf16` ride the same generic when `half` is on.
//! Requantized output needs `int8` + `epilogue`: [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and
//! the `u8`-output [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse
//! the requantize into a quantized `i8` (resp. `u8`) output. Complex-fused needs `complex` +
//! `epilogue`: the bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] (no activation: undefined
//! on complex numbers). Like the plain entries, these read raw parts out of the view and forward to
//! gemmkit's raw engine, so transposed, sub-matrix, and reversed (negative-stride) views all work
//! without copying
//!
//! faer has no rank-3 array / batch type, so the batched [`gemm_batched`] takes the batch as a slice
//! of per-element `(A, B)` [`MatRef`] inputs paired with a slice of `&mut C`
//! [`MatMut`] outputs (over gemmkit's pointer-array
//! [`gemmkit::gemm_batched_ptr_unchecked`] engine), not the 3-D strided form the ndarray adapter
//! uses; the one shared bias/activation `gemm_batched_fused` of the ndarray adapter has no
//! pointer-array analogue in the core, so it is not mirrored here

#![cfg_attr(docsrs, feature(doc_cfg))]

use faer::{Mat, MatMut, MatRef};
/// The fused-epilogue selectors, re-exported so callers of [`gemm_fused`] need not depend on
/// `gemmkit` directly
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
/// The prepacked-operand handles, re-exported so callers of [`prepack_rhs`] / [`prepack_lhs`] need
/// not depend on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
/// The requantization parameters ([`Requantize`]) and its per-tensor / per-row output scale
/// ([`RequantScale`]) for the `int8` fused entries, re-exported so callers of [`gemm_i8_requant`]
/// need not depend on `gemmkit` directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
// The unqualified `MatRef`/`MatMut` names here are faer's view types (imported above); the fused
// entries pull raw parts out of them and forward to gemmkit's raw engine
#[cfg(feature = "epilogue")]
use gemmkit::{
    BiasDim, FusedScalar, MapScalar, gemm_fused_unchecked, gemm_fused_unchecked_with,
    gemm_map_unchecked, gemm_map_unchecked_with, gemm_packed_a_fused_unchecked,
    gemm_packed_a_fused_unchecked_with, gemm_packed_b_fused_unchecked,
    gemm_packed_b_fused_unchecked_with,
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

// pointer-array-batched GEMM over a slice of per-element view triples
mod batched;
// shared view-adapter helpers (ref-part extraction, C-footprint / bias / requant validation)
mod common;
// complex GEMM entries with optional conjugation
#[cfg(feature = "complex")]
mod cplx;
// homogeneous f32/f64 (plus f16/bf16 under half) GEMM entries
mod float;
// fused-epilogue (bias/activation) GEMM entries
#[cfg(feature = "epilogue")]
mod fused;
// integer (i8 -> i32) and requantizing (i8 -> i8) GEMM entries
#[cfg(feature = "int8")]
mod int8;
// user-defined per-element map-epilogue GEMM entries
#[cfg(feature = "epilogue")]
mod map;
// prepacked-operand (PackedLhs/PackedRhs) entries
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
