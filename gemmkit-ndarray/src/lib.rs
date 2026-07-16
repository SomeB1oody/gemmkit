//! # gemmkit-ndarray
//!
//! A thin [`ndarray`] adapter over the [`gemmkit`] GEMM engine. It accepts
//! `&ArrayBase<S, Ix2>` for any storage `S: Data` (so both `ArrayView2` and
//! `&Array2` work), pulls the pointer and strides straight out of the array, and
//! forwards to gemmkit's raw engine, so C-order, F-order, general-stride, and
//! reversed (negative-stride) views all work without copying
//!
//! ```
//! use ndarray::array;
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//! let c = gemmkit_ndarray::dot(&a, &b);
//! assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! always, plus `f16`/`bf16` under the `half` feature. [`gemm_batched`]/[`gemm_batched_with`]/
//! [`dot_batched`] extend the same idea to a stack of matrices (a 3-D array with the batch on
//! axis 0) and [`prepack_rhs`]/[`prepack_lhs`] (with their [`gemm_packed_b`]/[`gemm_packed_a`]
//! consumers) pre-pack one reused operand for the fixed-weight loop. Complex
//! (`Complex<f32>`/`Complex<f64>`, with optional conjugation) needs the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since the
//! conj flags don't fit the homogeneous surface. The integer (`i8 -> i32`) path likewise gets its
//! own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused-epilogue entries mirror gemmkit's own:
//! [`gemm_fused`]/[`gemm_fused_with`] (`C <- act(alpha*A*B + beta*C + bias)` in 1 pass, an
//! optional [`Bias`] plus an optional [`Activation`]), their batched twins
//! [`gemm_batched_fused`]/[`gemm_batched_fused_with`] (one shared bias/activation over every element
//! of the stack), and the prepacked-operand twins
//! [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] (the same reused [`PackedRhs`]/[`PackedLhs`]
//! handle plus a fused bias/activation). `f16`/`bf16` ride the same generic when `half` is on.
//! Requantized output needs
//! `int8` + `epilogue`: [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse the requantize
//! into a quantized `i8` (resp. `u8`) output. Complex-fused needs `complex` + `epilogue`: the
//! bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] (no activation: undefined on complex numbers)

#![cfg_attr(docsrs, feature(doc_cfg))]

/// The fused-epilogue selectors, re-exported so callers of [`gemm_fused`] need not depend on
/// `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
#[cfg(feature = "epilogue")]
use gemmkit::{
    BiasDim, FusedScalar, MapScalar, gemm_batched_fused_unchecked,
    gemm_batched_fused_unchecked_with, gemm_fused_unchecked, gemm_fused_unchecked_with,
    gemm_map_unchecked, gemm_map_unchecked_with, gemm_packed_a_fused_unchecked,
    gemm_packed_a_fused_unchecked_with, gemm_packed_b_fused_unchecked,
    gemm_packed_b_fused_unchecked_with,
};
#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmScalar, Parallelism, Workspace, gemm_batched_unchecked, gemm_batched_unchecked_with,
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
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::{
    gemm_i8_requant_u8_unchecked, gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with,
};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};
use ndarray::{Array2, Array3, ArrayBase, Data, DataMut, Ix2, Ix3};

// strided-batched ndarray GEMM entries (batch on axis 0)
mod batched;
// shared dims/strides extraction and fused-epilogue validation helpers
mod common;
// complex ndarray GEMM entries with optional conjugation
#[cfg(feature = "complex")]
mod cplx;
// real f32/f64 (and f16/bf16 under half) ndarray GEMM entries
mod float;
// fused-epilogue (bias/activation) ndarray GEMM entries
#[cfg(feature = "epilogue")]
mod fused;
// integer (i8 -> i32) and requantizing (i8 -> i8) ndarray GEMM entries
#[cfg(feature = "int8")]
mod int8;
// user-defined per-element map-epilogue ndarray GEMM entries
#[cfg(feature = "epilogue")]
mod map;
// prepacked-operand (PackedLhs/PackedRhs) ndarray GEMM entries
mod packed;

pub use batched::{dot_batched, gemm_batched, gemm_batched_with};
#[cfg(feature = "epilogue")]
pub use batched::{gemm_batched_fused, gemm_batched_fused_with};
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
