//! # gemmkit-ndarray
//!
//! Thin [`ndarray`] adapter over the [`gemmkit`] GEMM engine. The 2-D entries take
//! `&ArrayBase<S, Ix2>` for any storage `S: Data`, so both `ArrayView2` and `&Array2` work.
//! Each entry reads the pointer and strides directly from the array and forwards them to
//! gemmkit's raw engine. C-order, F-order, general-stride, and reversed (negative-stride)
//! views all work without copying
//!
//! ```
//! use ndarray::array;
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//! let c = gemmkit_ndarray::dot(&a, &b);
//! assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`GemmScalar`]: `f32`/`f64` always, plus
//! `f16`/`bf16` under the `half` feature. [`gemm_batched`]/[`gemm_batched_with`]/
//! [`dot_batched`] extend the same idea to a stack of matrices, a 3-D array with the batch on
//! axis 0. [`prepack_rhs`]/[`prepack_lhs`], with their [`gemm_packed_b`]/[`gemm_packed_a`]
//! consumers, pre-pack 1 reused operand for a fixed-weight loop. Complex (`Complex<f32>`/
//! `Complex<f64>`, with optional conjugation) needs the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] entries under the `complex` feature, because
//! the conjugation flags do not fit the homogeneous surface. The integer (`i8 -> i32`) path
//! likewise gets its own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] entries under the `int8`
//! feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused entries mirror gemmkit's own.
//! [`gemm_fused`]/[`gemm_fused_with`] compute `C <- act(alpha*A*B + beta*C + bias)` in 1 pass,
//! with an optional [`Bias`] and an optional [`Activation`]. The batched twins
//! [`gemm_batched_fused`]/[`gemm_batched_fused_with`] apply 1 shared bias and activation to
//! every element of the stack. The prepacked-operand twins
//! [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] reuse the same
//! [`PackedRhs`]/[`PackedLhs`] handle with a fused bias and activation.
//! [`gemm_map`]/[`gemm_map_with`] fuse an arbitrary `f(value, row, col)` closure into the
//! store, for `f32`/`f64` only, with no `f16`/`bf16` extension under `half`.
//! Requantized output needs `int8` plus `epilogue`. [`gemm_i8_requant`]/[`gemm_i8_requant_with`]
//! (and the `u8`-output [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a
//! [`Requantize`] and fuse it into a quantized `i8` or `u8` output. Complex-fused needs
//! `complex` plus `epilogue`: the bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] has no
//! activation parameter, because an ordering activation is undefined on complex numbers

#![cfg_attr(docsrs, feature(doc_cfg))]

/// The element-type bound of the complex entries. This lets a caller writing a wrapper
/// generic over [`gemm_cplx`] name the bound without depending on `gemmkit` directly
#[cfg(feature = "complex")]
pub use gemmkit::ComplexScalar;
/// The element-type bound of the plain real entries. This lets a caller writing a wrapper
/// generic over [`gemm`], [`dot`], or [`prepack_rhs`] name the bound without depending on
/// `gemmkit` directly
pub use gemmkit::GemmScalar;
/// This re-exports gemmkit's heuristic thresholds: the `tuning::set_*` setters, their
/// getters, and the compiled defaults. A caller need not depend on `gemmkit` directly to
/// reach them. The knobs are process-global atomics, so reaching them through a separately
/// resolved second `gemmkit` would set a copy this adapter never reads
#[doc(no_inline)]
pub use gemmkit::tuning;
/// The bias and activation selectors for the fused entries. This lets a caller of
/// [`gemm_fused`] avoid depending on `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
/// The complex element type of [`gemm_cplx`] and its `c32`/`c64` aliases. This lets callers
/// avoid depending on `num-complex` directly, because [`ndarray`] supplies no complex type of
/// its own
#[cfg(feature = "complex")]
#[doc(no_inline)]
pub use gemmkit::{Complex, c32, c64};
/// The element-type bounds of the fused entries: [`FusedScalar`] for the bias/activation form
/// and [`MapScalar`] for [`gemm_map`]. This lets a caller writing a wrapper generic over them
/// name the bound without depending on `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{FusedScalar, MapScalar};
/// The prepacked-operand handles. This lets callers of [`prepack_rhs`] and [`prepack_lhs`]
/// avoid depending on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
/// The [`Parallelism`] selector taken by every entry, and the reusable [`Workspace`] taken by
/// the `_with` variants. This lets callers avoid depending on `gemmkit` directly
pub use gemmkit::{Parallelism, Workspace};
/// The requantize parameters ([`Requantize`]) and its per-tensor or per-row output scale
/// ([`RequantScale`]). This lets callers of [`gemm_i8_requant`] avoid depending on `gemmkit`
/// directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
/// The narrow float element types of the `half`-gated entries. This lets callers avoid
/// depending on `half` directly. Both accumulate in `f32`
#[cfg(feature = "half")]
#[doc(no_inline)]
pub use gemmkit::{bf16, f16};
#[cfg(feature = "epilogue")]
use gemmkit::{
    gemm_batched_fused_unchecked, gemm_batched_fused_unchecked_with, gemm_fused_unchecked,
    gemm_fused_unchecked_with, gemm_map_unchecked, gemm_map_unchecked_with,
    gemm_packed_a_fused_unchecked, gemm_packed_a_fused_unchecked_with,
    gemm_packed_b_fused_unchecked, gemm_packed_b_fused_unchecked_with,
};
use gemmkit::{
    gemm_batched_unchecked, gemm_batched_unchecked_with, gemm_packed_a_unchecked,
    gemm_packed_a_unchecked_with, gemm_packed_b_unchecked, gemm_packed_b_unchecked_with,
    gemm_unchecked, gemm_unchecked_with, prepack_lhs_unchecked, prepack_rhs_unchecked,
};
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
#[cfg(feature = "complex")]
use gemmkit::{gemm_cplx_unchecked, gemm_cplx_unchecked_with};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::{
    gemm_i8_requant_u8_unchecked, gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with,
};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};
use ndarray::{Array2, Array3, ArrayBase, Data, DataMut, Ix2, Ix3};

// Batched ndarray GEMM entries, batch on axis 0 of a 3-D array
mod batched;
// Shared dims/strides extraction for the entry modules (bias/requant validation lives in
// gemmkit's adapter module)
mod common;
// Complex ndarray GEMM entries with optional conjugation
#[cfg(feature = "complex")]
mod cplx;
// Real f32/f64 (plus f16/bf16 under half) ndarray GEMM entries
mod float;
// Fused bias/activation ndarray GEMM entries
#[cfg(feature = "epilogue")]
mod fused;
// Integer i8 -> i32 and requantizing i8/u8 ndarray GEMM entries
#[cfg(feature = "int8")]
mod int8;
// User-defined per-element map-epilogue ndarray GEMM entries
#[cfg(feature = "epilogue")]
mod map;
// Prepacked-operand (PackedLhs/PackedRhs) ndarray GEMM entries
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
