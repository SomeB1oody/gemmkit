//! # gemmkit-nalgebra
//!
//! A thin [`nalgebra`] adapter over the [`gemmkit`] GEMM engine. Every entry point takes
//! `&Matrix<T, R, C, S>` for any storage `S: RawStorage`, so `DMatrix`, a static `SMatrix`, and
//! every view type all work. Each entry reads the pointer and strides directly off the matrix
//! and forwards them to the gemmkit raw engine. Column-major (nalgebra's natural layout),
//! row-major, and general-stride views all work without copying
//!
//! ```
//! use nalgebra::{DMatrix, Matrix2};
//! let a = Matrix2::new(1.0_f32, 2.0, 3.0, 4.0);
//! let b = Matrix2::new(5.0_f32, 6.0, 7.0, 8.0);
//! let c = gemmkit_nalgebra::dot(&a, &b);
//! assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
//! ```
//!
//! [`gemm`], [`gemm_with`], and [`dot`] are generic over [`GemmScalar`]: `f32`/`f64` always, plus
//! `f16`/`bf16` under the `half` feature. [`prepack_rhs`] and [`prepack_lhs`] (consumed by
//! [`gemm_packed_b`] and [`gemm_packed_a`]) pre-pack 1 reused operand for a fixed-weight
//! inference loop. Complex types (`Complex<f32>`/`Complex<f64>`, with optional conjugation) need
//! the separate [`gemm_cplx`], [`gemm_cplx_with`], and [`dot_cplx`] entries under the `complex`
//! feature. The conjugation flags do not fit the homogeneous surface. The integer (`i8 -> i32`)
//! path likewise gets its own [`gemm_i8`], [`gemm_i8_with`], and [`dot_i8`] under
//! the `int8` feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused-epilogue entries mirror gemmkit's own. [`gemm_fused`]
//! and [`gemm_fused_with`] compute `C <- act(alpha*A*B + beta*C + bias)` in 1 pass, with an
//! optional [`Bias`] and an optional [`Activation`]. The prepacked-operand twins
//! [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] take the same reused
//! [`PackedRhs`]/[`PackedLhs`] handle, plus a fused bias and activation. `f16`/`bf16` ride the
//! same generic when `half` is also on. Requantized output needs `int8` + `epilogue`.
//! [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse it into a
//! quantized `i8` or `u8` output. Complex-fused needs `complex` +
//! `epilogue`: the bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] takes no activation
//! parameter, because an ordering activation is undefined on complex numbers
//!
//! nalgebra has no rank-3 array type. [`gemm_batched`] instead takes a batch as a slice of
//! per-element `(&A, &B)` inputs, paired with a slice of `&mut C` outputs. This runs over
//! gemmkit's pointer-array [`gemmkit::gemm_batched_ptr_unchecked`] engine, rather than the 3-D
//! strided form the ndarray adapter uses. The ndarray adapter's `gemm_batched_fused` (1 bias and
//! activation shared by the whole batch) has no pointer-array analogue in gemmkit's core. It has
//! no matching entry here

#![cfg_attr(docsrs, feature(doc_cfg))]

/// Re-exported element-type bound for the complex entries. A wrapper generic over
/// [`gemm_cplx`] can name the bound without a direct `gemmkit` dependency
#[cfg(feature = "complex")]
pub use gemmkit::ComplexScalar;
/// Re-exported element-type bound for the plain real entries. A wrapper generic over
/// [`gemm`]/[`dot`]/[`prepack_rhs`] can name the bound without a direct `gemmkit` dependency
pub use gemmkit::GemmScalar;
/// Re-exported heuristic thresholds from gemmkit: the `tuning::set_*` setters, their getters,
/// and the compiled defaults. Callers do not need a direct `gemmkit` dependency to reach them.
/// The knobs are process-global atomics, so a separately resolved 2nd `gemmkit` copy would set
/// values this adapter never reads
#[doc(no_inline)]
pub use gemmkit::tuning;
/// Fused-epilogue selectors ([`Bias`], [`Activation`]), re-exported so callers of [`gemm_fused`]
/// do not need a direct `gemmkit` dependency
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
/// The complex element type behind [`gemm_cplx`], with its `c32`/`c64` aliases, re-exported so a
/// caller does not need a 2nd source for them. These are the same `num_complex` types as
/// [`nalgebra::Complex`]
#[cfg(feature = "complex")]
#[doc(no_inline)]
pub use gemmkit::{Complex, c32, c64};
/// Re-exported element-type bounds for the fused entries: [`FusedScalar`] for the bias and
/// activation form, [`MapScalar`] for [`gemm_map`]. A wrapper generic over them can name the
/// bound without a direct `gemmkit` dependency
#[cfg(feature = "epilogue")]
pub use gemmkit::{FusedScalar, MapScalar};
use gemmkit::{
    GemmProblem, gemm_batched_ptr_unchecked, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_b_unchecked, gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with,
    prepack_lhs_unchecked, prepack_rhs_unchecked,
};
/// Prepacked-operand handles, re-exported so callers of [`prepack_rhs`]/[`prepack_lhs`] do not
/// need a direct `gemmkit` dependency
pub use gemmkit::{PackedLhs, PackedRhs};
/// Every entry except [`prepack_rhs`]/[`prepack_lhs`] takes the [`Parallelism`] selector,
/// re-exported here so callers do not need a direct `gemmkit` dependency. The `_with` variants
/// also take the reusable [`Workspace`], re-exported for the same reason
pub use gemmkit::{Parallelism, Workspace};
/// Requantization parameters ([`Requantize`], with its per-tensor or per-row output scale
/// [`RequantScale`]) for the `int8` fused entries. This is re-exported so callers of
/// [`gemm_i8_requant`] do not need a direct `gemmkit` dependency
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
/// The narrow float element types of the `half`-gated entries, re-exported so callers do not
/// need a direct `half` dependency ([`nalgebra`] exposes neither). Both accumulate in `f32`
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
use nalgebra::{DMatrix, Dim, Dyn, Matrix, RawStorage, RawStorageMut, VecStorage};

// batched GEMM over a slice of independently shaped (&A, &B) -> &mut C triples
mod batched;
// dims/strides extraction and output-buffer allocation shared by the entry modules (bias/scale
// validation lives in gemmkit's adapter module)
mod common;
// complex GEMM (Complex<f32>/Complex<f64>) with optional per-operand conjugation
#[cfg(feature = "complex")]
mod cplx;
// the plain real-scalar gemm/gemm_with/dot entries
mod float;
// bias + activation fused into the GEMM store
#[cfg(feature = "epilogue")]
mod fused;
// i8 x i8 -> i32 GEMM, plus the requantizing i8/u8-output entries
#[cfg(feature = "int8")]
mod int8;
// per-element closure fused into the GEMM store
#[cfg(feature = "epilogue")]
mod map;
// entries that reuse a pre-packed PackedLhs/PackedRhs operand
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
