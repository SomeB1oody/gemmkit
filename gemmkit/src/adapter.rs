//! Shared validation and epilogue-lowering surface for the view adapters (layer L8a support)
//!
//! The checked core entries in `crate::api` validate slice-backed [`crate::MatRef`] and
//! [`crate::MatMut`] views. Their `C`-overlap test compares the bias against `C`'s full backing
//! slice
//!
//! The out-of-crate view adapters (`gemmkit-ndarray`, `gemmkit-nalgebra`, `gemmkit-faer`) hold
//! raw-pointer views instead. A raw-pointer view may be gappy, with padded columns, or reversed,
//! with negative strides, so the slice-based tier cannot describe it. The bias and requantize
//! checks these adapters need live here, in one pointer-level form
//!
//! Every function here takes a raw pointer plus 1 `(dim, element-stride)` pair per axis, and
//! never forms a reference over the span. This lets it describe a `C` view the caller still
//! holds as an exclusive borrow and has not yet proven in-bounds
//!
//! This module holds the single source for the panic wording. Both the adapters and the checked
//! core entries share it, and the checked entries delegate here, treating `C`'s backing slice as
//! a unit-stride footprint
//!
//! This is a `#[doc(hidden)]` support surface for L8a, not a layer of its own. It is not part of
//! the documented API, and it is versioned in lockstep with the adapters that consume it. It
//! uses only `core` arithmetic, so it works without `std`

#[cfg(feature = "epilogue")]
use crate::Bias;
#[cfg(all(feature = "int8", feature = "epilogue"))]
use crate::RequantScale;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;

/// The half-open byte range `[lo, hi)` a strided view touches
///
/// The view is based at `cp` (its element `(0, ..., 0)`). A negative stride reverses the view
/// and extends `lo` below the base. A positive stride extends `hi` above it. A length-1 axis
/// contributes neither bound. A `dim == 0` axis makes the view empty, and the range collapses
/// to `[cp, cp)`
///
/// This function uses raw pointer arithmetic only and never forms a reference over the
/// (possibly gappy) span. It can therefore describe a `C` view the caller still holds as an
/// exclusive borrow and has not yet proven in-bounds
///
/// # Parameters
///
/// - `cp` - the raw pointer at the view's `(0, ..., 0)` element
/// - `dims` - the `(dim, element-stride)` pair for each axis
///
/// # Returns
///
/// - `(usize, usize)` - the half-open byte range `[lo, hi)` the view touches
#[cfg(feature = "epilogue")]
#[inline]
pub fn c_byte_range<T>(cp: *const T, dims: &[(usize, isize)]) -> (usize, usize) {
    let sz = core::mem::size_of::<T>() as isize;
    if dims.iter().any(|&(d, _)| d == 0) {
        let b = cp as usize;
        return (b, b);
    }
    let (mut lo, mut hi): (isize, isize) = (0, 0);
    for &(d, s) in dims {
        if d <= 1 {
            continue; // a length-1 axis has no extent, so its stride does not matter
        }
        let e = (d as isize - 1) * s;
        if e < 0 {
            lo += e;
        } else {
            hi += e;
        }
    }
    let base = cp as isize;
    ((base + lo * sz) as usize, (base + (hi + 1) * sz) as usize)
}

/// Whether a `len`-element `TB` slice at `bias` overlaps the byte range the strided `C` view
/// touches
///
/// This runs the standard half-open interval test `a0 < b1 && b0 < a1` over [`c_byte_range`].
/// It detects a bias and `C` overlap without ever forming a `C` slice
///
/// # Parameters
///
/// - `cp`/`c_dims` - the `C` view's base pointer and per-axis `(dim, element-stride)` pairs
/// - `bias`/`len` - the bias slice's base pointer and element length
///
/// # Returns
///
/// - `bool` - `true` when the 2 ranges overlap
#[cfg(feature = "epilogue")]
#[inline]
pub fn bias_overlaps_c<TC, TB>(
    cp: *const TC,
    c_dims: &[(usize, isize)],
    bias: *const TB,
    len: usize,
) -> bool {
    let (c_lo, c_hi) = c_byte_range(cp, c_dims);
    if c_lo == c_hi || len == 0 {
        return false;
    }
    let b_lo = bias as usize;
    let b_hi = b_lo + len * core::mem::size_of::<TB>();
    c_lo < b_hi && b_lo < c_hi
}

/// Validate a fused bias against `(m, n)` and `C`'s footprint
///
/// This checks the `Option<Bias<'_, T>>` against `(m, n)` and `C`'s footprint (`cp`/`c_dims`).
/// It panics with the exact wording every checked fused entry uses. It then lowers the bias to
/// the raw `(ptr, BiasDim, has_bias)` triple the `_unchecked` core entries take
///
/// A `PerRow` bias must have length `m`. A `PerCol` bias must have length `n`. Neither may
/// overlap `C` ([`bias_overlaps_c`], raw pointer math only)
///
/// # Parameters
///
/// - `bias` - the optional per-row or per-col bias to validate
/// - `m` - row count of `A` and `C`, the required `PerRow` length
/// - `n` - column count of `B` and `C`, the required `PerCol` length
/// - `cp`/`c_dims` - `C`'s base pointer and per-axis `(dim, element-stride)` pairs
///
/// # Returns
///
/// - `(*const T, BiasDim, bool)` - the lowered pointer, its dimension, and whether a bias is
///   present
///
/// # Panics
///
/// When the bias length does not match `m` or `n`, or when the bias overlaps `C`
#[cfg(feature = "epilogue")]
pub fn lower_bias<T>(
    bias: Option<Bias<'_, T>>,
    m: usize,
    n: usize,
    cp: *const T,
    c_dims: &[(usize, isize)],
) -> (*const T, BiasDim, bool) {
    match bias {
        None => (core::ptr::null(), BiasDim::PerRow, false),
        Some(Bias::PerRow(s)) => {
            assert_eq!(
                s.len(),
                m,
                "gemmkit: PerRow bias length ({}) != A.rows ({})",
                s.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, s.as_ptr(), s.len()) {
                panic!("gemmkit: bias slice overlaps C");
            }
            (s.as_ptr(), BiasDim::PerRow, true)
        }
        Some(Bias::PerCol(s)) => {
            assert_eq!(
                s.len(),
                n,
                "gemmkit: PerCol bias length ({}) != B.cols ({})",
                s.len(),
                n
            );
            if bias_overlaps_c(cp, c_dims, s.as_ptr(), s.len()) {
                panic!("gemmkit: bias slice overlaps C");
            }
            (s.as_ptr(), BiasDim::PerCol, true)
        }
    }
}

/// Validate an optional requantize `i32` bias against `C`'s footprint
///
/// The bias must have length `m == A.rows`. `C`'s footprint is its `i8`/`u8` output
/// (`cp`/`c_dims`). This panics with the requantizing entries' wording on a length mismatch or
/// an overlap. It then lowers the bias to the raw `(ptr, has_bias)` pair the `_unchecked`
/// requant entries take
///
/// # Parameters
///
/// - `m` - row count of `A`, the required bias length
/// - `cp`/`c_dims` - the `i8`/`u8` `C` view's base pointer and per-axis `(dim, element-stride)`
///   pairs
/// - `bias` - the optional `i32` bias to validate
///
/// # Returns
///
/// - `(*const i32, bool)` - the lowered pointer and whether a bias is present
///
/// # Panics
///
/// When the bias length does not match `m`, or when the bias overlaps `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn requant_bias<TC>(
    m: usize,
    cp: *const TC,
    c_dims: &[(usize, isize)],
    bias: Option<&[i32]>,
) -> (*const i32, bool) {
    match bias {
        Some(bias) => {
            assert_eq!(
                bias.len(),
                m,
                "gemmkit: requantize bias length ({}) != A.rows ({})",
                bias.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, bias.as_ptr(), bias.len()) {
                panic!("gemmkit: requantize bias overlaps C");
            }
            (bias.as_ptr(), true)
        }
        None => (core::ptr::null(), false),
    }
}

/// Validate a [`RequantScale`] against `C`'s footprint
///
/// `C`'s footprint is its `i8`/`u8` output (`cp`/`c_dims`). This panics with the requantizing
/// entries' wording on an invalid value. It then lowers the scale to the raw `(scale,
/// row_scales, has_row_scales)` triple the `_unchecked` requant entries take
///
/// A `PerTensor(s)` value must be finite and greater than 0. A `PerRow` value must have length
/// `m == A.rows`, with every element finite and greater than 0, and must not overlap `C`
///
/// # Parameters
///
/// - `m` - row count of `A`, the required `PerRow` length
/// - `cp`/`c_dims` - the `i8`/`u8` `C` view's base pointer and per-axis `(dim, element-stride)`
///   pairs
/// - `scale` - the per-tensor or per-row scale to validate
///
/// # Returns
///
/// - `(f32, *const f32, bool)` - the tensor scale (0 when per-row), the row-scale pointer, and
///   whether row scales are present
///
/// # Panics
///
/// When a scale value is not finite or not positive, or when per-row scales overlap `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn requant_scale<TC>(
    m: usize,
    cp: *const TC,
    c_dims: &[(usize, isize)],
    scale: RequantScale<'_>,
) -> (f32, *const f32, bool) {
    match scale {
        RequantScale::PerTensor(s) => {
            assert!(
                s.is_finite() && s > 0.0,
                "gemmkit: requantize scale ({s}) must be finite and > 0"
            );
            (s, core::ptr::null(), false)
        }
        RequantScale::PerRow(scales) => {
            assert_eq!(
                scales.len(),
                m,
                "gemmkit: requantize scales length ({}) != A.rows ({})",
                scales.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, scales.as_ptr(), scales.len()) {
                panic!("gemmkit: requantize scales overlap C");
            }
            for &s in scales {
                assert!(
                    s.is_finite() && s > 0.0,
                    "gemmkit: requantize scale ({s}) must be finite and > 0"
                );
            }
            (0.0, scales.as_ptr(), true)
        }
    }
}
