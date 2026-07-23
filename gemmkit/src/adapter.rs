//! Shared validation and epilogue-lowering surface for the view adapters (layer L8a support)
//!
//! The checked core entries (`crate::api`) validate slice-backed [`crate::MatRef`]/
//! [`crate::MatMut`] views, so their `C`-overlap test compares the bias against `C`'s full
//! backing slice. The out-of-crate view adapters (`gemmkit-ndarray`, `gemmkit-nalgebra`,
//! `gemmkit-faer`) instead hold raw-pointer views that may be gappy (padded columns) or
//! reversed (negative strides), which the slice-based tier cannot describe, so the bias/requant
//! checks they need live here in one pointer-level form. Every function works on a raw pointer
//! plus 1 `(dim, element-stride)` pair per axis and never forms a reference over the span, so it
//! can validate a `C` the caller still holds as an exclusive borrow and has not proven in-bounds.
//! The panic wording is the single source both the adapters and the checked core entries (which
//! delegate here over `C`'s backing slice as a unit-stride footprint) share
//!
//! This is a `#[doc(hidden)]` support surface for L8a, not a layer of its own: it is not part of
//! the documented API and is versioned in lockstep with the adapters that consume it. `core`
//! arithmetic only, so it stays `no_std`-clean

#[cfg(feature = "epilogue")]
use crate::Bias;
#[cfg(all(feature = "int8", feature = "epilogue"))]
use crate::RequantScale;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;

/// The half-open byte range `[lo, hi)` a strided view based at `cp` (its element `(0, ..., 0)`)
/// touches, given the raw pointer plus 1 `(dim, element-stride)` pair per axis. A negative stride
/// (a reversed view) extends `lo` below the base, a positive one extends `hi` above it (a length-1
/// axis contributes neither); any `dim == 0` axis makes the view empty and collapses the range to
/// `[cp, cp)`. Raw pointer arithmetic only: no reference is ever formed over the (possibly gappy)
/// span, so this can describe a `C` the caller has not yet proven in-bounds and still holds as an
/// exclusive borrow
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

/// `true` if a `len`-element `TB` slice at `bias` overlaps the byte range the strided `C` view
/// (`cp`/`c_dims`) touches: the standard half-open interval test `a0 < b1 && b0 < a1`, over
/// [`c_byte_range`], so a bias/`C` overlap is rejected without ever forming a `C` slice
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

/// Validate a fused `Option<Bias>` against `(m, n)` and `C`'s footprint (`cp`/`c_dims`), panicking
/// with the exact wording every checked fused entry uses, and lower it to the raw `(ptr, BiasDim,
/// has_bias)` triple the `_unchecked` core entries take. `PerRow` must have length `m`, `PerCol`
/// length `n`; either must not overlap `C` ([`bias_overlaps_c`], raw pointer math only)
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

/// Validate an optional requantize `i32` bias (length `m == A.rows`) against `C`'s footprint
/// (`cp`/`c_dims`, the `i8`/`u8` output), panicking with the requantizing entries' wording, and
/// lower it to the raw `(ptr, has_bias)` pair the `_unchecked` requant entries take
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

/// Validate a [`RequantScale`] against `C`'s footprint (`cp`/`c_dims`, the `i8`/`u8` output),
/// panicking with the requantizing entries' wording, and lower it to the raw `(scale, row_scales,
/// has_row_scales)` triple the `_unchecked` requant entries take. `PerTensor(s)` must be finite
/// and `> 0`; `PerRow` must have length `m == A.rows`, every element finite and `> 0`, and must not
/// overlap `C`
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
