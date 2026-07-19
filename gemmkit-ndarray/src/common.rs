//! Shared dims/strides extraction, plus the bias/requant validation used by the epilogue-gated
//! entries to replicate gemmkit's own checked-entry panics without materializing a `C` slice
use super::*;

#[inline]
pub(crate) fn dims_strides<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix2>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}

/// The half-open byte range `[lo, hi)` a strided view based at `cp` (element `(0, ..., 0)`)
/// touches, given the raw pointer plus its `(dim, element-stride)` pairs. A negative stride
/// extends `lo` below the base, a positive one extends `hi` above it (a length-1 axis
/// contributes neither); an empty (`dim == 0`) axis yields an empty range at `cp`. **Raw pointer
/// arithmetic only**: no reference is ever formed over the (possibly gappy) span, so this can
/// describe a `C` the caller has not yet proven in-bounds
#[cfg(feature = "epilogue")]
#[inline]
pub(crate) fn c_byte_range<T>(cp: *const T, dims: &[(usize, isize)]) -> (usize, usize) {
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
/// [`c_byte_range`], so the adapter can reject a bias/`C` overlap without ever forming a `C`
/// slice
#[cfg(feature = "epilogue")]
#[inline]
pub(crate) fn bias_overlaps_c<TC, TB>(
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
/// with the exact wording gemmkit's own checked entry uses, and lower it to the raw `(ptr,
/// BiasDim, has_bias)` triple the `_unchecked` core entries take. `PerRow` must have length `m`,
/// `PerCol` length `n`; either must not overlap `C` ([`bias_overlaps_c`], raw pointer math only)
#[cfg(feature = "epilogue")]
pub(crate) fn lower_bias<T>(
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
/// (`cp`/`c_dims`, the `i8`/`u8` output), panicking with gemmkit's own checked-entry wording, and
/// lower it to the raw `(ptr, has_bias)` pair the `_unchecked` requant entries take
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) fn requant_bias<TC>(
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
/// panicking with gemmkit's own checked-entry wording, and lower it to the raw `(scale,
/// row_scales, has_row_scales)` triple the `_unchecked` requant entries take. `PerTensor(s)` must
/// be finite and `> 0`; `PerRow` must have length `m == A.rows`, every element finite and `> 0`,
/// and must not overlap `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) fn requant_scale<TC>(
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
