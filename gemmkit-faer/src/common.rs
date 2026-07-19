//! Shared view-adapter helpers for the other modules: pulling raw parts out of a faer view, and
//! (under `epilogue`) replicating gemmkit's own bias/requant validation against a raw C footprint
//! without ever forming a reference over C
use super::*;

/// Pull `(rows, cols, row-stride, col-stride, ptr)` out of a [`MatRef`]. faer already reports
/// strides in element units as `isize` (negative for a reversed view), which is exactly what
/// gemmkit's raw entries take, so the values pass through unconverted
#[inline]
pub(crate) fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (
        a.nrows(),
        a.ncols(),
        a.row_stride(),
        a.col_stride(),
        a.as_ptr(),
    )
}

/// Allocate an `m x n` column-major [`Mat`] with every cell set to `zero`. Used by the `dot`-family
/// wrappers as the output buffer for a `beta == 0` call: gemm overwrites every element, so the fill
/// value is never read back, it only needs to exist so the buffer is initialized. `Mat::from_fn`
/// carries no numeric trait bound, so this works for element types (`f16`/`bf16`, `i32`) that don't
/// implement faer's own `ComplexField`
#[inline]
pub(crate) fn filled_mat<T: Copy>(m: usize, n: usize, zero: T) -> Mat<T> {
    Mat::from_fn(m, n, |_, _| zero)
}

/// The half-open byte range `[lo, hi)` that a strided view based at `cp` (element `(0, 0)`) actually
/// touches, given `(dim, element-stride)` pairs for each axis. A stride may be negative (a
/// `reverse_rows`/`reverse_cols` view), so a negative axis pushes `lo` below the base and a positive
/// one pushes `hi` above it; if any axis has `dim == 0` the view is empty and the range collapses to
/// `[cp, cp)`. Pure pointer arithmetic, no reference is formed over the span (which may be gappy: a
/// `Mat` whose column stride exceeds `nrows` has uninitialized padding between columns), so this is
/// safe to call even though C is an exclusive borrow the caller still holds
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
            continue; // length-1 axis spans nothing: its stride doesn't move the range
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

/// `true` if the `bias` slice (`len` elements of `TB`) overlaps the byte range the strided C view
/// touches. The standard `a0 < b1 && b0 < a1` interval-overlap test over [`c_byte_range`], so the
/// adapter can reject an aliasing bias the same way gemmkit's checked core entry does, without ever
/// forming a `C` slice
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

/// Validate a fused `Option<Bias>` against the output shape and against `C`'s footprint, with the
/// same panic wording as gemmkit's checked core entry, then lower it to the raw `(ptr, BiasDim,
/// has_bias)` triple the `_unchecked` core entries take. `cp`/`c_dims` describe `C` for the overlap
/// test via [`bias_overlaps_c`]
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

/// Validate a requantize per-row bias against `A.rows` and against `C`'s footprint, with the same
/// panic wording as gemmkit's core `requant_bias`, then lower it to the raw `(ptr, has_bias)` pair
/// the `_unchecked` requant entries take. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap
/// test
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

/// Validate a requantize [`RequantScale`] against `A.rows` and against `C`'s footprint, with the
/// same panic wording as gemmkit's core `requant_scale`, then lower it to the raw `(scale,
/// row_scales, has_row_scales)` triple the `_unchecked` requant entries take. A `PerTensor(s)` must
/// be finite and `> 0`; a `PerRow` slice must have length `m`, every element finite and `> 0`, and
/// must not overlap `C`. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap test
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
