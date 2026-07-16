//! Shared stride-extraction, C-footprint, and epilogue-lowering helpers
use super::*;

/// Pull `(rows, cols, row-stride, col-stride)` out of a matrix of any storage. nalgebra reports
/// non-negative `usize` strides in element units; widen to the `isize` gemmkit's raw engine takes
#[inline]
pub(crate) fn dims_strides<T, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    a: &Matrix<T, R, C, S>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.shape();
    let (rs, cs) = a.strides();
    (r, c, rs as isize, cs as isize)
}

/// The half-open byte range `[lo, hi)` a strided C view based at `cp` (element `(0, 0)`) actually
/// touches, from the raw pointer plus `(dim, element-stride)` pairs. Strides are non-negative on
/// nalgebra, but the negative branch is kept for parity with gemmkit's own `extent`; an empty
/// (`dim == 0`) axis yields an empty range. **Raw pointer arithmetic only**: no reference is ever
/// formed over the span, which is why the fused entries forward raw parts to gemmkit's `_unchecked`
/// engine instead of fabricating a slice here
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
            continue; // a length-1 axis spans nothing, so its stride (any sign) is irrelevant
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
/// touches: the raw-pointer replication of gemmkit's own byte-range overlap test (`a0 < b1 && b0 <
/// a1`), so the adapter reproduces the core checked entry's bias-vs-`C` rejection without ever
/// fabricating a `C` slice
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

/// Validate a fused `Option<Bias>` against the output shape and `C`'s footprint (replicating the
/// core checked entry's `validate_bias`; panic wording is byte-identical), and lower it to the raw
/// `(ptr, BiasDim, has_bias)` triple the `_unchecked` core entries take. `cp`/`c_dims` describe `C`
/// for the overlap test via [`bias_overlaps_c`] (raw pointer math; `C` is never referenced)
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

/// Validate a requantize per-row bias against `A.rows` and `C`'s footprint (replicating the core
/// `requant_bias`; panic wording is byte-identical), and lower it to the raw `(ptr, has_bias)` the
/// `_unchecked` requant entries take. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap
/// test; raw pointer math only
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

/// Validate a requantize [`RequantScale`] against `A.rows` and `C`'s footprint, replicating the
/// core `requant_scale` (byte-identical panic wording), and lower it to the raw `(scale,
/// row_scales, has_row_scales)` the `_unchecked` requant entries take. A `PerTensor(s)` must be
/// finite and `> 0`; a `PerRow` slice must have length `m`, every element finite and `> 0`, and
/// must not overlap `C`. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap test; raw
/// pointer math only
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

/// Allocate an `m x n` column-major [`DMatrix`] whose cells are all `zero`. Used only by the
/// `dot`-family convenience wrappers: they call gemm with `beta == 0`, so gemmkit overwrites every
/// element and the fill is never read: it exists solely to hand the engine an initialized buffer.
/// Passing the zero value in (rather than going through `DMatrix::zeros`) keeps the bound at
/// `T: Copy`, so the engine's element types (`f16`/`bf16`, `i32`) need not satisfy nalgebra's own
/// `Scalar`/`Zero`
#[inline]
pub(crate) fn filled_dmatrix<T: Copy>(m: usize, n: usize, zero: T) -> DMatrix<T> {
    DMatrix::from_data(VecStorage::new(Dyn(m), Dyn(n), vec![zero; m * n]))
}
