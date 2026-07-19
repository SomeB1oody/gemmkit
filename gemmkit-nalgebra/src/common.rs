//! Shared stride-extraction, C-footprint, and epilogue-lowering helpers used across the entry
//! modules: every `gemm*` wrapper pulls dims/strides through [`dims_strides`], and the epilogue
//! wrappers additionally validate a bias/scale slice against C's footprint through these before
//! forwarding to gemmkit's `_unchecked` engine
use super::*;

/// Reads `(rows, cols, row-stride, col-stride)` off a matrix of any storage. nalgebra's `strides()`
/// is `usize` (always non-negative); widen to the `isize` gemmkit's raw pointer API takes
#[inline]
pub(crate) fn dims_strides<T, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    a: &Matrix<T, R, C, S>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.shape();
    let (rs, cs) = a.strides();
    (r, c, rs as isize, cs as isize)
}

/// The half-open byte range `[lo, hi)` spanned by a strided view based at `cp` (element `(0, 0)`),
/// given its `(dim, element-stride)` pairs. Negative strides never occur on nalgebra (`strides()` is
/// `usize`), but the negative-offset accumulation is kept anyway for parity with gemmkit's own
/// range logic; any zero-length axis collapses the range to the empty `(cp, cp)`. Pure pointer
/// arithmetic: nothing here ever dereferences `cp`, which is why the epilogue wrappers can compute
/// this before `C` has necessarily been initialized
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
            continue; // a length-1 axis contributes no span regardless of its stride's sign
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

/// `true` if a `len`-element `TB` slice at `bias` overlaps the byte range `c_dims` describes for the
/// `TC` view at `cp`, via the standard half-open-interval overlap test (`a0 < b1 && b0 < a1`). Used
/// to reject a bias/scale argument that aliases `C` without ever forming a reference over `C`
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

/// Checks a fused `Option<Bias>` against `m`/`n` and `C`'s footprint, and lowers it to the raw
/// `(ptr, BiasDim, has_bias)` triple the `_unchecked` core entries take. Mirrors gemmkit's own
/// `validate_bias` including its panic wording (`PerRow`/`PerCol` length mismatch, or an overlap
/// with `C`), so the checked-tier behavior an adapter caller sees matches the core crate's. `cp`/
/// `c_dims` describe `C` for the [`bias_overlaps_c`] test; `C` itself is never referenced
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

/// Checks a requantize per-row `i32` bias against `A.rows` (`m`) and `C`'s footprint, and lowers it
/// to the raw `(ptr, has_bias)` pair the `_unchecked` requant entries take. Mirrors gemmkit's own
/// `requant_bias`, panic wording included. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the
/// overlap test; `C` itself is never referenced
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

/// Checks a requantize [`RequantScale`] against `A.rows` (`m`) and `C`'s footprint, and lowers it to
/// the raw `(scale, row_scales, has_row_scales)` triple the `_unchecked` requant entries take. A
/// `PerTensor(s)` must be finite and `> 0`; a `PerRow` slice must have length `m`, every element
/// finite and `> 0`, and must not overlap `C`. Mirrors gemmkit's own `requant_scale`, panic wording
/// included. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap test; `C` itself is never
/// referenced
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

/// Allocates an `m x n` column-major [`DMatrix`] filled with `zero`. The `dot`-family wrappers hand
/// this to gemm with `beta == 0`, under which every cell is overwritten and the fill value is never
/// read; it exists only to give the engine an initialized buffer to write into. Taking `zero` as a
/// caller-supplied value (the caller already has it: `T::ZERO` from [`GemmScalar`]/`ComplexScalar`)
/// rather than calling `DMatrix::zeros`, which needs `T: num_traits::Zero`, keeps the bound at the
/// `Copy` `GemmScalar` already implies, with no extra trait to stack onto `dot`'s signature
#[inline]
pub(crate) fn filled_dmatrix<T: Copy>(m: usize, n: usize, zero: T) -> DMatrix<T> {
    DMatrix::from_data(VecStorage::new(Dyn(m), Dyn(n), vec![zero; m * n]))
}
