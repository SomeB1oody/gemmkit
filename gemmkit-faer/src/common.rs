//! Shared view-adapter helpers for the other modules. These pull raw parts out of a faer view and
//! allocate the output buffer the `dot`-family wrappers write into
//!
//! The bias and requant validation the epilogue-gated entries need lives once in gemmkit's
//! `adapter` module, a raw-pointer-level surface shared with gemmkit's own checked entries. The
//! epilogue-gated entries import and reuse it instead of keeping a local copy
use super::*;

/// Pull `(rows, cols, row-stride, col-stride, ptr)` out of a [`MatRef`]
///
/// faer already reports strides in element units as `isize`, negative for a reversed view. This
/// matches what gemmkit's raw entries take, so the values pass through unconverted
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

/// Allocate an `m x n` column-major [`Mat`] with every cell set to `zero`
///
/// The `dot`-family wrappers use this as the output buffer for a `beta == 0` call. gemm
/// overwrites every element, so the fill value only needs to exist to initialize the buffer, and
/// is never read back. `Mat::from_fn` carries no numeric trait bound, so this works for element
/// types (`f16`/`bf16`, `i32`) that do not implement faer's own `ComplexField`
#[inline]
pub(crate) fn filled_mat<T: Copy>(m: usize, n: usize, zero: T) -> Mat<T> {
    Mat::from_fn(m, n, |_, _| zero)
}
