//! Shared view-adapter helpers for the other modules: pulling raw parts out of a faer view, and
//! allocating the output buffer the `dot`-family wrappers write into. The bias/requant validation
//! the epilogue-gated entries need lives once in gemmkit's `adapter` module (a raw-pointer-level
//! surface shared with gemmkit's own checked entries), which they import and reuse rather than
//! keeping a local copy
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
