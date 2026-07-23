//! Shared stride-extraction and buffer-allocation helpers used across the entry modules: every
//! `gemm*` wrapper pulls dims/strides through [`dims_strides`], and the `dot`-family wrappers
//! allocate their output through [`filled_dmatrix`]. The bias/scale validation the epilogue
//! wrappers need lives once in gemmkit's `adapter` module (a raw-pointer-level surface shared with
//! gemmkit's own checked entries), which they import and reuse rather than keeping a local copy
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
