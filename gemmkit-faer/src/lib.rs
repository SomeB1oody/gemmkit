//! # gemmkit-faer
//!
//! A thin [`faer`] adapter over the [`gemmkit`] GEMM engine. It accepts faer's
//! vocabulary view types — [`MatRef<'_, T>`](faer::MatRef) for inputs and
//! [`MatMut<'_, T>`](faer::MatMut) for the output — pulls the data pointer and the
//! element-unit `isize` row/column strides straight out of the view, and forwards to
//! gemmkit's raw engine. faer's natural column-major layout, transposed views,
//! sub-matrices, and reversed (negative-stride) views therefore all work without copying.
//!
//! ```
//! use faer::Mat;
//! let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
//! let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
//! let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
//! assert_eq!(c[(0, 0)], 19.0);
//! assert_eq!(c[(1, 1)], 50.0);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! always, plus `f16`/`bf16` under the `half` feature. [`prepack_rhs`]/[`prepack_lhs`]
//! (with their [`gemm_packed_b`]/[`gemm_packed_a`] consumers) pre-pack one reused operand
//! for the fixed-weight loop. Complex (`Complex<f32>`/`Complex<f64>` — i.e. faer's
//! `c32`/`c64`, with optional conjugation) needs the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since the conj
//! flags don't fit the homogeneous surface. The integer (`i8 -> i32`) path likewise gets its
//! own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature (`i8` inputs, `i32`
//! output).
//!
//! faer has no 3-D array / batch type, so the batched (`gemm_batched`) entries of the ndarray
//! adapter have no analogue here.

use faer::{Mat, MatMut, MatRef};
#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmScalar, Parallelism, Workspace, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_b_unchecked, gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with,
    prepack_lhs_unchecked, prepack_rhs_unchecked,
};
/// The prepacked-operand handles, re-exported so callers of [`prepack_rhs`] / [`prepack_lhs`] need
/// not depend on `gemmkit` directly.
pub use gemmkit::{PackedLhs, PackedRhs};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};

/// Pull `(rows, cols, row-stride, col-stride, ptr)` out of a [`MatRef`]. faer reports strides in
/// element units as `isize` (negative for a reversed view) — exactly what gemmkit's raw engine
/// takes — so no conversion is needed.
#[inline]
fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (
        a.nrows(),
        a.ncols(),
        a.row_stride(),
        a.col_stride(),
        a.as_ptr(),
    )
}

/// Allocate an `m×n` column-major [`Mat`] whose cells are all `zero`. Used only by the `dot`-family
/// convenience wrappers: they call gemm with `beta == 0`, so gemmkit overwrites every element and
/// the fill is never read — it exists solely to hand the engine an initialized buffer. `Mat::from_fn`
/// carries no numeric trait bound, so the engine's element types (`f16`/`bf16`, `i32`) need not
/// satisfy faer's own `ComplexField`.
#[inline]
fn filled_mat<T: Copy>(m: usize, n: usize, zero: T) -> Mat<T> {
    Mat::from_fn(m, n, |_, _| zero)
}

/// `C <- alpha·A·B + beta·C`.
///
/// # Panics
/// If the inner dimensions disagree.
pub fn gemm<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// If the inner dimensions disagree.
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(k, kb, "gemmkit-faer: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();

    // SAFETY: dims validated; faer's `MatRef`/`MatMut` guarantee the pointer + element-unit `isize`
    // strides describe a valid in-bounds layout (possibly negative for a reversed view, which
    // gemmkit's unchecked path handles). `c` is a `MatMut` — an exclusive borrow — so C cannot alias
    // A/B.
    unsafe {
        match ws {
            Some(ws) => gemm_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
            None => gemm_unchecked(
                m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
        }
    }
}

/// `A·B` into a fresh column-major [`Mat`] — the `.dot()`-style convenience.
pub fn dot<T: GemmScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the initial fill is never read.
    let mut c = filled_mat(m, n, T::ZERO);
    gemm(
        T::ONE,
        a,
        b,
        T::ZERO,
        c.as_dyn_stride_mut(),
        Parallelism::default(),
    );
    c
}

/// Pre-pack a RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack once
/// here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this `B`.
/// Reads B's pointer/strides directly, so any layout works without copying.
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    let (k, n, rsb, csb, bp) = ref_parts(b);
    // SAFETY: faer guarantees B's pointer/strides describe a valid in-bounds layout.
    unsafe { prepack_rhs_unchecked(bp, rsb, csb, k, n) }
}

/// `C <- alpha·A·B + beta·C` reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|col stride| >= |row stride|`) — a row-major `C` would swap A/B and invalidate
/// the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout.
///
/// # Panics
/// If the dimensions disagree, or if `C` is not column-major-ish.
pub fn gemm_packed_b<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_b_common(None, alpha, a, packed, beta, c, par);
}

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`] — the fixed-cost inference loop.
///
/// # Panics
/// Same conditions as [`gemm_packed_b`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_b_common(Some(ws), alpha, a, packed, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-faer: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-faer: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();
    // SAFETY: faer guarantees A/C layouts are valid in-bounds; `c` (a `MatMut` exclusive borrow)
    // can't alias A, and the prepacked B is a separate owned buffer.
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_unchecked_with(
                ws, alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, par,
            ),
            None => {
                gemm_packed_b_unchecked(alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, par)
            }
        }
    }
}

/// Pre-pack an LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of right
/// operands): pack once, then skip the per-call repack across many [`gemm_packed_a`] calls. Reads
/// A's pointer/strides directly, so any layout works without copying.
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    // SAFETY: faer guarantees A's pointer/strides describe a valid in-bounds layout.
    unsafe { prepack_lhs_unchecked(ap, rsa, csa, m, k) }
}

/// `C <- alpha·A·B + beta·C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be row-major-ish
/// (`|col stride| <= |row stride|`) — a column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout.
///
/// # Panics
/// If the dimensions disagree, or if `C` is not row-major-ish.
pub fn gemm_packed_a<T: GemmScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_a_common(None, alpha, packed, b, beta, c, par);
}

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_packed_a`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_a_common(Some(ws), alpha, packed, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-faer: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-faer: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();
    // SAFETY: faer guarantees B/C layouts are valid in-bounds; `c` (a `MatMut` exclusive borrow)
    // can't alias B, and the prepacked A is a separate owned buffer.
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_unchecked_with(
                ws, alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
            None => {
                gemm_packed_a_unchecked(alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, par)
            }
        }
    }
}

/// Integer `C(i32) <- alpha·A(i8)·B(i8) + beta·C`, the faer adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ — faer's view types are generic
/// over an arbitrary element, so an `i8`/`i32` `MatRef`/`MatMut` needs no special handling. Reads
/// pointers/strides directly, so transposed / reversed / general-stride views work without copying.
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "int8")]
pub fn gemm_i8(
    alpha: i32,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    gemm_i8_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`] — the fixed-cost quantized-inference
/// loop.
///
/// # Panics
/// Same conditions as [`gemm_i8`].
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_with(
    ws: &mut Workspace,
    alpha: i32,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(k, kb, "gemmkit-faer: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();
    // SAFETY: dims validated; faer guarantees valid in-bounds layouts; `c` (a `MatMut<i32>`
    // exclusive borrow) can't alias `a`/`b` (`&i8`) — different element types over distinct storage.
    unsafe {
        match ws {
            Some(ws) => gemm_i8_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
            None => gemm_i8_unchecked(
                m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
        }
    }
}

/// `A(i8)·B(i8)` into a fresh column-major `Mat<i32>` — the i8 analogue of [`dot`].
#[cfg(feature = "int8")]
pub fn dot_i8(a: MatRef<'_, i8>, b: MatRef<'_, i8>) -> Mat<i32> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the initial fill is never read.
    let mut c = filled_mat(m, n, 0i32);
    gemm_i8(1, a, b, 0, c.as_dyn_stride_mut(), Parallelism::default());
    c
}

/// Complex `C <- alpha·op(A)·op(B) + beta·C`, with `op(A) = A̅` when `conj_a` (resp.
/// `B̅` when `conj_b`). `T` is `Complex<f32>`/`Complex<f64>` (faer's `c32`/`c64`); needs the
/// `complex` feature. Like [`gemm`], it reads pointer/strides directly, so transposed, reversed,
/// and general-stride views work without copying.
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T: ComplexScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_cplx_common(None, alpha, a, conj_a, b, conj_b, beta, c, par);
}

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T: ComplexScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_cplx_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, par);
}

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_common<T: ComplexScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(k, kb, "gemmkit-faer: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();

    // SAFETY: dims validated; faer guarantees the pointer/strides describe a valid in-bounds layout
    // (element-unit `isize`, negative for reversed views — gemmkit handles that), and `c` (a
    // `MatMut` exclusive borrow) cannot alias `a`/`b`.
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, conj_a, bp, rsb, csb, conj_b, beta, cp, rsc, csc,
                par,
            ),
            None => gemm_cplx_unchecked(
                m, k, n, alpha, ap, rsa, csa, conj_a, bp, rsb, csb, conj_b, beta, cp, rsc, csc, par,
            ),
        }
    }
}

/// Non-conjugated complex `A·B` into a fresh column-major [`Mat`] — the complex
/// analogue of [`dot`]. For conjugated products use [`gemm_cplx`]. Needs the
/// `complex` feature.
#[cfg(feature = "complex")]
pub fn dot_cplx<T: ComplexScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the initial fill is never read.
    let mut c = filled_mat(m, n, T::ZERO);
    gemm_cplx(
        T::ONE,
        a,
        false,
        b,
        false,
        T::ZERO,
        c.as_dyn_stride_mut(),
        Parallelism::default(),
    );
    c
}
