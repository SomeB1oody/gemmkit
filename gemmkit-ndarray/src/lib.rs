//! # gemmkit-ndarray
//!
//! A thin [`ndarray`] adapter over the [`gemmkit`] GEMM engine. It accepts
//! `&ArrayBase<S, Ix2>` for any storage `S: Data` (so both `ArrayView2` and
//! `&Array2` work), pulls the pointer and strides straight out of the array, and
//! forwards to gemmkit's raw engine — so C-order, F-order, general-stride, and
//! reversed (negative-stride) views all work without copying.
//!
//! ```
//! use ndarray::array;
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//! let c = gemmkit_ndarray::dot(&a, &b);
//! assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! always, plus `f16`/`bf16` under the `half` feature. [`gemm_batched`]/[`gemm_batched_with`]/
//! [`dot_batched`] extend the same idea to a stack of matrices — a 3-D array with the batch on
//! axis 0 — and [`prepack_rhs`]/[`prepack_lhs`] (with their [`gemm_packed_b`]/[`gemm_packed_a`]
//! consumers) pre-pack one reused operand for the fixed-weight loop. Complex
//! (`Complex<f32>`/`Complex<f64>`, with optional conjugation) needs the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since the
//! conj flags don't fit the homogeneous surface. The integer (`i8 -> i32`) path likewise gets its
//! own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature (`i8` inputs, `i32` output).

#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmScalar, Parallelism, Workspace, gemm_batched_unchecked, gemm_batched_unchecked_with,
    gemm_packed_a_unchecked, gemm_packed_a_unchecked_with, gemm_packed_b_unchecked,
    gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with, prepack_lhs_unchecked,
    prepack_rhs_unchecked,
};
/// The prepacked-operand handles, re-exported so callers of [`prepack_rhs`] / [`prepack_lhs`] need
/// not depend on `gemmkit` directly.
pub use gemmkit::{PackedLhs, PackedRhs};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};
use ndarray::{Array2, Array3, ArrayBase, Data, DataMut, Ix2, Ix3};

#[inline]
fn dims_strides<T, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}

/// `C <- alpha·A·B + beta·C`.
///
/// # Panics
/// If the inner dimensions disagree.
pub fn gemm<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// If the inner dimensions disagree.
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`.
    unsafe {
        match ws {
            Some(ws) => gemm_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// `A·B` into a fresh row-major [`Array2`] — the `.dot()`-style convenience.
pub fn dot<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read.
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

#[inline]
fn dims_strides3<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix3>,
) -> (usize, usize, usize, isize, isize, isize) {
    let (b, r, c) = a.dim();
    let s = a.strides();
    (b, r, c, s[0], s[1], s[2])
}

/// Strided-batched `C_e <- alpha·A_e·B_e + beta·C_e`, batch on **axis 0**: `a` is `(batch, m, k)`,
/// `b` is `(batch, k, n)`, `c` is `(batch, m, n)`. The axis-0 stride is each operand's batch stride
/// and axes 1/2 the element strides — read directly, so C-order, F-order, and general-stride 3-D
/// views all work without copying. Parallelizes across the batch (each element serial on one
/// worker), so the result reproduces a loop of [`gemm`] calls.
///
/// # Panics
/// If the batch sizes or inner dimensions disagree.
pub fn gemm_batched<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_batched`] but reuses a caller-owned [`Workspace`] — zero heap allocation after the
/// first sufficiently large call, for a stream of batched products.
///
/// # Panics
/// Same conditions as [`gemm_batched`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_batched_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (batch, m, k, as0, as1, as2) = dims_strides3(a);
    let (bb, kb, n, bs0, bs1, bs2) = dims_strides3(b);
    let (cb, cm, cn, cs0, cs1, cs2) = dims_strides3(c);
    assert_eq!(
        batch, bb,
        "gemmkit-ndarray: A batch ({batch}) != B batch ({bb})"
    );
    assert_eq!(
        batch, cb,
        "gemmkit-ndarray: A batch ({batch}) != C batch ({cb})"
    );
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cp = c.as_mut_ptr();

    // SAFETY: ndarray guarantees each element's pointer/strides describe a valid in-bounds layout;
    // `c` (a `&mut` borrow) can't alias `a`/`b`, and its batch elements — distinct axis-0 slices of
    // a real array — are pairwise disjoint.
    unsafe {
        match ws {
            Some(ws) => gemm_batched_unchecked_with(
                ws,
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                par,
            ),
            None => gemm_batched_unchecked(
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                par,
            ),
        }
    }
}

/// `A_e · B_e` for each batch element into a fresh row-major `(batch, m, n)` [`Array3`] — the
/// batched analogue of [`dot`].
pub fn dot_batched<T, S1, S2>(a: &ArrayBase<S1, Ix3>, b: &ArrayBase<S2, Ix3>) -> Array3<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (batch, m, _) = a.dim();
    let (_, _, n) = b.dim();
    // beta == 0, so the initial fill is never read.
    let mut c = Array3::from_elem((batch, m, n), T::ZERO);
    gemm_batched(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

/// Pre-pack a 2-D RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack
/// once here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this
/// `B`. Reads B's pointer/strides directly, so any layout works without copying.
pub fn prepack_rhs<T, S>(b: &ArrayBase<S, Ix2>) -> PackedRhs<T>
where
    T: GemmScalar,
    S: Data<Elem = T>,
{
    let (k, n, rsb, csb) = dims_strides(b);
    // SAFETY: ndarray guarantees B's pointer/strides describe a valid in-bounds layout.
    unsafe { prepack_rhs_unchecked(b.as_ptr(), rsb, csb, k, n) }
}

/// `C <- alpha·A·B + beta·C` reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|col stride| >= |row stride|`) — a row-major `C` would swap A/B and invalidate
/// the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout.
///
/// # Panics
/// If the dimensions disagree, or if `C` is not column-major-ish.
pub fn gemm_packed_b<T, S1, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_common(None, alpha, a, packed, beta, c, par);
}

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`] — the fixed-cost inference loop.
///
/// # Panics
/// Same conditions as [`gemm_packed_b`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_with<T, S1, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_common(Some(ws), alpha, a, packed, beta, c, par);
}

fn gemm_packed_b_common<T, S1, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (cm, cn) = c.dim();
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-ndarray: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-ndarray: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();
    // SAFETY: ndarray guarantees A/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
    // A, and the prepacked B is a separate owned buffer.
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_unchecked_with(
                ws,
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_packed_b_unchecked(
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Pre-pack a 2-D LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of right
/// operands): pack once, then skip the per-call repack across many [`gemm_packed_a`] calls. Reads
/// A's pointer/strides directly, so any layout works without copying.
pub fn prepack_lhs<T, S>(a: &ArrayBase<S, Ix2>) -> PackedLhs<T>
where
    T: GemmScalar,
    S: Data<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    // SAFETY: ndarray guarantees A's pointer/strides describe a valid in-bounds layout.
    unsafe { prepack_lhs_unchecked(a.as_ptr(), rsa, csa, m, k) }
}

/// `C <- alpha·A·B + beta·C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be row-major-ish
/// (`|col stride| <= |row stride|`) — a column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout.
///
/// # Panics
/// If the dimensions disagree, or if `C` is not row-major-ish.
pub fn gemm_packed_a<T, S2, SC>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_common(None, alpha, packed, b, beta, c, par);
}

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_packed_a`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_with<T, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_common(Some(ws), alpha, packed, b, beta, c, par);
}

fn gemm_packed_a_common<T, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-ndarray: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-ndarray: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();
    // SAFETY: ndarray guarantees B/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
    // B, and the prepacked A is a separate owned buffer.
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_unchecked_with(
                ws,
                alpha,
                packed,
                n,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_packed_a_unchecked(
                alpha,
                packed,
                n,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Integer `C(i32) <- alpha·A(i8)·B(i8) + beta·C`, the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ. Reads pointers/strides directly,
/// so transposed / F-order / general-stride views work without copying.
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "int8")]
pub fn gemm_i8<S1, S2, SC>(
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    gemm_i8_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`] — the fixed-cost quantized-inference
/// loop.
///
/// # Panics
/// Same conditions as [`gemm_i8`].
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_with<S1, S2, SC>(
    ws: &mut Workspace,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();
    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut i32` borrow)
    // can't alias `a`/`b` (`&i8`) — different element types over distinct storage.
    unsafe {
        match ws {
            Some(ws) => gemm_i8_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_i8_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// `A(i8)·B(i8)` into a fresh row-major `Array2<i32>` — the i8 analogue of [`dot`].
#[cfg(feature = "int8")]
pub fn dot_i8<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<i32>
where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read.
    let mut c = Array2::<i32>::zeros((m, n));
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Complex `C <- alpha·op(A)·op(B) + beta·C`, with `op(A) = A̅` when `conj_a` (resp.
/// `B̅` when `conj_b`). `T` is `Complex<f32>`/`Complex<f64>`; needs the `complex`
/// feature. Like [`gemm`], it reads pointer/strides directly, so transposed, F-order,
/// and negative-stride views work without copying.
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_common(None, alpha, a, conj_a, b, conj_b, beta, c, par);
}

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// If the inner dimensions disagree.
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, par);
}

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`.
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_cplx_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Non-conjugated complex `A·B` into a fresh row-major [`Array2`] — the complex
/// analogue of [`dot`]. For conjugated products use [`gemm_cplx`]. Needs the
/// `complex` feature.
#[cfg(feature = "complex")]
pub fn dot_cplx<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read.
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemm_cplx(
        T::ONE,
        a,
        false,
        b,
        false,
        T::ZERO,
        &mut c,
        Parallelism::default(),
    );
    c
}
