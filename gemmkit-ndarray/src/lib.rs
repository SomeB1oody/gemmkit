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
//! The generic [`gemm`]/[`gemm_with`]/[`dot`] are over [`gemmkit::GemmScalar`], so
//! `f32`/`f64` work out of the box and `f16`/`bf16` work once the `half` feature
//! (forwarded to `gemmkit/half`) is enabled. **Complex** (`Complex<f32>`/`Complex<f64>`,
//! with optional conjugation) is served by the dedicated [`gemm_cplx`]/[`gemm_cplx_with`]
//! /[`dot_cplx`] under the `complex` feature — it can't ride the homogeneous surface
//! because it carries conj flags. gemmkit's integer (`i8 -> i32`) path is heterogeneous
//! and has no adapter yet.

#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{GemmScalar, Parallelism, Workspace, gemm_unchecked, gemm_unchecked_with};
use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2};

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
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`.
    unsafe {
        gemm_unchecked(
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
            c.as_mut_ptr(),
            rsc,
            csc,
            par,
        );
    }
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
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);

    // SAFETY: see `gemm`.
    unsafe {
        gemm_unchecked_with(
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
            c.as_mut_ptr(),
            rsc,
            csc,
            par,
        );
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

/// Complex `C <- alpha·op(A)·op(B) + beta·C`, where `op(A) = A̅` if `conj_a` (resp.
/// `B̅` if `conj_b`). `T` is `Complex<f32>`/`Complex<f64>`. Requires the `complex`
/// feature. Like [`gemm`], it reads the pointer/strides straight out of the arrays, so
/// transposed / F-order / negative-stride views work without copying.
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
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`.
    unsafe {
        gemm_cplx_unchecked(
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
            c.as_mut_ptr(),
            rsc,
            csc,
            par,
        );
    }
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
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);

    // SAFETY: see `gemm_cplx`.
    unsafe {
        gemm_cplx_unchecked_with(
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
            c.as_mut_ptr(),
            rsc,
            csc,
            par,
        );
    }
}

/// Non-conjugated complex `A·B` into a fresh row-major [`Array2`] — the complex
/// analogue of [`dot`]. For conjugated products use [`gemm_cplx`]. Requires the
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
