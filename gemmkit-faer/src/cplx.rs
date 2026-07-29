//! Complex GEMM entries with optional per-operand conjugation
use super::*;
use crate::common::{filled_mat, ref_parts};
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// Complex `C <- alpha*op(A)*op(B) + beta*C`. `op(A)` is `conj(A)` when `conj_a` is true, and
/// `op(B)` is `conj(B)` when `conj_b` is true. `conj_a = conj_b = false` gives the plain product
/// `A*B`. `T` is `Complex<f32>` or `Complex<f64>` (faer's `c32`/`c64`), and this function needs
/// the `complex` feature. Like [`gemm`], it reads the pointer and strides directly, so a
/// transposed, reversed, or general-stride view works without copying
///
/// # Panics
/// If the inner dimensions disagree
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

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// If the inner dimensions disagree
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

    // SAFETY: the dims are validated above. faer guarantees that the pointer and the
    // element-unit `isize` strides describe a valid in-bounds layout. A negative stride marks
    // a reversed view, which gemmkit handles. `c` is a `MatMut` exclusive borrow, so it cannot
    // alias `a` or `b`
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

/// Non-conjugated complex `A*B` into a fresh column-major [`Mat`]: the complex analogue of
/// [`dot`]. For a conjugated product use [`gemm_cplx`] directly. Needs the `complex` feature
#[cfg(feature = "complex")]
pub fn dot_cplx<T: ComplexScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the fill value below is never read
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

/// Complex `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 fused pass. `op(A)` is `conj(A)` when
/// `conj_a` is true, and `op(B)` is `conj(B)` when `conj_b` is true. This is the faer adapter
/// over gemmkit's [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`]
/// (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), added verbatim to every element of
/// that row or column and never conjugated. `bias == None` behaves exactly like [`gemm_cplx`]
///
/// There is no activation parameter, because an ordering activation such as ReLU is undefined
/// on complex numbers, which have no total order. Like [`gemm_cplx`], this function reads the
/// pointer and strides directly and forwards to gemmkit's raw engine. A transposed, sub-matrix,
/// or reversed (negative-stride) view works without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects: a wrong-length
/// `PerRow`/`PerCol` bias, or a bias slice that overlaps `C`
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T: ComplexScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) {
    gemm_cplx_fused_common(None, alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

/// Like [`gemm_cplx_fused`] but reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T: ComplexScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) {
    gemm_cplx_fused_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_fused_common<T: ComplexScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
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
    // Validates the bias the same way gemmkit's checked entry does, with the same panic
    // wording. Its length matches its axis, and it does not overlap C. There is no activation,
    // so no slope check runs here
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: the dims are validated above. faer guarantees that the pointer and the
    // element-unit `isize` strides describe a valid in-bounds layout. A negative stride marks
    // a reversed view, which the raw engine handles. `c` is a `MatMut` exclusive borrow, so it
    // cannot alias `a` or `b`, and the bias was validated disjoint from `C` above
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_fused_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, conj_a, bp, rsb, csb, conj_b, beta, cp, rsc, csc,
                bias_ptr, bias_dim, has_bias, par,
            ),
            None => gemm_cplx_fused_unchecked(
                m, k, n, alpha, ap, rsa, csa, conj_a, bp, rsb, csb, conj_b, beta, cp, rsc, csc,
                bias_ptr, bias_dim, has_bias, par,
            ),
        }
    }
}
