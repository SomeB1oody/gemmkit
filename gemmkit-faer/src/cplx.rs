//! Complex GEMM entries with optional per-operand conjugation
use super::*;
use crate::common::{filled_mat, ref_parts};
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// Complex `C <- alpha*op(A)*op(B) + beta*C`, with `op(A) = conj(A)` when `conj_a` (resp.
/// `conj(B)` when `conj_b`); `conj_a = conj_b = false` is the plain product `A*B`. `T` is
/// `Complex<f32>`/`Complex<f64>` (faer's `c32`/`c64`); needs the `complex` feature. Like [`gemm`],
/// it reads the pointer/strides directly, so transposed, reversed, and general-stride views work
/// without copying
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

/// [`gemm_cplx`], threading a caller-owned [`Workspace`] through instead of the thread-local pool
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

    // SAFETY: dims validated above; faer guarantees the pointer + element-unit `isize` strides
    // describe a valid in-bounds layout (negative for a reversed view, which gemmkit handles), and
    // `c` (a `MatMut` exclusive borrow) can't alias `a`/`b`
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

/// Non-conjugated complex `A*B` into a fresh column-major [`Mat`] (the complex analogue of
/// [`dot`]). For a conjugated product use [`gemm_cplx`] directly. Needs the `complex` feature
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

/// Complex `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 fused pass, with `op(A) = conj(A)` when
/// `conj_a` (resp. `conj(B)` when `conj_b`); the faer adapter over gemmkit's
/// [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added verbatim to every element of that row/column, never
/// conjugated; `bias == None` behaves exactly like [`gemm_cplx`]. There is **no** activation
/// parameter: an ordering activation such as ReLU is undefined on complex numbers, which have no
/// total order. Like [`gemm_cplx`], it reads the pointer/strides directly and forwards to gemmkit's
/// raw engine, so transposed, sub-matrix, and reversed (negative-stride) views all work without
/// copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects (a `PerRow`/`PerCol` bias of
/// the wrong length, or a bias slice overlapping `C`)
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

/// [`gemm_cplx_fused`], threading a caller-owned [`Workspace`] through instead of the thread-local
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
    // Bias validation, matching gemmkit's checked entry (same panic wording): the bias length
    // matches its axis and doesn't overlap C. No activation, so there is no slope check
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: dims validated above; faer guarantees the pointer + element-unit `isize` strides
    // describe a valid in-bounds layout (negative for a reversed view, which the raw engine
    // handles), `c` (a `MatMut` exclusive borrow) can't alias `a`/`b`, and the bias was validated
    // disjoint from C above
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
