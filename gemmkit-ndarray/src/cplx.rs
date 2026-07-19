//! Complex ndarray GEMM entries, with an optional conjugation of either operand
use super::*;
use crate::common::dims_strides;
#[cfg(feature = "epilogue")]
use crate::common::lower_bias;

/// Complex `C <- alpha*op(A)*op(B) + beta*C`, with `op(A) = conj(A)` when `conj_a` (resp.
/// `conj(B)` when `conj_b`). `T` is `Complex<f32>`/`Complex<f64>`; needs the `complex` feature.
/// Like [`gemm`], it reads the pointer/strides directly, so transposed, F-order, and
/// negative-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
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

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx`]
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

    // SAFETY: dims validated above; ndarray guarantees the pointer/strides are in-bounds, and
    // `c` (a `&mut` borrow) cannot alias `a`/`b`
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

/// Non-conjugated complex `A*B` into a fresh row-major [`Array2`]: the complex analogue of
/// [`dot`]. For a conjugated product use [`gemm_cplx`]. Needs the `complex` feature
#[cfg(feature = "complex")]
pub fn dot_cplx<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta is 0 here, so the fill value below is never read
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

/// Complex `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 fused pass, with `op(A) = conj(A)` when
/// `conj_a` (resp. `conj(B)` when `conj_b`): the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added verbatim (never conjugated); `bias == None` is
/// exactly [`gemm_cplx`]. There is **no** activation parameter: an ordering activation is
/// undefined on complex numbers. Like [`gemm_cplx`], it reads the pointer/strides directly and
/// forwards to gemmkit's raw engine, so transposed and reversed (negative-stride) views all work
/// without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects (a `PerRow`/`PerCol` bias of
/// the wrong length, or a bias slice overlapping `C`)
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_fused_common(None, alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

/// Like [`gemm_cplx_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_fused_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_fused_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
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
    // Bias validation matching gemmkit's checked-entry wording: the bias length matches its axis
    // and does not overlap C (raw pointer math only). No activation to check: complex has none
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: dims validated above; ndarray guarantees the pointer/strides are in-bounds and `c`
    // (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C above
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_fused_unchecked_with(
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
                bias_ptr,
                bias_dim,
                has_bias,
                par,
            ),
            None => gemm_cplx_fused_unchecked(
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
                bias_ptr,
                bias_dim,
                has_bias,
                par,
            ),
        }
    }
}
