//! Complex GEMM (`Complex<f32>`/`Complex<f64>`) with optional per-operand conjugation
use super::*;
use crate::common::{dims_strides, filled_dmatrix};
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// Complex `C <- alpha*op(A)*op(B) + beta*C`, where `op(A) = conj(A)` when `conj_a` is set (resp.
/// `op(B) = conj(B)` when `conj_b` is set). `T` is `Complex<f32>`/`Complex<f64>`; needs the
/// `complex` feature. As with [`gemm`], the pointer/strides are read directly off `a`/`b`/`c`, so
/// transposed, row-major, and general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_cplx_common(None, alpha, a, conj_a, b, conj_b, beta, c, par);
}

/// As [`gemm_cplx`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_cplx_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, par);
}

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.shape();
    assert_eq!(k, kb, "gemmkit-nalgebra: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-nalgebra: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-nalgebra: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();

    // SAFETY: dims checked above; nalgebra guarantees the storage's pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) can't alias `a`/`b`
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

/// Non-conjugated complex `A*B` into a fresh column-major [`DMatrix`]: the complex analogue of
/// [`dot`]. For a conjugated product use [`gemm_cplx`] directly. Needs the `complex` feature
#[cfg(feature = "complex")]
pub fn dot_cplx<T, R1, C1, S1, R2, C2, S2>(
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
) -> DMatrix<T>
where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta = 0, so gemm_cplx overwrites every cell; the fill value is never read
    let mut c = filled_dmatrix(m, n, T::ZERO);
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

/// Complex `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 fused pass, where `op(A) = conj(A)` when
/// `conj_a` is set (resp. `op(B) = conj(B)` when `conj_b` is set): the nalgebra adapter over
/// gemmkit's [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length
/// `A.rows`) or [`Bias::PerCol`] (length `B.cols`), added verbatim without conjugation;
/// `bias == None` behaves exactly like [`gemm_cplx`]. There is **no** activation parameter: an
/// ordering activation (`Relu`, `LeakyRelu`) is undefined on complex numbers. As with [`gemm_cplx`],
/// pointer/strides are read directly and forwarded to gemmkit's raw engine, so transposed and
/// general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects (a `PerRow`/`PerCol` bias of
/// the wrong length, or a bias slice overlapping `C`)
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_cplx_fused_common(None, alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

/// As [`gemm_cplx_fused`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_cplx_fused_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_fused_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.shape();
    assert_eq!(k, kb, "gemmkit-nalgebra: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-nalgebra: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-nalgebra: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();
    // Checks the bias length against its axis and rejects an overlap with C, matching the core
    // checked entry's wording; no slope check since complex has no activation parameter
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: dims checked above; nalgebra guarantees the pointer/strides describe a valid
    // in-bounds layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was checked disjoint
    // from C above
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
