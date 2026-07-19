//! Bias + activation fused into the GEMM store, over gemmkit's `gemm_fused`
use super::*;
use crate::common::dims_strides;
#[cfg(feature = "epilogue")]
use crate::common::lower_bias;

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 fused pass: the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added after the product; the optional [`Activation`] is
/// applied last. `bias == None && act == None` behaves exactly like [`gemm`]. `T` is `f32`/`f64`
/// (plus `f16`/`bf16` under `half`, whose epilogue runs in `f32` before the single narrowing back
/// to the output type). As with [`gemm`], the pointer/strides are read directly and forwarded to
/// gemmkit's raw engine, so column-major, row-major, and general-stride views all work without
/// copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias/activation the adapter rejects (a `PerRow`/`PerCol`
/// bias of the wrong length, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
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
    gemm_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// As [`gemm_fused`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
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
    gemm_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_fused_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
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

    // Checks the bias length against its axis and rejects an overlap with C, then a finite
    // LeakyRelu slope, matching the core checked entry's wording
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims checked above; nalgebra guarantees the pointer/strides describe a valid
    // in-bounds layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was checked disjoint
    // from C above
    unsafe {
        match ws {
            Some(ws) => gemm_fused_unchecked_with(
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
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
            None => gemm_fused_unchecked(
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
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
        }
    }
}
