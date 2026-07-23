//! Fused bias/activation ndarray GEMM entries
use super::*;
use crate::common::dims_strides;
use gemmkit::adapter::lower_bias;

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 fused pass: the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`) and the optional [`Activation`] is applied last;
/// `bias == None && act == None` is exactly [`gemm`]. `T` is `f32`/`f64` (plus `f16`/`bf16`
/// under `half`, whose epilogue applies in `f32` before the single narrowing). Like [`gemm`], it
/// reads the pointer/strides directly and forwards to gemmkit's raw engine, so C-order, F-order,
/// general-stride, transposed, and reversed (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias/activation the adapter rejects (a
/// `PerRow`/`PerCol` bias of the wrong length, a bias slice overlapping `C`, or a non-finite
/// `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_fused_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
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

    // Bias/activation validation matching gemmkit's checked-entry wording: the bias length
    // matches its axis and does not overlap C (raw pointer math only), and a LeakyRelu slope
    // must be finite
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated above; ndarray guarantees the pointer/strides are in-bounds and `c`
    // (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C above
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
