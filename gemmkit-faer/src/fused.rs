//! Fused bias/activation GEMM entries
use super::*;
use crate::common::ref_parts;
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 fused pass, the faer adapter over gemmkit's
/// [`gemmkit::gemm_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added before the optional [`Activation`], which runs last;
/// `bias == None && act == None` behaves exactly like [`gemm`]. `T` is `f32`/`f64`, plus `f16`/`bf16`
/// under `half` (whose epilogue runs in `f32`, narrowing to the output type only once, after the
/// activation). Like [`gemm`], it reads the pointer/strides directly and forwards to gemmkit's raw
/// engine, so transposed, sub-matrix, and reversed (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias/activation the adapter rejects (a `PerRow`/`PerCol`
/// bias of the wrong length, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T: FusedScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// [`gemm_fused`], threading a caller-owned [`Workspace`] through instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_fused_common<T: FusedScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
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

    // Bias/activation validation, matching gemmkit's checked entry (same panic wording): the bias
    // length matches its axis and doesn't overlap C, and a LeakyRelu slope is finite
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated above; faer guarantees the pointer + element-unit `isize` strides
    // describe a valid in-bounds layout (negative for a reversed view, which the raw engine
    // handles), `c` (a `MatMut` exclusive borrow) can't alias `a`/`b`, and the bias was validated
    // disjoint from C above
    unsafe {
        match ws {
            Some(ws) => gemm_fused_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, bias_ptr,
                bias_dim, has_bias, act, par,
            ),
            None => gemm_fused_unchecked(
                m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, bias_ptr, bias_dim,
                has_bias, act, par,
            ),
        }
    }
}
