//! i8-input GEMM entries: plain accumulation into i32, and fused requantization into i8/u8
use super::*;
use crate::common::{filled_mat, ref_parts};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::adapter::{requant_bias, requant_scale};

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`. This is the faer adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`, `beta`, and `C`
/// are all `i32`). Arithmetic wraps on overflow, the usual integer-GEMM semantics
///
/// The input and output element types differ, so this needs its own entry rather than riding
/// [`gemm`]. faer's view types are generic over an arbitrary element, so an `i8`/`i32`
/// `MatRef`/`MatMut` pair needs no special handling here. This function reads the pointer and
/// strides directly, so a transposed, reversed, or general-stride view works without copying
///
/// # Panics
/// If the inner dimensions disagree
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

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`] instead of the thread-local pool.
/// The caller-owned workspace avoids a fixed allocation cost in a repeated quantized-inference
/// loop
///
/// # Panics
/// Same conditions as [`gemm_i8`]
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
    // SAFETY: the dims are validated above, and faer guarantees valid in-bounds layouts. `c` is
    // a `MatMut<i32>` exclusive borrow, so it cannot alias `a` or `b` (`MatRef<i8>`). The
    // element types differ, so the 2 views address distinct storage
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

/// `A(i8)*B(i8)` into a fresh column-major `Mat<i32>` (the i8 analogue of [`dot`])
#[cfg(feature = "int8")]
pub fn dot_i8(a: MatRef<'_, i8>, b: MatRef<'_, i8>) -> Mat<i32> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the fill value below is never read
    let mut c = filled_mat(m, n, 0i32);
    gemm_i8(1, a, b, 0, c.as_dyn_stride_mut(), Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then
/// requantized to an `i8` output in 1 fused pass. This is the faer adapter over gemmkit's
/// [`gemmkit::gemm_i8_requant`]. The [`Requantize`] carries the per-tensor or per-row `scale`,
/// the `zero_point`, and an optional per-row `i32` bias. There is no `alpha`, because it folds
/// into `scale`. There is no `beta`, because `C` already holds quantized output, which cannot
/// accumulate further
///
/// This function reads the pointer and strides directly and forwards to gemmkit's raw engine,
/// so a transposed, sub-matrix, or reversed (negative-stride) view works without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a requant parameter the adapter rejects:
///
/// - a non-finite or non-positive `scale`, per-tensor or any per-row element
/// - a per-row scale slice whose length is not `A.rows`, or which overlaps `C`
/// - a `zero_point` outside `[-128, 127]`
/// - a bias whose length is not `A.rows`, or which overlaps `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant(
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, i8>,
    par: Parallelism,
) {
    gemm_i8_requant_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool. The caller-owned workspace avoids a fixed allocation cost in a repeated
/// quantized-inference loop
///
/// # Panics
/// Same conditions as [`gemm_i8_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_with(
    ws: &mut Workspace,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, i8>,
    par: Parallelism,
) {
    gemm_i8_requant_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_common(
    ws: Option<&mut Workspace>,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, i8>,
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
    // Validates the requant parameters the same way gemmkit's checked entry does, with the
    // same panic wording. The scale (per-tensor or per-row) is finite and positive, and a
    // per-row scale has length `A.rows` and does not overlap `C`. The `zero_point` sits inside
    // the i8 band. A per-row bias has length `A.rows` and does not overlap `C`
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: the dims are validated above, and faer guarantees valid in-bounds layouts. `c` is
    // a `MatMut<i8>` exclusive borrow, so it cannot alias `a` or `b`, and the bias was
    // validated disjoint from `C` above. A reversed stride forwards straight through, exactly
    // as the plain entry does
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_unchecked_with(
                ws,
                m,
                k,
                n,
                ap,
                rsa,
                csa,
                bp,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_i8_requant_unchecked(
                m,
                k,
                n,
                ap,
                rsa,
                csa,
                bp,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Requantizing integer GEMM with an unsigned `u8` output (ONNX-QLinearMatMul-style
/// activation). This is the faer adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`], the
/// `u8`-output twin of [`gemm_i8_requant`]. It differs only in the output domain `[0, 255]` and
/// the matching `zero_point` range
///
/// # Panics
/// If the inner dimensions disagree, or on a requant parameter the adapter rejects:
///
/// - a non-finite or non-positive `scale`, per-tensor or any per-row element
/// - a per-row scale slice whose length is not `A.rows`, or which overlaps `C`
/// - a `zero_point` outside `[0, 255]`
/// - a bias whose length is not `A.rows`, or which overlaps `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8(
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, u8>,
    par: Parallelism,
) {
    gemm_i8_requant_u8_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8_with(
    ws: &mut Workspace,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, u8>,
    par: Parallelism,
) {
    gemm_i8_requant_u8_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_u8_common(
    ws: Option<&mut Workspace>,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, u8>,
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
    // Validates the requant parameters the same way gemmkit's checked entry does, with the
    // same panic wording. The scale (per-tensor or per-row) is finite and positive, and a
    // per-row scale has length `A.rows` and does not overlap `C`. The `zero_point` sits inside
    // the u8 band. A per-row bias has length `A.rows` and does not overlap `C`
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: the dims are validated above, and faer guarantees valid in-bounds layouts. `c` is
    // a `MatMut<u8>` exclusive borrow, so it cannot alias `a` or `b`, and the bias was
    // validated disjoint from `C` above. A reversed stride forwards straight through, exactly
    // as the plain entry does
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_u8_unchecked_with(
                ws,
                m,
                k,
                n,
                ap,
                rsa,
                csa,
                bp,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_i8_requant_u8_unchecked(
                m,
                k,
                n,
                ap,
                rsa,
                csa,
                bp,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}
