//! i8-input GEMM entries: plain accumulation into i32, and fused requantization into i8/u8
use super::*;
use crate::common::{filled_mat, ref_parts};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::adapter::{requant_bias, requant_scale};

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`, the faer adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are also
/// `i32`); arithmetic wraps on overflow, the conventional integer-GEMM semantics. Input and output
/// element types differ, so this needs its own entry rather than riding [`gemm`]; faer's view types
/// are generic over an arbitrary element, so an `i8`/`i32` `MatRef`/`MatMut` pair needs no special
/// handling here. Reads pointers/strides directly, so transposed, reversed, and general-stride
/// views all work without copying
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

/// [`gemm_i8`], threading a caller-owned [`Workspace`] through instead of the thread-local pool
/// (the fixed-cost path for a quantized-inference loop)
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
    // SAFETY: dims validated above; faer guarantees valid in-bounds layouts; `c` (a `MatMut<i32>`
    // exclusive borrow) can't alias `a`/`b` (`MatRef<i8>`): distinct element types over distinct
    // storage
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

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then requantized to
/// an `i8` output in 1 fused pass, the faer adapter over gemmkit's [`gemmkit::gemm_i8_requant`]. The
/// [`Requantize`] carries the per-tensor or per-row `scale`, the `zero_point`, and an optional
/// per-row `i32` bias; there is no `alpha` (it folds into `scale`) and no `beta` (accumulating into
/// an already-quantized C is ill-defined). Reads the pointers/strides directly and forwards to
/// gemmkit's raw engine, so transposed, sub-matrix, and reversed (negative-stride) views all work
/// without copying
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale`, whether per-tensor or any per-row element; a per-row scale slice whose
/// length is not `A.rows` or which overlaps `C`; a `zero_point` outside `[-128, 127]`; or a bias
/// whose length is not `A.rows` or which overlaps `C`)
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

/// [`gemm_i8_requant`], threading a caller-owned [`Workspace`] through instead of the thread-local
/// pool (the fixed-cost path for a quantized-inference loop)
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
    // Requantize validation, matching gemmkit's checked entry (same panic wording): a finite,
    // positive per-tensor or per-row scale (per-row length A.rows, disjoint from C); zero_point in
    // the i8 band; a per-row bias of length A.rows, disjoint from C
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated above; faer guarantees valid in-bounds layouts; `c` (a `MatMut<i8>`
    // exclusive borrow) can't alias `a`/`b`, and the bias was validated disjoint from C above
    // Reversed strides forward straight through, exactly as the plain entry
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

/// Requantizing integer GEMM with an **unsigned `u8` output** (ONNX-QLinearMatMul-style
/// activation), the faer adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`]. The `u8`-output
/// twin of [`gemm_i8_requant`], differing only in the output domain `[0, 255]` and the matching
/// `zero_point` range
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale`, whether per-tensor or any per-row element; a per-row scale slice whose
/// length is not `A.rows` or which overlaps `C`; a `zero_point` outside `[0, 255]`; or a bias whose
/// length is not `A.rows` or which overlaps `C`)
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

/// [`gemm_i8_requant_u8`], threading a caller-owned [`Workspace`] through instead of the
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
    // Requantize validation, matching gemmkit's checked entry (same panic wording): a finite,
    // positive per-tensor or per-row scale (per-row length A.rows, disjoint from C); zero_point in
    // the u8 band; a per-row bias of length A.rows, disjoint from C
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated above; faer guarantees valid in-bounds layouts; `c` (a `MatMut<u8>`
    // exclusive borrow) can't alias `a`/`b`, and the bias was validated disjoint from C above
    // Reversed strides forward straight through, exactly as the plain entry
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
