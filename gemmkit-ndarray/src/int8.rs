//! Integer (i8 -> i32) and requantizing (i8 -> i8) ndarray GEMM entries
use super::*;
use crate::common::dims_strides;
#[cfg(all(feature = "int8", feature = "epilogue"))]
use crate::common::{requant_bias, requant_scale};

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`, the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ. Reads pointers/strides directly,
/// so transposed / F-order / general-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "int8")]
pub fn gemm_i8<S1, S2, SC>(
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    gemm_i8_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized-inference
/// loop
///
/// # Panics
/// Same conditions as [`gemm_i8`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_with<S1, S2, SC>(
    ws: &mut Workspace,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
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
    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut i32` borrow)
    // can't alias `a`/`b` (`&i8`): different element types over distinct storage
    unsafe {
        match ws {
            Some(ws) => gemm_i8_unchecked_with(
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
                par,
            ),
            None => gemm_i8_unchecked(
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
                par,
            ),
        }
    }
}

/// `A(i8)*B(i8)` into a fresh row-major `Array2<i32>`: the i8 analogue of [`dot`]
#[cfg(feature = "int8")]
pub fn dot_i8<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<i32>
where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read
    let mut c = Array2::<i32>::zeros((m, n));
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then requantized to
/// an `i8` output in 1 pass: the ndarray adapter over gemmkit's [`gemmkit::gemm_i8_requant`]. The
/// [`Requantize`] carries the per-tensor or per-row `scale`, `zero_point`, and an optional per-row
/// `i32` bias; there is no `alpha` (folds into `scale`) or `beta`. Reads the pointers/strides
/// directly and forwards to gemmkit's raw engine, so transposed, general-stride, and reversed
/// (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[-128, 127]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant<S1, S2, SC>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
{
    gemm_i8_requant_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized
/// inference loop
///
/// # Panics
/// Same conditions as [`gemm_i8_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_with<S1, S2, SC>(
    ws: &mut Workspace,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
{
    gemm_i8_requant_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): a
    // finite, positive per-tensor or per-row scale (per-row length A.rows disjoint from C);
    // zero_point in the i8 band; a per-row bias of length A.rows disjoint from C (raw pointer math,
    // C is never referenced)
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut i8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above. Reversed strides
    // forward straight through, exactly as the plain entry
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_unchecked_with(
                ws,
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
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
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
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
/// activation): the ndarray adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`]. The `i8`-output
/// twin of [`gemm_i8_requant`], differing only in the output domain `[0, 255]` and the `zero_point`
/// range
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[0, 255]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8<S1, S2, SC>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
{
    gemm_i8_requant_u8_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8_with<S1, S2, SC>(
    ws: &mut Workspace,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
{
    gemm_i8_requant_u8_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_u8_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): a
    // finite, positive per-tensor or per-row scale (per-row length A.rows disjoint from C);
    // zero_point in the u8 band; a per-row bias of length A.rows disjoint from C (raw pointer math,
    // C is never referenced)
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut u8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above. Reversed strides
    // forward straight through, exactly as the plain entry
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_u8_unchecked_with(
                ws,
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
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
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
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
