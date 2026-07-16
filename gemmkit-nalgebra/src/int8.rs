//! Integer (`i8` -> `i32`) and requantizing (`i8` -> `i8`) GEMM entries
use super::*;
use crate::common::{dims_strides, filled_dmatrix};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use crate::common::{requant_bias, requant_scale};

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`, the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ. Reads pointers/strides directly,
/// so transposed / general-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "int8")]
pub fn gemm_i8<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
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
pub fn gemm_i8_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
{
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
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
    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut i32` borrow)
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

/// `A(i8)*B(i8)` into a fresh column-major `DMatrix<i32>`: the i8 analogue of [`dot`]
#[cfg(feature = "int8")]
pub fn dot_i8<R1, C1, S1, R2, C2, S2>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
) -> DMatrix<i32>
where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta == 0, so the initial fill is never read
    let mut c = filled_dmatrix(m, n, 0i32);
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then requantized to
/// an `i8` output in 1 pass: the nalgebra adapter over gemmkit's [`gemmkit::gemm_i8_requant`]. The
/// [`Requantize`] carries the per-tensor or per-row `scale`, `zero_point`, and an optional per-row `i32` bias;
/// there is no `alpha` (folds into `scale`) or `beta`. Reads the pointers/strides directly and
/// forwards to gemmkit's raw engine, so transposed and general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[-128, 127]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
{
    gemm_i8_requant_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized
/// inference loop
///
/// # Panics
/// Same conditions as [`gemm_i8_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
{
    gemm_i8_requant_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
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

    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut i8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above
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
/// activation): the nalgebra adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`]. The `u8`-output
/// twin of [`gemm_i8_requant`], differing only in the output domain `[0, 255]` and the `zero_point`
/// range
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[0, 255]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
{
    gemm_i8_requant_u8_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
{
    gemm_i8_requant_u8_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_u8_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
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

    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut u8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above
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
