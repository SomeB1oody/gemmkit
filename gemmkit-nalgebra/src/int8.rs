//! `i8 x i8 -> i32` GEMM, plus the fused requantizing entries that collapse an `i32` accumulator
//! straight to a quantized `i8`/`u8` output
use super::*;
use crate::common::{dims_strides, filled_dmatrix};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::adapter::{requant_bias, requant_scale};

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`: the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output, so `alpha`/`beta`/`C` are
/// `i32`; arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry
/// from [`gemm`] because the input (`i8`) and output (`i32`) types differ, which `gemm`'s single
/// `T` can't express. Reads pointers/strides directly off `a`/`b`/`c`, so transposed and
/// general-stride views work without copying
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

/// As [`gemm_i8`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool: useful
/// for a quantized-inference loop that calls this repeatedly
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
    // SAFETY: dims checked above; nalgebra guarantees valid in-bounds layouts; `c` (`&mut i32`)
    // can't alias `a`/`b` (`&i8`): distinct element types can't share the same storage
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

/// `A(i8)*B(i8)` into a fresh column-major `DMatrix<i32>`: the `i8` analogue of [`dot`]
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
    // beta = 0, so gemm_i8 overwrites every cell; the fill value is never read
    let mut c = filled_dmatrix(m, n, 0i32);
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiply into an `i32` accumulator, which is then
/// requantized to an `i8` output in the same pass: the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_i8_requant`]. [`Requantize`] carries the per-tensor or per-row `scale`, an
/// integer `zero_point`, and an optional per-row `i32` bias; there is no separate `alpha` (it folds
/// into `scale`) or `beta` (accumulating into an already-quantized `C` is not meaningful). Reads
/// the pointers/strides directly and forwards to gemmkit's raw engine, so transposed and
/// general-stride views all work without copying
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

/// As [`gemm_i8_requant`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool:
/// useful for a quantized-inference loop that calls this repeatedly
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
    // Checks the scale (finite, positive, and if per-row, length A.rows and disjoint from C), the
    // zero_point range, and the bias (length A.rows, disjoint from C), matching the core checked
    // entry's wording
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims checked above; nalgebra guarantees valid in-bounds layouts; `c` (`&mut i8`)
    // can't alias `a`/`b`, and the bias was checked disjoint from C above
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

/// Requantizing integer GEMM with an **unsigned `u8` output** (the ONNX QLinearMatMul-style
/// activation quantization): the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_i8_requant_u8`]. The `u8`-output twin of [`gemm_i8_requant`], differing only in
/// the output domain (`[0, 255]`) and the matching `zero_point` range
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

/// As [`gemm_i8_requant_u8`], but reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool
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
    // Checks the scale (finite, positive, and if per-row, length A.rows and disjoint from C), the
    // zero_point range, and the bias (length A.rows, disjoint from C), matching the core checked
    // entry's wording
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims checked above; nalgebra guarantees valid in-bounds layouts; `c` (`&mut u8`)
    // can't alias `a`/`b`, and the bias was checked disjoint from C above
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
