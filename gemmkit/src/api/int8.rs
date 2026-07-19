//! Integer (`i8` -> `i32`) and requantizing (`i8` -> `i8`/`u8`) GEMM entries: the
//! heterogeneous-output counterpart of the plain `gemm` surface, needed because
//! `i8 -> i32` and `i8 -> i8`/`u8` cannot be expressed through a single `T` type param
use super::*;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;

/// Integer GEMM: `C <- alpha*A*B + beta*C` with `i8` inputs accumulated into an `i32`
/// output (`alpha`, `beta`, `C` are `i32`). Wraps on overflow, the standard integer-GEMM
/// convention. Uses the thread-local workspace pool
///
/// A separate entry point from [`gemm`] because the input/output types differ (`i8` vs
/// `i32`), which the homogeneous `gemm<T>` surface cannot express
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm`]: `A.cols == B.rows`,
/// `A.rows == C.rows`, `B.cols == C.cols`, every view in bounds, and `C` addresses each
/// element uniquely without overlapping `A`/`B`. For negative strides or raw pointers use
/// [`gemm_i8_unchecked`] ([`gemm_unchecked`] is homogeneous and cannot serve `i8 -> i32`)
#[cfg(feature = "int8")]
pub fn gemm_i8(
    alpha: i32,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_i8_with(ws, alpha, a, b, beta, c, par));
}

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_i8`]
#[cfg(feature = "int8")]
pub fn gemm_i8_with(
    ws: &mut Workspace,
    alpha: i32,
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    validate_gemm_views(&a, &b, &c);

    // SAFETY: validated above, shapes agree, every stride in bounds, C addresses
    // each (i,j) uniquely and does not overlap A/B
    unsafe {
        dispatch::execute_int(
            dispatch::IntTask {
                m: a.rows,
                k: a.cols,
                n: b.cols,
                alpha,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                b: b.data.as_ptr(),
                rsb: b.rs,
                csb: b.cs,
                beta,
                c: c.data.as_mut_ptr(),
                rsc: c.rs,
                csc: c.cs,
            },
            par,
            ws,
        );
    }
}

/// `C(i32) <- alpha*A(i8)*B(i8) + beta*C` over raw pointers and `isize` element strides,
/// with no bounds, alias, or shape checks: the `i8 -> i32` escape hatch for negative
/// strides or raw-pointer callers, since [`gemm_unchecked`] is typed for the homogeneous
/// surface and cannot express a differing output type. Uses the thread-local workspace pool
///
/// # Safety
/// `a`/`b` valid for reads and `c` valid for read+write over every `(i,j)` implied by the
/// dimensions and strides; `c` does not alias `a`/`b`; when `beta == 0`, `c` need not be
/// initialized
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_unchecked(
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    unsafe {
        workspace::with_thread_pool(|ws| {
            dispatch::execute_int(
                dispatch::IntTask {
                    m,
                    k,
                    n,
                    alpha,
                    a,
                    rsa,
                    csa,
                    b,
                    rsb,
                    csb,
                    beta,
                    c,
                    rsc,
                    csc,
                },
                par,
                ws,
            );
        });
    }
}

/// Like [`gemm_i8_unchecked`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
///
/// # Safety
/// See [`gemm_i8_unchecked`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_unchecked_with(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: the caller guarantees `a`/`b` valid for reads and `c` for read+write over the
    // shape/strides, `c` not aliasing `a`/`b`, and `beta == 0` may leave `c` uninitialized
    unsafe {
        dispatch::execute_int(
            dispatch::IntTask {
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                beta,
                c,
                rsc,
                csc,
            },
            par,
            ws,
        );
    }
}

/// The output scale for the requantizing entries: either 1 value applied to the whole tensor,
/// or 1 value per output row/channel (the per-channel quantized-inference convention). Every
/// scale must be finite and `> 0`
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub enum RequantScale<'a> {
    /// 1 scale applied to every output element (`alpha` folds into this)
    PerTensor(f32),
    /// 1 scale per output row / channel (length `A.rows == m`), the standard per-channel
    /// quantized-inference convention: `C[i,j]` is scaled by `scale_i`
    PerRow(&'a [f32]),
}

/// The quantization parameters for the requantizing entries: [`RequantScale`], an integer
/// `zero_point`, and an optional per-row `i32` bias (length `m`, the standard qlinear layer
/// bias). The output is `C[i,j] = clamp(zero_point + round_ne(scale*(sum_k A*B + bias[i])),
/// LO, HI)` with round-half-to-even, where `scale` is the per-tensor value or the per-row
/// `scale_i`, and `[LO, HI]` is set by the entry: `[-128, 127]` for [`gemm_i8_requant`],
/// `[0, 255]` for [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub struct Requantize<'a> {
    /// Output scale, per-tensor or per-row/channel; every value must be finite and `> 0`
    pub scale: RequantScale<'a>,
    /// Output zero-point, added after rounding. Must lie in the output domain of the chosen
    /// entry: `[-128, 127]` for [`gemm_i8_requant`], `[0, 255]` for [`gemm_i8_requant_u8`]
    pub zero_point: i32,
    /// Optional per-row `i32` bias (length `m`), added to the accumulator before scaling
    pub bias: Option<&'a [i32]>,
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then
/// requantized to an `i8` output in 1 pass, skipping the full `m x n` `i32` materialization
/// a separate [`gemm_i8`] followed by a requantize pass would need. No `alpha` (it folds into
/// `scale`) and no `beta` (accumulating into an already-quantized `C` is not well-defined).
/// Uses the thread-local workspace pool
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm_i8`], plus: a non-finite or
/// non-positive `scale` (per-tensor or any per-row element); a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`; a `zero_point` outside `[-128, 127]`; or a bias whose
/// length is not `A.rows` or which overlaps `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant(
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, i8>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_i8_requant_with(ws, a, b, req, c, par));
}

/// Bias validation shared by the `i8`- and `u8`-output requantizing entries: checks length
/// against `a_rows` and checks the byte ranges of `bias` and `C` for overlap (`TO` is 1 byte
/// either way), then lowers to the `(ptr, has_bias)` pair the dispatch task carries. `scale`
/// and `zero_point` are checked separately in each entry since the zero-point band differs
/// between `i8` and `u8`
#[cfg(all(feature = "int8", feature = "epilogue"))]
fn requant_bias<TO>(a_rows: usize, c: &MatMut<'_, TO>, bias: Option<&[i32]>) -> (*const i32, bool) {
    match bias {
        Some(bias) => {
            assert_eq!(
                bias.len(),
                a_rows,
                "gemmkit: requantize bias length ({}) != A.rows ({})",
                bias.len(),
                a_rows
            );
            // bias (i32) must not overlap C (a 1-byte quantized output)
            if overlaps_bytes(
                c.data.as_ptr() as *const u8,
                c.data.len(),
                core::mem::size_of::<TO>(),
                bias.as_ptr() as *const u8,
                bias.len(),
                4,
            ) {
                panic!("gemmkit: requantize bias overlaps C");
            }
            (bias.as_ptr(), true)
        }
        None => (core::ptr::null(), false),
    }
}

/// Scale validation shared by the `i8`- and `u8`-output requantizing entries: a
/// [`RequantScale::PerTensor`] must be finite and `> 0`; a [`RequantScale::PerRow`] must have
/// length `a_rows`, must not overlap `C`'s byte range, and every element must be finite and
/// `> 0`. Lowers to the `(scale, row_scales_ptr, has_row_scales)` triple the dispatch task
/// carries: `PerTensor(s) -> (s, null, false)`, `PerRow(p) -> (0.0, p, true)`
#[cfg(all(feature = "int8", feature = "epilogue"))]
fn requant_scale<TO>(
    a_rows: usize,
    c: &MatMut<'_, TO>,
    scale: RequantScale<'_>,
) -> (f32, *const f32, bool) {
    match scale {
        RequantScale::PerTensor(s) => {
            assert!(
                s.is_finite() && s > 0.0,
                "gemmkit: requantize scale ({s}) must be finite and > 0"
            );
            (s, core::ptr::null(), false)
        }
        RequantScale::PerRow(scales) => {
            assert_eq!(
                scales.len(),
                a_rows,
                "gemmkit: requantize scales length ({}) != A.rows ({})",
                scales.len(),
                a_rows
            );
            // Overlap checked before the per-element finite/> 0 loop below, so an
            // aliasing scales slice panics before any element is read
            if overlaps_bytes(
                c.data.as_ptr() as *const u8,
                c.data.len(),
                core::mem::size_of::<TO>(),
                scales.as_ptr() as *const u8,
                scales.len(),
                4,
            ) {
                panic!("gemmkit: requantize scales overlap C");
            }
            for &s in scales {
                assert!(
                    s.is_finite() && s > 0.0,
                    "gemmkit: requantize scale ({s}) must be finite and > 0"
                );
            }
            (0.0, scales.as_ptr(), true)
        }
    }
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
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
    validate_gemm_views(&a, &b, &c);

    let (scale, row_scales, has_row_scales) = requant_scale(a.rows, &c, req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );

    let (bias_ptr, has_bias) = requant_bias(a.rows, &c, req.bias);

    // SAFETY: validated above, shapes agree, strides in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B or the bias/scales; scale/zp are in range. The bias
    // and row-scale slices (borrowed for this call) outlive the `execute_int_requant` frame
    unsafe {
        dispatch::execute_int_requant(
            dispatch::RequantTask {
                m: a.rows,
                k: a.cols,
                n: b.cols,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                b: b.data.as_ptr(),
                rsb: b.rs,
                csb: b.cs,
                c: c.data.as_mut_ptr(),
                rsc: c.rs,
                csc: c.cs,
                scale,
                row_scales,
                has_row_scales,
                zp: req.zero_point,
                bias: bias_ptr,
                has_bias,
                bias_dim: BiasDim::PerRow,
            },
            par,
            ws,
        );
    }
}

/// `C(i8) <- clamp(zp + round_ne(scale*(A*B + bias)), -128, 127)` over raw pointers and
/// `isize` element strides, with no bounds, alias, or shape checks. `bias` is a per-row `i32`
/// pointer, read only when `has_bias`. The scale is the scalar `scale` unless `has_row_scales`
/// is set, in which case `row_scales` supplies 1 `f32` per output row (length `m`) instead.
/// Uses the thread-local workspace pool
///
/// # Safety
/// `a`/`b` valid for reads and `c` valid for writes over the shape/strides; `c` does not alias
/// `a`/`b`; when `has_bias`, `bias` is valid for `m` reads and disjoint from `c`; when
/// `has_row_scales`, `row_scales` is valid for `m` reads and disjoint from `c` (otherwise
/// `row_scales` may be null or dangling); every applied scale is finite and `> 0`, and
/// `zero_point` is in `[-128, 127]` (the checked API enforces the last 2)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_requant_unchecked(
    m: usize,
    k: usize,
    n: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    scale: f32,
    row_scales: *const f32,
    has_row_scales: bool,
    zero_point: i32,
    bias: *const i32,
    has_bias: bool,
    c: *mut i8,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_i8_requant_unchecked_with(
                ws,
                m,
                k,
                n,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                zero_point,
                bias,
                has_bias,
                c,
                rsc,
                csc,
                par,
            );
        });
    }
}

/// Like [`gemm_i8_requant_unchecked`] but reuses a caller-owned [`Workspace`] instead of
/// the thread-local pool
///
/// # Safety
/// See [`gemm_i8_requant_unchecked`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_requant_unchecked_with(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    scale: f32,
    row_scales: *const f32,
    has_row_scales: bool,
    zero_point: i32,
    bias: *const i32,
    has_bias: bool,
    c: *mut i8,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: the caller guarantees `a`/`b` valid for reads and `c` for writes over the
    // shape/strides, `c` not aliasing `a`/`b`, when `has_bias` a valid disjoint `m`-length bias,
    // when `has_row_scales` a valid disjoint `m`-length scale vector, and `scale`/`zero_point`
    // in range (see [`gemm_i8_requant_unchecked`])
    unsafe {
        dispatch::execute_int_requant(
            dispatch::RequantTask {
                m,
                k,
                n,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                c,
                rsc,
                csc,
                scale,
                row_scales,
                has_row_scales,
                zp: zero_point,
                bias,
                has_bias,
                bias_dim: BiasDim::PerRow,
            },
            par,
            ws,
        );
    }
}

/// Requantizing integer GEMM with an unsigned `u8` output (the ONNX QLinearMatMul
/// activation convention): `i8` inputs multiplied into an `i32` accumulator, then requantized
/// in 1 pass to `C[i,j] = clamp(zero_point + round_ne(scale*(sum_k A*B + bias[i])), 0, 255)`
/// with round-half-to-even, where `scale` is the per-tensor value or the per-row `scale_i`.
/// The `u8`-output twin of [`gemm_i8_requant`], differing only in the output domain (`[0, 255]`
/// instead of `[-128, 127]`) and the accepted `zero_point` range. No `alpha` (folds into
/// `scale`) and no `beta` (accumulating into an already-quantized `C` is not well-defined).
/// Uses the thread-local workspace pool
///
/// # Determinism
/// Same contract as [`gemm_i8_requant`]: the `i32` accumulation is exact and ISA-independent,
/// and the requantize step is bit-exact across every ISA (scalar, FMA, AVX-512, VNNI) and
/// across the vector and scalar store paths
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm_i8`], plus: a non-finite or non-positive
/// `scale` (per-tensor or any per-row element); a per-row scale slice whose length is not `A.rows`
/// or which overlaps `C`; a `zero_point` outside `[0, 255]`; or a bias whose length is not
/// `A.rows` or which overlaps `C`
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8(
    a: MatRef<'_, i8>,
    b: MatRef<'_, i8>,
    req: Requantize<'_>,
    c: MatMut<'_, u8>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_i8_requant_u8_with(ws, a, b, req, c, par));
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
    validate_gemm_views(&a, &b, &c);

    let (scale, row_scales, has_row_scales) = requant_scale(a.rows, &c, req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );

    let (bias_ptr, has_bias) = requant_bias(a.rows, &c, req.bias);

    // SAFETY: validated above, shapes agree, strides in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B or the bias/scales; scale/zp are in range. The bias
    // and row-scale slices (borrowed for this call) outlive the `execute_int_requant` frame
    unsafe {
        dispatch::execute_int_requant(
            dispatch::RequantTask {
                m: a.rows,
                k: a.cols,
                n: b.cols,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                b: b.data.as_ptr(),
                rsb: b.rs,
                csb: b.cs,
                c: c.data.as_mut_ptr(),
                rsc: c.rs,
                csc: c.cs,
                scale,
                row_scales,
                has_row_scales,
                zp: req.zero_point,
                bias: bias_ptr,
                has_bias,
                bias_dim: BiasDim::PerRow,
            },
            par,
            ws,
        );
    }
}

/// `C(u8) <- clamp(zp + round_ne(scale*(A*B + bias)), 0, 255)` over raw pointers and `isize`
/// element strides, with no bounds, alias, or shape checks: the unsigned twin of
/// [`gemm_i8_requant_unchecked`]. `bias` is a per-row `i32` pointer, read only when `has_bias`.
/// The scale is the scalar `scale` unless `has_row_scales` is set, in which case `row_scales`
/// supplies 1 `f32` per output row (length `m`) instead. Uses the thread-local workspace pool
///
/// # Safety
/// `a`/`b` valid for reads and `c` valid for writes over the shape/strides; `c` does not alias
/// `a`/`b`; when `has_bias`, `bias` is valid for `m` reads and disjoint from `c`; when
/// `has_row_scales`, `row_scales` is valid for `m` reads and disjoint from `c` (otherwise
/// `row_scales` may be null or dangling); every applied scale is finite and `> 0`, and
/// `zero_point` is in `[0, 255]` (the checked API enforces the last 2)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_requant_u8_unchecked(
    m: usize,
    k: usize,
    n: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    scale: f32,
    row_scales: *const f32,
    has_row_scales: bool,
    zero_point: i32,
    bias: *const i32,
    has_bias: bool,
    c: *mut u8,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_i8_requant_u8_unchecked_with(
                ws,
                m,
                k,
                n,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                scale,
                row_scales,
                has_row_scales,
                zero_point,
                bias,
                has_bias,
                c,
                rsc,
                csc,
                par,
            );
        });
    }
}

/// Like [`gemm_i8_requant_u8_unchecked`] but reuses a caller-owned [`Workspace`] instead
/// of the thread-local pool
///
/// # Safety
/// See [`gemm_i8_requant_u8_unchecked`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_requant_u8_unchecked_with(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    scale: f32,
    row_scales: *const f32,
    has_row_scales: bool,
    zero_point: i32,
    bias: *const i32,
    has_bias: bool,
    c: *mut u8,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: the caller guarantees `a`/`b` valid for reads and `c` for writes over the
    // shape/strides, `c` not aliasing `a`/`b`, when `has_bias` a valid disjoint `m`-length bias,
    // when `has_row_scales` a valid disjoint `m`-length scale vector, and `scale`/`zero_point`
    // in range (see [`gemm_i8_requant_u8_unchecked`])
    unsafe {
        dispatch::execute_int_requant(
            dispatch::RequantTask {
                m,
                k,
                n,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                c,
                rsc,
                csc,
                scale,
                row_scales,
                has_row_scales,
                zp: zero_point,
                bias,
                has_bias,
                bias_dim: BiasDim::PerRow,
            },
            par,
            ws,
        );
    }
}
