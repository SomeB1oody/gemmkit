//! Integer (`i8` -> `i32`) and requantizing (`i8` -> `i8`) GEMM entries
use super::*;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;

/// Integer GEMM: `C <- alpha*A*B + beta*C` with **`i8` inputs accumulated into an
/// `i32` output** (`alpha`/`beta`/`C` are `i32`). Arithmetic wraps on overflow, the
/// conventional integer-GEMM semantics. Uses the thread-local workspace pool
///
/// A separate entry point from [`gemm`] because input and output types differ
/// (`i8` vs `i32`), which the homogeneous `gemm<T>` surface cannot express
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm`] (`A.cols == B.rows`,
/// `A.rows == C.rows`, `B.cols == C.cols`; every view in bounds; `C` addresses each
/// element uniquely and does not overlap `A`/`B`). Negative-stride / raw-pointer
/// callers use [`gemm_i8_unchecked`] (the homogeneous [`gemm_unchecked`] cannot
/// serve `i8 -> i32`)
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

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`]
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

/// The raw integer engine: `C(i32) <- alpha*A(i8)*B(i8) + beta*C` over pointers and
/// `isize` strides, with **no** bounds/alias/shape checks: the heterogeneous
/// counterpart of [`gemm_unchecked`] (which is typed for the homogeneous surface and
/// cannot serve `i8 -> i32`). The escape hatch [`gemm_i8`] points negative-stride /
/// advanced callers to. Uses the thread-local workspace pool
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for read+write over every
/// `(i,j)` implied by the dimensions and strides; `c` does not alias `a`/`b`; and
/// when `beta == 0`, `c` need not be initialized
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

/// As [`gemm_i8_unchecked`] but with a caller-owned [`Workspace`]
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

/// The quantization parameters shared by the requantizing entries: a per-tensor `scale` and
/// integer `zero_point`, plus an optional **per-row** `i32` bias (length `m`, the standard
/// qlinear layer bias). The output is `C[i,j] = clamp(zero_point + round_ne(scale*(sum_k A*B +
/// bias[i])), LO, HI)`, with round-half-to-even, where the clamp band `[LO, HI]` is set by the
/// entry: [`gemm_i8_requant`] -> `[-128, 127]` (`i8`), [`gemm_i8_requant_u8`] -> `[0, 255]` (`u8`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub struct Requantize<'a> {
    /// Per-tensor output scale (`alpha` folds into this; must be finite and `> 0`)
    pub scale: f32,
    /// Output zero-point, joined in integer after rounding. Must lie in the output domain of the
    /// chosen entry: `[-128, 127]` for [`gemm_i8_requant`], `[0, 255]` for [`gemm_i8_requant_u8`]
    pub zero_point: i32,
    /// Optional per-row `i32` bias (length `m`), added to the accumulator before scaling
    pub bias: Option<&'a [i32]>,
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then
/// requantized to an `i8` output in 1 pass (deleting the full `m*n` `i32` materialization
/// a `gemm_i8` + separate requantize would pay). No `alpha` (folds into `scale`) and no
/// `beta` (accumulating into a quantized C is ill-defined). Uses the thread-local workspace
/// pool
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm_i8`], plus: a non-finite or
/// non-positive `scale`; a `zero_point` outside `[-128, 127]`; or a bias whose length is not
/// `A.rows` or which overlaps `C`
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

/// Shared bias validation for the requantizing entries (`i8` and `u8` output): the optional
/// per-row bias must have length `a_rows` and must not overlap `C` (byte ranges: `C` is a
/// 1-byte quantized output either way, `size_of::<TO>() == 1`). Returns the raw `(ptr, has_bias)`
/// the task carries. Panic messages are byte-identical to the `i8`-only wording (tests match the
/// substrings). The `scale` and `zero_point` range checks stay in each entry (the `zp` band
/// differs between `i8` and `u8`), so this factors only the axis-independent bias check
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

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`]
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

    assert!(
        req.scale.is_finite() && req.scale > 0.0,
        "gemmkit: requantize scale ({}) must be finite and > 0",
        req.scale
    );
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );

    let (bias_ptr, has_bias) = requant_bias(a.rows, &c, req.bias);

    // SAFETY: validated above, shapes agree, strides in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B or the bias; scale/zp are in range. The bias pointer
    // (borrowed for this call) stays valid for the whole `execute_int_requant` frame
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
                scale: req.scale,
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

/// The raw requantizing engine: `C(i8) <- clamp(zp + round_ne(scale*(A*B + bias)), -128,
/// 127)` over pointers and `isize` strides, with **no** bounds/alias/shape checks. `bias` is
/// a per-row `i32` pointer enabled by `has_bias` (ignored when `has_bias == false`). Uses the
/// thread-local workspace pool
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for writes over the shape/strides;
/// `c` does not alias `a`/`b`; when `has_bias`, `bias` is valid for `m` reads and disjoint
/// from `c`; and `scale` is finite and `> 0` with `zero_point in [-128, 127]` (the checked
/// API enforces the last 2)
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
                ws, m, k, n, a, rsa, csa, b, rsb, csb, scale, zero_point, bias, has_bias, c, rsc,
                csc, par,
            );
        });
    }
}

/// As [`gemm_i8_requant_unchecked`] but with a caller-owned [`Workspace`]
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
    // and `scale`/`zero_point` in range (see [`gemm_i8_requant_unchecked`])
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

/// Requantizing integer GEMM with an **unsigned `u8` output** (ONNX-QLinearMatMul-style
/// activation): `i8` inputs multiplied into an `i32` accumulator, then requantized in 1 pass to
/// `C[i,j] = clamp(zero_point + round_ne(scale*(sum_k A*B + bias[i])), 0, 255)` with
/// round-half-to-even. The `i8`-output twin of [`gemm_i8_requant`], differing only in the output
/// domain `[0, 255]` and the `zero_point` range. No `alpha` (folds into `scale`) and no `beta`
/// (accumulating into a quantized C is ill-defined). Uses the thread-local workspace pool
///
/// # Determinism
/// The same contracts as [`gemm_i8_requant`]: the `i32` accumulation is exact and ISA-independent,
/// and the requantize map is **bit-exact across every ISA** (scalar / FMA / AVX-512 / VNNI) and
/// across the vector and scalar store paths
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm_i8`], plus: a non-finite or non-positive
/// `scale`; a `zero_point` outside `[0, 255]`; or a bias whose length is not `A.rows` or which
/// overlaps `C`
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

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`]
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

    assert!(
        req.scale.is_finite() && req.scale > 0.0,
        "gemmkit: requantize scale ({}) must be finite and > 0",
        req.scale
    );
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );

    let (bias_ptr, has_bias) = requant_bias(a.rows, &c, req.bias);

    // SAFETY: validated above, shapes agree, strides in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B or the bias; scale/zp are in range. The bias pointer
    // (borrowed for this call) stays valid for the whole `execute_int_requant` frame
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
                scale: req.scale,
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

/// The raw `u8`-output requantizing engine: `C(u8) <- clamp(zp + round_ne(scale*(A*B + bias)), 0,
/// 255)` over pointers and `isize` strides, with **no** bounds/alias/shape checks: the unsigned
/// twin of [`gemm_i8_requant_unchecked`]. `bias` is a per-row `i32` pointer enabled by `has_bias`
/// (ignored when `has_bias == false`). Uses the thread-local workspace pool
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for writes over the shape/strides;
/// `c` does not alias `a`/`b`; when `has_bias`, `bias` is valid for `m` reads and disjoint
/// from `c`; and `scale` is finite and `> 0` with `zero_point in [0, 255]` (the checked
/// API enforces the last 2)
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
                ws, m, k, n, a, rsa, csa, b, rsb, csb, scale, zero_point, bias, has_bias, c, rsc,
                csc, par,
            );
        });
    }
}

/// As [`gemm_i8_requant_u8_unchecked`] but with a caller-owned [`Workspace`]
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
    // and `scale`/`zero_point` in range (see [`gemm_i8_requant_u8_unchecked`])
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
