//! Fused-epilogue GEMM entries (feature `epilogue`): bias and activation folded into the same
//! store the plain kernel would perform, so `C <- act(alpha*A*B + beta*C + bias)` costs 1 pass
//! over `C` instead of a GEMM followed by a separate scalar map
//!
//! [`Bias`] and [`Activation`] select the epilogue; [`gemm_fused`] / [`gemm_fused_with`] are the
//! checked entries over [`MatRef`]/[`MatMut`], and [`gemm_fused_unchecked`] /
//! [`gemm_fused_unchecked_with`] the raw pointer + `isize`-stride equivalents. With no bias and
//! no activation, the checked entries fall back to the plain, unfused engine; the raw entries
//! instead run the fused engine with a no-op epilogue
use super::*;
use crate::dispatch::FusedScalar;
use crate::kernel::epilogue::{BiasDim, FusedEpi};

/// 1 bias value broadcast across a whole output row or column, combined into a [`gemm_fused`]
/// call's epilogue after the `alpha*A*B + beta*C` product and before the activation
#[derive(Copy, Clone)]
pub enum Bias<'a, T> {
    /// 1 value per output row, added to every element of that row (length `m`)
    PerRow(&'a [T]),
    /// 1 value per output column, added to every element of that column (length `n`)
    PerCol(&'a [T]),
}

/// The activation a [`gemm_fused`] call applies last, after the bias add
pub enum Activation<T> {
    /// `max(v, 0)`; NaN maps to 0
    Relu,
    /// `max(v, 0) + slope*min(v, 0)`; NaN maps to 0, negative zero maps to positive zero
    LeakyRelu(T),
}

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 pass over safe slice views, using the
/// thread-local workspace pool. `bias == None && act == None` delegates to plain [`gemm`]
///
/// The fused kernel takes the same route plain `gemm` would for the same shape (the general
/// register-blocked driver, gemv for `m == 1` / `n == 1`, the small-`m,n` horizontal path, or
/// the small-`k` path) and applies the epilogue to the very register or scratch value the plain
/// store would otherwise have written. So for `f32`/`f64` the result is bit-identical, for
/// every shape, to `gemm()` followed by the same scalar map, and it stays deterministic across
/// thread counts
///
/// Under the `half` feature `T` may also be `f16`/`bf16`. The bias vector and `LeakyRelu` slope
/// are then the narrow type, widened exactly to `f32` (a narrow value is a strict subset of
/// `f32`), and the epilogue runs in `f32` on the accumulator before a single
/// round-to-nearest-even narrowing to the output. That is more precise than `gemm()` followed
/// by a separate narrow map, which would round to the narrow type, widen back, then round
/// again, so for `f16`/`bf16` the result is not bitwise-equal to `gemm`-then-map (the
/// `f32`/`f64` bitwise contract above does not extend to them). Determinism (serial ==
/// parallel, bit-for-bit) holds regardless
///
/// # Panics
/// Same conditions as [`gemm`], plus: a `PerRow` bias whose length is not `A.rows` (or a
/// `PerCol` bias not `B.cols`); a bias slice that overlaps `C`; or a non-finite `LeakyRelu`
/// slope
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
    workspace::with_thread_pool(|ws| gemm_fused_with(ws, alpha, a, b, beta, c, bias, act, par));
}

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`]. Accepts the same `f32`/`f64`
/// and, under `half`, `f16`/`bf16` types, with the same pre-narrow `f32` epilogue precision
/// described at [`gemm_fused`]
///
/// # Panics
/// Same conditions as [`gemm_fused`]
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
    // No bias or activation: delegate to plain gemm so the fused kernel is never instantiated,
    // and both paths share 1 set of validation panics
    if bias.is_none() && act.is_none() {
        gemm_with(ws, alpha, a, b, beta, c, par);
        return;
    }

    validate_gemm_views(&a, &b, &c);

    // Bias length matches its axis and stays clear of C
    validate_bias(&bias, a.rows, b.cols, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: shapes, bounds, and non-aliasing validated above; the bias (if any) is in-bounds
    // and disjoint from C, and its borrow outlives this execute_fused call
    unsafe {
        dispatch::execute_fused(
            Task {
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
            epi,
            par,
            ws,
        );
    }
}

/// The raw fused engine: `C <- act(alpha*A*B + beta*C + bias)` over pointers and `isize`
/// strides, with no bounds, alias, or shape checks. `bias` is a `(ptr, dim)` pair, read only
/// when `has_bias`. Uses the thread-local workspace pool
///
/// Accepts `f32`/`f64` and, under `half`, `f16`/`bf16`; narrow types get the same pre-narrow
/// `f32` epilogue precision as [`gemm_fused`], so the result is not bitwise-equal to
/// `gemm`-then-map for those types (unlike `f32`/`f64`)
///
/// # Safety
/// As [`gemm_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m` (`PerRow`)
/// or `n` (`PerCol`) elements and does not alias `c`; a non-finite `LeakyRelu` slope is the
/// caller's responsibility (the checked API rejects it)
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_fused_unchecked<T: FusedScalar>(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        fused_unchecked_impl(
            None, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, epi, par,
        );
    }
}

/// As [`gemm_fused_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_fused_unchecked`]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_fused_unchecked_with<T: FusedScalar>(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        fused_unchecked_impl(
            Some(ws),
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
            epi,
            par,
        );
    }
}

/// Shared body for [`gemm_fused_unchecked`] and [`gemm_fused_unchecked_with`]: builds the
/// [`Task`] and runs the fused engine, either on a caller-owned [`Workspace`] (`ws = Some`) or
/// the thread-local pool (`ws = None`). `epi` arrives already lowered by [`to_fused_epi_raw`]
///
/// # Safety
/// As [`gemm_fused_unchecked`]: `a`/`b` valid for reads and `c` for read+write over the shape
/// and strides, `c` not aliasing `a`/`b`, and `epi`'s bias (if any) valid for `m`/`n` reads and
/// disjoint from `c`
#[allow(clippy::too_many_arguments)]
unsafe fn fused_unchecked_impl<T: FusedScalar>(
    ws: Option<&mut Workspace>,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    epi: FusedEpi<T>,
    par: Parallelism,
) {
    let task = Task {
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
    };
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_fused(task, epi, par, ws),
            None => workspace::with_thread_pool(|ws| dispatch::execute_fused(task, epi, par, ws)),
        }
    }
}
