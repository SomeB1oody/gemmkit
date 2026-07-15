//! Fused-epilogue (bias / activation) GEMM entries
use super::*;
use crate::dispatch::FusedScalar;
use crate::kernel::epilogue::{BiasDim, FusedEpi};

/// A bias vector fused into a [`gemm_fused`] call: 1 value per output **row** (length `m`)
/// or per output **column** (length `n`), added to every element of that row / column after
/// the product and before the activation
pub enum Bias<'a, T> {
    /// 1 value per output row (length `m`)
    PerRow(&'a [T]),
    /// 1 value per output column (length `n`)
    PerCol(&'a [T]),
}

/// An activation fused into a [`gemm_fused`] call, applied last (after the bias add)
pub enum Activation<T> {
    /// `max(v, 0)` (NaN maps to 0)
    Relu,
    /// `max(v, 0) + slope*min(v, 0)` (NaN maps to 0, -0 to +0)
    LeakyRelu(T),
}

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 pass: a **fused** GEMM epilogue over safe
/// slice views, using the thread-local workspace pool. The bias is added by 1 IEEE add
/// after the final `beta`-fold, then the activation is applied. `bias == None && act == None`
/// delegates to plain [`gemm`]
///
/// The fused engine routes every shape through the **same** kernel `gemm` would use: the
/// general register-blocked driver, gemv (`m == 1` / `n == 1`), the small-`m,n` horizontal
/// path, or the small-`k` path, fusing the epilogue into that kernel's store without
/// perturbing its accumulation order. So for `f32`/`f64` the result is **bit-identical** to
/// `gemm()` followed by the same scalar map, for **every** shape, and deterministic across
/// thread counts
///
/// **Narrow types (`f16`/`bf16`, feature `half`).** `T` may also be a narrow float. The bias
/// vector and `LeakyRelu` slope are then the narrow type and are widened **exactly** to `f32`
/// (`f16`/`bf16` are a subset of `f32`); the epilogue applies in `f32` to the accumulator
/// **before** the single round-to-nearest-even narrowing to the output. This is *more* precise
/// than `gemm()` followed by a separate narrow map (which rounds to the narrow type, widens
/// back, then rounds again), so for narrow types it is **not** bitwise-equal to `gemm`-then-map:
/// the every-shape bitwise contract above holds for `f32`/`f64` only. Reproducibility and
/// determinism are unchanged (serial == parallel, bit-for-bit)
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

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`]. Accepts `f32`/`f64` and, under
/// the `half` feature, `f16`/`bf16` with the same pre-narrow `f32` epilogue semantics as
/// [`gemm_fused`] (more precise than `gemm`-then-map; bitwise-equal to it only for `f32`/`f64`)
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
    validate_gemm_views(&a, &b, &c);

    // Fused-epilogue validation: bias length matches its axis and does not overlap C
    validate_bias(&bias, a.rows, b.cols, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // The identity-fused case cannot even reach a fused monomorphization: delegate to plain
    // gemm so the zero-cost path is guaranteed
    if bias.is_none() && act.is_none() {
        gemm_with(ws, alpha, a, b, beta, c, par);
        return;
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validated above, shapes agree, every stride is in bounds, C addresses each
    // (i,j) uniquely and does not alias A/B, and the bias slice (borrowed for this call) does
    // not overlap C. The bias pointer stays valid for the whole `execute_fused` frame
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
/// strides, with **no** bounds/alias/shape checks. `bias` is a `(ptr, dim)` pair enabled by
/// `has_bias` (`bias` is ignored when `has_bias == false`). Uses the thread-local workspace
/// pool
///
/// Accepts `f32`/`f64` and, under the `half` feature, `f16`/`bf16`. For narrow types the bias /
/// slope are the narrow type, widened **exactly** to `f32`, and the epilogue applies in `f32`
/// before the single narrowing to the output: *more* precise than a separate narrow map, hence
/// not bitwise-equal to `gemm`-then-map (the `f32`/`f64` every-shape bitwise contract is
/// unchanged); reproducibility/determinism are unchanged
///
/// # Safety
/// As [`gemm_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m` (`PerRow`)
/// or `n` (`PerCol`) elements and does not alias `c`; and a non-finite `LeakyRelu` slope is
/// the caller's responsibility (the checked API rejects it)
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
    // SAFETY: preconditions forwarded to the caller (see # Safety)
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
    // SAFETY: preconditions forwarded to the caller (see # Safety)
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

/// Shared lowering for the 2 raw fused entries: build the [`Task`], then dispatch the fused
/// engine over either a caller-owned [`Workspace`] (`ws = Some`) or the thread-local pool
/// (`ws = None`). The bias/activation are already lowered into `epi` by [`to_fused_epi_raw`]
///
/// # Safety
/// As [`gemm_fused_unchecked`]: `a`/`b` valid for reads and `c` for read+write over the shape /
/// strides, `c` not aliasing `a`/`b`, and `epi`'s bias (if any) valid for `m`/`n` reads and
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
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_fused(task, epi, par, ws),
            None => workspace::with_thread_pool(|ws| dispatch::execute_fused(task, epi, par, ws)),
        }
    }
}
