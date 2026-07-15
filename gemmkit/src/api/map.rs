//! User-defined per-element map-epilogue GEMM entries
//!
//! `gemm_map` applies an arbitrary caller closure `f(value, row, col) -> value` to each output
//! element at its final value, in 1 pass fused into the store. It is the general extension
//! point for epilogues gemmkit does not ship a fast path for (GELU, sigmoid, clamps,
//! position-dependent transforms): use [`crate::gemm_fused`] for bias / activation, which
//! vectorizes; `gemm_map` trades 1 indirect call per output element (amortized by the `O(k)`
//! FLOPs per element) for total generality
use super::*;
use crate::dispatch::MapScalar;
use crate::kernel::epilogue::MapEpi;

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in 1 fused pass: a plain GEMM with a caller
/// closure applied to each output element at its final value, over safe slice views, using the
/// thread-local workspace pool. `(r, c)` are in the **user** frame of `C` (row `r`, column `c`),
/// and `f` fires **exactly once** per element, on the final depth panel
///
/// The map engine routes every shape through the **same** kernel `gemm` would use (the general
/// register-blocked driver, gemv for `m == 1` / `n == 1`, the small-`m,n` horizontal path, or the
/// small-`k` path), applying `f` to the very value the plain store would write. So for `f32`/`f64`
/// the result is **bit-identical** to [`gemm`] followed by mapping each `C[r, c]` through
/// `f(C[r, c], r, c)`, for **every** shape (strided / transposed / row-major C, gemv, degenerate),
/// and deterministic across thread counts (serial == parallel, bit-for-bit)
///
/// The closure may **capture its environment by reference** (`+ Sync`, so it is shared safely across
/// the parallel workers). It is invoked once per output element through a `dyn Fn` indirection: for
/// a simple bias or activation prefer [`gemm_fused`] (it applies the transform in-register on the
/// vector fast path); `gemm_map` is the extension point for arbitrary per-element / position-dependent
/// maps
///
/// `T` is `f32`/`f64` only. The narrow floats (`f16`/`bf16`), complex, and integer types are **not**
/// supported (a `T`-domain closure after the `f32` accumulate would double-round, breaking the
/// bitwise contract); nor are the batched or prepacked entries. Use [`gemm_fused`] for those
///
/// # Panics
/// Same conditions as [`gemm`] (dimension mismatch, out-of-bounds view, C self-aliasing, or C
/// overlapping A / B). The closure itself is total: it is applied to every stored element
///
/// [`gemm_fused`]: crate::gemm_fused
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map<T: MapScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_map_with(ws, alpha, a, b, beta, c, f, par));
}

/// Like [`gemm_map`] but reuses a caller-owned [`Workspace`]: zero heap allocation after the 1st
/// sufficiently large call
///
/// # Panics
/// Same conditions as [`gemm_map`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map_with<T: MapScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    // The standard GEMM view validation, byte-identical to `gemm`/`gemm_fused` (dimensions in
    // bounds, C addresses each (i,j) uniquely and does not alias A/B). The closure is total, so
    // there is nothing epilogue-specific to validate
    validate_gemm_views(&a, &b, &c);

    // SAFETY: validated above, shapes agree, every stride is in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B. The closure borrow outlives the whole `execute_map` frame
    unsafe {
        map_unchecked_impl(
            Some(ws),
            a.rows,
            a.cols,
            b.cols,
            alpha,
            a.data.as_ptr(),
            a.rs,
            a.cs,
            b.data.as_ptr(),
            b.rs,
            b.cs,
            beta,
            c.data.as_mut_ptr(),
            c.rs,
            c.cs,
            f,
            par,
        );
    }
}

/// The raw map engine: `C[r, c] <- f(alpha*A*B + beta*C, r, c)` over pointers and `isize` strides,
/// with **no** bounds/alias/shape checks. `(r, c)` are the user-frame coordinates of `C`. Uses the
/// thread-local workspace pool
///
/// # Safety
/// As [`gemm_unchecked`]: `a`/`b` valid for reads and `c` for read+write over every `(i,j)` implied
/// by the dimensions/strides, `c` not aliasing `a`/`b`, and `c` need not be initialized when
/// `beta == 0`. The closure is applied to every stored element (it is total)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_map_unchecked<T: MapScalar>(
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
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        map_unchecked_impl(
            None, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, f, par,
        );
    }
}

/// As [`gemm_map_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_map_unchecked`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_map_unchecked_with<T: MapScalar>(
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
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        map_unchecked_impl(
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
            f,
            par,
        );
    }
}

/// Shared lowering for the map entries: build the [`Task`] and the [`MapEpi`] (in the user frame,
/// so `swapped` starts `false`), then dispatch the map engine over either a caller-owned
/// [`Workspace`] (`ws = Some`) or the thread-local pool (`ws = None`)
///
/// # Safety
/// As [`gemm_map_unchecked`]: `a`/`b` valid for reads and `c` for read+write over the shape /
/// strides, and `c` not aliasing `a`/`b`
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
unsafe fn map_unchecked_impl<T: MapScalar>(
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
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
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
    // The epilogue starts in the user frame; the dispatch layer sets `swapped` if it orients C
    let epi = MapEpi { f, swapped: false };
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_map(task, epi, par, ws),
            None => workspace::with_thread_pool(|ws| dispatch::execute_map(task, epi, par, ws)),
        }
    }
}
