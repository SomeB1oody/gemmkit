//! User-defined per-element map-epilogue GEMM entries
//!
//! `gemm_map` applies an arbitrary caller closure `f(value, row, col) -> value` to each output
//! element, fused into the same store the plain kernel would perform. It is the general
//! extension point for epilogues gemmkit has no dedicated fast path for (GELU, sigmoid,
//! clamping, position-dependent transforms). For a plain bias or activation, prefer
//! [`crate::gemm_fused`], which vectorizes the transform; `gemm_map` trades that for an
//! indirect call per output element in exchange for full generality
use super::*;
use crate::dispatch::MapScalar;
use crate::kernel::epilogue::MapEpi;

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in 1 fused pass: a plain GEMM with a caller
/// closure applied to each output element at its final value, over safe slice views, using
/// the thread-local workspace pool. `(r, c)` is the user-frame coordinate of `C` (row `r`,
/// column `c`), and `f` fires exactly once per element, at the point the plain kernel would
/// store it
///
/// The map engine routes every shape through the same kernel [`gemm`] would use: the general
/// register-blocked driver, gemv for `m == 1` / `n == 1`, the small-`m,n` horizontal path, or
/// the small-`k` path, applying `f` to the value the plain store would write instead of
/// writing it directly. So for `f32`/`f64` the result is bit-identical to [`gemm`] followed by
/// mapping every `C[r, c]` through `f(C[r, c], r, c)`, for every shape (strided, transposed,
/// row-major `C`, gemv, degenerate), and deterministic across thread counts: serial and
/// parallel runs match bit-for-bit
///
/// The closure may capture its environment by reference: the `+ Sync` bound is what makes that
/// reference safe to share across the parallel workers. It runs through a `dyn Fn` indirection
/// once per output element; for a plain bias or activation prefer [`gemm_fused`], which applies
/// the transform in-register on the vector fast path. `gemm_map` is the extension point for an
/// arbitrary, position-dependent per-element function
///
/// `T` is `f32`/`f64` only. The narrow floats (`f16`/`bf16`), complex, and integer types are not
/// supported: a `T`-domain closure applied after the `f32` accumulate would double-round,
/// breaking the bitwise contract above. The batched and prepacked entries are likewise not
/// supported. Use [`gemm_fused`] for those
///
/// # Panics
/// Same conditions as [`gemm`]: dimension mismatch, an out-of-bounds view, `C` aliasing
/// itself, or `C` overlapping `A`/`B`. The closure itself is total, applied to every stored
/// element
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

/// Like [`gemm_map`] but reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool: zero heap allocation once the workspace has grown to fit the 1st sufficiently
/// large call
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
    // Same checks as `gemm`/`gemm_fused`; the closure is total, so nothing epilogue-specific
    // needs validating here
    validate_gemm_views(&a, &b, &c);

    // SAFETY: validated above, shapes agree, every stride is in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B; the closure borrow outlives the whole `execute_map` frame
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

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` over raw pointers and `isize` element strides,
/// with no bounds, alias, or shape checks. `(r, c)` is the user-frame coordinate of `C`. Uses
/// the thread-local workspace pool
///
/// # Safety
/// As [`gemm_unchecked`]: `a`/`b` valid for reads and `c` valid for read+write over every
/// `(i,j)` implied by the dimensions/strides, `c` not aliasing `a`/`b`, and `c` need not be
/// initialized when `beta == 0`. The closure is total, applied to every stored element
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

/// Like [`gemm_map_unchecked`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
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

/// Lowering shared by every unchecked map entry: build the [`Task`] and the [`MapEpi`] (in the
/// user frame, so `swapped` starts `false`), then dispatch through a caller-owned [`Workspace`]
/// (`ws = Some`) or the thread-local pool (`ws = None`)
///
/// # Safety
/// As [`gemm_map_unchecked`]: `a`/`b` valid for reads and `c` valid for read+write over the
/// shape/strides, and `c` not aliasing `a`/`b`
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
    // Starts in the user frame; the dispatch layer flips `swapped` if it orients C
    let epi = MapEpi { f, swapped: false };
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_map(task, epi, par, ws),
            None => workspace::with_thread_pool(|ws| dispatch::execute_map(task, epi, par, ws)),
        }
    }
}
