//! Complex GEMM entries with optional conjugation
use super::*;
use crate::dispatch::ComplexScalar;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{Act, BiasDim, BiasSpec, FusedEpi};
#[cfg(feature = "epilogue")]
use crate::parallel::Ptr;

/// Complex GEMM with optional conjugation: `C <- alpha*op(A)*op(B) + beta*C` where
/// `op(A) = conj(A)` if `conj_a` (resp. `conj(B)` if `conj_b`). `T` is `Complex<f32>` or
/// `Complex<f64>` (re-exported as [`crate::c32`] / [`crate::c64`]). Uses the
/// thread-local workspace pool
///
/// Complex is homogeneous, so the non-conjugated case could ride [`gemm`], but the
/// conj op-family gets its own entry; `conj_a = conj_b = false` is the plain product
/// `A*B`
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm`]
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T: ComplexScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_cplx_with(ws, alpha, a, conj_a, b, conj_b, beta, c, par));
}

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx`]
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T: ComplexScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    validate_gemm_views(&a, &b, &c);

    // SAFETY: validated above, shapes agree, strides in bounds, C unique and not
    // aliasing A/B
    unsafe {
        dispatch::execute_complex(
            conj_a,
            conj_b,
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
            par,
            ws,
        );
    }
}

/// Complex GEMM with an **optional fused per-row / per-col bias**:
/// `C <- alpha*op(A)*op(B) + beta*C + bias` in **1 pass**, where `op(A) = conj(A)` if
/// `conj_a` (resp. `conj(B)` if `conj_b`). `T` is `Complex<f32>` or `Complex<f64>`. Uses
/// the thread-local workspace pool
///
/// The bias is [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), added by
/// 1 complex IEEE add to every element of that row / column after the `alpha*op(A)*op(B) + beta*C`
/// combine. `bias == None` delegates to plain [`gemm_cplx`] (a zero-cost identity: no fused
/// monomorphization is even reached)
///
/// **No activation.** Unlike [`gemm_fused`], the complex entry has no activation parameter: an
/// ordering-based activation (ReLU / LeakyReLU) is mathematically undefined on complex numbers
/// (not an ordered field), so it is deliberately absent; bias is the only fusible complex epilogue
///
/// **Bitwise contract.** The result is **bit-identical** (on both the real and imaginary parts) to
/// [`gemm_cplx`] followed by the same element-wise bias add, for **every** shape and **every**
/// conj combination: the engine stores exactly the bits plain `gemm_cplx` would and maps them in
/// place on the final depth panel (a tile-local post-pass, applied once per element). It is also
/// deterministic across thread counts (serial == parallel, bit-for-bit, the same
/// thread-independent-blocking caveat as plain `gemm_cplx`)
///
/// `conj_a` / `conj_b` conjugate the **operands only**; the bias is added verbatim, never
/// conjugated
///
/// There is no `_unchecked` variant (the minimal-surface decision shared with the other fused
/// entries): advanced raw-stride callers use the checked entry or compose `gemm_cplx_unchecked`
/// with their own bias pass
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm_cplx`], plus: a `PerRow` bias whose length
/// is not `A.rows` (or a `PerCol` bias not `B.cols`), or a bias slice that overlaps `C`
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T: ComplexScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| {
        gemm_cplx_fused_with(ws, alpha, a, conj_a, b, conj_b, beta, c, bias, par)
    });
}

/// Like [`gemm_cplx_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T: ComplexScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    conj_a: bool,
    b: MatRef<'_, T>,
    conj_b: bool,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) {
    validate_gemm_views(&a, &b, &c);

    // Fused-bias validation: bias length matches its axis and does not overlap C
    validate_bias(&bias, a.rows, b.cols, &c);

    // No bias: the fused path is pure identity, so delegate to plain `gemm_cplx`, the zero-cost
    // path is then guaranteed (no fused monomorphization is instantiated)
    let Some(bias) = bias else {
        gemm_cplx_with(ws, alpha, a, conj_a, b, conj_b, beta, c, par);
        return;
    };

    let bias_spec = match bias {
        Bias::PerRow(s) => BiasSpec::Row(Ptr(s.as_ptr() as *mut T)),
        Bias::PerCol(s) => BiasSpec::Col(Ptr(s.as_ptr() as *mut T)),
    };
    // Complex has no activation (undefined on complex numbers): always `Act::None`
    let epi = FusedEpi {
        bias: bias_spec,
        act: Act::None,
    };

    // SAFETY: validated above, shapes agree, every stride is in bounds, C addresses each (i,j)
    // uniquely and does not alias A/B, and the bias slice (borrowed for this call) does not overlap
    // C. The bias pointer stays valid for the whole `execute_complex_fused` frame
    unsafe {
        dispatch::execute_complex_fused(
            conj_a,
            conj_b,
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

/// The raw complex fused engine: `C <- alpha*op(A)*op(B) + beta*C + bias` over pointers and `isize`
/// strides, with **no** bounds/alias/shape checks: the raw-parts form of [`gemm_cplx_fused`] (and
/// the complex sibling of [`gemm_fused_unchecked`]). `op` conjugates the operand when its `conj_*`
/// flag is set; `bias` is a `(ptr, dim)` pair enabled by `has_bias` (`bias` is ignored when
/// `has_bias == false`), added verbatim (never conjugated). There is **no** activation parameter:
/// an ordering activation is undefined on complex numbers. Uses the thread-local workspace pool
///
/// # Safety
/// As [`gemm_cplx_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m` (`PerRow`)
/// or `n` (`PerCol`) elements and does not alias `c`
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_cplx_fused_unchecked<T: ComplexScalar>(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    conj_a: bool,
    b: *const T,
    rsb: isize,
    csb: isize,
    conj_b: bool,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_cplx_fused_unchecked_with(
                ws, m, k, n, alpha, a, rsa, csa, conj_a, b, rsb, csb, conj_b, beta, c, rsc, csc,
                bias, bias_dim, has_bias, par,
            );
        });
    }
}

/// As [`gemm_cplx_fused_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_cplx_fused_unchecked`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_cplx_fused_unchecked_with<T: ComplexScalar>(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    conj_a: bool,
    b: *const T,
    rsb: isize,
    csb: isize,
    conj_b: bool,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    par: Parallelism,
) {
    // Complex has no activation (undefined on complex numbers): lower with `act = None`, giving
    // `Act::None`
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, None);
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        dispatch::execute_complex_fused(
            conj_a,
            conj_b,
            Task {
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
            epi,
            par,
            ws,
        );
    }
}

/// The raw complex engine: `C <- alpha*op(A)*op(B) + beta*C` over pointers and
/// `isize` strides, with **no** bounds/alias/shape checks: the complex counterpart
/// of [`gemm_unchecked`] (`op` conjugates the operand when its `conj_*` flag is set).
/// The raw path advanced callers (e.g. the ndarray adapter) use to express arbitrary
/// (transposed / negative) strides. Uses the thread-local workspace pool
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for read+write over every
/// `(i,j)` implied by the dimensions and strides; `c` does not alias `a`/`b`; and
/// when `beta == 0`, `c` need not be initialized
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_cplx_unchecked<T: ComplexScalar>(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    conj_a: bool,
    b: *const T,
    rsb: isize,
    csb: isize,
    conj_b: bool,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_cplx_unchecked_with(
                ws, m, k, n, alpha, a, rsa, csa, conj_a, b, rsb, csb, conj_b, beta, c, rsc, csc,
                par,
            );
        });
    }
}

/// As [`gemm_cplx_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_cplx_unchecked`]
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_cplx_unchecked_with<T: ComplexScalar>(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    conj_a: bool,
    b: *const T,
    rsb: isize,
    csb: isize,
    conj_b: bool,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    unsafe {
        dispatch::execute_complex(
            conj_a,
            conj_b,
            Task {
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
