//! Complex GEMM entries (`c32`/`c64`, feature `complex`): `C <- alpha*op(A)*op(B) + beta*C`,
//! where `op` optionally conjugates an operand. [`gemm_cplx`] and [`gemm_cplx_with`] are the
//! checked entries over [`MatRef`]/[`MatMut`]. [`gemm_cplx_unchecked`] and
//! [`gemm_cplx_unchecked_with`] are the raw pointer and `isize`-stride equivalents. Under
//! feature `epilogue`, [`gemm_cplx_fused`] and its siblings add a fused per-row or per-col
//! bias. Complex numbers have no activation, because an ordering-based activation has no
//! definition there
use super::*;
use crate::dispatch::ComplexScalar;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;

/// Complex GEMM with optional conjugation: `C <- alpha*op(A)*op(B) + beta*C`, where
/// `op(A) = conj(A)` when `conj_a` (independently, `op(B) = conj(B)` when `conj_b`). `T` is
/// `Complex<f32>` or `Complex<f64>` (re-exported as [`crate::c32`] / [`crate::c64`]). Uses the
/// thread-local workspace pool
///
/// Complex numbers dispatch through a path of their own, since `ComplexScalar` does not
/// extend `GemmScalar`. This entry exists to carry the conjugation flags through that path,
/// whether or not either operand is conjugated
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

    // SAFETY: shapes, bounds, and non-aliasing validated above. C addresses each (i,j) uniquely
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

/// Complex GEMM with an optional fused per-row / per-col bias:
/// `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 pass, where `op` conjugates an operand exactly
/// as in [`gemm_cplx`]. `T` is `Complex<f32>` or `Complex<f64>`. Uses the thread-local
/// workspace pool
///
/// The bias is [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`). It is
/// added by 1 complex add to every element of that row or column, after the
/// `alpha*op(A)*op(B) + beta*C` combine. It is added verbatim, never conjugated. There is no
/// activation parameter. An ordering-based activation such as ReLU has no definition on
/// complex numbers, so bias is the only fusible complex epilogue. `bias == None` delegates to
/// plain [`gemm_cplx`]
///
/// The kernel stores exactly the bits plain `gemm_cplx` would and applies the bias in a
/// tile-local post-pass on the final depth panel. The result is bit-identical to
/// [`gemm_cplx`] followed by the same element-wise bias add. This holds for every shape, every
/// conj combination, and both the real and imaginary parts. Serial and parallel runs also
/// agree bit-for-bit today, though the crate's reproducibility contract covers only a fixed
/// configuration
///
/// # Panics
/// Same conditions as [`gemm_cplx`], plus: a `PerRow` bias whose length is not `A.rows` (or a
/// `PerCol` bias not `B.cols`), or a bias slice that overlaps `C`
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
    // No bias: fall back to plain gemm_cplx so no fused kernel is instantiated, and both paths
    // share 1 set of validation panics
    let Some(bias) = bias else {
        gemm_cplx_with(ws, alpha, a, conj_a, b, conj_b, beta, c, par);
        return;
    };

    validate_gemm_views(&a, &b, &c);

    // Bias length matches its axis and stays clear of C
    validate_bias(&Some(bias), a.rows, b.cols, &c);

    // Complex numbers have no activation, because an ordering-based activation has no
    // definition there. act stays None
    let epi = to_fused_epi(Some(bias), None);

    // SAFETY: shapes, bounds, and non-aliasing validated above. The bias is in-bounds and
    // disjoint from C, and its borrow outlives this execute_complex_fused call
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

/// The raw complex fused engine: `C <- alpha*op(A)*op(B) + beta*C + bias` over pointers and
/// `isize` strides, with no bounds, alias, or shape checks: the raw-parts form of
/// [`gemm_cplx_fused`]. `op` conjugates an operand when its `conj_*` flag is set. `bias` is a
/// `(ptr, dim)` pair, read only when `has_bias`, and added verbatim, never conjugated. There is
/// no activation parameter, since it has no definition on complex numbers. Uses the
/// thread-local workspace pool
///
/// # Safety
/// As [`gemm_cplx_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m`
/// (`PerRow`) or `n` (`PerCol`) elements and does not alias `c`
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
    // SAFETY: preconditions satisfied by the caller, per # Safety above
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
    // No activation on complex: lower with act = None
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, None);
    // SAFETY: preconditions satisfied by the caller, per # Safety above
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

/// The raw complex engine: `C <- alpha*op(A)*op(B) + beta*C` over pointers and `isize`
/// strides, with no bounds, alias, or shape checks: the complex counterpart of
/// [`gemm_unchecked`], where `op` conjugates an operand when its `conj_*` flag is set. Adapter
/// crates (e.g. ndarray) use this path to express transposed or negative strides that the
/// checked API rejects. Uses the thread-local workspace pool
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for read+write over every
/// `(i,j)` implied by the dimensions and strides. `c` does not alias `a`/`b`. When
/// `beta == 0`, `c` need not be initialized
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
