//! Complex GEMM entries with optional conjugation.
use super::*;
use crate::dispatch::ComplexScalar;

/// Complex GEMM with optional conjugation: `C <- alpha·op(A)·op(B) + beta·C` where
/// `op(A) = A̅` if `conj_a` (resp. `B̅` if `conj_b`). `T` is `Complex<f32>` or
/// `Complex<f64>` (re-exported as [`crate::c32`] / [`crate::c64`]). Uses the
/// thread-local workspace pool.
///
/// Complex is homogeneous, so the non-conjugated case could ride [`gemm`], but the
/// conj op-family gets its own entry; `conj_a = conj_b = false` is the plain product
/// `A·B`.
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm`].
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

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_cplx`].
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

    // SAFETY: validated above — shapes agree, strides in bounds, C unique and not
    // aliasing A/B.
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

/// The raw complex engine: `C <- alpha·op(A)·op(B) + beta·C` over pointers and
/// `isize` strides, with **no** bounds/alias/shape checks — the complex counterpart
/// of [`gemm_unchecked`] (`op` conjugates the operand when its `conj_*` flag is set).
/// The raw path advanced callers (e.g. the ndarray adapter) use to express arbitrary
/// (transposed / negative) strides. Uses the thread-local workspace pool.
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for read+write over every
/// `(i,j)` implied by the dimensions and strides; `c` does not alias `a`/`b`; and
/// when `beta == 0`, `c` need not be initialized.
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

/// As [`gemm_cplx_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_cplx_unchecked`].
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
