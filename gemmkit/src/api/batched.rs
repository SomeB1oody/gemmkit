//! Batched GEMM entries: many independent products in 1 call
//!
//! 2 batch forms: the strided form ([`gemm_batched`] and its fused/unchecked siblings), where
//! every element shares 1 shape and strides and is spaced by a fixed `*_batch_stride`, and the
//! pointer-array form ([`gemm_batched_slice`] / [`gemm_batched_ptr_unchecked`]), where each
//! element carries its own shape and pointers. Both parallelize across the batch (whole GEMMs
//! assigned to workers) rather than splitting an individual element, and both forward to the
//! scheduling engine in `crate::special::batched`
#[cfg(feature = "epilogue")]
use super::fused::{Activation, Bias};
use super::*;
#[cfg(feature = "epilogue")]
use crate::dispatch::FusedScalar;
use crate::dispatch::GemmProblem;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::BiasDim;
use alloc::vec::Vec;

/// Bounds-checks a strided-batched view: every element (element `bi` based at slice offset
/// `bi * batch_stride`) must address inside `data`, including the last one. `batch_stride` must
/// be non-negative when `batch > 1` (only element 0 exists otherwise, so the stride is moot).
/// Returns the single element's extent (highest offset + 1) so the caller can reuse it, e.g.
/// for a disjointness check
///
/// # Panics
/// If the strides are negative or too large to address, if `batch_stride` is negative, or if
/// the last element's view runs past `data`
#[allow(clippy::too_many_arguments)]
fn check_batched_view<T>(
    data: &[T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
    batch: usize,
    batch_stride: isize,
    name: &str,
) -> usize {
    let e = match extent(rows, cols, rs, cs) {
        Some(e) => e,
        None => panic!(
            "gemmkit: {name} view has negative strides or is too large to address; use the unchecked API"
        ),
    };
    // batch <= 1: only element 0 exists, so the stride is irrelevant
    let last_base = if batch <= 1 {
        0
    } else {
        if batch_stride < 0 {
            panic!("gemmkit: {name} batch stride ({batch_stride}) must be non-negative");
        }
        (batch - 1).saturating_mul(batch_stride as usize)
    };
    let need = last_base.saturating_add(e);
    if need > data.len() {
        panic!(
            "gemmkit: {name} batched view ({batch}× {rows}x{cols}, batch stride {batch_stride}) \
             needs {need} elements but slice has {}",
            data.len()
        );
    }
    e
}

/// The shared checked-API validation for a strided-batched `(A, B, C)` trio, used by both plain
/// [`gemm_batched_with`] and fused [`gemm_batched_fused_with`]: per-element inner dimensions
/// agree, every element view (including the last) is in bounds, every `C` element addresses
/// uniquely, the `batch` `C` outputs are pairwise disjoint, and `C` does not overlap `A`/`B`.
/// Panics on any violation (the wording is what the tests assert on). Callers add any
/// entry-specific checks (fused bias) after this returns
///
/// Assumes `batch >= 1`: callers short-circuit `batch == 0` before validating, since the views
/// are unused there
#[allow(clippy::too_many_arguments)]
fn validate_batched_views<T>(
    batch: usize,
    a: &MatRef<'_, T>,
    a_batch_stride: isize,
    b: &MatRef<'_, T>,
    b_batch_stride: isize,
    c: &MatMut<'_, T>,
    c_batch_stride: isize,
) {
    assert_eq!(
        a.cols, b.rows,
        "gemmkit: A.cols ({}) != B.rows ({})",
        a.cols, b.rows
    );
    assert_eq!(
        a.rows, c.rows,
        "gemmkit: A.rows ({}) != C.rows ({})",
        a.rows, c.rows
    );
    assert_eq!(
        b.cols, c.cols,
        "gemmkit: B.cols ({}) != C.cols ({})",
        b.cols, c.cols
    );

    check_batched_view(
        a.data,
        a.rows,
        a.cols,
        a.rs,
        a.cs,
        batch,
        a_batch_stride,
        "A",
    );
    check_batched_view(
        b.data,
        b.rows,
        b.cols,
        b.rs,
        b.cs,
        batch,
        b_batch_stride,
        "B",
    );
    let c_extent = check_batched_view(
        c.data,
        c.rows,
        c.cols,
        c.rs,
        c.cs,
        batch,
        c_batch_stride,
        "C",
    );

    // C must address each (i,j) uniquely (self-aliasing would race under concurrent writes),
    // and the batch elements must not overlap each other either. Disjointness is enforced
    // conservatively: the batch stride must clear 1 whole element extent, which is simpler than
    // a per-offset overlap test and never accepts a real overlap (it can only reject some
    // exotic layout that threads a later element through this one's internal gaps). The cast
    // below is sound because check_batched_view already rejected a negative c_batch_stride
    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: batched C element aliases itself (strides {},{} map distinct elements to \
             the same memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }
    if batch > 1 && (c_batch_stride as usize) < c_extent {
        panic!(
            "gemmkit: C batch stride ({c_batch_stride}) must be at least the element extent \
             ({c_extent}) so the batched C outputs stay disjoint"
        );
    }

    // C must not alias A or B; the borrow checker already forbids this in safe Rust, so the
    // check below is defensive
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len())
        || overlaps(cp, cl, b.data.as_ptr(), b.data.len())
    {
        panic!("gemmkit: batched C aliases A or B");
    }
}

/// Strided-batched GEMM: `C_b <- alpha*A_b*B_b + beta*C_b` for `b in 0..batch`, in 1 call,
/// parallelized across the batch rather than within each element. Every element shares the
/// single-element shape and strides of `a`/`b`/`c`; element `b` is based at
/// `a.data + b*a_batch_stride` (likewise for `b`/`c`). A `*_batch_stride` of `0` broadcasts 1
/// operand across the whole batch, valid for the read-only `A`/`B` but never for `C`. Uses the
/// thread-local workspace pool
///
/// Every element re-dispatches through the full engine, so the batch reproduces a loop of
/// [`gemm`] calls and stays reproducible across thread counts. The serial and batch-parallel
/// schedules run each element on a single worker, so those 2 are additionally bit-identical
/// across thread counts; the few-but-large schedule instead runs an element through the
/// parallel engine and inherits that route's own serial == parallel behavior
///
/// # Panics
/// If the per-element dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`); if any element view (including the last, `b == batch - 1`) addresses
/// outside its slice; if a batch stride is negative; if the `batch` output regions overlap each
/// other (`C` batch stride below the element extent) or a `C` element aliases itself; or if
/// `C`'s storage overlaps `A`'s or `B`'s
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched<T: GemmScalar>(
    batch: usize,
    alpha: T,
    a: MatRef<'_, T>,
    a_batch_stride: isize,
    b: MatRef<'_, T>,
    b_batch_stride: isize,
    beta: T,
    c: MatMut<'_, T>,
    c_batch_stride: isize,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| {
        gemm_batched_with(
            ws,
            batch,
            alpha,
            a,
            a_batch_stride,
            b,
            b_batch_stride,
            beta,
            c,
            c_batch_stride,
            par,
        );
    });
}

/// Like [`gemm_batched`] but reuses a caller-owned [`Workspace`]. The serial and few-but-large
/// schedules pack through `ws`; the batch-parallel schedule instead has each worker pack
/// through its own thread-local pool, since 1 `Workspace` cannot back concurrent packing from
/// several threads, reused across calls exactly like a caller-owned `ws`
///
/// # Panics
/// Same conditions as [`gemm_batched`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_with<T: GemmScalar>(
    ws: &mut Workspace,
    batch: usize,
    alpha: T,
    a: MatRef<'_, T>,
    a_batch_stride: isize,
    b: MatRef<'_, T>,
    b_batch_stride: isize,
    beta: T,
    c: MatMut<'_, T>,
    c_batch_stride: isize,
    par: Parallelism,
) {
    // batch == 0: nothing to run, so skip validating the (unused) views
    if batch == 0 {
        return;
    }

    validate_batched_views(
        batch,
        &a,
        a_batch_stride,
        &b,
        b_batch_stride,
        &c,
        c_batch_stride,
    );

    // SAFETY: validate_batched_views has confirmed the shapes, bounds, disjointness, and
    // non-aliasing above
    unsafe {
        gemm_batched_unchecked_with(
            ws,
            batch,
            a.rows,
            a.cols,
            b.cols,
            alpha,
            a.data.as_ptr(),
            a.rs,
            a.cs,
            a_batch_stride,
            b.data.as_ptr(),
            b.rs,
            b.cs,
            b_batch_stride,
            beta,
            c.data.as_mut_ptr(),
            c.rs,
            c.cs,
            c_batch_stride,
            par,
        );
    }
}

/// Strided-batched GEMM with a fused epilogue shared by every element:
/// `C_b <- act(alpha*A_b*B_b + beta*C_b + bias)` for `b in 0..batch`, in 1 call, parallelized
/// across the batch. 1 bias vector and 1 activation apply to every element (the
/// batched-linear-layer case: 1 layer applied to a batch of inputs). Shape, stride, and
/// broadcast conventions match [`gemm_batched`]. Uses the thread-local workspace pool;
/// `bias == None && act == None` takes the plain [`gemm_batched`] path
///
/// Each element re-dispatches through the full fused engine, so element `b`'s output is
/// bit-identical to a standalone [`gemm_fused`] call on that element with the same bias and
/// activation. For `f32`/`f64` that means bit-identical to `gemm()` followed by the same
/// scalar map, for every shape; for `f16`/`bf16` the epilogue applies in `f32` before the
/// single narrowing (more precise than a separate narrow map, so not bitwise-equal to
/// `gemm`-then-map there). Elements are independent, so the batch stays reproducible across
/// thread counts, with the same serial / batch-parallel bit-identical guarantee as
/// [`gemm_batched`]
///
/// # Panics
/// The [`gemm_batched`] conditions, plus: a `PerRow` bias whose length is not the element
/// `A.rows` (or a `PerCol` bias not the element `B.cols`), since the bias is 1 shared vector
/// sized for a single element, not `batch*axis`; a bias slice overlapping `C`'s storage; or a
/// non-finite `LeakyRelu` slope
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_fused<T: FusedScalar>(
    batch: usize,
    alpha: T,
    a: MatRef<'_, T>,
    a_batch_stride: isize,
    b: MatRef<'_, T>,
    b_batch_stride: isize,
    beta: T,
    c: MatMut<'_, T>,
    c_batch_stride: isize,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| {
        gemm_batched_fused_with(
            ws,
            batch,
            alpha,
            a,
            a_batch_stride,
            b,
            b_batch_stride,
            beta,
            c,
            c_batch_stride,
            bias,
            act,
            par,
        );
    });
}

/// Like [`gemm_batched_fused`] but reuses a caller-owned [`Workspace`] (the same split as
/// [`gemm_batched_with`]: serial and few-but-large schedules pack through `ws`, batch-parallel
/// through each worker's own thread-local pool)
///
/// # Panics
/// Same conditions as [`gemm_batched_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    batch: usize,
    alpha: T,
    a: MatRef<'_, T>,
    a_batch_stride: isize,
    b: MatRef<'_, T>,
    b_batch_stride: isize,
    beta: T,
    c: MatMut<'_, T>,
    c_batch_stride: isize,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    // batch == 0: nothing to run, so skip validating the (unused) views and bias
    if batch == 0 {
        return;
    }

    // No bias or activation: delegate to plain gemm_batched so no fused kernel is instantiated,
    // and both paths share 1 set of validation panics
    if bias.is_none() && act.is_none() {
        gemm_batched_with(
            ws,
            batch,
            alpha,
            a,
            a_batch_stride,
            b,
            b_batch_stride,
            beta,
            c,
            c_batch_stride,
            par,
        );
        return;
    }

    validate_batched_views(
        batch,
        &a,
        a_batch_stride,
        &b,
        b_batch_stride,
        &c,
        c_batch_stride,
    );

    // The bias is 1 shared vector sized for a single element (its length matches the element
    // axis, not batch*axis) and must not overlap C's whole backing slice
    validate_bias(&bias, a.rows, b.cols, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validate_batched_views and validate_bias confirmed shapes, bounds, disjointness,
    // non-aliasing, and a finite slope above; the bias borrow outlives this run_fused call
    unsafe {
        crate::special::batched::run_fused(
            batch,
            a.rows,
            a.cols,
            b.cols,
            alpha,
            a.data.as_ptr(),
            a.rs,
            a.cs,
            a_batch_stride,
            b.data.as_ptr(),
            b.rs,
            b.cs,
            b_batch_stride,
            beta,
            c.data.as_mut_ptr(),
            c.rs,
            c.cs,
            c_batch_stride,
            epi,
            par,
            ws,
        );
    }
}

/// The raw strided-batched fused engine: `C_e <- act(alpha*A_e*B_e + beta*C_e + bias)` for
/// `e in 0..batch`, over pointers and `isize` strides, with no bounds, alias, or shape checks:
/// the raw-parts form of [`gemm_batched_fused`], combining [`gemm_batched_unchecked`]'s
/// per-element shape with the shared bias/activation of [`gemm_fused_unchecked`]. Element `e`
/// is based at `a + e*a_batch_stride` / `b + e*b_batch_stride` / `c + e*c_batch_stride`, all
/// sharing the single-element shape `(m, k, n)` and element strides; the 1 `bias` (a
/// `(ptr, dim)` pair, read only when `has_bias`) and 1 `act` apply to every element. Uses the
/// thread-local workspace pool
///
/// # Safety
/// For every element `e in 0..batch`: `a`/`b` are valid for reads and `c` for read+write over
/// every `(i, j)` implied by `(m, k, n)` and the element strides at the batch-strided base; the
/// `batch` `C` regions are pairwise disjoint and none aliases any `A`/`B`; and when `beta == 0`,
/// `c` need not be initialized. A batch stride may be `0` (broadcast) only for the read-only
/// `A`/`B`, never `C`. When `has_bias`, `bias` is a single shared vector, valid for reads of `m`
/// (`PerRow`) or `n` (`PerCol`) elements, sized for 1 element rather than `batch*axis`, and
/// disjoint from every `C` element; a non-finite `LeakyRelu` slope is the caller's
/// responsibility (the checked API rejects it)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_fused_unchecked<T: FusedScalar>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    a_batch_stride: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    b_batch_stride: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    c_batch_stride: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_batched_fused_unchecked_with(
                ws,
                batch,
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                a_batch_stride,
                b,
                rsb,
                csb,
                b_batch_stride,
                beta,
                c,
                rsc,
                csc,
                c_batch_stride,
                bias,
                bias_dim,
                has_bias,
                act,
                par,
            );
        });
    }
}

/// As [`gemm_batched_fused_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_batched_fused_unchecked`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_fused_unchecked_with<T: FusedScalar>(
    ws: &mut Workspace,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    a_batch_stride: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    b_batch_stride: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    c_batch_stride: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        crate::special::batched::run_fused(
            batch,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            a_batch_stride,
            b,
            rsb,
            csb,
            b_batch_stride,
            beta,
            c,
            rsc,
            csc,
            c_batch_stride,
            epi,
            par,
            ws,
        );
    }
}

/// The raw strided-batched engine: [`gemm_batched`] over pointers and `isize` strides, with no
/// bounds, alias, or shape checks. Element `e` is based at `a + e*a_batch_stride` /
/// `b + e*b_batch_stride` / `c + e*c_batch_stride`, all sharing the single-element shape
/// `(m, k, n)` and element strides. Adapter crates (e.g. an ndarray `Array3` batched on axis 0)
/// and FFI callers that supply their own pointers or arbitrary strides use this path. Uses the
/// thread-local workspace pool
///
/// # Safety
/// For every element `e in 0..batch`: `a`/`b` are valid for reads and `c` for read+write over
/// every `(i, j)` implied by `(m, k, n)` and the element strides at the batch-strided base; the
/// `batch` `C` regions are pairwise disjoint and none aliases any `A`/`B`; and when `beta == 0`,
/// `c` need not be initialized. A batch stride may be `0` (broadcast) only for the read-only
/// `A`/`B`, never `C`
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_unchecked<T: GemmScalar>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    a_batch_stride: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    b_batch_stride: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    c_batch_stride: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions satisfied by the caller, per # Safety above
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_batched_unchecked_with(
                ws,
                batch,
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                a_batch_stride,
                b,
                rsb,
                csb,
                b_batch_stride,
                beta,
                c,
                rsc,
                csc,
                c_batch_stride,
                par,
            );
        });
    }
}

/// As [`gemm_batched_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_batched_unchecked`]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched_unchecked_with<T: GemmScalar>(
    ws: &mut Workspace,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    a_batch_stride: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    b_batch_stride: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    c_batch_stride: isize,
    par: Parallelism,
) {
    // SAFETY: caller guarantees valid, pairwise-disjoint, non-aliasing C regions per element,
    // and that beta == 0 may leave C uninitialized
    unsafe {
        crate::special::batched::run(
            batch,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            a_batch_stride,
            b,
            rsb,
            csb,
            b_batch_stride,
            beta,
            c,
            rsc,
            csc,
            c_batch_stride,
            par,
            ws,
        );
    }
}

/// Runs a pointer-array batched GEMM: every element in `problems` is an independent product
/// with its own shape and pointers ([`GemmProblem`]), parallelized across the batch (whole
/// GEMMs assigned to workers, each run serially and cache-hot). The raw counterpart of
/// [`gemm_batched_slice`], for callers (FFI, adapters) that validate their own inputs and may
/// use arbitrary pointers or negative strides. Deterministic across thread counts, since each
/// element runs wholly on 1 worker, and takes the `problems` slice as-is with no per-call
/// allocation
///
/// # Safety
/// For each problem: `a`/`b` valid for reads and `c` for read+write over the shape/strides; when
/// `beta == 0`, `c` need not be initialized. Across the batch: the `c` regions must be pairwise
/// disjoint and none may alias any `a`/`b` (concurrent writes)
pub unsafe fn gemm_batched_ptr_unchecked<T: GemmScalar>(
    problems: &[GemmProblem<T>],
    par: Parallelism,
) {
    // SAFETY: caller guarantees each problem's pointers are valid and the outputs are pairwise
    // disjoint and don't alias inputs
    unsafe {
        workspace::with_thread_pool(|ws| crate::special::batched::run_ptr(problems, par, ws));
    }
}

/// 1 element of a checked pointer-array batched GEMM ([`gemm_batched_slice`]):
/// `C <- alpha*A*B + beta*C` over safe views
pub struct BatchProblem<'a, T> {
    /// Product scale
    pub alpha: T,
    /// LHS view
    pub a: MatRef<'a, T>,
    /// RHS view
    pub b: MatRef<'a, T>,
    /// Accumulator scale
    pub beta: T,
    /// Output view: a distinct `&mut` borrow per element, so the batch's outputs can't overlap
    pub c: MatMut<'a, T>,
}

/// Runs a checked pointer-array batched GEMM: `problems[i].c <- alpha*A*B + beta*C` for each
/// element, each an independent product over safe views, parallelized across the batch. The
/// safe counterpart of [`gemm_batched_ptr_unchecked`]: because every `c` is a distinct
/// `MatMut`, the outputs are pairwise disjoint and can't alias the inputs by construction (the
/// borrow checker already forbids 2 overlapping `&mut` borrows), so validation only covers
/// per-element shape agreement and in-bounds strides. Deterministic across thread counts
///
/// # Panics
/// If any element's dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`, `B.cols != C.cols`),
/// a view addresses outside its slice, or an element's `C` aliases itself
pub fn gemm_batched_slice<T: GemmScalar>(problems: &mut [BatchProblem<'_, T>], par: Parallelism) {
    let raw: Vec<GemmProblem<T>> = problems
        .iter_mut()
        .enumerate()
        .map(|(i, p)| {
            assert_eq!(
                p.a.cols, p.b.rows,
                "gemmkit: batch element {i} A.cols ({}) != B.rows ({})",
                p.a.cols, p.b.rows
            );
            assert_eq!(
                p.a.rows, p.c.rows,
                "gemmkit: batch element {i} A.rows ({}) != C.rows ({})",
                p.a.rows, p.c.rows
            );
            assert_eq!(
                p.b.cols, p.c.cols,
                "gemmkit: batch element {i} B.cols ({}) != C.cols ({})",
                p.b.cols, p.c.cols
            );
            check_view(p.a.data, p.a.rows, p.a.cols, p.a.rs, p.a.cs, "A");
            check_view(p.b.data, p.b.rows, p.b.cols, p.b.rs, p.b.cs, "B");
            check_view(p.c.data, p.c.rows, p.c.cols, p.c.rs, p.c.cs, "C");
            if self_aliases(p.c.rows, p.c.cols, p.c.rs, p.c.cs) {
                panic!(
                    "gemmkit: batch element {i} C view aliases itself (strides {},{}); C must \
                     address each (i,j) uniquely",
                    p.c.rs, p.c.cs
                );
            }
            GemmProblem {
                m: p.a.rows,
                k: p.a.cols,
                n: p.b.cols,
                alpha: p.alpha,
                a: p.a.data.as_ptr(),
                rsa: p.a.rs,
                csa: p.a.cs,
                b: p.b.data.as_ptr(),
                rsb: p.b.rs,
                csb: p.b.cs,
                beta: p.beta,
                c: p.c.data.as_mut_ptr(),
                rsc: p.c.rs,
                csc: p.c.cs,
            }
        })
        .collect();
    // SAFETY: shapes validated above; distinct &mut C borrows vs & A/B mean the outputs are
    // pairwise disjoint and alias nothing, by construction, so the parallel writes are race-free
    workspace::with_thread_pool(|ws| unsafe { crate::special::batched::run_ptr(&raw, par, ws) });
}
