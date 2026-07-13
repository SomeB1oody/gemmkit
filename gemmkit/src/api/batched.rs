//! Strided- and pointer-array-batched GEMM entries.
use super::*;
use crate::dispatch::GemmProblem;
use alloc::vec::Vec;

/// Bounds check for a **strided-batched** view: the `batch` element views (element `bi` based at
/// slice offset `bi * batch_stride`) must all address inside `data`, including the last element.
/// The batch stride must be non-negative for the safe API (like the element strides). Returns the
/// element extent (highest offset + 1) so the caller can reuse it for the inter-element
/// disjointness check.
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
    // Only element 0 exists when batch <= 1, so the batch stride is irrelevant there.
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

/// Strided-batched GEMM: `C_b <- alpha·A_b·B_b + beta·C_b` for `b in 0..batch`, one call,
/// parallelized **across the batch**. All elements share the single-element shape and strides of
/// `a`/`b`/`c`; consecutive elements are `*_batch_stride` apart in their slices (`A_b` at
/// `a.data + b·a_batch_stride`, etc.). A `*_batch_stride` of `0` broadcasts one operand across the
/// whole batch (valid for the read-only `A`/`B`, never for `C`). Uses the thread-local workspace
/// pool.
///
/// Each element re-dispatches through the full engine, so the result **reproduces** a loop of
/// [`gemm`] calls and is **deterministic** across thread counts (reproducible under a fixed
/// config, the library's determinism contract). The serial and batch-parallel schedules run each
/// element serially, so they are additionally bit-identical across thread counts; the
/// few-but-large schedule runs each element through the parallel engine and so inherits its
/// per-element serial==parallel behavior.
///
/// # Panics
/// If the per-element dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`); if any element view (including the last, `b == batch-1`) addresses outside
/// its slice; if a batch stride is negative; if the `batch` output regions overlap each other
/// (`C` batch stride below the element extent) or a `C` element aliases itself; or if `C`'s
/// storage overlaps `A`'s or `B`'s.
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
/// schedules pack through `ws`; the batch-parallel schedule instead packs through each worker's
/// own persistent per-thread batched pool (a single shared `ws` cannot back concurrent packing),
/// which is reused across calls the same way `ws` is.
///
/// # Panics
/// Same conditions as [`gemm_batched`].
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
    // A zero-length batch is a pure no-op — nothing to validate (the views are unused).
    if batch == 0 {
        return;
    }

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

    // Each C element must address every (i,j) uniquely (a self-aliasing element would race in
    // parallel), and — since the batch writes them concurrently — the elements must not overlap
    // each other. Disjointness is enforced conservatively: the batch stride must clear one
    // element's whole extent (sufficient, and simpler than a per-offset overlap test — it can
    // reject an exotic layout that threads later elements through a strided element's internal
    // gaps, but never accepts a real overlap). (`c_batch_stride >= 0` here: `check_batched_view`.)
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

    // C must not alias A or B (it is written). The whole-slice check is defensive — safe Rust's
    // borrow checker already forbids overlapping &mut/& slices.
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len())
        || overlaps(cp, cl, b.data.as_ptr(), b.data.len())
    {
        panic!("gemmkit: batched C aliases A or B");
    }

    // SAFETY: validated above — per-element shapes agree, every element view (incl. the last) is
    // in bounds, the C outputs are pairwise disjoint and address uniquely, and C does not alias
    // A/B.
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

/// The raw strided-batched engine: [`gemm_batched`] over pointers and `isize` strides, with **no**
/// bounds/alias/shape checks. Element `e` is based at `a + e·a_batch_stride` / `b + e·b_batch_stride`
/// / `c + e·c_batch_stride`, all sharing the single-element shape `(m, k, n)` and element strides.
/// The raw counterpart of [`gemm_batched`], for adapters (e.g. an ndarray `Array3`, batch on axis 0)
/// and FFI callers that supply their own valid pointers / arbitrary strides. Uses the thread-local
/// workspace pool.
///
/// # Safety
/// For every element `e in 0..batch`: `a`/`b` are valid for reads and `c` for read+write over every
/// `(i, j)` implied by `(m, k, n)` and the element strides at the batch-strided base; the `batch`
/// `C` regions are pairwise disjoint and none aliases any `A`/`B`; and when `beta == 0`, `c` need not
/// be initialized. A batch stride may be `0` (broadcast) only for the read-only `A`/`B`, never `C`.
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
    // SAFETY: preconditions forwarded to the caller (see this fn's # Safety).
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

/// As [`gemm_batched_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_batched_unchecked`].
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
    // SAFETY: the caller guarantees per-element pointer validity, pairwise-disjoint `C` regions that
    // don't alias `A`/`B`, and that `beta == 0` may leave `C` uninitialized (see # Safety).
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

/// Run a **pointer-array batched** GEMM: every element in `problems` is an independent product with
/// its own shape and pointers ([`GemmProblem`]), parallelized across the batch (whole GEMMs assigned
/// to workers, each run serially and cache-hot). The raw counterpart of [`gemm_batched_slice`], for
/// callers (FFI, adapters) that validate their own inputs and may use arbitrary pointers / negative
/// strides. Deterministic across thread counts (each element runs wholly on one worker), and it
/// takes the `problems` slice as-is — no per-call allocation.
///
/// # Safety
/// For each problem: `a`/`b` valid for reads and `c` for read+write over the shape/strides; when
/// `beta == 0`, `c` need not be initialized. Across the batch: the `c` regions must be pairwise
/// disjoint and none may alias any `a`/`b` (concurrent writes).
pub unsafe fn gemm_batched_ptr_unchecked<T: GemmScalar>(
    problems: &[GemmProblem<T>],
    par: Parallelism,
) {
    // SAFETY: the caller guarantees each problem's pointers are valid and the outputs are pairwise
    // disjoint and don't alias inputs.
    unsafe {
        workspace::with_thread_pool(|ws| crate::special::batched::run_ptr(problems, par, ws));
    }
}

/// One element of a checked pointer-array batched GEMM ([`gemm_batched_slice`]):
/// `C <- alpha·A·B + beta·C` over safe views.
pub struct BatchProblem<'a, T> {
    /// Product scale.
    pub alpha: T,
    /// LHS view.
    pub a: MatRef<'a, T>,
    /// RHS view.
    pub b: MatRef<'a, T>,
    /// Accumulator scale.
    pub beta: T,
    /// Output view (a distinct `&mut` per element ⇒ the batch's outputs are disjoint).
    pub c: MatMut<'a, T>,
}

/// Run a **checked pointer-array batched** GEMM: `problems[i].c <- α·A·B + β·C` for each element,
/// each an independent product over safe views, parallelized across the batch. The safe counterpart
/// of [`gemm_batched_ptr_unchecked`]: because every `c` is a distinct `MatMut`, the outputs are
/// pairwise disjoint and don't alias the inputs *by construction*, so validation only covers
/// per-element shape agreement and in-bounds strides. Deterministic across thread counts.
///
/// # Panics
/// If any element's dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`, `B.cols != C.cols`),
/// a view addresses outside its slice, or an element's `C` aliases itself.
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
    // SAFETY: each element validated above; the outputs are pairwise disjoint and don't alias the
    // inputs by construction (distinct `&mut` C vs `&` A/B), so the parallel writes are race-free.
    workspace::with_thread_pool(|ws| unsafe { crate::special::batched::run_ptr(&raw, par, ws) });
}
