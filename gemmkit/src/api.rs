//! Public core API (layer L8a).
//!
//! Two layers of safety over the same engine:
//!
//! * [`gemm`] / [`gemm_with`] — safe `&[T]` + stride views. Shape mismatches,
//!   out-of-bounds strides, and C aliasing A/B all **panic** before any unsafe
//!   work runs.
//! * [`gemm_unchecked`] — the raw pointer + `isize` stride engine for advanced
//!   callers (e.g. the ndarray adapter), which validate their own inputs.
//!
//! Semantics are exactly `C <- alpha·A·B + beta·C`. Transposition is expressed
//! through strides (a transposed view swaps `rs`/`cs`, no copy). When `beta == 0`
//! the output C is **not read**, so it may be uninitialized.

#[cfg(feature = "complex")]
use crate::dispatch::ComplexScalar;
use crate::dispatch::{self, GemmProblem, GemmScalar, PackedConsume, Task};
use crate::parallel::Parallelism;
use crate::workspace::{self, Workspace};

/// An immutable strided matrix view over a slice.
///
/// Element `(i, j)` lives at slice offset `i*rs + j*cs`. Strides must be
/// non-negative for the safe API (use [`gemm_unchecked`] for negative strides /
/// pointers into the middle of a buffer).
#[derive(Copy, Clone)]
pub struct MatRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

/// A mutable strided matrix view over a slice.
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

impl<'a, T> MatRef<'a, T> {
    /// A view with explicit strides. Panics in [`gemm`] if the strides address
    /// outside `data`.
    pub fn new(data: &'a [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) `rows × cols` view.
    pub fn from_row_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, cols as isize, 1)
    }
    /// A column-major (Fortran-order) `rows × cols` view.
    pub fn from_col_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, 1, rows as isize)
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, T> MatMut<'a, T> {
    /// A mutable view with explicit strides.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) mutable view.
    pub fn from_row_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let cs = cols as isize;
        Self::new(data, rows, cols, cs, 1)
    }
    /// A column-major (Fortran-order) mutable view.
    pub fn from_col_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let rs = rows as isize;
        Self::new(data, rows, cols, 1, rs)
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Highest slice offset (exclusive) and lowest offset reached by a strided view,
/// or `None` if the strides are negative (unsupported by the safe API).
fn extent(rows: usize, cols: usize, rs: isize, cs: isize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let mut lo: isize = 0;
    let mut hi: isize = 0;
    for &(dim, s) in &[(rows, rs), (cols, cs)] {
        let e = (dim as isize - 1) * s;
        if e < 0 {
            lo += e;
        } else {
            hi += e;
        }
    }
    if lo < 0 {
        None // negative strides — not allowed in the safe API
    } else {
        Some(hi as usize + 1)
    }
}

fn check_view<T>(data: &[T], rows: usize, cols: usize, rs: isize, cs: isize, name: &str) {
    match extent(rows, cols, rs, cs) {
        Some(need) if need <= data.len() => {}
        Some(need) => panic!(
            "gemmkit: {name} view of {rows}x{cols} (strides {rs},{cs}) needs {need} elements but slice has {}",
            data.len()
        ),
        None => panic!("gemmkit: {name} view has negative strides; use gemm_unchecked"),
    }
}

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
        None => panic!("gemmkit: {name} view has negative strides; use the unchecked API"),
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

/// `true` if a strided `rows×cols` view maps two *distinct* `(i,j)` to the same
/// offset. Such a view is fine to read from (a broadcast input) but invalid as an
/// output: the parallel driver assumes output tiles are disjoint, so writing
/// through it would be a data race. Strides are taken by magnitude (negative
/// strides are already rejected by [`extent`]). A dimension of length ≤ 1 spans
/// nothing, so its stride is irrelevant; for two real dimensions, no collision is
/// possible exactly when the larger stride clears the smaller dimension's whole
/// span (`big ≥ small_stride · small_dim`).
fn self_aliases(rows: usize, cols: usize, rs: isize, cs: isize) -> bool {
    if rows == 0 || cols == 0 {
        return false; // empty view: nothing is written, so nothing can race
    }
    let r = (rows > 1).then_some((rs.unsigned_abs(), rows));
    let c = (cols > 1).then_some((cs.unsigned_abs(), cols));
    match (r, c) {
        (None, None) => false,
        (Some((s, _)), None) | (None, Some((s, _))) => s == 0,
        (Some(a), Some(b)) => {
            let (sm, big) = if a.0 <= b.0 { (a, b.0) } else { (b, a.0) };
            sm.0 == 0 || big < sm.0.saturating_mul(sm.1)
        }
    }
}

/// `true` if the byte ranges of two `[T]`-typed views overlap (same element type;
/// the heterogeneous integer API uses [`overlaps_bytes`] directly).
fn overlaps<T>(pa: *const T, na: usize, pb: *const T, nb: usize) -> bool {
    let s = core::mem::size_of::<T>();
    overlaps_bytes(pa as *const u8, na, s, pb as *const u8, nb, s)
}

/// `C <- alpha·A·B + beta·C` over safe slice views, using the thread-local
/// workspace pool.
///
/// # Panics
/// If the inner dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if any view addresses outside its slice, or if `C`'s
/// storage overlaps `A`'s or `B`'s.
pub fn gemm<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_with(ws, alpha, a, b, beta, c, par));
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`] — zero heap allocation
/// after the first sufficiently large call.
///
/// # Panics
/// Same conditions as [`gemm`].
pub fn gemm_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
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

    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");

    // C is written, so its strides must address each (i,j) uniquely. A
    // self-aliasing output (e.g. `rsc == 0`) would otherwise become a data race
    // in parallel mode — reachable from entirely safe code. (A/B may alias
    // themselves: they are only read, so broadcast strides are allowed there.)
    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C must not alias A or B (it is written). In safe Rust the borrow checker
    // already forbids overlapping &mut/& slices; this is a defensive guarantee.
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len())
        || overlaps(cp, cl, b.data.as_ptr(), b.data.len())
    {
        panic!("gemmkit: C aliases A or B");
    }

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    // SAFETY: validated above — shapes agree, every stride stays in bounds, and
    // C does not alias A/B.
    unsafe {
        dispatch::execute(
            Task {
                m,
                k,
                n,
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
        crate::special::batched::run(
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

/// The raw engine: `C <- alpha·A·B + beta·C` over pointers and `isize` strides,
/// with **no** bounds/alias/shape checks. Uses the thread-local workspace pool.
///
/// # Safety
/// The caller guarantees: `a`/`b` are valid for reads and `c` for read+write
/// over every `(i,j)` implied by the dimensions and strides; `c` does not alias
/// `a`/`b`; and when `beta == 0`, `c` need not be initialized.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_unchecked<T: GemmScalar>(
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
    par: Parallelism,
) {
    unsafe {
        workspace::with_thread_pool(|ws| {
            dispatch::execute(
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
        });
    }
}

/// A right-hand-side matrix pre-packed once into gemmkit's internal
/// micropanel-major layout, for reuse across many products that share the same
/// `B` (the inference pattern: fixed weights, a stream of activation batches).
/// Produced by [`prepack_rhs`] and consumed by [`gemm_packed_b`] /
/// [`gemm_packed_b_with`], which skip the per-call RHS repack.
///
/// The buffer records the blocking geometry it was built for; the consuming call
/// reads it back verbatim, so a panel is always read against its own tiling. It is
/// read-only during the GEMM, so it is shared across worker threads with no
/// synchronization.
pub struct PackedRhs<T> {
    buf: Vec<T>,
    k: usize,
    n: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedRhs<T> {
    /// Rows of the original `B` (the shared `k` dimension).
    pub fn rows(&self) -> usize {
        self.k
    }
    /// Columns of the original `B` (the `n` dimension).
    pub fn cols(&self) -> usize {
        self.n
    }
}

/// Pre-pack a `k × n` RHS into [`PackedRhs`] for reuse across many [`gemm_packed_b`]
/// calls. The pack happens once, single-threaded, here; later products skip it.
///
/// Any layout of `B` is accepted (the pack reads it through its strides). The
/// resulting buffer is valid for products whose `(k, n)` match this `B` and whose
/// `C` is column-major-ish (`|csc| >= |rsc|`); [`gemm_packed_b`] enforces both.
///
/// # Panics
/// If `B`'s view addresses outside its slice (same bounds check as [`gemm`]).
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    let (k, n) = (b.rows, b.cols);
    // Resolve the panel geometry through the same ISA tile the consuming call will
    // use; the `m = 65` sentinel dodges the tiny-matrix branch so the geometry is
    // `m`-independent (the consume reads it back verbatim). Block with the packed
    // input (`Lhs` == `Rhs`) element size — the unit the panels are stored in, same
    // as the driver
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    let lhs_size = core::mem::size_of::<T>().max(1);
    let blk = crate::cache::topology().blocking(mr, nr, lhs_size, 65, n, k);
    let kc = if T::OUT_IS_ACC {
        blk.kc.max(1)
    } else {
        k.max(1)
    };
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    // A dot kernel (bf16 `vdpbf16ps`) packs depth in groups, so the panel depth is rounded
    // up to its `DEPTH_MULTIPLE`; `1` (every other kernel) leaves this unchanged.
    let k_pad = k.next_multiple_of(<T as GemmScalar>::rhs_depth_multiple());
    let total = n.div_ceil(nr) * nr * k_pad;
    let mut buf = vec![T::ZERO; total];
    if total > 0 {
        // SAFETY: `buf` holds `ceil(n/nr)*nr*k_pad` elements (the exact layout size, with
        // the depth padded to the dispatched family's `DEPTH_MULTIPLE`); `b` is validated
        // in-bounds above; `pack_rhs_full` writes only that range.
        // `GemmScalar::pack_rhs_full` selects the right kernel family per type.
        unsafe {
            T::pack_rhs_full(
                buf.as_mut_ptr(),
                b.data.as_ptr(),
                b.rs,
                b.cs,
                k,
                n,
                kc,
                nc,
                nr,
            );
        }
    }
    PackedRhs {
        buf,
        k,
        n,
        nr,
        kc,
        nc,
    }
}

/// `C <- alpha·A·B + beta·C` reusing a [`PackedRhs`] (pre-packed `B`), via the
/// thread-local workspace pool. Skips the per-call RHS repack.
///
/// The result **reproduces** a plain [`gemm`] under the same config, except in two
/// cases that stay correct but may differ in the last ULP: very small
/// (`m <= 64 && n <= 64`) products and gemv-shaped (`m == 1` or `n == 1`) products.
/// Output is deterministic across thread counts regardless.
///
/// # Panics
/// If the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `A` or `C` addresses outside its slice, if `C` aliases
/// itself or `A`, or if `C` is **not** column-major-ish (`|csc| >= |rsc|`) — a
/// row-major `C` would make the engine swap `A`/`B`, which a prepacked `B` cannot
/// support (use plain [`gemm`] there).
pub fn gemm_packed_b<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_packed_b_with(ws, alpha, a, packed, beta, c, par));
}

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_packed_b`].
pub fn gemm_packed_b_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    assert_eq!(
        a.cols, packed.k,
        "gemmkit: A.cols ({}) != packed B.rows ({})",
        a.cols, packed.k
    );
    assert_eq!(
        packed.n, c.cols,
        "gemmkit: packed B.cols ({}) != C.cols ({})",
        packed.n, c.cols
    );
    assert_eq!(
        a.rows, c.rows,
        "gemmkit: A.rows ({}) != C.rows ({})",
        a.rows, c.rows
    );

    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");

    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C must not alias A (it is written). The prepacked B is a separate owned
    // buffer, so it cannot alias C.
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len()) {
        panic!("gemmkit: C aliases A");
    }

    // A prepacked B is only valid for the no-swap orientation. The engine computes
    // Cᵀ = Bᵀ·Aᵀ when C is row-major-ish (|csc| < |rsc|), which would make the
    // prepacked operand play the LHS role — wrong layout. Require column-major-ish
    // C and direct callers to plain `gemm` otherwise.
    assert!(
        c.cs.unsigned_abs() >= c.rs.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );

    // SAFETY: validated above — shapes agree, A/C strides are in bounds, C does not
    // alias A, and the packed buffer (owned by `packed`, read-only) outlives the
    // call and matches the recorded (nr, kc, nc) geometry.
    unsafe {
        dispatch::execute_packed(
            PackedConsume {
                m: a.rows,
                k: a.cols,
                n: packed.n,
                alpha,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                packed: packed.buf.as_ptr(),
                nr: packed.nr,
                kc: packed.kc,
                nc: packed.nc,
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

/// A left-hand-side matrix pre-packed once into gemmkit's internal
/// micropanel-major layout, for reuse across many products that share the same
/// `A` (a fixed weight matrix `A` against a stream of differently-shaped right
/// operands `B`). Produced by [`prepack_lhs`] and consumed by [`gemm_packed_a`] /
/// [`gemm_packed_a_with`], which skip the per-call LHS repack.
///
/// By the engine's A/B symmetry, a prepacked LHS is the prepacked RHS of the
/// transposed product `Cᵀ = Bᵀ·Aᵀ`; the buffer records that transposed problem's
/// blocking geometry, which the consuming call (driven transposed) reads back
/// verbatim. Read-only during the GEMM, so it is shared across worker threads with
/// no synchronization.
pub struct PackedLhs<T> {
    buf: Vec<T>,
    m: usize,
    k: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedLhs<T> {
    /// Rows of the original `A` (the `m` dimension).
    pub fn rows(&self) -> usize {
        self.m
    }
    /// Columns of the original `A` (the shared `k` dimension).
    pub fn cols(&self) -> usize {
        self.k
    }
}

/// Pre-pack an `m × k` LHS into [`PackedLhs`] for reuse across many [`gemm_packed_a`]
/// calls. The pack happens once, single-threaded, here; later products skip it.
///
/// Any layout of `A` is accepted (the pack reads it through its strides). The
/// resulting buffer is valid for products whose `(m, k)` match this `A` and whose
/// `C` is row-major-ish (`|csc| <= |rsc|`); [`gemm_packed_a`] enforces both.
///
/// # Panics
/// If `A`'s view addresses outside its slice (same bounds check as [`gemm`]).
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    let (m, k) = (a.rows, a.cols);
    // Resolve the panel geometry through the consuming call's ISA tile, for the
    // *transposed* problem (whose RHS is this A): its `N` is the LHS's `m` rows, its
    // `K` is `k`. The `m = 65` sentinel for the transposed `M` (the unknown-here `n`)
    // dodges the tiny-matrix branch so the geometry is `n`-independent.
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    // Packed input (`Lhs` == `Rhs`) element size — the unit the panels are stored in,
    // matching the driver (the prepacked buffer below is `T`-typed).
    let lhs_size = core::mem::size_of::<T>().max(1);
    let blk = crate::cache::topology().blocking(mr, nr, lhs_size, 65, m, k);
    let kc = if T::OUT_IS_ACC {
        blk.kc.max(1)
    } else {
        k.max(1)
    };
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    // Depth-padded for a dot kernel (see `prepack_rhs`); identity for `DEPTH_MULTIPLE == 1`.
    let k_pad = k.next_multiple_of(<T as GemmScalar>::rhs_depth_multiple());
    let total = m.div_ceil(nr) * nr * k_pad;
    let mut buf = vec![T::ZERO; total];
    if total > 0 {
        // SAFETY: `buf` holds `ceil(m/nr)*nr*k_pad` elements (the exact layout size, with
        // the depth padded to the dispatched family's `DEPTH_MULTIPLE`); `a` is validated
        // in-bounds above; `pack_lhs_full` writes only that range and selects the right
        // kernel family per type.
        unsafe {
            T::pack_lhs_full(
                buf.as_mut_ptr(),
                a.data.as_ptr(),
                a.rs,
                a.cs,
                m,
                k,
                kc,
                nc,
                nr,
            );
        }
    }
    PackedLhs {
        buf,
        m,
        k,
        nr,
        kc,
        nc,
    }
}

/// `C <- alpha·A·B + beta·C` reusing a [`PackedLhs`] (pre-packed `A`), via the
/// thread-local workspace pool. Skips the per-call LHS repack.
///
/// The result **reproduces** a plain [`gemm`] under the same config, except in two
/// cases that stay correct but may differ in the last ULP: very small
/// (`m <= 64 && n <= 64`) products and gemv-shaped (`m == 1` or `n == 1`) products.
/// Output is deterministic across thread counts regardless.
///
/// # Panics
/// If the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `B` or `C` addresses outside its slice, if `C` aliases
/// itself or `B`, or if `C` is **not** row-major-ish (`|csc| <= |rsc|`) — a
/// column-major `C` would leave `A` in the genuine LHS role, which a prepacked `A`
/// (laid out as the transposed RHS) cannot serve (use plain [`gemm`] there).
pub fn gemm_packed_a<T: GemmScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_packed_a_with(ws, alpha, packed, b, beta, c, par));
}

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_packed_a`].
pub fn gemm_packed_a_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    assert_eq!(
        packed.k, b.rows,
        "gemmkit: packed A.cols ({}) != B.rows ({})",
        packed.k, b.rows
    );
    assert_eq!(
        packed.m, c.rows,
        "gemmkit: packed A.rows ({}) != C.rows ({})",
        packed.m, c.rows
    );
    assert_eq!(
        b.cols, c.cols,
        "gemmkit: B.cols ({}) != C.cols ({})",
        b.cols, c.cols
    );

    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");

    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C must not alias B (it is written). The prepacked A is a separate owned
    // buffer, so it cannot alias C.
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, b.data.as_ptr(), b.data.len()) {
        panic!("gemmkit: C aliases B");
    }

    // A prepacked A is only valid for the orientation in which A keeps the RHS role
    // of the transposed product. The engine computes `Cᵀ = Bᵀ·Aᵀ` exactly when C is
    // row-major-ish (`|csc| < |rsc|`); a column-major-ish C would leave A as the
    // genuine LHS — the wrong role for a buffer packed as the transposed RHS.
    // Require row-major-ish C and direct callers to plain `gemm` otherwise.
    assert!(
        c.cs.unsigned_abs() <= c.rs.unsigned_abs(),
        "gemmkit: gemm_packed_a requires row-major-ish C (|csc| <= |rsc|); a column-major C \
         would keep A in the LHS role and invalidate the prepacked LHS — use gemm() for that layout"
    );

    // Reframe as the transposed product `Cᵀ = Bᵀ·(prepacked Aᵀ)` and reuse the
    // existing prepacked-*RHS* engine — a prepacked LHS *is* the prepacked RHS of
    // that transpose. So the genuine `B` becomes the in-place LHS (`a`) with its
    // strides swapped (Bᵀ row/col = B col/row), the prepacked `A` stays the `packed`
    // RHS, the dims swap (`m↔n`), and `C`'s strides swap so the driver writes Cᵀ down
    // contiguous columns. The required row-major-ish C makes that transposed Cᵀ
    // column-major-ish — the no-extra-swap orientation `execute_packed` expects, so
    // no new dispatch path is needed.
    //
    // SAFETY: validated above — shapes agree, B/C strides are in bounds, C does not
    // alias B, and the packed buffer (owned by `packed`, read-only) outlives the call
    // and matches the recorded (nr, kc, nc) geometry.
    unsafe {
        dispatch::execute_packed(
            PackedConsume {
                m: b.cols,
                k: packed.k,
                n: packed.m,
                alpha,
                a: b.data.as_ptr(),
                rsa: b.cs,
                csa: b.rs,
                packed: packed.buf.as_ptr(),
                nr: packed.nr,
                kc: packed.kc,
                nc: packed.nc,
                beta,
                c: c.data.as_mut_ptr(),
                rsc: c.cs,
                csc: c.rs,
            },
            par,
            ws,
        );
    }
}

/// `true` if two byte ranges (base pointer + element count + element size) overlap.
/// The heterogeneous analogue of [`overlaps`] for the integer API, where C (`i32`)
/// and A/B (`i8`) have different element sizes.
fn overlaps_bytes(
    pa: *const u8,
    na: usize,
    sa: usize,
    pb: *const u8,
    nb: usize,
    sb: usize,
) -> bool {
    let a0 = pa as usize;
    let a1 = a0 + na * sa;
    let b0 = pb as usize;
    let b1 = b0 + nb * sb;
    a0 < b1 && b0 < a1
}

/// Integer GEMM: `C <- alpha·A·B + beta·C` with **`i8` inputs accumulated into an
/// `i32` output** (`alpha`/`beta`/`C` are `i32`). Arithmetic wraps on overflow, the
/// conventional integer-GEMM semantics. Uses the thread-local workspace pool.
///
/// A separate entry point from [`gemm`] because input and output types differ
/// (`i8` vs `i32`), which the homogeneous `gemm<T>` surface cannot express.
///
/// # Panics
/// Same shape / bounds / aliasing conditions as [`gemm`] (`A.cols == B.rows`,
/// `A.rows == C.rows`, `B.cols == C.cols`; every view in bounds; `C` addresses each
/// element uniquely and does not overlap `A`/`B`). Negative-stride / raw-pointer
/// callers use [`gemm_i8_unchecked`] (the homogeneous [`gemm_unchecked`] cannot
/// serve `i8 -> i32`).
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

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_i8`].
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

    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");

    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C (i32) must not alias A or B (i8) — heterogeneous element sizes.
    let cp = c.data.as_ptr() as *const u8;
    let cl = c.data.len();
    if overlaps_bytes(cp, cl, 4, a.data.as_ptr() as *const u8, a.data.len(), 1)
        || overlaps_bytes(cp, cl, 4, b.data.as_ptr() as *const u8, b.data.len(), 1)
    {
        panic!("gemmkit: C aliases A or B");
    }

    // SAFETY: validated above — shapes agree, every stride in bounds, C addresses
    // each (i,j) uniquely and does not overlap A/B.
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
    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");
    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len())
        || overlaps(cp, cl, b.data.as_ptr(), b.data.len())
    {
        panic!("gemmkit: C aliases A or B");
    }

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

/// The raw integer engine: `C(i32) <- alpha·A(i8)·B(i8) + beta·C` over pointers and
/// `isize` strides, with **no** bounds/alias/shape checks — the heterogeneous
/// counterpart of [`gemm_unchecked`] (which is typed for the homogeneous surface and
/// cannot serve `i8 -> i32`). The escape hatch [`gemm_i8`] points negative-stride /
/// advanced callers to. Uses the thread-local workspace pool.
///
/// # Safety
/// The caller guarantees `a`/`b` valid for reads and `c` for read+write over every
/// `(i,j)` implied by the dimensions and strides; `c` does not alias `a`/`b`; and
/// when `beta == 0`, `c` need not be initialized.
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

/// As [`gemm_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_unchecked`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_unchecked_with<T: GemmScalar>(
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
    par: Parallelism,
) {
    unsafe {
        dispatch::execute(
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
