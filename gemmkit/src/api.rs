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

use crate::dispatch::{self, GemmScalar, PackedConsume, Task};
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

/// `true` if the byte ranges `[pa, pa+na)` and `[pb, pb+nb)` overlap.
fn overlaps<T>(pa: *const T, na: usize, pb: *const T, nb: usize) -> bool {
    let a0 = pa as usize;
    let a1 = a0 + na * core::mem::size_of::<T>();
    let b0 = pb as usize;
    let b1 = b0 + nb * core::mem::size_of::<T>();
    a0 < b1 && b0 < a1
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
/// `B` (the classic inference pattern: fixed weights, a stream of activation
/// batches). Produced by [`prepack_rhs`] and consumed by [`gemm_packed_b`] /
/// [`gemm_packed_b_with`], which skip the per-call RHS repack.
///
/// The packed layout is tied to the ISA tile and the blocking geometry the host
/// resolves for `(k, n)`, both recorded here; the consuming call re-derives them
/// at its real `m` and **panics** on any mismatch, so a panel can never be read
/// against a different tiling. Because the buffer is read-only during the GEMM it
/// is shared immutably across all worker threads with no extra synchronization.
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
    // Resolve the panel geometry through the *same* ISA tile the consuming call
    // will use; the `m = 65` sentinel dodges the tiny-matrix branch so `kc`/`nc`
    // are `m`-independent and match any (non-both-tiny) consume.
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    let blk = crate::cache::topology().blocking(mr, nr, core::mem::size_of::<T>().max(1), 65, n, k);
    let kc = blk.kc.max(1);
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    let total = n.div_ceil(nr) * nr * k;
    let mut buf = vec![T::ZERO; total];
    if total > 0 {
        // SAFETY: `buf` holds `ceil(n/nr)*nr*k` elements (the exact layout size);
        // `b` is validated in-bounds above; `pack_rhs_full` writes only that range.
        unsafe {
            crate::driver::pack_rhs_full::<crate::kernel::FloatGemm<T>>(
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
/// The result is **bit-identical** to a plain [`gemm`] on the same inputs for all
/// but very small (`m <= 64 && n <= 64`) products: those alone block via a
/// small-matrix shortcut that the prepacked path bypasses (it reuses the buffer's
/// own blocking), so the result is still numerically correct but may differ from
/// plain [`gemm`] in the last ULP. The output is bit-identical across thread counts
/// regardless.
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
