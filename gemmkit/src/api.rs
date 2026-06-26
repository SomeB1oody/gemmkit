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

use crate::dispatch::{self, GemmScalar, Task};
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
