//! Prepacked-operand (`PackedLhs`/`PackedRhs`) entries.
use super::*;
use crate::dispatch::PackedConsume;
use alloc::vec;
use alloc::vec::Vec;

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
/// If `B`'s view addresses outside its slice (same bounds check as [`gemm`]),
/// or if `B` is so large (broadcast strides allow logical dimensions up to
/// `isize::MAX`) that the pack buffer size overflows `usize`.
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    // SAFETY: `b` is validated in-bounds directly above.
    unsafe { prepack_rhs_unchecked(b.data.as_ptr(), b.rs, b.cs, b.rows, b.cols) }
}

/// As [`prepack_rhs`] but over a raw `k × n` `B` pointer + strides, with **no** bounds check — the
/// raw counterpart for adapters / FFI that validate their own inputs.
///
/// # Safety
/// `b` must be valid for reads at every offset `i·rsb + j·csb`, for `i in 0..k` and `j in 0..n`.
pub unsafe fn prepack_rhs_unchecked<T: GemmScalar>(
    b: *const T,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
) -> PackedRhs<T> {
    // Resolve the panel geometry through the same ISA tile the consuming call will
    // use; the `tiny_block_dim() + 1` sentinel row count dodges the tiny-matrix branch
    // so the geometry is `m`-independent (the consume reads it back verbatim). Block
    // with the packed input (`Lhs` == `Rhs`) element size — the unit the panels are
    // stored in, same as the driver.
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    // An empty operand packs to nothing. Short-circuit before the geometry/size
    // arithmetic, which would otherwise overflow for a huge free dimension — an
    // empty view's extent is 0, so `check_view` accepts e.g. a `0 x usize::MAX` B.
    // The consume path never reads the buffer for a `k == 0`/`n == 0` problem
    // (it only beta-scales C), so an empty pack round-trips. Mirrors the
    // zero-batch early return in `gemm_batched`.
    if k == 0 || n == 0 {
        return PackedRhs {
            buf: Vec::new(),
            k,
            n,
            nr,
            kc: 1,
            nc: nr,
        };
    }
    let lhs_size = core::mem::size_of::<T>().max(1);
    let dodge_tiny = crate::tuning::tiny_block_dim().saturating_add(1);
    let blk = crate::cache::topology().blocking(mr, nr, lhs_size, dodge_tiny, n, k);
    let kc = if T::OUT_IS_ACC {
        blk.kc.max(1)
    } else {
        k.max(1)
    };
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    // A dot kernel (bf16 `vdpbf16ps`) packs depth in groups, so the panel depth is rounded
    // up to its `DEPTH_MULTIPLE`; `1` (every other kernel) leaves this unchanged.
    let k_pad = k.next_multiple_of(<T as GemmScalar>::rhs_depth_multiple());
    // Checked: a broadcast (zero-stride) view passes `check_view` with a tiny
    // backing slice, so a logically huge `n`/`k` can reach this product; a wrapped
    // size would under-allocate the buffer the pack then writes past.
    let total = n
        .div_ceil(nr)
        .checked_mul(nr)
        .and_then(|v| v.checked_mul(k_pad))
        .unwrap_or_else(|| {
            panic!("gemmkit: prepacked RHS of {k}x{n} is too large; the pack buffer size overflows usize")
        });
    let mut buf = vec![T::ZERO; total];
    if total > 0 {
        // SAFETY: `buf` holds `ceil(n/nr)*nr*k_pad` elements (the exact layout size, with
        // the depth padded to the dispatched family's `DEPTH_MULTIPLE`); `b` is caller-promised
        // valid for the `(k, n)` strided reads; `pack_rhs_full` writes only that range.
        // `GemmScalar::pack_rhs_full` selects the right kernel family per type.
        unsafe {
            T::pack_rhs_full(buf.as_mut_ptr(), b, rsb, csb, k, n, kc, nc, nr);
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
/// cases that stay correct but may differ in the last ULP: very small products (both
/// `m` and `n` at or below [`crate::tuning::tiny_block_dim`], default 64) and gemv-shaped
/// (`m == 1` or `n == 1`) products. Output is deterministic across thread counts regardless.
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

    // SAFETY: A/C strides are in bounds and C does not alias A (checked above).
    unsafe {
        gemm_packed_b_unchecked_with(
            ws,
            alpha,
            a.rows,
            a.data.as_ptr(),
            a.rs,
            a.cs,
            packed,
            beta,
            c.data.as_mut_ptr(),
            c.rs,
            c.cs,
            par,
        );
    }
}

/// As [`gemm_packed_b`] but over raw `A`/`C` pointers + strides, with **no** bounds/alias checks.
/// The shared `k` and output `n` come from `packed`; `m` is A's rows (= C's rows). Uses the
/// thread-local workspace pool.
///
/// # Safety
/// `a` valid for reads over `(m, packed.rows())` and `c` for read+write over `(m, packed.cols())`
/// at the given strides; `c` does not alias `a`; and when `beta == 0`, `c` need not be initialized.
///
/// # Panics
/// If `C` is not column-major-ish (`|csc| >= |rsc|`) — a prepacked RHS cannot serve a row-major C
/// (use plain [`gemm`] for that layout).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_b_unchecked<T: GemmScalar>(
    alpha: T,
    m: usize,
    a: *const T,
    rsa: isize,
    csa: isize,
    packed: &PackedRhs<T>,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety).
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_packed_b_unchecked_with(ws, alpha, m, a, rsa, csa, packed, beta, c, rsc, csc, par);
        });
    }
}

/// As [`gemm_packed_b_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_packed_b_unchecked`].
///
/// # Panics
/// See [`gemm_packed_b_unchecked`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_b_unchecked_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    m: usize,
    a: *const T,
    rsa: isize,
    csa: isize,
    packed: &PackedRhs<T>,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // A prepacked B is only valid for the no-swap orientation
    assert!(
        csc.unsigned_abs() >= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );
    // SAFETY: the caller guarantees A/C validity and that C does not alias A (see # Safety); the
    // packed buffer (owned by `packed`, read-only) outlives the call and matches its recorded
    // (nr, kc, nc) geometry.
    unsafe {
        dispatch::execute_packed(
            PackedConsume {
                m,
                k: packed.k,
                n: packed.n,
                alpha,
                a,
                rsa,
                csa,
                packed: packed.buf.as_ptr(),
                nr: packed.nr,
                kc: packed.kc,
                nc: packed.nc,
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
/// If `A`'s view addresses outside its slice (same bounds check as [`gemm`]),
/// or if `A` is so large (broadcast strides allow logical dimensions up to
/// `isize::MAX`) that the pack buffer size overflows `usize`.
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    // SAFETY: `a` is validated in-bounds directly above.
    unsafe { prepack_lhs_unchecked(a.data.as_ptr(), a.rs, a.cs, a.rows, a.cols) }
}

/// As [`prepack_lhs`] but over a raw `m × k` `A` pointer + strides, with **no** bounds check — the
/// raw counterpart for adapters / FFI that validate their own inputs.
///
/// # Safety
/// `a` must be valid for reads at every offset `i·rsa + j·csa`, for `i in 0..m` and `j in 0..k`.
pub unsafe fn prepack_lhs_unchecked<T: GemmScalar>(
    a: *const T,
    rsa: isize,
    csa: isize,
    m: usize,
    k: usize,
) -> PackedLhs<T> {
    // By the engine's A/B symmetry, a prepacked LHS *is* the prepacked RHS of the
    // transposed product `Cᵀ = Bᵀ·Aᵀ`: the `m × k` LHS is that problem's `k × m` RHS
    // (depth `k`, leading `m`), so the LHS row stride plays the RHS column stride and
    // the LHS column stride the RHS depth stride. Delegating to `prepack_rhs_unchecked`
    // keeps one geometry + pack path as the single source of truth (it lays down the
    // identical micropanel-major buffer, which the consuming call — driven transposed —
    // reads back verbatim); we only relabel the recorded dimensions into LHS terms.
    // (One benign consequence: the effectively-unreachable overflow panic reports the
    // problem as an RHS of `{k}x{m}` rather than an LHS of `{m}x{k}`.)
    //
    // SAFETY: `a` is caller-promised valid for the `(m, k)` reads at `i·rsa + j·csa`;
    // those are exactly the `(k, n = m)` reads `prepack_rhs_unchecked` performs with the
    // transposed strides `(rsb = csa, csb = rsa)`.
    let packed = unsafe { prepack_rhs_unchecked(a, csa, rsa, k, m) };
    PackedLhs {
        buf: packed.buf,
        m: packed.n,
        k: packed.k,
        nr: packed.nr,
        kc: packed.kc,
        nc: packed.nc,
    }
}

/// `C <- alpha·A·B + beta·C` reusing a [`PackedLhs`] (pre-packed `A`), via the
/// thread-local workspace pool. Skips the per-call LHS repack.
///
/// The result **reproduces** a plain [`gemm`] under the same config, except in two
/// cases that stay correct but may differ in the last ULP: very small products (both
/// `m` and `n` at or below [`crate::tuning::tiny_block_dim`], default 64) and gemv-shaped
/// (`m == 1` or `n == 1`) products. Output is deterministic across thread counts regardless.
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

    // C must not alias B (it is written). The prepacked A is a separate owned buffer
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, b.data.as_ptr(), b.data.len()) {
        panic!("gemmkit: C aliases B");
    }

    // SAFETY: B/C strides are in bounds and C does not alias B (checked above)
    unsafe {
        gemm_packed_a_unchecked_with(
            ws,
            alpha,
            packed,
            b.cols,
            b.data.as_ptr(),
            b.rs,
            b.cs,
            beta,
            c.data.as_mut_ptr(),
            c.rs,
            c.cs,
            par,
        );
    }
}

/// As [`gemm_packed_a`] but over raw `B`/`C` pointers + strides, with **no** bounds/alias checks.
/// The shared `k` and output-row count `m` come from `packed`; `n` is B's cols (= C's cols). Uses
/// the thread-local workspace pool.
///
/// # Safety
/// `b` valid for reads over `(packed.cols(), n)` and `c` for read+write over `(packed.rows(), n)`
/// at the given strides; `c` does not alias `b`; and when `beta == 0`, `c` need not be initialized.
///
/// # Panics
/// If `C` is not row-major-ish (`|csc| <= |rsc|`) — a prepacked LHS cannot serve a column-major C
/// (use plain [`gemm`] for that layout).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_a_unchecked<T: GemmScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    n: usize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety).
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_packed_a_unchecked_with(ws, alpha, packed, n, b, rsb, csb, beta, c, rsc, csc, par);
        });
    }
}

/// As [`gemm_packed_a_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_packed_a_unchecked`].
///
/// # Panics
/// See [`gemm_packed_a_unchecked`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_a_unchecked_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    n: usize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // A prepacked A is only valid for the orientation in which A keeps the RHS role of the
    // transposed product
    assert!(
        csc.unsigned_abs() <= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_a requires row-major-ish C (|csc| <= |rsc|); a column-major C \
         would keep A in the LHS role and invalidate the prepacked LHS — use gemm() for that layout"
    );

    // SAFETY: the caller guarantees B/C validity and that C does not alias B (see # Safety); the
    // packed buffer (owned by `packed`, read-only) outlives the call and matches its recorded
    // (nr, kc, nc) geometry.
    unsafe {
        dispatch::execute_packed(
            PackedConsume {
                m: n,
                k: packed.k,
                n: packed.m,
                alpha,
                a: b,
                rsa: csb,
                csa: rsb,
                packed: packed.buf.as_ptr(),
                nr: packed.nr,
                kc: packed.kc,
                nc: packed.nc,
                beta,
                c,
                rsc: csc,
                csc: rsc,
            },
            par,
            ws,
        );
    }
}
