//! Prepacked-operand (`PackedLhs`/`PackedRhs`) entries
use super::*;
#[cfg(feature = "epilogue")]
use crate::dispatch::FusedScalar;
use crate::dispatch::PackedConsume;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{BiasDim, BiasSpec, FusedEpi};
// The `vec!` macro is now used only by the i8 prepack path (`vec![0i8; ..]`, which already
// specializes to `alloc_zeroed`); the float/half path allocates uninit via `Vec::with_capacity`
#[cfg(feature = "int8")]
use alloc::vec;
use alloc::vec::Vec;

/// A right-hand-side matrix pre-packed once into gemmkit's internal
/// micropanel-major layout, for reuse across many products that share the same
/// `B` (the inference pattern: fixed weights, a stream of activation batches).
/// Produced by [`prepack_rhs`] and consumed by [`gemm_packed_b`] /
/// [`gemm_packed_b_with`], which skip the per-call RHS repack
///
/// The buffer records the blocking geometry it was built for; the consuming call
/// reads it back verbatim, so a panel is always read against its own tiling. It is
/// read-only during the GEMM, so it is shared across worker threads with no
/// synchronization
pub struct PackedRhs<T> {
    buf: Vec<T>,
    k: usize,
    n: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedRhs<T> {
    /// Rows of the original `B` (the shared `k` dimension)
    pub fn rows(&self) -> usize {
        self.k
    }
    /// Columns of the original `B` (the `n` dimension)
    pub fn cols(&self) -> usize {
        self.n
    }
}

/// Pre-pack a `k x n` RHS into [`PackedRhs`] for reuse across many [`gemm_packed_b`]
/// calls. The pack happens once, single-threaded, here; later products skip it
///
/// Any layout of `B` is accepted (the pack reads it through its strides). The
/// resulting buffer is valid for products whose `(k, n)` match this `B` and whose
/// `C` is column-major-ish (`|csc| >= |rsc|`); [`gemm_packed_b`] enforces both
///
/// # Panics
/// If `B`'s view addresses outside its slice (same bounds check as [`gemm`]),
/// or if `B` is so large (broadcast strides allow logical dimensions up to
/// `isize::MAX`) that the pack buffer size overflows `usize`
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    // SAFETY: `b` is validated in-bounds directly above
    unsafe { prepack_rhs_unchecked(b.data.as_ptr(), b.rs, b.cs, b.rows, b.cols) }
}

/// As [`prepack_rhs`] but over a raw `k x n` `B` pointer + strides, with **no** bounds check: the
/// raw counterpart for adapters / FFI that validate their own inputs
///
/// # Safety
/// `b` must be valid for reads at every offset `i*rsb + j*csb`, for `i in 0..k` and `j in 0..n`
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
    // with the packed input (`Lhs` == `Rhs`) element size: the unit the panels are
    // stored in, same as the driver
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    // An empty operand packs to nothing. Short-circuit before the geometry/size
    // arithmetic, which would otherwise overflow for a huge free dimension: an
    // empty view's extent is 0, so `check_view` accepts e.g. a `0 x usize::MAX` B
    // The consume path never reads the buffer for a `k == 0`/`n == 0` problem
    // (it only beta-scales C), so an empty pack round-trips. Mirrors the
    // zero-batch early return in `gemm_batched`
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
    // up to its `DEPTH_MULTIPLE`; `1` (every other kernel) leaves this unchanged
    let k_pad = k.next_multiple_of(<T as GemmScalar>::rhs_depth_multiple());
    // Checked: a broadcast (zero-stride) view passes `check_view` with a tiny
    // backing slice, so a logically huge `n`/`k` can reach this product; a wrapped
    // size would under-allocate the buffer the pack then writes past
    let total = n
        .div_ceil(nr)
        .checked_mul(nr)
        .and_then(|v| v.checked_mul(k_pad))
        .unwrap_or_else(|| {
            panic!("gemmkit: prepacked RHS of {k}x{n} is too large; the pack buffer size overflows usize")
        });
    // Allocate the pack buffer *without* zero-init: `pack_rhs_full` writes every one of the
    // `total` slots below before any of them is read, so the zero pass is dead. For `f32`/`f64`
    // `vec![ZERO; ..]` specializes to `alloc_zeroed` (already free), but the `half` types
    // (`f16`/`bf16`) lack std's `IsZero` specialization and would run a genuine dead `O(k*n)`
    // write; `with_capacity` + `set_len` avoids it uniformly. The write coverage is the same
    // tiling proof the pack oracle tests cover: `pack_rhs_full` lays `ceil(n/nr)` panels of
    // `nr x k_pad` end to end (its cursor advances `nr*k_pad_of_slice` per panel, summing to
    // exactly `total`), and each panel's `pack_(kgroup_)panels` writes every leading lane and
    // every padded-depth slot. So all `total` elements are initialized before `PackedRhs`
    // returns (hence before any read). `T: Scalar` is `Copy` with no drop glue, so even the
    // unreachable pack-panic path drops the (possibly uninit) `Vec` soundly
    let mut buf: Vec<T> = Vec::with_capacity(total);
    if total > 0 {
        // SAFETY: `buf`'s capacity is exactly `total`; `set_len(total)` exposes those slots,
        // every one written by `pack_rhs_full` below before it is read. `buf` holds
        // `ceil(n/nr)*nr*k_pad` elements (the exact layout size, with the depth padded to the
        // dispatched family's `DEPTH_MULTIPLE`); `b` is caller-promised valid for the `(k, n)`
        // strided reads; `pack_rhs_full` writes only that range and selects the right family
        unsafe {
            buf.set_len(total);
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

/// `C <- alpha*A*B + beta*C` reusing a [`PackedRhs`] (pre-packed `B`), via the
/// thread-local workspace pool. Skips the per-call RHS repack
///
/// The result **reproduces** a plain [`gemm`] under the same config, except in 2
/// cases that stay correct but may differ in the last ULP: very small products (both
/// `m` and `n` at or below [`crate::tuning::tiny_block_dim`], default 64) and gemv-shaped
/// (`m == 1` or `n == 1`) products. Output is deterministic across thread counts regardless
///
/// # Panics
/// If the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `A` or `C` addresses outside its slice, if `C` aliases
/// itself or `A`, or if `C` is **not** column-major-ish (`|csc| >= |rsc|`): a
/// row-major `C` would make the engine swap `A`/`B`, which a prepacked `B` cannot
/// support (use plain [`gemm`] there)
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

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_b`]
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
    // buffer, so it cannot alias C
    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len()) {
        panic!("gemmkit: C aliases A");
    }

    // SAFETY: A/C strides are in bounds and C does not alias A (checked above)
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
/// thread-local workspace pool
///
/// # Safety
/// `a` valid for reads over `(m, packed.rows())` and `c` for read+write over `(m, packed.cols())`
/// at the given strides; `c` does not alias `a`; and when `beta == 0`, `c` need not be initialized
///
/// # Panics
/// If `C` is not column-major-ish (`|csc| >= |rsc|`): a prepacked RHS cannot serve a row-major C
/// (use plain [`gemm`] for that layout)
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
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_packed_b_unchecked_with(ws, alpha, m, a, rsa, csa, packed, beta, c, rsc, csc, par);
        });
    }
}

/// As [`gemm_packed_b_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_packed_b_unchecked`]
///
/// # Panics
/// See [`gemm_packed_b_unchecked`]
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
    // (nr, kc, nc) geometry
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

/// Pre-pack a `k x n` **`i8`** RHS into a [`PackedRhs<i8>`] for reuse across many
/// [`gemm_i8_packed_b`] calls: the fixed-weight quantized-inference pattern (constant `i8`
/// weights, a stream of `i8` activation batches). The pack happens once, single-threaded, here;
/// later products skip it. For the VNNI `vpdpbusd` kernel this matters most: its RHS pack is
/// **mandatory every call** (the k-quad-interleaved layout can't be read in place), so at small
/// `m` the per-call `O(k*n)` pack otherwise dominates the `O(m*k*n)` compute
///
/// The buffer is packed through whichever integer kernel the process's memoized dispatch selected
/// (the VNNI k-quad-interleaved layout, or the widen kernel's plain panels) and records the
/// blocking geometry it was built for; [`gemm_i8_packed_b`] reads it back verbatim and always runs
/// that same family, so the buffer is never misread. Any layout of `B` is accepted (the pack reads
/// it through its strides); the result is valid for products whose `(k, n)` match this `B` and
/// whose `C` is column-major-ish (`|csc| >= |rsc|`)
///
/// # Panics
/// If `B`'s view addresses outside its slice (same bounds check as [`gemm_i8`]), or if `B` is so
/// large (broadcast strides allow logical dimensions up to `isize::MAX`) that the pack buffer size
/// overflows `usize`
#[cfg(feature = "int8")]
pub fn prepack_rhs_i8(b: MatRef<'_, i8>) -> PackedRhs<i8> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    // SAFETY: `b` is validated in-bounds directly above
    unsafe { prepack_rhs_i8_unchecked(b.data.as_ptr(), b.rs, b.cs, b.rows, b.cols) }
}

/// As [`prepack_rhs_i8`] but over a raw `k x n` `B` pointer + strides, with **no** bounds check:
/// the raw counterpart for adapters / FFI that validate their own inputs
///
/// # Safety
/// `b` must be valid for reads at every offset `i*rsb + j*csb`, for `i in 0..k` and `j in 0..n`
#[cfg(feature = "int8")]
pub unsafe fn prepack_rhs_i8_unchecked(
    b: *const i8,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
) -> PackedRhs<i8> {
    // Resolve the panel geometry through the same ISA tile the consuming call will use; the
    // `tiny_block_dim() + 1` sentinel row count dodges the tiny-matrix branch so the geometry is
    // `m`-independent (the consume reads it back verbatim). i8 packs in 1-byte units
    let (mr, nr) = dispatch::i8_rhs_tile();
    // An empty operand packs to nothing (see the `prepack_rhs` rationale)
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
    let dodge_tiny = crate::tuning::tiny_block_dim().saturating_add(1);
    let blk = crate::cache::topology().blocking(mr, nr, 1, dodge_tiny, n, k);
    let depth_multiple = dispatch::i8_rhs_depth_multiple();
    // The VNNI dot kernel (`DEPTH_MULTIPLE = 4`) packs the whole contraction as one depth slice
    // (the driver's prepacked single-slice guard for a depth-padded family); the widen kernel keeps
    // the cache-model `kc`. Integer accumulation is exact (wrapping i32 is associative), so single
    // vs multi-slice `kc` is bit-identical either way. (i8's `Out == Acc == i32` makes it
    // `OUT_IS_ACC`, so the depth multiple, not `OUT_IS_ACC`, is what forces the single slice here,
    // unlike the bf16 dot path)
    let kc = if depth_multiple > 1 {
        k.max(1)
    } else {
        blk.kc.max(1)
    };
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    // A dot kernel (VNNI `vpdpbusd`) packs depth in groups, so the panel depth is rounded up to its
    // `DEPTH_MULTIPLE`; `1` (the widen kernel) leaves this unchanged
    let k_pad = k.next_multiple_of(depth_multiple);
    // Checked: a broadcast (zero-stride) view passes `check_view` with a tiny backing slice, so a
    // logically huge `n`/`k` can reach this product; a wrapped size would under-allocate the buffer
    let total = n
        .div_ceil(nr)
        .checked_mul(nr)
        .and_then(|v| v.checked_mul(k_pad))
        .unwrap_or_else(|| {
            panic!("gemmkit: prepacked RHS of {k}x{n} is too large; the pack buffer size overflows usize")
        });
    let mut buf = vec![0i8; total];
    if total > 0 {
        // SAFETY: `buf` holds `ceil(n/nr)*nr*k_pad` elements (the exact layout size, depth padded to
        // the selected family's `DEPTH_MULTIPLE`); `b` is caller-promised valid for the `(k, n)`
        // strided reads; `pack_rhs_full_i8` writes only that range through the selected family's pack
        unsafe {
            dispatch::pack_rhs_full_i8(buf.as_mut_ptr(), b, rsb, csb, k, n, kc, nc, nr);
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

/// `C(i32) <- alpha*A(i8)*(prepacked B) + beta*C` reusing a [`PackedRhs<i8>`] (pre-packed `i8` `B`),
/// via the thread-local workspace pool. The integer (`i8 -> i32`) twin of [`gemm_packed_b`]: it
/// skips the per-call RHS repack, which for the VNNI kernel is otherwise mandatory on every call
///
/// The result is **bit-identical** to a plain [`gemm_i8`] under the same config for every valid
/// shape/stride (integer accumulation is exact and ISA-independent, so the prepacked and plain
/// paths agree exactly, and the output is deterministic across thread counts)
///
/// # Panics
/// If the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`, `B.cols != C.cols`), if `A`
/// or `C` addresses outside its slice, if `C` aliases itself or `A`, or if `C` is **not**
/// column-major-ish (`|csc| >= |rsc|`): a row-major `C` would make the engine swap `A`/`B`, which a
/// prepacked `B` cannot support (use plain [`gemm_i8`] there)
#[cfg(feature = "int8")]
pub fn gemm_i8_packed_b(
    alpha: i32,
    a: MatRef<'_, i8>,
    packed: &PackedRhs<i8>,
    beta: i32,
    c: MatMut<'_, i32>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_i8_packed_b_with(ws, alpha, a, packed, beta, c, par));
}

/// Like [`gemm_i8_packed_b`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_i8_packed_b`]
#[cfg(feature = "int8")]
pub fn gemm_i8_packed_b_with(
    ws: &mut Workspace,
    alpha: i32,
    a: MatRef<'_, i8>,
    packed: &PackedRhs<i8>,
    beta: i32,
    c: MatMut<'_, i32>,
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

    // C (i32) must not alias A (i8); byte ranges (heterogeneous element sizes). The prepacked B is
    // a separate owned buffer, so it cannot alias C
    if overlaps_bytes(
        c.data.as_ptr() as *const u8,
        c.data.len(),
        core::mem::size_of::<i32>(),
        a.data.as_ptr() as *const u8,
        a.data.len(),
        core::mem::size_of::<i8>(),
    ) {
        panic!("gemmkit: C aliases A");
    }

    // SAFETY: A/C strides are in bounds and C does not alias A (checked above)
    unsafe {
        gemm_i8_packed_b_unchecked_with(
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

/// As [`gemm_i8_packed_b`] but over raw `A`/`C` pointers + strides, with **no** bounds/alias
/// checks: the heterogeneous (`i8 -> i32`) counterpart of [`gemm_packed_b_unchecked`]. The shared
/// `k` and output `n` come from `packed`; `m` is A's rows (= C's rows). Uses the thread-local
/// workspace pool
///
/// # Safety
/// `a` valid for reads over `(m, packed.rows())` and `c` for read+write over `(m, packed.cols())`
/// at the given strides; `c` does not alias `a`; and when `beta == 0`, `c` need not be initialized
///
/// # Panics
/// If `C` is not column-major-ish (`|csc| >= |rsc|`): a prepacked RHS cannot serve a row-major C
/// (use plain [`gemm_i8`] for that layout)
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_packed_b_unchecked(
    alpha: i32,
    m: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    packed: &PackedRhs<i8>,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_i8_packed_b_unchecked_with(
                ws, alpha, m, a, rsa, csa, packed, beta, c, rsc, csc, par,
            );
        });
    }
}

/// As [`gemm_i8_packed_b_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_i8_packed_b_unchecked`]
///
/// # Panics
/// See [`gemm_i8_packed_b_unchecked`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_i8_packed_b_unchecked_with(
    ws: &mut Workspace,
    alpha: i32,
    m: usize,
    a: *const i8,
    rsa: isize,
    csa: isize,
    packed: &PackedRhs<i8>,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    // A prepacked B is only valid for the no-swap orientation (same assert + message as the float
    // `gemm_packed_b_unchecked_with`)
    assert!(
        csc.unsigned_abs() >= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );
    // SAFETY: the caller guarantees A/C validity and that C does not alias A (see # Safety); the
    // packed buffer (owned by `packed`, read-only) outlives the call and matches its recorded
    // (nr, kc, nc) geometry
    unsafe {
        dispatch::execute_int_packed(
            dispatch::IntPackedConsume {
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

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in 1 pass: a **fused** epilogue over a reused
/// [`PackedRhs`], via the thread-local workspace pool. The fused twin of [`gemm_packed_b`]: the
/// bias is added by 1 IEEE add after the final `beta`-fold, then the activation is applied, fused
/// into the store the packed kernel already runs. `bias == None && act == None` reproduces
/// [`gemm_packed_b`] bit-for-bit
///
/// The **same** [`PackedRhs`] handle serves both [`gemm_packed_b`] and this fused entry: the
/// epilogue is store-side only, so the pack (and its recorded geometry) is untouched. For
/// `f32`/`f64` the result is **bit-identical** to [`gemm_packed_b`] followed by the same scalar map,
/// for every valid shape/stride; for `f16`/`bf16` the epilogue applies in `f32` before the single
/// narrowing (more precise than, so not bitwise-equal to, packed-gemm-then-map). Unlike plain
/// [`gemm_fused`], the packed path is **not** rerouted to gemv / small-`m,n` / small-`k`: it always
/// drives the general prepacked kernel (the special-path divergence the plain packed entries
/// document), and it never orientation-swaps, so the user-frame per-row / per-col bias passes
/// straight through
///
/// # Panics
/// Same conditions as [`gemm_packed_b`], plus the fused conditions of [`gemm_fused`]: a `PerRow`
/// bias whose length is not `A.rows` (or a `PerCol` bias not `B.cols`); a bias slice overlapping
/// `C`; or a non-finite `LeakyRelu` slope
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused<T: FusedScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| {
        gemm_packed_b_fused_with(ws, alpha, a, packed, beta, c, bias, act, par)
    });
}

/// Like [`gemm_packed_b_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_b_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    // The exact validation `gemm_packed_b_with` runs (byte-identical messages)
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

    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, a.data.as_ptr(), a.data.len()) {
        panic!("gemmkit: C aliases A");
    }

    // Fused-epilogue validation (shared wording with `gemm_fused`): the bias length matches its
    // axis (PerRow == A.rows, PerCol == packed B.cols == C.cols) and does not overlap C; a
    // LeakyRelu slope is finite. The packed path never swaps, so the user-frame bias is passed
    // straight through (no axis flip)
    validate_bias(&bias, a.rows, packed.n, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validated above; A/C strides are in bounds, C addresses each (i,j) uniquely and does
    // not alias A, the prepacked B is a separate owned buffer, and the bias (borrowed for this
    // call) is the right length for its axis and disjoint from C. The bias pointer stays valid for
    // the whole `execute_packed_fused` frame
    unsafe {
        packed_b_fused_impl(
            Some(ws),
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
            epi,
            par,
        );
    }
}

/// As [`gemm_packed_b_fused`] but over raw `A`/`C` pointers + strides, with **no** bounds/alias
/// checks. `bias` is a `(ptr, dim)` pair enabled by `has_bias` (ignored when `has_bias == false`),
/// in the user frame (the packed path never swaps, so no axis flip); `act` is applied last. Uses
/// the thread-local workspace pool
///
/// # Safety
/// As [`gemm_packed_b_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m`
/// (`PerRow`) or `packed.cols()` (`PerCol`) elements and does not alias `c`; and a non-finite
/// `LeakyRelu` slope is the caller's responsibility (the checked API rejects it)
///
/// # Panics
/// As [`gemm_packed_b_unchecked`] (a non-column-major-ish C)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_b_fused_unchecked<T: FusedScalar>(
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
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        packed_b_fused_impl(
            None, alpha, m, a, rsa, csa, packed, beta, c, rsc, csc, epi, par,
        );
    }
}

/// As [`gemm_packed_b_fused_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_packed_b_fused_unchecked`]
///
/// # Panics
/// See [`gemm_packed_b_fused_unchecked`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_b_fused_unchecked_with<T: FusedScalar>(
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
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        packed_b_fused_impl(
            Some(ws),
            alpha,
            m,
            a,
            rsa,
            csa,
            packed,
            beta,
            c,
            rsc,
            csc,
            epi,
            par,
        );
    }
}

/// Shared lowering for the 4 fused B-packed entries: assert the packed-B orientation (the
/// byte-identical panic `gemm_packed_b_unchecked_with` raises), build the [`PackedConsume`], and
/// dispatch the prepacked-fused engine over either a caller-owned [`Workspace`] (`ws = Some`) or the
/// thread-local pool (`ws = None`). The bias/activation are already lowered into `epi`. The packed-B
/// consume frame is the user frame (no swap), so `epi` is passed through unflipped
///
/// # Safety
/// As [`gemm_packed_b_fused_unchecked`]: `a` valid for reads and `c` for read+write over the shape /
/// strides, `c` not aliasing `a`, and `epi`'s bias (if any) valid for `m`/`packed.cols()` reads and
/// disjoint from `c`
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
unsafe fn packed_b_fused_impl<T: FusedScalar>(
    ws: Option<&mut Workspace>,
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
    epi: FusedEpi<T>,
    par: Parallelism,
) {
    // A prepacked B is only valid for the no-swap orientation (same assert + message as
    // `gemm_packed_b_unchecked_with`)
    assert!(
        csc.unsigned_abs() >= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );
    let req = PackedConsume {
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
    };
    // SAFETY: caller guarantees A/C validity and that C does not alias A; the packed buffer (owned
    // by `packed`, read-only) outlives the call and matches its recorded geometry; `epi`'s bias is
    // valid and disjoint from C (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_packed_fused(req, epi, par, ws),
            None => {
                workspace::with_thread_pool(|ws| dispatch::execute_packed_fused(req, epi, par, ws))
            }
        }
    }
}

/// A left-hand-side matrix pre-packed once into gemmkit's internal
/// micropanel-major layout, for reuse across many products that share the same
/// `A` (a fixed weight matrix `A` against a stream of differently-shaped right
/// operands `B`). Produced by [`prepack_lhs`] and consumed by [`gemm_packed_a`] /
/// [`gemm_packed_a_with`], which skip the per-call LHS repack
///
/// By the engine's A/B symmetry, a prepacked LHS is the prepacked RHS of the
/// transposed product `C^T = B^T*A^T`; the buffer records that transposed problem's
/// blocking geometry, which the consuming call (driven transposed) reads back
/// verbatim. Read-only during the GEMM, so it is shared across worker threads with
/// no synchronization
pub struct PackedLhs<T> {
    buf: Vec<T>,
    m: usize,
    k: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedLhs<T> {
    /// Rows of the original `A` (the `m` dimension)
    pub fn rows(&self) -> usize {
        self.m
    }
    /// Columns of the original `A` (the shared `k` dimension)
    pub fn cols(&self) -> usize {
        self.k
    }
}

/// Pre-pack an `m x k` LHS into [`PackedLhs`] for reuse across many [`gemm_packed_a`]
/// calls. The pack happens once, single-threaded, here; later products skip it
///
/// Any layout of `A` is accepted (the pack reads it through its strides). The
/// resulting buffer is valid for products whose `(m, k)` match this `A` and whose
/// `C` is row-major-ish (`|csc| <= |rsc|`); [`gemm_packed_a`] enforces both
///
/// # Panics
/// If `A`'s view addresses outside its slice (same bounds check as [`gemm`]),
/// or if `A` is so large (broadcast strides allow logical dimensions up to
/// `isize::MAX`) that the pack buffer size overflows `usize`
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    // SAFETY: `a` is validated in-bounds directly above
    unsafe { prepack_lhs_unchecked(a.data.as_ptr(), a.rs, a.cs, a.rows, a.cols) }
}

/// As [`prepack_lhs`] but over a raw `m x k` `A` pointer + strides, with **no** bounds check: the
/// raw counterpart for adapters / FFI that validate their own inputs
///
/// # Safety
/// `a` must be valid for reads at every offset `i*rsa + j*csa`, for `i in 0..m` and `j in 0..k`
pub unsafe fn prepack_lhs_unchecked<T: GemmScalar>(
    a: *const T,
    rsa: isize,
    csa: isize,
    m: usize,
    k: usize,
) -> PackedLhs<T> {
    // By the engine's A/B symmetry, a prepacked LHS *is* the prepacked RHS of the
    // transposed product `C^T = B^T*A^T`: the `m x k` LHS is that problem's `k x m` RHS
    // (depth `k`, leading `m`), so the LHS row stride plays the RHS column stride and
    // the LHS column stride the RHS depth stride. Delegating to `prepack_rhs_unchecked`
    // keeps one geometry + pack path as the single source of truth (it lays down the
    // identical micropanel-major buffer, which the consuming call, driven transposed,
    // reads back verbatim); only the recorded dimensions are relabeled into LHS terms
    // (One benign consequence: the effectively-unreachable overflow panic reports the
    // problem as an RHS of `{k}x{m}` rather than an LHS of `{m}x{k}`.)
    //
    // SAFETY: `a` is caller-promised valid for the `(m, k)` reads at `i*rsa + j*csa`;
    // those are exactly the `(k, n = m)` reads `prepack_rhs_unchecked` performs with the
    // transposed strides `(rsb = csa, csb = rsa)`
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

/// `C <- alpha*A*B + beta*C` reusing a [`PackedLhs`] (pre-packed `A`), via the
/// thread-local workspace pool. Skips the per-call LHS repack
///
/// The result **reproduces** a plain [`gemm`] under the same config, except in 2
/// cases that stay correct but may differ in the last ULP: very small products (both
/// `m` and `n` at or below [`crate::tuning::tiny_block_dim`], default 64) and gemv-shaped
/// (`m == 1` or `n == 1`) products. Output is deterministic across thread counts regardless
///
/// # Panics
/// If the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `B` or `C` addresses outside its slice, if `C` aliases
/// itself or `B`, or if `C` is **not** row-major-ish (`|csc| <= |rsc|`): a
/// column-major `C` would leave `A` in the genuine LHS role, which a prepacked `A`
/// (laid out as the transposed RHS) cannot serve (use plain [`gemm`] there)
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

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_a`]
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
/// the thread-local workspace pool
///
/// # Safety
/// `b` valid for reads over `(packed.cols(), n)` and `c` for read+write over `(packed.rows(), n)`
/// at the given strides; `c` does not alias `b`; and when `beta == 0`, `c` need not be initialized
///
/// # Panics
/// If `C` is not row-major-ish (`|csc| <= |rsc|`): a prepacked LHS cannot serve a column-major C
/// (use plain [`gemm`] for that layout)
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
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        workspace::with_thread_pool(|ws| {
            gemm_packed_a_unchecked_with(ws, alpha, packed, n, b, rsb, csb, beta, c, rsc, csc, par);
        });
    }
}

/// As [`gemm_packed_a_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_packed_a_unchecked`]
///
/// # Panics
/// See [`gemm_packed_a_unchecked`]
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
    // (nr, kc, nc) geometry
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

/// `C <- act(alpha*(prepacked A)*B + beta*C + bias)` in 1 pass: a **fused** epilogue over a reused
/// [`PackedLhs`], via the thread-local workspace pool. The fused twin of [`gemm_packed_a`]: the bias
/// is added by 1 IEEE add after the final `beta`-fold, then the activation is applied, fused into
/// the store the packed kernel already runs. `bias == None && act == None` reproduces
/// [`gemm_packed_a`] bit-for-bit
///
/// The **same** [`PackedLhs`] handle serves both [`gemm_packed_a`] and this fused entry: the
/// epilogue is store-side only, so the pack is untouched. For `f32`/`f64` the result is
/// **bit-identical** to [`gemm_packed_a`] followed by the same scalar map, for every valid
/// shape/stride; for `f16`/`bf16` the epilogue applies in `f32` before the single narrowing (more
/// precise than, so not bitwise-equal to, packed-gemm-then-map). Like [`gemm_packed_b_fused`] the
/// packed path is **not** rerouted to a special path (always the general prepacked kernel) and the
/// per-row / per-col bias is specified in the **user** frame (a `PerRow` bias of length `A.rows`,
/// added to every column of that row): the engine's internal transpose of the packed-A product is
/// handled inside, so the user never sees an axis flip
///
/// # Panics
/// Same conditions as [`gemm_packed_a`], plus the fused conditions of [`gemm_fused`]: a `PerRow`
/// bias whose length is not `A.rows` (or a `PerCol` bias not `B.cols`); a bias slice overlapping
/// `C`; or a non-finite `LeakyRelu` slope
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused<T: FusedScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| {
        gemm_packed_a_fused_with(ws, alpha, packed, b, beta, c, bias, act, par)
    });
}

/// Like [`gemm_packed_a_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_a_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    // The exact validation `gemm_packed_a_with` runs (byte-identical messages)
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

    let cp = c.data.as_ptr();
    let cl = c.data.len();
    if overlaps(cp, cl, b.data.as_ptr(), b.data.len()) {
        panic!("gemmkit: C aliases B");
    }

    // Fused-epilogue validation (shared wording with `gemm_fused`): the bias length matches its
    // USER axis (PerRow == packed A.rows == C.rows, PerCol == B.cols) and does not overlap C; a
    // LeakyRelu slope is finite. The bias stays in the user frame here; `packed_a_fused_impl` flips
    // the axis to match the transposed consume the packed-A path drives
    validate_bias(&bias, packed.m, b.cols, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validated above; B/C strides are in bounds, C addresses each (i,j) uniquely and does
    // not alias B, the prepacked A is a separate owned buffer, and the bias (borrowed for this
    // call) is the right length for its axis and disjoint from C. The bias pointer stays valid for
    // the whole `execute_packed_fused` frame
    unsafe {
        packed_a_fused_impl(
            Some(ws),
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
            epi,
            par,
        );
    }
}

/// As [`gemm_packed_a_fused`] but over raw `B`/`C` pointers + strides, with **no** bounds/alias
/// checks. `bias` is a `(ptr, dim)` pair enabled by `has_bias` (ignored when `has_bias == false`),
/// in the **user** frame (a `PerRow` bias indexes `A.rows` = `C.rows`); `act` is applied last. Uses
/// the thread-local workspace pool
///
/// # Safety
/// As [`gemm_packed_a_unchecked`], plus: when `has_bias`, `bias` is valid for reads of
/// `packed.rows()` (`PerRow`) or `n` (`PerCol`) elements and does not alias `c`; and a non-finite
/// `LeakyRelu` slope is the caller's responsibility (the checked API rejects it)
///
/// # Panics
/// As [`gemm_packed_a_unchecked`] (a non-row-major-ish C)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_a_fused_unchecked<T: FusedScalar>(
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
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        packed_a_fused_impl(
            None, alpha, packed, n, b, rsb, csb, beta, c, rsc, csc, epi, par,
        );
    }
}

/// As [`gemm_packed_a_fused_unchecked`] but with a caller-owned [`Workspace`]
///
/// # Safety
/// See [`gemm_packed_a_fused_unchecked`]
///
/// # Panics
/// See [`gemm_packed_a_fused_unchecked`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_packed_a_fused_unchecked_with<T: FusedScalar>(
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
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let epi = to_fused_epi_raw(bias, bias_dim, has_bias, act);
    // SAFETY: preconditions forwarded to the caller (see # Safety)
    unsafe {
        packed_a_fused_impl(
            Some(ws),
            alpha,
            packed,
            n,
            b,
            rsb,
            csb,
            beta,
            c,
            rsc,
            csc,
            epi,
            par,
        );
    }
}

/// Shared lowering for the 4 fused A-packed entries: assert the packed-A orientation (the
/// byte-identical panic `gemm_packed_a_unchecked_with` raises), **flip the user-frame bias axis**
/// to the transposed consume frame the packed-A path drives, build the transposed [`PackedConsume`]
/// (exactly as `gemm_packed_a_unchecked_with` does), and dispatch the prepacked-fused engine over
/// either a caller-owned [`Workspace`] (`ws = Some`) or the thread-local pool (`ws = None`)
///
/// By the engine's A/B symmetry the packed-A product is driven as the transposed problem
/// `C^T = B^T*A^T` (m<->n swapped in the consume frame), so a user per-row bias (indexed by the user
/// output row) becomes per-col in the consume frame and vice versa: the same field-write flip
/// `run_typed_fused` applies on a dynamic orientation swap, here baked into the always-transposed
/// packed-A path so the user bias axis is honoured
///
/// # Safety
/// As [`gemm_packed_a_fused_unchecked`]: `b` valid for reads and `c` for read+write over the shape /
/// strides, `c` not aliasing `b`, and `epi`'s bias (if any) valid for `packed.rows()`/`n` reads and
/// disjoint from `c`
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
unsafe fn packed_a_fused_impl<T: FusedScalar>(
    ws: Option<&mut Workspace>,
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
    mut epi: FusedEpi<T>,
    par: Parallelism,
) {
    // A prepacked A is only valid for the row-major-ish orientation (same assert + message as
    // `gemm_packed_a_unchecked_with`)
    assert!(
        csc.unsigned_abs() <= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_a requires row-major-ish C (|csc| <= |rsc|); a column-major C \
         would keep A in the LHS role and invalidate the prepacked LHS — use gemm() for that layout"
    );
    // The packed-A path drives the transposed product, so the user bias axis flips to the consume
    // (oriented) frame: user per-row -> per-col and vice versa. `execute_packed_fused` (compute and
    // degenerate) then applies `epi` in that consume frame, which maps back to the user axis
    epi.bias = match epi.bias {
        BiasSpec::None => BiasSpec::None,
        BiasSpec::Row(p) => BiasSpec::Col(p),
        BiasSpec::Col(p) => BiasSpec::Row(p),
    };
    // The transposed consume `gemm_packed_a_unchecked_with` builds (m<->n, A=B, strides swapped)
    let req = PackedConsume {
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
    };
    // SAFETY: caller guarantees B/C validity and that C does not alias B; the packed buffer (owned
    // by `packed`, read-only) outlives the call and matches its recorded geometry; `epi`'s bias is
    // valid and disjoint from C (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_packed_fused(req, epi, par, ws),
            None => {
                workspace::with_thread_pool(|ws| dispatch::execute_packed_fused(req, epi, par, ws))
            }
        }
    }
}
