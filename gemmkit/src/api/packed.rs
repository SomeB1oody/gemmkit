//! Prepacked-operand entries: pack `A` or `B` once into gemmkit's internal micropanel
//! layout, then reuse that buffer across many GEMM calls. The packed operand stays fixed
//! while the other operand varies
//!
//! [`PackedRhs`] and [`prepack_rhs`] fix `B` (for example, a weight matrix) and stream
//! `A` (for example, activations). [`gemm_packed_b`] consumes the result and requires a
//! column-major-ish `C`. [`PackedLhs`] and [`prepack_lhs`] fix `A` and stream `B` instead.
//! [`gemm_packed_a`] consumes the result and requires a row-major-ish `C`, because a
//! prepacked LHS is packed as the RHS of the transposed product. Each pair comes in a
//! checked form, an `_unchecked` form, a workspace-owning `_with` form, and, under the
//! `epilogue` feature, a bias and activation `_fused` form. Under the `int8` feature, the
//! `B`-packed family also gains an `i8 -> i32` twin
use super::*;
#[cfg(feature = "epilogue")]
use crate::dispatch::FusedScalar;
use crate::dispatch::PackedConsume;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{BiasDim, BiasSpec, FusedEpi};
// `vec!` specializes `vec![0i8; total]` to `alloc_zeroed`, so only the i8 prepack path
// uses it. The float and half path allocates uninitialized memory with `Vec::with_capacity`
#[cfg(feature = "int8")]
use alloc::vec;
use alloc::vec::Vec;

/// A `B` matrix packed once into gemmkit's internal micropanel-major layout, for reuse
/// across many products that share the same `B`. This fits a fixed weight matrix streamed
/// against many activation matrices. [`prepack_rhs`] builds it. [`gemm_packed_b`] and its
/// `_with`, `_unchecked`, and `_fused` siblings consume it and skip the per-call RHS pack
///
/// The buffer stores the blocking geometry (`nr`, `kc`, `nc`) it was packed for. Every
/// consuming call reads panels back with that exact geometry instead of deriving a new one,
/// so reuse always matches the original tiling. The buffer is read-only for the whole GEMM,
/// so it needs no synchronization across worker threads
pub struct PackedRhs<T> {
    buf: Vec<T>,
    k: usize,
    n: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedRhs<T> {
    /// Row count of the original `B` (the shared contraction dimension `k`)
    pub fn rows(&self) -> usize {
        self.k
    }
    /// Column count of the original `B` (the `n` dimension)
    pub fn cols(&self) -> usize {
        self.n
    }
}

/// Pack a `k x n` `B` view into a [`PackedRhs`] for reuse across many [`gemm_packed_b`]
/// calls. The pack runs once, single-threaded, right here, so every later call skips it
///
/// The pack accepts any layout of `B` and reads it through its strides. The resulting
/// buffer is only valid for a product whose `(k, n)` match this `B` and whose `C` is
/// column-major-ish (`|csc| >= |rsc|`). [`gemm_packed_b`] checks both before consuming it
///
/// # Panics
///
/// Panics if `B`'s view addresses outside its slice, the same bounds check [`gemm`] runs.
/// Also panics if `B` is so large that the pack buffer size overflows `usize`. A broadcast
/// stride can make `B`'s logical dimensions run up to `isize::MAX` while addressing only a
/// few elements
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    // SAFETY: `b` is validated in-bounds directly above
    unsafe { prepack_rhs_unchecked(b.data.as_ptr(), b.rs, b.cs, b.rows, b.cols) }
}

/// As [`prepack_rhs`] but over a raw `k x n` `B` pointer and strides, with no bounds check.
/// Use this raw form for an adapter or FFI caller that validates its own inputs
///
/// # Safety
///
/// `b` must be valid for reads at every offset `i*rsb + j*csb`, for `i in 0..k` and
/// `j in 0..n`
pub unsafe fn prepack_rhs_unchecked<T: GemmScalar>(
    b: *const T,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
) -> PackedRhs<T> {
    // Resolve panel geometry from the ISA microtile this element type dispatches to
    // `blocking()` picks a small-matrix shortcut based on row count, which would change
    // `kc`/`nc`. A `tiny_block_dim() + 1` sentinel skips that branch, so the packed geometry
    // works for every `m` a later call brings
    let (mr, nr) = <T as GemmScalar>::rhs_tile();
    // Guard the degenerate case before the geometry math below, which assumes `k` and `n`
    // are nonzero. A broadcast view can make them logically huge while backed by only a few
    // elements. The consuming dispatch never reads the packed buffer when `k == 0` or
    // `n == 0`, so this empty pack round-trips safely, the same way `gemm_batched` handles
    // a `batch == 0` call
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

    // A dot-product kernel, such as bf16's `vdpbf16ps`, packs depth in fused pairs, so the
    // panel depth rounds up to its `DEPTH_MULTIPLE`. Every other kernel's multiple is 1, so
    // this has no effect there
    let k_pad = k.next_multiple_of(<T as GemmScalar>::rhs_depth_multiple());
    // A broadcast (zero-stride) view can pass `check_view` while backed by a tiny slice, so
    // `n` and `k` may be logically huge here. Checked multiplication avoids under-sizing `buf`
    let total = n
        .div_ceil(nr)
        .checked_mul(nr)
        .and_then(|v| v.checked_mul(k_pad))
        .unwrap_or_else(|| {
            panic!("gemmkit: prepacked RHS of {k}x{n} is too large; the pack buffer size overflows usize")
        });
    // `buf` skips zero-initializing: the pack loop below fills every slot before any read,
    // so a zero pass is wasted work. `Vec::with_capacity` plus `set_len` avoids that write
    // uniformly, including for `half` types that do not get the `alloc_zeroed`
    // specialization a plain `vec![T::ZERO; total]` gets for `f32` and `f64`
    let mut buf: Vec<T> = Vec::with_capacity(total);
    if total > 0 {
        // SAFETY: `buf` has capacity `total`, so `set_len(total)` only exposes allocated
        // memory. `pack_rhs_full` writes every one of those slots before any read, and
        // `T: Copy` rules out a `Drop` impl, so even an unreachable panic mid-pack drops a
        // partly uninitialized `Vec` soundly. `b` is valid for the `(k, n)` strided reads
        // per the caller's promise (see # Safety), and `pack_rhs_full` dispatches through
        // `T`'s own kernel family, so it writes the layout that family expects
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

/// `C <- alpha*A*B + beta*C`, consuming a [`PackedRhs`] (`B` prepacked once) instead of `B`
/// itself, using the thread-local workspace pool. This skips the per-call RHS pack that
/// [`gemm`] would run
///
/// The result reproduces plain [`gemm`] under the same config, except that 2 shapes may
/// differ in the last ULP while staying correct. This can happen for a small product,
/// where both `m` and `n` are at or below [`crate::tuning::small_mn_dim`], or for a
/// gemv-shaped product, where `m == 1` or `n == 1`. Both cases arise because this path
/// always drives the general driver. Plain `gemm` instead reroutes them to a dedicated
/// kernel with a different accumulation order
///
/// # Panics
///
/// Panics if the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `A` or `C` addresses outside its slice, or if `C` aliases itself
/// or `A`. Also panics if `C` is not column-major-ish (`|csc| >= |rsc|`), because a
/// row-major `C` would make the engine swap `A` and `B`, which a prepacked `B` cannot
/// support. Use plain [`gemm`] for that layout
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
///
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

    // C must not alias A, because C is written. The prepacked B is a separate owned buffer,
    // so it can never alias C
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

/// As [`gemm_packed_b`] but over raw `A`/`C` pointers and strides, with no bounds or alias
/// checks. The shared `k` and output `n` come from `packed`, and `m` is `A`'s row count,
/// the same as `C`'s. Uses the thread-local workspace pool
///
/// # Safety
///
/// `a` must be valid for reads over `(m, packed.rows())`. `c` must be valid for read and
/// write over `(m, packed.cols())` at the given strides, and must not alias `a`. When
/// `beta == 0`, `c` need not be initialized, because the store overwrites it instead of
/// reading it
///
/// # Panics
///
/// Panics if `C` is not column-major-ish (`|csc| >= |rsc|`), because a prepacked RHS
/// cannot serve a row-major `C`. Use plain [`gemm`] for that layout
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
///
/// See [`gemm_packed_b_unchecked`]
///
/// # Panics
///
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
    // A prepacked B was packed as the genuine RHS, so it only serves the no-swap
    // (column-major-ish C) orientation
    assert!(
        csc.unsigned_abs() >= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );
    // SAFETY: the caller guarantees A/C validity and that C does not alias A (see
    // # Safety). `packed` outlives this call, is read-only, and its `(nr, kc, nc)` fields
    // are exactly the geometry its buffer was packed with
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

/// Pack a `k x n` `i8` RHS into a [`PackedRhs<i8>`] for reuse across many
/// [`gemm_i8_packed_b`] calls. This fits the quantized-inference pattern of constant `i8`
/// weights against a stream of `i8` activation batches. The pack runs once, single-threaded,
/// right here, so later calls skip it
///
/// This matters most for the VNNI `vpdpbusd` kernel. Its RHS pack is otherwise mandatory on
/// every call, because its layout cannot be read from `B` in place. The pack cost depends
/// on `k` and `n` alone, so a small `m` does not amortize it. Prepacking removes it from
/// the hot path
///
/// The buffer is packed through whichever integer kernel the process's dispatch selected,
/// either VNNI's interleaved layout or a widen kernel's plain panels. It records the
/// blocking geometry it was packed with. [`gemm_i8_packed_b`] reads that geometry back and
/// always runs the same family, so the buffer is never misread. The pack accepts any layout
/// of `B` and reads it through its strides. The result is valid for a product whose `(k, n)`
/// match this `B` and whose `C` is column-major-ish (`|csc| >= |rsc|`)
///
/// # Panics
///
/// Panics if `B`'s view addresses outside its slice, the same bounds check [`gemm_i8`]
/// runs. Also panics if `B` is so large that the pack buffer size overflows `usize`
#[cfg(feature = "int8")]
pub fn prepack_rhs_i8(b: MatRef<'_, i8>) -> PackedRhs<i8> {
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    // SAFETY: `b` is validated in-bounds directly above
    unsafe { prepack_rhs_i8_unchecked(b.data.as_ptr(), b.rs, b.cs, b.rows, b.cols) }
}

/// As [`prepack_rhs_i8`] but over a raw `k x n` `B` pointer and strides, with no bounds
/// check. Use this raw form for an adapter or FFI caller that validates its own inputs
///
/// # Safety
///
/// `b` must be valid for reads at every offset `i*rsb + j*csb`, for `i in 0..k` and
/// `j in 0..n`
#[cfg(feature = "int8")]
pub unsafe fn prepack_rhs_i8_unchecked(
    b: *const i8,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
) -> PackedRhs<i8> {
    // Resolve panel geometry from the memoized integer kernel's own microtile, the i8 mirror
    // of the float and half path above. A `tiny_block_dim() + 1` sentinel row count dodges
    // `blocking()`'s small-matrix shortcut, so the geometry works for every `m` a later call
    // brings. i8 packs in 1-byte units
    let (mr, nr) = dispatch::i8_rhs_tile();
    // An empty operand packs to nothing, the same short-circuit `prepack_rhs` uses
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
    // The driver's prepacked-RHS consume path assumes a single depth slice for any
    // `depth_multiple > 1` family, so VNNI (`depth_multiple == 4`) packs the whole
    // contraction as one panel here. The widen kernel (`depth_multiple == 1`) has no such
    // restriction and keeps the cache-model `kc`. Wrapping `i32` addition is associative, so
    // either choice of `kc` gives the same result
    let kc = if depth_multiple > 1 {
        k.max(1)
    } else {
        blk.kc.max(1)
    };
    let nc = blk.nc.next_multiple_of(nr).max(nr);

    // VNNI packs 4 depth steps per lane, so the panel depth pads up to `DEPTH_MULTIPLE`. The
    // widen kernel's multiple of 1 leaves this unchanged
    let k_pad = k.next_multiple_of(depth_multiple);
    // A broadcast (zero-stride) view can pass `check_view` while backed by a tiny slice, so
    // `n` and `k` may be logically huge here. Checked multiplication avoids under-sizing `buf`
    let total = n
        .div_ceil(nr)
        .checked_mul(nr)
        .and_then(|v| v.checked_mul(k_pad))
        .unwrap_or_else(|| {
            panic!("gemmkit: prepacked RHS of {k}x{n} is too large; the pack buffer size overflows usize")
        });
    let mut buf = vec![0i8; total];
    if total > 0 {
        // SAFETY: `buf` holds exactly `total = ceil(n/nr)*nr*k_pad` elements, the packed
        // layout size for the selected family. `b` is caller-promised valid for the `(k, n)`
        // strided reads (see # Safety), and `pack_rhs_full_i8` writes only within that range
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

/// `C(i32) <- alpha*A(i8)*(prepacked B) + beta*C`, consuming a [`PackedRhs<i8>`] (`B`
/// prepacked once) instead of `B` itself, using the thread-local workspace pool. This is the
/// integer (`i8 -> i32`) twin of [`gemm_packed_b`]. It skips the RHS pack that, for the VNNI
/// kernel, would otherwise run on every call
///
/// The result is bit-identical to plain [`gemm_i8`] under the same config, for every valid
/// shape and stride. Wrapping `i32` addition is associative regardless of grouping or ISA, so
/// every route, VNNI or widen, prepacked or per-call, produces the same sum. Output is
/// deterministic across thread counts
///
/// # Panics
///
/// Panics if the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `A` or `C` addresses outside its slice, or if `C` aliases itself
/// or `A`. Also panics if `C` is not column-major-ish (`|csc| >= |rsc|`), because a
/// row-major `C` would make the engine swap `A` and `B`, which a prepacked `B` cannot
/// support. Use plain [`gemm_i8`] for that layout
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
///
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

    // C (i32) must not alias A (i8). The check compares byte ranges because the element
    // sizes differ. The prepacked B is a separate owned buffer, so it can never alias C
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

/// As [`gemm_i8_packed_b`] but over raw `A`/`C` pointers and strides, with no bounds or
/// alias checks. This is the heterogeneous (`i8 -> i32`) counterpart of
/// [`gemm_packed_b_unchecked`]. The shared `k` and output `n` come from `packed`, and `m` is
/// `A`'s row count, the same as `C`'s. Uses the thread-local workspace pool
///
/// # Safety
///
/// `a` must be valid for reads over `(m, packed.rows())`. `c` must be valid for read and
/// write over `(m, packed.cols())` at the given strides, and must not alias `a`. When
/// `beta == 0`, `c` need not be initialized
///
/// # Panics
///
/// Panics if `C` is not column-major-ish (`|csc| >= |rsc|`), because a prepacked RHS cannot
/// serve a row-major `C`. Use plain [`gemm_i8`] for that layout
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
///
/// See [`gemm_i8_packed_b_unchecked`]
///
/// # Panics
///
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
    // A prepacked B was packed as the genuine RHS, so it only serves the no-swap orientation
    // (same assert and message as the float `gemm_packed_b_unchecked_with`)
    assert!(
        csc.unsigned_abs() >= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_b requires column-major-ish C (|csc| >= |rsc|); a row-major C \
         would swap A/B and invalidate the prepacked RHS — use gemm() for that layout"
    );
    // SAFETY: the caller guarantees A/C validity and that C does not alias A (see
    // # Safety). `packed` outlives this call, is read-only, and its `(nr, kc, nc)` fields
    // are exactly the geometry its buffer was packed with
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

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in one pass. This is a fused epilogue
/// over a reused [`PackedRhs`], using the thread-local workspace pool. It is the fused twin
/// of [`gemm_packed_b`]. The bias is folded in with 1 IEEE add right after the final
/// `beta`-scaled store. The activation applies next, fused into the same store the packed
/// kernel already runs. `bias == None && act == None` reproduces [`gemm_packed_b`]
/// bit-for-bit
///
/// The same [`PackedRhs`] handle serves both [`gemm_packed_b`] and this fused entry. The
/// epilogue is store-side only, so the pack and its recorded geometry stay untouched. For
/// `f32` and `f64` the result is bit-identical to [`gemm_packed_b`] followed by the same
/// scalar map, for every valid shape and stride. For `f16` and `bf16` the epilogue applies
/// in `f32` before the single narrowing, which is more precise than, and so not bitwise
/// equal to, packed-gemm-then-map
///
/// Unlike plain [`gemm_fused`], this path never reroutes to gemv or a small-shape special
/// path. It always drives the general prepacked kernel. Because it never swaps orientation,
/// the user-frame per-row or per-col bias passes straight through
///
/// # Panics
///
/// Same conditions as [`gemm_packed_b`], plus the fused conditions of [`gemm_fused`]. A
/// `PerRow` bias whose length is not `A.rows`, or a `PerCol` bias whose length is not
/// `B.cols`, causes a panic. A bias slice that overlaps `C`, or a non-finite `LeakyRelu`
/// slope, also causes a panic
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
///
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
    // The same validation `gemm_packed_b_with` runs, with identical panic wording
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

    // Fused-epilogue validation, shared wording with `gemm_fused`. Bias length must match
    // its axis (PerRow == A.rows, PerCol == packed B.cols == C.cols) and must not overlap C
    // A LeakyRelu slope must be finite. This path never swaps orientation, so the bias stays
    // in the user frame with no axis flip
    validate_bias(&bias, a.rows, packed.n, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validated above. A/C strides are in bounds, C addresses each (i,j) uniquely and
    // does not alias A, the prepacked B is a separate owned buffer, and the bias, borrowed for
    // this call, is the right length for its axis and disjoint from C. The bias pointer stays
    // valid for the whole `execute_packed_fused` frame
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

/// As [`gemm_packed_b_fused`] but over raw `A`/`C` pointers and strides, with no bounds or
/// alias checks. `bias` is a `(ptr, dim)` pair, enabled by `has_bias` and ignored when
/// `has_bias == false`, in the user frame, because this path never swaps orientation. `act`
/// applies last. Uses the thread-local workspace pool
///
/// # Safety
///
/// As [`gemm_packed_b_unchecked`], plus: when `has_bias`, `bias` must be valid for reads of
/// `m` (`PerRow`) or `packed.cols()` (`PerCol`) elements and must not alias `c`. A non-finite
/// `LeakyRelu` slope is the caller's responsibility, because the checked API rejects it
///
/// # Panics
///
/// As [`gemm_packed_b_unchecked`], for a non-column-major-ish `C`
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
///
/// See [`gemm_packed_b_fused_unchecked`]
///
/// # Panics
///
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

/// Shared lowering for the 4 fused B-packed entries. It asserts the packed-B orientation,
/// the same panic `gemm_packed_b_unchecked_with` raises. It builds the [`PackedConsume`]
/// and dispatches the prepacked-fused engine, either over a caller-owned [`Workspace`]
/// (`ws = Some`) or the thread-local pool (`ws = None`). `epi` arrives already lowered from
/// the bias and activation selectors. The packed-B consume frame is the user frame, with no
/// orientation swap, so `epi` passes through unflipped
///
/// # Safety
///
/// As [`gemm_packed_b_fused_unchecked`]. `a` must be valid for reads and `c` for read and
/// write over the shape and strides. `c` must not alias `a`, and `epi`'s bias, if any, must
/// be valid for `m` or `packed.cols()` reads and disjoint from `c`
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
    // A prepacked B was packed as the genuine RHS, so it only serves the no-swap orientation
    // (same assert and message as `gemm_packed_b_unchecked_with`)
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
    // SAFETY: the caller guarantees A/C validity and that C does not alias A. The packed
    // buffer, owned by `packed` and read-only, outlives the call and matches its recorded
    // geometry. `epi`'s bias is valid and disjoint from C (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_packed_fused(req, epi, par, ws),
            None => {
                workspace::with_thread_pool(|ws| dispatch::execute_packed_fused(req, epi, par, ws))
            }
        }
    }
}

/// An `A` matrix packed once into gemmkit's internal micropanel-major layout, for reuse
/// across many products that share the same `A`. This fits a fixed weight matrix streamed
/// against differently shaped `B` operands. [`prepack_lhs`] builds it. [`gemm_packed_a`] and
/// its `_with`, `_unchecked`, and `_fused` siblings consume it and skip the per-call LHS pack
///
/// By the engine's A/B symmetry, a prepacked LHS is exactly the prepacked RHS of the
/// transposed product `C^T = B^T*A^T`. The buffer stores that transposed problem's blocking
/// geometry, and the consuming call, itself driven transposed, reads it back unchanged. The
/// buffer is read-only for the whole GEMM, so it needs no synchronization across worker
/// threads
pub struct PackedLhs<T> {
    buf: Vec<T>,
    m: usize,
    k: usize,
    nr: usize,
    kc: usize,
    nc: usize,
}

impl<T> PackedLhs<T> {
    /// Row count of the original `A` (the `m` dimension)
    pub fn rows(&self) -> usize {
        self.m
    }
    /// Column count of the original `A` (the shared contraction dimension `k`)
    pub fn cols(&self) -> usize {
        self.k
    }
}

/// Pack an `m x k` `A` view into a [`PackedLhs`] for reuse across many [`gemm_packed_a`]
/// calls. The pack runs once, single-threaded, right here, so every later call skips it
///
/// The pack accepts any layout of `A` and reads it through its strides. The resulting
/// buffer is only valid for a product whose `(m, k)` match this `A` and whose `C` is
/// row-major-ish (`|csc| <= |rsc|`). [`gemm_packed_a`] checks both before consuming it
///
/// # Panics
///
/// Panics if `A`'s view addresses outside its slice, the same bounds check [`gemm`] runs.
/// Also panics if `A` is so large that the pack buffer size overflows `usize`. A broadcast
/// stride can make `A`'s logical dimensions run up to `isize::MAX` while addressing only a
/// few elements
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    // SAFETY: `a` is validated in-bounds directly above
    unsafe { prepack_lhs_unchecked(a.data.as_ptr(), a.rs, a.cs, a.rows, a.cols) }
}

/// As [`prepack_lhs`] but over a raw `m x k` `A` pointer and strides, with no bounds check.
/// Use this raw form for an adapter or FFI caller that validates its own inputs
///
/// # Safety
///
/// `a` must be valid for reads at every offset `i*rsa + j*csa`, for `i in 0..m` and
/// `j in 0..k`
pub unsafe fn prepack_lhs_unchecked<T: GemmScalar>(
    a: *const T,
    rsa: isize,
    csa: isize,
    m: usize,
    k: usize,
) -> PackedLhs<T> {
    // By the engine's A/B symmetry, a prepacked LHS is exactly the prepacked RHS of the
    // transposed product `C^T = B^T*A^T`. This `m x k` LHS is that problem's `k x m` RHS
    // The LHS column stride plays the RHS row (depth) stride, and the LHS row stride plays
    // the RHS column stride. Delegating to `prepack_rhs_unchecked` keeps one pack and
    // geometry path as the single source of truth. It lays down the same micropanel-major
    // buffer the transposed-driven consuming call reads back
    //
    // Only the recorded dimensions are relabeled into LHS terms below. One side effect: the
    // unreachable overflow panic reports the problem as an RHS of `{k}x{m}` rather than an
    // LHS of `{m}x{k}`
    //
    // SAFETY: `a` is caller-promised valid for the `(m, k)` reads at `i*rsa + j*csa` (see
    // # Safety). Those are exactly the `(k, n = m)` reads `prepack_rhs_unchecked` performs
    // under the transposed strides `(rsb = csa, csb = rsa)`
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

/// `C <- alpha*A*B + beta*C`, consuming a [`PackedLhs`] (`A` prepacked once) instead of `A`
/// itself, using the thread-local workspace pool. This skips the per-call LHS pack that
/// [`gemm`] would run
///
/// The result reproduces plain [`gemm`] under the same config, except that 2 shapes may
/// differ in the last ULP while staying correct. This can happen for a small product,
/// where both `m` and `n` are at or below [`crate::tuning::small_mn_dim`], or for a
/// gemv-shaped product, where `m == 1` or `n == 1`. As with [`gemm_packed_b`], this path
/// always drives the general driver through the transposed consume. Plain `gemm` instead
/// reroutes those shapes to a dedicated kernel with a different accumulation order
///
/// # Panics
///
/// Panics if the dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if `B` or `C` addresses outside its slice, or if `C` aliases itself
/// or `B`. Also panics if `C` is not row-major-ish (`|csc| <= |rsc|`), because a
/// column-major `C` would leave `A` in the genuine LHS role, which a prepacked `A`, laid out
/// as the transposed RHS, cannot serve. Use plain [`gemm`] for that layout
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
///
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

    // C must not alias B, because C is written. The prepacked A is a separate owned buffer,
    // so it can never alias C
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

/// As [`gemm_packed_a`] but over raw `B`/`C` pointers and strides, with no bounds or alias
/// checks. The shared `k` and output-row count `m` come from `packed`, and `n` is `B`'s
/// column count, the same as `C`'s. Uses the thread-local workspace pool
///
/// # Safety
///
/// `b` must be valid for reads over `(packed.cols(), n)`. `c` must be valid for read and
/// write over `(packed.rows(), n)` at the given strides, and must not alias `b`. When
/// `beta == 0`, `c` need not be initialized
///
/// # Panics
///
/// Panics if `C` is not row-major-ish (`|csc| <= |rsc|`), because a prepacked LHS cannot
/// serve a column-major `C`. Use plain [`gemm`] for that layout
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
///
/// See [`gemm_packed_a_unchecked`]
///
/// # Panics
///
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
    // A prepacked A was packed as the transposed problem's RHS, so it only serves the
    // orientation where A keeps that role
    assert!(
        csc.unsigned_abs() <= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_a requires row-major-ish C (|csc| <= |rsc|); a column-major C \
         would keep A in the LHS role and invalidate the prepacked LHS — use gemm() for that layout"
    );

    // SAFETY: the caller guarantees B/C validity and that C does not alias B (see
    // # Safety). `packed` outlives this call, is read-only, and its `(nr, kc, nc)` fields
    // are exactly the geometry its buffer was packed with
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

/// `C <- act(alpha*(prepacked A)*B + beta*C + bias)` in one pass. This is a fused epilogue
/// over a reused [`PackedLhs`], using the thread-local workspace pool. It is the fused twin
/// of [`gemm_packed_a`]. The bias is folded in with 1 IEEE add right after the final
/// `beta`-scaled store. The activation applies next, fused into the same store the packed
/// kernel already runs. `bias == None && act == None` reproduces [`gemm_packed_a`]
/// bit-for-bit
///
/// The same [`PackedLhs`] handle serves both [`gemm_packed_a`] and this fused entry. The
/// epilogue is store-side only, so the pack stays untouched. For `f32` and `f64` the result
/// is bit-identical to [`gemm_packed_a`] followed by the same scalar map, for every valid
/// shape and stride. For `f16` and `bf16` the epilogue applies in `f32` before the single
/// narrowing, which is more precise than, and so not bitwise equal to, packed-gemm-then-map
///
/// Like [`gemm_packed_b_fused`], this path never reroutes to a special kernel and always
/// drives the general prepacked one. The bias is specified in the user frame: a `PerRow`
/// bias of length `A.rows` adds to every column of that row. The engine handles its internal
/// transpose of the packed-A product underneath, so the caller never sees an axis flip
///
/// # Panics
///
/// Same conditions as [`gemm_packed_a`], plus the fused conditions of [`gemm_fused`]. A
/// `PerRow` bias whose length is not `A.rows`, or a `PerCol` bias whose length is not
/// `B.cols`, causes a panic. A bias slice that overlaps `C`, or a non-finite `LeakyRelu`
/// slope, also causes a panic
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
///
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
    // The same validation `gemm_packed_a_with` runs, with identical panic wording
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

    // Fused-epilogue validation, shared wording with `gemm_fused`. Bias length must match
    // its user axis (PerRow == packed A.rows == C.rows, PerCol == B.cols) and must not
    // overlap C. A LeakyRelu slope must be finite. Bias stays in the user frame here
    // `packed_a_fused_impl` flips the axis to match the transposed consume the packed-A
    // path actually drives
    validate_bias(&bias, packed.m, b.cols, &c);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    let epi = to_fused_epi(bias, act);

    // SAFETY: validated above. B/C strides are in bounds, C addresses each (i,j) uniquely and
    // does not alias B, the prepacked A is a separate owned buffer, and the bias, borrowed for
    // this call, is the right length for its axis and disjoint from C. The bias pointer stays
    // valid for the whole `execute_packed_fused` frame
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

/// As [`gemm_packed_a_fused`] but over raw `B`/`C` pointers and strides, with no bounds or
/// alias checks. `bias` is a `(ptr, dim)` pair, enabled by `has_bias` and ignored when
/// `has_bias == false`, in the user frame, where a `PerRow` bias indexes `A.rows`, the same
/// as `C.rows`. `act` applies last. Uses the thread-local workspace pool
///
/// # Safety
///
/// As [`gemm_packed_a_unchecked`], plus: when `has_bias`, `bias` must be valid for reads of
/// `packed.rows()` (`PerRow`) or `n` (`PerCol`) elements and must not alias `c`. A
/// non-finite `LeakyRelu` slope is the caller's responsibility, because the checked API
/// rejects it
///
/// # Panics
///
/// As [`gemm_packed_a_unchecked`], for a non-row-major-ish `C`
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
///
/// See [`gemm_packed_a_fused_unchecked`]
///
/// # Panics
///
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

/// Shared lowering for the 4 fused A-packed entries. It asserts the packed-A orientation,
/// the same panic `gemm_packed_a_unchecked_with` raises. It flips the user-frame bias axis
/// to the transposed consume frame the packed-A path drives. It then builds the transposed
/// [`PackedConsume`] the same way `gemm_packed_a_unchecked_with` does. It dispatches the
/// prepacked-fused engine over either a caller-owned [`Workspace`] (`ws = Some`) or the
/// thread-local pool (`ws = None`)
///
/// By the engine's A/B symmetry, the packed-A product is driven as the transposed problem
/// `C^T = B^T*A^T`, with `m` and `n` swapped in the consume frame. A user per-row bias,
/// indexed by the user output row, becomes per-col in the consume frame, and vice versa.
/// This is the same field-write flip `run_typed_fused` applies on a dynamic orientation
/// swap. Here it is built into the always-transposed packed-A path, so the user bias axis
/// stays correct
///
/// # Safety
///
/// As [`gemm_packed_a_fused_unchecked`]. `b` must be valid for reads and `c` for read and
/// write over the shape and strides. `c` must not alias `b`. `epi`'s bias, if any, must be
/// valid for `packed.rows()` or `n` reads and disjoint from `c`
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
    // A prepacked A was packed as the transposed problem's RHS, so it only serves the
    // row-major-ish orientation (same assert and message as `gemm_packed_a_unchecked_with`)
    assert!(
        csc.unsigned_abs() <= rsc.unsigned_abs(),
        "gemmkit: gemm_packed_a requires row-major-ish C (|csc| <= |rsc|); a column-major C \
         would keep A in the LHS role and invalidate the prepacked LHS — use gemm() for that layout"
    );
    // The packed-A path always drives the transposed product, so the user's bias axis must
    // flip to the oriented consume frame before dispatch. Per-row becomes per-col and vice
    // versa. `execute_packed_fused` then applies `epi` in that oriented frame, which maps the
    // flipped value back onto the user's intended row or column
    epi.bias = match epi.bias {
        BiasSpec::None => BiasSpec::None,
        BiasSpec::Row(p) => BiasSpec::Col(p),
        BiasSpec::Col(p) => BiasSpec::Row(p),
    };
    // The same transposed consume `gemm_packed_a_unchecked_with` builds: m<->n swapped, A
    // takes B's role, and the strides swap along with it
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
    // SAFETY: the caller guarantees B/C validity and that C does not alias B. The packed
    // buffer, owned by `packed` and read-only, outlives the call and matches its recorded
    // geometry. `epi`'s bias is valid and disjoint from C (see # Safety)
    unsafe {
        match ws {
            Some(ws) => dispatch::execute_packed_fused(req, epi, par, ws),
            None => {
                workspace::with_thread_pool(|ws| dispatch::execute_packed_fused(req, epi, par, ws))
            }
        }
    }
}
