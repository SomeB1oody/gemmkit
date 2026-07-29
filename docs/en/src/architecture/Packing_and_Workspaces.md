# Packing and Workspaces

The microkernel wants its inputs in exactly one shape. For each depth step, it wants `mr` elements of `A` contiguous in memory, and `nr` elements of `B` contiguous in memory. It wants them panel after panel, with nothing between them. User matrices almost never look like that. They have arbitrary row and column strides, tails that do not divide the microtile, and depth walks that can stride across memory pages.

Packing is the copy that closes this gap. It rearranges each macro-block into *micropanel-major* layout once, so the innermost loop reads pure unit-stride streams, full `mr`/`nr` vectors every time, from 64-byte-aligned scratch. The copy costs `O(mc*kc)`, against the `O(mc*kc*nc)` compute that reuses it. That is why it amortizes, and why the driver still skips it when reuse is too low to pay for it.

## One routine, both operands

The mechanical copy lives in a single routine, `pack_panels` (`gemmkit/src/pack.rs`, layer L1). The LHS and RHS layouts are the same layout, viewed from different sides. An LHS macro-block packs into panels `mr` rows tall, stored column by column. Panel 0 holds rows `0..mr`, with each depth step's `mr` elements contiguous. Panel 1 holds rows `mr..2*mr`, and so on. An RHS macro-block packs into panels `nr` columns wide, stored row by row.

Both layouts are `width` contiguous leading elements per depth step. The only difference is which matrix axis plays the leading role. So the 2 `KernelFamily` hooks call the same routine, with the strides swapped:

```rust
// gemmkit/src/kernel/float.rs
#[inline]
unsafe fn pack_rhs(
    dst: *mut T,
    src: *const T,
    rs: isize,
    cs: isize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    // RHS panels are `nr` columns wide, stored row-by-row: the "leading"
    // direction is columns (stride `cs`) and the "depth" is rows (stride
    // `rs`), the transpose of the LHS case, handled by swapping strides
    unsafe {
        pack_panels(
            dst, src, /*lead*/ cs, /*depth*/ rs, /*n_lead*/ nc, kc, nr,
        )
    }
}
```

`pack_lhs` is the mirror image: `lead = rs`, `depth = cs`, `width = mr`. When the block does not divide evenly, the routine zero-fills the tail panel's dead lanes. The kernel then always reads full `mr`/`nr` vectors, and edge tiles need no masking in the multiply itself.

Inside the routine, 2 paths write byte-identical output. The 1st case is a contiguous leading dimension (`lead == 1`), for a column-major `A` or a row-major `B`. There, each depth step's `live` elements already sit adjacent in the source. The panel is then a sequence of straight `copy_nonoverlapping` calls, plus a tail zero-fill.

The 2nd case is a strided leading dimension. There, a naive gather would take a cache miss per element (`width` strided loads per depth step). Instead the routine runs a *cache-blocked transpose*. It walks the source along its contiguous dimension in strips of `GEMMKIT_PACK_TRANSPOSE_TILE` depth steps (default 16), and scatters each strip into the panel. This produces the same packed bytes as a pure reordered copy, but far more cheaply for a strided source. That is what makes row-major-`A` layouts cost little more than column-major ones.

The dot-product families (`i8` VNNI, `bf16` `vdpbf16ps`) have a sibling routine, `pack_kgroup_panels`. It additionally interleaves `DEPTH_MULTIPLE` consecutive depth steps per lane, so one dot instruction can consume a whole group. That layout belongs to [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md).

Whether to pack at all is the driver's call, and the 2 operands are asymmetric.

The microkernel reads `A` as `mr`-wide vectors. So the driver *must* pack `A` whenever its rows are not unit-stride, or the row panel is partial. Beyond that, the driver packs `A` when each worker's column reuse clears `GEMMKIT_LHS_PACK_THRESHOLD` (default 256 columns on aarch64, 1024 elsewhere).

The driver also packs a column-major `A` under one more condition. Its depth walk must be page-scale in stride, wide in span, and reused by enough column tiles to be worth the cost. All 3 conditions must hold together:

1. The per-step stride reaches half a memory page (`GEMMKIT_LHS_PACK_STRIDE`, auto-derived from the page size memoized in `Machine`).
2. The whole depth-slice walk (`csa * sizeof(Lhs) * kc`) reaches `GEMMKIT_LHS_PACK_SPAN` bytes (auto: 4 MiB).
3. At least `GEMMKIT_LHS_PACK_REUSE` `nr`-wide column tiles reuse each packed panel (`min(n, nc) / nr`, rounded up, default 128 on x86, 4 on aarch64).

Each gate rules out a different case where packing would not pay for itself. A page-scale stride over a span that still fits in cache only re-walks lines that are already warm. Reading `A` in place then costs less than paying for a pack. The span condition keeps `A` in place until the walk is wide enough to thrash the TLB, regardless of how much reuse follows.

The reuse floor prices the opposite failure mode. Take a tall, skinny shape, where `m` is much greater than `n`. It reaches a large span from very few column tiles. Packing it would then amortize an expensive copy over too little reuse to be worth it.

The reuse floor differs by architecture, because the pack-versus-read-in-place trade differs by architecture. On x86 a pack costs more relative to an in-place read. So the driver waits for more reuse before it pays for one, and the default floor is 128 tiles. On aarch64 a pack costs less relative to an in-place strided read. So the driver packs sooner, and the default floor is 4 tiles.

`B`, by contrast, is only ever read by broadcasting single elements, so any layout works unpacked. The driver packs `B` purely for reuse: once per depth slice, when `m` clears `GEMMKIT_RHS_PACK_THRESHOLD` (default 2048) and enough row blocks will re-read it. Who performs these packs, and the barriers between packing and compute, are scheduling questions. [Parallel Execution](Parallel_Execution.md) covers them.

## Prepacked operands

A per-call pack is wasted work when the same operand appears in call after call. This is the inference pattern: fixed weights multiplied against a stream of activations. The prepack entries in `gemmkit/src/api/packed.rs` pack a whole operand once, up front. `prepack_rhs` walks any-layout `B` through its strides and returns a `PackedRhs<T>`. `gemm_packed_b` then multiplies against it, skipping the per-call RHS pack entirely. [Prepacked Operands](../gemmkit-guide/Prepacked_Operands.md) covers the usage side of this API. Architecturally, 3 properties matter.

First, the buffer records the blocking geometry it was built for: `nr`, `kc`, and `nc`. The consuming call reads that geometry back verbatim. The driver substitutes the recorded `kc` and `nc` for its own model output. Only `mc` still derives from the real `m`. So panel addresses always match what the buffer packed, even if a tuning knob changed between the pack call and the consuming call. The geometry itself resolves through the same `blocking` model as a plain call, with a sentinel row count of `tiny_block_dim() + 1`. This keeps it off the tiny-matrix branch, so it stays independent of the eventual `m`.

Second, the layout has one source of truth. `prepack_rhs` fills the buffer through `driver::pack_rhs_full`. This lays panels down in exactly the order the driver's own per-slice pack writes them. The order is `jc` blocks outermost, then depth slices, then the `nr`-wide panels of each slice. The prepacked bytes therefore equal the per-call packed bytes. So a prepacked GEMM *reproduces* a plain `gemm` under the same configuration. The documented exceptions are tiny products (`m` and `n` at or below `tiny_block_dim`) and gemv-shaped products. Plain `gemm` reroutes those to special paths, so they may differ in the last ULP.

Third, the buffer stays read-only during the GEMM, so every worker shares it with no synchronization. Unlike the per-call `B` pack, it needs no barrier.

`PackedLhs` costs almost no extra code, because of the engine's A/B symmetry. An `m x k` LHS *is* the RHS of the transposed product `C^T = B^T*A^T`. So `prepack_lhs` delegates to `prepack_rhs_unchecked` with the strides swapped, and `gemm_packed_a` consumes it through the transposed problem. This symmetry also explains the orientation asserts. A prepacked `B` requires a column-major-ish `C` (`|csc| >= |rsc|`), and a prepacked `A` requires a row-major-ish one. The other orientation would make dispatch swap the operand roles, and the baked-in layout could not serve that swap.

The `int8` feature adds a heterogeneous twin, `prepack_rhs_i8` and `gemm_i8_packed_b`, with 3 deliberate differences.

First, `prepack_rhs_i8` pins its layout to whichever integer kernel the process's memoized dispatch selected: the VNNI k-quad-interleaved layout, or the widen kernel's plain panels. The consuming entry always runs that same family, so it can never misread the buffer.

Second, it rounds the buffer depth up to the dot kernel's `DEPTH_MULTIPLE = 4` and packs the whole contraction as one depth slice. This satisfies the driver's single-slice guard for depth-padded families.

Third, it deliberately bypasses the dynamic small-parallel widen fallback that plain `gemm_i8` applies below `GEMMKIT_I8_VNNI_MIN_PAR_MNK`. A `vpdpbusd` buffer is quad-interleaved, so the widen kernel simply cannot consume it. Because integer accumulation is exact, the result is bit-identical to plain `gemm_i8` either way.

Prepacking matters most on exactly this path. The VNNI RHS pack is otherwise mandatory on every call, so at small `m` the per-call `O(k*n)` pack cost dominates the `O(m*k*n)` compute.

## The workspace

All of this packing needs scratch memory. `Workspace` (`gemmkit/src/workspace.rs`) is its allocator: a growable buffer, 64-byte aligned (enough for AVX-512 stores), that grows to the next power of 2 and never shrinks. Per call, `Workspace::regions` carves it into `a_regions` equal LHS regions plus one shared RHS region, with each region rounded up to the alignment.

The LHS region count is the worker count on the per-worker pack path, or the row-block count on the shared-`A` path. The carving works the same way either way. When neither operand packs, the driver skips the reservation entirely, so an all-in-place workload never grows the pool.

### Fail closed at the byte product

The sizing arithmetic is where a memory-safety subtlety hides. gemmkit accepts broadcast (zero-stride) views. These pass bounds validation with a tiny backing slice, while presenting *logical* dimensions up to `isize::MAX`. So the products that size the pack buffer can genuinely overflow `usize`. A wrapped (too-small) size would then under-allocate a buffer that the pack writes past.

The driver guards its element-count products with `checked_mul`, but element counts alone are not enough. Take `k = 2^56` on the mixed-precision path, where `kc == k`. An LHS region of `mc * kc` elements, say `32 * 2^56 = 2^61`, fits `usize` comfortably and sails through every element-level check. Multiply that count by the element size and round up to the 64-byte alignment, though, and the value wraps. The overflow only appears at the element-to-byte conversion. That is where the guard must sit: at the chokepoint every region size funnels through:

```rust
// gemmkit/src/workspace.rs
fn region_bytes(elems: usize, esize: usize) -> usize {
    elems
        .checked_mul(esize)
        .and_then(|b| b.checked_next_multiple_of(ALIGN))
        .unwrap_or_else(|| workspace_too_large())
}
```

`Workspace` checks every step: the byte product, the alignment round-up, the region sum, and the final `A + B` total. Any overflow panics with the same "too large" contract as the driver's own sizing. This is fail closed. The code rejects an absurd problem loudly, instead of corrupting memory. The driver runs its element-count guards unconditionally for the same reason, even on routes that end up packing nothing. Skipping them would also skip the abort, and send the absurd `k` into the in-place loops to spin for an effectively unbounded time.

### The pool, `_with`, and `no_std`

Callers rarely see a `Workspace`, because a thread-local pool supplies one transparently. The common `gemm` call allocates at most once per thread, and reuses that buffer for every later call.

The pool is also re-entrancy-safe. Nested rayon can re-enter a GEMM on a thread already inside one. For example, a worker might work-steal another GEMM while blocked in its own `for_each`, or a batch-parallel worker might run an element inline. In that case the pool's `RefCell` is already borrowed. So `with_thread_pool` hands out a fresh scratch workspace for that one call, instead of panicking. Packing buffers hold no result state between calls, so this fallback is invisible. Only that one call skips the buffer reuse.

For explicit control there is the `*_with` tier. Every entry has a variant (`gemm_with`, `gemm_packed_b_with`, and so on) that threads a caller-owned `Workspace` through. From the second sufficiently large call on, this gives zero heap allocation. It is the tool for hot loops of small products and for latency-sensitive code, and `Workspace::with_capacity` avoids even the first-call allocation spike.

Without `std` there is no thread-local storage, so `with_thread_pool` simply builds a fresh workspace per call. Because `parallel` requires `std`, there are no threads to re-enter either. A caller who wants reuse on such a build holds its own `Workspace` and uses `*_with`. [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md) recommends this same pattern.
