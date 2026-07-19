# Packing and Workspaces

The microkernel wants its inputs in exactly one shape: for each depth step, `mr` elements of `A` contiguous in memory and `nr` elements of `B` contiguous in memory, panel after panel, with nothing between them. User matrices almost never look like that - they have arbitrary row and column strides, tails that do not divide the microtile, and depth walks that can stride across memory pages. Packing is the copy that closes the gap: it rearranges each macro-block into *micropanel-major* layout once, so the innermost loop reads pure unit-stride streams, full `mr`/`nr` vectors every time, from 64-byte-aligned scratch. The copy is `O(mc*kc)` against the `O(mc*kc*nc)` compute that reuses it, which is why it amortizes - and why the driver still skips it when reuse is too low to pay for it.

## One routine, both operands

The mechanical copy lives in a single routine, `pack_panels` (`gemmkit/src/pack.rs`, layer L2), because the LHS and RHS layouts are the same layout viewed from different sides. An LHS macro-block packs into panels `mr` rows tall, stored column by column: panel 0 holds rows `0..mr` with each depth step's `mr` elements contiguous, panel 1 holds rows `mr..2*mr`, and so on. An RHS macro-block packs into panels `nr` columns wide, stored row by row. Both are "`width` contiguous leading elements per depth step"; the only difference is which matrix axis is the leading one, so the two `KernelFamily` hooks call the same routine with the strides swapped:

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

`pack_lhs` is the mirror image: `lead = rs`, `depth = cs`, `width = mr`. When the block does not divide evenly, the tail panel's dead lanes are zero-filled, so the kernel always reads full `mr`/`nr` vectors and edge tiles need no masking in the multiply itself.

Inside the routine there are two paths that write byte-identical output. When the leading dimension is contiguous (`lead == 1` - a column-major `A` or row-major `B`), each depth step's `live` elements are already adjacent in the source, so the panel is a sequence of straight `copy_nonoverlapping` calls plus tail zero-fill. When the leading dimension is strided, a naive gather would take a cache miss per element (`width` strided loads per depth step); instead the routine runs a *cache-blocked transpose*: it walks the source along its contiguous dimension in strips of `GEMMKIT_PACK_TRANSPOSE_TILE` depth steps (default 16) and scatters each strip into the panel. A pure reordered copy - the packed bytes are identical - but far cheaper for a strided source, which is what makes row-major-`A` layouts cost little more than column-major ones. The dot-product families (`i8` VNNI, `bf16` `vdpbf16ps`) have a sibling routine, `pack_kgroup_panels`, that additionally interleaves `DEPTH_MULTIPLE` consecutive depth steps per lane so one dot instruction can consume a whole group; that layout belongs to [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md).

Whether to pack at all is the driver's call, and the two operands are asymmetric. The microkernel reads `A` as `mr`-wide vectors, so `A` *must* be packed whenever its rows are not unit-stride or the row panel is partial; beyond that it is packed when each worker's column reuse clears `GEMMKIT_LHS_PACK_THRESHOLD` (default 256 columns on aarch64, 1024 elsewhere), or when a column-major `A`'s depth stride reaches half a memory page (`GEMMKIT_LHS_PACK_STRIDE`, auto-derived from the page size memoized in `Machine`) - past that point the in-place depth walk thrashes the TLB regardless of reuse. `B`, by contrast, is only ever read by broadcasting single elements, so any layout works unpacked, and it is packed purely for reuse: once per depth slice, when `m` clears `GEMMKIT_RHS_PACK_THRESHOLD` (default 2048) and enough row blocks will re-read it. Who performs these packs, and the barriers between packing and compute, are scheduling questions covered in [Parallel Execution](Parallel_Execution.md).

## Prepacked operands

A per-call pack is wasted work when the same operand appears in call after call - the inference pattern of fixed weights against a stream of activations. The prepack entries in `gemmkit/src/api/packed.rs` pack a whole operand once, up front: `prepack_rhs` walks any-layout `B` through its strides and returns a `PackedRhs<T>`; `gemm_packed_b` then multiplies against it with the per-call RHS pack skipped entirely. The usage side of this API lives in [Prepacked Operands](../gemmkit-guide/Prepacked_Operands.md); architecturally, three properties matter.

First, the buffer records the blocking geometry it was built for - `nr`, `kc`, `nc` - and the consuming call reads it back verbatim: the driver substitutes the recorded `kc`/`nc` for its own model output (only `mc` is still derived at the real `m`), so panel addresses always match what was packed, even if a tuning knob changed between pack and consume. The geometry itself is resolved through the same `blocking` model as a plain call, with a sentinel row count of `tiny_block_dim() + 1` so it never takes the tiny-matrix branch and is therefore independent of the eventual `m`.

Second, the layout has one source of truth: `prepack_rhs` fills the buffer via `driver::pack_rhs_full`, which lays panels down in exactly the order the driver's own per-slice pack writes them - `jc` blocks outermost, then depth slices, then the `nr`-wide panels of each slice. Because the prepacked bytes equal the per-call packed bytes, a prepacked GEMM *reproduces* a plain `gemm` under the same config; the documented exceptions are tiny (`m` and `n` at or below `tiny_block_dim`) and gemv-shaped products, which plain `gemm` reroutes to special paths and may therefore differ in the last ULP. Third, the buffer is read-only during the GEMM, so all workers share it with no synchronization - unlike the per-call `B` pack, it needs no barrier.

`PackedLhs` costs almost no extra code because of the engine's A/B symmetry: an `m x k` LHS *is* the RHS of the transposed product `C^T = B^T*A^T`, so `prepack_lhs` delegates to `prepack_rhs_unchecked` with the strides swapped, and `gemm_packed_a` consumes through the transposed problem. The symmetry also explains the orientation asserts: a prepacked `B` requires a column-major-ish `C` (`|csc| >= |rsc|`) and a prepacked `A` a row-major-ish one, because the other orientation would make dispatch swap the operand roles and the baked-in layout could not serve.

The `int8` feature adds a heterogeneous twin, `prepack_rhs_i8` / `gemm_i8_packed_b`, with three deliberate differences. Its layout is pinned to whichever integer kernel the process's memoized dispatch selected - the VNNI k-quad-interleaved layout or the widen kernel's plain panels - and the consume entry always runs that same family, so the buffer can never be misread. It rounds the buffer depth up to the dot kernel's `DEPTH_MULTIPLE = 4` and packs the whole contraction as one depth slice, satisfying the driver's single-slice guard for depth-padded families. And it deliberately bypasses the dynamic small-parallel widen fallback that plain `gemm_i8` applies below `GEMMKIT_I8_VNNI_MIN_PAR_MNK`: a `vpdpbusd` buffer is quad-interleaved and simply not consumable by the widen kernel, and since integer accumulation is exact the result is bit-identical to plain `gemm_i8` either way. Prepacking matters most on exactly this path - the VNNI RHS pack is otherwise mandatory on every call, so at small `m` the per-call `O(k*n)` pack dominates the `O(m*k*n)` compute.

## The workspace

All of this packing needs scratch memory, and `Workspace` (`gemmkit/src/workspace.rs`) is its allocator: a growable buffer, 64-byte aligned (enough for AVX-512 stores), that grows to the next power of two and never shrinks. Per call, `Workspace::regions` carves it into `a_regions` equal LHS regions plus one shared RHS region, each region rounded up to the alignment. The LHS region count is the worker count on the per-worker pack path or the row-block count on the shared-`A` path - the carving is identical either way - and when neither operand packs, the driver skips the reservation entirely so an all-in-place workload never grows the pool.

### Fail closed at the byte product

The sizing arithmetic is where a memory-safety subtlety hides. gemmkit accepts broadcast (zero-stride) views, which pass bounds validation with a tiny backing slice while presenting *logical* dimensions up to `isize::MAX` - so the products that size the pack buffer can genuinely overflow `usize`, and a wrapped (too-small) size would under-allocate a buffer the pack then writes past. The driver guards its element-count products with `checked_mul`, but element counts are not enough: take `k = 2^56` on the mixed-precision path, where `kc == k`. An LHS region of `mc * kc` elements - say `32 * 2^56 = 2^61` - fits `usize` comfortably and sails through every element-level check; multiply by the element size and round up to the 64-byte alignment, and the value wraps. The overflow only materializes at the element-to-byte conversion, so that is where the guard must sit - at the chokepoint every region size funnels through:

```rust
// gemmkit/src/workspace.rs
fn region_bytes(elems: usize, esize: usize) -> usize {
    elems
        .checked_mul(esize)
        .and_then(|b| b.checked_next_multiple_of(ALIGN))
        .unwrap_or_else(|| workspace_too_large())
}
```

Every step - the byte product, the alignment round-up, the region sum, the final `A + B` total - is checked, and any overflow panics with the same "too large" contract as the driver's own sizing. Fail closed: an absurd problem is rejected loudly instead of corrupting memory. The driver runs its element-count guards unconditionally for the same reason, even on routes that end up packing nothing - skipping them would also skip the abort and send the absurd `k` into the in-place loops to spin effectively forever.

### The pool, `_with`, and `no_std`

Callers rarely see a `Workspace` because a thread-local pool supplies one transparently: the common `gemm` call allocates at most once per thread and reuses that buffer for every later call. The pool is re-entrancy-safe - nested rayon can re-enter a GEMM on a thread already inside one (a worker that work-steals another GEMM while blocked in its own `for_each`, or a batch-parallel worker running an element inline), and since the pool's `RefCell` is then already borrowed, `with_thread_pool` hands out a fresh scratch workspace that one time instead of panicking. Packing buffers hold no result state, so the fallback is invisible; only the buffer reuse is skipped.

For explicit control there is the `*_with` tier: every entry has a variant (`gemm_with`, `gemm_packed_b_with`, ...) that threads a caller-owned `Workspace` through, giving zero heap allocation from the second sufficiently large call on - the tool for hot loops of small products and latency-sensitive code, and `Workspace::with_capacity` avoids even the first-call spike. Without `std` there is no thread-local storage, so `with_thread_pool` simply builds a fresh workspace per call; since `parallel` requires `std`, there are no threads to re-enter, and callers who want reuse hold a `Workspace` and use `*_with` - which is also the recommended pattern in [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md).
