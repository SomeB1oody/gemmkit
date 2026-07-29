# Life of a GEMM Call

The previous page described the stack at rest. This page follows a single call through
it. The specimen is the quick-start example from the crate docs:

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

// 2x3 * 3x2 = 2x2, all row-major
let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
let mut c = [0.0_f32; 4];
gemm(
    1.0,
    MatRef::from_row_major(&a, 2, 3),
    MatRef::from_row_major(&b, 3, 2),
    0.0,
    MatMut::from_row_major(&mut c, 2, 2),
    Parallelism::Serial,
);
assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
```

This toy shape takes one of the early exits below. The walk keeps 2 problems in
mind: the 2x2x3 example above, and a 2048x2048x2048 `f32` product on an AVX-512
machine. That larger product goes all the way down, through every layer. Here is the
route, compressed:

```
gemm(alpha, A, B, beta, C, par)
  |  validate_gemm_views: shapes, bounds, aliasing     [api.rs]
  v
Task<T>: raw pointers + isize strides
  |  m == 0 || n == 0        -> return                 [dispatch.rs]
  |  k == 0 || alpha == 0    -> C <- beta*C, done
  v
memoized per-type kernel (OnceLock fn pointer)
  |  gemv shape (m==1||n==1) -> special/gemv.rs        [dispatch/float.rs]
  |  orient: row-major-ish C -> compute C^T = B^T*A^T
  |  small m,n + long k      -> special/small_mn.rs
  |  k <= small_k_threshold  -> special/small_k.rs
  v
driver::run                                            [driver.rs]
  jc over NC -> pc over KC (never parallel)
    -> flat job list (ic row-block x jt column-tile),
       workers drain a shared JobCursor, pack A/B adaptively
  v
Fam::microkernel_epi: MR x NR tile in registers        [kernel/float.rs]
  alpha/beta epilogue store (vector fast path | scratch drain)
```

## Stage 1: validation and lowering

`gemm` itself is one line. It borrows the thread-local workspace and forwards to
`gemm_with` (`gemmkit/src/api.rs`). `gemm_with` runs `validate_gemm_views`, the full
panic catalog from [Design Goals](Design_Goals_and_the_Big_Picture.md). Shapes must
agree. Every view must stay inside its slice. `C` must address each `(i, j)` uniquely.
`C` must overlap neither input.

Then the views dissolve. Everything below this point speaks `Task<T>`: a `Copy` struct
of `m, k, n`, `alpha`/`beta`, and 3 raw pointers with `isize` row/column strides.
Transposition never exists as a flag. A transposed view is just swapped strides. When
`beta == 0`, the contract says `C` is never read, so it may arrive uninitialized.

The `unsafe` boundary is crossed exactly here, justified by the validation that just
ran. `gemm_unchecked` enters one step later, with the caller carrying that
justification instead.

## Stage 2: dispatch early exits

`dispatch::execute` (`gemmkit/src/dispatch.rs`) handles the degenerate algebra while
the element type is still concrete:

```rust
if task.m == 0 || task.n == 0 {
    return;
}
// k == 0 or alpha == 0 => the A*B term vanishes: C <- beta*C only
if task.k == 0 || task.alpha == T::ZERO {
    T::scale_c(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
    return;
}
T::dispatch(task, par, ws);
```

An empty output means nothing to do. A vanished `A*B` term (`k == 0` or `alpha == 0`)
degrades the call to a `C <- beta*C` scale. That scale never reads `A` or `B`. Within
it, `beta == 0` stores zeros without reading `C`, which keeps the uninitialized-C
contract honest.

Only a real product reaches `T::dispatch`, which reads the per-type `OnceLock` slot.
On first use, the selection ladder probes CPU features, honoring a
`GEMMKIT_REQUIRE_ISA` pin, which panics rather than falls back. It then caches the
winning monomorphized entry points plus the tile geometry. Every later call is a
single indirect call. On the AVX-512 machine, `f32` resolves to
`run_typed::<f32, Avx512F, 2, 12>`, the 32x12 tile.

## Stage 3: routing in `run_typed`

`run_typed` (`gemmkit/src/dispatch/float.rs`) is a short gauntlet of gates, each
rerouting a shape the register-tiling driver would serve poorly.

First, gemv. If `m == 1 || n == 1`, and the path is not capped off through
`GEMMKIT_GEMV_THRESHOLD`, the call goes straight to `special/gemv.rs`. This happens
*before* orientation normalization, in the user's original frame. gemv resolves its
own orientation. It treats the `m == 1` case as the transposed `rows x k` problem, and
it partitions output rows itself.

Everything else is orientation-normalized by `orient_transpose`. If `C` is
row-major-ish (`|csc| < |rsc|`), the dispatcher rewrites the problem as its transpose:
`C^T = B^T * A^T`. This swaps `m` with `n`, the `A`/`B` pointers and strides, and
`rsc` with `csc`.

The identity is free. No data moves, only the descriptor changes. It buys a strong
invariant. After this point, the output's *row* stride is the small one (`rsc == 1`
for a fully contiguous C). Each output column then occupies consecutive memory, and
the kernel walks down contiguous columns.

The microkernel's fast store path needs exactly that: `rsc == 1`, so it can use
vector stores of `LANES` consecutive rows in a column. Every layer below optimizes
for one orientation instead of 2. The all-row-major 2048-cube hits this swap. The
engine actually computes `C^T`, and nobody below dispatch knows.

Then 2 more gates apply to the normalized task.

A small-`m,n` shape goes to `special/small_mn.rs`. Both dimensions must be at or below
`small_mn_dim`, with a contraction longer than `small_k_threshold`. There, each output
element is one horizontal SIMD dot. This is zero-copy when both operands stream
unit-stride along `k`. When one operand is strided (`k > small_mn_pack_min_k`), it
goes through a pack tier that copies just the offending operand.

A small-`k` shape, `k <= small_k_threshold` (16 on x86, 8 on aarch64 by default), goes
to `special/small_k.rs`. This computes the whole product as one in-place depth panel
over the microkernel, with no blocking or packing setup.

Whatever passes all the gates, and the 2048-cube does, enters `driver::run`. The
driver states its preconditions: `m, n, k > 0`, `alpha != 0`, and orientation
normalized.

## Stage 4: the driver loop nest

`driver::run` forwards to `run_inner` (`gemmkit/src/driver.rs`) with the zero-cost
`Identity` epilogue. The fused entries land in the same function, with a real
epilogue instead. The driver is generic over the family and the ISA token. For this
call, that is `FloatGemm<f32>` and `Avx512F`, with `mr = MR_REG * LANES = 32` and
`nr = 12`.

Blocking comes first. `cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k)` yields
`(MC, KC, NC)` from the BLIS cache model. These are sized in *packed-input* elements,
`sizeof(Lhs)`, not the accumulator, so narrow types get deeper blocks. The loop nest
then runs in BLIS order:

- **`jc` over `NC`**: column blocks, sized so the packed B macro-panel stays
  L3-resident.
- **`pc` over `KC`**: depth slices. This loop is *never parallel*. All depth slices
  accumulate into the same C tiles. Parallelizing depth would mean a synchronized
  read-modify-write on C, or split reductions. Keeping depth serial lets every output
  element be reduced start to finish by one worker, which is half of the
  reproducibility contract. `beta` participates only on the first slice (`pc == 0`).
  Later slices run with an effective beta of one, and accumulate. For
  mixed-precision families (`OUT_IS_ACC = false`), there is exactly one slice,
  `kc = k`, so the running sum never rounds through the narrow output type.
- **A flat 1-D job list**: inside each depth slice, the remaining work is `n_mc` row
  blocks times `n_nt` column tiles. These flatten into `n_jobs = n_mc * n_nt` indices.
  Workers pull contiguous chunks from a shared, lock-free `JobCursor` on demand. There
  is no static partition, so a faster core absorbs proportionally more work. The
  chunk grain oversamples the worker count (`job_grain`). The packed-LHS path uses a
  row-block-aligned `packed_block_grain` instead, so chunks never straddle a pack
  boundary. The worker count itself comes from `par.resolve(m*n*k, n_jobs)`. This is
  work-based: it scales with the total work `m*n*k` over a per-worker floor, rather
  than jumping straight to all cores. If that count would leave the job list
  shallower than a few chunks per worker, the driver first shrinks `mc`. This only
  cuts more, smaller row blocks, so it cannot move a result bit. Shrinking `mc`
  deepens the list before the cursor hands work out.

Packing is adaptive, and each side decides separately.

B is packed once per depth slice, when `m` clears `rhs_pack_threshold`. The packed
panel is reused across all `n_mc` row blocks, so the copy only pays off when that
reuse is high. Otherwise B is read in place, through its original strides. When
packing does happen, it is itself parallel. Workers pull `nr`-wide column panels from
a cursor. The `for_each_worker` join doubles as the write-before-read barrier. Packed
B is the one buffer all compute workers share, so this barrier matters.

A has 3 modes. Each worker can pack the row block it is working on into its own
private workspace region. This is forced when `rsa != 1`, or when the block is a
partial `mr` multiple. It is chosen otherwise when per-worker column reuse, or a
TLB-hostile column stride, makes it pay off. On large parallel problems, a shared
pre-pass can instead pack each row block exactly once. It packs into a per-block
region behind its own barrier (the `shared_lhs_mnk` gate). This eliminates redundant
per-worker packing. When reuse is too low to pay for any copy, A is read in place.

Sizing for these regions happens up front, through `Workspace::regions`, with the
fail-closed overflow checks noted earlier. On a no-pack route, the workspace is not
even touched.

For each job, the worker resolves its A panel, packed or in-place. It locates the B
panel: per-call packed, prepacked buffer, or in-place. It then calls the microkernel
for every `mr`-row strip of the block. All of this runs inside `simd.vectorize`, so
the entire strip executes in target-feature codegen.

## Stage 5: the microkernel and its store

`Fam::microkernel_epi` (`gemmkit/src/kernel/float.rs`, `microkernel_impl`) computes
one `MR x NR` tile. For this call, that is 32x12 `f32` values, held in 24 ZMM
accumulator registers as a `[[Reg; MR_REG]; NR]` array.

A full-width tile runs `SimdOps::accumulate_tile`, the ascending-`k`
fused-multiply-add schedule. This is the seam a load-bound ISA like NEON overrides,
with a software-pipelined variant that reorders loads, never arithmetic. An edge
column tile instead takes a runtime-bounded loop. That loop reads exactly `nr_eff`
columns, so an unpacked B is never read past its last real column.

Then `alpha` folds into the accumulators. This step is skipped entirely when
`alpha == 1`, thanks to the `AlphaStatus` the driver precomputed.

The store is where `beta` and the epilogue live. It has 2 routes.

The fast path fires for a full tile with unit output row stride:
`mr_eff == mr && nr_eff == NR && rsc == 1`. The orientation normalization from stage 3
is what makes this common. Each accumulator register combines with `C` directly. It
is stored as-is for `beta == 0` (C unread), added for `beta == 1`, or
fused-multiply-added for a general `beta`. The result is written back with vector
stores.

Edge tiles and strided outputs take the general path instead. All accumulators drain
into a stack scratch tile, a `SCRATCH_LEN` array in the worker's frame, with no
allocation. A scalar loop then applies the same beta arithmetic element-wise, through
whatever strides `C` has.

Plain `gemm` threads the `Identity` epilogue through all of this. Every epilogue hook
is gated on `!E::IS_IDENTITY`, an associated `const`. The guards fold away at
monomorphization, so the emitted kernel is byte-for-byte the pre-epilogue code.

A fused call, such as `gemm_fused`, `gemm_map`, or requantize, runs the same engine
with a real epilogue. That epilogue fires only when `last_k` is true, on the final
depth slice, once per output element. That story continues in
[Epilogue Fusion](Epilogue_Fusion.md).

## The short way home

The 2x2x3 example never saw most of this. It entered `execute` with `m, n, k` all
positive and `alpha == 1`. It reached `run_typed`, and failed the gemv gate
(`n != 1`, `m != 1`). It was orientation-swapped, then failed the small-`m,n` gate,
since `k = 3` is not a long contraction. With `k = 3 <= 16`, it took
`special/small_k.rs`. That route is one in-place depth panel over the same
microkernel, with no blocking, no packing, and no workspace traffic.

The 2048-cube took the full driver, with parallel B-packing. At
`Parallelism::Rayon(0)`, its worker count scaled to its total work.

Same entry point, same result contract, 2 very different journeys. The layers below
decide, and the caller never has to. For the deeper mechanics of each stage, see
[Blocking and the Cache Model](Blocking_and_the_Cache_Model.md),
[Packing and Workspaces](Packing_and_Workspaces.md),
[Parallel Execution](Parallel_Execution.md), and
[Special Paths](Special_Paths.md).
