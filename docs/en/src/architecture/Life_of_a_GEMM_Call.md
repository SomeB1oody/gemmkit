# Life of a GEMM Call

The previous page described the stack at rest; this one follows a single call through it. The specimen is the quick-start example from the crate docs:

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

This toy shape will take one of the early exits below, so the walk keeps two problems in mind: the 2x2x3 above, and a 2048x2048x2048 `f32` product on an AVX-512 machine that goes all the way down. The route, compressed:

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

`gemm` itself is one line: it borrows the thread-local workspace and forwards to `gemm_with` (`gemmkit/src/api.rs`). `gemm_with` runs `validate_gemm_views` — the full panic catalog from [Design Goals](Design_Goals_and_the_Big_Picture.md): shapes agree, every view stays inside its slice, `C` addresses each `(i, j)` uniquely, `C` overlaps neither input. Then the views dissolve. Everything below this point speaks `Task<T>`: a `Copy` struct of `m, k, n`, `alpha`/`beta`, and three raw pointers with `isize` row/column strides. Transposition never exists as a flag — a transposed view is just swapped strides — and when `beta == 0` the contract says `C` is never read, so it may arrive uninitialized. The `unsafe` boundary is crossed exactly here, justified by the validation that just ran; `gemm_unchecked` enters one step later with the caller carrying that justification instead.

## Stage 2: dispatch early exits

`dispatch::execute` (`gemmkit/src/dispatch.rs`) handles the degenerate algebra while the element type is still concrete:

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

An empty output means nothing to do. A vanished `A*B` term (`k == 0` or `alpha == 0`) degrades the call to a `C <- beta*C` scale that never reads `A` or `B` — and within it, `beta == 0` stores zeros without reading `C`, keeping the uninitialized-C contract honest. Only a real product reaches `T::dispatch`, which reads the per-type `OnceLock` slot: on first use the selection ladder probes CPU features (honoring a `GEMMKIT_REQUIRE_ISA` pin, which panics rather than falls back) and caches the winning monomorphized entry points plus tile geometry; every later call is a single indirect call. On the AVX-512 machine, `f32` resolves to `run_typed::<f32, Avx512F, 2, 12>` — the 32x12 tile.

## Stage 3: routing in `run_typed`

`run_typed` (`gemmkit/src/dispatch/float.rs`) is a short gauntlet of gates, each rerouting a shape the register-tiling driver would serve poorly.

First, gemv: if `m == 1 || n == 1` (and the path is not capped off via `GEMMKIT_GEMV_THRESHOLD`), the call goes to `special/gemv.rs` immediately — notably *before* orientation normalization, in the user's original frame, because gemv resolves its own orientation by viewing the `m == 1` case as the transposed `rows x k` problem and partitions output rows itself.

Everything else is orientation-normalized by `orient_transpose`: if `C` is row-major-ish (`|csc| < |rsc|`), the dispatcher rewrites the problem as its transpose — `C^T = B^T * A^T`, swapping `m` with `n`, the `A`/`B` pointers and strides, and `rsc` with `csc`. The identity is free (no data moves, only the descriptor changes), and it buys a strong invariant: after this point the output's *row* stride is the small one (`rsc == 1` for a fully contiguous C), so each output column occupies consecutive memory and the kernel walks down contiguous columns. The microkernel's fast store path requires exactly that (`rsc == 1` — vector stores of `LANES` consecutive rows of a column), and every layer below optimizes for one orientation instead of two. Our all-row-major 2048-cube hits this swap: the engine actually computes `C^T` and nobody below dispatch knows.

Then two more gates on the normalized task. A small-`m,n` shape (both dimensions at or below `small_mn_dim`, contraction longer than `small_k_threshold`) goes to `special/small_mn.rs`, where each output element is one horizontal SIMD dot — either zero-copy when both operands stream unit-stride along `k`, or through a pack tier that copies just the offending operand when one is strided (`k > small_mn_pack_min_k`). A small-`k` shape (`k <= small_k_threshold`, 16 on x86, 8 on aarch64 by default) goes to `special/small_k.rs`, which computes the whole product as one in-place depth panel over the microkernel with no blocking or packing setup. Whatever passes all gates — our 2048-cube does — enters `driver::run` with the preconditions the driver states: `m, n, k > 0`, `alpha != 0`, orientation normalized.

## Stage 4: the driver loop nest

`driver::run` forwards to `run_inner` (`gemmkit/src/driver.rs`) with the zero-cost `Identity` epilogue — the fused entries land in the same function with a real one. The driver is generic over the family and ISA token; for our call that is `FloatGemm<f32>` and `Avx512F`, `mr = MR_REG * LANES = 32`, `nr = 12`.

Blocking comes first: `cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k)` yields `(MC, KC, NC)` from the BLIS cache model, sized in *packed-input* elements (`sizeof(Lhs)`, not the accumulator — narrow types get deeper blocks). The loop nest then runs in BLIS order:

- **`jc` over `NC`** — column blocks, sized so the packed B macro-panel stays L3-resident.
- **`pc` over `KC`** — depth slices, and this loop is *never parallel*. All depth slices accumulate into the same C tiles, so parallelizing depth would mean synchronized read-modify-write on C or split reductions; keeping depth serial is what lets every output element be reduced start-to-finish by one worker, which is half of the reproducibility contract. `beta` participates only on the first slice (`pc == 0`); later slices run with an effective beta of one and accumulate. For mixed-precision families (`OUT_IS_ACC = false`) there is exactly one slice — `kc = k` — so the running sum never rounds through the narrow output type.
- **A flat 1-D job list** — inside each depth slice, the remaining work is `n_mc` row blocks times `n_nt` column tiles, flattened into `n_jobs = n_mc * n_nt` indices. Workers pull contiguous chunks from a shared lock-free `JobCursor` on demand: no static partition, so faster cores absorb proportionally more, and the chunk grain oversamples the worker count (`job_grain`; the packed-LHS path uses a row-block-aligned `packed_block_grain` so chunks never straddle a pack boundary). The worker count itself came from `par.resolve(m*n*k, n_jobs)` — work-based, scaling with the total flops (`m*n*k` over a per-worker floor) rather than jumping to all cores. If that count would leave the job list shallower than a few chunks per worker, the driver first shrinks `mc` — which only cuts more, smaller row blocks and so cannot move a result bit — to deepen the list before the cursor hands it out.

Packing is adaptive, decided per side. **B** is packed once per depth slice when `m` clears `rhs_pack_threshold` — the packed panel is reused across all `n_mc` row blocks, so the copy amortizes only when that reuse is high; otherwise B is read in place through its original strides. When packing does happen it is itself parallel: workers pull `nr`-wide column panels from a cursor, and the `for_each_worker` join doubles as the write-before-read barrier, because packed B is the one buffer all compute workers share. **A** has three modes: each worker packs the row block it is working on into its private workspace region (forced when `rsa != 1` or the block is a partial `mr` multiple, chosen otherwise when per-worker column reuse or a TLB-hostile column stride makes it pay); on large parallel problems a shared pre-pass packs each row block exactly once into a per-block region behind its own barrier (`shared_lhs_mnk` gate), eliminating redundant per-worker packing; and when reuse is too low to amortize any copy, A is read in place. Sizing for these regions happens up front through `Workspace::regions`, with the fail-closed overflow checks noted earlier, and on a no-pack route the workspace is not even touched.

For each job, the worker resolves its A panel (packed or in-place), locates the B panel (per-call packed, prepacked buffer, or in-place), and calls the microkernel for every `mr`-row strip of the block — all inside `simd.vectorize`, so the entire strip executes in target-feature codegen.

## Stage 5: the microkernel and its store

`Fam::microkernel_epi` (`gemmkit/src/kernel/float.rs`, `microkernel_impl`) computes one `MR x NR` tile — for our call, 32x12 `f32` values held in 24 ZMM accumulator registers as a `[[Reg; MR_REG]; NR]` array. A full-width tile runs `SimdOps::accumulate_tile`, the ascending-`k` fused-multiply-add schedule (the seam a load-bound ISA like NEON overrides with a software-pipelined variant that reorders loads, never arithmetic); an edge column tile takes a runtime-bounded loop that reads exactly `nr_eff` columns so an unpacked B is never read past its last real column. Then `alpha` folds into the accumulators — skipped entirely when `alpha == 1`, thanks to the `AlphaStatus` the driver precomputed.

The store is where `beta` and the epilogue live, and it has two routes. The fast path fires for a full tile with unit output row stride (`mr_eff == mr && nr_eff == NR && rsc == 1` — the orientation normalization from stage 3 is what makes this common): each accumulator register is combined with `C` directly — stored as-is for `beta == 0` (C unread), added for `beta == 1`, or fused-multiply-added for general `beta` — and written back with vector stores. Edge tiles and strided outputs take the general path: all accumulators drain into a stack scratch tile (a `SCRATCH_LEN` array in the worker's frame, no allocation), then a scalar loop applies the same beta arithmetic element-wise through whatever strides `C` has.

Plain `gemm` threads the `Identity` epilogue through all of this, and every epilogue hook is gated on `!E::IS_IDENTITY` — an associated `const`, so the guards fold away at monomorphization and the emitted kernel is byte-for-byte the pre-epilogue code. A fused call (`gemm_fused`, `gemm_map`, requantize) runs the same engine with a real epilogue that fires only when `last_k` is true — on the final depth slice, once per output element. That story continues in [Epilogue Fusion](Epilogue_Fusion.md).

## The short way home

Our 2x2x3 example never saw most of this: it entered `execute` with `m, n, k` all positive and `alpha == 1`, reached `run_typed`, failed the gemv gate (`n != 1`, `m != 1`), was orientation-swapped, failed the small-`m,n` gate (`k = 3` is not a long contraction), and with `k = 3 <= 16` took `special/small_k.rs`: one in-place depth panel over the same microkernel, no blocking, no packing, no workspace traffic. The 2048-cube took the full driver with parallel B-packing and, at `Parallelism::Rayon(0)`, a worker count scaled to its total work. Same entry point, same result contract, two very different journeys — the layers below decide, and the caller never has to. For the deeper mechanics of each stage, see [Blocking and the Cache Model](Blocking_and_the_Cache_Model.md), [Packing and Workspaces](Packing_and_Workspaces.md), [Parallel Execution](Parallel_Execution.md), and [Special Paths](Special_Paths.md).
