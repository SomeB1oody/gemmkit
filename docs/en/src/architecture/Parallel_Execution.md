# Parallel Execution

Parallelism in gemmkit lives in one small layer, `gemmkit/src/parallel.rs` (layer L2), with 2 jobs. It decides *how many* workers a problem deserves, and hands *which* work to each of them.

Both decisions are deliberately conservative, because more threads are not free. The layer's design starts from one observation: the wrong worker count loses more performance than the wrong schedule does. Both decisions are also shaped so that the numerical result never depends on either one.

The user-facing surface is a single enum. It is `Parallelism::Serial`, or `Parallelism::Rayon(n)`. `Rayon(0)`, the default, means auto.

## Workload-aware worker resolution

`Parallelism::resolve` turns the request into an actual partition count. The request is only one input. The workload is the other.

First comes the total-work serial gate. Below `GEMMKIT_PARALLEL_THRESHOLD` (default `48*48*256` for `m*n*k`), everything stays serial, before the resolver even samples the core count. Forking rayon for a product that takes only microseconds would cost more than the product itself. The gate precedes the request check, so even an explicit `Rayon(n)` stays serial below it.

Above the gate, the resolver honors an explicit count, capped by the core count and by the number of available jobs. So `Rayon(huge)` can neither over-subscribe the machine nor over-allocate per-worker pack regions. Only the auto path is heuristic. This keeps forced widths exact for tests and for scaling diagnostics.

The auto path is work-based. It divides the total work `m*n*k` by `GEMMKIT_PAR_MNK_PER_WORKER` (default `2_000_000`, the per-worker floor below which fork/join overhead outweighs the gain). This division gives the worker count, floored at 1 and capped by the core count and the job count.

The path is work-based rather than dimension-based, because the optimal worker count tracks total flops, not linear size. A small cube runs fastest serial. A mid-size cube wants only a few workers. A large cube wants every hardware thread the machine has. No single stride on a linear dimension can fit that spread.

Scaling to full width at mid sizes depends on one more thing: avoiding redundant per-worker packing of the same `A` panel. [Packing and Workspaces](Packing_and_Workspaces.md) covers the LHS in-place gate that keeps that redundancy from happening. The `GEMMKIT_PAR_MNK_PER_WORKER` knob is the escape hatch for a machine whose per-worker floor differs from the compiled default.

Bandwidth-bound shapes get a different rule entirely. A gemv or gevv does O(1) arithmetic per byte, so the compute ramp's logic does not transfer. `resolve_bandwidth` gates on *bytes touched* instead.

Below a cache-derived byte floor, the matrix fits one core's private cache, and that core saturates it alone. Splitting then only adds fork/join and shared-cache contention, with no bandwidth to gain. `gemv_parallel_floor_bytes` (in `cache.rs`) derives this floor from the topology. On parts with an L3, the floor is the per-core private L2. On parts without one, it is a fraction of the full shared cluster L2. `GEMMKIT_GEMV_PARALLEL_BYTES` overrides the floor directly.

Above the floor, the matrix spills to the shared L3, whose bandwidth a single core cannot saturate on its own. So the auto count steps *straight* to a wider width for those bytes, instead of ramping up to it. That width climbs a ladder built from the exact-fit pool tiers described below. The smallest tier sits at the floor, and the count climbs one tier for each `GEMMKIT_GEMV_TIER_STEP` (default auto, 8) factor of touched bytes. The ladder stops at the largest tier rather than the full machine width. A gemv saturates its bandwidth well before the machine runs out of cores, and more workers past that point can cost more than they gain.

gemmkit builds the rungs from the pool tiers themselves, not from a separate set of fractions. So an auto gemv width always has an exact-fit pool waiting for it, and it never pays the slack tax those tiers exist to remove. `GEMMKIT_GEMV_THREAD_CAP` replaces the whole ladder with one flat width, for a deployment that wants to fix it.

There is no ramp between rungs, because a bandwidth-bound scaling curve dips at a handful of workers. Fork/join and contention costs are already paid there, and aggregate bandwidth has not yet arrived. Any ramp through that dip would lose to both of its endpoints. So the rule stays simple: serial below the floor, a tier width above it, and nothing in between.

Batched GEMM has its own resolver, `resolve_batch`, which chooses between 3 plans. `Serial` runs every element in turn on the calling thread. `BatchParallel(n)` gives each worker whole, cache-hot GEMMs to run, paying one fork/join for the whole batch. Because no element ever splits, this plan stays bit-identical across worker counts. `SequentialInternal` instead loops the batch on the calling thread, giving each large, DRAM-bound element the full engine parallelism in turn.

`resolve_batch` gates the `SequentialInternal` split to `m, n > 1` shapes, whose routes are worker-count independent. A gemv-shaped element always stays whole on one worker instead. [Special Paths](Special_Paths.md) covers the routing, and [Batched GEMM](../gemmkit-guide/Batched_GEMM.md) covers the API.

## Demand-driven work distribution

Given a worker count, the driver does not build a nested task tree. For each column block and depth slice, it flattens the inner work into a flat 1-D job list: `n_mc` row blocks times `n_nt` column tiles. Job `q` decodes to `(ic_idx, jt) = (q / n_nt, q % n_nt)`. Workers pull contiguous chunks from a shared, lock-free cursor until it is empty:

```rust
// gemmkit/src/parallel.rs
impl JobCursor {
    /// Atomically claim the next `[start, end)` chunk, or `None` once the job space
    /// is exhausted
    #[inline]
    pub(crate) fn next_chunk(&self) -> Option<(usize, usize)> {
        let start = self.next.fetch_add(self.grain, Ordering::Relaxed);
        if start >= self.n_jobs {
            None
        } else {
            Some((start, (start + self.grain).min(self.n_jobs)))
        }
    }
}
```

Each claim costs one `fetch_add`. There are no locks and no per-job queues. Demand-driven pulling is what makes heterogeneous cores work well. On a big.LITTLE part, a P-core that finishes chunks faster simply pulls proportionally more of them. A static `n_jobs / n_threads` split would instead leave every core waiting on the slowest one. The same property also absorbs OS noise and frequency differences on homogeneous machines.

The chunk grain balances 2 costs. Too coarse a grain leaves the tail of the job list idling workers at the join. Too fine a grain lets the atomic claims start to show, along with, on the packed-LHS path, re-packing at chunk edges.

The general grain oversamples the worker count. `job_grain` targets `GEMMKIT_PARALLEL_OVERSAMPLE` chunks per worker (default 8), so each worker expects several pulls, and imbalance self-corrects.

The packed-LHS path is special. Its natural chunk is a whole row block (`n_nt` consecutive jobs). A worker packs the block's `A` panel once, and reuses it across all the block's column tiles. That yields only `n_mc` chunks. So when the row-block count is small, `packed_block_grain` splits each block into power-of-two column sub-chunks. It splits until there are about `GEMMKIT_PACKED_OVERSAMPLE * n_threads` chunks (default target 2), and only by divisors of `n_nt`. So a chunk never straddles a row-block boundary and re-packs `A` mid-chunk. Splitting harder than this target re-packs too often and makes performance worse.

2 parallel phases run before the compute region in each depth slice, and their boundaries are the only barriers in the driver.

When `B` packs, workers pull `nr`-wide column panels from their own cursor. The fork/join of that region is the write-before-read barrier the compute region depends on, because packed `B` is the one buffer shared non-disjointly across workers.

The shared-`A` pre-pass packs each row block's panel once into a shared slot, under the same discipline. It opens above a size gate, and it also opens from 16 workers up regardless of size. At that width, every extra worker is another redundant copy of each panel it touches, so deduplicating pays off even on mid-size problems.

Everything else is disjoint by construction. Workers write only their own output tiles and their own pack regions. That invariant is what lets the `Ptr` shim declare the captured raw pointers `Send + Sync`, in one audited place.

## Size-class thread pools

Rayon's fork/join cost does not scale with the problem. It scales with the pool's *idle slack*: the gap between the threads a pool owns and the workers a given call actually engages. Forking `k` workers into a `w`-wide pool wakes `w` threads, not `k`. The `w - k` threads that get no work still pay their share of the barrier and the OS-level wake/park round trip.

The full-width global pool is the worst case for a small parallel GEMM. A mid-size product often wants only a fraction of the machine's threads. Forking it into the full-width global pool then wastes most of that width as pure slack tax on every call.

gemmkit's answer is a small set of private, persistent pools, each sized to exactly one of the worker counts the auto path actually asks for. Below full machine width, it keeps up to `GEMMKIT_POOL_CLASSES` halving tiers (default 2 on x86_64, 1 on aarch64, clamped to 3, 0 elsewhere). The tiers are a half-width pool, a quarter-width pool, and so on, each halving the machine's physical width again. Each tier pool builds lazily, on first use, at a small one-time cost, and it never rebuilds afterward. The tiers are a fixed halving of the machine width, not a value tuned per shape.

The auto path snaps its worker count exactly onto a tier, so it carries zero slack by construction. It stays on its largest tier until the total work `m*n*k` clears `GEMMKIT_FULL_WIDTH_MNK` (default auto, arch-split: 110_000_000 on x86_64, 14_000_000 on aarch64). Past that point, the extra full-width workers finally pay for the fork/join they add, and full machine width takes over.

3 rules keep this mechanism from ever fighting the caller's own scheduling.

1. A call already running inside a rayon pool, whether the caller's own `install` or a nested gemmkit call, never redirects to a tier pool. The ambient pool always wins, exactly as it did before tier pools existed.
2. An explicit `Rayon(n)` keeps its exact semantics of precisely `n` workers. It only picks the smallest tier pool that fits `n`, instead of forking into the global pool. The worker count stays unaffected. Only the pool it forks into changes.
3. Threaded wasm keeps its own dedicated pool, described below, untouched by any of this. Tier pools are a native, non-wasm concern.

## The wasm story

On `wasm32-wasip1` there are no threads to spawn, and rayon would trap if it tried. The compile-time constant `RAYON_USABLE` captures whether the target can run workers at all. On a wasm build without the threading opt-in, every resolver returns 1, and `for_each_worker` runs the plain serial loop. `parallel` degrades gracefully instead of trapping.

The opt-in is the `wasm_threads` feature, which targets `wasm32-wasip1-threads` or a browser with `SharedArrayBuffer`. Because `available_parallelism` is unsupported on wasm, rayon's global pool would otherwise silently size itself to 1 thread. So gemmkit builds its own pool instead, sized by the `GEMMKIT_WASM_THREADS` knob (default 8), and installs it around the worker loop. The deployer states the width, and everything else stays unchanged. [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md) has the details of the wasm builds.

## The reproducibility contract, assembled

gemmkit's reproducibility contract is simple. For a fixed machine and a fixed configuration, the engine produces reproducible output. This is not a promise of bitwise-identical results across different configurations, and the worker count counts as part of that configuration.

Today, gemmkit also holds a stronger property, as an engineering fact rather than a separate promise. Changing only the worker count, with everything else held fixed, still produces bitwise-identical output. 3 mechanisms make that true. Here is how they fit together.

First, the numerics do not depend on the worker count. [Blocking](Blocking_and_the_Cache_Model.md) derives `kc` and `nc` from the cache model alone, never from the thread count. The depth slices always run in the same fixed `pc` order. So every output element's floating-point reduction takes the same shape, whether one worker drains the cursor or many do. The one dimension a worker count can move is `mc`. A wide count shrinks it, through the driver's parallel job-depth floor, so the flat job list stays several chunks deep per worker. The job list itself is therefore not always identical, chunk for chunk, across widths. But `mc` always stays an `mr` multiple. So the microtile set (every `mr`-aligned row offset plus the one `m`-tail tile) is the same under any split. With `kc` and the `pc` order untouched, no result bit moves. The worker count changes how work is *grouped and partitioned*. It never changes what each tile computes.

Second, no reduction ever splits across workers. Within a depth slice, one worker computes an output tile's whole update, so a chunk is always a set of whole tiles. The depth slices themselves run sequentially, since the `pc` loop is never parallel. `beta` applies on the first slice, and later slices accumulate. So every output element's floating-point reduction runs in one fixed order, determined by the blocking alone. The special paths keep the same discipline. gemv partitions output *rows* on register-panel boundaries, so each row's SIMD/scalar split does not depend on the partition. That makes gemv bit-identical across worker counts outright. Batched plans either keep whole elements on one worker, or split only shapes whose routes are already worker-count independent.

Third, packed bytes do not depend on who packs them. `pack_panels` is a pure reorder, and both of its branches write identical bytes. A panel packed by 1 worker, by the shared-`A` pre-pass, or by `prepack_rhs` earlier, is therefore the same sequence of bytes. See [Packing and Workspaces](Packing_and_Workspaces.md). The kernel cannot observe who staged its input.

Which worker computes a given tile genuinely varies from run to run, since the cursor hands chunks to whoever asks first. By the 3 mechanisms above, though, nothing numerical depends on that choice. This bitwise agreement across worker counts is a property of gemmkit's design today, not a wider guarantee. It leaves room for a tolerance-held kernel, such as the bf16 dot path, to reshape its accumulation without breaking the actual contract. [Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md) covers how to choose worker counts in practice, and what the contract means for testing.
