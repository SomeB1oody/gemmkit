# Parallelism in Practice

Every GEMM entry takes a `Parallelism` argument as its last parameter. It is a small enum with three practical modes, and the difference between using it well and using it badly is mostly about understanding what the `auto` mode decides on your behalf and when to take the wheel yourself.

## The three modes

```rust
pub enum Parallelism {
    Serial,       // single-threaded
    Rayon(usize), // rayon with at most n threads; Rayon(0) auto-detects
}
```

`Serial` runs the whole call on the calling thread. `Rayon(n)` asks for at most `n` workers. `Rayon(0)` is auto, and it is also the `Default`, so `Parallelism::default()` gives you auto. The `n` in `Rayon(n)` is a ceiling on partitions, not a promise to use them all - a problem with less work than `n` chunks, or fewer cores than `n`, gets fewer.

## What auto actually does

Auto is not "use all cores." It is two decisions layered on the problem size.

First, a **workload gate**. Below a total-work threshold on `m*n*k` (the `GEMMKIT_PARALLEL_THRESHOLD` knob, default `48*48*256`), the call stays serial no matter what - the fork/join overhead would swamp any gain on a small matrix. This gate runs before everything else, so it applies even to an explicit `Rayon(n)`: below it, `Rayon(8)` still runs on one thread.

Above the gate, auto **scales the worker count with the total work** rather than jumping to the full core count. Concretely it targets `m*n*k` divided by `GEMMKIT_PAR_MNK_PER_WORKER` (default `2_000_000` - one worker per that much work), then caps that by the machine's core count and by the number of available job chunks, floored at one. The count is work-based, not dimension-based, because the measured optimum tracks total flops rather than linear size: on a Ryzen 9950X, `128^3` runs fastest serial while `384^3` already wants every hardware thread - a spread no single stride on the linear dimension can fit. So a small product uses a handful of workers and a large one uses many, and setting `GEMMKIT_PAR_MNK_PER_WORKER` to `0` (which behaves as `1`) forces full width for anything above the serial gate.

## Explicit counts

`Rayon(n)` with `n > 0` bypasses the ramp heuristic and asks for exactly `n` partitions - but still capped, for safety, by the machine's core count (`available_parallelism`) and by the number of job chunks the problem actually splits into. So `Rayon(1000)` on a 16-core box computing a small product will not oversubscribe; it collapses to what the machine and the work can absorb. This exactness is why the test suite and the scaling diagnostics use explicit counts: `Rayon(4)` gives you four-way partitioning (when there is that much work and that many cores), not a heuristic guess. Use an explicit count when you have measured your own workload and know the sweet spot, or when you want reproducible partitioning across runs for benchmarking.

## How gemmkit uses the rayon pool

On native targets gemmkit does not build or own a rayon thread pool. When it decides to run `k` partitions, it drives them through rayon's parallel iterator on the *ambient* pool - rayon's global pool by default. That has a useful consequence: if you wrap a call in your own pool's `install`, the GEMM's workers run on your pool.

```rust
let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
pool.install(|| {
    gemm(1.0, a, b, 0.0, c, Parallelism::Rayon(0)); // runs on `pool`
});
```

The worker *count* gemmkit chooses is still bounded by `available_parallelism` (the whole machine), and the partitions are distributed over whatever threads the current pool has by rayon's work-stealing scheduler - so a smaller custom pool simply runs the same partitions on fewer threads. Work distribution inside a call is demand-driven: workers pull contiguous chunks from a shared lock-free cursor, so faster cores on a heterogeneous part (P/E core layouts) absorb proportionally more instead of everyone waiting on the slowest. The threaded-wasm story is different - there gemmkit *does* size a dedicated pool - and is covered in [no_std and WebAssembly](no_std_and_WebAssembly.md).

Beneath that ambient-pool story sits a second mechanism, native and x86_64-only: gemmkit also keeps up to `GEMMKIT_POOL_CLASSES` (default 2) private, persistent pools sized to exact halving tiers of the machine width - on a 32-thread part, 16 and 8 threads - built lazily on first use and never rebuilt afterward. Auto snaps its worker count exactly onto one of these tiers instead of forking the full-width global pool, because a fork's overhead tracks the pool's idle slack (threads it owns beyond the ones actually engaged), and a small GEMM drowns in that slack on a full-width pool. None of this changes what you already know above: it never fires inside your own pool - an `install`'d call is still fully respected and never redirected to a tier pool - and an explicit `Rayon(n)` still gets exactly `n` workers, just routed into whichever tier pool is the smallest fit. What does change is idle memory: by default an x86_64 process now parks about 24 extra threads (the 16- and 8-wide tier pools) alongside the global pool, all asleep until a small GEMM needs them. Set `GEMMKIT_POOL_CLASSES=0` to disable tier pools entirely and fall back to the ambient pool for every call.

## The reproducibility promise, precisely

For a **fixed input, environment, and configuration, the output is identical regardless of the worker count.** That is the contract. It holds because `kc`, `nc`, and the fixed depth-panel order - the only things that shape each output element's summation - are computed independently of how many threads will run them, and every output element is reduced start-to-finish by a single worker over the full contraction depth, so there are no split reductions whose order would depend on the schedule. (The flat job list itself is *not* strictly identical across widths - a wide worker count can shrink `mc` to keep the list deep enough - but `mc` stays an `mr` multiple, so the set of microtiles and their numerics are unchanged.) The packed bytes do not depend on who packs them. Which worker computes a given tile varies from run to run; the numerical result does not.

What is **not** promised is bitwise identity between `Serial` and `Rayon(n)`. It happens to hold today on the driver paths - serial and parallel run the same kernel - but the guarantee you should build on is reproducibility under a fixed config, not serial-versus-parallel bit-equality. If you need cross-machine or cross-config bit-equality you will not get it here; floating-point GEMM is order-sensitive and the config (ISA, blocking, thread cap) is part of the fixed input. Integer `gemm_i8` is the exception: it is bit-identical across ISAs and worker counts because `i32` addition is order-independent.

## When Serial is the right call

Reach for `Serial` in three situations. First, **small problems**: below the workload gate auto is serial anyway, but passing `Serial` explicitly also skips the `available_parallelism` probe and the fork machinery entirely, which is measurably cheaper in a tight loop of tiny GEMMs. Second, **when you own the outer parallelism**: if you are already running many independent GEMMs across a rayon pool, or parallelizing a batch loop yourself, letting each inner call also fan out oversubscribes the machine and usually regresses - run the inner calls `Serial` and keep the parallelism at the outer level. (For a batch of products, prefer the built-in [Batched GEMM](Batched_GEMM.md) entries, which schedule the batch as a whole.) Third, **determinism-sensitive debugging**, where you want the single-threaded path to rule out a scheduling variable.

## Bandwidth-bound shapes get their own policy

A matrix-vector product (`m == 1` or `n == 1`) and other memory-bound shapes are not compute-bound, so the work-based worker count is the wrong model for them. Those routes use a separate rule: they stay serial below an LLC-derived byte floor - one core already gets the full last-level-cache bandwidth, so splitting only adds contention - and above it they step straight to a bandwidth cap (a fraction of the logical core count, since DRAM saturates well before the last core). A *few* workers is the worst point on a bandwidth scaling curve, so the policy jumps over it rather than ramping through. This is automatic; the byte floor and the cap are tunable (`GEMMKIT_GEMV_PARALLEL_BYTES`, `GEMMKIT_GEMV_THREAD_CAP`), and the full treatment is in [Small Shapes and GEMV](Small_Shapes_and_GEMV.md).

## Where to next

- [Small Shapes and GEMV](Small_Shapes_and_GEMV.md) - the bandwidth-bound policy in detail.
- [Batched GEMM](Batched_GEMM.md) - scheduling many products as one batch instead of nesting parallelism.
- [Tuning Knobs](Tuning_Knobs.md) - the `GEMMKIT_*` thresholds behind these decisions.
- [Parallel Execution](../architecture/Parallel_Execution.md) - the job cursor and worker-count resolution internals.
