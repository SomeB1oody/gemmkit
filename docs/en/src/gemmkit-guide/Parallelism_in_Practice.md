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

Above the gate, auto **ramps the worker count with the problem's linear size** rather than jumping to the full core count. Concretely it targets `cbrt(m*n*k)` - roughly `n` for a square problem - divided by a stride, then caps that by the machine's core count and by the number of available job chunks. The stride is derived from the core count (a small machine ramps faster, a large one slower) and is itself tunable via `GEMMKIT_THREAD_DIM_STRIDE`. The effect is that a `256^3` problem uses a handful of workers and a `4096^3` problem uses many, with the count climbing smoothly in between. This is deliberate: spinning up all cores for a mid-size matrix loses to memory and scheduling contention, and the ramp was fit to measured scaling curves rather than assumed.

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

## The reproducibility promise, precisely

For a **fixed input, environment, and configuration, the output is identical regardless of the worker count.** That is the contract. It holds because the blocking geometry and the job list are computed independently of how many threads will run them, and every output element is reduced start-to-finish by a single worker over the full contraction depth - there are no split reductions whose order would depend on the schedule. The packed bytes do not depend on who packs them. Which worker computes a given tile varies from run to run; the numerical result does not.

What is **not** promised is bitwise identity between `Serial` and `Rayon(n)`. It happens to hold today on the driver paths - serial and parallel run the same kernel - but the guarantee you should build on is reproducibility under a fixed config, not serial-versus-parallel bit-equality. If you need cross-machine or cross-config bit-equality you will not get it here; floating-point GEMM is order-sensitive and the config (ISA, blocking, thread cap) is part of the fixed input. Integer `gemm_i8` is the exception: it is bit-identical across ISAs and worker counts because `i32` addition is order-independent.

## When Serial is the right call

Reach for `Serial` in three situations. First, **small problems**: below the workload gate auto is serial anyway, but passing `Serial` explicitly also skips the `available_parallelism` probe and the fork machinery entirely, which is measurably cheaper in a tight loop of tiny GEMMs. Second, **when you own the outer parallelism**: if you are already running many independent GEMMs across a rayon pool, or parallelizing a batch loop yourself, letting each inner call also fan out oversubscribes the machine and usually regresses - run the inner calls `Serial` and keep the parallelism at the outer level. (For a batch of products, prefer the built-in [Batched GEMM](Batched_GEMM.md) entries, which schedule the batch as a whole.) Third, **determinism-sensitive debugging**, where you want the single-threaded path to rule out a scheduling variable.

## Bandwidth-bound shapes get their own policy

A matrix-vector product (`m == 1` or `n == 1`) and other memory-bound shapes are not compute-bound, so the `cbrt(m*n*k)` ramp is the wrong model for them. Those routes use a separate rule: they stay serial below an LLC-derived byte floor - one core already gets the full last-level-cache bandwidth, so splitting only adds contention - and above it they step straight to a bandwidth cap (a fraction of the logical core count, since DRAM saturates well before the last core). A *few* workers is the worst point on a bandwidth scaling curve, so the policy jumps over it rather than ramping through. This is automatic; the byte floor and the cap are tunable (`GEMMKIT_GEMV_PARALLEL_BYTES`, `GEMMKIT_GEMV_THREAD_CAP`), and the full treatment is in [Small Shapes and GEMV](Small_Shapes_and_GEMV.md).

## Where to next

- [Small Shapes and GEMV](Small_Shapes_and_GEMV.md) - the bandwidth-bound policy in detail.
- [Batched GEMM](Batched_GEMM.md) - scheduling many products as one batch instead of nesting parallelism.
- [Tuning Knobs](Tuning_Knobs.md) - the `GEMMKIT_*` thresholds behind these decisions.
- [Parallel Execution](../architecture/Parallel_Execution.md) - the job cursor and worker-count resolution internals.
