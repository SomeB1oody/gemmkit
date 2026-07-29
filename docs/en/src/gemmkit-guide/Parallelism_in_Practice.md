# Parallelism in Practice

Every GEMM entry takes a `Parallelism` argument as its last parameter. It is a small
enum with 3 practical modes. Using it well comes down to 2 things: understand
what the `auto` mode decides for you, and know when to take over yourself.

## The 3 modes

```rust
pub enum Parallelism {
    Serial,       // single-threaded
    Rayon(usize), // rayon with at most n threads; Rayon(0) auto-detects
}
```

`Serial` runs the whole call on the calling thread. `Rayon(n)` asks for at most `n`
workers. `Rayon(0)` is auto, and it is also the `Default`, so `Parallelism::default()`
gives you auto. The `n` in `Rayon(n)` is a ceiling on partitions, not a promise to use
them all. A problem with less work than `n` chunks, or fewer cores than `n`, gets fewer.

## What auto actually does

Auto does not mean "use all cores." It makes 2 decisions based on the problem size.

First, a **workload gate** applies. Below a total-work threshold on `m*n*k` (the
`GEMMKIT_PARALLEL_THRESHOLD` knob, default `48*48*256`), the call stays serial no
matter what. On a matrix that small, fork/join overhead would swamp any gain. This
gate runs before everything else, so it applies even to an explicit `Rayon(n)`.
Below the gate, `Rayon(8)` still runs on one thread.

Above the gate, auto scales the worker count with the total work instead of jumping
straight to the full core count. It targets `m*n*k` divided by
`GEMMKIT_PAR_MNK_PER_WORKER` (default `2_000_000`, one worker per that much work).
It then caps the result by the machine's core count and by the number of available
job chunks, floored at one. The count is work-based, not dimension-based, because the
best worker count tracks total flops, not linear size. No single stride on a linear
dimension can fit that whole range. A small product uses a handful of workers, and a
large one uses many. Setting `GEMMKIT_PAR_MNK_PER_WORKER` to `0` (which behaves as
`1`) forces full width for anything above the serial gate.

## Explicit counts

`Rayon(n)` with `n > 0` bypasses the ramp heuristic and asks for exactly `n`
partitions. For safety, this is still capped by the machine's core count
(`available_parallelism`) and by the number of job chunks the problem actually splits
into. So `Rayon(1000)` on a 16-core box computing a small product does not
oversubscribe: it collapses to what the machine and the work can absorb. This
exactness is why the test suite and the scaling diagnostics use explicit counts.
`Rayon(4)` gives you four-way partitioning, when there is that much work and that many
cores, not a heuristic guess. Use an explicit count once you have measured your own
workload and know the sweet spot. Also use it when you want reproducible
partitioning across runs for benchmarking.

## How gemmkit uses the rayon pool

gemmkit does not require you to hand it a rayon pool. If you wrap a call in your own
pool's `install`, the GEMM's workers run on that pool instead of anywhere else.

```rust
let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
pool.install(|| {
    gemm(1.0, a, b, 0.0, c, Parallelism::Rayon(0)); // runs on `pool`
});
```

The worker *count* gemmkit chooses is still bounded by `available_parallelism`, the
whole machine. Rayon's work-stealing scheduler distributes the partitions over
whatever threads the current pool has. A smaller custom pool simply runs the same
partitions on fewer threads. Work distribution inside a call is demand-driven:
workers pull contiguous chunks from a shared lock-free cursor. On a heterogeneous
part (a mix of P-cores and E-cores), a faster core absorbs more chunks instead of
everyone waiting on the slowest one.

If a call does not run inside a pool you installed yourself, gemmkit reaches for one
of its own pools instead. This is on by default on native targets, x86_64 and
aarch64. gemmkit keeps up to `GEMMKIT_POOL_CLASSES` (default 2
on x86_64, 1 on aarch64) private, persistent pools. Each is sized to an exact halving
tier of the machine width. On a 32-thread part that means tiers of 16 and 8 threads.
On a 14-core M4 Max it means a single 7-wide tier. Each pool builds lazily on first
use and is never rebuilt.

Auto snaps its worker count exactly onto one of these tiers instead of forking the
full-width global pool. A fork's overhead tracks the pool's idle slack: the threads
it owns beyond the ones actually engaged. A small GEMM drowns in that slack on a
full-width pool, so matching the pool to the work avoids the drag.

None of this changes what you already know above. An `install`'d call is still fully
respected and never redirected to a tier pool. An explicit `Rayon(n)` still gets
exactly `n` workers, just routed into whichever tier pool is the smallest fit. What
does change is idle memory. By default, an x86_64 process now parks about 24 extra
threads, the 16- and 8-wide tier pools, alongside the global pool. An aarch64 M4 Max
process parks 7 (its single half-width tier). All of them stay asleep until a small
GEMM needs them. Set `GEMMKIT_POOL_CLASSES=0` to disable tier pools entirely. Every
call then falls back to the ambient pool.

The threaded-wasm story is different: there gemmkit always sizes a dedicated pool of
its own. See [no_std and WebAssembly](no_std_and_WebAssembly.md) for that case.

## The reproducibility promise, precisely

For a fixed input, environment, and configuration, the output is identical
regardless of the worker count. That is the contract.

It holds for 2 reasons. First, `kc`, `nc`, and the fixed depth-panel order are the
only things that shape each output element's summation. gemmkit computes all 3
independently of how many threads will run them. Second, a single worker reduces
each output element start to finish, over the full contraction depth. No split
reduction exists whose order could depend on the schedule.

The flat job list itself is not strictly identical across worker counts: a wide
worker count can shrink `mc` to keep the list deep enough. But `mc` always stays a
multiple of `mr`, so the set of microtiles, and their numerics, stay unchanged. The
packed bytes do not depend on who packs them, either. Which worker computes a given
tile varies from run to run. The numerical result does not.

What is **not** promised is bitwise identity between `Serial` and `Rayon(n)`. It
happens to hold today on the driver paths, since serial and parallel run the same
kernel. But build on reproducibility under a fixed config, not on
serial-versus-parallel bit equality. You will not get cross-machine or cross-config
bit equality here. Floating-point GEMM is order-sensitive, and the config (ISA,
blocking, thread cap) is part of the fixed input. Integer `gemm_i8` is the
exception. It is bit-identical across ISAs and worker counts, because `i32`
addition is order-independent.

## When Serial is the right call

Reach for `Serial` in 3 situations.

1. **Small problems.** Below the workload gate, auto is serial anyway. Passing
   `Serial` explicitly also skips the `available_parallelism` probe and the fork
   machinery entirely, which is cheaper in a tight loop of tiny GEMMs.
2. **When you own the outer parallelism.** Suppose you already run many independent
   GEMMs across a rayon pool, or you parallelize a batch loop yourself. Do not let
   each inner call also fan out. That oversubscribes the machine, and it usually
   hurts performance instead of helping. Run the inner calls `Serial` and keep the
   parallelism at the outer level. For a batch of products, prefer the built-in
   [Batched GEMM](Batched_GEMM.md) entries instead, since they schedule the whole
   batch as one unit.
3. **Determinism-sensitive debugging.** Use the single-threaded path to rule out
   scheduling as a variable.

## Bandwidth-bound shapes get their own policy

A matrix-vector product (`m == 1` or `n == 1`), and other memory-bound shapes, are
not compute-bound. The work-based worker count above is the wrong model for them, so
these routes use a separate rule.

Below a cache-derived byte floor, the matrix fits one core's private cache. That core
already saturates the cache alone, so splitting the work only adds contention, and
the route stays serial. Above the floor, the route steps straight to a width chosen
for the bytes touched. This width tops out at half the logical core count, because a
gemv saturates its bandwidth well before the last core joins in.

That width climbs in steps, not smoothly. Each step is one of the exact-fit thread
pools described above, so a bandwidth-bound call also gets a pool sized exactly to
it. A few workers is the worst point on a bandwidth scaling curve, so the policy
jumps over that point instead of ramping through it.

This whole policy is automatic. The byte floor, the step spacing, and a flat override
are all tunable, through `GEMMKIT_GEMV_PARALLEL_BYTES`, `GEMMKIT_GEMV_TIER_STEP`, and
`GEMMKIT_GEMV_THREAD_CAP`. See [Small Shapes and GEMV](Small_Shapes_and_GEMV.md) for
the full treatment.

## Where to next

- [Small Shapes and GEMV](Small_Shapes_and_GEMV.md): the bandwidth-bound policy in
  detail.
- [Batched GEMM](Batched_GEMM.md): scheduling many products as one batch instead of
  nesting parallelism.
- [Tuning Knobs](Tuning_Knobs.md): the `GEMMKIT_*` thresholds behind these decisions.
- [Parallel Execution](../architecture/Parallel_Execution.md): the job cursor and
  worker-count resolution internals.
