# Blocking and the Cache Model

The microkernel at the bottom of the driver multiplies one `MR x NR` tile of `C`. It holds the tile in registers and streams a micropanel of `A` and a micropanel of `B` along the depth axis. This loop reaches machine peak only when both streams come from nearby cache. A GEMM touches far more data than any cache holds. It reuses every element of `A` across `n` output columns, and every element of `B` across `m` output rows.

Blocking is how the driver arranges this reuse. It partitions the problem so each operand block enters a specific cache level once, then gets read many times before anything evicts it. `KC` slices the depth axis so the 2 micropanels a tile multiplication reads stay resident in L1 for the whole tile. `MC` sizes the packed `A` macro-panel so it stays in L2 while the driver sweeps it across every column tile of the current column block. `NC` sizes the packed `B` macro-panel so it stays in L3 while every row block sweeps over it. Panel residency across the loop nest is the entire point. Without it, the same bytes would stream from DRAM `m`, `n`, or `k` times over.

Many libraries hard-code `(MC, KC, NC)` per microarchitecture. gemmkit instead computes them analytically for each call, in `CacheTopology::blocking` (`gemmkit/src/cache.rs`, layer L3). The function follows the BLIS model and derives the sizes from cache geometry detected at runtime. Its inputs are the microtile `(mr, nr)`, the byte size of one packed input element, and the problem shape `(m, n, k)`. Its output is the `Blocking { mc, kc, nc }` triple that the driver's loop nest iterates by. [Life of a GEMM Call](Life_of_a_GEMM_Call.md) shows where each value lands in the nest. This page explains how the model derives the values, and why it takes this shape.

## 3 constraints, 3 block sizes

### KC: both micropanels in L1 without self-eviction

Every microtile call walks `kc` depth steps. Each step reads `mr` packed `A` elements and `nr` packed `B` elements. So an `mr x kc` micropanel and an `nr x kc` micropanel must both fit in L1d for the whole tile. The subtle requirement is *without self-eviction*. A cache is not a byte pool. It holds sets of ways. A panel that maps too many of its own lines onto the same sets evicts itself before its total size even reaches the cache size.

The model therefore works in lines and sets, not bytes. It computes how many L1 lines one depth step of each micropanel claims. Then it picks the largest `kc` whose combined footprint stays within the L1 associativity. It raises that result to the `GEMMKIT_KC_MIN` floor (default 512), so a small L1 never starves the microkernel's depth walk, and clamps it to `k`. A final rebalance splits `k` into `ceil(k / kc)` panels of near-equal size, so the last depth slice is never a sliver.

### MC: the A macro-panel in L2, minus what B needs

Within one row block, the driver reuses the packed `mc x kc` `A` panel across every column tile of the current column block. So the panel should fill L2. It cannot fill all of L2, though. The `nr x kc` `B` micropanel of the same depth slice also streams through L2 on every tile call. The model counts how many L2 ways that micropanel occupies. It reserves those ways plus one spare way, then hands the rest of the capacity to `A`.

It divides that remaining capacity by `kc` to get `mc`, and rounds the result down to a multiple of `mr`. It then rebalances the result so the row blocks come out even. A BLIS-style hard cap of `GEMMKIT_MC_REG_PANELS * MR` rows (default 8 microtile rows) clamps the result last. This cap is a calibration point, not a bound strictly derived from the L2 term. In practice the cap binds before the L2 capacity term does. So `MC` ends up as a small multiple of `MR`, and the L2 capacity term mostly serves as headroom.

### NC: the B macro-panel in L3, or a panel cap without one

When an L3 is present, the model reserves one way for the `A` traffic passing through. It budgets the rest for the packed `kc x nc` `B` macro-panel. It divides that capacity by `kc` to get `nc`, rounds the result down to a multiple of `nr`, and rebalances it across `n`.

Some machines report no L3 at all. On Apple Silicon, for example, a cluster-shared L2 tops the hierarchy. There, the model runs full-`N` up to a panel-count cap instead. `nc` is `GEMMKIT_NC_NO_L3_PANELS * nr` (default 512 panels, that is, 2048 columns at `nr = 4`), capped by `n`. With no L3 to keep `B` resident, `B` streams from DRAM regardless. The cap only bounds the shared packed-`B` buffer. It does not model residency.

## Sized in packed elements, not accumulator elements

The `sizeof` argument is the size of one **packed input** element. The driver passes `size_of::<Fam::Lhs>()`, not the accumulator size, because the model stores the panels it budgets in packed `Lhs`/`Rhs` units. For `f32` and `f64` the 2 sizes coincide, so nothing changes.

For narrow types the distinction matters more. `i8` packs 1 byte per element against a 4-byte `i32` accumulator. `f16` and `bf16` pack 2 bytes against a 4-byte `f32` accumulator. Sizing by the accumulator would cut their `kc` and `nc` to a quarter or half of what the caches actually fit. Narrow types instead get proportionally deeper blocks, which is why they can outperform `f32` on the same hardware. The prepack entries reuse the same model with a sentinel row count, so a [prepacked operand](Packing_and_Workspaces.md)'s geometry stays independent of the eventual `m`.

## Prefetching the output tile past the LLC

The 3 block sizes keep the `A` and `B` panels resident, but they say nothing about `C`. Every microtile call reads, modifies, and writes its `mr x nr` output tile. Once a call's working set (the `A`, `B`, and `C` bytes together) outgrows the per-core-reachable LLC, that output tile no longer lives in cache. Its store then reaches into DRAM.

The driver answers with a software prefetch, decided once per call. It compares the working set against `cache::prefetch_ws_bytes` (the `GEMMKIT_PREFETCH_MIN_BYTES` gate, where `0` means auto: the per-core-reachable LLC, L3 where present, otherwise L2). Once the gate clears, the driver issues a T0 prefetch of each output microtile just ahead of that tile's microkernel call. This pulls the lines the store will touch into L1 while the microkernel still computes.

The hint walks whole 64-byte cache lines along the tile's unit-stride dimension. A tile strided in both dimensions has no contiguous lines, so the driver skips it. The driver emits the hint only on `x86_64` (`prefetcht0`, baseline SSE, needs no feature gate). On any other target it lowers to nothing, so aarch64 and wasm stay unaffected. The prefetch moves cache lines only, never arithmetic, so results stay bit-identical whether the gate is on, off, or forced. Below the gate, where the tiles stay cache-resident, the prefetch path adds no extra cost.

## The tiny-matrix shortcut

When both `m` and `n` are at or below `GEMMKIT_TINY_BLOCK_DIM` (default 64), the driver skips the full model. It sets `kc` to `k`, clamped to the `GEMMKIT_KC` ceiling (default 2048, or 16384 on aarch64). It sets `mc` to whatever row count keeps the panel in L2 at that depth, capped by `m` itself. It sets `nc` to `n` rounded up to `nr`.

A problem whose whole working set fits in L2 gains nothing from 3 levels of residency analysis. The shortcut spends the saved arithmetic where it matters: on the fixed per-call overhead that dominates small products.

The ceiling counts 4-byte elements, and the shortcut divides it by the packed element size. So `f64` gets half the depth of `f32`, and `int8` gets 4 times as much. What stays fixed is the byte budget, which is what the hardware limit is about. One number then calibrates every element family, and a tuner sweep run on `f32` transfers to the rest.

The ceiling sets the depth-slice count, and the slice count drives 2 costs that pull in opposite directions. Each extra slice re-reads and re-writes `C`, re-enters the driver, and forks the workers once more on the parallel path. That argues for a deep ceiling.

A deeper slice also grows the packed A and B panels. Those panels must stay in a private L2, so that argues for a shallow ceiling. On x86 the panels reach about 1.1 MiB at the default, which is one Zen5 L2. The parallel cost is the larger of the 2, so the default sits at the residency limit rather than below it.

## Detection: a fallback chain that cannot fail

The model is only as good as the geometry it receives, and there is no portable way to ask for cache geometry. gemmkit instead runs a best-effort chain, where `#[cfg]` only ever picks the *sniffing method*, never the *values*. A `#[cfg(target_arch)]` check cannot tell an Intel part apart from an AMD part, and a VM or container can mask CPUID or hide `/sys`. So every backend returns an `Option`, and the chain bottoms out in a constant that cannot fail.

```rust
// gemmkit/src/cache.rs
#[cfg(feature = "std")]
fn detect() -> CacheTopology {
    // try the CPUID backend
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
    if let Some(t) = cpuid::detect().filter(plausible) {
        return t;
    }
    // try the sysfs backend
    #[cfg(all(target_os = "linux", not(miri)))]
    if let Some(t) = sysfs::detect().filter(plausible) {
        return t;
    }
    // try the sysctl backend
    #[cfg(all(target_os = "macos", not(miri)))]
    if let Some(t) = sysctl::detect().filter(plausible) {
        return t;
    }
    ZEN5_FALLBACK
}
```

The backends run in this order.

**CPUID** (`cache/cpuid.rs`) is a single instruction, so it works regardless of OS, in containers, and in most VMs. CPUID reads both vendors through the per-cache topology leaf (Intel `04h`, AMD `0x8000_001D`), which describes each cache *as reachable from the executing core*. On a multi-die part such as a 2-CCD Ryzen, the L3 figure is the one complex a core can reach (32 MiB on the 9950X). It is not the package total. That per-core-reachable figure is the semantic every consumer of the value wants. AMD parts or hypervisors without that leaf fall back to the legacy L1 (`0x8000_0005`) and L2/L3 (`0x8000_0006`) leaves. There, L3 size arrives in units of 512 KiB as a package total, and the legacy fields cannot even encode 16-way associativity.

**Linux sysfs** (`cache/sysfs.rs`) parses `/sys/devices/system/cpu/cpu0/cache/index*/` with plain `std::fs`. It serves as a fallback on x86 Linux, for a hypervisor that masks CPUID. It is also the primary source on aarch64 Linux, which has no CPUID instruction.

**macOS sysctl** (`cache/sysctl.rs`) reads `sysctlbyname` keys through a 2-line `extern "C"` block, with no `libc` dependency. It prefers the Apple Silicon per-performance-level keys (`hw.perflevel0.*`, the P-cores), with the flat Intel-Mac keys as a fallback. `sysctl` does not expose associativity, so the backend assumes conservative typical values. This is safe, because the model clamps associativity with `.max(2)` and needs it only approximately.

The bottom of the chain is `ZEN5_FALLBACK`, a static default calibrated on the Ryzen 9950X dev machine. L1d is 48 KiB and 12-way. L2 is 1 MiB, 16-way, and private. L3 is 32 MiB and 16-way.

2 guards make the chain robust rather than merely ordered. `plausible` rejects half-populated reads. Any level smaller than 4 KiB, a line under 16 bytes, or zero associativity fails the whole backend. A masked leaf therefore cannot poison blocking with zeros. Detection also runs at most once per process. `Machine::current()` memoizes the topology behind a `OnceLock`. It also memoizes the OS page size (`getpagesize`, validated as a power of two between 4 KiB and 2 MiB). That page size drives the LHS-packing stride gate described in [Packing and Workspaces](Packing_and_Workspaces.md). A `no_std` build skips detection entirely and uses the Zen5 fallback with a 4 KiB page.

## `shared_by`: contention for what the driver puts there

Each `Level` carries `bytes`, `assoc`, `line`, and one derived field, `shared_by`, which divides the level's capacity into the `effective_bytes` that the model actually budgets. Storing the hardware core-sharing count there would seem natural, but it would be wrong. `shared_by` instead models *per-worker contention for the data the driver actually places at that level*. The driver's placement is: per-worker `A`/`B` micropanels in L1d, each worker's private `A` macro-panel in L2, and one shared `B` macro-panel in L3.

That placement fixes the values. L1d is per-core, so its whole capacity serves one worker's micropanels, and `shared_by = 1`. L3 is shared by every core in hardware. The data the driver keeps there, though, is a single panel. All workers *read* that panel *in common*: the same bytes, not per-worker copies. So the whole level belongs to that one panel, and `shared_by` is again `1`. Dividing by the raw core count would shrink the budget many times over and crater `NC` for no reason.

Only L2 holds genuinely private per-worker data, so only L2 uses the *physical-core* L2-sharing degree. That degree is `1` on parts with a private L2, such as mainstream x86 and Neoverse. It is the cluster size on parts where a core cluster shares one L2, such as Apple Silicon. There, several workers' private `A` panels really do contend for the same ways. Each backend must *derive* this value rather than copy a raw count. sysfs divides the raw L2 `shared_cpu_list` count by the SMT degree read from L1d's sharing list, so hyperthread siblings are not double-counted. sysctl reads `hw.perflevel0.cpusperl2`. The CPUID backend hard-sets `1`, because x86 L2s are private to each physical core.

On x86 and Graviton the whole mechanism reduces to all-ones. It exists for the cluster-L2 parts. There, it decides whether the model blocks for the L2 a worker actually gets, or for one it must share across a whole cluster.

## What the thread count moves, and what it cannot

`blocking` takes no thread-count parameter. For `KC` and `NC` that omission carries real weight. Both depend only on the machine and the problem. So a serial run and a wide parallel run derive the same `KC` and the same `NC`. That gives every run the same depth slices, and the same fixed-order depth chain for every output element.

`MC` is the one blocking dimension the driver *does* adjust for parallelism. A wide worker count can leave the flat job list too shallow, with fewer than a handful of chunks per worker. The run's tail then degenerates into idle workers waiting on whoever drew the last chunks. When that happens, the driver shrinks `MC` to cut more row blocks and deepen the list. [Parallel Execution](Parallel_Execution.md) details this parallel job-depth floor. So the panel boundaries and the flat job list are *not* strictly worker-count independent anymore.

Bit-identity survives the `MC` shrink regardless, because the shrink itself carries no numerics. `MC` always stays an `MR` multiple. So the set of microtiles it produces (every `MR`-aligned row offset plus the single `m`-tail tile) stays identical under any split. A wider worker count only regroups the same tiles into more, smaller row blocks. `KC`, the only blocking dimension that shapes a tile's accumulation order, never moves with the thread count. So under a fixed configuration, changing only the worker count leaves every output element's accumulation order unchanged. That is the mechanism behind gemmkit's reproducibility contract. [Parallel Execution](Parallel_Execution.md) assembles the full contract and states its exact scope.

Parallelism otherwise influences packing *decisions* only. The LHS pack gate reads per-worker column reuse, and the shared-`A` pre-pass engages only on large parallel problems. Those decisions choose where the driver stages packed bytes and who writes them. They never change what values the kernel computes. [Parallel Execution](Parallel_Execution.md) covers how the job list splits, and how the contract holds end to end. [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md) catalogs every `GEMMKIT_*` threshold named on this page, and every other one.
