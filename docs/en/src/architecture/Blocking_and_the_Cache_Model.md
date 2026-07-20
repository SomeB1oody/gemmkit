# Blocking and the Cache Model

The microkernel at the bottom of the driver multiplies one `MR x NR` tile of `C` held in registers, streaming a micropanel of `A` and a micropanel of `B` along the depth axis. That loop runs at machine peak only while both streams come out of nearby cache - and a GEMM touches far more data than any cache holds, reusing every element of `A` across `n` output columns and every element of `B` across `m` output rows. Blocking is how that reuse is choreographed: partition the problem so each operand block is brought into a specific cache level once and then read many times before anything evicts it. `KC` slices the depth so the two micropanels a tile multiplication reads stay resident in L1 for the whole tile; `MC` sizes the packed `A` macro-panel so it stays in L2 while the driver sweeps it across every column tile of the current column block; `NC` sizes the packed `B` macro-panel so it stays in L3 while every row block sweeps over it. Panel residency across the loop nest is the entire point - without it the same bytes would stream from DRAM `m`, `n`, or `k` times over.

Where many libraries hard-code `(MC, KC, NC)` per micro-architecture, gemmkit computes them analytically per call in `CacheTopology::blocking` (`gemmkit/src/cache.rs`, layer L3), following the BLIS model, from cache geometry detected at runtime. The inputs are the microtile `(mr, nr)`, the byte size of one packed input element, and the problem shape `(m, n, k)`; the output is the `Blocking { mc, kc, nc }` triple the driver's nest iterates by. [Life of a GEMM Call](Life_of_a_GEMM_Call.md) shows where each value lands in the nest; this page is about how the values are derived and why the model is shaped the way it is.

## Three constraints, three block sizes

### KC: both micropanels in L1 without self-eviction

Every microtile call walks `kc` depth steps, reading `mr` packed `A` elements and `nr` packed `B` elements per step, so an `mr x kc` and an `nr x kc` micropanel must coexist in L1d for the duration of the tile. The subtle requirement is *without self-eviction*: a cache is not a byte pool but sets of ways, and a panel that maps too many of its own lines onto the same sets evicts itself long before its total size reaches the cache size. The model therefore works in lines and sets, not bytes: it computes how many L1 lines one depth step of each micropanel claims, picks the largest `kc` whose combined footprint stays within the L1 associativity, raises the result to the `GEMMKIT_KC_MIN` floor (default 512) so a small L1 never starves the microkernel's depth walk, and clamps it to `k`. A final rebalance splits `k` into `ceil(k / kc)` equal-ish panels so the last depth slice is never a sliver.

### MC: the A macro-panel in L2, minus what B needs

Within one row block, the packed `mc x kc` `A` panel is reused across every column tile of the current column block, so it should fill L2 - but not all of it, because the `nr x kc` `B` micropanel of the same depth slice streams through L2 on every tile call. The model counts how many L2 ways that micropanel occupies, reserves those plus one spare way, and hands the remaining capacity to `A`: `mc` is that capacity divided by `kc`, rounded down to a multiple of `mr`, rebalanced so the row blocks come out even, and finally clamped by the BLIS-style hard cap of `GEMMKIT_MC_REG_PANELS * MR` rows (default 8 microtile rows). The cap is a calibration point rather than an invariant - it currently binds on every measured topology, so in practice `MC` is a small multiple of `MR` and the L2 capacity term is headroom.

### NC: the B macro-panel in L3, or a panel cap without one

With an L3 present, one way is reserved for the `A` traffic passing through and the rest budgets the packed `kc x nc` `B` macro-panel: `nc` is that capacity divided by `kc`, rounded down to a multiple of `nr` and rebalanced across `n`. Some machines report no L3 at all - Apple Silicon's cluster-shared L2 tops the hierarchy - and there the model runs full-`N` up to a panel-count cap: `nc` is `GEMMKIT_NC_NO_L3_PANELS * nr` (default 512 panels, i.e. 2048 columns at `nr = 4`) capped by `n`. With no L3 to keep `B` resident in, `B` streams from DRAM anyway; the cap only bounds the shared packed-`B` buffer rather than modeling residency.

## Sized in packed elements, not accumulator elements

The `sizeof` argument is the size of one **packed input** element - the driver passes `size_of::<Fam::Lhs>()`, not the accumulator size - because the panels the model is budgeting are stored in packed `Lhs`/`Rhs` units. For `f32`/`f64` the two coincide and nothing changes, but for narrow types the distinction is worth real depth: `i8` packs 1 byte per element against a 4-byte `i32` accumulator, `f16`/`bf16` pack 2 bytes against a 4-byte `f32` accumulator, so sizing by the accumulator would cut their `kc` and `nc` to a quarter or half of what the caches actually fit. Narrow types get proportionally deeper blocks, which is exactly why they can outrun `f32` on the same hardware. The prepack entries reuse the same model with a sentinel row count so a [prepacked operand](Packing_and_Workspaces.md)'s geometry is independent of the eventual `m`.

## The tiny-matrix shortcut

When both `m` and `n` are at or below `GEMMKIT_TINY_BLOCK_DIM` (default 64), the full model is skipped: `kc` is `k` clamped to the `GEMMKIT_KC` ceiling (default 512), `mc` is whatever row count keeps the panel in L2 at that depth (capped by `m` itself), and `nc` is `n` rounded up to `nr`. A problem whose whole working set fits in L2 gains nothing from three levels of residency analysis; the shortcut spends the saved arithmetic where it matters, on the fixed per-call overhead small products are dominated by.

## Detection: a fallback chain that cannot fail

The model is only as good as the geometry it is fed, and there is no portable way to ask for cache geometry. gemmkit runs a best-effort chain in which `#[cfg]` only ever picks the *sniffing method*, never the *values*: a `#[cfg(target_arch)]` cannot tell an Intel from an AMD, and a VM or container can mask CPUID or hide `/sys`, so every backend returns an `Option` and the chain bottoms out in a constant that cannot fail.

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

The backends, in order: **CPUID** (`cache/cpuid.rs`) is an instruction, so it works regardless of OS, in containers and most VMs - Intel parts are read through the deterministic cache leaf `04h`, AMD parts through the legacy L1 (`0x8000_0005`) and L2/L3 (`0x8000_0006`) leaves, where L3 size arrives in units of 512 KiB. **Linux sysfs** (`cache/sysfs.rs`) parses `/sys/devices/system/cpu/cpu0/cache/index*/` with plain `std::fs` - a fallback on x86 Linux (a hypervisor that masks CPUID), the primary source on aarch64 Linux, which has no CPUID instruction. **macOS sysctl** (`cache/sysctl.rs`) reads `sysctlbyname` keys through a two-line `extern "C"` block (no `libc` dependency), preferring the Apple Silicon per-performance-level keys (`hw.perflevel0.*`, the P-cores) with the flat Intel-Mac keys as fallback; `sysctl` does not expose associativity, so conservative typical values are assumed - safe because the model clamps associativity with `.max(2)` and needs it only approximately. The bottom of the chain is `ZEN5_FALLBACK`, a static default calibrated on the Ryzen 9950X dev machine: 48 KiB / 12-way L1d, 1 MiB / 16-way private L2, 32 MiB / 16-way L3.

Two guards make the chain robust rather than merely ordered. `plausible` rejects half-populated reads - any level smaller than 4 KiB, a line under 16 bytes, or zero associativity fails the whole backend, so a masked leaf cannot poison blocking with zeros. And detection runs at most once per process: `Machine::current()` memoizes the topology behind a `OnceLock`, together with the OS page size (`getpagesize`, validated as a power of two between 4 KiB and 2 MiB), which drives the LHS-packing stride gate described in [Packing and Workspaces](Packing_and_Workspaces.md). A `no_std` build skips detection entirely and uses the Zen5 fallback with a 4 KiB page.

## `shared_by`: contention for what the driver puts there

Each `Level` carries `bytes`, `assoc`, `line`, and one derived field: `shared_by`, which divides the level's capacity into the `effective_bytes` the model actually budgets. It would be natural - and wrong - to store the hardware core-sharing count there. `shared_by` instead models *per-worker contention for the data the driver actually places at that level*, and the driver's placement is: per-worker `A`/`B` micropanels in L1d, each worker's private `A` macro-panel in L2, and one shared `B` macro-panel in L3.

That placement fixes the values. L1d is per-core, so its whole capacity serves one worker's micropanels: `shared_by = 1`. L3 is shared by every core in hardware, but the data the driver keeps there is a single panel that all workers *read in common* - the same bytes, not per-worker copies - so the whole level belongs to that one panel and `shared_by` is again `1`; dividing by the raw core count would shrink the budget 16- or 32-fold and crater `NC` for no reason. Only L2 holds genuinely private per-worker data, so only L2 uses the *physical-core* L2-sharing degree: `1` on parts with a private L2 (mainstream x86, Neoverse), the cluster size on parts where a core cluster shares one L2 (Apple Silicon), where several workers' private `A` panels really do contend for the same ways. Each backend must *derive* this rather than copy a raw count: sysfs divides the raw L2 `shared_cpu_list` count by the SMT degree read from L1d's sharing list (so hyperthread siblings are not double-counted), sysctl reads `hw.perflevel0.cpusperl2`, and the CPUID backend hard-sets `1` because x86 L2s are per-physical-core. On x86 and Graviton the whole mechanism therefore reduces to all-ones; it exists for the cluster-L2 parts, where it is the difference between blocking for the L2 a worker actually gets and blocking for one it has to share five ways.

## What the thread count moves, and what it cannot

`blocking` has no thread-count parameter, and for `KC` and `NC` that is a load-bearing omission: both depend only on the machine and the problem, so a serial run and a 32-worker run derive the same `KC` and `NC`, hence the same depth slices and the same fixed-order depth chain for every output element. `MC` is the one blocking dimension the driver *does* adjust for parallelism. When a wide worker count would leave the flat job list too shallow - fewer than a handful of chunks per worker, so the run's tail degenerates into idle workers waiting on whoever drew the last chunks - the driver shrinks `MC` to cut more row blocks and deepen the list (the parallel job-depth floor, detailed in [Parallel Execution](Parallel_Execution.md)). So the panel boundaries and the flat job list are *not* strictly worker-count independent anymore.

Bit-identity survives that regardless, because the shrink is numerics-free. `MC` always stays an `MR` multiple, so the set of microtiles it produces - every `MR`-aligned row offset plus the single `m`-tail tile - is identical under any split; a wider worker count only regroups the same tiles into more, smaller row blocks. And `KC`, the only blocking dimension that shapes a tile's accumulation order, never moves with the thread count. So a fixed input, environment, and configuration still gives identical output at any worker count - the reproducibility contract holds, `MC` shrink and all. Parallelism otherwise influences packing *decisions* - the LHS pack gate is per-worker column reuse, and the shared-`A` pre-pass engages only on large parallel problems - but those choose where packed bytes are staged and who writes them, never what values are computed. How the job list is split and why the contract holds end to end is the subject of [Parallel Execution](Parallel_Execution.md); the knobs named on this page, and every other `GEMMKIT_*` threshold, are cataloged in [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md).
