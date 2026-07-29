# Tuning Knobs

Every heuristic in gemmkit is a named threshold with a shipped default, not a
hard-coded constant. This covers questions like when to go parallel, when to pack an
operand, and where a shape stops being small. A handful of knobs are split by
architecture, where aarch64 needs a different value from the rest. The defaults are
good on most hardware. When one is not right for your machine, you can reach it
3 ways without touching the source.

## Resolution order

A knob resolves at the point it is read, taking the first of these that is set:

1. **Per-call argument.** Where a knob has a call-site equivalent, that wins
   outright. The clearest case is parallelism: the
   [`Parallelism`](Parallelism_in_Practice.md) argument you pass to `gemm` overrides
   any global thread policy. This layer lives in the API, not in `tuning`.
2. **Programmatic setter.** `gemmkit::tuning::set_*(v)` stores a value
   unconditionally. Once set, later reads never consult the environment again. This
   is for an application that tunes itself in code. It should win over whatever the
   deployment environment supplies.
3. **Environment variable.** `GEMMKIT_*`. This is the deployment layer. `source` a
   profile, for instance one emitted by
   [gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md), to retune an
   already-built binary for a host with no recompile.
4. **Compiled default.** The calibrated constant, arch-split where it needed to be.

The ordering of setter over env is deliberate. An app that calls the setters has
opted out of the environment. An app that wants a deployment profile to apply simply
does not call them.

Environment variables are **read once, on the first access to that knob, then
cached** as an atomic. A value set after the first read of a given knob is ignored,
so export the profile before the process starts. A `GEMMKIT_*` var that is set but
does not parse as a non-negative integer is treated as a typo, not a silent no-op.
gemmkit warns on stderr and falls back to the default. The warning fires once per
knob, since the fallback is then cached. It never panics: a perf-knob typo must not
crash the process.

## The knobs

The table below is the full catalog, across every feature and target configuration.
The internal `tuning::knob_env_names` registry is the source of truth these are drawn
from. 2 knobs are feature- or target-gated and only exist when compiled in. Every
getter has a matching `set_*`. The env var name is the getter's name, upper-cased,
with the `GEMMKIT_` prefix.

### Serial / parallel gate

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_PARALLEL_THRESHOLD` | `set_parallel_threshold` | 48*48*256 | Below this `m*n*k`, work is forced onto a single thread. This is the serial-to-parallel break-even. Raise it if your thread pool is expensive to fork. Lower it if you have cheap threads and small products worth splitting. |

### Pack gates and strides

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_RHS_PACK_THRESHOLD` | `set_rhs_pack_threshold` | 2048 | Pack the RHS macro-panel only when `m` (how many row blocks reuse it) exceeds this. Below it, B is read in place. |
| `GEMMKIT_LHS_PACK_THRESHOLD` | `set_lhs_pack_threshold` | 1024 (aarch64: 256) | Pack the LHS only when per-worker column reuse exceeds this. Packing is cheaper on aarch64, so it pays from lower reuse there. |
| `GEMMKIT_LHS_PACK_STRIDE` | `set_lhs_pack_stride` | 0 (auto) | Byte gate on the column-major depth stride `csa * sizeof(Lhs)`. Once the stride reaches this many bytes, A is packed to dodge a TLB- and cache-hostile strided read, independent of reuse. `0` derives it from the OS page size. This gate is ANDed with the span and reuse gates below, so stride, span, and reuse must all hold before the force-pack fires. |
| `GEMMKIT_LHS_PACK_SPAN` | `set_lhs_pack_span` | 0 (auto) | Address-span companion to the stride gate above. The page-scale stride only force-packs a column-major A when the whole depth-slice walk (`csa * sizeof(Lhs) * kc`) also reaches this many bytes. Below that span, the walk stays cache-resident and re-reads warm lines, so it is faster in place than the pack it would cost. `0` means auto (4 MiB). |
| `GEMMKIT_LHS_PACK_REUSE` | `set_lhs_pack_reuse` | 128 (aarch64: 4) | Reuse floor that prices the force-pack's benefit, not its cost. The stride and span gates above only fire above a reuse floor. That floor is measured in `nr`-wide column tiles reusing each packed panel (`min(n, nc) / nr`, rounded up). A tall, skinny shape (`m` much greater than `n`) has a huge span but few column tiles. It would amortize an expensive pack over too little reuse, so this floor holds it back. `0` drops the floor and lets the stride+span pair decide alone. On aarch64 the tradeoff nearly inverts: packing is cheap there, and the in-place walk strides small pages. So the aarch64 default packs from far lower reuse than the x86 one. |
| `GEMMKIT_SHARED_LHS_MNK` | `set_shared_lhs_mnk` | 8e9 (aarch64: 6e6, 32-bit: disabled) | `m*n*k` gate for the shared-A pre-pass on the parallel packed path, which removes redundant per-worker packs at the cost of a fork-join barrier. The crossover trades the barrier's cost against the packing it saves. Packing costs relatively less on aarch64, so its gate sits far below the x86 one. Independent of this gate, the pre-pass also opens from 16 workers up, where the per-worker redundancy always outweighs the barrier. |
| `GEMMKIT_PACK_TRANSPOSE_TILE` | `set_pack_transpose_tile` | 16 | Strip length for the cache-blocked transpose used when a packed operand is strided, turning a per-element gather into blocked copies. Backs both the real and complex packers. |

### Special-path thresholds

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_GEMV_THRESHOLD` | `set_gemv_threshold` | unbounded | Caps `min(m, n)` for the dedicated gemv path when the other dimension is 1. Shape, not size, triggers gemv. This knob only bounds it. |
| `GEMMKIT_SMALL_K_THRESHOLD` | `set_small_k_threshold` | 16 (aarch64: 8) | At or below this `k`, a shape takes the generic small-`k` route (one depth panel, no packing) instead of the register-tiling driver. |
| `GEMMKIT_SMALL_MN_DIM` | `set_small_mn_dim` | 16 (aarch64: 32) | Both `m` and `n` at or below this (with a long `k`) take the horizontal inner-product route, where each output is one SIMD-reduced dot. `0` disables the route. The register-tiling driver instead pads small row and column tiles up to a full microtile. It spends much of its work on that padding. The point where the driver starts to win differs by machine, which is why the aarch64 cap sits above the x86 one. |
| `GEMMKIT_SMALL_MN_PACK_MIN_K` | `set_small_mn_pack_min_k` | 16 | The `k` gate for the small-`m,n` pack tier: a strided small shape copies the failing operand into `k`-contiguous scratch only above this `k`. |
| `GEMMKIT_GEMV_PARALLEL_BYTES` | `set_gemv_parallel_bytes` | 0 (auto) | Byte floor below which a bandwidth-bound gemv/gevv stays single-threaded. Below it, the matrix fits one core's private cache, which that core saturates alone, so splitting only loses. `0` derives it from the detected cache. On a part with an L3, that is the per-core private L2. On an L3-less aarch64 part, it is an eighth of the shared cluster L2. |
| `GEMMKIT_GEMV_TIER_STEP` | `set_gemv_tier_step` | 0 (auto) | Byte spacing between the rungs of the auto gemv/gevv worker ladder. The width climbs one exact-fit pool tier per factor of this in bytes touched, starting from the byte floor above. `0` means 8. `1` collapses the ladder onto its top tier. This knob has no effect with fewer than 2 pool tiers active, so it does nothing under the single-tier aarch64 default. |
| `GEMMKIT_GEMV_AXPY_PAR_MIN_ROWS` | `set_gemv_axpy_par_min_rows` | 16384 (x86), 1024 (aarch64) | Output-row floor below which a **column-major** gemv stays serial instead of splitting its rows. For a column-major matrix, the output-row axis is the inner memory axis. A split then gives every worker a strided walk over the whole matrix, while the serial route makes one sequential pass. Only once each worker's run is long enough does splitting pay for the sequentiality it gives up. `0` disables the floor. The 2 defaults differ by an order of magnitude because the crossover sits much lower on aarch64. On that target the crossover also follows the bytes in one column rather than the row count, so an `f64`-dominated workload there wants half the default. A row-major gemv and the `half` mixed twin are never gated: both scale well when split, at every size. |
| `GEMMKIT_GEMV_THREAD_CAP` | `set_gemv_thread_cap` | 0 (auto) | A flat worker count for a bandwidth-bound gemv/gevv, replacing the ladder above outright. Use it to pin an exact width for a known machine. `0` keeps the ladder, which tops out at half the logical cores, since a gemv saturates its bandwidth far below the full width. |
| `GEMMKIT_K_STREAM_MAX` | `set_k_stream_max` | 32 | The `k` ceiling below which an axpy-shape gemv holds its output panel in registers across the whole depth sweep. Above it, the plain column-outer form wins. |
| `GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER` | `set_seq_internal_bytes_per_worker` | 128 KiB | aarch64 batched-GEMM crossover: a batch element splits across the machine rather than running one-per-worker cache-hot once its per-batch-worker byte share exceeds this. Only consulted on aarch64. |
| `GEMMKIT_I8_VNNI_MIN_PAR_MNK` | `set_i8_vnni_min_par_mnk` | 768^3 | Below this `m*n*k`, an auto-selected VNNI `i8` kernel hands a *multi-threaded* problem to the widen fallback instead. VNNI's mandatory RHS-pack barrier does not pay on a small parallel problem. Bit-identical to VNNI. Requires the `int8` feature. |

### Scheduler grains

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_PARALLEL_OVERSAMPLE` | `set_parallel_oversample` | 8 | The parallel driver aims for this many work chunks per worker, drained from a shared cursor on demand. Higher gives finer load balance with a smaller tail, at the cost of more atomic claims. Lower is coarser with less overhead. |
| `GEMMKIT_PAR_MNK_PER_WORKER` | `set_par_mnk_per_worker` | 2000000 (threaded wasm: 262144) | Auto worker-count granularity. The auto path targets `m*n*k` divided by this much work per worker, then caps the result by cores and jobs, floored at 1. The count then scales with total flops rather than linear size. A wasm worker costs far less to engage than a native thread, hence the lower wasm floor. `0` behaves as `1` (always full width). |
| `GEMMKIT_PACKED_OVERSAMPLE` | `set_packed_oversample` | 2 | The packed-LHS path's split target, distinct from the general grain above. Splitting harder re-packs A too often and regresses, so this optimum is lower. |
| `GEMMKIT_POOL_CLASSES` | `set_pool_classes` | 2 (aarch64: 1, elsewhere: 0) | Number of halving tiers below full machine width: half, then quarter. For each active tier, gemmkit keeps a private, persistent rayon pool, built lazily on first use and never rebuilt. The auto path snaps its worker count exactly to a tier so no thread sits idle at the fork/join barrier. An explicit `Rayon(n)` still gets exactly `n` workers and merely runs in the smallest tier pool that fits. `0` disables tier pools, leaving every call on the ambient pool. Clamped to 3. Defaults to 2 tiers on x86_64, 1 on aarch64, and 0 (off) on every other target. |
| `GEMMKIT_FULL_WIDTH_MNK` | `set_full_width_mnk` | 0 (auto) | The `m*n*k` above which the auto path leaves its largest tier pool for full machine width. Below it, auto stays on its largest tier even though more cores exist. The extra full-width workers would not yet pay for the added fork/join cost. `0` derives it per architecture: 110_000_000 on x86, 14_000_000 on aarch64, where full width, including the E-cores, pays off at a smaller problem size. `MAX` pins the auto path to the largest tier unconditionally, so full width never engages. |

### Blocking caps

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_MC_REG_PANELS` | `set_mc_reg_panels` | 8 | The A macro-panel is bounded to this many microtile rows (`this * MR`), following BLIS's rule that MC stays a small multiple of MR. |
| `GEMMKIT_NC_NO_L3_PANELS` | `set_nc_no_l3_panels` | 512 | The no-L3 column block (Apple Silicon and the like) is `min(this * NR, N)`. Dead where an L3 exists. |
| `GEMMKIT_TINY_BLOCK_DIM` | `set_tiny_block_dim` | 64 | A shape with both `m` and `n` at or below this skips the full BLIS blocking model and just keeps A/B panels in L2. |
| `GEMMKIT_KC` | `set_kc` | 512 (aarch64: 16384) | The depth block in the tiny-matrix shortcut: `k` clamped to this. Deeper slices keep winning further on aarch64 than on x86, so the aarch64 shortcut runs close to a single slice. |
| `GEMMKIT_KC_MIN` | `set_kc_min` | 512 | The main-model `kc` floor: the L1-fit depth estimate is raised to at least this, so a small L1 never starves the microkernel's depth walk. |

### Deep-contraction and wasm

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_DEEP_KC_BYTES` | `set_deep_kc_bytes` | 0 (auto) | The engage gate, in bytes, for the deep-contraction path. A narrow-output family (`f16`/`bf16`) normally runs the whole contraction as a single depth panel. It switches to an f32-output, multi-slice twin once its RHS micropanel (`nr * k * sizeof(N)`) outgrows this. `0` derives it from half the detected L2. |
| `GEMMKIT_PREFETCH_MIN_BYTES` | `set_prefetch_min_bytes` | 0 (auto) | The engage gate, in bytes, for the driver's C-tile software prefetch. Once a call's working set (`A + B + C` bytes) exceeds this, the output microtiles stream from beyond the LLC. The driver then issues a T0 prefetch of each microtile just ahead of its microkernel call, hiding the read-modify-write latency. Below it, the tiles are cache-resident and the hint would be pure overhead. `0` derives it from the per-core-reachable LLC (L3 where present, else L2). A non-zero value is the threshold verbatim, so `usize::MAX` disables the prefetch and `1` forces it on. This knob is x86_64-only, a no-op on other targets, so aarch64 and wasm are unchanged. It is also numerics-invisible: bit-identical on or off. |
| `GEMMKIT_WASM_THREADS` | `set_wasm_threads` | 8 | The worker count for a threaded wasm build, since wasm has no `available_parallelism` to query. Sizes gemmkit's wasm rayon pool. Only exists on wasm32 with the `wasm_threads` feature. |

## A note on GEMMKIT_FAST_TEST

You may see `GEMMKIT_FAST_TEST` in the test harness. It shrinks the correctness
sweeps to run faster, and it is a **test-suite-only** switch. The library itself
never reads it, and setting it has no effect on a production GEMM.

## Beyond hand-tuning

Setting knobs by hand is for when you already know which one to move. To calibrate
the whole set for a specific machine, run the autotuner. It sweeps each knob over a
probe-shape set and writes a `GEMMKIT_*` profile you `source` before running, with no
recompile. That is the subject of the
[gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md) chapter.
