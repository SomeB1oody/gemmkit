# Tuning Knobs

Every heuristic in gemmkit, when to go parallel, when to pack an operand, where a shape stops being small, is a named threshold with a shipped default, not a hard-coded constant. The defaults were calibrated on a Ryzen 9950X (Zen5), with a handful split by architecture where aarch64 measured differently, and they are good on most hardware. When they are not, each one is reachable three ways without touching the source.

## Resolution order

A knob resolves at the point it is read, taking the first of these that is set:

1. **Per-call argument.** Where a knob has a call-site equivalent, that wins outright. The clearest case is parallelism: the [`Parallelism`](Parallelism_in_Practice.md) argument you pass to `gemm` overrides any global thread policy. This layer lives in the API, not in `tuning`.
2. **Programmatic setter.** `gemmkit::tuning::set_*(v)` stores a value unconditionally. Once set, later reads never consult the environment again. This is for an application that tunes itself in code: it should win over whatever the deployment environment supplies.
3. **Environment variable.** `GEMMKIT_*`. This is the deployment layer: `source` a profile (for instance one emitted by [gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md)) to retune an already-built binary for a host with no recompile.
4. **Compiled default.** The calibrated constant, arch-split where it needed to be.

The ordering of setter over env is deliberate: an app that calls the setters has opted out of the environment; an app that wants a deployment profile to apply simply does not call them.

Environment variables are **read once, on the first access to that knob, then cached** as an atomic. A value set after the first read of a given knob is ignored, so export the profile before the process starts. A `GEMMKIT_*` var that is set but does not parse as a non-negative integer is treated as a typo, not a silent no-op: gemmkit warns on stderr and falls back to the default (the warning fires once per knob, since the fallback is then cached). It never panics, because a perf-knob typo must not crash the process.

## The knobs

The knobs below are the full catalog across every feature and target configuration; the internal `tuning::knob_env_names` registry is the source of truth these are drawn from. Two are feature- or target-gated and only exist when compiled in. Every getter has a matching `set_*`; the env var name is the getter's name upper-cased with the `GEMMKIT_` prefix.

### Serial / parallel gate

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_PARALLEL_THRESHOLD` | `set_parallel_threshold` | 48*48*256 | Below this `m*n*k`, work is forced onto a single thread. This is the serial-to-parallel break-even; raise it if your thread pool is expensive to fork, lower it if you have cheap threads and small products worth splitting. |

### Pack gates and strides

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_RHS_PACK_THRESHOLD` | `set_rhs_pack_threshold` | 2048 | Pack the RHS macro-panel only when `m` (how many row blocks reuse it) exceeds this; below it, B is read in place. |
| `GEMMKIT_LHS_PACK_THRESHOLD` | `set_lhs_pack_threshold` | 1024 (aarch64: 256) | Pack the LHS only when per-worker column reuse exceeds this. Packing is cheaper on aarch64, so it pays from lower reuse there. |
| `GEMMKIT_LHS_PACK_STRIDE` | `set_lhs_pack_stride` | 0 (auto) | Byte gate: a column-major A whose per-step depth stride `csa * sizeof(Lhs)` reaches this is packed to dodge a TLB- and cache-hostile strided read, independent of reuse. `0` derives it from the OS page size. ANDed with the span and reuse gates below: stride, span, and reuse must all hold before the force-pack fires. |
| `GEMMKIT_LHS_PACK_SPAN` | `set_lhs_pack_span` | 0 (auto) | Address-span companion to the stride gate: the page-scale stride only force-packs a column-major A when the whole depth-slice walk (`csa * sizeof(Lhs) * kc`) also reaches this many bytes. A page-scale stride over a span that stays cache-resident re-walks warm lines and is faster in place than the pack it would pay for. `0` means auto (4 MiB). |
| `GEMMKIT_LHS_PACK_REUSE` | `set_lhs_pack_reuse` | 128 | Reuse floor that prices the force-pack's benefit rather than its cost: the stride and span gates above only fire when at least this many `nr`-wide column tiles reuse each packed panel (`min(n, nc) / nr`, rounded up). A tall/skinny shape (`m` much greater than `n`) has a huge span but few column tiles, so it would amortize an expensive pack over too little reuse; `0` drops the floor and lets the stride+span pair decide alone. No arch split. |
| `GEMMKIT_SHARED_LHS_MNK` | `set_shared_lhs_mnk` | 8e9 (aarch64: 5e7; 32-bit: disabled) | `m*n*k` gate for the shared-A pre-pass on the parallel packed path, which removes redundant per-worker packs at the cost of a fork-join barrier. Independent of this gate, the pre-pass also opens from 16 workers up, where the per-worker redundancy always outweighs the barrier. |
| `GEMMKIT_PACK_TRANSPOSE_TILE` | `set_pack_transpose_tile` | 16 | Strip length for the cache-blocked transpose used when a packed operand is strided, turning a per-element gather into blocked copies. Backs both the real and complex packers. |

### Special-path thresholds

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_GEMV_THRESHOLD` | `set_gemv_threshold` | unbounded | Caps `min(m, n)` for the dedicated gemv path when the other dimension is 1. Shape, not size, triggers gemv; this only bounds it. |
| `GEMMKIT_SMALL_K_THRESHOLD` | `set_small_k_threshold` | 16 (aarch64: 8) | At or below this `k`, a shape takes the generic small-`k` route (one depth panel, no packing) instead of the register-tiling driver. |
| `GEMMKIT_SMALL_MN_DIM` | `set_small_mn_dim` | 16 | Both `m` and `n` at or below this (with a long `k`) take the horizontal inner-product route, where each output is one SIMD-reduced dot. `0` disables the route. |
| `GEMMKIT_SMALL_MN_PACK_MIN_K` | `set_small_mn_pack_min_k` | 16 | The `k` gate for the small-`m,n` pack tier: a strided small shape copies the failing operand into `k`-contiguous scratch only above this `k`. |
| `GEMMKIT_GEMV_PARALLEL_BYTES` | `set_gemv_parallel_bytes` | 0 (auto) | Byte floor below which a bandwidth-bound gemv/gevv stays single-threaded (below it the touched data is LLC-resident and splitting only loses). `0` derives it from the LLC size. |
| `GEMMKIT_GEMV_THREAD_CAP` | `set_gemv_thread_cap` | 0 (auto) | Max workers a bandwidth-bound gemv/gevv may use, since DRAM saturates far below the logical core count. `0` derives a proxy from the core count; raise it on a high-bandwidth part. |
| `GEMMKIT_K_STREAM_MAX` | `set_k_stream_max` | 32 | The `k` ceiling below which an axpy-shape gemv holds its output panel in registers across the whole depth sweep; above it the plain column-outer form wins. |
| `GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER` | `set_seq_internal_bytes_per_worker` | 128 KiB | aarch64 batched-GEMM crossover: a batch element splits across the machine rather than running one-per-worker cache-hot once its per-batch-worker byte share exceeds this. Only consulted on aarch64. |
| `GEMMKIT_I8_VNNI_MIN_PAR_MNK` | `set_i8_vnni_min_par_mnk` | 768^3 | Below this `m*n*k`, an auto-selected VNNI `i8` kernel hands a *multi-threaded* problem to the widen fallback (VNNI's mandatory RHS-pack barrier does not pay on a small parallel problem). Bit-identical to VNNI. Requires the `int8` feature. |

### Scheduler grains

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_PARALLEL_OVERSAMPLE` | `set_parallel_oversample` | 8 | The parallel driver aims for this many work chunks per worker, drained from a shared cursor on demand. Higher is finer load balance with a smaller tail but more atomic claims; lower is coarser with less overhead. |
| `GEMMKIT_PAR_MNK_PER_WORKER` | `set_par_mnk_per_worker` | 2000000 (threaded wasm: 262144) | Auto worker-count granularity: the auto path targets `m*n*k` divided by this much work per worker (then capped by cores and jobs, floored at 1), so the count scales with total flops rather than linear size. A wasm worker costs far less to engage than a native thread, hence the lower wasm floor. `0` behaves as `1` (always full width). |
| `GEMMKIT_PACKED_OVERSAMPLE` | `set_packed_oversample` | 2 | The packed-LHS path's split target (distinct from the general grain above): splitting harder re-packs A too often and regresses, so this optimum is lower. |

### Blocking caps

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_MC_REG_PANELS` | `set_mc_reg_panels` | 8 | The A macro-panel is bounded to this many microtile rows (`this * MR`), following BLIS's rule that MC stays a small multiple of MR. |
| `GEMMKIT_NC_NO_L3_PANELS` | `set_nc_no_l3_panels` | 512 | The no-L3 column block (Apple Silicon and the like) is `min(this * NR, N)`. Dead where an L3 exists. |
| `GEMMKIT_TINY_BLOCK_DIM` | `set_tiny_block_dim` | 64 | A shape with both `m` and `n` at or below this skips the full BLIS blocking model and just keeps A/B panels in L2. |
| `GEMMKIT_KC` | `set_kc` | 512 | The depth block in the tiny-matrix shortcut: `k` clamped to this. |
| `GEMMKIT_KC_MIN` | `set_kc_min` | 512 | The main-model `kc` floor: the L1-fit depth estimate is raised to at least this, so a small L1 never starves the microkernel's depth walk. |

### Deep-contraction and wasm

| Env var | Setter | Default | Controls |
| --- | --- | --- | --- |
| `GEMMKIT_DEEP_KC_BYTES` | `set_deep_kc_bytes` | 0 (auto) | The engage gate, in bytes, for the deep-contraction path: a narrow-output family (`f16`/`bf16`) switches from its single-panel form to an f32-output multi-slice twin once its single RHS micropanel (`nr * k * sizeof(N)`) outgrows this. `0` derives it from half the detected L2. |
| `GEMMKIT_WASM_THREADS` | `set_wasm_threads` | 8 | The worker count for a threaded wasm build, since wasm has no `available_parallelism` to query. Sizes gemmkit's wasm rayon pool. Only exists on wasm32 with the `wasm_threads` feature. |

## A note on GEMMKIT_FAST_TEST

You may see `GEMMKIT_FAST_TEST` in the test harness. It shrinks the correctness sweeps to run faster and is a **test-suite-only** switch; the library itself never reads it, and setting it has no effect on a production GEMM.

## Beyond hand-tuning

Setting knobs by hand is for when you already know which one to move. To calibrate the whole set for a specific machine, run the autotuner: it sweeps each knob over a probe-shape set and writes a `GEMMKIT_*` profile you `source` before running, with no recompile. That is the subject of the [gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md) chapter.
