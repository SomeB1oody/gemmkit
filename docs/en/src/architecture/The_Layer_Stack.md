# The Layer Stack

Every module in the core crate opens by declaring its place in a stack: `api.rs` says "Public core API (layer L8a)", `driver.rs` says "The generic GEMM driver (layer L4)", and so on down to `simd.rs` at L0. The labels are not decoration — they are the crate's dependency discipline written where you cannot miss it. This page walks the stack from the bottom up, so that by the time we reach the public API every word it uses has already been defined. The next page, [Life of a GEMM Call](Life_of_a_GEMM_Call.md), traverses the same stack in the other direction, following one call.

```
L8a  api        safe slice entries, *_with, *_unchecked; MatRef/MatMut
L7   dispatch   runtime ISA selection, one memoized fn pointer per type
L6   special    gemv, small-k, small-m,n, batched reroutes
L5   parallel   worker-count resolution, JobCursor work distribution
L4   driver     the generic 5-loop blocked GEMM, one for all families
L3   cache      topology detection + BLIS analytical blocking
L2   pack       micropanel packing primitives
L1   kernel     KernelFamily seam (float/mixed/int/complex) + Epilogue
L0   simd       ISA tokens + SimdOps vocabulary;  scalar: Scalar/Acc types
     ---        cross-cutting: tuning (GEMMKIT_* knobs), workspace (buffers)
```

## L0: the vocabulary — `scalar.rs` and `simd.rs`

The bottom layer defines what the rest of the crate is allowed to talk about. `gemmkit/src/scalar.rs` holds the data-type seam, and it is deliberately tiny:

```rust
pub trait Scalar: Copy + Send + Sync + PartialEq + 'static {
    /// The type in which products are accumulated. `Self` for `f32`/`f64`
    type Acc: Scalar<Acc = Self::Acc>;
    /// The additive identity
    const ZERO: Self;
    /// The multiplicative identity
    const ONE: Self;
}
```

That is the whole trait: identity constants and the accumulator type (`f16`/`bf16` accumulate in `f32`, `i8` in `i32`, `f32`/`f64`/complex in themselves). No arithmetic lives on it — all real math happens vectorized in `SimdOps` or in per-family scalar epilogues — so adding an element type never drags in a scalar arithmetic surface. The refinement traits `Float`, `NarrowFloat`, and `ComplexFloat` layer on the few extra capabilities specific paths need. What `scalar.rs` deliberately does not know: that SIMD exists. It has no idea its constants will end up broadcast into vector registers.

`gemmkit/src/simd.rs` plus the `simd/` backends (`avx512.rs`, `fma.rs`, `neon.rs`, `scalar.rs`, `wasm.rs`, and the complex glue in `complex.rs`) form the load-bearing wall. Two traits split the job: `Simd` is a zero-sized ISA *token* (`Avx512`, `Fma`, `Neon`, `ScalarTok`, `Simd128`, plus the dot-capable `Avx512Vnni`/`Avx512Bf16`) whose sole method, `vectorize`, is the `#[target_feature]` trampoline that puts runtime-selected intrinsics into feature-enabled codegen; `SimdOps<T>` is the thick per-element-type vocabulary — register type, `LANES`, load/store/broadcast/mul/add/fma/reduce, and the overridable `accumulate_tile` schedule. `KernelSimd<L, R, A, O>` on top of them is the widen/narrow seam that makes mixed precision work without a driver branch. What this module deliberately does not know: anything above it. Its module doc states it depends only on `scalar` and `core`, so it could be split into its own crate unchanged — `SimdOps` has no idea what a micropanel, a cache, or a GEMM is.

## L1: the operation-family seam — `kernel.rs`

`gemmkit/src/kernel.rs` and `kernel/` (`float.rs`, `mixed.rs`, `int.rs`, `complex.rs`, `epilogue.rs`) define `KernelFamily`: the bundle of everything that distinguishes one kind of GEMM from another — the `Lhs`/`Rhs`/`Acc`/`Out` types, the pack layout (`pack_lhs`/`pack_rhs`), the microkernel (`microkernel_epi`), and constants like `OUT_IS_ACC` and `DEPTH_MULTIPLE` that tell the driver how to block for the family. `FloatGemm<T>` is the baseline; `MixedGemm`, `IntGemm`/`IntGemmVnni`, and `ComplexGemm` are siblings that reuse the driver unchanged. This layer also owns the `Epilogue` trait with its zero-cost `Identity`, and the `AlphaStatus`/`BetaStatus` enums the driver precomputes so the microkernel never compares floats. What a family deliberately does not know: its own tile size. `MR_REG` and `NR` are const generics on the microkernel method, chosen per `(type, ISA)` at the dispatch site three layers up — the family compiles for any geometry, and a new tile is a new instantiation, never a new type.

## L2: the mechanical copy — `pack.rs`

`gemmkit/src/pack.rs` holds the two shared packing primitives the families' pack hooks delegate to (the complex family's plane-splitting pack is the one exception, and it lives with its family): `pack_panels`, the micropanel-major copy (LHS panels `mr` rows tall, RHS panels `nr` columns wide — the same routine with the "leading" and "depth" strides swapped, tails zero-filled, with a cache-blocked transpose for strided sources), and `pack_kgroup_panels`, the k-group-interleaved variant the dot-product families use. What it deliberately does not know: where its output goes. The same routine fills a transient per-call scratch region, a shared parallel pack buffer, and a caller-held `PackedRhs` that lives for the whole process — `pack.rs` never sees a `Workspace`, a worker, or a lifetime, only `dst`, `src`, and strides. That indifference is what makes the prepacked path byte-identical to the per-call path.

## L3: the machine model — `cache.rs`

`gemmkit/src/cache.rs` and its backends (`cache/cpuid.rs`, `cache/sysfs.rs`, `cache/sysctl.rs`) answer two questions: what does the cache hierarchy look like, and what blocking follows from it. Detection is a best-effort fallback chain that cannot fail — CPUID on x86, then Linux sysfs, then macOS sysctl, then a static default calibrated on a Zen5 part — with `#[cfg]` only ever picking the sniffing *method*, never the values, and the result memoized once in `Machine`. `blocking()` then computes `(MC, KC, NC)` analytically from the BLIS model: `KC` so the A and B micropanels coexist in L1, `MC` so the A macro-panel fits L2, `NC` so the B macro-panel fits L3. The key types are `Level` (with its carefully documented `shared_by` contention field), `CacheTopology`, and the blocking result. What this layer deliberately does not know: the thread count. `blocking()` has no worker parameter, and that omission is load-bearing — thread-count-independent blocking is the mechanism behind the reproducibility contract described in [Design Goals](Design_Goals_and_the_Big_Picture.md). Full detail in [Blocking and the Cache Model](Blocking_and_the_Cache_Model.md).

## L4: the engine — `driver.rs`

`gemmkit/src/driver.rs` is the one blocked loop nest that serves every family: the BLIS-order `jc -> pc -> flat job list` structure, the adaptive packing decisions (pack B per depth slice or read in place, pack A per worker, via a shared pre-pass, or not at all), the prepacked-RHS consumption, and the `pack_rhs_full` layout that the prepack API reuses so prepacked and plain GEMM produce identical panel bytes. Its public faces are `run`, `run_epilogue`, `run_packed_rhs`, and `run_packed_rhs_epilogue`, all funneling into the private `run_inner`. What it deliberately does not know: any concrete element type or ISA. The whole file is generic over `Fam: KernelFamily` and a `KernelSimd` token; it never names `f32`, never names AVX-512, and never branches on element type. That is the open/closed property — adding a family or an ISA leaves this file untouched, and `gemmkit/tests/open_closed.rs` proves it by driving the driver with a second, trivial family the crate does not ship.

## L5: work distribution — `parallel.rs`

`gemmkit/src/parallel.rs` owns the `Parallelism` enum (`Serial` or `Rayon(n)`, `Rayon(0)` = auto), workload-aware worker-count resolution (a serial gate below a total-work threshold, an explicit count honored but capped, an auto ramp that grows with `cbrt(m*n*k)` rather than jumping to all cores, and a separate bandwidth rule for gemv shapes), and the demand-driven machinery: `JobCursor`, a lock-free atomic cursor workers pull contiguous chunks from, plus the `job_grain`/`packed_block_grain` chunk sizing and the `for_each_worker` fork-join the driver uses as a barrier. It also provides `Ptr`, the `Send + Sync` pointer shim that lets raw pointers cross into rayon closures. What it deliberately does not know: what a job is. `JobCursor` hands out index ranges over an abstract count — nothing in this file mentions tiles, matrices, or families, which is why the same cursor schedules driver tiles, B-pack panels, A-pack row blocks, and gemv row panels. More in [Parallel Execution](Parallel_Execution.md).

## L6: the reroutes — `special.rs`

`gemmkit/src/special.rs` and `special/` (`gemv.rs`, `small_k.rs`, `small_mn.rs`, `batched.rs`) hold the paths for shapes the register-tiling driver fits poorly: matrix-times-vector, low-depth GEMM, small-`m,n` long-`k` inner products, and the batched orchestration layer. All sit behind the same public entries and are covered in [Special Paths](Special_Paths.md). What a special path deliberately does not know: why it was chosen. The gates — `gemv_threshold`, `small_k_threshold`, `small_mn_dim` — live in the dispatch layer above and the tuning module beside; `small_k::run` cannot even tell whether it is serving `gemm`, `gemm_fused`, or `gemm_map`, because the epilogue arrives as an opaque generic parameter.

## L7: runtime ISA selection — `dispatch.rs`

`gemmkit/src/dispatch.rs` and `dispatch/` (`isa.rs`, `float.rs`, `mixed.rs`, `int.rs`, `complex.rs`) turn "which kernel should this machine run" into a one-time decision. Each element type has one `OnceLock<Dispatched<T>>` slot: feature detection runs once, the winning monomorphized entry points (plain, prepacked, fused) plus the tile geometry are cached, and every later call is a plain indirect call through a *typed* function pointer — no `transmute`, no `AtomicPtr<()>`. This layer also owns the `Task<T>` problem descriptor, the degenerate-case handling in `execute`, the orientation normalization `orient_transpose`, the special-path gates, and the `GEMMKIT_REQUIRE_ISA` pin that forces (or fails loudly on) a specific kernel. What it deliberately does not know: where the pointers in a `Task` came from. Checked slice views and raw unchecked pointers arrive identical; validation happened above or not at all, and dispatch neither knows nor cares. See [SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md) and the user-facing [Runtime ISA Dispatch](../gemmkit-guide/Runtime_ISA_Dispatch.md).

## L8a: the public boundary — `api.rs`

`gemmkit/src/api.rs` and `api/` (`batched.rs`, `cplx.rs`, `fused.rs`, `int8.rs`, `map.rs`, `packed.rs`) define the `MatRef`/`MatMut` strided views, the per-family safe entries with their `*_with` (caller-owned workspace) and `*_unchecked` (raw engine) variants, the `validate_gemm_views` panic catalog, and the lowering of views into `Task`s. What it deliberately does not know: everything below dispatch. The API layer cannot see which ISA will run, what blocking will be chosen, or whether packing will happen — after validation it hands a `Task` to `dispatch::execute` and its job is done. Symmetrically, `MatRef` never appears below this layer; the rest of the crate speaks only pointers and strides.

## Why the arrows only point down

The dependency direction is the architecture's one hard rule: each layer is driven by the layers above and knows nothing about them. `simd` depends only on `scalar` and `core`; the driver never names an element type or ISA; nothing below L7 knows dispatch exists; nothing below L8a has ever heard of a slice. Three payoffs justify the discipline. First, extension cost: because knowledge only flows downward, a new ISA, element type, or family plugs in at its own layer and everything below is provably untouched — the seams described in [Extension Points](Extension_Points.md) work only because no lower layer could have special-cased what sits above it. Second, review locality: to audit the microkernel you read `kernel/float.rs` and the `SimdOps` contract, nothing else; to audit the scheduling you read `driver.rs` and `parallel.rs`. Third, testability: lower layers are exercised in isolation (SIMD conformance tests check every token against scalar models; the open/closed test drives the driver with a foreign family), which is what makes the correctness story in [Testing and Verification](Testing_and_Verification.md) tractable.

## The two cross-cutting modules

Two modules sit beside the stack rather than in it, because every layer needs them and neither depends on anything above `core`/`alloc`.

`gemmkit/src/tuning.rs` is the unified knob surface. Every heuristic threshold in the engine — the serial/parallel gate, pack gates and strides, the special-path thresholds, scheduler grains, blocking caps — lives here and resolves as: per-call argument > programmatic setter (`tuning::set_*`) > environment variable (`GEMMKIT_*`) > compiled default. Env vars are read once and cached; a malformed value warns on stderr and falls back rather than panics, because a perf-knob typo must not crash the process. The full set of `GEMMKIT_*` names is enumerated in the `tuning::knob_env_names` registry, which the out-of-crate consumers (the gemmkit-tune sweep table, the knob property tests, the fuzz setters) assert their lists against, so a new knob cannot silently escape coverage. The user-facing tour is [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md).

`gemmkit/src/workspace.rs` is the scratch-memory story. `Workspace` is a growable 64-byte-aligned buffer; `Workspace::regions` carves it into per-worker (or per-row-block) LHS regions plus one shared RHS region, with fail-closed overflow checks at the element-to-byte chokepoint. Under `std` a re-entrancy-safe thread-local pool supplies the default, so plain `gemm` allocates at most once per thread; the `*_with` entries thread a caller-owned workspace through instead, giving zero heap allocation after the first sufficiently large call; without `std` each call uses a fresh workspace. Details in [Packing and Workspaces](Packing_and_Workspaces.md).
