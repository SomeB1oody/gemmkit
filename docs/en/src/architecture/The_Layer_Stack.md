# The Layer Stack

Every module in the core crate opens by declaring its place in a stack. `api.rs` says
"Public core API (layer L8a)". `driver.rs` says "The generic GEMM driver (layer L5)",
and so on down to `simd.rs` at L0. The labels are not decoration. They write the
crate's dependency discipline where a reader cannot miss it. The map below lists the
modules in dependency order, which is what makes the downward claim checkable. This
page walks the stack from the bottom up. By the time it reaches the public API, every
word the API uses has already been defined. The next page,
[Life of a GEMM Call](Life_of_a_GEMM_Call.md), traverses the same stack in the other
direction, following one call.

```
L8a  api        safe slice entries, *_with, *_unchecked; MatRef/MatMut
L7   dispatch   runtime ISA selection, one memoized fn pointer per type
L6   special    gemv, small-k, small-m,n, batched reroutes
L5   driver     the generic 5-loop blocked GEMM, one for all families
L4   kernel     KernelFamily seam (float/mixed/int/complex) + Epilogue
L3   cache      topology detection + BLIS analytical blocking
L2   parallel   worker-count resolution, JobCursor work distribution
L1   pack       micropanel packing primitives
L0   simd       ISA tokens + SimdOps vocabulary;  scalar: Scalar/Acc types
     ---        cross-cutting: tuning (GEMMKIT_* knobs), workspace (buffers)
```

2 placements are worth calling out, because they are what keep every arrow pointing
down. `parallel` sits low, below the kernel families, the cache model, and the driver,
because it is self-contained worker vocabulary. The `Parallelism` policy enum, the
`Ptr` Send-pointer wrapper, and the `JobCursor` depend only on `tuning`. `kernel`,
`driver`, `special`, and `dispatch` all reach down to them. `pack` sits below `kernel`
because the families' pack hooks build on the packing primitives, not the other way
around.

## L0: the vocabulary, `scalar.rs` and `simd.rs`

The bottom layer defines what the rest of the crate is allowed to talk about.
`gemmkit/src/scalar.rs` holds the data-type seam, and it is deliberately tiny:

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

That is the whole trait: identity constants and the accumulator type. `f16` and
`bf16` accumulate in `f32`. `i8` accumulates in `i32`. `f32`, `f64`, and the complex
types accumulate in themselves. No arithmetic lives on `Scalar` itself. All real math
happens vectorized in `SimdOps`, or in per-family scalar epilogues, so adding an
element type never drags in a scalar arithmetic surface. The refinement traits
`Float`, `NarrowFloat`, and `ComplexFloat` layer on the few extra capabilities specific
paths need. What `scalar.rs` deliberately does not know: that SIMD exists. It has no
idea its constants will end up broadcast into vector registers.

`gemmkit/src/simd.rs` forms the load-bearing wall. The `simd/` backends join it:
`avx512.rs`, `fma.rs`, `neon.rs`, `scalar.rs`, `wasm.rs`, and the complex glue in
`complex.rs`. 3 traits split the job. `Simd` is a zero-sized ISA *token*. Examples
include `Avx512F`, `Fma`, `Neon`, `ScalarTok`, `Simd128`, and the dot-capable
`Avx512Vnni` and `Avx512Bf16`. Its sole method is `vectorize`, the `#[target_feature]`
trampoline. It puts runtime-selected intrinsics into feature-enabled codegen.
`SimdOps<T>` is the thick per-element-type vocabulary: register type, `LANES`,
load/store/broadcast/mul/add/fma/reduce, and the overridable `accumulate_tile`
schedule. `KernelSimd<L, R, A, O>` on top of them is the widen/narrow seam that makes
mixed precision work without a driver branch. What this module deliberately does not
know: anything above it. Its module doc states it depends only on `scalar` and `core`,
so it could be split into its own crate unchanged. `SimdOps` has no idea what a
micropanel, a cache, or a GEMM is.

## L1: the mechanical copy, `pack.rs`

`gemmkit/src/pack.rs` holds the 2 shared packing primitives that turn a strided A or
B region into contiguous, microkernel-sized panels. These are the copies the kernel
families of L4 delegate their pack hooks to. The complex family's plane-splitting pack
is the one exception, and it lives with its family.

The 2 primitives are `pack_panels` and `pack_kgroup_panels`. `pack_panels` is the
micropanel-major copy. LHS panels are `mr` rows tall, and RHS panels are `nr` columns
wide. It is the same routine for both, with the "leading" and "depth" strides swapped.
Tails are zero-filled, and a cache-blocked transpose handles strided sources.
`pack_kgroup_panels` is the k-group-interleaved variant the dot-product families use.

What `pack.rs` deliberately does not know: where its output goes. The same routine
fills a transient per-call scratch region, a shared parallel pack buffer, and a
caller-held `PackedRhs` that lives for the whole process. `pack.rs` never sees a
`Workspace`, a worker, or a lifetime, only `dst`, `src`, and strides. That indifference
is what makes the prepacked path byte-identical to the per-call path. Depending only on
`scalar`, it names no family and no cache, which is why it can sit this low.

## L2: work distribution, `parallel.rs`

`gemmkit/src/parallel.rs` owns 3 things.

First, the `Parallelism` enum: `Serial`, or `Rayon(n)` with `Rayon(0)` meaning auto.

Second, workload-aware worker-count resolution. A serial gate applies below a
total-work threshold. An explicit count is honored, but capped. An auto count scales
with the total work `m*n*k`, rather than jumping straight to all cores. A separate
bandwidth rule covers the memory-bound matrix-vector shapes.

Third, the demand-driven machinery. `JobCursor` is a lock-free atomic cursor workers
pull contiguous chunks from. The `job_grain` and `packed_block_grain` knobs size those
chunks. The `for_each_worker` fork-join is the barrier the higher layers use.
`parallel.rs` also provides `Ptr`, the `Send + Sync` pointer shim that lets raw
pointers cross into rayon closures.

Depending only on `tuning`, `parallel.rs` sits below everything that calls it. The same
worker vocabulary serves `kernel`, `driver`, `special`, and `dispatch` alike. What it
deliberately does not know: what a job is. `JobCursor` hands out index ranges over an
abstract count. Nothing in this file mentions tiles, matrices, or families. That is why
the same cursor later schedules driver tiles, B-pack panels, A-pack row blocks, and
gemv row panels, without distinguishing between them. More in
[Parallel Execution](Parallel_Execution.md).

## L3: the machine model, `cache.rs`

`gemmkit/src/cache.rs` and its backends (`cache/cpuid.rs`, `cache/sysfs.rs`,
`cache/sysctl.rs`) answer 2 questions. What does the cache hierarchy look like? What
blocking follows from it? Detection is a best-effort fallback chain that cannot fail.
It tries CPUID on x86, then Linux sysfs, then macOS sysctl, then a static default
calibrated on a Zen5 part. `#[cfg]` only ever picks the sniffing *method*, never the
values, and the result is memoized once in `Machine`.

`blocking()` then computes `(MC, KC, NC)` analytically from the BLIS model. `KC` is
sized so the A and B micropanels coexist in L1. `MC` is sized so the A macro-panel fits
L2. `NC` is sized so the B macro-panel fits L3. The key types are `Level` (with its
carefully documented `shared_by` contention field), `CacheTopology`, and the blocking
result. What this layer deliberately does not know: the thread count. `blocking()` has
no worker parameter, and that omission is load-bearing. Thread-count-independent
blocking is the mechanism behind the reproducibility contract described in
[Design Goals](Design_Goals_and_the_Big_Picture.md). Full detail in
[Blocking and the Cache Model](Blocking_and_the_Cache_Model.md).

## L4: the operation-family seam, `kernel.rs`

`gemmkit/src/kernel.rs` and `kernel/` (`float.rs`, `mixed.rs`, `int.rs`, `complex.rs`,
`epilogue.rs`) define `KernelFamily`. This is the bundle of everything that
distinguishes one kind of GEMM from another. A family bundles the `Lhs`, `Rhs`, `Acc`,
and `Out` types. It bundles the pack layout too: `pack_lhs` and `pack_rhs`, which
delegate to the L1 primitives. It bundles the microkernel, `microkernel_epi`. It also
bundles constants like `OUT_IS_ACC` and `DEPTH_MULTIPLE`, which tell the driver how to
block for the family.

`FloatGemm<T>` is the baseline. `MixedGemm`, `IntGemm`/`IntGemmVnni`, and `ComplexGemm`
are siblings that reuse the driver unchanged. This layer also owns the `Epilogue`
trait, with its zero-cost `Identity`. It owns the `AlphaStatus` and `BetaStatus` enums
too. The driver precomputes both, so the microkernel never compares floats. What a
family deliberately does not know: its own tile size. `MR_REG` and `NR` are const
generics on the microkernel method, chosen per `(type, ISA)` at the dispatch site three
layers up. The family compiles for any geometry, and a new tile is a new instantiation,
never a new type.

## L5: the engine, `driver.rs`

`gemmkit/src/driver.rs` is the one blocked loop nest that serves every family. It has
the BLIS-order `jc -> pc -> flat job list` structure. It makes the adaptive packing
decisions. It can pack B per depth slice, or read B in place. It can pack A per
worker, pack A through a shared pre-pass, or not pack A at all. It also handles
prepacked-RHS consumption, through the `pack_rhs_full` layout the prepack API reuses.
That reuse is why prepacked and plain GEMM produce identical panel bytes.

Its public faces are `run`, `run_epilogue`, `run_packed_rhs`, and
`run_packed_rhs_epilogue`, all funneling into the private `run_inner`. What it
deliberately does not know: any concrete element type or ISA. The whole file is
generic over `Fam: KernelFamily` and a `KernelSimd` token. It never names `f32`, never
names AVX-512, and never branches on element type. That is the open/closed property.
Adding a family or an ISA leaves this file untouched. `gemmkit/tests/open_closed.rs`
proves this. It drives the driver with a second, trivial family the crate does not
ship.

## L6: the reroutes, `special.rs`

`gemmkit/src/special.rs` and `special/` (`gemv.rs`, `small_k.rs`, `small_mn.rs`,
`batched.rs`) hold the paths for shapes the register-tiling driver fits poorly. These
are matrix-times-vector, low-depth GEMM, small-`m,n` long-`k` inner products, and the
batched orchestration layer. All sit behind the same public entries and are covered in
[Special Paths](Special_Paths.md). What a special path deliberately does not know: why
it was chosen. The gates, `gemv_threshold`, `small_k_threshold`, and `small_mn_dim`,
live in the dispatch layer above and the tuning module beside it. `small_k::run`
cannot even tell whether it is serving `gemm`, `gemm_fused`, or `gemm_map`, because the
epilogue arrives as an opaque generic parameter. The batched path is the one place a
layer reaches back up. `batched.rs` forwards each element through `dispatch::execute`
(L7), so it inherits the whole ladder above. That is the single annotated exception
discussed at the end of this page.

## L7: runtime ISA selection, `dispatch.rs`

`gemmkit/src/dispatch.rs` and `dispatch/` (`isa.rs`, `float.rs`, `mixed.rs`, `int.rs`,
`complex.rs`) turn "which kernel should this machine run" into a one-time decision.
Each element type has one `OnceLock<Dispatched<T>>` slot. Feature detection runs once.
The winning monomorphized entry points (plain, prepacked, fused) get cached, along
with the tile geometry. Every later call is a plain indirect call through a *typed*
function pointer, with no `transmute` and no `AtomicPtr<()>`.

This layer also owns the `Task<T>` problem descriptor, the degenerate-case handling in
`execute`, the orientation normalization `orient_transpose`, and the special-path
gates. It also owns the `GEMMKIT_REQUIRE_ISA` pin, which forces, or fails loudly on, a
specific kernel. What it deliberately does not know: where the pointers in a `Task`
came from. Checked slice views and raw unchecked pointers arrive identical. Validation
happened above or not at all, and dispatch neither knows nor cares. See
[SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md) and the user-facing
[Runtime ISA Dispatch](../gemmkit-guide/Runtime_ISA_Dispatch.md).

## L8a: the public boundary, `api.rs`

`gemmkit/src/api.rs` and `api/` (`batched.rs`, `cplx.rs`, `fused.rs`, `int8.rs`,
`map.rs`, `packed.rs`) define several things. These are the `MatRef`/`MatMut` strided
views, the per-family safe entries, and the `validate_gemm_views` panic catalog. The
safe entries come in `*_with` (caller-owned workspace) and `*_unchecked` (raw engine)
variants. This layer also lowers views into `Task`s.

What it deliberately does not know: everything below dispatch. The API layer cannot
see which ISA will run, what blocking will be chosen, or whether packing will happen.
After validation it hands a `Task` to `dispatch::execute` and its job is done.
Symmetrically, `MatRef` never appears below this layer. The rest of the crate speaks
only pointers and strides.

## Why the arrows only point down

The dependency direction is the architecture's one hard rule. Each layer is driven by
the layers above it, and knows nothing about them. `simd` depends only on `scalar` and
`core`. The driver never names an element type or ISA. Nothing below L7 knows dispatch
exists. Nothing below L8a has ever heard of a slice.

The rule has exactly one deliberate, annotated exception. `special/batched.rs` (L6)
forwards each batch element back through `dispatch::execute` (L7). This re-entry lets
every element inherit the same driver, small-k, small-mn, and gemv routing a
standalone `gemm` call would take. It avoids a second dispatch ladder maintained by
hand. It is an upward arrow by design, and the only one in the crate.

3 payoffs justify the discipline. First, extension cost. Because knowledge only
flows downward, a new ISA, element type, or family plugs in at its own layer.
Everything below it is provably untouched. The seams described in
[Extension Points](Extension_Points.md) work only because no lower layer could have
special-cased what sits above it. Second, review locality: to audit the microkernel,
read `kernel/float.rs` and the `SimdOps` contract, nothing else. To audit the
scheduling, read `driver.rs` and `parallel.rs`. Third, testability: lower layers are
exercised in isolation. SIMD conformance tests check every token against scalar
models, and the open/closed test drives the driver with a foreign family. This is what
makes the correctness story in
[Testing and Verification](Testing_and_Verification.md) tractable.

## The 2 cross-cutting modules

2 modules sit beside the stack rather than in it, because every layer needs them and
neither depends on anything above `core`/`alloc`.

`gemmkit/src/tuning.rs` is the unified knob surface. Every heuristic threshold in the
engine lives here: the serial/parallel gate, pack gates and strides, the special-path
thresholds, scheduler grains, and blocking caps. Each one resolves in this order:
per-call argument, then programmatic setter (`tuning::set_*`), then environment
variable (`GEMMKIT_*`), then compiled default. Env vars are read once and cached. A
malformed value warns on stderr and falls back rather than panics, because a perf-knob
typo must not crash the process. The full set of `GEMMKIT_*` names is enumerated in
the `tuning::knob_env_names` registry. The out-of-crate consumers, the gemmkit-tune
sweep table, the knob property tests, and the fuzz setters, assert their lists against
it. So a new knob cannot silently escape coverage. The user-facing tour is
[Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md).

`gemmkit/src/workspace.rs` is the scratch-memory story. `Workspace` is a growable
64-byte-aligned buffer. `Workspace::regions` carves it into per-worker (or
per-row-block) LHS regions plus one shared RHS region, with fail-closed overflow
checks at the element-to-byte chokepoint. Under `std` a re-entrancy-safe thread-local
pool supplies the default, so plain `gemm` allocates at most once per thread. The
`*_with` entries thread a caller-owned workspace through instead, giving zero heap
allocation after the first sufficiently large call. Without `std`, each call uses a
fresh workspace. Details in
[Packing and Workspaces](Packing_and_Workspaces.md).
