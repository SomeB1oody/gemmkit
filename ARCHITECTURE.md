# Architecture

This document describes the design of the gemmkit workspace as it exists in
the code. It covers the layer structure of the core engine and the path a
call takes from the public API to the microkernel. It also covers the seams
that make new instruction sets and element types cheap to add. File
references are repo-relative paths. The API reference lives at
[docs.rs/gemmkit](https://docs.rs/gemmkit).

## Goals and constraints

gemmkit computes `C <- alpha*A*B + beta*C` over `&[T]` slices with stride
views, or over raw pointers with `isize` strides. It selects the best
available instruction set at runtime. The crate uses edition 2024 and
`rust-version` 1.89, and is licensed MIT OR Apache-2.0 (`LICENSE-MIT`,
`LICENSE-APACHE` at the repo root). The design targets:

- **Safety at the boundary.** The checked entries (`gemm`, `gemm_fused`, and
  so on) validate before any unsafe work runs. They panic on a shape
  mismatch, an out-of-bounds stride, a self-aliasing output, or `C`
  overlapping `A`/`B` (`validate_gemm_views` in `gemmkit/src/api.rs`). The
  `*_unchecked` tier exposes the raw engine and allows negative strides, for
  callers that validate their own inputs, such as the adapters.
- **Reproducible, not bitwise, parallel results.** For a fixed machine and a
  fixed configuration, gemmkit returns reproducible output. The worker count
  is part of that configuration, so the contract does not promise
  bitwise-identical output across 2 different worker counts.

  The blocking parameters explain why serial and parallel runs agree in
  practice. `kc` and the fixed depth-panel order set every element's
  accumulation order. Neither depends on the thread count. `mc` may shrink
  with the worker count to keep the job list deep enough. It always stays
  an `mr` multiple, though, so the microtile set does not change. Each
  output element is reduced start to finish by one worker. This is why
  bitwise
  serial-versus-parallel identity holds on the driver paths today. That
  agreement is a result of the design, not an added guarantee beyond the
  fixed-configuration promise.
- **No macros, no `transmute` in the variation points.** ISA, element type,
  and operation family are traits. Dispatch slots are typed function
  pointers in `OnceLock`s. Tile geometry is a pair of const generics.
- **`no_std` and zero mandatory dependencies.** With default features off,
  the crate is `#![no_std]`. It needs only `core` and `alloc`, and depends
  on nothing else. Compile-time target features replace runtime CPU
  detection, env knobs turn off, and a per-call workspace replaces the
  thread-local pool. Optional features each pull in one crate: `std` adds
  `raw-cpuid` (x86 only), `parallel` adds `rayon`, `half` adds `half`, and
  `complex` adds `num-complex`. `int8` and `epilogue` add no dependency.

## Workspace layout

5 crates release in lockstep at version 0.1.1, and all 5 are
published on crates.io. The fuzz crate is a separate nightly-only root,
excluded from the workspace.

| Path | Crate | Role |
|---|---|---|
| `gemmkit/` | [gemmkit](https://crates.io/crates/gemmkit) | The core GEMM engine (everything below) |
| `gemmkit-ndarray/` | [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) | Zero-copy adapter over `ndarray` (>= 0.17.1) views |
| `gemmkit-nalgebra/` | [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra) | Zero-copy adapter over `nalgebra` 0.35 matrices |
| `gemmkit-faer/` | [gemmkit-faer](https://crates.io/crates/gemmkit-faer) | Zero-copy adapter over `faer` 0.24 matrices |
| `gemmkit-tune/` | [gemmkit-tune](https://crates.io/crates/gemmkit-tune) | Install-time autotuner binary (emits a `GEMMKIT_*` env profile) |
| `gemmkit/fuzz/` | gemmkit-fuzz | cargo-fuzz targets, its own workspace root, excluded from the stable workspace |

The adapters pull matrix pointers and strides straight out of each
library's native views (C-order, F-order, general and reversed strides, no
copies). They forward them to the `*_unchecked` engine. The fused and
requantizing entries still need a bias/scale length check and a
`C`-overlap check. Over
raw, possibly gappy, or reversed views, the core's slice-based
`validate_gemm_views` cannot describe those checks, so they live once in
gemmkit's `#[doc(hidden)]` `adapter` module. That module is a pointer-level
support surface for the L8a checked entries, not a layer of its own. It is
versioned in lockstep with the adapters, following the `knob_env_names`
precedent. All 3 adapters and the core's own checked entries consume
that single implementation, instead of each re-deriving gemmkit's exact
panic wording.

Batched GEMM is exposed in the shape each library's types allow. The
ndarray adapter has a 3-D strided `gemm_batched` that batches on axis 0,
over the strided-batched engine. The nalgebra and faer adapters instead
take `gemm_batched` over a slice of per-element `(A, B)` inputs, paired
with a slice of `&mut C` outputs. That form runs over the pointer-array
`gemm_batched_ptr_unchecked` engine, since neither library has a rank-3
type. Each adapter feature (default `parallel`, plus `wasm_threads`,
`half`, `complex`, `int8`, `epilogue`) forwards to the same-named gemmkit
feature.

Each adapter re-exports every gemmkit item that its own signatures name, so
an adapter user never needs a direct `gemmkit` dependency. The re-exported
items include:

- the `Parallelism` and `Workspace` arguments
- the `Bias`, `Activation`, `Requantize`, and `RequantScale` epilogue
  selectors
- the `PackedLhs` and `PackedRhs` handles
- the `GemmScalar`, `FusedScalar`, `MapScalar`, and `ComplexScalar`
  element-type bounds, so a caller can name them and write a wrapper generic
  over an entry
- the feature-gated element types (`f16`, `bf16`, `Complex`, `c32`, `c64`),
  so `half` and `num-complex` also stay out of the caller's manifest
- the `tuning` module

Reaching `tuning` through the adapter is a correctness requirement, not only
a convenience. The knobs are process-global atomics. A setter reached
through a separately resolved second `gemmkit` would write a copy that the
adapter never reads. When a new gemmkit type appears in an adapter
signature, the same change must add it to that crate's re-exports.

## Layer map

Module docs in the core crate carry explicit layer labels. The map below
lists the modules in dependency order, with the public API at the top, so
the downward claim is checkable against the code.

`parallel` sits low in the map because it is self-contained worker
vocabulary: the `Parallelism` policy enum, the `Ptr` Send-pointer wrapper,
and the `JobCursor`. It depends only on `tuning`, and `kernel`, `driver`,
`special`, and `dispatch` all use it alike. `pack` sits below `kernel`
because the families' pack hooks build on the packing primitives.

Dependencies point strictly downward, with one annotated re-entry detailed
below the chart. `simd` depends only on `scalar` and `core`. The driver
never names a concrete element type or ISA:

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

The one exception to the downward rule is `special/batched.rs` (L6), which
re-enters `dispatch::execute` (L7) once per batch element. So each element
inherits the same driver, small-k, small-mn, or gemv routing that a
standalone `gemm` call on its shape would take. It does not need a second
dispatch ladder maintained by hand.

| Layer | Path | Responsibility |
|---|---|---|
| L8a | `gemmkit/src/api.rs` + `api/` | Public entries per family (`batched`, `cplx`, `fused`, `int8`, `map`, `packed`), validation, lowering to dispatch tasks |
| L7 | `gemmkit/src/dispatch.rs` + `dispatch/` | Per-type `OnceLock<fn>` selection ladders (`float`, `mixed`, `int`, `complex`, `isa`), orientation normalization, special-path gates |
| L6 | `gemmkit/src/special.rs` + `special/` | `gemv`, `small_k`, `small_mn`, `batched` orchestration (`batched` re-enters L7 `dispatch::execute` per element) |
| L5 | `gemmkit/src/driver.rs` | The blocked loop nest, packing decisions, prepacked-RHS consumption |
| L4 | `gemmkit/src/kernel.rs` + `kernel/` | `KernelFamily` trait, the families, `Epilogue` trait and built-ins |
| L3 | `gemmkit/src/cache.rs` + `cache/` | Cache detection (`cpuid`, `sysfs`, `sysctl`), `blocking()` model |
| L2 | `gemmkit/src/parallel.rs` | `Parallelism`, worker ramps, `JobCursor`, rayon integration |
| L1 | `gemmkit/src/pack.rs` | `pack_panels` and the k-group-interleaved `pack_kgroup_panels` |
| L0 | `gemmkit/src/simd.rs` + `simd/`, `gemmkit/src/scalar.rs` | `Simd` tokens, `SimdOps`/`KernelSimd`, `Scalar`/`Float`/`NarrowFloat`/`ComplexFloat` |
| - | `gemmkit/src/tuning.rs`, `gemmkit/src/workspace.rs` | Threshold knobs, packing-buffer pool |

## Life of a gemm call

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

fn main() {
    // 2x3 * 3x2 = 2x2, all row-major
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0_f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 3, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}
```

1. **Validate** (`gemm_with` in `gemmkit/src/api.rs`). The call checks that
   the shapes agree and that every view stays inside its slice. It also
   checks that `C` addresses each `(i, j)` uniquely, without overlapping `A`
   or `B`. The views then lower to a `Task<T>` of raw pointers and strides.
2. **Dispatch** (`dispatch::execute` in `gemmkit/src/dispatch.rs`).
   Degenerate cases exit early. `m == 0 || n == 0` returns right away.
   `k == 0 || alpha == 0` becomes a `C <- beta*C` scale that never reads
   `A` or `B`. Otherwise the memoized per-type function pointer runs (see
   the next section).
3. **Route** (`run_typed` in `gemmkit/src/dispatch/float.rs`). A gemv shape
   (`m == 1 || n == 1`) goes straight to `special/gemv.rs`, in the caller's
   own frame. Every other shape first goes through orientation
   normalization (`orient_transpose`). If `C` is row-major-ish, the driver
   computes `C^T = B^T*A^T` instead, so the output columns stay contiguous.
   From there the shape is gated to `special/small_mn.rs` or
   `special/small_k.rs`, or else falls through to the general driver.
4. **Block, loop, pack** (`driver::run_inner` in `gemmkit/src/driver.rs`).
   `(MC, KC, NC)` come from `cache::topology().blocking(...)`, sized in
   packed-input elements. The BLIS-order nest is `jc` (columns, at the L3
   scale), then `pc` (depth, never parallel). Inside that, a flat 1-D job
   list covers every `(ic` row-block `x jt` column-tile`)` pair, and workers
   drain it from a shared `JobCursor`. `beta` applies only on the first
   depth slice. Later slices accumulate instead.

   Packing is adaptive. The driver packs `B` once per depth slice, in
   parallel behind a fork-join barrier, when `m` clears
   `rhs_pack_threshold`. It packs `A` per worker, or once per row block
   through a shared pre-pass on large parallel problems. Where reuse is too
   low to pay for the copy, it reads `A` in place instead.

   A shape's working set is its `A + B + C` bytes. When that set outgrows
   the per-core-reachable LLC, the driver also prefetches each output
   microtile just ahead of its microkernel call, as a T0 hint. This hides
   the out-of-cache read-modify-write latency. The hint runs only on
   `x86_64`, and is a no-op elsewhere, so aarch64 and wasm are untouched. It
   moves cache lines only and never changes arithmetic, so results stay
   bit-identical whether the hint is on or off. `GEMMKIT_PREFETCH_MIN_BYTES`
   gates it, where `0` means auto, the per-core-reachable LLC.
5. **Microkernel** (`Fam::microkernel_epi`, for example
   `gemmkit/src/kernel/float.rs`). `SimdOps::accumulate_tile` accumulates
   one `MR x NR` tile in registers, in an ascending-`k` fused-multiply-add
   schedule. The alpha/beta epilogue then stores the tile. A full
   unit-stride tile stores its vectors directly. An edge tile or a strided
   tile drains through a stack scratch tile instead. Plain `gemm` runs the
   zero-cost `Identity` epilogue, which const-folds away. The fused entries
   thread a real `Epilogue` through the same path.

## ISA dispatch

Each element type has one `OnceLock<Dispatched<T>>` slot
(`gemmkit/src/dispatch/`). Feature detection runs once. Dispatch caches the
winning monomorphized entry points (plain, prepacked, fused) and the
microtile geometry there, so every later call is a plain indirect call.

The auto-selection ladder prefers AVX-512F, then FMA/AVX2, then scalar, on
x86. NEON is the baseline on aarch64. On wasm32, `simd128` is chosen at
compile time, since there is no runtime feature detection there. A wasm32
build must pass `-C target-feature=+simd128` to use it. Scalar is the
portable floor everywhere.

Runtime detection cannot pair with a fixed `#[target_feature]` attribute on
the generic kernel. So each ISA token's `Simd::vectorize` runs a closure
inside a small `#[target_feature]`-annotated trampoline instead
(`gemmkit/src/simd.rs`). The kernel and its `#[inline(always)]` primitives
inline into that trampoline, so every intrinsic lands in feature-enabled
codegen.

Tile geometry, the pair `(MR_REG, NR)`, is the only per-(type, ISA) knob.
The dispatch site chooses it as const generics. `MR` is `MR_REG * LANES`,
and `f64` halves the lane count. For `f32`: AVX-512F runs a 32x12 tile,
FMA runs 16x6, NEON runs 16x4, simd128 runs 8x4, and scalar runs 4x4.

`GEMMKIT_REQUIRE_ISA` pins the kernel end to end. Its values are `scalar`,
`fma`, `avx512f`, `avx512vnni` (the i8 dot kernel), `avx512bf16` (the bf16
dot kernel), `neon`, `simd128`, or `auto`. If the CPU or the target does
not support the requested ISA, dispatch panics instead of falling back. So
a CI job that means to exercise one kernel fails loudly if it cannot.
gemmkit reads and memoizes the value once.

## Element-type families

2 small traits carry the type variation. `Scalar` (`gemmkit/src/scalar.rs`,
L0) holds only the identity constants and the accumulator type `Acc`. `f32`
and `f64` accumulate in themselves, `f16` and `bf16` accumulate in `f32`,
`i8` accumulates in `i32`, and complex types accumulate in themselves. No
arithmetic lives on `Scalar`.

`KernelFamily` (`gemmkit/src/kernel.rs`, L4) bundles what distinguishes one
kind of GEMM: the `Lhs`, `Rhs`, `Acc`, and `Out` types, the pack layout, and
the microkernel. The driver is generic over the family and never branches on
element type. The families:

| Family | Types | Notes |
|---|---|---|
| `FloatGemm<T>` | `f32`, `f64` | The baseline, one generic microkernel for every ISA |
| `MixedGemm<N>` | `f16`, `bf16` in/out, `f32` acc | Widen on load, narrow on store via the `KernelSimd` seam |
| `Bf16DotGemm` | `bf16`, AVX-512 BF16 | `vdpbf16ps` dot kernel, `DEPTH_MULTIPLE = 2` |
| `MixedGemmF32<N>` / `Bf16DotGemmF32` | `f16`/`bf16` in, `f32` out | Deep-contraction twins (`OUT_IS_ACC = true`, multi-slice), see below |
| `IntGemm` / `IntGemmVnni` | `i8 -> i32` | Exact, wrapping, and a VNNI `vpdpbusd` dot kernel (`DEPTH_MULTIPLE = 4`) with `+128` signedness correction, bit-identical to the widen path |
| `IntGemmQ` / `IntGemmVnniQ` | `i8 -> i8`/`u8` | The requantizing variants (feature `epilogue`) |
| `ComplexGemm<T, CONJ_A, CONJ_B>` | `c32`, `c64` | Split (SoA) kernel, see below |

`KernelSimd<L, R, A, O>` (`gemmkit/src/simd.rs`) is the widen/narrow seam. A
blanket impl covers the homogeneous case, and mixed impls add widening
loads and narrowing stores, so mixed precision needs no branch in the
driver. A family whose output is narrower than its accumulator sets
`OUT_IS_ACC = false`. The driver then uses `kc = k`, one depth panel: the
whole contraction accumulates in `f32` and rounds to the narrow output
once.

At large `k`, that single panel streams an L2-overflowing RHS micropanel
(`nr * k * sizeof(N)`) from L3 or DRAM on every microtile call. Above an
engage gate (`GEMMKIT_DEEP_KC_BYTES`, auto-derived from half the L2 in
`cache::deep_k_engage_bytes`), the mixed dispatch instead runs an
**f32-output twin** family, `MixedGemmF32<N>` or `Bf16DotGemmF32`. The twin
sets `Out = f32 = Acc`, so `OUT_IS_ACC = true`.

The twin reuses the narrow pack and the widen-FMA or `vdpbf16ps` accumulate
step, but stores `f32`. This lets the driver's existing multi-slice
blocking apply unchanged, since each slice's panels stay L2-resident. The
twin accumulates into an `m x n` f32 scratch (a dedicated `Workspace`) with
`alpha = 1` and `beta = 0`. One vectorized sweep then narrows
`alpha*scratch + beta*C` back to `N`.

The twin seeds each slice's accumulators from that scratch. A third
`KernelSimd<N, N, f32, f32>` seam supplies the plain-f32 `C` load and
store. This continues the single panel's ascending-`k` chain, split at
slice boundaries by an exact f32 store and reload. For the common case
`beta in {0, 1}`, the deep-k route is byte-for-byte the single panel. For a
general `beta` it holds to tolerance instead.

The dot twin's interior slices round `kc` up to `DEPTH_MULTIPLE`, so a
k-pair never straddles a slice boundary. Shallow `k`, the fused-epilogue
path, and prepacked RHS all keep the single panel.

Dot-product families declare `DEPTH_MULTIPLE = Q` and pack through
`pack_kgroup_panels` (`gemmkit/src/pack.rs`), which interleaves `Q`
consecutive depth steps contiguously per lane. The ISA token's
`dot_accumulate` override then consumes whole instruction groups at once.
VNNI is bit-exact against the widen path, since its arithmetic is integer.
The bf16 dot kernel reshapes the accumulation rounding instead, and is held
to a tolerance, within the reproducibility contract.

Complex GEMM does not ride on `FloatGemm`. Its pack de-interleaves each
micropanel into planar real and imaginary planes, and applies conjugation
as a sign flip during packing, selected by const generics. The hot loop is
pure real FMAs through the `SimdOps::cplx_microkernel` seam: 4 fused
real steps per complex multiply-accumulate, with no in-loop shuffles. Both
operands are always packed, since only the planar layout is consumable.

## Blocking and the cache model

`gemmkit/src/cache.rs` computes `(MC, KC, NC)` analytically from the
detected cache geometry, using the BLIS model. `KC` is sized so the A and B
micropanels coexist in L1 without self-eviction. `MC` is sized so the A
macro-panel fits L2, with one way reserved for B. `NC` is sized so the B
macro-panel fits L3, with a panel-count cap on machines that report no L3.

A tiny-matrix shortcut skips the model for small shapes. Sizing uses the
packed-input element size, so narrower types get deeper blocks. `KC` and
`NC` are deliberately independent of the thread count. `MC` may shrink with
the worker count, to keep the parallel job list deep enough. It always
stays an `MR` multiple, though. So the microtile set, and each tile's
accumulation order, stay the same at any worker count.

Parallelism otherwise feeds the packing decisions. The LHS pack gate is
per-worker column reuse. The shared-A pre-pass engages only on the parallel
packed path, above a workload threshold.

Detection is best-effort, through a fallback chain that cannot fail. It
tries CPUID on x86 (`cache/cpuid.rs`), then Linux sysfs
(`cache/sysfs.rs`), then macOS sysctl (`cache/sysctl.rs`). If none of those
report a value, it falls back to a static default calibrated on a Zen5
part. `#[cfg]` only ever picks the sniffing method, never the values.
gemmkit filters out implausible reads, and memoizes the result once,
together with the OS page size, in `Machine`.

Every backend reports each cache level as reachable from one core. On a
multi-die part, the L3 figure is the local complex's slice, 32 MiB on a
2-CCD 9950X. It is never the package total. No single core reaches the
whole package's L3 as one cache.

`Level::shared_by` models per-worker contention for the data the driver
actually places at each level. It does not model raw hardware sharing. L1
and L3 are always `1`. Only L2 uses the physical-core sharing degree.

## Packing and workspaces

`pack_panels` (`gemmkit/src/pack.rs`) is the one micropanel copy that both
operands share. An LHS panel is `mr` rows tall, stored column by column. An
RHS panel is `nr` columns wide, stored row by row, using the same routine
with the strides swapped. Both zero-fill their tails. A contiguous leading
dimension takes a straight copy. A strided source takes a cache-blocked
transpose instead, and writes the same bytes either way.

Prepacked operands (`gemmkit/src/api/packed.rs`) serve the fixed-weight
loop. `prepack_rhs` and `prepack_lhs` pack a whole operand once into a
`PackedRhs<T>` or `PackedLhs<T>`, recording the blocking geometry (`nr`,
`kc`, `nc`) it was built for. `gemm_packed_b` and `gemm_packed_a` (and
their fused twins) read that geometry back verbatim, so panels always
match their tiling. `gemm_packed_a` consumes through the transposed
problem.

The layout comes from `driver::pack_rhs_full`, the same code the per-call
pack uses. So a prepacked GEMM reproduces a plain one under the same
config. The buffer is read-only, and workers share it with no
synchronization.

The `int8` feature adds a heterogeneous twin, `prepack_rhs_i8` and
`gemm_i8_packed_b`, that builds a `PackedRhs<i8>`. It is bit-identical to
plain `gemm_i8`, since integer accumulation is exact. Its layout is pinned
to whichever integer kernel the memoized dispatch chose, so the consuming
call always runs that same family. This deliberately bypasses the dynamic
small-parallel widen fallback, since a `vpdpbusd` buffer is k-quad-
interleaved and the widen kernel cannot consume it.

For that VNNI dot kernel, the RHS pack is otherwise mandatory on every
call, so prepacking wins the most there. `prepack_rhs_i8` rounds the
buffer depth up to `DEPTH_MULTIPLE = 4`. It then packs the whole
contraction as one depth slice, using the driver's single-slice guard for a
depth-padded family.

`Workspace` (`gemmkit/src/workspace.rs`) is a growable, 64-byte-aligned
scratch buffer. `Workspace::regions` carves out per-worker (or
per-row-block) LHS regions, plus one shared RHS region. It applies
fail-closed overflow checks at the element-to-byte chokepoint. A broadcast
stride can present logical dimensions near `isize::MAX`. A wrapped size
there would under-allocate the buffer, and the pack would then write past
it.

A re-entrancy-safe thread-local pool supplies the default workspace. The
`*_with` entries thread a caller-owned workspace through instead, so there
is zero heap allocation after the first call large enough to need one.
Without `std`, there is no pool, and each call uses a fresh workspace.

## Parallel execution

`Parallelism` is either `Serial` or `Rayon(n)`, where `Rayon(0)` means auto.
Resolution (`gemmkit/src/parallel.rs`) is workload-aware. Below a
total-work gate, everything stays serial. An explicit worker count is
honored, capped by the core count and the available jobs. The auto count
instead scales with the total work: one worker per
`GEMMKIT_PAR_MNK_PER_WORKER` block of `m*n*k` (default 2_000_000). It is
capped by cores and jobs, and floored at 1. This rule is work-based rather
than dimension-based. Total work predicts the optimal worker count better
than any single dimension does.

Bandwidth-bound shapes (gemv and gevv) use a different rule. They stay
serial below a cache-derived byte floor, the per-core private L2, above
which the matrix spills to the shared L3. Above that floor they jump
straight to a width sized for those bytes. A few workers is the worst
point on a bandwidth scaling curve, so there is no gain in stopping there.
That width is a ladder over the same pool tiers described below. It climbs
one tier per
`GEMMKIT_GEMV_TIER_STEP` (default auto, 8) factor of touched bytes above
the floor, and stops at the largest tier. It never reaches the full
machine width, since a gemv saturates its bandwidth well before that.
`GEMMKIT_GEMV_THREAD_CAP` replaces the whole ladder with one flat width
instead.

Rayon's fork/join overhead scales with a pool's idle slack, its thread
count minus its engaged workers. So a small parallel GEMM that forks into
the full-width global pool pays for threads it never uses.

On x86_64 and aarch64, gemmkit also keeps private, persistent rayon pools,
at halving tiers of the machine width. `GEMMKIT_POOL_CLASSES` caps how
many: default 2 on x86_64, 1 on aarch64, clamped to 3, and 0 (disabled) on
every other target. gemmkit builds each pool lazily on first use, and
never rebuilds it.

`GEMMKIT_FULL_WIDTH_MNK` is the work-size gate at which the auto path
leaves its largest tier pool for the full machine width. Its default is
auto and arch-split: 110_000_000 on x86_64, 14_000_000 on aarch64. The
auto path snaps its worker count exactly onto a tier, and stays on its
largest tier below the gate. `MAX` pins auto to the largest tier
unconditionally.

A call already running inside a rayon pool is never redirected. An
explicit `Rayon(n)` keeps exactly `n` workers, and only picks the smallest
tier pool that fits. Threaded wasm's dedicated pool is unaffected by any of
this.

Work distribution is demand-driven. The driver flattens its inner work
into a 1-D job list, and workers pull contiguous chunks from a shared
lock-free `JobCursor`. This lets faster cores on heterogeneous parts absorb
proportionally more work. The chunk grain oversamples the worker count.
The packed-LHS path uses a row-block-aligned grain instead, so chunks
never straddle a pack boundary.

On wasm32, rayon is usable only under the `wasm_threads` feature, which
targets `wasm32-wasip1-threads` and sizes a dedicated pool from the
`GEMMKIT_WASM_THREADS` knob. Without that feature, `parallel` degrades to
the serial loop instead of trapping.

To restate the mechanism concretely: `kc` and `nc`, and the fixed
depth-panel order, do not change with the worker count. So each output
element accumulates in one fixed order, regardless of how many workers run
the call. A wider worker count can shrink `mc`, to keep the job list deep
enough. `mc` always stays an `mr` multiple, though, so the microtile set
does not change either. Each output tile is computed whole by one worker
over the full depth, and packed bytes do not depend on who packs them.

Which worker computes a given tile varies from run to run. The bits of the
result do not, on the driver path, for a fixed input and a fixed worker
count. gemmkit's actual promise stays the narrower one from Goals and
constraints: reproducible output for a fixed machine and a fixed
configuration, not a cross-configuration guarantee.

## Special paths

Dispatch reroutes shapes that the register-tiling driver fits poorly
(`gemmkit/src/special/`). All of them sit behind the same public entries,
and all are tunable:

- **gemv** (`gemv.rs`): triggers on `m == 1 || n == 1`. It is
  memory-bound. Output rows are partitioned across workers with no split
  reductions, so the result is bit-identical across worker counts.

  2 bit-identical axpy strategies exist, a register-blocked output form
  and a plain column-outer form, chosen by output cache residency. The axpy
  forms vectorize over output rows, and the dot form vectorizes over `k`.
  So a sweep with fewer rows than one SIMD register yields to the dot form
  wherever that form is also legal. That case is the single row of a pure
  dot product (`m == n == 1`), whose strides are both 1, so it fits either
  classification.

  Whether the rows split at all is decided separately. For a column-major
  matrix, the output-row axis is the inner memory axis. So a split trades
  the serial route's one sequential pass for a strided walk per worker.
  Below `gemv_axpy_par_min_rows` output rows, that trade loses, so the
  sweep stays serial. A row-major matrix hands each worker whole
  `k`-contiguous rows instead, so it is never gated this way. Neither is
  the mixed twin, since splitting helps it at every size.

  The mixed-precision (`f16`/`bf16`) twin `run_mixed` reuses the same
  partition. It widens each load to `f32` through the `KernelSimd` seam,
  accumulates in `f32`, and rounds to the narrow type once at the store.
  This applies only to the register-blocked axpy, since the narrow output
  must round exactly once. The mixed *fused* gemv deliberately stays on the
  driver instead, since the driver already rounds once after the epilogue.
  So only the plain mixed path routes through `gemv.rs`.
- **small-k** (`small_k.rs`): triggers when `k` is at or below
  `small_k_threshold`. The whole product runs as one depth panel over the
  family's microkernel, reading `A` and `B` in place with no packing or
  blocking setup. It is generic over families.
- **small-m,n** (`small_mn.rs`): triggers when both `m` and `n` are at or
  below `small_mn_dim`, with a long contraction. Each output is a single
  horizontal SIMD dot, since the driver would spend most of its time on
  microtile padding here.

  When both operands stream unit-stride along `k` (row-major `A`,
  column-major `B`), the dots read `A` and `B` in place. One operand can be
  strided instead, in an all-row-major or all-column-major shape, above
  `small_mn_pack_min_k`. Then a shared pre-pack step copies only the
  strided operand into a padded, `k`-contiguous scratch buffer. That copy
  costs roughly `1/m` or `1/n` of the total work, still cheaper than
  falling through to the driver. The same kernel then runs over the packed
  operand. The pack is a pure reorder, so the packed route is bit-identical
  to the already-eligible layout.
- **batched** (`batched.rs`): `gemm_batched*` orchestrates the single-GEMM
  engine over a batch. `Parallelism::resolve_batch` picks among 3
  policies:
  - assign whole cache-hot GEMMs to workers
  - run a sequential loop, giving each large element the engine's full
    parallelism (only for `m, n > 1` shapes, whose routes ignore the
    worker count)
  - run serial

  The pointer-array form, `gemm_batched_ptr_unchecked` over a slice of
  `GemmProblem`s, allows per-element shapes.

## Epilogue fusion

The `epilogue` feature fuses a per-element transform into the microkernel's
store instead of a second pass over `C`. The seam is the `Epilogue` trait
(`gemmkit/src/kernel/epilogue.rs`), threaded through
`KernelFamily::microkernel_epi`:

- **Zero-cost identity**: plain `gemm` passes `Identity`, whose hooks
  const-fold away (`IS_IDENTITY`), so the non-fused kernel is unchanged.
- **Fire-once semantics**: the driver passes `last_k` and the epilogue
  applies only on the final depth panel. Earlier panels store raw
  accumulator partials (`OUT_IS_ACC = false` families have a single panel
  by construction).
- **Whole-tile application**: the vector path hands the entire
  `MR_REG x NR` register tile to `apply_tile`, not one register at a time.
  The kernel's store pass is unrolled by the tile const generics. So a
  per-register hook replicates whatever the epilogue branches on, once per
  accumulator, and that branch web costs the compiler the accumulator tile
  itself. `apply_tile` defaults to a loop over the per-register
  `apply_reg`. An epilogue with a runtime discriminant overrides it
  instead, and decodes once.
- **Built-ins**:
  - `FusedEpi` (per-row or per-col bias, ReLU or LeakyReLU) behind
    `gemm_fused*`, its batched and prepacked variants, and the bias-only
    `gemm_cplx_fused*`
  - `MapEpi` (a user per-element closure, `f32`/`f64`) behind `gemm_map*`
  - `KRequantize` (`i32` accumulator to quantized `i8`/`u8`, with a
    per-tensor or per-row scale, a zero point, an optional `i32` bias, and
    round-half-to-even) behind `gemm_i8_requant*`

The correctness contract has 3 parts. A fused call routes every shape
through the same kernel that plain `gemm` would use (driver, gemv, small-k,
small-m,n), each with its fused version. The engine itself stays
epilogue-independent. The vector and scalar apply paths must agree bit for
bit. So for `f32` and `f64`, `gemm_fused` equals `gemm()` followed by the
same scalar map, bitwise, for every shape.

The documented exception is `f16` and `bf16`. Their epilogue applies in
`f32` before the single narrowing step, which is more precise than
narrowing first and then mapping. So those entries are deliberately not
bitwise-equal to gemm-then-map. Reproducibility is unaffected. The
requantize vector store is proven bit-equal to its scalar map, per lane.

## Extension points

- **A new ISA backend** needs:
  - a zero-sized token with a `Simd::vectorize` trampoline
  - `SimdOps<T>` impls for the element types it accelerates
  - a `Dispatched` descriptor with its tile geometry
  - one arm per `select_*` ladder, plus a `GEMMKIT_REQUIRE_ISA` name

  The driver, the families, packing, and blocking stay untouched.
- **A new element type** needs a `Scalar` impl that chooses its `Acc`. It
  also needs a `KernelFamily` (or reuse of one through the `KernelSimd`
  widen/narrow seam), plus a dispatch module with its own `OnceLock` slot.
  The open/closed property is enforced by `gemmkit/tests/open_closed.rs`,
  which drives the driver with a second trivial family.
- **A dot-product instruction** (VNNI-style) arrives as a family with
  `DEPTH_MULTIPLE > 1`, plus a `KernelSimd::dot_accumulate` override on the
  capable token. `accumulate_tile` overrides stay reserved for scheduling
  changes that keep the same rounding shape.
- **A new fused transform** is an `Epilogue` impl. Its vector and scalar
  paths must agree bitwise. An impl that dispatches on a runtime
  discriminant must also override `apply_tile`, so the decode gets hoisted
  out of the unrolled store pass.

## Tuning knobs

Every heuristic threshold lives in `gemmkit/src/tuning.rs`. Each one
resolves in this order: a per-call argument, then a programmatic setter
(`tuning::set_*`), then an environment variable (`GEMMKIT_*`), then a
compiled default. The compiled default is calibrated on a Zen5 x86 part,
with a different default on aarch64 for some knobs. gemmkit reads and
caches each env var once. A malformed value warns on stderr and falls back
to the default instead of panicking.

The knobs cover the serial/parallel gate, pack gates and strides,
special-path thresholds, scheduler grains, and blocking caps.
`tuning::knob_env_names` enumerates the full set of `GEMMKIT_*` names once,
as a `#[doc(hidden)]`, zero-cost, `no_std` registry. The out-of-crate knob
consumers (the `gemmkit-tune` sweep table, `tests/props_knobs.rs`, the
fuzz `KNOB_SETTERS`) each assert their own hand-maintained list against
that registry. So a new knob cannot silently escape their coverage.

`gemmkit-tune` (`gemmkit-tune/src/main.rs`) automates host calibration. Run
on the deploy machine, it sweeps each knob independently over a set of
probe shapes. It scores each candidate by geometric-mean throughput, with
a tie-break that favors the default and accounts for measurement noise. It
then writes a `gemmkit-tune.env` profile of `export GEMMKIT_*=...` lines,
for the deployer to source before running. No recompile is involved.

## Testing strategy

The test suites live in `gemmkit/tests/`. The performance harnesses
(`tests/perf/` and `gemmkit/benches/`) are measurement tools, not CI gates.
`tests/perf/` is the exhaustive internal investigation suite, made of
`#[ignore]` tests over a median-of-9 harness. `benches/gemm_bench.rs` is the
curated public `cargo bench` surface instead, built on criterion and
grouped into `sgemm`, `dtypes`, `gemv`, `prepacked`, and `batched`, for
`--save-baseline` regression tracking.

- **Correctness** (`tests/correctness/`): sweeps shapes, layouts, and
  alpha/beta values against an independent `f64` reference GEMM
  (`tests/oracle_common/`). It checks per-type accuracy gates, and
  cross-checks against the external `gemm` crate. It also checks parallel
  bit-identity where the contract promises it, per-ISA kernel runs, and the
  safe API's exact panic wording.
- **Property tests** (`tests/props_api.rs`, `props_packed.rs`,
  `props_knobs.rs`): proptest over shapes, strides, and knob values.
- **Conformance** (`tests/simd_conformance.rs`, plus in-module sweeps like
  `requant_store` in `gemmkit/src/simd.rs`): every available token checked
  against scalar models.
- **Fuzzing** (`gemmkit/fuzz/`): 6 libFuzzer targets (gemm, batched,
  prepack, prepack-i8, API validation, knobs) in a nightly-only
  sub-workspace.
- **Knob and env surface** (`tests/tuning.rs`, `tests/env.rs`,
  `tests/props_knobs.rs`, `tests/deep_k_narrow.rs`): tests that mutate the
  process-global tuning knobs or the `GEMMKIT_*` environment live in their
  own binaries, since a separate process cannot race another's knob state.
  Each binary serializes its own mutations under a per-binary `KNOB_LOCK`.
- **ISA pins** (`tests/env_isa_*.rs`): one binary per
  `GEMMKIT_REQUIRE_ISA` value (`avx512f`, `vnni`, `bf16`, `scalar`, `neon`,
  `wasm`), plus a garbage-value guard. Each binary pins its ISA once,
  through a shared `Once` (`tests/env_isa_common/`), before any dispatch
  runs. So the memoized per-ISA dispatch resolves the pinned kernel in its
  own isolated process. The write overrides any inherited pin, so the SDE
  and pinned CI jobs still exercise these routes.
- **Adapter re-export surface**
  (`gemmkit-{ndarray,nalgebra,faer}/tests/reexports.rs`): one binary per
  adapter that may not name a `gemmkit::` path. It reaches every gemmkit
  item through the adapter instead, the way a downstream crate would. It
  writes the same generic wrapper a caller would write over each entry
  family. So a gemmkit type that reaches a public signature without being
  re-exported fails to compile there. Every other adapter test binary
  imports from `gemmkit::` directly, so it cannot catch an incomplete
  adapter surface.
- **Miri**: CI runs the scalar-path correctness suite and the complex
  negative-stride entry under Miri. `cfg(miri)` detours exist only where
  Miri cannot interpret a hardware conversion.
- **ISA pinning in CI** (`.github/workflows/ci.yml`): jobs pin each kernel
  through `GEMMKIT_REQUIRE_ISA` (AVX-512F, VNNI, and BF16 under Intel SDE,
  NEON on aarch64, simd128 on wasm). Other jobs cover no_std builds, an
  MSRV job, and feature-matrix builds. `GEMMKIT_FAST_TEST` is a
  test-suite-only switch that shrinks the sweeps. The library itself never
  reads it.
