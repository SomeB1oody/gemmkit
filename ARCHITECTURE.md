# Architecture

This document describes the design of the gemmkit workspace as it exists in
the code: the layer structure of the core engine, the path a call takes from
public API to microkernel, and the seams that make new instruction sets and
element types cheap to add. File references are repo-relative paths; the API
reference lives at [docs.rs/gemmkit](https://docs.rs/gemmkit).

## Goals and constraints

gemmkit computes `C <- alpha*A*B + beta*C` over `&[T]` + stride views (or raw
pointers with `isize` strides), selecting the best available instruction set
at runtime. Edition 2024, `rust-version` 1.89, licensed MIT OR Apache-2.0
(`LICENSE-MIT`, `LICENSE-APACHE` at the repo root). The design targets:

- **Safety at the boundary.** The checked entries (`gemm`, `gemm_fused`, ...)
  panic on shape mismatch, out-of-bounds strides, a self-aliasing output, or
  `C` overlapping `A`/`B` before any unsafe work runs (`validate_gemm_views`
  in `gemmkit/src/api.rs`). The `*_unchecked` tier exposes the raw engine
  (negative strides allowed) for callers that validate their own inputs, such
  as the adapters.
- **Reproducible, not bitwise, parallel results.** For a fixed input,
  environment, and configuration the output is identical regardless of the
  worker count: blocking is thread-count independent and every output element
  is reduced start-to-finish by one worker. Bitwise serial-vs-parallel
  identity happens to hold on the driver paths today, but the promised
  contract is reproducibility under a fixed config.
- **No macros, no `transmute` in the variation points.** ISA, element type,
  and operation family are traits; dispatch slots are typed function pointers
  in `OnceLock`s; tile geometry is a pair of const generics.
- **`no_std` and zero mandatory dependencies.** With default features off the
  crate is `#![no_std]`, needs only `core` + `alloc`, and depends on nothing:
  compile-time target features replace runtime CPU detection, env knobs are
  off, and a per-call workspace replaces the thread-local pool. Optional
  features pull one crate each: `std` -> `raw-cpuid` (x86 only), `parallel`
  -> `rayon`, `half` -> `half`, `complex` -> `num-complex`; `int8` and
  `epilogue` add none.

## Workspace layout

Five crates release in lockstep at version 0.1.0 (not yet published to
crates.io); the fuzz crate is a separate nightly-only root.

| Path | Crate | Role |
|---|---|---|
| `gemmkit/` | [gemmkit](https://crates.io/crates/gemmkit) | The core GEMM engine (everything below) |
| `gemmkit-ndarray/` | [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) | Zero-copy adapter over `ndarray` (>= 0.17.1) views |
| `gemmkit-nalgebra/` | [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra) | Zero-copy adapter over `nalgebra` 0.35 matrices |
| `gemmkit-faer/` | [gemmkit-faer](https://crates.io/crates/gemmkit-faer) | Zero-copy adapter over `faer` 0.24 matrices |
| `gemmkit-tune/` | [gemmkit-tune](https://crates.io/crates/gemmkit-tune) | Install-time autotuner binary (emits a `GEMMKIT_*` env profile) |
| `gemmkit/fuzz/` | gemmkit-fuzz | cargo-fuzz targets; its own workspace root, excluded from the stable workspace |

The adapters pull matrix pointers and strides straight out of each library's
native views (C-order, F-order, general and reversed strides, no copies) and
forward to the `*_unchecked` engine. Batched GEMM is exposed in the shape each
library's types allow: the ndarray adapter's 3-D strided `gemm_batched`
(batch on axis 0, over the strided-batched engine), and the nalgebra/faer
`gemm_batched` over a slice of per-element `(A, B)` inputs paired with a slice
of `&mut C` outputs (over the pointer-array `gemm_batched_ptr_unchecked`
engine, since neither library has a rank-3 type). Each adapter feature
(default `parallel`, plus `wasm_threads`, `half`, `complex`, `int8`,
`epilogue`) forwards to the same-named gemmkit feature.

## Layer map

Module docs in the core crate carry explicit layer labels; this is the stack
they declare, public API at the top. Dependencies point strictly downward
(`simd` depends only on `scalar` and `core`; the driver never names a
concrete element type or ISA):

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

| Layer | Path | Responsibility |
|---|---|---|
| L8a | `gemmkit/src/api.rs` + `api/` | Public entries per family (`batched`, `cplx`, `fused`, `int8`, `map`, `packed`); validation; lowering to dispatch tasks |
| L7 | `gemmkit/src/dispatch.rs` + `dispatch/` | Per-type `OnceLock<fn>` selection ladders (`float`, `mixed`, `int`, `complex`, `isa`); orientation normalization; special-path gates |
| L6 | `gemmkit/src/special.rs` + `special/` | `gemv`, `small_k`, `small_mn`, `batched` orchestration |
| L5 | `gemmkit/src/parallel.rs` | `Parallelism`, worker ramps, `JobCursor`, rayon integration |
| L4 | `gemmkit/src/driver.rs` | The blocked loop nest, packing decisions, prepacked-RHS consumption |
| L3 | `gemmkit/src/cache.rs` + `cache/` | Cache detection (`cpuid`, `sysfs`, `sysctl`), `blocking()` model |
| L2 | `gemmkit/src/pack.rs` | `pack_panels` and the k-group-interleaved `pack_kgroup_panels` |
| L1 | `gemmkit/src/kernel.rs` + `kernel/` | `KernelFamily` trait, the families, `Epilogue` trait and built-ins |
| L0 | `gemmkit/src/simd.rs` + `simd/`, `gemmkit/src/scalar.rs` | `Simd` tokens, `SimdOps`/`KernelSimd`, `Scalar`/`Float`/`NarrowFloat`/`ComplexFloat` |
| - | `gemmkit/src/tuning.rs`, `gemmkit/src/workspace.rs` | Threshold knobs; packing-buffer pool |

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

1. **Validate** (`gemm_with` in `gemmkit/src/api.rs`): shapes agree, every
   view stays inside its slice, `C` addresses each `(i, j)` uniquely and does
   not overlap `A`/`B`; the views lower to a `Task<T>` of raw pointers and
   strides.
2. **Dispatch** (`dispatch::execute` in `gemmkit/src/dispatch.rs`): degenerate
   cases exit early (`m == 0 || n == 0` returns; `k == 0 || alpha == 0`
   becomes a `C <- beta*C` scale that never reads `A`/`B`), then the memoized
   per-type function pointer runs (next section).
3. **Route** (`run_typed` in `gemmkit/src/dispatch/float.rs`): a gemv shape
   (`m == 1 || n == 1`) goes to `special/gemv.rs` first, in the user frame.
   Everything else is orientation-normalized (`orient_transpose`: if `C` is
   row-major-ish, compute `C^T = B^T*A^T` so output columns are contiguous),
   then gated to `special/small_mn.rs` or `special/small_k.rs`, else the
   general driver.
4. **Block, loop, pack** (`driver::run_inner` in `gemmkit/src/driver.rs`):
   `(MC, KC, NC)` come from `cache::topology().blocking(...)`, sized in
   packed-input elements. The BLIS-order nest is `jc` (columns, L3) -> `pc`
   (depth; never parallel) -> a flat 1-D job list over `(ic row-block x jt
   column-tile)` that workers drain from a shared `JobCursor`. `beta` applies
   only on the first depth slice; later slices accumulate. Packing is
   adaptive: `B` is packed once per depth slice (in parallel, behind a
   fork-join barrier) when `m` clears `rhs_pack_threshold`; `A` is packed per
   worker, or once per row block by a shared pre-pass on large parallel
   problems, or read in place when reuse is too low to amortize the copy.
5. **Microkernel** (`Fam::microkernel_epi`, e.g. `gemmkit/src/kernel/float.rs`):
   one `MR x NR` tile is accumulated in registers by
   `SimdOps::accumulate_tile` (an ascending-`k` fused-multiply-add schedule),
   then the alpha/beta epilogue stores it: full unit-stride tiles store
   vectors directly, edge or strided tiles drain through a stack scratch tile.
   Plain `gemm` runs the zero-cost `Identity` epilogue, which const-folds
   away; the fused entries thread a real `Epilogue` through the same path.

## ISA dispatch

Each element type has one `OnceLock<Dispatched<T>>` slot
(`gemmkit/src/dispatch/`): feature detection runs once, the winning
monomorphized entry points (plain, prepacked, fused) plus the microtile
geometry are cached, and every later call is a plain indirect call. The
auto-selection ladders prefer AVX-512 then FMA/AVX2 then scalar on x86; NEON
is baseline on aarch64; `simd128` is chosen at compile time on wasm32 (no
runtime feature detection there, so the build must pass
`-C target-feature=+simd128`); scalar is the portable floor everywhere.

Runtime detection cannot pair with a fixed `#[target_feature]` on the generic
kernel, so each ISA token's `Simd::vectorize` runs a closure inside a small
`#[target_feature]`-annotated trampoline (`gemmkit/src/simd.rs`); the kernel
and its `#[inline(always)]` primitives inline into that context, landing every
intrinsic in feature-enabled codegen.

Tile geometry `(MR_REG, NR)` is the only per-(type, ISA) knob, chosen at the
dispatch site as const generics: for `f32`, AVX-512 runs 32x12, FMA 16x6, NEON
16x4, simd128 8x4, scalar 4x4 (`MR = MR_REG * LANES`; `f64` halves the lane
count).

`GEMMKIT_REQUIRE_ISA` pins the kernel end to end: `scalar`, `fma`, `avx512`,
`avx512vnni` (the i8 dot kernel), `avx512bf16` (the bf16 dot kernel), `neon`,
`simd128`, or `auto`. If the CPU or target does not support the request,
dispatch panics rather than falling back, so a CI job that means to exercise
a given kernel fails loudly. The value is read once and memoized.

## Element-type families

Two small traits carry the type variation. `Scalar`
(`gemmkit/src/scalar.rs`, L0) holds only the identity constants and the
accumulator type `Acc` (`f32`/`f64` accumulate in themselves, `f16`/`bf16` in
`f32`, `i8` in `i32`, complex in itself); no arithmetic lives on it.
`KernelFamily` (`gemmkit/src/kernel.rs`, L1) bundles what distinguishes one
kind of GEMM: the `Lhs`/`Rhs`/`Acc`/`Out` types, the pack layout, and the
microkernel; the driver is generic over the family and never branches on
element type. The families:

| Family | Types | Notes |
|---|---|---|
| `FloatGemm<T>` | `f32`, `f64` | The baseline; one generic microkernel for every ISA |
| `MixedGemm<N>` | `f16`, `bf16` in/out, `f32` acc | Widen on load, narrow on store via the `KernelSimd` seam |
| `Bf16DotGemm` | `bf16`, AVX-512 BF16 | `vdpbf16ps` dot kernel, `DEPTH_MULTIPLE = 2` |
| `MixedGemmF32<N>` / `Bf16DotGemmF32` | `f16`/`bf16` in, `f32` out | Deep-contraction twins (`OUT_IS_ACC = true`, multi-slice); see below |
| `IntGemm` / `IntGemmVnni` | `i8 -> i32` | Exact, wrapping; VNNI `vpdpbusd` dot kernel (`DEPTH_MULTIPLE = 4`) with `+128` signedness correction, bit-identical to the widen path |
| `IntGemmQ` / `IntGemmVnniQ` | `i8 -> i8`/`u8` | The requantizing variants (feature `epilogue`) |
| `ComplexGemm<T, CONJ_A, CONJ_B>` | `c32`, `c64` | Split (SoA) kernel; see below |

`KernelSimd<L, R, A, O>` (`gemmkit/src/simd.rs`) is the widen/narrow seam: a
blanket impl covers the homogeneous case, and mixed impls add widening loads
and narrowing stores, so mixed precision needs no branch in the driver. A
family whose output is narrower than its accumulator sets `OUT_IS_ACC = false`
and the driver uses `kc = k` (one depth panel): the whole contraction
accumulates in `f32` and rounds to the narrow output once.

At large `k` that single panel streams an L2-overflowing RHS micropanel
(`nr * k * sizeof(N)`) from L3/DRAM on every microtile call, so above an engage
gate (`GEMMKIT_DEEP_KC_BYTES`, auto-derived from half the L2 in
`cache::deep_k_engage_bytes`) the mixed dispatch instead runs an **f32-output
twin** family - `MixedGemmF32<N>` / `Bf16DotGemmF32`, `Out = f32 = Acc`, so
`OUT_IS_ACC = true`. The twin reuses the narrow pack and widen-FMA / `vdpbf16ps`
accumulate but stores `f32`, so the driver's existing multi-slice blocking
applies unchanged (each slice's panels L2-resident); it accumulates into an
`m x n` f32 scratch (a dedicated `Workspace`) with `alpha = 1`, `beta = 0`, then
one vectorized sweep narrows `alpha*scratch + beta*C` back to `N`. The twin
seeds each slice's accumulators from the scratch (a third `KernelSimd<N, N, f32,
f32>` seam supplies the plain-f32 C load/store), continuing the single panel's
ascending-`k` chain split at slice boundaries - an exact f32 store/reload - so
for the common `beta in {0, 1}` the deep-k route is byte-for-byte the single
panel; for a general `beta` it holds to tolerance. The dot twin's interior
slices round `kc` up to `DEPTH_MULTIPLE`, so a k-pair never straddles a boundary.
Shallow `k`, the fused-epilogue path, and prepacked RHS keep the single panel.

Dot-product families declare `DEPTH_MULTIPLE = Q` and pack via
`pack_kgroup_panels` (`gemmkit/src/pack.rs`), which interleaves `Q` consecutive
depth steps contiguously per lane; the ISA token's `dot_accumulate` override
consumes whole instruction groups. VNNI is bit-exact against the widen path
(integer arithmetic); the bf16 dot kernel reshapes the accumulation rounding
and is held to a tolerance, within the reproducibility contract.

Complex GEMM does not ride `FloatGemm`: its pack de-interleaves each
micropanel into planar real/imaginary planes (conjugation is a sign flip
applied during packing, selected by const generics), and the hot loop is pure
real FMAs through the `SimdOps::cplx_microkernel` seam, four fused real steps
per complex multiply-accumulate with no in-loop shuffles. Both operands are
always packed: only the planar layout is consumable.

## Blocking and the cache model

`gemmkit/src/cache.rs` computes `(MC, KC, NC)` analytically from the detected
cache geometry using the BLIS model: `KC` so the A and B micropanels coexist
in L1 without self-eviction, `MC` so the A macro-panel fits L2 (one way
reserved for B), `NC` so the B macro-panel fits L3 (with a panel-count cap on
L3-less machines). A tiny-matrix shortcut skips the model for small shapes,
and sizing uses the packed-input element size, so narrow types get deeper
blocks. Blocking is deliberately independent of the thread count: that is the
mechanism behind reproducible output. Parallelism instead feeds the packing
decisions: the LHS pack gate is per-worker column reuse, and the shared-A
pre-pass engages only on the parallel packed path above a workload threshold.

Detection is best-effort with a fallback chain that cannot fail: CPUID on x86
(`cache/cpuid.rs`) -> Linux sysfs (`cache/sysfs.rs`) -> macOS sysctl
(`cache/sysctl.rs`) -> a static default calibrated on a Zen5 part. `#[cfg]`
only ever picks the sniffing method, never the values; implausible reads are
filtered out, and results are memoized once (with the OS page size) in
`Machine`. `Level::shared_by` models per-worker contention for the data the
driver actually places at each level, not raw hardware sharing: L1 and L3 are
always 1, and only L2 uses the physical-core sharing degree.

## Packing and workspaces

`pack_panels` (`gemmkit/src/pack.rs`) is the one micropanel copy both operands
share: LHS panels `mr` rows tall stored column-by-column, RHS panels `nr`
columns wide stored row-by-row (the same routine with strides swapped), tails
zero-filled. A contiguous leading dimension takes a straight copy; a strided
source takes a cache-blocked transpose that writes identical bytes.

Prepacked operands (`gemmkit/src/api/packed.rs`) serve the fixed-weight loop:
`prepack_rhs`/`prepack_lhs` pack a whole operand once into a `PackedRhs<T>` /
`PackedLhs<T>` recording the blocking geometry (`nr`, `kc`, `nc`) it was built
for; `gemm_packed_b`/`gemm_packed_a` (and their fused twins) read that
geometry back verbatim so panels always match their tiling (`gemm_packed_a`
consumes through the transposed problem). The layout comes from
`driver::pack_rhs_full`, the same code the per-call pack uses, so a prepacked
GEMM reproduces a plain one under the same config; the buffer is read-only and
shared across workers with no synchronization. The `int8` feature adds the
heterogeneous twin `prepack_rhs_i8`/`gemm_i8_packed_b` (a `PackedRhs<i8>`, bit-
identical to plain `gemm_i8` since integer accumulation is exact): its layout is
pinned to whichever integer kernel the memoized dispatch chose, so the consume
call always runs that same family and deliberately bypasses the dynamic small-
parallel widen fallback (a `vpdpbusd` buffer is k-quad-interleaved and not
consumable by the widen kernel). For that VNNI dot kernel the RHS pack is
otherwise mandatory on every call, so prepacking is the bigger win there;
`prepack_rhs_i8` rounds the buffer depth up to `DEPTH_MULTIPLE = 4` and packs the
whole contraction as one depth slice (the driver's single-slice guard for a
depth-padded family).

`Workspace` (`gemmkit/src/workspace.rs`) is a growable 64-byte-aligned scratch
buffer. `Workspace::regions` carves per-worker (or per-row-block) LHS regions
plus one shared RHS region, with fail-closed overflow checks at the
element-to-byte chokepoint (a broadcast stride can present logical dimensions
near `isize::MAX`; a wrapped size would under-allocate the buffer the pack
then writes past). A re-entrancy-safe thread-local pool supplies the default
workspace; the `*_with` entries thread a caller-owned one through instead,
giving zero heap allocation after the first sufficiently large call. Without
`std` there is no pool and each call uses a fresh workspace.

## Parallel execution

`Parallelism` is `Serial` or `Rayon(n)` (`Rayon(0)` = auto). Resolution
(`gemmkit/src/parallel.rs`) is workload-aware: below a total-work gate
everything stays serial; an explicit count is honored (capped by cores and
available jobs); the auto count ramps with the linear problem size,
`cbrt(m*n*k)` divided by a core-count-derived stride, instead of jumping to
all cores. Bandwidth-bound shapes (gemv/gevv) use a different rule: serial
below an LLC-derived byte floor, then straight to a bandwidth cap, because a
few workers is the worst point on a bandwidth scaling curve.

Work distribution is demand-driven: the driver flattens its inner work into a
1-D job list and workers pull contiguous chunks from a shared lock-free
`JobCursor`, so faster cores on heterogeneous parts absorb proportionally
more. The chunk grain oversamples the worker count; the packed-LHS path uses a
row-block-aligned grain so chunks never straddle a pack boundary. On wasm32,
rayon is usable only under `wasm_threads` (targeting `wasm32-wasip1-threads`),
which sizes a dedicated pool from the `GEMMKIT_WASM_THREADS` knob; without it,
`parallel` degrades to the serial loop instead of trapping.

The reproducibility contract, concretely: blocking and the job list are
identical for every worker count, each output tile is computed whole by one
worker over the full depth, and packed bytes do not depend on who packs them.
Which worker computes a tile varies run to run; the result never does.

## Special paths

Dispatch reroutes shapes the register-tiling driver fits poorly
(`gemmkit/src/special/`), all behind the same public entries and all tunable:

- **gemv** (`gemv.rs`): `m == 1 || n == 1`. Memory-bound; output rows are
  partitioned across workers with no split reductions, so it is bit-identical
  across worker counts. Two bit-identical axpy strategies (register-blocked
  output vs plain column-outer) are chosen by output cache residency. The
  mixed-precision (`f16`/`bf16`) twin `run_mixed` reuses the same partition but
  widens each load to `f32` through the `KernelSimd` seam, accumulates in `f32`,
  and rounds to the narrow type once at the store (only the register-blocked
  axpy, since the narrow output must round exactly once); the mixed *fused*
  gemv deliberately stays on the driver (which already rounds once after the
  epilogue), so only the plain mixed path routes here.
- **small-k** (`small_k.rs`): `k` at or below `small_k_threshold`. The whole
  product is one depth panel over the family's microkernel, reading A/B in
  place with no packing or blocking setup; generic over families.
- **small-m,n** (`small_mn.rs`): both `m, n` at or below `small_mn_dim`, long
  contraction. Each output is a single horizontal SIMD dot; the driver would
  compute mostly microtile padding here. When both operands stream unit-stride
  along `k` (row-major A, col-major B) the dots read A/B in place; when one is
  strided (an all-row-major or all-col-major shape, `k > small_mn_pack_min_k`)
  a shared pre-pack copies only the failing operand into a padded `k`-contiguous
  scratch (a `~1/m` or `~1/n` tax that still beats the driver) and the same
  kernel runs over it. The pack is a pure reorder, so the packed route is
  bit-identical to the eligible layout.
- **batched** (`batched.rs`): `gemm_batched*` is orchestration over the
  single-GEMM engine. `Parallelism::resolve_batch` picks between assigning
  whole cache-hot GEMMs to workers, a sequential loop giving each large
  element the full engine parallelism (gated to `m, n > 1` shapes whose routes
  are worker-count independent), and serial. The pointer-array form
  (`gemm_batched_ptr_unchecked` over `GemmProblem`s) allows per-element shapes.

## Epilogue fusion

The `epilogue` feature fuses a per-element transform into the microkernel's
store instead of a second pass over `C`. The seam is the `Epilogue` trait
(`gemmkit/src/kernel/epilogue.rs`), threaded through
`KernelFamily::microkernel_epi`:

- **Zero-cost identity**: plain `gemm` passes `Identity`, whose hooks
  const-fold away (`IS_IDENTITY`), so the non-fused kernel is unchanged.
- **Fire-once semantics**: the driver passes `last_k` and the epilogue applies
  only on the final depth panel; earlier panels store raw accumulator partials
  (`OUT_IS_ACC = false` families have a single panel by construction).
- **Built-ins**: `FusedEpi` (per-row/per-col bias, ReLU / LeakyReLU) behind
  `gemm_fused*`, its batched/prepacked variants, and the bias-only
  `gemm_cplx_fused*`; `MapEpi` (a user per-element closure, `f32`/`f64`)
  behind `gemm_map*`; `KRequantize` (`i32` accumulator to quantized `i8`/`u8`:
  per-tensor or per-row scale, zero point, optional `i32` bias,
  round-half-to-even) behind `gemm_i8_requant*`.

The correctness contract: a fused call routes every shape through the same
kernel plain `gemm` would (driver, gemv, small-k, small-m,n, each fused), the
engine is epilogue-independent, and the vector and scalar apply paths must
agree bit-for-bit, so for `f32`/`f64` `gemm_fused` equals `gemm()` followed by
the same scalar map, bitwise, for every shape. The documented exception is
`f16`/`bf16`: the epilogue applies in `f32` before the single narrowing (more
precise than narrow-then-map), so those entries are deliberately not
bitwise-equal to gemm-then-map; reproducibility is unaffected. The requantize
vector store is proven bit-equal to its scalar map per lane.

## Extension points

- **A new ISA backend** is a zero-sized token with a `Simd::vectorize`
  trampoline, `SimdOps<T>` impls for the element types it accelerates, a
  `Dispatched` descriptor with its tile geometry, and one arm per `select_*`
  ladder (plus a `GEMMKIT_REQUIRE_ISA` name). Driver, families, packing, and
  blocking are untouched.
- **A new element type** is a `Scalar` impl (choosing its `Acc`), a
  `KernelFamily` (or reuse of one through the `KernelSimd` widen/narrow seam),
  and a dispatch module with its own `OnceLock` slot. The open/closed property
  is enforced by `gemmkit/tests/open_closed.rs`, which drives the driver with
  a second trivial family.
- **A dot-product instruction** (VNNI-style) arrives as a family with
  `DEPTH_MULTIPLE > 1` plus a `KernelSimd::dot_accumulate` override on the
  capable token; `accumulate_tile` overrides are reserved for scheduling
  changes that keep the rounding shape.
- **A new fused transform** is an `Epilogue` impl; the vector and scalar paths
  must agree bitwise.

## Tuning knobs

Every heuristic threshold lives in `gemmkit/src/tuning.rs` and resolves as:
per-call argument > programmatic setter (`tuning::set_*`) > environment
variable (`GEMMKIT_*`) > compiled default (calibrated on a Zen5 x86 part, with
arch-split defaults where aarch64 measured differently). Env vars are read
once and cached; malformed values warn on stderr and fall back rather than
panic. Knobs cover the serial/parallel gate, pack gates and strides,
special-path thresholds, scheduler grains, and blocking caps.

`gemmkit-tune` (`gemmkit-tune/src/main.rs`) automates host calibration: run on
the deploy machine, it sweeps each knob independently over a probe-shape set,
scores candidates by geometric-mean throughput with a default-biased,
noise-aware tie-break, and writes a `gemmkit-tune.env` profile of
`export GEMMKIT_*=...` lines to source before running; no recompile involved.

## Testing strategy

The suites live in `gemmkit/tests/`; the performance harnesses (`tests/perf/`
and `gemmkit/benches/`) are measurement tools, not CI gates. `tests/perf/` is
the exhaustive internal investigation suite (`#[ignore]` tests over a
median-of-9 harness); `benches/gemm_bench.rs` is the curated public `cargo
bench` surface (criterion, grouped `sgemm`/`dtypes`/`gemv`/`prepacked`/`batched`)
for `--save-baseline` regression tracking.

- **Correctness** (`tests/correctness/`): shapes x layouts x alpha/beta swept
  against an independent `f64` reference GEMM (`tests/oracle_common/`) with
  per-type accuracy gates, cross-checks against the external `gemm` crate,
  parallel bit-identity where promised, per-ISA kernel runs, and the safe
  API's exact panic wording.
- **Property tests** (`tests/props_api.rs`, `props_packed.rs`,
  `props_knobs.rs`): proptest over shapes, strides, and knob values.
- **Conformance** (`tests/simd_conformance.rs`, plus in-module sweeps like
  `requant_store` in `gemmkit/src/simd.rs`): every available token checked
  against scalar models.
- **Fuzzing** (`gemmkit/fuzz/`): five libFuzzer targets (gemm, batched,
  prepack, API validation, knobs) in a nightly-only sub-workspace.
- **Knob and env surface** (`tests/tuning.rs`, `tests/env.rs`,
  `tests/props_knobs.rs`, `tests/deep_k_narrow.rs`): the tests that mutate the
  process-global tuning knobs or `GEMMKIT_*` environment live in their own
  binaries (a separate process cannot race another's knob state) and serialize
  their mutations under a per-binary `KNOB_LOCK`.
- **ISA pins** (`tests/env_isa_*.rs`): one binary per `GEMMKIT_REQUIRE_ISA`
  value (`avx512`, `vnni`, `bf16`, `scalar`, `neon`, `wasm`, plus a
  garbage-value guard). Each pins its ISA once through a shared `Once`
  (`tests/env_isa_common/`) before any dispatch, so the memoized per-ISA
  dispatch resolves the pinned kernel in an isolated process; the write
  overrides an inherited pin, so the SDE/pinned CI jobs still exercise these
  routes.
- **Miri**: CI runs the scalar-path correctness suite and the complex
  negative-stride entry under Miri; `cfg(miri)` detours exist only where Miri
  cannot interpret hardware conversions.
- **ISA pinning in CI** (`.github/workflows/ci.yml`): jobs pin each kernel via
  `GEMMKIT_REQUIRE_ISA` (AVX-512, VNNI, and BF16 under Intel SDE, NEON on
  aarch64, simd128 on wasm), plus no_std builds, an MSRV job, and
  feature-matrix builds. `GEMMKIT_FAST_TEST` is a test-suite-only switch that
  shrinks the sweeps; the library never reads it.
