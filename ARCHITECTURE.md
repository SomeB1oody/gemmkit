# Architecture

This document is a tour of how gemmkit is built and *why*. The short version: the
combinatorial explosion of **{element type} × {ISA} × {tile} × {operation family}**
is dissolved in the type system — a thick SIMD trait, const generics, and an
operation-family seam — so the algorithm is written **once**, with **no macros**
and **no `transmute`**.

## Design principles

1. **The algorithm is written once.** Blocking, packing, parallelism, and cache
   modelling are all generic. There is exactly one five-loop driver, and each family's
   microkernel is a single generic function across all ISAs and tiles (the real
   `FloatGemm` one; the shared split-layout complex one).
2. **Variation points collapse onto traits.** ISA → [`SimdOps`]; element type →
   [`Scalar`]; operation family → [`KernelFamily`]. Tile geometry is a pair of
   const generics chosen at the dispatch site.
3. **Runtime detection + compile-time monomorphization.** CPU features are probed
   once at runtime; the chosen path is a fully monomorphized function. The only
   `unsafe` *codegen* boundary is a per-ISA `#[target_feature]` trampoline.
4. **Best-effort with a guaranteed floor.** Cache detection, feature detection,
   and packing are all adaptive but always have a correct fallback.

## Layer map

```
L0  scalar.rs        Scalar (+ Float) — the data-type seam
L0  simd/            Simd + SimdOps<T> — the ISA seam (tokens: ScalarTok/Fma/Avx512/Neon/Simd128)
L1  kernel/          KernelFamily — the operation-family seam; float microkernel
L2  pack.rs          micropanel packing primitive
L3  cache/           CacheTopology + analytical BLIS blocking (cpuid/sysctl/sysfs backends + fallback)
L4  driver.rs        the generic five-loop nest
L5  parallel.rs      Parallelism + 1-D job split + work-gate
L6  special/         bandwidth-bound special paths (gemv.rs matrix·vector, small_k.rs skinny/low-depth)
L7  dispatch.rs      OnceLock<fn> ISA selection + orientation + per-(type,ISA) entry
L8a api.rs           MatRef/MatMut safe API + unchecked raw engine
L8b gemmkit-ndarray  ArrayBase adapter + dot
        tuning.rs / workspace.rs    cross-cutting
```

Dependencies flow strictly upward. The `simd` module depends only on `scalar` and
`core` — no reverse dependency on anything above it — so it could be split into its
own crate unchanged.

## L0 — `Scalar` and `SimdOps`

[`Scalar`] is deliberately tiny: identity constants plus a mixed-precision
accumulator type `Acc` (`Self` for f32/f64; a future f16 would set `Acc = f32`).
No arithmetic lives on it, so adding an element type does not grow the trait.
[`Float`] adds the scalar arithmetic the float epilogues need.

[`SimdOps<T>`] is the **load-bearing wall**. It is *thick*: it exposes every
primitive the whole microkernel needs — `zero`, `splat`, `load`/`loadu`,
`store`/`storeu`, `mul`, `add`, `mul_add`, `fnma` (fused `c - a·b`, for the complex
kernel's `acc_re -= ai·bi`), `reduce_sum` — plus `LANES` and `ALIGN`. Because the ISA
*token* and the element *type* are decoupled, `LANES` varies with the `(ISA, T)` pair
(f32@FMA = 8, f32@AVX-512 = 16, f32@NEON = 4, f32@wasm-simd128 = 4, f64 halved).

This is the direct answer to matrixmultiply's thin-trait trap. matrixmultiply's
kernel trait abstracts only the multiply-add, so the entire microkernel had to be
hand-written per ISA (and only one tile per ISA was maintainable). Here, *all* the
primitives are in `SimdOps`, so the microkernel is one generic function and each
ISA contributes only a set of primitives plus a one-line trampoline.

The hot inner loop — the `kc`-deep accumulation of one full `MR_REG × NR` tile —
is factored behind a single overridable seam, [`SimdOps::accumulate_tile`], so an
ISA can re-shape *the schedule* without forking the microkernel. The shared
microkernel calls it directly and carries **no per-ISA branch**; all the variation
lives in that method's *default implementation*, which holds two schedules: the
portable per-column `splat` + FMA, and an optional lane-indexed fast path
(`const LANE_FMA` + `fma_bvec`, NEON `vfmaq_laneq`, broadcasting B multipliers
straight from a loaded vector lane). The lane path is gated on a `const` that is
`false` for every ISA by default, so it and its provided `fma_bvec` fallback
compile away everywhere except the one token (NEON) that overrides them.

On a wide out-of-order core LLVM already lowers the default to the canonical
register-blocked kernel, so the *whole-seam* override exists only for cores it
will **not** fix on its own — an in-order / narrow-OoO core that needs explicit
software pipelining, or a scalable-vector ISA (SVE/RVV) whose length is not a
compile-time `LANES`. Every path here — the lane fast path and any override — is
bound by one contract: **same-run consistency**. Within a run the accumulation
stays a per-element fused `a·b + c` in ascending `p`, so a full tile and an edge
tile of the same matrix round identically (the L4 driver's determinism guarantee);
software pipelining reorders *loads*, never the arithmetic, so it is legal. The
lane path performs the same per-element fused `a·b + c` as the `splat` path and on
the benchmarked M-series is perf-neutral (that kernel is FMA-throughput-bound, not
B-load-bound) — it earns its place as a vocabulary option, not a measured win. An
override need only stay **deterministic and accurate to the same tolerance** under
a fixed config; it need not be bitwise-identical to the default. Instructions that
*reshape the accumulation rounding itself* — matrix/dot widening (`bfmmla`, `sdot`,
VNNI, bf16) — are out of scope for *this* seam: they arrive as a new
[`KernelFamily`] (L1) with a dedicated dot seam, never an `accumulate_tile`
override.

### `#[target_feature]` correctness

AVX/AVX-512 intrinsics must be code-generated where the feature is enabled, but
the CPU is chosen at runtime, so we can't put a fixed `#[target_feature]` on the
generic kernel. Each token's [`Simd::vectorize`] runs a closure inside a tiny
`#[target_feature]`-annotated inner function; the closure and the `#[inline]`
primitives fold into it, so every intrinsic lands in a feature-enabled context.
The driver wraps each output column strip in `vectorize`, which works for both the
serial path and rayon worker closures (the proven pulp/faer pattern). The scalar
token's `vectorize` is the identity.

## L1 — `KernelFamily` and the float microkernel

The driver is generic over [`KernelFamily`], not over "do an FMA on `T`". A family
bundles the input/accumulator/output types, the pack layout, the microkernel, and
the epilogue. It ships four families: `FloatGemm<T>` (homogeneous `f32`/`f64`),
`MixedGemm<N>` (mixed precision — `f16`/`bf16` in, **`f32` accumulator**, narrow
out; its bf16 `vdpbf16ps` dot sibling `Bf16DotGemm` is below), `IntGemm` (`i8` in, **`i32` accumulator/output** — the first family with
`Lhs != Out`, reached via the public `gemm_i8` since the homogeneous `gemm<T>`
surface can't express `Out != Lhs`; it reuses the `KernelSimd` widen seam with an
`i8 -> i32` sign-extend and exact wrapping `i32` arithmetic; its sibling `IntGemmVnni`
is the denser AVX-512 **VNNI dot kernel**, below), and `ComplexGemm<T, CONJ_A, CONJ_B>`
(`Complex<f32>`/`Complex<f64>`, via the public `gemm_cplx`).

**Dot-kernel families** add a second L0 seam, [`KernelSimd::dot_accumulate`], for ISAs
whose hardware folds several depth steps into one *dot* instruction — `vpdpbusd` (4 i8
steps), `vdpbf16ps` (2 bf16 steps). These reshape the accumulation rounding, so they
cannot ride `accumulate_tile` (whose contract forbids that); instead the family packs
its operands **k-group-interleaved** and reports `KernelFamily::DEPTH_MULTIPLE = Q` (the
group size), the one driver-visible knob: the driver rounds each packed panel's depth up
to `Q` (`1` for every other family, so they are byte-for-byte unchanged), and the family
depth-pads the tail. Two dot families ship, both AVX-512, each a sibling of the widen
family it accelerates (the pack layout differs, and `pack_lhs`/`pack_rhs` have no ISA
parameter, so the interleave must key off the family):

* `IntGemmVnni` (`Q = 4`, sibling of `IntGemm`) offsets A by `+128` for the `u8 × i8`
  `vpdpbusd` and subtracts the per-column bias `128·Σ_k B[k][j]` in the dot seam, so it is
  **bit-for-bit equal to `IntGemm`** (exact wrapping `i32`). The auto dispatch falls back
  to the widen kernel for small *parallel* problems, where the dot kernel's mandatory pack
  barrier outweighs its compute win.
* `Bf16DotGemm` (`Q = 2`, sibling of `MixedGemm<bf16>`) packs bf16 in pairs and accumulates
  in `f32` with `vdpbf16ps`, sharing `MixedGemm`'s `kc = k` rule and narrow epilogue. Its
  fused 2-term MAC is only *tolerance*-equal to the widen path (not bitwise), but serial,
  parallel, and prepacked runs share the one kernel and pack layout, so they reproduce each
  other bit-for-bit. Because the widen bf16 load is load-bound, the dot kernel wins at every
  size, so it has no widen fallback.

`FloatGemm` is always built; the other three families are **optional, off-by-default
Cargo features** — `half` (`MixedGemm`, pulls `half`), `int8` (`IntGemm`, no extra
dep), and `complex` (`ComplexGemm`, pulls `num-complex`). Gating is purely additive
and orthogonal to the seam: each feature toggles a family's kernel, its per-ISA
`SimdOps`/`KernelSimd` impls, its dispatch ladder, and its public entry — the driver,
packing framework, cache model, and parallelism are untouched, so a plain `f32`/`f64`
build pays for none of their codegen or dependencies.

**Complex (op-family seam) — a dedicated split (SoA) kernel.** `ComplexGemm` does
**not** ride `FloatGemm`. The product runs in the **structure-of-arrays** layout: real
and imaginary planes in **separate accumulator registers**, so one complex
multiply-accumulate is four fused *real* FMAs into two banks — `acc_re += ar·br`,
`acc_re -= ai·bi`, `acc_im += ar·bi`, `acc_im += ai·br` — with no in-loop shuffle or
`fmaddsub` (that `-=` step is the only new L0 primitive, `SimdOps::fnma`, x86 `fnmadd` /
NEON `vfms`). The de-interleave moves *out* of the `kc` loop into the pack: each
micropanel is laid down **planar** (per depth step, the `mr`/`nr` reals then the imags),
amortized `O(MK+KN)` instead of `O(MNK)`, so the kernel loads a register of reals and a
register of imags with plain contiguous loads. Because the kernel only consumes that
layout, both operands are **always** packed (`FORCE_PACK_LHS = FORCE_PACK_RHS = true`).
The epilogue de-interleaves `C`, folds the complex `alpha`/`beta`, and re-interleaves on
store. This is the only path to NEON's full FMA throughput on **stable** Rust (the fused
`FCMLA` is nightly-gated — `stdarch_neon_fcma`, rust-lang/rust#117222 — and would also
change the rounding relative to the SoA real-FMA path) and it raises x86 throughput over
the old interleaved-`fmaddsub` kernel.

The family stays homogeneous (`Acc = T`, so the complex `alpha`/`beta` thread through the
unchanged driver), but the hot loop runs on the *real* component, which the
`KernelSimd<T, T, T, T>` bound — yielding only `SimdOps<Complex>` — cannot name. The
bridge is `SimdOps::cplx_microkernel`, the **complex analogue of `accumulate_tile`**: an
L0 seam whose default is unreachable and whose per-ISA `SimdOps<Complex<_>>` override has
the real `SimdOps<f32>`/`<f64>` concretely and forwards to one shared, ISA-generic SoA
kernel. The thin `SimdOps<Complex<_>>` glue exists only so the driver can read `LANES`
(set to the **real** lane count: real lanes = complex rows the tile spans) and the
homogeneous `KernelSimd` blanket applies; complex GEMM never calls its element ops.

**Conjugation is a sign flip on the packed imaginary plane:** `CONJ_A`/`CONJ_B` are
`const` params, and a set flag negates the imag plane *during packing*, so `A̅·B` falls
out of the same real-FMA loop — no per-element conj branch. Dispatch maps the runtime
conj flags to the const-generic variant (and the orientation swap swaps the flags too,
since `(A̅·B)ᵀ = Bᵀ·A̅ᵀ`); `conjC` is deferred. The deferred integer VNNI (a dot kernel
with interleaved-K packing and a requantize epilogue) would arrive the same way, with the
driver, packing framework, cache model, and parallelism untouched.

**Mixed precision (`Acc != Lhs`).** `MixedGemm<N>` is the seam's first asymmetric
family: it packs narrow `N` panels (plain micropanels, like the float pack),
**widens them to `f32` registers on load**, accumulates in `f32`, and **rounds back
to `N` on store** (reading a narrow `C` widened for the `β != 0` term). The widening
lives entirely behind a small L0 capability, `KernelSimd<L, R, A, O>`, whose
`load_lhs`/`splat_rhs`/`load_out`/`store_out` are the widen-load / narrow-store
primitives an ISA token must provide. A single blanket impl makes the homogeneous
case (`L = R = A = O`) plain `SimdOps` load/splat/store, so `FloatGemm` and every
external homogeneous family get it for free; the all-equal blanket can never overlap
a mixed impl (which has `L != A`), so coherence is clean. The microkernel and driver
bound on `KernelSimd` instead of `SimdOps`, and the driver derives `MR` from the
**accumulator** lane count — so the five-loop nest carries no per-type branch. The
dispatch bound is `GemmScalar: Scalar` (not `Float<Acc = Self>`); per-type details
that can't be expressed generically (the `f32`-mediated `β`-scale; which family to
pack/dispatch through) are `GemmScalar` methods.

**Tile geometry is not on the trait.** `MR_REG` (register rows) and `NR` (columns)
are const generics on the driver/microkernel, chosen per `(family, ISA)` at the
dispatch site. So a new tile is a new instantiation — never a new type, never a
macro. `MR = MR_REG * LANES`.

The float microkernel is a rank-1 outer-product update: load `MR_REG` LHS vectors,
broadcast each of `NR` RHS scalars, FMA into `NR * MR_REG` accumulators that stay
in registers. The const-bounded `for` loops monomorphize and fully unroll, so the
optimizer keeps every accumulator in a register without any `seq!`-style macro.
The full-tile `kc`-loop runs through the overridable [`SimdOps::accumulate_tile`]
seam (L0); only the partial-column edge tile and the epilogue stay inline in the
kernel. v1 tiles (register usage in parentheses):

| | f32 | f64 |
|---|---|---|
| FMA (AVX2) | 16×6 = `MR_REG=2, NR=6` (15 YMM) | 8×6 (15 YMM) |
| AVX-512 | 32×12 = `MR_REG=2, NR=12` (27 ZMM) | 16×12 (27 ZMM) |
| scalar | 4×4 | 4×4 |

**Epilogue.** A full tile with column-major output (`rsc == 1`) takes the fast
vector path (load C, `β·C + α·AB`, store). Partial tiles or general/negative
strides drain the accumulators to a stack scratch buffer and copy back with the
real strides — one scratch tile per call, no lookup grid. `β == 0` never reads C.

## L2 — packing

Packed layout is **micropanel-major**: A as panels of `MR` rows (column-by-column,
`MR` contiguous rows per depth step), B as panels of `NR` columns, tails
zero-filled. The same `pack_panels` primitive serves both LHS and RHS — they
differ only in which stride is "leading" vs "depth". The *choice* of layout
belongs to the family, but the mechanical copy is shared and never changes when a family
is added: `pack_panels` for the plain micropanel layout, and `pack_kgroup_panels` for the
**k-group-interleaved** dot layout (`Q` consecutive depth steps contiguous per lane/column,
depth padded to `Q`, with a per-element transform — identity for bf16, the `+128` `u8` bias
for VNNI). Both dot families (`IntGemmVnni` `Q = 4`, `Bf16DotGemm` `Q = 2`) route through
that one routine, so the interleave index math has a single source of truth.

**Adaptive.** Packing is skipped when it doesn't pay. RHS is packed once per
panel (always, in v1) and reused across all row blocks. LHS packing has **two
independent triggers**:

1. *Reuse* — a non-unit row stride or a partial row panel forces packing (the
   microkernel reads full `MR`-row vectors); otherwise a column-major full block is
   packed only when each worker reuses it across enough columns to amortize the
   copy. Historically every worker packed into its own buffer, so redundant packing
   made mid-size parallel runs cheaper *unpacked* — column-major inputs flowed
   straight into the kernel and packed only when reuse was genuinely high. (The L5
   dynamic scheduler now hands each row-block to one worker on the packed path, so
   that redundancy is largely gone.)
2. *Stride* — even with low reuse, an in-place column-major A is read by walking K
   with stride `csa`, so once `csa · sizeof` reaches ~a memory page every depth
   step lands on a fresh page and the strided read collapses (TLB thrash). Above
   that gate A is packed regardless of reuse, which recovers large-`m` parallel
   throughput dramatically — measured ~2.4× at m = 4096 on Apple Silicon's 16 KiB
   pages (50% → 111% of the `gemm` crate). The gate scales with the page, so —
   keeping with the "detect geometry, don't hardcode" rule — it is **derived from
   the runtime page size** (`cache::page_size()` via `getpagesize`, half a page),
   not a fixed constant: 8 KiB on 16 KiB-page Apple Silicon, 2 KiB on 4 KiB-page
   x86/Linux, automatically. `GEMMKIT_LHS_PACK_STRIDE` overrides it (`0` = auto).

## L3 — cache topology and analytical blocking

`#[cfg]` chooses the *sniffing method*, never the *values*. CPUID is an
instruction (OS-independent; works in containers/VMs), so the x86 backend reads
L1/L2/L3 via CPUID (`raw-cpuid`: Intel deterministic leaf, AMD legacy leaves). On
macOS the `sysctl` backend reads `hw.perflevel0.{l1d,l2}cachesize` etc. through a
tiny `sysctlbyname` FFI (no `libc` dependency) — the primary source on Apple
Silicon, where there is no CPUID; it divides the cluster-shared L2 by
`hw.perflevel0.cpusperl2` for a realistic per-core budget. On Linux a `sysfs`
backend reads `/sys/devices/system/cpu`. Any backend that fails or returns
implausible values, or a VM that masks CPUID, falls through a chain: backend →
micro-arch hint → a static default calibrated on the Ryzen 9950X. Detection runs
once (memoized) and a plausibility gate rejects half-populated reads.

Blocking is the **BLIS analytical model**: `KC` so the A and B micro-panels coexist
in L1 without self-eviction; `MC` so the A macro-panel resides in L2 (reserving one
way for B, capped at `8·MR`); `NC` so the B macro-panel resides in L3 (reserving
one way for A). The result is **independent of thread count**, which is the
mechanism behind reproducible serial/parallel output: a fixed config yields the
same result regardless of how many workers run.

## L4 — the driver

One BLIS five-loop nest, generic over the family and the ISA token, with no
concrete type, ISA, or macro:

```
for jc in 0..n step nc:               # L3 column macro-panel
  for pc in 0..k step kc:             # depth — NOT parallel (C accumulates)
    beta_eff = (pc==0) ? beta : 1     # user β only on the first depth slice
    pack B panel in parallel
    for each (ic row-block × jt column-tile) job, parallel over a flat 1-D list:
      pack/reuse this row-block's A
      vectorize: for each MR×NR microtile -> microkernel(...)
```

Invariants: the depth loop is serial (the C tile is read-modify-written across
depth); `β` is applied only on the first depth slice so `C ← βC + αAB` holds and
`β==0` never reads C; each output tile is computed start-to-finish by one worker
over the full K; the blocking is thread-count-independent. Together these make the
output **reproducible** for any [`Parallelism`] — a fixed config gives the same
result regardless of thread count. (Serial and parallel happen to be bitwise-equal
today because both run the same kernel, but the contract is reproducibility under a
fixed config, not bitwise serial-vs-parallel identity.)

**Orientation.** If C is row-major-ish (`|csc| < |rsc|`), the dispatch layer
computes `Cᵀ = Bᵀ·Aᵀ` by swapping A/B and the strides, so the kernel always writes
columns contiguously. (This swap needs `Lhs == Rhs`, so it lives in the
concretely-typed dispatch layer, not the fully generic driver.)

## L5 — parallelism

`Parallelism::{Serial, Rayon(n)}` (`Rayon(0)` = auto). A single work-gate: below an
`m·n·k` threshold the run is forced serial; above it the worker count *scales with
the workload* (half-gate granularity) and is capped by the available parallelism
and the job count. Work is a flat 1-D list of column strips; workers pull
contiguous chunks from a shared lock-free cursor (`JobCursor`) **on demand**, so
faster cores absorb proportionally more — the makespan approaches `work / Σ core
rates` instead of `n · slowest`, which matters on heterogeneous big.LITTLE layouts
(Apple P/E, ARM DynamIQ, Intel hybrid) where an equal indivisible split is bounded
by the slowest core. This forfeits *nothing*: blocking stays thread-count
independent, the depth loop stays serial, and each output tile is still computed
wholly by one worker over the full K, so *which* worker computes a tile does not
change the result — the output is reproducible (and, with today's single-kernel
design, bitwise-equal) across thread counts. On the common in-place-LHS path, chunk size targets
`GEMMKIT_PARALLEL_OVERSAMPLE` chunks per worker (default 8: ~8 chunks/worker bounds
the heterogeneous tail imbalance to ~⅛ of a worker's share). When the LHS is packed
*and* there are at least as many row-blocks as workers, the chunk is a whole
row-block instead — each block's A is then packed by exactly one worker, trading
some balance granularity for pack-once reuse; with fewer row-blocks it falls back to
the fine grain. RHS packing is parallelized the same way (its own cursor) with a
barrier before compute.

The `cbrt(mnk)` ramp above models *compute* work. The bandwidth-bound L6 special paths
instead call `Parallelism::resolve_bandwidth(bytes_touched, rows)`: below an LLC-derived byte
floor the touched data is cache-resident and one core already gets the full LLC bandwidth, so
it stays serial (splitting there only loses — the thread-scaling curve *dips* at 2–4 workers
on fork/join and shared-cache contention, with no DRAM to gain); above the floor the auto
count steps straight to a topology bandwidth cap (`bandwidth_cap` — a documented fraction of
the logical core count, since DRAM saturates far below it). It is a *step*, not a ramp,
because for a bandwidth-bound shape a few workers is the worst point on the curve.
`GEMMKIT_GEMV_PARALLEL_BYTES` / `GEMMKIT_GEMV_THREAD_CAP` tune the floor and cap.

## L6 — special paths (bandwidth-bound shapes)

These shapes have O(1) arithmetic intensity, so the ceiling is memory bandwidth, not
compute. Both compute each output element in a **single pass over `k`** and parallelize by
partitioning disjoint **output** tiles — no cross-thread reduction — so the result is
bit-identical to the serial run for any worker count. Worker counts come from
`Parallelism::resolve_bandwidth` (a linear-in-bytes ramp capped by a topology bandwidth
proxy), not the driver's `cbrt(mnk)` compute ramp: past the few cores that saturate DRAM,
more workers stop helping.

- **gemv** (`gemv.rs`, `m == 1` or `n == 1`): both cases reduce to one core routine by
  viewing the matrix (transposed for `m == 1`) as `rows × k`. Column-major (axpy) shape has
  two bit-identical strategies chosen by cache fit and `k` — column-outer axpy when the
  output stays L2-resident (its re-reads are cheap and its single contiguous matrix stream is
  ideal; it folds `KB` columns per output load/store, so the axpy form is limited by DRAM
  bandwidth rather than load/store-port throughput), and **output register-blocking** (hold
  the output panel in registers across the
  whole `k`-sweep, output/matrix read once) when the output spills L2 *and* `k` is small
  enough that the register-blocked form's `k` in-place matrix column-streams stay within the
  prefetcher's window. Row-major uses the dot form.
- **small_k** (`small_k.rs`, `k <= small_k_threshold`): skinny / low-depth GEMM (gevv,
  rank-`k`, tall-skinny). Computes the whole product in one depth panel over the family's
  microkernel, reading A/B **in place** (unpacked), skipping the driver's blocking/packing/
  workspace setup that is pure overhead at tiny `k`. Requires column-major A (`rsa == 1`) and
  a non-`FORCE_PACK` family; otherwise defers to `driver::run`. Wired into every family's
  dispatch except complex (its planar SoA kernel cannot read in place).

Deferred: the small-matrix horizontal kernel and batched GEMM.

## L7 — dispatch

Each element type has one `OnceLock<fn>`. Feature detection (`avx512f → fma →
scalar`) runs once; the winning monomorphized entry point is cached; later calls
are a plain indirect call. **No `transmute`, no `AtomicPtr<()>`** — the slot is a
typed function pointer. A per-`(type, ISA)` wrapper picks the `(MR_REG, NR)` tile
and calls the generic driver; that is the *only* per-(type,ISA) code, and it is one
line each.

## L8 — API

- **Safe** ([`gemm`] / [`gemm_with`]): `MatRef`/`MatMut` slice + stride views.
  Shape mismatch, out-of-bounds strides, and C aliasing A/B all **panic** before
  any unsafe work. (In safe Rust, `&mut` C cannot overlap `&` A/B anyway; the alias
  check is a defensive guarantee.)
- **Unchecked** ([`gemm_unchecked`]): the raw pointer + `isize` stride engine for
  advanced callers (e.g. the ndarray adapter) that validate their own inputs and
  may use negative strides.

`gemmkit-ndarray` is a thin adapter: it accepts `&ArrayBase<S, Ix2>` for any
`S: Data` (both `ArrayView2` and `&Array2`), reads the pointer and strides, and
forwards to the unchecked engine — so C-order, F-order, general-stride, and
reversed views all work without copying. `dot` is the `.dot()`-style convenience.

## Cross-cutting

- **Tuning** (`tuning.rs`): every heuristic threshold in one place, each resolving
  *per-call argument > setter > `GEMMKIT_*` env var > compile-time default*.
- **Workspace** (`workspace.rs`): a 64-byte-aligned growable packing buffer. The
  default path uses a transparent thread-local pool; `gemm_with` accepts a
  caller-owned `Workspace` whose second-and-later uses allocate nothing.

## How this maps to the rigor criteria

- **No macro-generated kernels** — each microkernel is a single generic function (the
  real one; the one shared SoA complex one); the only `macro_rules!` in the crate
  generate `SimdOps` *impl boilerplate* — the scalar token's element ops and the thin
  complex glue — never kernel bodies.
- **One kernel, all ISAs** — adding an ISA is a `SimdOps` impl + one `vectorize`
  trampoline + a few dispatch lines, with the driver, packing, blocking,
  parallelism, API, and microkernel all untouched. The AArch64 NEON token is the
  worked proof: it was added purely additively (new `simd/neon.rs`, two `mod`
  lines, the dispatch wiring, one `isa_neon` test) on a different architecture
  with 32 vector registers and a wider tile than AVX2. The WebAssembly `simd128`
  token (`simd/wasm.rs`, `Simd128`) is a second, differently-shaped proof: a
  *compile-time* feature (`cfg`-selected, no runtime detection — like NEON's
  baseline-by-cfg arm) on a register-poor backend with **no hardware FMA**, so
  its `mul_add` is the two-rounding `add(mul(a,b),c)` that matches the scalar
  reference and keeps the path reproducible. It spans the **whole element-type
  matrix** purely additively — f32/f64 via the homogeneous blanket; i8 (`IntGemm`
  widen-and-multiply, no `vpdpbusd`), f16/bf16 (`MixedGemm` scalar widen/narrow,
  no native fp16), and complex (the shared SoA `cplx_microkernel` macro) reuse the
  exact same seams as the x86/NEON tokens — all with zero kernel/driver edits.
  Multithreading on wasm is opt-in: `parallel` alone stays safe-serial (the `RAYON_USABLE`
  guard degrades to the serial loop, since baseline `wasm32-wasip1` has no thread runtime),
  and the `wasm_threads` feature turns on a gemmkit-owned rayon pool for
  `wasm32-wasip1-threads`. wasm has no `available_parallelism` and on stable Rust the
  `atomics` cfg is unsettable, so both the opt-in and the worker count
  (`GEMMKIT_WASM_THREADS`, default 8) are explicit rather than auto-detected.
- **No `transmute`** — `OnceLock<fn>`.
- **Open/closed** — `tests/open_closed.rs` declares a second `KernelFamily`
  entirely outside the crate and drives it through the unchanged `driver::run`.
- **Single crate, many types; no reverse deps; localized unsafe** — by construction
  of the layer map above.

[`Scalar`]: crate::scalar::Scalar
[`Float`]: crate::scalar::Float
[`Simd::vectorize`]: crate::simd::Simd::vectorize
[`SimdOps`]: crate::simd::SimdOps
[`SimdOps<T>`]: crate::simd::SimdOps
[`SimdOps::accumulate_tile`]: crate::simd::SimdOps::accumulate_tile
[`KernelFamily`]: crate::kernel::KernelFamily
[`Parallelism`]: crate::Parallelism
[`gemm`]: crate::gemm
[`gemm_with`]: crate::gemm_with
[`gemm_unchecked`]: crate::gemm_unchecked
