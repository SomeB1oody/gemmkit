# Architecture

This document is a tour of how gemmkit is built and *why*. The short version: the
combinatorial explosion of **{element type} × {ISA} × {tile} × {operation family}**
is dissolved in the type system — a thick SIMD trait, const generics, and an
operation-family seam — so the algorithm is written **once**, with **no macros**
and **no `transmute`**.

## Design principles

1. **The algorithm is written once.** Blocking, packing, parallelism, and cache
   modelling are all generic. There is exactly one five-loop driver and exactly
   one floating-point microkernel.
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
L0  simd/            Simd + SimdOps<T> — the ISA seam (tokens: ScalarTok/Fma/Avx512)
L1  kernel/          KernelFamily — the operation-family seam; float microkernel
L2  pack.rs          micropanel packing primitive
L3  cache/           CacheTopology + analytical BLIS blocking (cpuid backend + fallback)
L4  driver.rs        the generic five-loop nest
L5  parallel.rs      Parallelism + 1-D job split + work-gate
L6  special/gemv.rs  matrix·vector special path
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
`store`/`storeu`, `mul`, `add`, `mul_add`, `reduce_sum` — plus `LANES` and
`ALIGN`. Because the ISA *token* and the element *type* are decoupled, `LANES`
varies with the `(ISA, T)` pair (f32@FMA = 8, f32@AVX-512 = 16, f64 halved).

This is the direct answer to matrixmultiply's thin-trait trap. matrixmultiply's
kernel trait abstracts only the multiply-add, so the entire microkernel had to be
hand-written per ISA (and only one tile per ISA was maintainable). Here, *all* the
primitives are in `SimdOps`, so the microkernel is one generic function and each
ISA contributes only a set of primitives plus a one-line trampoline.

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
the epilogue. v1 ships one family, `FloatGemm<T>`. Complex or integer GEMM would
arrive as *new families* (complex: an `fmaddsub` two-step kernel; integer: a VNNI
dot kernel with interleaved-K packing and a requantize epilogue) with the driver,
packing framework, cache model, and parallelism untouched.

**Tile geometry is not on the trait.** `MR_REG` (register rows) and `NR` (columns)
are const generics on the driver/microkernel, chosen per `(family, ISA)` at the
dispatch site. So a new tile is a new instantiation — never a new type, never a
macro. `MR = MR_REG * LANES`.

The float microkernel is a rank-1 outer-product update: load `MR_REG` LHS vectors,
broadcast each of `NR` RHS scalars, FMA into `NR * MR_REG` accumulators that stay
in registers. The const-bounded `for` loops monomorphize and fully unroll, so the
optimizer keeps every accumulator in a register without any `seq!`-style macro.
v1 tiles (register usage in parentheses):

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
belongs to the family (so a future integer family can interleave for VNNI); the
mechanical copy is shared and never changes when a family is added.

**Adaptive.** Packing is skipped when it doesn't pay. RHS is packed once per
panel (always, in v1) and reused across all row blocks. LHS packing is
**reuse-aware**: a non-unit row stride or a partial row panel forces packing
(the microkernel reads full `MR`-row vectors), but a column-major full block is
packed only when each worker reuses it across enough columns to amortize the copy.
Because every worker packs into its own buffer, redundant packing across workers
makes mid-size parallel runs cheaper *unpacked* — so column-major inputs flow
straight into the kernel there, and pack only when reuse is genuinely high (serial
or very wide). This single change roughly doubled mid-size multi-threaded
throughput.

## L3 — cache topology and analytical blocking

`#[cfg]` chooses the *sniffing method*, never the *values*. CPUID is an
instruction (OS-independent; works in containers/VMs), so the x86 backend reads
L1/L2/L3 via CPUID (`raw-cpuid`: Intel deterministic leaf, AMD legacy leaves). A
VM that masks CPUID, or a future aarch64 target, falls through a chain: backend →
micro-arch hint → a static default calibrated on the Ryzen 9950X. Detection runs
once (memoized) and a plausibility gate rejects half-populated reads.

Blocking is the **BLIS analytical model**: `KC` so the A and B micro-panels coexist
in L1 without self-eviction; `MC` so the A macro-panel resides in L2 (reserving one
way for B, capped at `8·MR`); `NC` so the B macro-panel resides in L3 (reserving
one way for A). The result is **independent of thread count**, which is a
prerequisite for bit-identical serial/parallel output.

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
output **bit-identical** for any [`Parallelism`].

**Orientation.** If C is row-major-ish (`|csc| < |rsc|`), the dispatch layer
computes `Cᵀ = Bᵀ·Aᵀ` by swapping A/B and the strides, so the kernel always writes
columns contiguously. (This swap needs `Lhs == Rhs`, so it lives in the
concretely-typed dispatch layer, not the fully generic driver.)

## L5 — parallelism

`Parallelism::{Serial, Rayon(n)}` (`Rayon(0)` = auto). A single work-gate: below an
`m·n·k` threshold the run is forced serial; above it the worker count *scales with
the workload* (half-gate granularity) and is capped by the available parallelism
and the job count. Work is a flat 1-D list of column strips; each worker takes a
balanced contiguous slice. RHS packing is parallelized the same way with a barrier
before compute.

## L6 — gemv

`m == 1` or `n == 1` routes to a memory-bound axpy/dot sweep (both cases reduce to
one core routine by viewing the matrix, transposed for `m == 1`, as `rows × k`).
Vectorized for contiguous layouts, correct for all. Deferred: gevv (k ≤ 2), the
small-matrix horizontal kernel, batched GEMM.

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

- **No macro-generated kernels** — the microkernel is the single generic function;
  the only `macro_rules!` in the crate generates the *scalar* `SimdOps` impl
  boilerplate, not kernel bodies.
- **One kernel, all ISAs** — adding an ISA is a `SimdOps` impl + one `vectorize`
  trampoline + one dispatch line.
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
[`KernelFamily`]: crate::kernel::KernelFamily
[`Parallelism`]: crate::Parallelism
[`gemm`]: crate::gemm
[`gemm_with`]: crate::gemm_with
[`gemm_unchecked`]: crate::gemm_unchecked
