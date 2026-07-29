# Design Goals and the Big Picture

gemmkit is a pure-Rust GEMM engine. It computes `C <- alpha*A*B + beta*C` over `&[T]`
slices with explicit strides, or over raw pointers with `isize` strides. At runtime it
selects the best instruction set the machine offers. On x86-64 that is AVX-512 or
FMA/AVX2, with dedicated VNNI and BF16 dot kernels. On aarch64 it is NEON. On wasm32 it
is `simd128`. A portable scalar path runs everywhere else. The workspace uses edition
2024, `rust-version` 1.89, and the MIT OR Apache-2.0 license.

3 kinds of callers shape the API surface. Application code uses the safe slice
entries, such as `gemm`, `gemm_fused`, and `gemm_i8`. These entries validate everything
before any unsafe work runs. Linear algebra libraries use the `*_unchecked` tier. This
group includes the shipped `ndarray`, `nalgebra`, and `faer` adapters, plus anything
built the same way. The `*_unchecked` tier trusts the caller's own invariants and
accepts layouts the safe tier cannot express. Constrained deployments get a core that
builds `#![no_std]` with zero mandatory dependencies, down to wasm32 with compile-time
SIMD. Everything else in this chapter follows from 4 design tenets.
[ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md)
states them compactly under "Goals and constraints". This chapter expands each one with
the reasoning behind it.

## Safety at the boundary

The checked entries run `validate_gemm_views` (`gemmkit/src/api.rs`) before touching any
unsafe code. Its panic catalog is deliberately exhaustive:

- **Shape mismatch**: gemmkit checks `A.cols != B.rows`, `A.rows != C.rows`, and
  `B.cols != C.cols`. Each panic message names the 2 numbers that disagree.
- **A view addressing outside its slice**: for A, B, and C, gemmkit computes the
  highest offset each view's strides can reach (`extent`). It checks that offset
  against the slice length. A view that needs more elements than its slice holds
  panics with the exact shortfall.
- **Negative strides**: the safe tier rejects these, with a message that points to
  `gemm_unchecked`. A `&[T]` view with a negative stride would address below the slice
  start. The safe extent math cannot vouch for that address.
- **A self-aliasing output**: a stride on `C` can map 2 distinct `(i, j)` pairs to the
  same offset. A zero stride is the common case. This is fine on `A` or `B`, since a
  broadcast input is only read. On `C` it panics, because the parallel driver assumes
  output tiles do not overlap. Writing through such a view would create a data race that
  entirely safe code could reach.
- **`C` overlapping `A` or `B`**: gemmkit checks this as byte ranges. This stays exact
  even when C (`i32`) and A/B (`i8`) have different element sizes. The fused entries
  also check bias length (`PerRow` = m, `PerCol` = n) and bias and C disjointness.
- **A problem too large to size**: broadcast strides allow logical dimensions near
  `isize::MAX`. The internal pack-buffer sizing can then overflow `usize`. Every such
  product panics, fail-closed, at the element-to-byte chokepoint (`Workspace::regions`),
  instead of wrapping and under-allocating.

The panic wording is itself a tested contract. The correctness suite asserts the exact
strings. Changing an error message is therefore a deliberate, visible act.

The `*_unchecked` tier exists because this validation is only meaningful at one
boundary. The adapters pull pointers and strides straight out of `ndarray`, `nalgebra`,
and `faer` types. Those types already guarantee validity through their own invariants,
so re-checking would be pure overhead. The slice-based checks could not even express
what the adapters need. For example, a reversed `ndarray` view has a negative stride and
a base pointer in the middle of its allocation. Both are legal and sound for the raw
engine. Safety is therefore paid exactly once, either by gemmkit's validator or by the
caller's type system, never both.

The unchecked entries are ordinary `unsafe fn`s with documented contracts. The guide
covers them in [The Unchecked Tier](../gemmkit-guide/The_Unchecked_Tier.md).

## Reproducible, not bitwise, parallel results

gemmkit promises reproducible parallel results. For a fixed input, a fixed environment,
and a fixed configuration, the output does not depend on the worker count. Three
mechanisms carry this promise.

The first mechanism is the blocking sizes. `KC` and `NC` come from the cache model
alone and never depend on the thread count. `MC` only ever changes by an `MR`-aligned
regroup. Every run therefore reduces each output element in the same fixed order.

The second mechanism is the reduction order. One worker reduces each output element
start to finish, over the full depth. The engine never splits a reduction across
workers.

The third mechanism is demand-driven scheduling. Packed bytes do not depend on who
packs them, so any worker can take any tile. Which worker computes a tile varies from
run to run. The result never does.

Just as important is what gemmkit does not promise. Bitwise serial-versus-parallel
identity is not part of the contract. It happens to hold on the driver paths today,
because serial and parallel runs execute the same kernel over the same blocking.
Nothing pins this fact in place. Bitwise identity across configurations is explicitly
out too. Change a tuning knob, and the blocking, and so the floating-point summation
order, may legitimately change.

Bitwise identity across kernels of the same type is out as well. The bf16 `vdpbf16ps`
dot kernel reshapes the accumulation rounding relative to the widen-and-FMA path.
gemmkit holds that kernel to a tolerance, not to exact equality.

Why draw the line there? A permanent promise of bitwise serial-versus-parallel identity
would forbid useful engineering. It would rule out dot-product instructions that fuse
depth pairs. It would also rule out a blocking choice that takes parallelism into
account. Such a promise would buy nothing a user can rely on across machines or library
versions anyway.

Parallelism-aware blocking is no longer hypothetical. The driver already has a
job-depth floor that shrinks `MC` with the worker count, to keep the parallel job list
deep enough. This stays bitwise-reproducible precisely because the weaker contract
leaves room for it. `MC` still stays an `MR` multiple, so the microtile set and every
element's `KC`-shaped accumulation order stay unchanged.

Reproducibility under a fixed configuration is the property tests can assert. Deployments
can depend on it, and the engine can keep it while it evolves. Where a path
can promise more cheaply, it does. gemv partitions output rows and is bit-identical
across worker counts. The `i8` integer path uses exact arithmetic, so its VNNI dot
kernel is bit-identical to the widen kernel.

## No macros, no `transmute` at the variation points

The engine varies along 3 axes: instruction set, element type, and operation
family. Each axis is an ordinary trait. `Simd` and `SimdOps` cover the ISA. `Scalar`
covers the element type. `KernelFamily` covers the operation family.

Dispatch slots are typed function pointers, cached in `OnceLock`s. Microtile geometry
is a pair of const generics, chosen at the dispatch site. Here is what a "kernel
variant" actually looks like, from `gemmkit/src/dispatch/float.rs`:

```rust
unsafe fn gemm_f32_fma(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 6 -> 12 acc + 2 lhs + 1 rhs = 15 of 16 YMM
    unsafe { run_typed::<f32, Fma, 2, 6>(Fma, t, par, ws) }
}

unsafe fn gemm_f32_avx512f(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*16 = 32, NR = 12 -> 24 acc + 2 lhs + 1 rhs = 27 of 32 ZMM
    unsafe { run_typed::<f32, Avx512F, 2, 12>(Avx512F, t, par, ws) }
}
```

That is the entire per-(type, ISA) surface. Each variant is one wrapper that names a
token and a tile. The alternative, macro-stamped or hand-copied per-ISA kernels in the C
BLAS tradition, was rejected for its cost to review and to extend.

Traits and const generics leave exactly one generic microkernel to read, step through,
and fix. A scheduling improvement lands once, and every ISA inherits it. The compiler
type-checks every monomorphization. The `OnceLock` slots hold typed function pointers,
not type-erased ones, so a signature drift is a compile error, not a latent `transmute`
bug.

Extension follows the same shape. A new ISA needs a zero-sized token, its `SimdOps`
impls, and one arm per selection ladder. A new element type needs a `Scalar` impl, a
family (or a reuse through the widen/narrow seam), and a dispatch slot. The driver,
packing, and blocking never change. A test (`gemmkit/tests/open_closed.rs`) enforces
this by driving the driver with a second, trivial family. 2 follow-up pages walk each
seam in detail: [SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md) and
[Scalars and Kernel Families](Scalars_and_Kernel_Families.md).

## `no_std` and a zero-mandatory-dependency core

With default features off, the core crate builds `#![no_std]`. It needs only `core` and
`alloc`, and depends on nothing else. Every optional feature pulls in at most one crate:

| Feature | Dependency added | What it buys |
|---|---|---|
| `std` (default) | `raw-cpuid` (x86/x86-64 targets only) | runtime cache and CPU-feature detection, `GEMMKIT_*` env knobs, the thread-local workspace pool |
| `parallel` (default) | `rayon` | `Parallelism::Rayon` multi-threading |
| `half` | `half` | `f16`/`bf16` mixed-precision GEMM |
| `complex` | `num-complex` | `c32`/`c64` complex GEMM |
| `int8` | none | `i8 -> i32` integer GEMM |
| `epilogue` | none | fused bias/activation/map epilogues (requantize additionally needs `int8`) |
| `wasm_threads` | none beyond `parallel` | an explicitly sized rayon pool on threaded wasm |

Without `std`, compile-time target features replace runtime CPU detection. The env
knobs turn off, though the programmatic `tuning::set_*` setters still work, since they
are plain atomics. A per-call workspace replaces the thread-local pool.

A kernel this low in the stack should not force a dependency policy on its hosts. An
embedded or wasm deployment gets the same driver, the same families, and the same
reproducibility contract as a desktop build. It loses only the machinery that
genuinely needs an OS. The practical how-to lives in
[no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md).

## The workspace map

5 crates release in lockstep at version 0.1.1, plus a fuzzing crate that
deliberately sits in its own workspace root:

| Path | Crate | Role |
|---|---|---|
| `gemmkit/` | gemmkit | The core GEMM engine (everything this chapter describes) |
| `gemmkit-ndarray/` | gemmkit-ndarray | Zero-copy adapter over `ndarray` (>= 0.17.1) views |
| `gemmkit-nalgebra/` | gemmkit-nalgebra | Zero-copy adapter over `nalgebra` 0.35 matrices |
| `gemmkit-faer/` | gemmkit-faer | Zero-copy adapter over `faer` 0.24 matrices |
| `gemmkit-tune/` | gemmkit-tune | Install-time autotuner binary emitting a `GEMMKIT_*` env profile |
| `gemmkit/fuzz/` | gemmkit-fuzz | cargo-fuzz targets, nightly-only, excluded from the stable workspace |

The adapters are thin by design. Each one pulls the matrix pointer and strides straight
out of the host library's native view. That view may be C-order, F-order, general
strides, or reversed strides, and the adapter never copies data. Each adapter then
forwards to the `*_unchecked` engine,
relying on the host type's own invariants for the validation the safe tier would
otherwise do. Each adapter also forwards the same-named Cargo features (`parallel`,
`wasm_threads`, `half`, `complex`, `int8`, `epilogue`) to gemmkit, so the feature story
stays identical everywhere. The adapter chapters cover their full surfaces. See
[ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md),
[nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md), and
[faer](../gemmkit-faer/Using_gemmkit_with_faer.md).

`gemmkit-tune` is the out-of-process calibrator. Every heuristic threshold in the
engine is a runtime knob (see [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md)). The
compiled defaults were calibrated on one machine, so the tuner exists to redo that
calibration on yours.

Run the tuner binary once on the deploy host. It sweeps each knob over a set of probe
shapes, then writes a `gemmkit-tune.env` profile of `export GEMMKIT_*=...` lines. Source
that file before you launch your application. This needs no recompile and no
build-time coupling. The only contract between the tuner and the library is the
documented env-var surface. The `tuning::knob_env_names` registry keeps that contract
honest, since the tuner's sweep table is checked against it. The
[gemmkit-tune chapter](../gemmkit-tune/Tuning_with_gemmkit-tune.md) has the practical
guide.

The fuzz crate sits outside the workspace on purpose. cargo-fuzz needs nightly, for
build-std and AddressSanitizer. Excluding the fuzz crate keeps `cargo test --workspace`
and the MSRV build on stable.

## This chapter and ARCHITECTURE.md

The repository's `ARCHITECTURE.md` is the compact map. It gives the layer table, the
call path, the seams, and one section per subsystem. It is written for a reader who has
the code open in another pane. This book chapter is the guided tour of the same material.
It uses the same layer labels and the same file references, but leaves room for the
reasoning, the rejected alternatives, and worked examples. When the two disagree, the
code wins, and both documents have a bug.

Read [The Layer Stack](The_Layer_Stack.md) next for the structure. Then read
[Life of a GEMM Call](Life_of_a_GEMM_Call.md) for the motion.
