# Design Goals and the Big Picture

gemmkit is a pure-Rust GEMM engine: it computes `C <- alpha*A*B + beta*C` over `&[T]` slices with explicit strides, or over raw pointers with `isize` strides, and selects the best instruction set available on the machine at runtime — AVX-512 or FMA/AVX2 on x86-64 (with dedicated VNNI and BF16 dot kernels), NEON on aarch64, `simd128` on wasm32, and a portable scalar floor everywhere. The workspace is edition 2024, `rust-version` 1.89, licensed MIT OR Apache-2.0.

Three kinds of callers shape the API surface. Application code uses the safe slice entries (`gemm`, `gemm_fused`, `gemm_i8`, ...), which validate everything before any unsafe work runs. Linear algebra libraries — the shipped `ndarray`, `nalgebra`, and `faer` adapters, and anything built like them — use the `*_unchecked` tier, which trusts the caller's invariants and accepts layouts the safe tier cannot express. And constrained deployments get a core that builds `#![no_std]` with zero mandatory dependencies, down to wasm32 with compile-time SIMD. Everything else in this chapter follows from four design tenets, stated compactly in [ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md) under "Goals and constraints" and expanded here with the reasoning behind each.

## Safety at the boundary

The checked entries run `validate_gemm_views` (`gemmkit/src/api.rs`) before touching any unsafe code, and its panic catalog is deliberately exhaustive:

- **Shape mismatch**: `A.cols != B.rows`, `A.rows != C.rows`, or `B.cols != C.cols`, each with a message naming the two disagreeing numbers.
- **A view addressing outside its slice**: for each of A, B, C, the highest offset the strides can reach is computed (`extent`) and checked against the slice length; a view that needs more elements than its slice has panics with the exact shortfall.
- **Negative strides**: rejected in the safe tier with a message pointing at `gemm_unchecked`. A `&[T]` view with a negative stride would have to address below the slice start, so the safe extent math simply cannot vouch for it.
- **A self-aliasing output**: strides on `C` that map two distinct `(i, j)` to the same offset — a zero stride is the common case — are fine on A or B (a broadcast input is only read) but panic on C, because the parallel driver assumes output tiles are disjoint and writing through such a view would be a data race reachable from entirely safe code.
- **`C` overlapping `A` or `B`**: checked as byte ranges, so it stays exact even for the heterogeneous integer entries where C (`i32`) and A/B (`i8`) have different element sizes. The fused entries additionally validate bias length (`PerRow` = m, `PerCol` = n) and bias/C disjointness.
- **A problem too large to size**: broadcast strides allow logical dimensions near `isize::MAX`, so the internal pack-buffer sizing can overflow `usize`; every such product panics fail-closed at the element-to-byte chokepoint (`Workspace::regions`) instead of wrapping and under-allocating.

The panic wording is itself a tested contract: the correctness suite asserts the exact strings, so an error message change is a deliberate, visible act.

The `*_unchecked` tier exists because this validation is only meaningful at one boundary. The adapters pull pointers and strides straight out of `ndarray`/`nalgebra`/`faer` types whose own invariants already guarantee validity — re-checking would be pure overhead, and the slice-based checks could not even express what the adapters need: a reversed `ndarray` view has a negative stride and a base pointer in the middle of its allocation, both legal and sound for the raw engine. So safety is paid exactly once, either by gemmkit's validator or by the caller's type system, never both. The unchecked entries are ordinary `unsafe fn`s with documented contracts; the guide covers them in [The Unchecked Tier](../gemmkit-guide/The_Unchecked_Tier.md).

## Reproducible, not bitwise, parallel results

The promise: for a fixed input, environment, and configuration, the output is identical regardless of the worker count. Three mechanisms carry it. `KC` and `NC` come from the cache model alone and never depend on the thread count, and `MC` only ever changes by an `MR`-aligned regroup, so every run reduces each output element in the same fixed order. Every output element is reduced start-to-finish by one worker over the full depth — there are no split reductions anywhere in the engine. And packed bytes do not depend on who packs them, so demand-driven scheduling can hand any tile to any worker. Which worker computes a tile varies run to run; the result never does.

Just as important is what is *not* promised. Bitwise serial-vs-parallel identity is not the contract — it happens to hold on the driver paths today, because serial and parallel runs execute the same kernel over the same blocking, but nothing pins it. Bitwise identity across configurations is explicitly out: change a tuning knob and the blocking, hence the floating-point summation order, may legitimately change. And bitwise identity across kernels of the same type is out too: the bf16 `vdpbf16ps` dot kernel reshapes the accumulation rounding relative to the widen-and-FMA path and is held to a tolerance, not to equality.

Why draw the line there? Promising bitwise serial-vs-parallel identity forever would forbid genuinely useful engineering — dot-product instructions that fuse depth pairs, or a blocking choice that considers parallelism — while buying nothing a user can rely on across machines or library versions anyway. That parallelism-aware blocking is no longer hypothetical: the driver's job-depth floor already shrinks `MC` with the worker count to keep the parallel job list deep enough, and it stays bitwise-reproducible precisely because the weaker contract left room for it — `MC` stays an `MR` multiple, so the microtile set and every element's `KC`-shaped accumulation order are unchanged. Reproducibility under a fixed config is the property tests can assert, deployments can depend on, and the engine can keep while it evolves. Where a path can promise more cheaply, it does: gemv partitions output rows and is bit-identical across worker counts, and the `i8` integer path is exact arithmetic, so its VNNI dot kernel is bit-identical to the widen kernel.

## No macros, no `transmute` at the variation points

The engine varies along three axes — instruction set, element type, operation family — and each one is an ordinary trait: `Simd`/`SimdOps` for the ISA, `Scalar` for the element type, `KernelFamily` for the family. Dispatch slots are typed function pointers cached in `OnceLock`s, and microtile geometry is a pair of const generics chosen at the dispatch site. Here is what a "kernel variant" actually is, from `gemmkit/src/dispatch/float.rs`:

```rust
unsafe fn gemm_f32_fma(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 6 -> 12 acc + 2 lhs + 1 rhs = 15 YMM
    unsafe { run_typed::<f32, Fma, 2, 6>(Fma, t, par, ws) }
}

unsafe fn gemm_f32_avx512f(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*16 = 32, NR = 12 -> 24 acc + 2 lhs + 1 rhs = 27 ZMM
    unsafe { run_typed::<f32, Avx512F, 2, 12>(Avx512F, t, par, ws) }
}
```

That is the entire per-(type, ISA) surface: one wrapper naming a token and a tile. The alternative — macro-stamped or hand-copied per-ISA kernels, the C BLAS tradition — was rejected for reviewability and extension cost. With traits and const generics there is exactly one generic microkernel to read, step through, and fix; a scheduling improvement lands once and every ISA inherits it. The compiler type-checks every monomorphization, and because the `OnceLock` slots hold *typed* function pointers rather than type-erased ones, a signature drift is a compile error, not a latent `transmute` bug. Extension follows the same shape: a new ISA is a zero-sized token, its `SimdOps` impls, and one arm per selection ladder; a new element type is a `Scalar` impl, a family (or a reuse through the widen/narrow seam), and a dispatch slot. The driver, packing, and blocking never change — a property enforced by a test (`gemmkit/tests/open_closed.rs`) that drives the driver with a second, trivial family. The follow-up pages [SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md) and [Scalars and Kernel Families](Scalars_and_Kernel_Families.md) walk each seam in detail.

## `no_std` and a zero-mandatory-dependency core

With default features off, the core crate is `#![no_std]`, needs only `core` + `alloc`, and depends on nothing at all. Every optional feature pulls at most one crate:

| Feature | Dependency added | What it buys |
|---|---|---|
| `std` (default) | `raw-cpuid` (x86/x86-64 targets only) | runtime cache and CPU-feature detection, `GEMMKIT_*` env knobs, the thread-local workspace pool |
| `parallel` (default) | `rayon` | `Parallelism::Rayon` multi-threading |
| `half` | `half` | `f16`/`bf16` mixed-precision GEMM |
| `complex` | `num-complex` | `c32`/`c64` complex GEMM |
| `int8` | none | `i8 -> i32` integer GEMM |
| `epilogue` | none | fused bias/activation/map epilogues (requantize additionally needs `int8`) |
| `wasm_threads` | none beyond `parallel` | an explicitly sized rayon pool on threaded wasm |

Without `std`, compile-time target features replace runtime CPU detection, the env knobs are off (the programmatic `tuning::set_*` setters still work — they are plain atomics), and a per-call workspace replaces the thread-local pool. The point is that a kernel this low in the stack should not force a dependency policy on its hosts: an embedded or wasm deployment gets the same driver, the same families, and the same reproducibility contract as a desktop build, minus only the machinery that genuinely requires an OS. The practical how-to lives in [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md).

## The workspace map

Five crates release in lockstep at version 0.1.0, plus a fuzzing crate that is deliberately its own workspace root:

| Path | Crate | Role |
|---|---|---|
| `gemmkit/` | gemmkit | The core GEMM engine — everything this chapter describes |
| `gemmkit-ndarray/` | gemmkit-ndarray | Zero-copy adapter over `ndarray` (>= 0.17.1) views |
| `gemmkit-nalgebra/` | gemmkit-nalgebra | Zero-copy adapter over `nalgebra` 0.35 matrices |
| `gemmkit-faer/` | gemmkit-faer | Zero-copy adapter over `faer` 0.24 matrices |
| `gemmkit-tune/` | gemmkit-tune | Install-time autotuner binary emitting a `GEMMKIT_*` env profile |
| `gemmkit/fuzz/` | gemmkit-fuzz | cargo-fuzz targets; nightly-only, excluded from the stable workspace |

The adapters are thin by design: each pulls the matrix pointer and strides straight out of the host library's native view — C-order, F-order, general and reversed strides, no copies — and forwards to the `*_unchecked` engine, relying on the host types' invariants for the validation the safe tier would otherwise do. Each adapter forwards the same-named Cargo features (`parallel`, `wasm_threads`, `half`, `complex`, `int8`, `epilogue`) to gemmkit, so the feature story is identical everywhere. The adapter chapters ([ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md), [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md), [faer](../gemmkit-faer/Using_gemmkit_with_faer.md)) cover their surfaces.

`gemmkit-tune` is the out-of-process calibrator. Every heuristic threshold in the engine is a runtime knob (see [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md)), and the compiled defaults were calibrated on one machine, so the tuner exists to redo that calibration on yours: run the binary once on the deploy host, it sweeps each knob over a probe-shape set and writes a `gemmkit-tune.env` profile of `export GEMMKIT_*=...` lines to source before launching your application. No recompile, no build-time coupling — the only contract between the tuner and the library is the documented env-var surface, kept honest by the `tuning::knob_env_names` registry that the tuner's sweep table is asserted against. The [gemmkit-tune chapter](../gemmkit-tune/Tuning_with_gemmkit-tune.md) has the practical guide.

The fuzz crate sits outside the workspace on purpose: cargo-fuzz needs nightly (build-std, AddressSanitizer), and excluding it keeps `cargo test --workspace` and the MSRV build on stable.

## This chapter and ARCHITECTURE.md

The repository's `ARCHITECTURE.md` is the compact map: layer table, call path, seams, one section per subsystem, written for someone with the code open in another pane. This book chapter is the guided tour of the same material — same layer labels, same file references, but with room for the reasoning, the rejected alternatives, and worked examples. When the two disagree, the code wins and both documents have a bug. Read on to [The Layer Stack](The_Layer_Stack.md) for the structure, then [Life of a GEMM Call](Life_of_a_GEMM_Call.md) for the motion.
