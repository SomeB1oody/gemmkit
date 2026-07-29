# no_std and WebAssembly

gemmkit's core does not need an operating system. Turn the default features off, and the crate becomes `#![no_std]`. It then needs only `core` and `alloc`, and depends on nothing else. This makes it usable in kernels, embedded firmware, and WebAssembly. The same code path is also how the wasm SIMD backend gets built. This page covers what a `no_std` build gives up, what it keeps, and the extra steps a wasm target needs.

## The no_std core

The `std` feature is on by default, as part of `default = ["std", "parallel"]`. Switch the defaults off, and you get the `alloc`-only path:

```toml
[dependencies]
gemmkit = { version = "0.1", default-features = false }
```

`alloc` is always required, because packing scratch is heap-backed in both builds. Beyond that, the crate pulls in nothing at all in this configuration. Each optional feature adds at most one dependency:

- `std` pulls `raw-cpuid`, on x86 only, for CPUID cache and feature detection.
- `parallel` pulls `rayon`.
- `half` pulls `half`.
- `complex` pulls `num-complex`.

The `int8` and `epilogue` features add no dependency at all. The element-type features compose freely with `no_std`. A `default-features = false, features = ["half", "int8"]` build, for example, is a valid, dependency-free `f16`/`bf16`/`i8` engine.

Note that `parallel` implies `std`, because rayon needs the standard library. So a `no_std` build is always single-threaded. Everything still compiles and runs. It just runs on one thread.

## What changes without std

3 things move from runtime to compile time, or from automatic to explicit:

**Feature detection becomes compile-time.** With `std`, x86 dispatch calls `is_x86_feature_detected!`. It picks the best kernel the running CPU reports. Without `std`, there is no runtime CPU detection. That capability lives in the `std`-gated `raw-cpuid`. So the ISA ladder falls back to `cfg!(target_feature = ...)` instead, and the build runs whatever its compile-time target features guarantee.

To get an accelerated x86 kernel from a `no_std` build, you must compile for it. For example, pass `-C target-cpu=native` or an explicit `-C target-feature=+avx512f`. Otherwise you get the scalar floor. On aarch64 and on wasm, this is already how selection works, so nothing is lost there.

**The env knobs are off.** Reading an environment variable needs `std`. Without it, `GEMMKIT_REQUIRE_ISA` is never consulted (dispatch always auto-selects), and every `GEMMKIT_*` tuning knob resolves straight to its compiled default. The programmatic `tuning::set_*` setters still work, so you retune a `no_std` build in code rather than through the environment. See [Tuning Knobs](Tuning_Knobs.md) for the setter layer.

**A per-call workspace replaces the pool.** The default thread-local packing pool is a `std` construct. Without `std`, there is no pool. Each call instead allocates a fresh `Workspace` for its scratch, and frees it on return. That is correct, but it allocates on every call. To reach a zero-allocation steady state, create a [`Workspace`](The_Unchecked_Tier.md) once, and thread it through the `*_with` entries: `gemm_with`, and the `_with` variant of every family. After the first sufficiently large call, those entries reuse the buffer with no further heap traffic.

## Building for WebAssembly

wasm32 has no runtime feature detection, so the `simd128` backend is selected by a compile-time `cfg`. The build must enable that target feature explicitly. If you forget it, the wasm build still compiles and runs correctly, but only on the scalar floor, which is much slower. Pass the flag through `RUSTFLAGS`:

```sh
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --target wasm32-wasip1 --no-default-features --features std
```

To run the result, you need a wasm runtime. gemmkit's CI uses `wasmtime`, so point Cargo's target runner at it. Sometimes you want to be certain the SIMD path is actually live, rather than silently falling back to scalar. Pin the ISA for that: `GEMMKIT_REQUIRE_ISA=simd128` turns a missing `+simd128` into a panic, instead of a quiet fallback. That is exactly what a test job wants. This pin needs `std`, which is already on in the wasm builds.

```sh
RUSTFLAGS="-C target-feature=+simd128" \
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --env GEMMKIT_REQUIRE_ISA=simd128" \
  cargo test --target wasm32-wasip1 --no-default-features --features std
```

Baseline `wasm32-wasip1` has no threads. If you build with `parallel` on for a baseline wasm target, gemmkit does not trap. An internal guard makes rayon unusable there, so `Parallelism::Rayon(_)` degrades to the serial loop instead. A portable wasm binary can therefore carry the `parallel` feature and simply run single-threaded, with no target-specific build.

## Threaded wasm

Real multithreading on wasm needs the threads-capable target and the matching feature:

```sh
RUSTFLAGS="-C target-feature=+simd128" \
CARGO_TARGET_WASM32_WASIP1_THREADS_RUNNER="wasmtime -W threads=y -W shared-memory=y -S threads=y" \
  cargo test --target wasm32-wasip1-threads \
  --no-default-features --features std,parallel,wasm_threads
```

The `wasm_threads` feature implies `parallel`, and it targets `wasm32-wasip1-threads`. It turns on gemmkit's dedicated wasm rayon pool. Because a wasm runtime cannot report a core count, the pool's width is not auto-derived. It comes from the `GEMMKIT_WASM_THREADS` knob instead, default 8, which both caps the auto worker count and sizes the pool. Set it to match the number of workers your runtime actually provisions. Everything else behaves exactly as on a native threaded build: blocking, the job list, and reproducibility.
