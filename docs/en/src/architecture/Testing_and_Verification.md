# Testing and Verification

A library whose headline promises are "bit-identical here, tolerance there, reproducible everywhere" lives or dies by how precisely its tests pin those words down. gemmkit's suites live in `gemmkit/tests/`, and the first structural decision is what is *not* a test: the performance harnesses are measurement tools, never CI gates. `tests/perf/` is the exhaustive internal investigation suite — `#[ignore]` benchmarks over a median-of-9 harness, serialized behind a shared lock because each one saturates every core — run manually when a change needs numbers. `gemmkit/benches/gemm_bench.rs` is the curated public `cargo bench` surface: criterion benchmarks in five headline groups (`sgemm`, `dtypes`, `gemv`, `prepacked`, `batched`) meant for `--save-baseline` regression tracking against the `gemm` crate and `matrixmultiply`. Neither can fail a merge, because performance assertions in shared CI runners assert mostly noise.

## Correctness, properties, conformance, fuzzing

The correctness suite (`tests/correctness/`) sweeps shapes x layouts x alpha/beta combinations against an independent `f64` reference GEMM. The reference and its accuracy machinery live once in `tests/oracle_common/` — element traits, deterministic fills, the `f64` reference, relative-Frobenius accuracy gates per element type — and are `#[path]`-included by both the correctness and property harnesses, so there is exactly one oracle to trust. On top of the oracle sweeps sit cross-checks against the external `gemm` crate (an independent implementation catching shared-blind-spot bugs the in-repo reference cannot), parallel bit-identity tests on the routes where that is promised, per-ISA kernel runs through the generic driver, and — easy to underrate — the safe API's exact panic wording, held by `#[should_panic(expected = ...)]` substrings like `"A.cols"` and `"aliases itself"`, so a validation message can't silently degrade into a less useful one.

The property tests generalize the sweeps: `tests/props_api.rs` (oracle accuracy, run-to-run bit determinism, serial == parallel, `beta == 0` overwrite semantics, broadcast strides, batched, panic guarantees), `tests/props_packed.rs` (prepacked-vs-plain bit-identity in the general regime and tolerance on the documented tiny/gemv exception set), and `tests/props_knobs.rs` (behavior under randomized knob settings) drive proptest over shapes, strides, and knob values.

One layer down, `tests/simd_conformance.rs` checks the L0 vocabulary itself: every ISA token the host supports is constructed directly — bypassing dispatch — and each `SimdOps` primitive, the homogeneous `KernelSimd` blanket, and the portable `fma_bvec` fallback are compared lane-by-lane against scalar references. This is where primitives the product kernels rarely touch (integer `reduce_sum`, `fnma`, the widen seam) get exercised at all; in-module sweeps such as the `requant_store` bit-equality tests in `gemmkit/src/simd.rs` do the same for the vector requantize contract. The suite has no proptest dependency, so it also runs on wasm and conformance-tests the compile-time `simd128` token.

Fuzzing lives in `gemmkit/fuzz/`, a nightly-only cargo-fuzz sub-workspace (its own workspace root, excluded from the stable build) with six libFuzzer targets: `fuzz_gemm` (valid-by-construction problems differentially checked against naive references — any panic is a library bug), `fuzz_batched`, `fuzz_prepack` and `fuzz_prepack_i8` (round-trips through the prepack APIs, the i8 one gated bit-exactly), `fuzz_api_validation` (adversarial geometry into the *checked* entries, where a documented `"gemmkit:"` panic is an accepted outcome and anything else is a validation gap), and `fuzz_knobs`, which sets every process-global tuning knob to adversarial values before each run — the target that mechanically finds arithmetic-overflow classes in the blocking model.

## Isolation discipline

Two kinds of global state make naive test organization racy, and the suite's layout is shaped around them.

Tuning knobs are process-global atomics. Every test that mutates one lives in a dedicated binary — `tests/tuning.rs` (setters), `tests/env.rs` (env resolution; it holds exactly one test so environment access is single-threaded by construction), `tests/props_knobs.rs`, `tests/deep_k_narrow.rs` (toggling `GEMMKIT_DEEP_KC_BYTES` to force each deep-k route) — because a separate binary is a separate process that cannot race another binary's knob state. *Within* each binary, libtest still runs tests concurrently, so every knob-touching test serializes under a per-binary `KNOB_LOCK` mutex and restores what it changed (the property binary adds an RAII guard that survives proptest's internal `catch_unwind`).

`GEMMKIT_REQUIRE_ISA` is stickier: dispatch memoizes it once per process. So there is one pin binary per value — `tests/env_isa_avx512f.rs`, `_vnni`, `_bf16`, `_scalar`, `_neon`, `_wasm`, plus `env_isa_garbage.rs` asserting the unknown-value panic. Each binary routes every test through a shared `Once` in `tests/env_isa_common/` that performs the single `set_var` before any dispatch resolves; since all tests in a binary pin the same value, it does not matter which one wins the race to the `Once`. The write deliberately *overrides* an inherited `GEMMKIT_REQUIRE_ISA`, which is what lets the SDE-pinned CI jobs below run these binaries and still exercise the real per-ISA routes.

Miri rounds out the memory-safety story where fuzzing's sanitizers stop: CI runs the scalar-path correctness suite (`miri_scalar_path`) and the complex negative-stride unchecked entry under Miri, interpreting the actual unsafe pointer arithmetic of the pack and microkernel paths. `cfg(miri)` detours exist only where Miri cannot interpret a hardware conversion, never to skip logic.

## The CI matrix

`.github/workflows/ci.yml` turns the pinning machinery into coverage of kernels the runners do not physically have:

| Job | What it exercises |
|---|---|
| `test` | Default features, then `--all-features`, then `parallel` off; `no_std`-style builds with `std` off in four feature combinations |
| `kernel-scalar` / `kernel-fma` | The full suite with `GEMMKIT_REQUIRE_ISA` pinned to each natively available kernel |
| `avx512f_test` / `avx512vnni_test` / `avx512bf16_test` | The suite under Intel SDE (`sde64 -spr`) with the AVX-512F, VNNI-dot, and BF16-dot kernels pinned — emulated silicon, real code paths |
| `kernel-neon` | The whole workspace natively on an arm64 macOS runner, then re-run with `neon` pinned |
| `wasm_simd128` / `wasm_simd128_threads` | Correctness and conformance under wasmtime on `wasm32-wasip1` with `simd128` pinned; the threads job runs real 8-way parallelism on `wasm32-wasip1-threads` |
| `no_std` | Builds for `x86_64-unknown-none`, `aarch64-unknown-none`, `wasm32-unknown-unknown` |
| `i686_check` / `msrv` / `lint` / `miri` / `coverage` | 32-bit check, Rust 1.89.0 build, fmt + clippy `-D warnings`, the Miri jobs above, and report-only `cargo-llvm-cov` with a pinned ISA list so the percentage cannot swing with the runner pool |

The SDE jobs run roughly 50x slower than native, which is where `GEMMKIT_FAST_TEST` earns its keep: it is a **test-suite-only** switch — implemented once in `tests/fast_test_common/` and included from the harnesses; the library never reads it — that shrinks the deterministic dimension/coefficient sweeps to one representative per redundant combination while still visiting every branch and path class. The SDE jobs set it (alongside `PROPTEST_CASES=16`); native jobs keep the full sweeps. Keeping it out of the library proper means no risk that a test-convenience flag changes shipping behavior.

The net effect ties the chapter together: every claim earlier pages made — the [special paths'](Special_Paths.md) bit-identity across worker counts, the [epilogue](Epilogue_Fusion.md) gemm-then-map equivalence, the [extension seams'](Extension_Points.md) open/closed property, each pinned kernel's correctness on hardware the project does not own — is held by a test you can point to, in a binary whose isolation rules make the result trustworthy.
