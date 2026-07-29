# Testing and Verification

A library whose headline promises are "bit-identical here, tolerance there, reproducible everywhere" lives or dies by how precisely its tests pin those words down. gemmkit's suites live in `gemmkit/tests/`. The first structural decision is what is *not* a test. The performance harnesses are measurement tools. They never gate CI.

`tests/perf/` is the exhaustive internal investigation suite. It runs `#[ignore]` benchmarks over a median-of-9 harness, serialized behind a shared lock, because each one saturates every core. Someone runs it by hand when a change needs numbers. `gemmkit/benches/gemm_bench.rs` is the curated public `cargo bench` surface instead. It holds criterion benchmarks in 5 headline groups, `sgemm`, `dtypes`, `gemv`, `prepacked`, and `batched`, meant for `--save-baseline` regression tracking against the `gemm` crate and `matrixmultiply`. Neither suite can fail a merge. A performance assertion on a shared CI runner would mostly assert noise.

## Correctness, properties, conformance, fuzzing

The correctness suite (`tests/correctness/`) sweeps shapes, layouts, and alpha/beta combinations against an independent `f64` reference GEMM. The reference, and the accuracy machinery around it, live once in `tests/oracle_common/`. This includes the element traits, the deterministic fills, the `f64` reference itself, and the relative-Frobenius accuracy gates for each element type. Both the correctness and the property harnesses include this module with `#[path]`, so there is exactly 1 oracle to trust.

On top of the oracle sweeps sit several more checks. Cross-checks against the external `gemm` crate catch shared-blind-spot bugs the in-repo reference cannot, since it is an independent implementation. Parallel bit-identity tests cover the routes where the library promises that agreement. Per-ISA kernel runs go through the generic driver. The safe API's exact panic wording, easy to underrate, gets held by `#[should_panic(expected = ...)]` substrings, such as `"A.cols"` and `"aliases itself"`. So a validation message cannot quietly degrade into a less useful one.

The property tests generalize these sweeps, and all 3 drive proptest over shapes, strides, and knob values. `tests/props_api.rs` covers oracle accuracy, run-to-run bit determinism, serial-equals-parallel agreement, the `beta == 0` overwrite semantics, broadcast strides, batching, and panic guarantees. `tests/props_packed.rs` covers prepacked-versus-plain bit-identity in the general regime, and tolerance on the documented tiny/gemv exception set. `tests/props_knobs.rs` covers behavior under randomized knob settings.

One layer down, `tests/simd_conformance.rs` checks the L0 vocabulary itself. It constructs every ISA token the host supports directly, bypassing dispatch. It then compares each `SimdOps` primitive, the homogeneous `KernelSimd` blanket, and the portable `fma_bvec` fallback, lane-by-lane, against scalar references. This is where primitives the product kernels rarely touch, such as integer `reduce_sum`, `fnma`, and the widen seam, get exercised at all. In-module sweeps, such as the `requant_store` bit-equality tests in `gemmkit/src/simd.rs`, do the same for the vector requantize contract. The suite has no proptest dependency, so it also runs on wasm and conformance-tests the compile-time `simd128` token.

Fuzzing lives in `gemmkit/fuzz/`, a nightly-only cargo-fuzz sub-workspace with its own workspace root, excluded from the stable build. It holds 6 libFuzzer targets. `fuzz_gemm` builds valid-by-construction problems and checks them differentially against naive references, so any panic there is a library bug. `fuzz_batched` does the same for batched calls. `fuzz_prepack` and `fuzz_prepack_i8` round-trip through the prepack APIs, with the i8 one gated bit-exactly. `fuzz_api_validation` throws adversarial geometry at the *checked* entries. There, a documented `"gemmkit:"` panic counts as an accepted outcome, and anything else counts as a validation gap. `fuzz_knobs` sets every process-global tuning knob to an adversarial value before each run. This is the target that mechanically finds arithmetic-overflow classes in the blocking model.

## Isolation discipline

A naive test layout turns racy around 2 kinds of global state, and the suite's structure exists to work around both.

Tuning knobs are process-global atomics. Every test that mutates one lives in its own dedicated binary. `tests/tuning.rs` holds the setters, `tests/env.rs` holds environment-variable resolution, and there is also `tests/props_knobs.rs` and `tests/deep_k_narrow.rs`, which toggles `GEMMKIT_DEEP_KC_BYTES` to force each deep-k route. `tests/env.rs` holds exactly 1 test, so its environment access is single-threaded by construction. A separate binary is a separate process, and it cannot race another binary's knob state.

*Within* one binary, though, libtest still runs tests concurrently. So every knob-touching test there serializes under a per-binary `KNOB_LOCK` mutex, and restores whatever it changed before it releases that lock. The property-test binary adds an RAII guard on top, one that survives proptest's internal `catch_unwind`.

`GEMMKIT_REQUIRE_ISA` is stickier still, because dispatch memoizes it once per process. So there is 1 pin binary per value: `tests/env_isa_avx512f.rs`, `_vnni`, `_bf16`, `_scalar`, `_neon`, and `_wasm`, plus `env_isa_garbage.rs`, which asserts the unknown-value panic. Each binary routes every test through a shared `Once` in `tests/env_isa_common/`. That `Once` performs the single `set_var` call before any dispatch resolves. Because every test in one binary pins the same value, it does not matter which test wins the race to run that `Once`. The write deliberately *overrides* an inherited `GEMMKIT_REQUIRE_ISA`. That override is what lets the SDE-pinned CI jobs below run these same binaries and still exercise the real, per-ISA routes.

Miri rounds out the memory-safety story where fuzzing's sanitizers stop. CI runs the scalar-path correctness suite (`miri_scalar_path`) and the complex negative-stride unchecked entry under Miri. Miri interprets the actual unsafe pointer arithmetic of the pack and microkernel paths directly. A `cfg(miri)` detour exists only where Miri cannot interpret a hardware conversion. It never exists to skip logic.

## The CI matrix

`.github/workflows/ci.yml` turns this pinning machinery into coverage of kernels the runners do not physically have:

| Job | What it exercises |
|---|---|
| `test` | Default features, then `--all-features`, then `parallel` off. `no_std`-style builds with `std` off, in 4 feature combinations. |
| `kernel-scalar` / `kernel-fma` | The full suite with `GEMMKIT_REQUIRE_ISA` pinned to each natively available kernel. |
| `avx512f_test` / `avx512vnni_test` / `avx512bf16_test` | The suite under Intel SDE (`sde64 -spr`), with the AVX-512F, VNNI-dot, and BF16-dot kernels pinned. SDE emulates the silicon, but the code paths are real. |
| `kernel-neon` | The whole workspace, run natively on an arm64 macOS runner, then run again with `neon` pinned. |
| `wasm_simd128` / `wasm_simd128_threads` | Correctness and conformance under wasmtime on `wasm32-wasip1`, with `simd128` pinned. The threads job runs real 8-way parallelism on `wasm32-wasip1-threads`. |
| `no_std` | Builds for `x86_64-unknown-none`, `aarch64-unknown-none`, and `wasm32-unknown-unknown`. |
| `i686_check` / `msrv` / `lint` / `miri` / `coverage` | A 32-bit check, a build on Rust 1.89.0 (the minimum supported version), `fmt` plus `clippy -D warnings`, and the Miri jobs above.<br>Coverage is report-only: `cargo-llvm-cov` with a pinned ISA list, so the reported percentage cannot swing with the runner pool. |

SDE emulation runs far slower than native execution. This is where `GEMMKIT_FAST_TEST` earns its keep. It is a **test-suite-only** switch, implemented once in `tests/fast_test_common/` and included from the harnesses. The library itself never reads it. The switch shrinks the deterministic dimension and coefficient sweeps down to 1 representative per redundant combination, while still visiting every branch and path class. The SDE jobs set it, alongside `PROPTEST_CASES=16`. Native jobs keep the full sweeps instead. Keeping this switch out of the library proper means a test-convenience flag can never change shipping behavior.

The net effect ties this chapter together. Every claim the earlier pages made is held by a test you can point to, in a binary whose isolation rules make the result trustworthy. This includes the bit-identity guarantees in [Special Paths](Special_Paths.md), the gemm-then-map equivalence in [Epilogue Fusion](Epilogue_Fusion.md), and the open/closed property in [Extension Points](Extension_Points.md). It also includes each pinned kernel's correctness on hardware the project does not own.
