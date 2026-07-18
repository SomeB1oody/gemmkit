//! Performance suite: `#[ignore]` benchmarks (not correctness gates), run manually
//!
//! * **Native cross-library benchmarks**: gemmkit vs the `gemm` crate / `matrixmultiply`.
//!   These depend on those dev-deps, which **do not build for wasm** (they are
//!   `cfg(all(not(miri), not(target_family = "wasm")))` dev-deps, see `Cargo.toml`), so
//!   each bench that calls them is individually gated `cfg(not(target_family = "wasm"))`.
//! * **The wasm `simd128` benchmark** (`perf_simd128`): simd128 vs the scalar token,
//!   mirroring the native `NativeTok` + `bench_native_equal_isa` pattern with the *scalar
//!   token* as the reference (no external crate on wasm). The shared harness
//!   (`fill`/`measure`/`gflops`/`Stat`) is `std`-only, so it serves both worlds and the
//!   file compiles on wasm. (Correctness of the simd128 path is gated separately by
//!   `isa_simd128` in `tests/correctness/isa.rs`; this is the throughput sanity print.)
//!
//! The whole file compiles away under Miri. The benchmarks each saturate every core, so
//! they must not run concurrently: they take a shared `BENCH_GUARD` lock, so even the
//! default multi-threaded harness serializes them and `--test-threads=1` is optional.
//! Run them with:
//!   cargo test -p gemmkit --release --test perf -- --ignored --nocapture
//! Run the wasm benchmark (compile-time `+simd128`) under a wasm runtime:
//!   RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//!     cargo test -p gemmkit --release --target wasm32-wasip1 \
//!       --no-default-features --features std --test perf -- --ignored --nocapture
#![cfg(not(miri))]

// Shared bench harness: BENCH_GUARD, fill/measure/Stat, native-ISA token
mod harness;

// Bandwidth-bound shapes: STREAM ceilings, gemv, gevv, small-k, gemv scaling
mod bandwidth;
// Batched GEMM vs naive gemm() loops
mod batched;
// f16 / i8 / c32 element-type throughput benches
#[cfg(all(
    not(target_family = "wasm"),
    any(feature = "half", feature = "int8", feature = "complex")
))]
mod dtypes;
// Prepacked-RHS/LHS reuse, prepack buffer setup cost, shared-LHS gate sweep
mod prepack;
// f32 sgemm vs gemm crate / matrixmultiply, thread-scaling, per-call latency
#[cfg(not(target_family = "wasm"))]
mod sgemm;
// wasm simd128 vs scalar-token throughput
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
// Small-m,n horizontal (inner-product) route benches
#[cfg(not(target_family = "wasm"))]
mod small_mn;
