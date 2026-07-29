//! Root of the `perf` integration-test binary: `#[ignore]`d throughput benchmarks, meant
//! to be run and read by hand, not correctness gates
//!
//! Native benches (`bandwidth`, `batched`, `dtypes`, `prepack`, `sgemm`, `small_mn`)
//! compare gemmkit against the `gemm` crate and/or `matrixmultiply`; those dev-dependencies
//! are excluded on wasm, so those modules are too. The wasm bench (`simd128`) has no
//! external crate to compare against on that target, so it instead measures gemmkit's
//! `simd128` token against its own scalar token. This whole file is `cfg(not(miri))`:
//! Miri cannot execute the target-feature-gated SIMD intrinsics these benches drive
//!
//! Every bench saturates all available cores, so each takes the shared `BENCH_GUARD` lock
//! as its first line to keep 2 from corrupting each other's numbers. Run with:
//!   cargo test -p gemmkit --release --test perf -- --ignored --nocapture
//! Run the wasm benchmark (compile-time `+simd128`) under a wasm runtime:
//!   RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//!     cargo test -p gemmkit --release --target wasm32-wasip1 \
//!       --no-default-features --features std --test perf -- --ignored --nocapture
#![cfg(not(miri))]

// Shared harness (BENCH_GUARD, fill, measure/measure_gbps, Stat, the native-ISA token)
mod harness;

// Bandwidth-bound shapes: STREAM Triad/Copy ceilings, gemv (axpy/dot/mixed), gevv, small-k
mod bandwidth;
// Batched GEMM (gemm_batched) vs naive gemm() loops, serial and parallel
mod batched;
// Fused-epilogue overhead vs plain gemm, same shape and ISA; asserts a ratio bound
#[cfg(feature = "epilogue")]
mod fused;
// f16 / bf16 / i8 / c32 element-type throughput, each vs its available external baseline
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
// wasm simd128 vs the scalar token, single-threaded
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
// Small-m,n horizontal (inner-product) route benches
#[cfg(not(target_family = "wasm"))]
mod small_mn;
