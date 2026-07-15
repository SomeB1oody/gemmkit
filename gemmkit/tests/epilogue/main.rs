//! Fused-epilogue suite (spec section 10): the determinism/precision contract for `gemm_fused`
//! (bias + ReLU/LeakyReLU) and `gemm_i8_requant`
//!
//! Every comparison is **bitwise** (raw bit patterns, so NaN/-0 are exercised). The oracle is
//! "plain GEMM, then the exact scalar map": a fused GEMM must equal it bit-for-bit, for **every**
//! shape: the fused engine routes each shape through the *same* kernel `gemm` uses (the general
//! driver, gemv, small-`m,n`, or small-`k`), fusing the epilogue into that route without
//! perturbing its accumulation order. All shapes are platform-independent; no machine numbers
//!
//! The suite is split by concern: [`common`] holds the harness (RNG, the `Flt` element trait, the
//! exact reference map, C-layout helpers, and the driver-shape `check_fused` oracle); [`float`]
//! covers the general-driver fused tests; [`special`] the gemv / small-`m,n` / small-`k` routes;
//! and [`requant`] the `i8 -> i8` requantize path (whose exact `i32` accumulation makes its oracle
//! hold bitwise under every `GEMMKIT_REQUIRE_ISA` pin)

// This whole suite exercises the fused-epilogue / requantize surface, which is gated behind the
// `epilogue` cargo feature; compile it away entirely when the feature is off
#![cfg(feature = "epilogue")]
// The index loops walk C and the bias vectors at different (strided) offsets, so explicit
// indices read clearer than zipped iterators here
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

// deterministic RNG, `Flt` element trait, reference epilogue map, `check_fused` oracle
mod common;

// strided-batched fused-epilogue tests
mod batched;
// complex-domain fused-epilogue tests (bias only, no activation)
#[cfg(feature = "complex")]
mod cplx;
// general-driver fused-epilogue tests (f32/f64)
mod float;
// user-defined per-element map-epilogue tests (gemm_map, f32/f64)
mod map;
// narrow-float (f16/bf16) fused-epilogue tests
#[cfg(feature = "half")]
mod mixed;
// prepacked-operand fused-epilogue tests (gemm_packed_{a,b}_fused)
mod packed;
// i8 -> i8/u8 requantize epilogue tests
#[cfg(feature = "int8")]
mod requant;
// gemv / small-m,n / small-k route fused-epilogue tests
mod special;
