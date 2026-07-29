//! Root of the fused-epilogue integration-test binary: every comparison is **bitwise** against a
//! self-computed oracle, plain GEMM then the exact scalar map applied by hand. Covers
//! `gemm_fused` (bias + ReLU/LeakyReLU), `gemm_map` (user closures), `gemm_batched_fused`,
//! `gemm_cplx_fused`, and `gemm_i8_requant`
//!
//! Split by concern: [`common`] is the shared harness; [`float`] and [`special`] cover
//! `gemm_fused` (general driver, then gemv / small-`m,n` / small-`k`); [`requant`] covers the
//! `i8 -> i8`/`u8` requantize path

// The whole suite exercises the epilogue cargo feature's surface, so compile it away entirely
// when that feature is off
#![cfg(feature = "epilogue")]
// C and the bias vectors are walked at independent strided offsets, so explicit indices read
// clearer here than zipped iterators would
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

// shared harness: RNG, Flt element trait, reference epilogue map, check_fused oracle
mod common;

// strided-batched gemm_batched_fused tests
mod batched;
// complex gemm_cplx_fused tests (bias only, no activation: ordering is undefined on complex)
#[cfg(feature = "complex")]
mod cplx;
// general-driver gemm_fused tests (f32/f64)
mod float;
// user-defined per-element gemm_map tests (f32/f64)
mod map;
// narrow-float (f16/bf16) gemm_fused tests
#[cfg(feature = "half")]
mod mixed;
// prepacked-operand gemm_packed_{a,b}_fused tests
mod packed;
// i8 -> i8/u8 requantize epilogue tests
#[cfg(feature = "int8")]
mod requant;
// gemv / small-m,n / small-k route gemm_fused tests
mod special;
