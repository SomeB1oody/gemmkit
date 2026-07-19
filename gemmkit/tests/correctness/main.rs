//! Entry point for the `correctness` integration test binary: wires up every
//! per-family test module below and re-exports the shared harness they all use

#![allow(clippy::too_many_arguments)]

// Element traits, random fills, matrix views, f64 references, and accuracy gates
mod common;

// Safe-API panic guarantees and cache-topology sanity checks
mod api;
// Complex (c32/c64) GEMM: conj variants, gemm-crate cross-check
#[cfg(feature = "complex")]
mod complex;
// Real float GEMM: shapes, layouts, alpha/beta, workspace reuse, parallel/serial bit-identity, gemv
mod float;
// Integer (i8 -> i32) GEMM, including overflow-wrap and small_mn routing
#[cfg(feature = "int8")]
mod int8;
// Per-ISA kernel checks through the generic driver, plus the Miri scalar-only suite
mod isa;
// Mixed-precision (f16/bf16) GEMM: shapes, gemv, gemm-crate cross-check
#[cfg(feature = "half")]
mod mixed;
// Prepacked LHS/RHS GEMM against plain gemm, and their orientation/shape panics
mod packed;
