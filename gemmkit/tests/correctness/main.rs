//! Correctness suite: numerical accuracy, full shape / layout / alpha-beta
//! coverage, parallel/serial reproducibility, per-ISA kernels, gemv, and the safe
//! API's panic guarantees

#![allow(clippy::too_many_arguments)]

// Shared test harness: element traits, random fills, views, references, accuracy gates
mod common;

// Safe-API panic guarantees and cache-topology sanity
mod api;
// Complex GEMM correctness (c32/c64, conj variants)
#[cfg(feature = "complex")]
mod complex;
// Float shapes x layouts x alpha/beta, workspace reuse, parallel bit-identity, gemv shapes
mod float;
// Integer GEMM (i8 -> i32) correctness
#[cfg(feature = "int8")]
mod int8;
// Per-ISA kernels via the generic driver, plus the Miri scalar-path suite
mod isa;
// Mixed-precision (f16/bf16) correctness and gemm-crate cross-check
#[cfg(feature = "half")]
mod mixed;
// Prepacked LHS/RHS vs plain gemm: bit-identity, both-tiny accuracy, packed-orientation panics
mod packed;
