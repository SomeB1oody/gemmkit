//! Correctness suite: numerical accuracy, full shape / layout / alpha-beta
//! coverage, parallel/serial reproducibility, per-ISA kernels, gemv, and the safe
//! API's panic guarantees.

#![allow(clippy::too_many_arguments)]

mod common;

mod api;
#[cfg(feature = "complex")]
mod complex;
mod float;
#[cfg(feature = "int8")]
mod int8;
mod isa;
#[cfg(feature = "half")]
mod mixed;
mod packed;
