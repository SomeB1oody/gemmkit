//! Shared harness for the gemmkit fuzz targets
//!
//! Everything the 5 targets need lives here so the target files stay thin
//! (`fuzz_target!(|p| run_x(p))`) and the differential-oracle logic is in one
//! testable place. 4 targets feed *valid-by-construction* problems and treat
//! **any** panic as a bug; `fuzz_api_validation` instead drives adversarial
//! geometry into the checked APIs and accepts documented `gemmkit:` panics
//!
//! Numerical bars mirror `gemmkit/tests/correctness/`: the `8*k*EPS` /
//! `16*k*EPS` relative-Frobenius gates, the `beta == 0` "C is not read" rule, and
//! the exact wrapping-i32 reference for `i8`. Each `Plan` carries already-bounded,
//! resolved values (manual `Arbitrary`), so a crash artifact decoded with
//! `cargo fuzz fmt` is directly translatable into a stable regression test

// Adversarial-geometry plan and driver for fuzz_api_validation
mod api_validation;
// RNG, element/canary traits, and strided-layout operand construction
mod common;
// Differential drivers gating gemmkit output against the naive reference
mod differential;
// Valid-by-construction plans/entries for fuzz_gemm, fuzz_knobs, fuzz_batched, fuzz_prepack
mod plans;
// Naive dense references and the tolerance/exact result gates
mod reference;

pub use api_validation::{DimClass, EntryKind, StrideClass, ValidationPlan, drive_validation};
pub use common::LayoutPlan;
pub use plans::{
    BatchedPlan, GemmPlan, KnobsPlan, PrepackPlan, Scenario, TypeTag, run_batched, run_gemm,
    run_knobs, run_prepack,
};
