//! Shared harness for the gemmkit fuzz targets: the 6 `fuzz_target!` binaries under
//! `fuzz_targets/` are thin wrappers (`fuzz_target!(|p| run_x(p))`) that decode an
//! `Arbitrary` plan and hand it to a driver defined here, so the differential-oracle
//! logic lives in 1 testable place. 5 of the 6 feed valid-by-construction problems and
//! treat any panic as a bug; `fuzz_api_validation` instead drives adversarial geometry
//! into the checked APIs and only accepts a documented `gemmkit:` panic
//!
//! The accuracy bars mirror `gemmkit/tests/correctness/`: the `8*k*EPS` / `16*k*EPS`
//! relative-Frobenius gates, the `beta == 0` "C is not read" rule, and the exact
//! wrapping-i32 reference for i8. Every `Plan` is built by a manual `Arbitrary` impl
//! (so its fields are already bounded/resolved) and derives `Debug`, so a crash
//! artifact decoded with `cargo fuzz fmt` prints as a literal ready to paste into a
//! stable regression test

// Adversarial-geometry plan and driver behind fuzz_api_validation
mod api_validation;
// The Rng, per-element traits, and strided operand builders every driver shares
mod common;
// Differential drivers: run a gemmkit entry point and gate it against the reference
mod differential;
// Valid-by-construction plans/entries for fuzz_gemm, fuzz_knobs, fuzz_batched, fuzz_prepack, fuzz_prepack_i8
mod plans;
// Triple-loop references and the tolerance/exact gates the drivers check output against
mod reference;

pub use api_validation::{DimClass, EntryKind, StrideClass, ValidationPlan, drive_validation};
pub use common::LayoutPlan;
pub use plans::{
    BatchedPlan, GemmPlan, KnobsPlan, PrepackI8Plan, PrepackPlan, Scenario, TypeTag, run_batched,
    run_gemm, run_knobs, run_prepack, run_prepack_i8,
};
