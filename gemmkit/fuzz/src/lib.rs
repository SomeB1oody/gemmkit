//! Shared harness for the gemmkit fuzz targets.
//!
//! Everything the five targets need lives here so the target files stay thin
//! (`fuzz_target!(|p| run_x(p))`) and the differential-oracle logic is in one
//! testable place. Four targets feed *valid-by-construction* problems and treat
//! **any** panic as a bug; `fuzz_api_validation` instead drives adversarial
//! geometry into the checked APIs and accepts documented `gemmkit:` panics.
//!
//! Numerical bars mirror `gemmkit/tests/correctness/`: the `8·k·EPS` /
//! `16·k·EPS` relative-Frobenius gates, the `beta == 0` "C is not read" rule, and
//! the exact wrapping-i32 reference for `i8`. Each `Plan` carries already-bounded,
//! resolved values (manual `Arbitrary`), so a crash artifact decoded with
//! `cargo fuzz fmt` is directly translatable into a stable regression test.

mod api_validation;
mod common;
mod differential;
mod plans;
mod reference;

pub use api_validation::{DimClass, EntryKind, StrideClass, ValidationPlan, drive_validation};
pub use common::LayoutPlan;
pub use plans::{
    BatchedPlan, GemmPlan, KnobsPlan, PrepackPlan, Scenario, TypeTag, run_batched, run_gemm,
    run_knobs, run_prepack,
};
