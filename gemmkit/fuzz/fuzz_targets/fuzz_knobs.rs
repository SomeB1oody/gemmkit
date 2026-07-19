//! Fuzz target that sets every one of gemmkit's 24 process-global tuning knobs
//! (`gemmkit::tuning::set_*`) to an adversarial value class on every input, then runs
//! one small differential GEMM scenario against the naive reference. Values include 0,
//! 1, tile-boundary sizes, and near/at `usize::MAX`, probing the blocking-model
//! arithmetic (kc/mc/nc derivation, thread striding) for overflow. Contract: for ANY
//! knob values, no panic / no UB / result within the type's gate, so no catch_unwind
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::KnobsPlan| {
    gemmkit_fuzz::run_knobs(plan);
});
