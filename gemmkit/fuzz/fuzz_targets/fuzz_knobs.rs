//! Fuzz target that sets every process-global tuning knob gemmkit compiles under this
//! crate's features (`gemmkit::tuning::set_*`) to an adversarial value class on every
//! input. It then runs one small differential GEMM scenario against the naive reference.
//! Value classes include 0, 1, tile-boundary sizes, and near or at `usize::MAX`. These
//! probe the blocking-model arithmetic (kc/mc/nc derivation, thread striding) for
//! overflow. The contract is: for any knob values, no panic, no undefined behavior, and
//! a result inside the type's gate. So there is no catch_unwind
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::KnobsPlan| {
    gemmkit_fuzz::run_knobs(plan);
});
