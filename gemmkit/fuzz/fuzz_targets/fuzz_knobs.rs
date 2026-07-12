#![no_main]

use libfuzzer_sys::fuzz_target;

// Sets all 22 process-global tuning knobs to adversarial value classes every input,
// then runs one small differential scenario. This is the target that auto-finds the
// arithmetic-overflow bug classes in the blocking model. Contract: for ANY knob
// values, no panic / no UB / result within the type's gate — so no catch_unwind.
fuzz_target!(|plan: gemmkit_fuzz::KnobsPlan| {
    gemmkit_fuzz::run_knobs(plan);
});
