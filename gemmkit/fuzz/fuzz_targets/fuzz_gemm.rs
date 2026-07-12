#![no_main]

use libfuzzer_sys::fuzz_target;

// Valid-by-construction problems differentially checked against an f64/i32/complex
// naive reference. Inputs are valid by construction, so ANY panic (gate violation,
// debug-assert, overflow, or ASan report) is a library bug — no catch_unwind.
fuzz_target!(|plan: gemmkit_fuzz::GemmPlan| {
    gemmkit_fuzz::run_gemm(plan);
});
