//! Fuzz target for plain GEMM: valid-by-construction problems differentially
//! checked against an f64/i32/complex naive reference. Since inputs are valid by
//! construction, ANY panic (a gate violation, a debug-assert, overflow, or an ASan
//! report) is a library bug: no catch_unwind
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::GemmPlan| {
    gemmkit_fuzz::run_gemm(plan);
});
