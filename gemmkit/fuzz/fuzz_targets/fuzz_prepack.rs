//! Fuzz target for prepack round trips over f32, f64, and bf16. `prepack_rhs` feeds
//! `gemm_packed_b`, which requires column-major-ish C. `prepack_lhs` feeds
//! `gemm_packed_a`, which requires row-major-ish C. Each gates against the naive
//! reference at tolerance rather than bit-exact. The packed path may differ from plain
//! `gemm` by the last ULP on tiny or gemv-shaped products. Every problem is valid by
//! construction, so any panic is a bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::PrepackPlan| {
    gemmkit_fuzz::run_prepack(plan);
});
