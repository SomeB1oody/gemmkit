//! Fuzz target for prepack round trips over f32/f64/bf16: `prepack_rhs` -> `gemm_packed_b`
//! (requires column-major-ish C) and `prepack_lhs` -> `gemm_packed_a` (requires
//! row-major-ish C), each gated against the naive reference at tolerance rather than
//! bit-exact, since the packed path may differ from plain `gemm` by the last ULP on
//! tiny or gemv-shaped products. Every problem is valid by construction, so any panic
//! is a bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::PrepackPlan| {
    gemmkit_fuzz::run_prepack(plan);
});
