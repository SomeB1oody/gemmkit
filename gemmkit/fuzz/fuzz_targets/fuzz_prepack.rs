//! Fuzz target for prepack round-trips: `prepack_rhs` -> `gemm_packed_b`
//! (col-major-ish C) and `prepack_lhs` -> `gemm_packed_a` (row-major-ish C), each
//! gated against the naive reference at tolerance (not bitwise: the API allows
//! tiny/gemv last-ULP drift). Any panic = bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::PrepackPlan| {
    gemmkit_fuzz::run_prepack(plan);
});
