//! Fuzz target for strided-batched GEMM: valid-by-construction problems (batch
//! strides always valid, broadcast A/B allowed) checked element-wise against the
//! naive reference, plus a `gemm_batched_slice` cross-check. Any panic = bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::BatchedPlan| {
    gemmkit_fuzz::run_batched(plan);
});
