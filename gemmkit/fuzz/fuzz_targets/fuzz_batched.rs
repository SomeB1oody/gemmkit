#![no_main]

use libfuzzer_sys::fuzz_target;

// Valid-by-construction strided-batched problems (batch strides valid by
// construction, broadcast A/B allowed) checked element-wise against the naive
// reference, plus a gemm_batched_slice cross-check. Any panic = bug.
fuzz_target!(|plan: gemmkit_fuzz::BatchedPlan| {
    gemmkit_fuzz::run_batched(plan);
});
