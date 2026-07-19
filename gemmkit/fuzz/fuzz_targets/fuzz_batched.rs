//! Fuzz target for strided-batched GEMM (`gemm_batched`): builds a valid-by-construction
//! `BatchedPlan` (f32 or f64, batch size 0-4, optional zero-stride broadcast of A/B,
//! padded batch strides) and gates each batch element's output against a per-element
//! naive reference at tolerance, then cross-checks the `gemm_batched_slice`
//! pointer-array entry point on the same problem over separate per-element buffers.
//! Every problem is valid by construction, so any panic is a bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::BatchedPlan| {
    gemmkit_fuzz::run_batched(plan);
});
