//! Fuzz target for the i8 prepack round-trip: `prepack_rhs_i8` -> `gemm_i8_packed_b`
//! (col-major-ish C), gated EXACTLY (integer GEMM is exact): the packed output must
//! equal both the wrapping-i32 reference and a plain `gemm_i8` bit-for-bit. Any panic = bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::PrepackI8Plan| {
    gemmkit_fuzz::run_prepack_i8(plan);
});
