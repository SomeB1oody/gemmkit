//! Fuzz target for the i8 prepack round trip. `prepack_rhs_i8` feeds
//! `gemm_i8_packed_b`, which requires column-major-ish C. The gate is exact rather
//! than by tolerance, since integer GEMM has no rounding to absorb. The packed output
//! must equal both the wrapping-i32 reference and a plain `gemm_i8` call bit-for-bit,
//! per the API's documented bit-identity. Every problem is valid by construction, so
//! any panic is a bug
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::PrepackI8Plan| {
    gemmkit_fuzz::run_prepack_i8(plan);
});
