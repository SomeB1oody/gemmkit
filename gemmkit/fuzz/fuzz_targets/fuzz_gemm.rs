//! Fuzz target for plain GEMM across every scalar type gemmkit supports. f32, f64, f16,
//! bf16, and complex c32/c64 go through `gemm` or `gemm_cplx`, and i8 goes through
//! `gemm_i8`. The plan also exercises the caller-owned `Workspace` reuse path via
//! `gemm_with`. Every `GemmPlan` is valid by construction and gated against an f64,
//! wrapping-i32 for i8, naive reference. So any panic is a library bug: a gate
//! violation, a debug assert, an overflow, or an ASan report. There is no catch_unwind
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::GemmPlan| {
    gemmkit_fuzz::run_gemm(plan);
});
