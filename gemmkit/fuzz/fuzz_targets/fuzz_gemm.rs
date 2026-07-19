//! Fuzz target for plain GEMM across every scalar type gemmkit supports: f32/f64/f16/
//! bf16 and complex c32/c64 via `gemm`/`gemm_cplx`, i8 via `gemm_i8`, plus the
//! caller-owned `Workspace` reuse path (`gemm_with`). Every `GemmPlan` is valid by
//! construction and gated against an f64 (wrapping-i32 for i8) naive reference, so ANY
//! panic (a gate violation, a debug assert, an overflow, an ASan report) is a library
//! bug: no catch_unwind
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|plan: gemmkit_fuzz::GemmPlan| {
    gemmkit_fuzz::run_gemm(plan);
});
