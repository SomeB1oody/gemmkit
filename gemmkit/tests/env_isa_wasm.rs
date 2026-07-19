//! `GEMMKIT_REQUIRE_ISA=wasm` cross-architecture pin
//!
//! `wasm` is the accepted alias for `simd128`, so this exercises both that alias's parse arm and
//! every dtype's "not wasm32 with +simd128" panic arm. Only builds/runs on native (non-wasm)
//! targets, where the pin is always unsatisfiable, so every dtype ladder must panic. Its own
//! single-test binary so the one `set_var` below runs before any dispatch. Each dtype's dispatch
//! memoizes into its own `OnceLock`, and a panicking initializer leaves that lock unset, so 1
//! process can drive every dtype's ladder to completion even though each one panics identically
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

// Tiny 2x2 GEMM entry point per dtype, shared by the cross-arch ISA pin binaries
mod isa_dtypes;

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn wasm_pin_sweeps_every_dtype_ladder() {
    // SAFETY: the only test in this binary, so no other thread reads the environment
    // concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "wasm");
    }
    // Suppress the expected panics' default output while sweeping, then restore the hook so a
    // genuine assertion failure below still prints its diagnostics
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let results: Vec<(&str, bool)> = isa_dtypes::dtype_cases()
        .into_iter()
        .map(|(name, f)| (name, catch_unwind(AssertUnwindSafe(f)).is_err()))
        .collect();
    std::panic::set_hook(prev);

    // Native target: simd128 is never available, so every ladder must have panicked
    for (name, panicked) in results {
        assert!(
            panicked,
            "GEMMKIT_REQUIRE_ISA=wasm dtype {name} must panic on a native target"
        );
    }
}
