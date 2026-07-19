//! `GEMMKIT_REQUIRE_ISA=neon` cross-architecture pin
//!
//! Its own single-test binary (a separate process) so the one `set_var` below runs before any
//! dispatch and cannot race another test's pin. Off aarch64 the `neon` pin is unsatisfiable, so
//! every dtype's `select_*` ladder must hit its "not aarch64" panic arm; on aarch64 the pin is
//! valid and every ladder must succeed instead. Each dtype's dispatch memoizes into its own
//! `OnceLock`, and a panicking initializer leaves that lock unset, so 1 process can drive every
//! dtype's ladder to completion even though each one panics identically. Only meaningful with
//! `std`: a no-`std` build never reads the environment
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

// Tiny 2x2 GEMM entry point per dtype, shared by the cross-arch ISA pin binaries
mod isa_dtypes;

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn neon_pin_sweeps_every_dtype_ladder() {
    // SAFETY: the only test in this binary, so no other thread reads the environment
    // concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "neon");
    }
    // On aarch64 the pin is satisfiable and every ladder should succeed instead of panic
    let expect_panic = !cfg!(target_arch = "aarch64");

    // Suppress the expected panics' default output while sweeping, then restore the hook so a
    // genuine assertion failure below still prints its diagnostics
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let results: Vec<(&str, bool)> = isa_dtypes::dtype_cases()
        .into_iter()
        .map(|(name, f)| (name, catch_unwind(AssertUnwindSafe(f)).is_err()))
        .collect();
    std::panic::set_hook(prev);

    for (name, panicked) in results {
        assert_eq!(
            panicked, expect_panic,
            "GEMMKIT_REQUIRE_ISA=neon dtype {name}: panicked={panicked}, expected={expect_panic}"
        );
    }
}
