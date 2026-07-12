//! `GEMMKIT_REQUIRE_ISA=neon` cross-arch pin. Its own single-test binary (a separate process) so
//! the one `set_var` runs before any dispatch and cannot race the other pin/knob tests. On a
//! non-aarch64 host the `neon` pin is unsatisfiable, so every dtype's `select_*` ladder hits its
//! "not aarch64" panic arm; on aarch64 the pin is valid and the same ladders succeed. Each ISA
//! ladder is a distinct `OnceLock`, and a panicking init leaves it uninitialized, so one process
//! sweeps all supported dtypes. Only meaningful with `std` (no-`std` never reads the environment).
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

mod isa_dtypes;

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn neon_pin_sweeps_every_dtype_ladder() {
    // Set the pin exactly once, before any dispatch. SAFETY: this is the only test in this binary,
    // so no other thread reads the environment concurrently with this write.
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "neon");
    }
    // On aarch64 the `neon` pin is valid; elsewhere every ladder panics on the "not aarch64" arm.
    let expect_panic = !cfg!(target_arch = "aarch64");

    // Silence the intentional pin panics only while sweeping, then restore the hook so an
    // assertion failure below still prints its diagnostics.
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
