//! `GEMMKIT_REQUIRE_ISA=wasm` cross-arch pin. `wasm` is the accepted alias for `simd128`; this
//! binary drives the parse arm for that alias plus every dtype's `simd128`-on-non-wasm panic arm.
//! Its own single-test binary so the one `set_var` runs before any dispatch. This binary only
//! builds/runs on native (non-wasm) targets — where the `simd128` pin is always unsatisfiable — so
//! every dtype ladder must panic. Each ladder is a distinct `OnceLock`, and a panicking init leaves
//! it uninitialized, so one process sweeps all supported dtypes.
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

mod isa_dtypes;

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn wasm_pin_sweeps_every_dtype_ladder() {
    // Set the pin exactly once, before any dispatch. SAFETY: this is the only test in this binary,
    // so no other thread reads the environment concurrently with this write.
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "wasm");
    }
    // Silence the intentional pin panics only while sweeping, then restore the hook so an
    // assertion failure below still prints its diagnostics.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let results: Vec<(&str, bool)> = isa_dtypes::dtype_cases()
        .into_iter()
        .map(|(name, f)| (name, catch_unwind(AssertUnwindSafe(f)).is_err()))
        .collect();
    std::panic::set_hook(prev);

    // This binary runs only on native targets, where `simd128` is never available: every ladder
    // must panic on its "not wasm32 with +simd128" arm.
    for (name, panicked) in results {
        assert!(
            panicked,
            "GEMMKIT_REQUIRE_ISA=wasm dtype {name} must panic on a native target"
        );
    }
}
