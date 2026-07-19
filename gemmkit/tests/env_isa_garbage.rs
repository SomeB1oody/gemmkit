//! An unrecognized `GEMMKIT_REQUIRE_ISA` value must panic rather than silently fall back to
//! auto-selection, so a typo in a pinned CI job's config fails loudly instead of masking itself
//!
//! Its own single-test binary so the one `set_var` below runs before any dispatch in the process
//! and cannot race a pin set by another test. Only meaningful with `std`: the pin is parsed from
//! the environment, which a no-`std` build never reads
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

use gemmkit::{MatMut, MatRef, Parallelism};

#[test]
#[should_panic(expected = "unknown value")]
fn garbage_pin_is_a_hard_error() {
    // SAFETY: the only test in this binary, so no other thread reads the environment
    // concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "definitely-not-an-isa");
    }
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    // This 1st dispatch resolves and memoizes the pin, hitting the unknown-value panic
    gemmkit::gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}
