//! `GEMMKIT_REQUIRE_ISA=<garbage>` must be a hard error (catches typos in CI config). Its own
//! single-test `should_panic` binary so the one `set_var` runs before any dispatch and cannot race
//! other tests. The unknown-value arm of the pin parser fires the 1st time any `select_*` ladder
//! resolves the pin. Only meaningful with `std`
#![cfg(all(feature = "std", not(target_family = "wasm"), not(miri)))]

use gemmkit::{MatMut, MatRef, Parallelism};

#[test]
#[should_panic(expected = "unknown value")]
fn garbage_pin_is_a_hard_error() {
    // Set the pin exactly once, before any dispatch. SAFETY: the only test in this binary, so no
    // other thread reads the environment concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "definitely-not-an-isa");
    }
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    // The 1st dispatch resolves the pin and panics on the unknown-value arm
    gemmkit::gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}
