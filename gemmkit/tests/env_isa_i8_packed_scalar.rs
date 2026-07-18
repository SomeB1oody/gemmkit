//! `GEMMKIT_REQUIRE_ISA=scalar` prepacked-i8 bit-parity pin. Its own single-test binary (a
//! separate process) so the one `set_var` runs before any i8 dispatch and cannot race the other
//! pin/knob tests. Scalar is valid on every architecture, so no host feature gate is needed: it
//! forces the widen scalar kernel's plain-panel prepack layout, deterministically, regardless of
//! what the auto path would pick. Only meaningful with `std` (no-`std` never reads the environment)
#![cfg(all(feature = "std", feature = "int8", not(miri)))]

// Shared prepacked-i8 bit-parity check driven by the ISA pin binaries
mod i8_packed_common;

#[test]
fn scalar_pin_i8_packed_matches_plain() {
    // Set the pin exactly once, before any dispatch. SAFETY: the only test in this binary, so no
    // other thread reads the environment concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "scalar");
    }
    i8_packed_common::check();
}
