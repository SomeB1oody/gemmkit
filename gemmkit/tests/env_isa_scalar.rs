//! `GEMMKIT_REQUIRE_ISA=scalar` prepacked-i8 bit-parity pin. Scalar is valid on every architecture,
//! so no host feature gate is needed: it forces the widen scalar kernel's plain-panel prepack
//! layout deterministically, regardless of what the auto path would pick. The single test pins
//! through [`env_isa_common::pin`] (single `set_var` under a `Once` before any dispatch; the shared
//! write overrides an inherited pin). Only meaningful with `std` (no-`std` never reads the
//! environment)
#![cfg(all(feature = "std", feature = "int8", not(miri)))]

// Shared single-set_var pin helper (Once before any dispatch; this test pins `scalar`)
mod env_isa_common;
// Shared prepacked-i8 bit-parity check driven by the ISA pin binaries
mod i8_packed_common;

#[test]
fn scalar_pin_i8_packed_matches_plain() {
    env_isa_common::pin("scalar");
    i8_packed_common::check();
}
