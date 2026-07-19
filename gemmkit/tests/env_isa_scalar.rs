//! `GEMMKIT_REQUIRE_ISA=scalar` prepacked-i8 bit-parity pin
//!
//! `scalar` is valid on every architecture, so unlike the other `env_isa_*` binaries this needs
//! no host feature gate: it deterministically forces the plain-panel widen kernel's prepack
//! layout regardless of what auto-selection would otherwise pick on this host. Pins through
//! [`env_isa_common::pin`] (a single `set_var` under a `Once`, before any dispatch). Only
//! meaningful with `std`: a no-`std` build never reads the environment
#![cfg(all(feature = "std", feature = "int8", not(miri)))]

// Shared GEMMKIT_REQUIRE_ISA pin helper; this binary's only test pins `scalar` with it
mod env_isa_common;
// Shared prepacked-vs-plain i8 bit-parity check, run under whichever ISA is pinned
mod i8_packed_common;

#[test]
fn scalar_pin_i8_packed_matches_plain() {
    env_isa_common::pin("scalar");
    i8_packed_common::check();
}
