//! Shared `GEMMKIT_REQUIRE_ISA` pin for the per-ISA env-var test binaries (`env_isa_*`)
//!
//! Every test in a given `env_isa_*` binary pins the **same** ISA value; [`pin`] performs that
//! single `set_var` through a `std::sync::Once`, so the write lands before any dispatch in the
//! binary. libtest runs a binary's tests concurrently, but the `Once` makes the one write
//! happen-before every `pin` return, and each test dispatches only after its own `pin` call, so
//! no `select_*` ladder reads the environment concurrently with the write. All tests wanting the
//! same value is what makes the shared write sound: the binary is per-pin-value, so the memoized
//! dispatch resolves that one ISA regardless of which test resolves it first. This also overrides
//! an inherited `GEMMKIT_REQUIRE_ISA` (the SDE/pinned CI jobs export one), the same as the old
//! single-test binaries did
//!
//! Lives in a subdirectory so cargo does not compile it as its own test binary
#![allow(dead_code)]

use std::sync::Once;

/// Pin `GEMMKIT_REQUIRE_ISA` to `isa` exactly once for this process, before any dispatch. Every
/// caller blocks until the single write completes, so the pin is visible to every later `select_*`
/// resolve in the binary. All tests in a binary pass the same `isa`, so a repeat call is a no-op
pub fn pin(isa: &str) {
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        // SAFETY: the Once serializes this single write, and every test calls `pin` before it
        // dispatches, so nothing reads the environment concurrently with the write. Every test in
        // this binary pins the same value, so which test wins the Once does not matter
        unsafe {
            std::env::set_var("GEMMKIT_REQUIRE_ISA", isa);
        }
    });
}
