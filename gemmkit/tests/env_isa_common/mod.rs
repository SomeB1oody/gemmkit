//! One-shot `GEMMKIT_REQUIRE_ISA` setter shared by the `env_isa_*` test binaries
//!
//! libtest runs a binary's `#[test]` functions on multiple threads, and `std::env::set_var` is
//! unsound if it races a read on another thread. Every test in one `env_isa_*` binary calls
//! [`pin`] with the same ISA string before it dispatches any gemm, so gating the single
//! `set_var` behind a `Once` guarantees it happens-before every dispatch in the process and no
//! `select_*` ladder ever reads the var mid-write. Because every caller in the binary passes the
//! same value, it does not matter which test's call actually performs the write
//!
//! Kept in its own subdirectory (`env_isa_common/mod.rs`) so cargo treats it as a shared module
//! rather than compiling it as an independent test binary
#![allow(dead_code)]

use std::sync::Once;

/// Set `GEMMKIT_REQUIRE_ISA` to `isa` the first time any test in this binary calls it
///
/// Blocks on the `Once` so every caller, not just the 1st, only returns after the write has
/// landed. Callers must all pass the same `isa` within one binary: later calls are no-ops, they do
/// not overwrite with a different value
pub fn pin(isa: &str) {
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        // SAFETY: Once ensures a single write; every test calls `pin` (with the same isa) before
        // its own dispatch, so no thread reads the var while this write is in flight
        unsafe {
            std::env::set_var("GEMMKIT_REQUIRE_ISA", isa);
        }
    });
}
