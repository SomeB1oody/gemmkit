//! Single source of the `GEMMKIT_FAST_TEST` knob shared across the test suites
//!
//! `GEMMKIT_FAST_TEST` is a TEST-SUITE-ONLY switch: the library never reads it (grep of
//! `gemmkit/src` for the name returns nothing). When on, the deterministic dimension /
//! coefficient sweeps in the test bodies shrink to one representative of each redundant
//! cross-product combo while still hitting every branch/path class the full sweep does;
//! when off, the full byte-for-byte sweep runs unchanged. The result is cached in a
//! `OnceLock`, so the variable is read once per process.
//!
//! This is the only implementation site. It is reached via the existing `#[path = ...]`
//! include pattern: `tests/oracle_common/mod.rs` includes and re-exports it (so the
//! correctness and property suites pick it up through their existing `pub use
//! oracle_common::*`), and `tests/epilogue/common.rs` includes it directly. It is never a
//! test target of its own (cargo only builds top-level `tests/*.rs` and `tests/*/main.rs`),
//! and each including binary uses it in a different subset of tests, so `dead_code` is
//! allowed.
#![allow(dead_code)]

use std::sync::OnceLock;

/// True iff `GEMMKIT_FAST_TEST` selects the shrunk sweeps: the value is exactly `1`, or
/// `true` (ASCII case-insensitive). Unset, empty, `0`, or anything else is off. Read once
/// and cached for the life of the process
pub fn fast_test() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| match std::env::var("GEMMKIT_FAST_TEST") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    })
}
