//! Single source of the `GEMMKIT_FAST_TEST` env-var switch shared by the test suites
//!
//! `GEMMKIT_FAST_TEST` is read only by test code: the library itself never checks it. When
//! set, the deterministic dimension/coefficient sweeps in the test bodies shrink to a smaller
//! set that still reaches every branch or path class the full sweep does, instead of the full
//! cross product. The result is cached in a `OnceLock`, so the variable is read at most once
//! per process
//!
//! This is the only implementation site, reached through the existing `#[path = ...]` include
//! pattern rather than a normal `mod`: `tests/oracle_common/mod.rs` includes and re-exports it,
//! so the correctness and property suites pick it up through their own `pub use
//! oracle_common::*`, and `tests/epilogue/common.rs` includes it directly. It is never a test
//! target of its own (cargo's default harness only builds top-level `tests/*.rs` and
//! `tests/*/main.rs`, and this file is neither), and each including binary only exercises a
//! subset of it, so `dead_code` is allowed
#![allow(dead_code)]

use std::sync::OnceLock;

/// True when `GEMMKIT_FAST_TEST` is `1` or `true` (ASCII case-insensitive); unset, empty,
/// `0`, or anything else leaves the full sweep active. Read from the environment once and
/// cached in a `OnceLock` for the life of the process
pub fn fast_test() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| match std::env::var("GEMMKIT_FAST_TEST") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    })
}
