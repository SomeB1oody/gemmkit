//! Resolution contract for the `GEMMKIT_*` env vars behind the tuning knobs
//!
//! A separate test binary from `tests/tuning.rs` (which exercises the `set_*` setters), so the
//! process-global `std::env::set_var`/`remove_var` calls here never race those tests. Holds
//! exactly 1 test, so libtest's concurrent-by-default execution never puts 2 env writers in this
//! process at once. Only meaningful under `std`: a no-`std` build never reads the environment
#![cfg(feature = "std")]

use gemmkit::tuning;

/// Exercises every branch of the env-resolution contract in 1 test (so no 2nd test can race
/// these `set_var` calls): an unset var falls through to the compile-time default; a malformed
/// value falls back to the default (and, by reading the source, takes the warning-and-fallback
/// branch rather than panicking, though its exact stderr text is not checked here, since
/// capturing this process's own output needs a child process); a well-formed value is parsed and
/// used; and a programmatic `set_*` call wins over whatever the env var says
#[test]
fn env_knobs_resolution_contract() {
    // Clear any ambient value (a dev's shell profile could export this) so the assertion below
    // checks the library's fall-through, not whatever the shell happened to set
    unsafe {
        std::env::remove_var("GEMMKIT_SMALL_MN_DIM");
    }
    assert_eq!(
        tuning::small_mn_dim(),
        tuning::SMALL_MN_DIM_DEFAULT,
        "an unset GEMMKIT_* var must fall through to the default"
    );

    // SAFETY: the only test in this binary, so nothing else reads the environment concurrently
    unsafe {
        std::env::set_var("GEMMKIT_K_STREAM_MAX", "not-a-number"); // malformed
        std::env::set_var("GEMMKIT_MC_REG_PANELS", "5"); // well-formed
        std::env::set_var("GEMMKIT_KC_MIN", "7"); // overridden by the setter below
    }

    // Malformed: falls back to the compile-time default without panicking
    assert_eq!(
        tuning::k_stream_max(),
        tuning::K_STREAM_MAX_DEFAULT,
        "a malformed GEMMKIT_* value must fall back to the default"
    );

    // Well-formed: parsed and used as-is
    assert_eq!(
        tuning::mc_reg_panels(),
        5,
        "a well-formed GEMMKIT_* value must be parsed and used"
    );

    // A programmatic set_* overrides whatever the env var holds
    tuning::set_kc_min(999);
    assert_eq!(
        tuning::kc_min(),
        999,
        "set_* must override the GEMMKIT_* env var"
    );
}
