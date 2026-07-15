//! `GEMMKIT_*` env-var resolution for the tuning knobs. Its own test binary (a separate process)
//! so the process-global `std::env::set_var`/`remove_var` here cannot race the setter-based tests
//! in `tests/tuning.rs`; it also holds exactly **one** test, so all env access is single-threaded
//! within this binary under every `cargo test` invocation (including `--include-ignored`). Only
//! meaningful with `std` (no-`std` never reads the environment)
#![cfg(feature = "std")]

use gemmkit::tuning;

/// The env-resolution contract:
/// * an **unset** var falls through to the compile-time default, silently;
/// * a **well-formed** value is parsed and used;
/// * a **malformed** value falls back to the default without panicking (and, by inspection, hits
///   `resolve_env`'s `eprintln!` warning branch, its exact stderr text is not asserted here:
///   capturing this process's own stderr needs a child process, and a 2nd in-harness test/entry
///   point to drive it races these `set_var`s, so the sound choice is to verify the *behavior*);
/// * a programmatic `set_*` overrides the env var (env is the deployment layer)
#[test]
fn env_knobs_resolution_contract() {
    // Unset -> default, silently. Remove any ambient value first (a dev who `source`d a tuned
    // profile may have GEMMKIT_SMALL_MN_DIM exported), so this checks the fall-through, not the
    // shell. Sound: this is the only test in this binary, so nothing reads env concurrently
    unsafe {
        std::env::remove_var("GEMMKIT_SMALL_MN_DIM");
    }
    assert_eq!(
        tuning::small_mn_dim(),
        16,
        "an unset GEMMKIT_* var must fall through to the default"
    );

    // SAFETY: single test, single thread; no gemm runs here, so nothing reads the environment
    // concurrently with these writes. Each knob is read once below, cached thereafter
    unsafe {
        std::env::set_var("GEMMKIT_K_STREAM_MAX", "not-a-number"); // malformed -> default (+warn)
        std::env::set_var("GEMMKIT_MC_REG_PANELS", "5"); // well-formed -> parsed
        std::env::set_var("GEMMKIT_KC_MIN", "7"); // set, but the setter below wins
    }

    // Malformed value: exercises resolve_env's warn+fallback branch; the value is the compile-time
    // default (32) and the process does not panic
    assert_eq!(
        tuning::k_stream_max(),
        32,
        "a malformed GEMMKIT_* value must fall back to the default"
    );

    // Well-formed value is applied verbatim
    assert_eq!(
        tuning::mc_reg_panels(),
        5,
        "a well-formed GEMMKIT_* value must be parsed and used"
    );

    // Setter beats env: env is the deployment layer, an in-code `set_*` takes precedence
    tuning::set_kc_min(999);
    assert_eq!(
        tuning::kc_min(),
        999,
        "set_* must override the GEMMKIT_* env var"
    );
}
