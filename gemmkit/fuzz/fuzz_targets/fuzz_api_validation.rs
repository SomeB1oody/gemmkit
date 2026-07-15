//! Fuzz target for the checked validation APIs: drives adversarial geometry into the
//! CHECKED entry points only, never the `*_unchecked` ones, whose contract makes
//! invalid input UB rather than a bug. A documented validation panic carries the
//! "gemmkit:" prefix and is an ACCEPTED outcome; anything else (an index OOB, an
//! arithmetic-overflow panic, an ASan report) is a validation gap
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

static HOOK: Once = Once::new();

// libfuzzer-sys installs a panic hook that aborts before catch_unwind can run, so
// the hook below is replaced once with a silent one; unexpected panics are
// re-raised as aborts further down, so nothing is lost
fuzz_target!(|plan: gemmkit_fuzz::ValidationPlan| {
    HOOK.call_once(|| panic::set_hook(Box::new(|_| {})));

    let r = panic::catch_unwind(AssertUnwindSafe(|| gemmkit_fuzz::drive_validation(&plan)));
    if let Err(payload) = r {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        if msg.contains("gemmkit") {
            return; // documented rejection: accepted outcome
        }
        eprintln!("UNEXPECTED PANIC: {msg}\nplan: {plan:?}");
        std::process::abort();
    }
    // Ok(()) is also fine: geometry passed validation and the GEMM ran (or was
    // skipped by the work cap); ASan watches for validation wrongly accepting
});
