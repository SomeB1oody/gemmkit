//! Fuzz target for the checked GEMM/prepack API: drives adversarial geometry (huge or
//! negative strides, dims near `usize::MAX`) into `drive_validation`, which calls only
//! the checked entry points (`gemm`, `gemm_i8`, `gemm_cplx`, `gemm_batched`,
//! `prepack_rhs`, `prepack_lhs`), never their `*_unchecked` twins whose contract makes
//! bad input UB rather than a bug. A panic carrying the "gemmkit:" prefix is a
//! documented validation reject and is an accepted outcome; anything else (an
//! out-of-bounds index, an arithmetic-overflow panic, an ASan report) is a validation
//! gap
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

static HOOK: Once = Once::new();

// libfuzzer-sys's own panic hook runs the default hook then aborts, which would kill
// the process before catch_unwind below ever sees the panic; swap in a silent hook
// once so a documented "gemmkit:" panic can be caught and dismissed instead
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
            return; // documented reject, not a bug
        }
        eprintln!("UNEXPECTED PANIC: {msg}\nplan: {plan:?}");
        std::process::abort();
    }
    // no panic: the geometry passed validation and the GEMM ran (or was skipped by
    // the work cap); ASan is what would catch validation wrongly letting bad input through
});
