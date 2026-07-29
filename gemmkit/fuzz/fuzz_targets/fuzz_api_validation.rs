//! Fuzz target for the checked GEMM/prepack API. It drives adversarial geometry, such
//! as huge or negative strides and dims near `usize::MAX`, into `drive_validation`.
//! That driver calls only the checked entry points: `gemm`, `gemm_i8`, `gemm_cplx`,
//! `gemm_batched`, `prepack_rhs`, and `prepack_lhs`. It never calls their
//! `*_unchecked` twins, whose contract makes bad input undefined behavior rather than
//! a bug. A panic that carries the "gemmkit:" prefix is a documented validation reject
//! and counts as an accepted outcome. Anything else, an out-of-bounds index, an
//! arithmetic-overflow panic, or an ASan report, marks a validation gap
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

static HOOK: Once = Once::new();

// libfuzzer-sys's own panic hook runs the default hook, then aborts. That would kill
// the process before catch_unwind below ever sees the panic. This swaps in a silent
// hook once, so a documented "gemmkit:" panic can be caught and dismissed instead
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
    // No panic means the geometry passed validation and the GEMM ran, or was skipped by
    // the work cap. ASan is what would catch validation that wrongly lets bad input
    // through
});
