//! Unified tuning surface (cross-cutting).
//!
//! Every heuristic threshold lives here, not scattered across globals. Each one
//! resolves with the priority **per-call argument > programmatic setter > env var
//! (`GEMMKIT_*`) > compile-time default** (calibrated on the Ryzen 9950X). The
//! per-call layer is expressed elsewhere (e.g. the [`crate::Parallelism`]
//! argument); this module owns the setter / env / default layers.

use core::sync::atomic::{AtomicUsize, Ordering};

const UNSET: usize = usize::MAX;

struct Threshold {
    value: AtomicUsize,
    env: &'static str,
    default: usize,
}

impl Threshold {
    const fn new(env: &'static str, default: usize) -> Self {
        Self {
            value: AtomicUsize::new(UNSET),
            env,
            default,
        }
    }

    #[inline]
    fn get(&self) -> usize {
        let v = self.value.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
        let resolved = self.resolve_env().unwrap_or(self.default);
        self.value.store(resolved, Ordering::Relaxed);
        resolved
    }

    #[inline]
    fn set(&self, v: usize) {
        // `usize::MAX` is reserved as the "unset" sentinel; clamp so a caller
        // asking for the maximum still takes effect (as `usize::MAX - 1`).
        self.value.store(v.min(UNSET - 1), Ordering::Relaxed);
    }

    #[cfg(feature = "std")]
    fn resolve_env(&self) -> Option<usize> {
        let v: usize = std::env::var(self.env).ok()?.trim().parse().ok()?;
        Some(v.min(UNSET - 1))
    }
    #[cfg(not(feature = "std"))]
    fn resolve_env(&self) -> Option<usize> {
        None
    }
}

// Below the product `m*n*k`, work is forced onto a single thread. Default
// 48*48*256, matching the empirical serial→parallel break-even.
static PARALLEL_THRESHOLD: Threshold = Threshold::new("GEMMKIT_PARALLEL_THRESHOLD", 48 * 48 * 256);

// Pack the RHS macro-panel only when `m` (the number of rows, i.e. how many row
// blocks reuse the packed B) exceeds this. Below it the RHS is read in place — it
// is only ever broadcast, so any layout works unpacked, and skipping the copy is
// a clear win for small/medium problems. The shared pack buffer has no
// per-worker redundancy, so the gate is purely about copy-cost vs reuse.
static RHS_PACK_THRESHOLD: Threshold = Threshold::new("GEMMKIT_RHS_PACK_THRESHOLD", 2048);

// Pack the LHS macro-panel only when each worker reuses it across more than this
// many columns (per-worker reuse, which falls with the thread count). A non-unit
// row stride or a partial panel always forces packing regardless. The default is
// calibrated so column-major inputs stay unpacked through mid-size parallel runs
// (where redundant per-worker packing would dominate) and pack only when reuse
// is genuinely high.
static LHS_PACK_THRESHOLD: Threshold = Threshold::new("GEMMKIT_LHS_PACK_THRESHOLD", 1024);

// Maximum `min(m, n)` for which the dedicated gemv (matrix·vector) path is taken
// when the other dimension is 1. (Shape, not size, decides; this only caps it.)
static GEMV_THRESHOLD: Threshold = Threshold::new("GEMMKIT_GEMV_THRESHOLD", usize::MAX - 1);

/// Get the serial/parallel work gate (`m*n*k` threshold).
pub fn parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.get()
}
/// Override the serial/parallel work gate.
pub fn set_parallel_threshold(v: usize) {
    PARALLEL_THRESHOLD.set(v);
}

/// Get the RHS-packing gate (on `m`).
pub fn rhs_pack_threshold() -> usize {
    RHS_PACK_THRESHOLD.get()
}
/// Override the RHS-packing gate.
pub fn set_rhs_pack_threshold(v: usize) {
    RHS_PACK_THRESHOLD.set(v);
}

/// Get the LHS-packing gate (per-worker column reuse).
pub fn lhs_pack_threshold() -> usize {
    LHS_PACK_THRESHOLD.get()
}
/// Override the LHS-packing gate.
pub fn set_lhs_pack_threshold(v: usize) {
    LHS_PACK_THRESHOLD.set(v);
}

/// Get the gemv special-path cap on `min(m, n)`.
pub fn gemv_threshold() -> usize {
    GEMV_THRESHOLD.get()
}
/// Override the gemv special-path cap.
pub fn set_gemv_threshold(v: usize) {
    GEMV_THRESHOLD.set(v);
}
