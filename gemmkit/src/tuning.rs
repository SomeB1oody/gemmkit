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
    // Read only by the `std` `resolve_env` below; the no-`std` build never looks
    // at an env var, so the name is stored but unread there.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
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

// Avoid a TLB/cache-hostile strided read, not amortize a copy.
// A column-major A is walked down K in the microkernel with stride `csa`,
// so when `csa * sizeof(Lhs)` reaches ~a memory page every depth
// step lands on a fresh page and the in-place read collapses
static LHS_PACK_STRIDE: Threshold = Threshold::new("GEMMKIT_LHS_PACK_STRIDE", 0);

// Maximum `min(m, n)` for which the dedicated gemv (matrix·vector) path is taken
// when the other dimension is 1. (Shape, not size, decides; this only caps it.)
static GEMV_THRESHOLD: Threshold = Threshold::new("GEMMKIT_GEMV_THRESHOLD", usize::MAX - 1);

// At or below this `k`, a (non-gemv) shape takes the generic small-`k` route — computing
// the whole product in one depth panel over the microkernel, reading A/B in place, no
// packing. Above it the register-tiling driver wins: packing A into contiguous panels pays
// for the better microkernel depth-walk once `k` is large enough. Calibrated on Zen5 —
// in-place stays ahead through `k = 16` (~120-140% of the driver on skinny GEMM) and falls
// behind by `k = 32`, so the crossover sits between.
static SMALL_K_THRESHOLD: Threshold = Threshold::new("GEMMKIT_SMALL_K_THRESHOLD", 16);

// Byte floor below which a bandwidth-bound gemv/gevv stays single-threaded, and the
// per-worker ramp quantum above it: the auto worker count is `bytes_touched / this`,
// capped by the topology bandwidth proxy. Below roughly one core's L2 the touched data is
// cache-resident and a single core already saturates its own bandwidth, so fork/join would
// only add overhead. Calibrated on Zen5 (1 MiB L2).
static GEMV_PARALLEL_BYTES: Threshold = Threshold::new("GEMMKIT_GEMV_PARALLEL_BYTES", 1024 * 1024);

// Maximum workers a bandwidth-bound gemv/gevv may use. `0` (the default) derives a proxy
// from the logical core count (see `parallel::bandwidth_cap`); any non-zero value is a hard
// cap. This is the escape hatch for the fact that the memory-parallel width is a heuristic:
// no physical-core / memory-channel count is exposed, and DRAM saturates at far fewer
// workers than the logical core count. Raise it on a high-bandwidth shared-L2 part (Apple).
static GEMV_THREAD_CAP: Threshold = Threshold::new("GEMMKIT_GEMV_THREAD_CAP", 0);

// Dynamic-scheduling granularity: the parallel driver aims for this many work
// chunks *per worker*, handed out from a shared cursor on demand, so faster cores
// (heterogeneous big.LITTLE P/E layouts) pull proportionally more. Higher = finer
// load balance and a smaller tail, at the cost of more atomic claims (and, on the
// rare packed-LHS path, more re-packing at chunk edges); lower = coarser
static PARALLEL_OVERSAMPLE: Threshold = Threshold::new("GEMMKIT_PARALLEL_OVERSAMPLE", 8);

// Auto worker-count ramp granularity (units of linear problem dimension per
// worker): the auto `Rayon(0)` path targets `cbrt(m*n*k).div_ceil(this)` workers.
// `0` (the default) means *auto* — derive the stride from the core count (see
// [`thread_dim_stride`]); any non-zero env/setter value overrides verbatim.
static THREAD_DIM_STRIDE: Threshold = Threshold::new("GEMMKIT_THREAD_DIM_STRIDE", 0);

// Worker count for a threaded wasm build (`wasm_threads` feature). wasm has no
// `available_parallelism`, so the deployer sets the parallel width here instead — it caps
// `auto_threads` and sizes gemmkit's wasm rayon pool. Off-target builds stay serial via the
// `RAYON_USABLE` guard regardless.
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
static WASM_THREADS: Threshold = Threshold::new("GEMMKIT_WASM_THREADS", 8);

// Minimum `m*n*k` for the shared-LHS A-pack to engage (on top of the runtime
// `n_mc < n_threads` redundancy guard in the driver). The shared pre-pass removes
// redundant per-worker packs but adds a fork-join barrier per depth slice; it pays
// only once the problem is large enough to amortize that barrier, so small/mid
// sizes regress and it is gated above the crossover.
//
// The crossover is a **machine** property, not a tile property
#[cfg(target_arch = "aarch64")]
const SHARED_LHS_MNK_DEFAULT: usize = 50_000_000;
#[cfg(all(not(target_arch = "aarch64"), target_pointer_width = "64"))]
const SHARED_LHS_MNK_DEFAULT: usize = 8_000_000_000;
// On a 32-bit target `usize` 8e9 literal above would not compile
// So on 32-bit set the default to `usize::MAX` to disable the pre-pass
#[cfg(all(not(target_arch = "aarch64"), not(target_pointer_width = "64")))]
const SHARED_LHS_MNK_DEFAULT: usize = usize::MAX;
static SHARED_LHS_MNK: Threshold = Threshold::new("GEMMKIT_SHARED_LHS_MNK", SHARED_LHS_MNK_DEFAULT);

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

/// Get the LHS-packing depth-stride gate, in bytes: a column-major A whose
/// `csa * sizeof(Lhs)` reaches this is packed to avoid a TLB/cache-hostile strided
/// read, independent of the reuse gate above. `0` (the default) means *auto* — the
/// driver derives the gate from the OS page size.
pub fn lhs_pack_stride() -> usize {
    LHS_PACK_STRIDE.get()
}
/// Override the LHS-packing depth-stride gate (bytes); `0` restores auto.
pub fn set_lhs_pack_stride(v: usize) {
    LHS_PACK_STRIDE.set(v);
}

/// Get the gemv special-path cap on `min(m, n)`.
pub fn gemv_threshold() -> usize {
    GEMV_THRESHOLD.get()
}
/// Override the gemv special-path cap.
pub fn set_gemv_threshold(v: usize) {
    GEMV_THRESHOLD.set(v);
}

/// Get the small-`k` route threshold (`k` at/below this takes the generic small-`k` path).
pub fn small_k_threshold() -> usize {
    SMALL_K_THRESHOLD.get()
}
/// Override the small-`k` route threshold.
pub fn set_small_k_threshold(v: usize) {
    SMALL_K_THRESHOLD.set(v);
}

/// Get the gemv/gevv parallelism byte floor and per-worker ramp quantum. Always `>= 1`
/// so it can never be a zero divisor in the worker-count ramp.
pub fn gemv_parallel_bytes() -> usize {
    GEMV_PARALLEL_BYTES.get().max(1)
}
/// Override the gemv/gevv parallelism byte floor / ramp quantum.
pub fn set_gemv_parallel_bytes(v: usize) {
    GEMV_PARALLEL_BYTES.set(v);
}

/// Get the gemv/gevv worker cap. `0` means *auto* — derive a bandwidth proxy from the
/// core count (see `crate::parallel::bandwidth_cap`); any non-zero value is a hard cap.
pub fn gemv_thread_cap() -> usize {
    GEMV_THREAD_CAP.get()
}
/// Override the gemv/gevv worker cap (`0` restores the core-derived auto proxy).
pub fn set_gemv_thread_cap(v: usize) {
    GEMV_THREAD_CAP.set(v);
}

/// Get the parallel dynamic-scheduling oversample factor (chunks per worker).
/// Always `>= 1` so the scheduler can never receive a zero grain.
pub fn parallel_oversample() -> usize {
    PARALLEL_OVERSAMPLE.get().max(1)
}
/// Override the parallel dynamic-scheduling oversample factor.
pub fn set_parallel_oversample(v: usize) {
    PARALLEL_OVERSAMPLE.set(v);
}

/// Get the shared-LHS A-pack workload gate (`m*n*k` threshold): the shared
/// pre-pack engages at or above it; below, each worker packs its own A.
pub fn shared_lhs_mnk() -> usize {
    SHARED_LHS_MNK.get()
}
/// Override the shared-LHS A-pack workload gate (`m*n*k`).
pub fn set_shared_lhs_mnk(v: usize) {
    SHARED_LHS_MNK.set(v);
}

/// Get the auto worker-count ramp granularity (units of linear problem dimension
/// per worker). `0` (the default) derives the stride from the machine's core count
/// (see `auto_thread_dim_stride`); any non-zero env/setter value is used verbatim.
/// Always `>= 1` so the `cbrt(mnk).div_ceil(stride)` ramp cannot divide by zero.
pub fn thread_dim_stride() -> usize {
    match THREAD_DIM_STRIDE.get() {
        0 => auto_thread_dim_stride(),
        v => v.max(1),
    }
}
/// Override the auto worker-count ramp granularity (`0` restores the core-derived
/// auto value).
pub fn set_thread_dim_stride(v: usize) {
    THREAD_DIM_STRIDE.set(v);
}

/// Get the worker count for a threaded wasm build (default 8). See [`WASM_THREADS`].
/// Only exists on `wasm32` with the `wasm_threads` feature, where the runtime cannot
/// report a core count; elsewhere `available_parallelism` is used instead.
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn wasm_threads() -> usize {
    WASM_THREADS.get().max(1)
}
/// Override the threaded-wasm worker count (clamped to `>= 1`).
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn set_wasm_threads(v: usize) {
    WASM_THREADS.set(v.max(1));
}

/// Core-count-derived auto ramp granularity. The ramp saturates all `cores` workers at
/// the linear size `cbrt(mnk) == stride * cores`, so the stride sets how fast a problem
/// ramps to full width. This is an *empirical calibration, not a derivation*: it is fit
/// to two measured points — a low/mid-core part that benefits from a fast ramp (small
/// stride) and a higher-core part that wants a slow one (large stride) — as
/// `stride = clamp(cores²/16, 16, 64)`. The real driver is memory-domain topology
/// (cross-domain traffic favors a slower ramp), which we can't robustly detect, so core
/// count is only a proxy and the interpolation between the two anchors is unvalidated.
/// The `16` floor keeps small machines from ramping *more* aggressively than measured (a
/// bare `cores²/16` gives `1` at 4 cores); the `64` ceiling keeps large ones no more
/// aggressive than the legacy default. `available_parallelism` is cheap and `resolve`
/// already calls it per region, so this is recomputed, not memoized. Override
/// `GEMMKIT_THREAD_DIM_STRIDE` on any topology this two-point fit misses.
#[cfg(feature = "std")]
fn auto_thread_dim_stride() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    (cores * cores / 16).clamp(16, 64)
}
/// Without `std` there is no `available_parallelism`; keep the legacy constant.
#[cfg(not(feature = "std"))]
fn auto_thread_dim_stride() -> usize {
    64
}
