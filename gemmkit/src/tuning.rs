//! Unified tuning surface (cross-cutting)
//!
//! Every heuristic threshold consulted by the dispatch, driver, and parallel layers is a
//! `Threshold` declared in this module, instead of a constant or global scattered at its call
//! site. A knob resolves in priority order: a per-call argument (where one exists, e.g. the
//! [`crate::Parallelism`] request) beats a programmatic `set_*` call, which beats the
//! `GEMMKIT_*` env var, which beats the compiled default. The per-call layer is defined wherever
//! that argument lives; this module owns the setter, env, and default layers. The reference
//! machines are a Ryzen 9950X (x86) and an M4 Max (aarch64); a knob whose optimal crossover
//! differs by architecture carries a `#[cfg(target_arch = "aarch64")]`-split default, each side
//! calibrated on its own machine
//!
//! ## Setter vs env precedence
//!
//! `GEMMKIT_*` is the deployment layer: source a profile emitted by the `gemmkit-tune`
//! autotuner to retune an already-built binary for its host, with no recompile. Each var is
//! read once, on the knob's first access, then cached for the rest of the process. A `set_*`
//! call stores its value unconditionally, so a later `get` never consults the env at all: an
//! application that tunes itself in code always wins over a deployment-supplied profile. An app
//! that wants the env var to take effect simply never calls the setter
//!
//! ## Malformed values
//!
//! A `GEMMKIT_*` var that is set but does not parse as a non-negative integer is treated as a
//! typo, not a silent no-op: the first access that finds it warns on stderr and falls back to
//! the default. That resolved default is then cached like any other value, so the warning fires
//! at most once per knob per process. A bad env var never panics: a perf-knob typo must not
//! bring the process down

use core::sync::atomic::{AtomicUsize, Ordering};

const UNSET: usize = usize::MAX;

struct Threshold {
    value: AtomicUsize,
    // `resolve_env` (std only) is the sole reader of this field; a no_std build never touches it
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
        // A resolved value of exactly `UNSET` would look like "not cached yet" on the next call,
        // so clamp it down 1. This only bites a knob whose compiled default is `usize::MAX`
        // itself (the 32-bit `SHARED_LHS_MNK_DEFAULT`); losing that top value is harmless since
        // `MAX` and `MAX - 1` mean the same "effectively unbounded" wherever the knob is read
        let resolved = self.resolve_env().unwrap_or(self.default).min(UNSET - 1);
        self.value.store(resolved, Ordering::Relaxed);
        resolved
    }

    #[inline]
    fn set(&self, v: usize) {
        // `UNSET` is reserved to mean "not cached yet"; a caller passing that exact value is
        // clamped down 1 instead of silently reverting the knob to auto-resolve on the next `get`
        self.value.store(v.min(UNSET - 1), Ordering::Relaxed);
    }

    #[cfg(feature = "std")]
    fn resolve_env(&self) -> Option<usize> {
        // An unset var is the common case: return `None` and let the caller fall through to the
        // default without comment. A var that IS set but fails to parse is almost always a typo
        // in a hand-edited or autotuner-generated profile, so surface it on stderr rather than
        // silently keeping the default. `get` caches whichever value results, so this runs (and
        // can warn) at most once per knob per process, modulo a race between concurrent first
        // accesses on different threads
        let raw = std::env::var(self.env).ok()?;
        match raw.trim().parse::<usize>() {
            Ok(v) => Some(v.min(UNSET - 1)),
            Err(_) => {
                eprintln!(
                    "gemmkit: ignoring malformed {}={raw:?} (expected a non-negative integer); \
                     using default {}",
                    self.env, self.default
                );
                None
            }
        }
    }
    #[cfg(not(feature = "std"))]
    fn resolve_env(&self) -> Option<usize> {
        None
    }
}

// Work gate consulted by `Parallelism::resolve`: a problem whose `m*n*k` product falls below
// this runs on a single thread no matter how many workers were requested. The default,
// 48*48*256, approximates the measured serial/parallel break-even
/// Compiled default for [`parallel_threshold`]: overridden by `GEMMKIT_PARALLEL_THRESHOLD` or
/// [`set_parallel_threshold`]
pub const PARALLEL_THRESHOLD_DEFAULT: usize = 48 * 48 * 256;
static PARALLEL_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_PARALLEL_THRESHOLD", PARALLEL_THRESHOLD_DEFAULT);

// Gate on `m` for packing the RHS macro-panel: below it B is read in place across all `m` row
// blocks. B is only ever broadcast into the kernel, so an unpacked read works under any layout,
// and the shared pack buffer carries no per-worker redundancy to amortize away - the only thing
// this trades is one copy of B against `m` reuses of it, so the gate is purely copy cost vs reuse
/// Compiled default for [`rhs_pack_threshold`]: overridden by `GEMMKIT_RHS_PACK_THRESHOLD` or
/// [`set_rhs_pack_threshold`]
pub const RHS_PACK_THRESHOLD_DEFAULT: usize = 2048;
static RHS_PACK_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_RHS_PACK_THRESHOLD", RHS_PACK_THRESHOLD_DEFAULT);

// Gate on per-worker column reuse for packing the LHS macro-panel: a worker that walks more than
// this many reused columns before moving to the next row block packs A first, since the pack
// cost is amortized over more reuse than a low-reuse worker gets. A non-unit row stride or a
// partial last panel forces packing regardless (see `driver::run`). The crossover trades a
// **machine's** pack cost against its reuse benefit, so it is arch-split:
// * x86 (Zen5): redundant per-worker packing dominates through mid-size parallel runs, so
//   column-major input stays unpacked until reuse is genuinely high (1024)
// * aarch64 (M4 Max): packing is cheap there, so it pays off from far less reuse: 256 sits at the
//   top of a flat 32..256 plateau, packing high-reuse shapes (n >= 512 gains about 30%) without
//   over-packing the low-reuse ones, which are unaffected either way
/// Compiled default for [`lhs_pack_threshold`]: overridden by `GEMMKIT_LHS_PACK_THRESHOLD` or
/// [`set_lhs_pack_threshold`]. Public so the `gemmkit-tune` calibration tool can read this
/// arch-split value as its baseline instead of hard-coding a copy that could drift
#[cfg(target_arch = "aarch64")]
pub const LHS_PACK_THRESHOLD_DEFAULT: usize = 256;
/// The non-aarch64 default; see the aarch64 doc above for what this knob controls
#[cfg(not(target_arch = "aarch64"))]
pub const LHS_PACK_THRESHOLD_DEFAULT: usize = 1024;
static LHS_PACK_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_LHS_PACK_THRESHOLD", LHS_PACK_THRESHOLD_DEFAULT);

// Byte gate on the column-major LHS depth stride (`csa * sizeof(Lhs)`): once it reaches this many
// bytes, A is packed even though reuse alone would not call for it. A column-major A is walked
// down K with stride `csa` in the microkernel; once that stride approaches a memory page, every
// depth step lands on a fresh page and the in-place strided read thrashes the TLB, so packing
// into a contiguous panel wins independent of reuse
/// Compiled default for [`lhs_pack_stride`]: overridden by `GEMMKIT_LHS_PACK_STRIDE` or
/// [`set_lhs_pack_stride`]; `0` means auto (derived from the OS page size)
pub const LHS_PACK_STRIDE_DEFAULT: usize = 0;
static LHS_PACK_STRIDE: Threshold =
    Threshold::new("GEMMKIT_LHS_PACK_STRIDE", LHS_PACK_STRIDE_DEFAULT);

// Address-span companion to the stride gate above: a page-scale per-step stride is only
// actually TLB/cache-hostile when the whole depth-slice walk (`csa * sizeof(Lhs) * kc`) also
// spans more address range than stays resident under it. Both gates must hold before the
// driver force-packs a column-major A: measured on the Zen5 9950X (f32, 32 workers), a
// page-scale stride over a <= 2 MiB slice span re-walks warm lines and beats the redundant
// per-worker packing it replaces by 1.5-2.7x, while a >= 4 MiB span is where packing wins
/// Compiled default for [`lhs_pack_span`]: overridden by `GEMMKIT_LHS_PACK_SPAN` or
/// [`set_lhs_pack_span`]; `0` means auto (4 MiB; see `cache::lhs_pack_span_bytes`)
pub const LHS_PACK_SPAN_DEFAULT: usize = 0;
static LHS_PACK_SPAN: Threshold = Threshold::new("GEMMKIT_LHS_PACK_SPAN", LHS_PACK_SPAN_DEFAULT);

// Reuse floor companion to the stride+span gates above: those 2 price the pack's cost (a
// page-scale depth stride over a wide address span), but a pack only pays off in proportion to
// how many `nr`-wide column tiles re-read each packed A panel (`n_nt_max = nc.div_ceil(nr)`). A
// tall, skinny shape (m >> n) has a huge span but few column tiles, so the stride+span gate
// fires and amortizes an expensive pack over too little reuse. This floor holds the force-pack
// back until at least this many column tiles reuse the panel. The crossover trades a
// **machine's** pack cost against its strided-read penalty, so it is arch-split:
// * x86 (Zen5, f32, auto parallelism): in-place beats the pack by 18-71% through n_nt = 86
//   (m = 4096-8192, k = 512-1024, skinny n) while deep-k squares from n_nt = 171 up (2048^3)
//   still want the pack by 8%, with every n_nt in between a tie - hence the floor at 128
// * aarch64 (M4 Max): the trade nearly inverts - packing is cheap and the in-place walk
//   strides 16 KiB pages, so the pack wins from n_nt = 4 up (m = k = 4096: n = 16 +38%,
//   n = 32 +88%, n = 256 +220%) and in-place only holds n_nt = 2 (n = 8: 177 vs 149
//   GFLOP/s) - hence the floor at 4
/// Compiled default for [`lhs_pack_reuse`]: overridden by `GEMMKIT_LHS_PACK_REUSE` or
/// [`set_lhs_pack_reuse`]; `0` drops the reuse floor (force-pack on the stride+span gate alone)
#[cfg(target_arch = "aarch64")]
pub const LHS_PACK_REUSE_DEFAULT: usize = 4;
/// The non-aarch64 default; see the aarch64 doc above for what this knob controls
#[cfg(not(target_arch = "aarch64"))]
pub const LHS_PACK_REUSE_DEFAULT: usize = 128;
static LHS_PACK_REUSE: Threshold = Threshold::new("GEMMKIT_LHS_PACK_REUSE", LHS_PACK_REUSE_DEFAULT);

// Cap on `min(m, n)` for taking the dedicated gemv (matrix*vector) path when the other dimension
// is 1. The shape (m == 1 or n == 1) decides whether gemv is even a candidate; this only caps how
// large the vector side may be before falling back to the general driver
/// Compiled default for [`gemv_threshold`]: overridden by `GEMMKIT_GEMV_THRESHOLD` or
/// [`set_gemv_threshold`]; effectively unbounded, so gemv-shaped problems always take the
/// dedicated path unless the knob is lowered
pub const GEMV_THRESHOLD_DEFAULT: usize = usize::MAX - 1;
static GEMV_THRESHOLD: Threshold = Threshold::new("GEMMKIT_GEMV_THRESHOLD", GEMV_THRESHOLD_DEFAULT);

// Depth cutoff for the generic small-`k` route (not gemv): at or below this `k`, a shape computes
// the whole product as a single depth panel over the microkernel, reading A/B in place with no
// packing. Above it the register-tiling driver wins, since packing A into contiguous panels pays
// for a better microkernel depth-walk once there is enough depth to amortize the pack. The
// crossover trades a **machine's** depth-walk speed against its pack cost, so it is arch-split:
// * x86 (Zen5, AVX-512): in-place stays ahead through k = 16 (about 120-140% of the driver on
//   skinny GEMM) and falls behind by k = 32, so the crossover sits between them
// * aarch64 (M4, NEON): the narrower 16x4 tile packs cheaply and its depth-walk wins sooner:
//   in-place leads only through k = 8 (about 115% of the driver) and the driver is already ahead
//   by k = 16 (about 76%), so the crossover is half of the x86 one
/// Compiled default for [`small_k_threshold`]: overridden by `GEMMKIT_SMALL_K_THRESHOLD` or
/// [`set_small_k_threshold`]. Public so `gemmkit-tune` can read this arch-split value directly
/// as its baseline rather than hand-copying it
#[cfg(target_arch = "aarch64")]
pub const SMALL_K_THRESHOLD_DEFAULT: usize = 8;
/// The non-aarch64 default; see the aarch64 doc above for what this knob controls
#[cfg(not(target_arch = "aarch64"))]
pub const SMALL_K_THRESHOLD_DEFAULT: usize = 16;
static SMALL_K_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_SMALL_K_THRESHOLD", SMALL_K_THRESHOLD_DEFAULT);

// Dimension cap (applied to both m and n) for the small-matrix horizontal (inner-product) route:
// a shape that clears the cap on both sides, has k above the small-k threshold, and streams both
// operands contiguously along k computes each output element as one SIMD-reduced dot over k,
// reading A/B in place with no packing or blocking. The register-tiling driver would instead pad
// tiny row/column tiles up to a full microtile and spend most of its work on padding; this route
// computes exactly the m*n outputs it needs
/// Compiled default for [`small_mn_dim`]: overridden by `GEMMKIT_SMALL_MN_DIM` or
/// [`set_small_mn_dim`]
pub const SMALL_MN_DIM_DEFAULT: usize = 16;
static SMALL_MN_DIM: Threshold = Threshold::new("GEMMKIT_SMALL_MN_DIM", SMALL_MN_DIM_DEFAULT);

// Depth floor (exclusive) for the small-`m,n` horizontal route's PACK tier: a shape that clears
// the `small_mn_dim` caps but has an operand strided along k (an all-row-major or all-col-major
// small-m,n GEMM) copies just the failing operand into k-contiguous scratch and runs the same
// horizontal dot, instead of falling back to the register-tiling driver - but only once k exceeds
// this floor. Below it, a strided shape stays on the driver: the copy no longer amortizes against
// the driver's padding overhead. The zero-copy tier (both operands already unit-stride along k)
// ignores this knob entirely and keeps using `small_k_threshold`. The crossover is the same class
// of machine property as `small_k_threshold` (copy cost and cache geometry vs the driver's padded
// deficit), calibrated on Zen5 (AVX-512): the packed route beats the driver at every measured k
// for every small shape (1.1x at 16x16 k=32, up to about 6.8x at 4x4, never a regression), so the
// gate is set right at the small-k boundary - a strided small-m,n shape starts packing exactly
// where an in-place shape would have left the small-k route anyway
/// Compiled default for [`small_mn_pack_min_k`]: overridden by `GEMMKIT_SMALL_MN_PACK_MIN_K` or
/// [`set_small_mn_pack_min_k`]
pub const SMALL_MN_PACK_MIN_K_DEFAULT: usize = 16;
static SMALL_MN_PACK_MIN_K: Threshold =
    Threshold::new("GEMMKIT_SMALL_MN_PACK_MIN_K", SMALL_MN_PACK_MIN_K_DEFAULT);

// Byte floor below which a bandwidth-bound gemv/gevv stays single-threaded: below it, the touched
// data (the matrix for gemv, the output for gevv) is LLC-resident, one core already gets the
// full LLC bandwidth, and splitting the work only adds fork/join overhead and shared-LLC
// contention with no DRAM bandwidth to gain by spreading across cores
/// Compiled default for [`gemv_parallel_bytes`]: overridden by `GEMMKIT_GEMV_PARALLEL_BYTES` or
/// [`set_gemv_parallel_bytes`]; `0` means auto (derived from the LLC size, see
/// `crate::cache::gemv_parallel_floor_bytes`)
pub const GEMV_PARALLEL_BYTES_DEFAULT: usize = 0;
static GEMV_PARALLEL_BYTES: Threshold =
    Threshold::new("GEMMKIT_GEMV_PARALLEL_BYTES", GEMV_PARALLEL_BYTES_DEFAULT);

// Hard cap on the worker count a bandwidth-bound gemv/gevv may use. This is the escape hatch for
// the fact that the memory-parallel width is a heuristic proxy: no physical-core or
// memory-channel count is exposed at runtime, and DRAM saturates at far fewer workers than the
// logical core count. Raise it on a high-bandwidth shared-cache part (e.g. Apple Silicon)
/// Compiled default for [`gemv_thread_cap`]: overridden by `GEMMKIT_GEMV_THREAD_CAP` or
/// [`set_gemv_thread_cap`]; `0` means auto (derived from the logical core count, see
/// `crate::parallel::bandwidth_cap`)
pub const GEMV_THREAD_CAP_DEFAULT: usize = 0;
static GEMV_THREAD_CAP: Threshold =
    Threshold::new("GEMMKIT_GEMV_THREAD_CAP", GEMV_THREAD_CAP_DEFAULT);

// Dynamic-scheduling granularity for the general parallel path: the driver aims for this many
// work chunks per worker, handed out from a shared atomic cursor on demand, so a faster core
// (e.g. a P-core on a heterogeneous big.LITTLE layout) pulls proportionally more chunks than a
// slower one. A higher value means finer load balance and a smaller tail at the cost of more
// atomic claims (and, on the packed-LHS path, more re-packing at chunk edges); a lower value
// means coarser balance but less overhead
/// Compiled default for [`parallel_oversample`]: overridden by `GEMMKIT_PARALLEL_OVERSAMPLE` or
/// [`set_parallel_oversample`]
pub const PARALLEL_OVERSAMPLE_DEFAULT: usize = 8;
static PARALLEL_OVERSAMPLE: Threshold =
    Threshold::new("GEMMKIT_PARALLEL_OVERSAMPLE", PARALLEL_OVERSAMPLE_DEFAULT);

// Auto worker-count ramp granularity: how much total work (`m*n*k`) each additional worker
// must bring before the auto `Rayon(0)` path widens by one, i.e. it targets `mnk / this`
// workers, floored at 1 and capped by the core and job counts. The ramp is work-based rather
// than dimension-based because the measured optimum tracks total flops, not linear size: on
// the Zen5 9950X, 128^3 (2e6) runs fastest serial, 192^3 (7e6) at 2-3 workers, and 384^3
// (5.7e7) already wants all 32 hardware threads - a linear-dimension stride cannot fit both
// ends of that curve. Calibrated on the 9950X; sweep with gemmkit-tune on other machines
/// Compiled default for [`par_mnk_per_worker`]: overridden by `GEMMKIT_PAR_MNK_PER_WORKER`
/// or [`set_par_mnk_per_worker`]. Public so the `gemmkit-tune` calibration tool can read this
/// target-split value as its baseline instead of hard-coding a copy that could drift
#[cfg(not(all(target_arch = "wasm32", feature = "wasm_threads")))]
pub const PAR_MNK_PER_WORKER_DEFAULT: usize = 2_000_000;
/// The threaded-wasm default: a wasm worker costs far less to engage than a native thread
/// (no `available_parallelism` walk, a dedicated pre-sized pool, cheap wasmtime scheduling),
/// and the measured optimum under wasmtime keeps all 8 default workers profitable down to
/// `128^3` (5x serial there, where the native machine wants that shape serial) - so the
/// per-worker work floor sits 8x lower than the native calibration
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub const PAR_MNK_PER_WORKER_DEFAULT: usize = 262_144;
static PAR_MNK_PER_WORKER: Threshold =
    Threshold::new("GEMMKIT_PAR_MNK_PER_WORKER", PAR_MNK_PER_WORKER_DEFAULT);

// Number of exact-fit private rayon pool tiers, halving down from half the machine width: 1
// tier is width/2, 2 adds width/4, 3 adds width/8. The auto worker count snaps to a tier so a
// small parallel GEMM runs in a pool sized to it with no idle slack, since rayon's fork-join
// tax scales with a pool's SLACK (its width minus the active workers), not just the worker
// count. The tier pools are persistent (built once, reused warm), not rebuilt per call. `0`
// disables the tiers and keeps the current ambient-pool behavior. The default is arch-split:
// * x86 (Zen5): 2 tiers. Measured on the 9950X (32 HW threads, 16 physical cores), the width/4
//   tier of 8 wins 96^3 through 288^3 by 2-3x, and the width/2 physical-core tier of 16 wins
//   320^3 through 448^3, beating full SMT width by 7-11% by staying on the physical cores
// * aarch64 (M4 Max): 1 tier. Measured on the M4 Max (14 cores, 10P + 4E, no SMT), the width/2
//   tier of 7 wins 128^3 through 224^3 by 2.1-2.6x over the ambient-pool ramp; a deeper ladder
//   loses (the tier margin would route 128^3..192^3 onto the width/4 tier of 3 and halve those
//   wins), and an exact 10-worker P-core tier loses to 7 everywhere except 256^3 (+7%), not
//   worth a 2nd tier. Full width takes over from 256^3 up through the aarch64 arm of the
//   full-width gate below
// * other targets: 0 (disabled), pending on-device validation; wasm has its own dedicated pool
//   and never reaches this path
/// Compiled default for [`pool_classes`]: overridden by `GEMMKIT_POOL_CLASSES` or
/// [`set_pool_classes`]; `0` disables the private size-class pools (ambient-pool behavior).
/// Clamped to at most 3 tiers at the consumer (see `crate::parallel::class_sizes`)
#[cfg(target_arch = "x86_64")]
pub const POOL_CLASSES_DEFAULT: usize = 2;
/// The aarch64 default; see the x86_64 doc above for what this knob controls
#[cfg(target_arch = "aarch64")]
pub const POOL_CLASSES_DEFAULT: usize = 1;
/// The default on every other target; see the x86_64 doc above for what this knob controls
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const POOL_CLASSES_DEFAULT: usize = 0;
static POOL_CLASSES: Threshold = Threshold::new("GEMMKIT_POOL_CLASSES", POOL_CLASSES_DEFAULT);

// Work gate (`m*n*k`) at which the auto path leaves the largest private pool tier for the full
// machine width. Below it a tier pool (the physical cores on x86, no SMT) wins; only the largest
// problems pay for the full machine width. Calibrated on the Zen5 9950X between 448^3 (89.9M, where
// the physical-core tier still wins by 7%) and 512^3 (134M, where full width wins by 5%). The
// auto value is arch-split at the consumer: aarch64 (M4 Max, no SMT) crosses over an order of
// magnitude earlier, between 224^3 and 256^3. Only consulted when pool tiers are active
// (`pool_classes` above is non-zero)
// the physical-core tier still wins by 7%) and 512^3 (134M, where full width wins by 5%). The
// auto value is arch-split at the consumer: aarch64 (M4 Max, no SMT) crosses over an order of
// magnitude earlier, between 224^3 and 256^3. Only consulted when pool tiers are active
// (`pool_classes` above is non-zero)
/// Compiled default for [`full_width_mnk`]: overridden by `GEMMKIT_FULL_WIDTH_MNK` or
/// [`set_full_width_mnk`]; `0` means auto (arch-split, 110_000_000 on x86 and 14_000_000 on
/// aarch64; see `crate::parallel::FULL_WIDTH_MNK_AUTO`)
pub const FULL_WIDTH_MNK_DEFAULT: usize = 0;
static FULL_WIDTH_MNK: Threshold = Threshold::new("GEMMKIT_FULL_WIDTH_MNK", FULL_WIDTH_MNK_DEFAULT);

// Worker count for a threaded wasm build (the `wasm_threads` feature). wasm has no
// `available_parallelism`, so the deployer sets the parallel width here instead: it both caps
// `auto_threads` and sizes gemmkit's own wasm rayon pool. An off-target build stays serial via
// the `RAYON_USABLE` guard regardless of this knob
/// Compiled default for [`wasm_threads`]: overridden by `GEMMKIT_WASM_THREADS` or
/// [`set_wasm_threads`]
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub const WASM_THREADS_DEFAULT: usize = 8;
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
static WASM_THREADS: Threshold = Threshold::new("GEMMKIT_WASM_THREADS", WASM_THREADS_DEFAULT);

// Workload gate (`m*n*k`) for the shared-LHS A-pack, on top of the driver's own
// `n_mc < n_threads` redundancy check. The shared pre-pass removes redundant per-worker packing
// of the same A panel, but adds a fork-join barrier per depth slice; that barrier only pays for
// itself once the problem is large enough, so small/mid sizes are left on the per-worker path
// and only problems above this crossover use the shared pre-pass
//
// The crossover is a **machine** property (barrier cost vs redundant-pack savings), not a tile
// property, so it does not scale with mr/nr/kc
/// Compiled default for [`shared_lhs_mnk`]: overridden by `GEMMKIT_SHARED_LHS_MNK` or
/// [`set_shared_lhs_mnk`]. Public for the same reason as [`SMALL_K_THRESHOLD_DEFAULT`]: a
/// calibration tool can read the shipped, arch-split value directly instead of mirroring it
#[cfg(target_arch = "aarch64")]
pub const SHARED_LHS_MNK_DEFAULT: usize = 50_000_000;
/// The 64-bit non-aarch64 default; see the aarch64 doc above for what this knob controls
#[cfg(all(not(target_arch = "aarch64"), target_pointer_width = "64"))]
pub const SHARED_LHS_MNK_DEFAULT: usize = 8_000_000_000;
// A 32-bit `usize` cannot hold the 8e9 literal above, so the 32-bit default is `usize::MAX`
// instead, which disables the shared pre-pass entirely (no `m*n*k` can ever reach it)
/// The 32-bit non-aarch64 default; see the aarch64 doc above for what this knob controls
#[cfg(all(not(target_arch = "aarch64"), not(target_pointer_width = "64")))]
pub const SHARED_LHS_MNK_DEFAULT: usize = usize::MAX;
static SHARED_LHS_MNK: Threshold = Threshold::new("GEMMKIT_SHARED_LHS_MNK", SHARED_LHS_MNK_DEFAULT);

// Depth ceiling for register-blocking the output of an axpy-shape gemv (holding the output panel
// in registers across the whole k-sweep). Above it, the many in-place matrix column-streams (one
// per depth step) exceed the hardware prefetcher's window and the plain column-outer form wins
// instead. Calibrated on Zen5: register-blocking wins by k <= 16, is a wash near k = 32, and
// regresses by k ~ 48
/// Compiled default for [`k_stream_max`]: overridden by `GEMMKIT_K_STREAM_MAX` or
/// [`set_k_stream_max`]. Public for the same reason as [`SMALL_K_THRESHOLD_DEFAULT`]: a
/// calibration tool reads it directly instead of hard-coding a copy of `32`
pub const K_STREAM_MAX_DEFAULT: usize = 32;
static K_STREAM_MAX: Threshold = Threshold::new("GEMMKIT_K_STREAM_MAX", K_STREAM_MAX_DEFAULT);

// aarch64-only batched-GEMM crossover: when there are fewer batch elements than workers, a batch
// element is split across the whole machine (`SequentialInternal`) rather than run one-per-worker
// cache-hot, once its per-batch-worker byte share (`elem_bytes / batch`, where `elem_bytes` sums
// A + B + C) exceeds this. Only the aarch64 `resolve_batch` path reads it; the knob and its env
// var still exist on other targets, but nothing there consults them. Calibrated on M4 Max (shared
// cluster-L2 plus high unified bandwidth): about 128 KiB separates elements that scale with
// DRAM/L2 bandwidth when split across the cluster from small cache-hot ones better run
// one-per-worker
/// Compiled default for [`seq_internal_bytes_per_worker`]: overridden by
/// `GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER` or [`set_seq_internal_bytes_per_worker`]. Public so a
/// calibration tool can read the shipped default as its baseline. Only consulted on aarch64
/// (inert elsewhere), but the default itself is not `#[cfg]`-split: there is nothing
/// arch-specific to calibrate for a value no other target reads
pub const SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT: usize = 128 * 1024;
static SEQ_INTERNAL_BYTES_PER_WORKER: Threshold = Threshold::new(
    "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
    SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT,
);

// Dynamic-scheduling split target for the packed-LHS path: `packed_block_grain` splits each row
// block into power-of-two column sub-chunks until there are at least `this * n_threads` chunks,
// trading pack reuse for load balance. Distinct from `PARALLEL_OVERSAMPLE` (the general path's
// grain, default 8): this is the packed path's own swept optimum, since splitting harder there
// also re-packs A more often and regresses throughput instead of improving balance
/// Compiled default for [`packed_oversample`]: overridden by `GEMMKIT_PACKED_OVERSAMPLE` or
/// [`set_packed_oversample`]
pub const PACKED_OVERSAMPLE_DEFAULT: usize = 2;
static PACKED_OVERSAMPLE: Threshold =
    Threshold::new("GEMMKIT_PACKED_OVERSAMPLE", PACKED_OVERSAMPLE_DEFAULT);

// MC cap, in microtile rows: the A macro-panel is bounded to `this * MR` rows, matching the BLIS
// guidance of keeping MC a small multiple of MR. This is a calibration point, not an invariant
// derived from cache size - it currently binds on every measured topology ahead of the
// L2-capacity-derived MC, so a larger L2 does not by itself move MC today
/// Compiled default for [`mc_reg_panels`]: overridden by `GEMMKIT_MC_REG_PANELS` or
/// [`set_mc_reg_panels`]
pub const MC_REG_PANELS_DEFAULT: usize = 8;
static MC_REG_PANELS: Threshold = Threshold::new("GEMMKIT_MC_REG_PANELS", MC_REG_PANELS_DEFAULT);

// NC cap, in microtile columns, used only when the machine reports no L3 (e.g. Apple Silicon):
// the no-L3 column block becomes `min(this * NR, N)`, i.e. full-N up to this ceiling. With no L3,
// the BLIS model wants NC large since B streams straight from DRAM and it is the A panel that must
// fit the last-level cache; this cap just bounds the shared packed-B panel's size. Dead on any
// topology that reports an L3. `512` (2048 columns at NR = 4) keeps typical widths at full-N
/// Compiled default for [`nc_no_l3_panels`]: overridden by `GEMMKIT_NC_NO_L3_PANELS` or
/// [`set_nc_no_l3_panels`]
pub const NC_NO_L3_PANELS_DEFAULT: usize = 512;
static NC_NO_L3_PANELS: Threshold =
    Threshold::new("GEMMKIT_NC_NO_L3_PANELS", NC_NO_L3_PANELS_DEFAULT);

// Small-matrix shortcut gate: a shape with both m and n at or below this skips the full BLIS
// blocking model entirely and just keeps the A/B panels resident in L2. One knob bounds both
// dimensions. The prepack paths derive their own branch-dodging sentinel as `this + 1`, so raising
// this gate can never make a prepack call take the tiny-matrix branch by surprise
/// Compiled default for [`tiny_block_dim`]: overridden by `GEMMKIT_TINY_BLOCK_DIM` or
/// [`set_tiny_block_dim`]
pub const TINY_BLOCK_DIM_DEFAULT: usize = 64;
static TINY_BLOCK_DIM: Threshold = Threshold::new("GEMMKIT_TINY_BLOCK_DIM", TINY_BLOCK_DIM_DEFAULT);

// Depth-block ceiling used only inside the small-matrix shortcut above: there, kc is k clamped to
// this value
/// Compiled default for [`kc`]: overridden by `GEMMKIT_KC` or [`set_kc`]
pub const KC_DEFAULT: usize = 512;
static KC: Threshold = Threshold::new("GEMMKIT_KC", KC_DEFAULT);

// Depth-block floor used by the main BLIS model (not the small-matrix shortcut): the L1-fit
// estimate for kc is raised to at least this before the final rebalance pass, so a small L1 cache
// can never starve the microkernel's depth-walk down to an impractically small kc
/// Compiled default for [`kc_min`]: overridden by `GEMMKIT_KC_MIN` or [`set_kc_min`]
pub const KC_MIN_DEFAULT: usize = 512;
static KC_MIN: Threshold = Threshold::new("GEMMKIT_KC_MIN", KC_MIN_DEFAULT);

// Deep-contraction engage gate, in bytes. A narrow-output family (f16/bf16, where OUT_IS_ACC is
// false) runs the whole contraction as one depth panel (kc = k) so the narrow output rounds only
// once; at large k, its single RHS micropanel (nr * k * sizeof(N) bytes) outgrows L2, and every
// microtile call then streams that panel from L3/DRAM instead of cache. Once the micropanel size
// exceeds this gate, dispatch switches to the family's f32-output twin (OUT_IS_ACC = true), which
// re-blocks the contraction at the cache-model kc (multi-slice, panels kept L2-resident) into an
// f32 scratch buffer and narrows the result once at the end. The twin is bit-identical to the
// single panel for the common beta in {0, 1} case (it continues the same ascending-k accumulation
// across slices); accurate only to tolerance otherwise
/// Compiled default for [`deep_kc_bytes`]: overridden by `GEMMKIT_DEEP_KC_BYTES` or
/// [`set_deep_kc_bytes`]; `0` means auto (derived from half the detected L2, see
/// `crate::cache::deep_k_engage_bytes`, where the measured throughput cliff sits)
pub const DEEP_KC_BYTES_DEFAULT: usize = 0;
static DEEP_KC_BYTES: Threshold = Threshold::new("GEMMKIT_DEEP_KC_BYTES", DEEP_KC_BYTES_DEFAULT);

// Working-set gate for the driver's C-tile prefetch: once a call's working set (the A, B and C
// bytes together) exceeds the last-level cache, the output tiles stream from DRAM and the driver
// issues a T0 prefetch of each C microtile just ahead of its microkernel call, hiding part of the
// tile's read-modify-write latency; below the gate the tiles are cache-resident and the hint
// would be pure overhead. The prefetch only reorders cache traffic, never arithmetic, so results
// are bit-identical with the gate on, off, or forced. Measured on the Zen5 9950X (f32, AVX-512):
// +1.4% parallel and about +1% serial at 2048^3 (48 MiB working set), +2-3% parallel at
// 3072^3 and deep-k
// 2048x2048x24576, neutral at 1536^3 and below (under the 32 MiB per-CCD LLC) and at in-cache
// sizes. Stream-level prefetching inside the kc loop (LHS/RHS, plain or chunked) measured
// strictly worse on the same hardware and is deliberately absent
/// Compiled default for [`prefetch_min_bytes`]: overridden by `GEMMKIT_PREFETCH_MIN_BYTES` or
/// [`set_prefetch_min_bytes`]; `0` means auto (the per-core-reachable last-level cache, see
/// `crate::cache::prefetch_ws_bytes`)
pub const PREFETCH_MIN_BYTES_DEFAULT: usize = 0;
static PREFETCH_MIN_BYTES: Threshold =
    Threshold::new("GEMMKIT_PREFETCH_MIN_BYTES", PREFETCH_MIN_BYTES_DEFAULT);

// Strip length for the cache-blocked transpose used by every strided packing path: the real
// packer, the complex packer, and the small-m,n PACK tier's k-contiguous copy. The source is
// walked along its contiguous dimension in strips of this many depth steps, and each strip is
// scattered into the destination panel, turning what would be a per-element strided gather into
// a sequence of blocked copies. One knob backs all of them, since they share the identical strip
// geometry
/// Compiled default for [`pack_transpose_tile`]: overridden by `GEMMKIT_PACK_TRANSPOSE_TILE` or
/// [`set_pack_transpose_tile`]
pub const PACK_TRANSPOSE_TILE_DEFAULT: usize = 16;
static PACK_TRANSPOSE_TILE: Threshold =
    Threshold::new("GEMMKIT_PACK_TRANSPOSE_TILE", PACK_TRANSPOSE_TILE_DEFAULT);

// Work gate (m*n*k) below which an auto-selected VNNI i8 kernel, on a multi-threaded request,
// hands the problem to the widen fallback instead: VNNI's mandatory RHS-pack barrier outweighs the
// compute it saves on a small parallel problem. The fallback is bit-identical to VNNI (both
// compute in exact i32), so swapping between them never perturbs the result. Calibrated on Zen5:
// VNNI wins the serial case at every size and the parallel case from about n ~ 1024 up. Inert
// unless the x86 VNNI auto-select path is actually chosen for the element type
/// Compiled default for [`i8_vnni_min_par_mnk`]: overridden by `GEMMKIT_I8_VNNI_MIN_PAR_MNK` or
/// [`set_i8_vnni_min_par_mnk`]
#[cfg(feature = "int8")]
pub const I8_VNNI_MIN_PAR_MNK_DEFAULT: usize = 768 * 768 * 768;
#[cfg(feature = "int8")]
static I8_VNNI_MIN_PAR_MNK: Threshold =
    Threshold::new("GEMMKIT_I8_VNNI_MIN_PAR_MNK", I8_VNNI_MIN_PAR_MNK_DEFAULT);

// Canonical list of every knob's `GEMMKIT_*` env name, kept as the single source of truth that
// out-of-crate consumers assert against: the `gemmkit-tune` sweep table, `tests/props_knobs.rs`'s
// `KNOBS` list, and the fuzz crate's `KNOB_SETTERS` list. Each of those hand-maintained lists checks
// itself against this one, so a knob added above cannot silently escape their coverage. A static
// cannot be iterated over without a macro (which this crate avoids), so this manual mirror is the
// chosen tradeoff: every `Threshold` static declared above must have its env name appear here
// The 28 knobs that exist in every build live in `KNOB_ENV_NAMES_BASE`; the 2 that are cfg-gated
// (whose `Threshold` statics above carry the same cfg) are appended only when actually compiled
// in - `I8_VNNI_MIN_PAR_MNK` under the `int8` feature and `WASM_THREADS` under
// `wasm32 + wasm_threads`. Declaring a `Threshold` here without adding its name to one of these 2
// lists is a small diff away from being caught: the consumer sync tests assert against the count
const KNOB_ENV_NAMES_BASE: [&str; 28] = [
    "GEMMKIT_PARALLEL_THRESHOLD",
    "GEMMKIT_RHS_PACK_THRESHOLD",
    "GEMMKIT_LHS_PACK_THRESHOLD",
    "GEMMKIT_LHS_PACK_STRIDE",
    "GEMMKIT_LHS_PACK_SPAN",
    "GEMMKIT_LHS_PACK_REUSE",
    "GEMMKIT_GEMV_THRESHOLD",
    "GEMMKIT_SMALL_K_THRESHOLD",
    "GEMMKIT_SMALL_MN_DIM",
    "GEMMKIT_SMALL_MN_PACK_MIN_K",
    "GEMMKIT_GEMV_PARALLEL_BYTES",
    "GEMMKIT_GEMV_THREAD_CAP",
    "GEMMKIT_PARALLEL_OVERSAMPLE",
    "GEMMKIT_PAR_MNK_PER_WORKER",
    "GEMMKIT_POOL_CLASSES",
    "GEMMKIT_FULL_WIDTH_MNK",
    "GEMMKIT_SHARED_LHS_MNK",
    "GEMMKIT_K_STREAM_MAX",
    "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
    "GEMMKIT_PACKED_OVERSAMPLE",
    "GEMMKIT_MC_REG_PANELS",
    "GEMMKIT_NC_NO_L3_PANELS",
    "GEMMKIT_TINY_BLOCK_DIM",
    "GEMMKIT_KC",
    "GEMMKIT_KC_MIN",
    "GEMMKIT_DEEP_KC_BYTES",
    "GEMMKIT_PREFETCH_MIN_BYTES",
    "GEMMKIT_PACK_TRANSPOSE_TILE",
];

// Count of cfg-gated knob names appended to the base list on this target. `cfg!` evaluates to a
// compile-time bool, so this entire expression folds to a constant with no runtime cost
const KNOB_ENV_NAMES_GATED: usize = cfg!(feature = "int8") as usize
    + cfg!(all(target_arch = "wasm32", feature = "wasm_threads")) as usize;

const KNOB_ENV_NAMES_LEN: usize = KNOB_ENV_NAMES_BASE.len() + KNOB_ENV_NAMES_GATED;

// Assemble the base names plus whichever cfg-gated names are compiled in, entirely at compile
// time, so the public getter below is just a plain slice return with no init-time work. The
// trailing `assert!` both consumes the running index (keeping it live) and turns a length
// miscount into a compile error instead of a silently truncated list
const fn build_knob_env_names() -> [&'static str; KNOB_ENV_NAMES_LEN] {
    let mut out = [""; KNOB_ENV_NAMES_LEN];
    let mut i = 0;
    while i < KNOB_ENV_NAMES_BASE.len() {
        out[i] = KNOB_ENV_NAMES_BASE[i];
        i += 1;
    }
    if cfg!(feature = "int8") {
        out[i] = "GEMMKIT_I8_VNNI_MIN_PAR_MNK";
        i += 1;
    }
    if cfg!(all(target_arch = "wasm32", feature = "wasm_threads")) {
        out[i] = "GEMMKIT_WASM_THREADS";
        i += 1;
    }
    assert!(i == KNOB_ENV_NAMES_LEN);
    out
}

static KNOB_ENV_NAMES: [&str; KNOB_ENV_NAMES_LEN] = build_knob_env_names();

/// Every knob's `GEMMKIT_*` env name, accurate for the current target and enabled features
///
/// A `#[doc(hidden)]` machine-readable registry, not a stable API: the knob consumers
/// (`gemmkit-tune`, `tests/props_knobs.rs`, the fuzz crate's `KNOB_SETTERS`) assert their
/// hand-maintained lists against this one, so a newly added knob cannot quietly escape their
/// coverage. Zero runtime cost (backed by a compile-time `static`) and `no_std`-compatible
#[doc(hidden)]
pub fn knob_env_names() -> &'static [&'static str] {
    &KNOB_ENV_NAMES
}

/// Get the serial/parallel work gate: an `m*n*k` product below this runs single-threaded
pub fn parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.get()
}
/// Override the serial/parallel work gate
pub fn set_parallel_threshold(v: usize) {
    PARALLEL_THRESHOLD.set(v);
}

/// Get the RHS-packing gate, applied to `m`: B is packed once it is reused across more than this
/// many row blocks
pub fn rhs_pack_threshold() -> usize {
    RHS_PACK_THRESHOLD.get()
}
/// Override the RHS-packing gate
pub fn set_rhs_pack_threshold(v: usize) {
    RHS_PACK_THRESHOLD.set(v);
}

/// Get the LHS-packing gate on per-worker column reuse: A is packed once a worker's share of
/// column strips exceeds this
pub fn lhs_pack_threshold() -> usize {
    LHS_PACK_THRESHOLD.get()
}
/// Override the LHS-packing gate
pub fn set_lhs_pack_threshold(v: usize) {
    LHS_PACK_THRESHOLD.set(v);
}

/// Get the LHS-packing depth-stride gate, in bytes: a column-major A whose `csa * sizeof(Lhs)`
/// reaches this many bytes is packed to avoid a TLB/cache-hostile strided read, independent of
/// the reuse gate above. `0` (the default) means auto: the gate is derived from the OS page size
pub fn lhs_pack_stride() -> usize {
    LHS_PACK_STRIDE.get()
}
/// Override the LHS-packing depth-stride gate, in bytes; `0` restores the page-size-derived auto
/// value
pub fn set_lhs_pack_stride(v: usize) {
    LHS_PACK_STRIDE.set(v);
}

/// Get the LHS-packing address-span gate, in bytes: the stride gate only force-packs a
/// column-major A when the whole depth-slice walk (`csa * sizeof(Lhs) * kc`) also reaches this
/// many bytes of address range. `0` (the default) means auto (4 MiB)
pub fn lhs_pack_span() -> usize {
    LHS_PACK_SPAN.get()
}
/// Override the LHS-packing address-span gate, in bytes; `0` restores the auto value
pub fn set_lhs_pack_span(v: usize) {
    LHS_PACK_SPAN.set(v);
}

/// Get the LHS-packing reuse floor for the strided-LHS force-pack: the stride+span gate only
/// fires when at least this many `nr`-wide column tiles reuse each packed A panel; `0` drops
/// the floor (force-pack on the stride+span gate alone)
pub fn lhs_pack_reuse() -> usize {
    LHS_PACK_REUSE.get()
}
/// Override the LHS-packing reuse floor; `0` drops the floor (force-pack on stride+span alone)
pub fn set_lhs_pack_reuse(v: usize) {
    LHS_PACK_REUSE.set(v);
}

/// Get the gemv special-path cap on `min(m, n)`
pub fn gemv_threshold() -> usize {
    GEMV_THRESHOLD.get()
}
/// Override the gemv special-path cap
pub fn set_gemv_threshold(v: usize) {
    GEMV_THRESHOLD.set(v);
}

/// Get the small-`k` route threshold: a `k` at or below this takes the generic small-`k` in-place
/// path instead of the register-tiling driver
pub fn small_k_threshold() -> usize {
    SMALL_K_THRESHOLD.get()
}
/// Override the small-`k` route threshold
pub fn set_small_k_threshold(v: usize) {
    SMALL_K_THRESHOLD.set(v);
}

/// Get the small-matrix horizontal route's dimension cap: a shape with both `m` and `n` at or
/// below this (and `k` above the small-`k` threshold) takes the horizontal inner-product path.
/// `0` disables the route entirely
pub fn small_mn_dim() -> usize {
    SMALL_MN_DIM.get()
}
/// Override the small-matrix horizontal route's dimension cap (`0` disables the route)
pub fn set_small_mn_dim(v: usize) {
    SMALL_MN_DIM.set(v);
}

/// Get the small-`m,n` horizontal route's PACK-tier `k` gate: a small-`m,n` shape with an operand
/// strided along `k` is copied into `k`-contiguous scratch and run through the horizontal dot
/// only once `k` exceeds this (otherwise it stays on the register-tiling driver). The zero-copy
/// tier, where both operands are already unit-stride along `k`, ignores this knob and gates on
/// `small_k_threshold` instead
pub fn small_mn_pack_min_k() -> usize {
    SMALL_MN_PACK_MIN_K.get()
}
/// Override the small-`m,n` horizontal route's PACK-tier `k` gate
pub fn set_small_mn_pack_min_k(v: usize) {
    SMALL_MN_PACK_MIN_K.set(v);
}

/// Get the gemv/gevv parallelism byte floor. `0` means auto: derive it from the LLC size (see
/// `crate::cache::gemv_parallel_floor_bytes`); a non-zero value is the floor verbatim
pub fn gemv_parallel_bytes() -> usize {
    GEMV_PARALLEL_BYTES.get()
}
/// Override the gemv/gevv parallelism byte floor (`0` restores the LLC-derived auto value)
pub fn set_gemv_parallel_bytes(v: usize) {
    GEMV_PARALLEL_BYTES.set(v);
}

/// Get the gemv/gevv worker cap. `0` means auto: derive a bandwidth proxy from the core count
/// (see `crate::parallel::bandwidth_cap`); a non-zero value is a hard cap
pub fn gemv_thread_cap() -> usize {
    GEMV_THREAD_CAP.get()
}
/// Override the gemv/gevv worker cap (`0` restores the core-derived auto proxy)
pub fn set_gemv_thread_cap(v: usize) {
    GEMV_THREAD_CAP.set(v);
}

/// Get the parallel dynamic-scheduling oversample factor (target chunks per worker). Always
/// `>= 1` so the scheduler can never be handed a zero-sized grain
pub fn parallel_oversample() -> usize {
    PARALLEL_OVERSAMPLE.get().max(1)
}
/// Override the parallel dynamic-scheduling oversample factor
pub fn set_parallel_oversample(v: usize) {
    PARALLEL_OVERSAMPLE.set(v);
}

/// Get the shared-LHS A-pack workload gate (`m*n*k`): the shared pre-pack engages at or above it;
/// below it, each worker still packs its own copy of A
pub fn shared_lhs_mnk() -> usize {
    SHARED_LHS_MNK.get()
}
/// Override the shared-LHS A-pack workload gate (`m*n*k`)
pub fn set_shared_lhs_mnk(v: usize) {
    SHARED_LHS_MNK.set(v);
}

/// Get the axpy-gemv output register-blocking depth ceiling: register-blocking is used when
/// `k <= this`
pub fn k_stream_max() -> usize {
    K_STREAM_MAX.get()
}
/// Override the axpy-gemv output register-blocking depth ceiling
pub fn set_k_stream_max(v: usize) {
    K_STREAM_MAX.set(v);
}

/// Get the aarch64 batched-GEMM `SequentialInternal` byte crossover (per-batch-worker share)
pub fn seq_internal_bytes_per_worker() -> usize {
    SEQ_INTERNAL_BYTES_PER_WORKER.get()
}
/// Override the aarch64 batched-GEMM `SequentialInternal` byte crossover
pub fn set_seq_internal_bytes_per_worker(v: usize) {
    SEQ_INTERNAL_BYTES_PER_WORKER.set(v);
}

/// Get the packed-LHS dynamic-scheduling split target (target chunks per worker). Always `>= 1`
/// so the grain computation can never target zero chunks
pub fn packed_oversample() -> usize {
    PACKED_OVERSAMPLE.get().max(1)
}
/// Override the packed-LHS dynamic-scheduling split target
pub fn set_packed_oversample(v: usize) {
    PACKED_OVERSAMPLE.set(v);
}

/// Get the MC cap, in microtile rows (the A macro-panel is bounded to `this * MR` rows). Always
/// `>= 1` so `MC` stays a positive multiple of `MR`
pub fn mc_reg_panels() -> usize {
    MC_REG_PANELS.get().max(1)
}
/// Override the MC cap, in microtile rows
pub fn set_mc_reg_panels(v: usize) {
    MC_REG_PANELS.set(v);
}

/// Get the no-L3 NC cap, in microtile columns (`NC <= this * NR`; consulted only when the machine
/// reports no L3)
pub fn nc_no_l3_panels() -> usize {
    NC_NO_L3_PANELS.get()
}
/// Override the no-L3 NC cap, in microtile columns
pub fn set_nc_no_l3_panels(v: usize) {
    NC_NO_L3_PANELS.set(v);
}

/// Get the small-matrix shortcut gate: a shape with both `m` and `n` at or below this skips the
/// full blocking model. The prepack paths dodge this branch via a `this + 1` sentinel
pub fn tiny_block_dim() -> usize {
    TINY_BLOCK_DIM.get()
}
/// Override the small-matrix shortcut gate
pub fn set_tiny_block_dim(v: usize) {
    TINY_BLOCK_DIM.set(v);
}

/// Get the tiny-branch `kc` ceiling (the small-matrix shortcut's depth block is `k` clamped to
/// this). Always `>= 1` so the clamp's upper bound never falls below its lower bound
pub fn kc() -> usize {
    KC.get().max(1)
}
/// Override the tiny-branch `kc` ceiling
pub fn set_kc(v: usize) {
    KC.set(v);
}

/// Get the main-model `kc` floor: the L1-fit depth estimate is raised to at least this before the
/// last-panel rebalance
pub fn kc_min() -> usize {
    KC_MIN.get()
}
/// Override the main-model `kc` floor
pub fn set_kc_min(v: usize) {
    KC_MIN.set(v);
}

/// Get the deep-contraction engage gate, in bytes: a narrow-output family switches to its
/// f32-output multi-slice twin once its single RHS micropanel (`nr * k * sizeof(N)`) exceeds
/// this. `0` (the default) means auto: derive the gate from the detected L2 (see
/// `crate::cache::deep_k_engage_bytes`); a non-zero value is the byte threshold verbatim
pub fn deep_kc_bytes() -> usize {
    DEEP_KC_BYTES.get()
}
/// Override the deep-contraction engage gate, in bytes (`0` restores the L2-derived auto value)
pub fn set_deep_kc_bytes(v: usize) {
    DEEP_KC_BYTES.set(v);
}

/// Get the C-tile prefetch engage gate, in bytes: the driver prefetches each output microtile
/// just ahead of its microkernel call only when the call's working set (the A, B and C bytes
/// together) exceeds this. `0` (the default) means auto: derive the gate from the detected
/// last-level cache (see `crate::cache::prefetch_ws_bytes`); a non-zero value is the byte
/// threshold verbatim, so `usize::MAX` disables the prefetch and `1` forces it on
pub fn prefetch_min_bytes() -> usize {
    PREFETCH_MIN_BYTES.get()
}
/// Override the C-tile prefetch engage gate, in bytes (`0` restores the LLC-derived auto value)
pub fn set_prefetch_min_bytes(v: usize) {
    PREFETCH_MIN_BYTES.set(v);
}

/// Get the cache-blocked-transpose strip length used by the strided packing paths. Always `>= 1`
/// so the strip loop always advances
pub fn pack_transpose_tile() -> usize {
    PACK_TRANSPOSE_TILE.get().max(1)
}
/// Override the cache-blocked-transpose strip length
pub fn set_pack_transpose_tile(v: usize) {
    PACK_TRANSPOSE_TILE.set(v);
}

/// Get the i8 VNNI small-parallel fallback gate (`m*n*k`): below it, an auto-selected VNNI kernel
/// hands a multi-threaded problem to the widen fallback instead. The 2 kernels are bit-identical,
/// so which one runs never changes the result
#[cfg(feature = "int8")]
pub fn i8_vnni_min_par_mnk() -> usize {
    I8_VNNI_MIN_PAR_MNK.get()
}
/// Override the i8 VNNI small-parallel fallback gate
#[cfg(feature = "int8")]
pub fn set_i8_vnni_min_par_mnk(v: usize) {
    I8_VNNI_MIN_PAR_MNK.set(v);
}

/// Get the auto worker-count ramp granularity: the `m*n*k` work each additional worker must
/// bring before the auto `Rayon(0)` path widens by one (it targets `mnk / this` workers).
/// Returned verbatim; the consumer clamps to `>= 1` so a `0` cannot divide by zero
pub fn par_mnk_per_worker() -> usize {
    PAR_MNK_PER_WORKER.get()
}
/// Override the auto worker-count ramp granularity (`0` behaves as `1`: always full width)
pub fn set_par_mnk_per_worker(v: usize) {
    PAR_MNK_PER_WORKER.set(v);
}

/// Get the number of exact-fit private rayon pool tiers, halving from half the machine width
/// (`0` disables the size-class pools, restoring the ambient-pool behavior). Clamped to at
/// most 3 tiers at the consumer (see `crate::parallel::class_sizes`)
pub fn pool_classes() -> usize {
    POOL_CLASSES.get()
}
/// Override the number of exact-fit private rayon pool tiers (`0` disables them)
pub fn set_pool_classes(v: usize) {
    POOL_CLASSES.set(v);
}

/// Get the full-machine-width work gate (`m*n*k`): above it the auto path leaves the largest
/// private pool tier for the full machine width. `0` means auto (arch-split, 110_000_000 on
/// x86 and 14_000_000 on aarch64). Consulted only when the size-class pools are active
/// (`pool_classes` is non-zero)
pub fn full_width_mnk() -> usize {
    FULL_WIDTH_MNK.get()
}
/// Override the full-machine-width work gate (`0` restores the arch-split auto value)
pub fn set_full_width_mnk(v: usize) {
    FULL_WIDTH_MNK.set(v);
}

/// Get the worker count for a threaded wasm build (default 8; see `WASM_THREADS_DEFAULT`). Only
/// exists on `wasm32` with the `wasm_threads` feature, where the runtime cannot report a core
/// count; every other target uses `available_parallelism` instead
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn wasm_threads() -> usize {
    WASM_THREADS.get().max(1)
}
/// Override the threaded-wasm worker count (clamped to `>= 1`)
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn set_wasm_threads(v: usize) {
    WASM_THREADS.set(v.max(1));
}
