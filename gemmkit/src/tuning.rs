//! Unified tuning surface (cross-cutting)
//!
//! Every heuristic threshold lives here, not scattered across globals. Each one
//! resolves with the priority **per-call argument > programmatic setter > env var
//! (`GEMMKIT_*`) > compile-time default** (calibrated on the Ryzen 9950X). The
//! per-call layer is expressed elsewhere (e.g. the [`crate::Parallelism`]
//! argument); this module owns the setter / env / default layers
//!
//! ## Setter vs env precedence
//!
//! The **env var is the deployment layer**: set `GEMMKIT_*` (e.g. `source` a profile emitted by
//! the `gemmkit-tune` autotuner) to retune an already-built binary for the host with no recompile.
//! It is read once per knob, on the first access, then cached. A programmatic **`set_*` call wins
//! over the env var** (a `set_*` stores the value unconditionally, so a later `get` never consults
//! the env); this is deliberate: an application that tunes itself in code takes precedence over a
//! deployment-supplied profile. Apps that want the env to apply simply don't call the setters
//!
//! ## Malformed values
//!
//! A `GEMMKIT_*` var that is set but does not parse as a non-negative integer is a typo, not a
//! silent no-op: `resolve_env` warns on stderr and falls back to the default (the value is then
//! cached, so the warning fires once per knob after the first access). It never panics: a
//! perf-knob typo must not crash the process

use core::sync::atomic::{AtomicUsize, Ordering};

const UNSET: usize = usize::MAX;

struct Threshold {
    value: AtomicUsize,
    // Read only by the `std` `resolve_env` below; the no-`std` build never looks
    // at an env var, so the name is stored but unread there
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
        // Clamp to `UNSET - 1` so the cached value can never itself be the `UNSET` sentinel: a
        // knob whose default is `usize::MAX` (e.g. the 32-bit `SHARED_LHS_MNK`) would otherwise
        // never cache and re-resolve (re-warning) on every access. `MAX` and `MAX - 1` are
        // interchangeable for every threshold that uses `MAX` to mean "effectively unbounded"
        let resolved = self.resolve_env().unwrap_or(self.default).min(UNSET - 1);
        self.value.store(resolved, Ordering::Relaxed);
        resolved
    }

    #[inline]
    fn set(&self, v: usize) {
        // `usize::MAX` is reserved as the "unset" sentinel; clamp so a caller
        // asking for the maximum still takes effect (as `usize::MAX - 1`)
        self.value.store(v.min(UNSET - 1), Ordering::Relaxed);
    }

    #[cfg(feature = "std")]
    fn resolve_env(&self) -> Option<usize> {
        // A missing var is the normal case: fall through to the default silently. A var that
        // *is* set but does not parse is almost always a typo in an autotuner-generated profile,
        // so make it visible: warn and fall back to the default rather than panic. A perf-knob
        // typo must not crash the process. `get` caches the resolved value, so the warning fires
        // once per knob after the first access (a burst of concurrent first-accesses may repeat it)
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

// Below the product `m*n*k`, work is forced onto a single thread. Default
// 48*48*256, matching the empirical serial->parallel break-even
/// Compiled default for [`parallel_threshold`] (before any env/setter override)
pub const PARALLEL_THRESHOLD_DEFAULT: usize = 48 * 48 * 256;
static PARALLEL_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_PARALLEL_THRESHOLD", PARALLEL_THRESHOLD_DEFAULT);

// Pack the RHS macro-panel only when `m` (the number of rows, i.e. how many row
// blocks reuse the packed B) exceeds this. Below it the RHS is read in place: it
// is only ever broadcast, so any layout works unpacked, and skipping the copy is
// a clear win for small/medium problems. The shared pack buffer has no
// per-worker redundancy, so the gate is purely about copy-cost vs reuse
/// Compiled default for [`rhs_pack_threshold`] (before any env/setter override)
pub const RHS_PACK_THRESHOLD_DEFAULT: usize = 2048;
static RHS_PACK_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_RHS_PACK_THRESHOLD", RHS_PACK_THRESHOLD_DEFAULT);

// Pack the LHS macro-panel only when each worker reuses it across more than this many columns
// (per-worker reuse, which falls with the thread count). A non-unit row stride or a partial panel
// always forces packing. The crossover is a **machine** property (per-worker pack cost vs reuse),
// so it is arch-specific:
// * **x86** (Zen5): redundant per-worker packing dominates through mid-size parallel runs, so keep
//   column-major inputs unpacked until reuse is genuinely high (1024)
// * **aarch64** (M4 Max): packing is cheap, so it pays from much lower reuse: 256 (the top of a
//   flat 32..256 plateau) packs high-reuse shapes (n >= 512 gains ~30%) without over-packing the
//   low-reuse ones (which are unaffected either way)
/// Compiled default for [`lhs_pack_threshold`], ignoring any env override. Public so a calibration
/// tool (gemmkit-tune) can use the shipped default as its baseline without mirroring this arch-split
/// value
#[cfg(target_arch = "aarch64")]
pub const LHS_PACK_THRESHOLD_DEFAULT: usize = 256;
/// See the aarch64 variant
#[cfg(not(target_arch = "aarch64"))]
pub const LHS_PACK_THRESHOLD_DEFAULT: usize = 1024;
static LHS_PACK_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_LHS_PACK_THRESHOLD", LHS_PACK_THRESHOLD_DEFAULT);

// Avoid a TLB/cache-hostile strided read, not amortize a copy. A column-major A is walked
// down K in the microkernel with stride `csa`; once `csa * sizeof(Lhs)` approaches a memory
// page, every depth step lands on a fresh page and the in-place read collapses
/// Compiled default for [`lhs_pack_stride`] (before any env/setter override); `0` = auto (page-derived)
pub const LHS_PACK_STRIDE_DEFAULT: usize = 0;
static LHS_PACK_STRIDE: Threshold =
    Threshold::new("GEMMKIT_LHS_PACK_STRIDE", LHS_PACK_STRIDE_DEFAULT);

// Maximum `min(m, n)` for which the dedicated gemv (matrix*vector) path is taken
// when the other dimension is 1. (Shape, not size, decides; this only caps it)
/// Compiled default for [`gemv_threshold`] (before any env/setter override); effectively unbounded
pub const GEMV_THRESHOLD_DEFAULT: usize = usize::MAX - 1;
static GEMV_THRESHOLD: Threshold = Threshold::new("GEMMKIT_GEMV_THRESHOLD", GEMV_THRESHOLD_DEFAULT);

// At or below this `k`, a (non-gemv) shape takes the generic small-`k` route: computing
// the whole product in one depth panel over the microkernel, reading A/B in place, no
// packing. Above it the register-tiling driver wins: packing A into contiguous panels pays
// for the better microkernel depth-walk once `k` is large enough. The crossover is a
// **machine** property (microkernel depth-walk vs pack cost), so it is arch-specific:
// * **x86** (Zen5, AVX-512): in-place stays ahead through `k = 16` (~120-140% of the driver
//   on skinny GEMM) and falls behind by `k = 32`, so the crossover sits between
// * **aarch64** (M4, NEON): the narrower 16x4 tile packs cheaply and its depth-walk wins
//   sooner: in-place leads through `k = 8` (~115% of the driver) and the driver is ahead by
//   `k = 16` (~76%), so the crossover is halved to 8
/// Compiled default for [`small_k_threshold`], ignoring any env override. Public so a calibration
/// tool (gemmkit-tune) can neutralize the env and use the shipped default as its baseline without
/// hand-copying this arch-split value (which could silently desync)
#[cfg(target_arch = "aarch64")]
pub const SMALL_K_THRESHOLD_DEFAULT: usize = 8;
/// See the aarch64 variant
#[cfg(not(target_arch = "aarch64"))]
pub const SMALL_K_THRESHOLD_DEFAULT: usize = 16;
static SMALL_K_THRESHOLD: Threshold =
    Threshold::new("GEMMKIT_SMALL_K_THRESHOLD", SMALL_K_THRESHOLD_DEFAULT);

// Largest `m` *and* `n` for which a shape (with the contraction `k` not itself tiny, and A/B
// streaming contiguously along `k`) takes the small-matrix horizontal (inner-product) route:
// each output element is a single SIMD-reduced dot over `k`, reading A/B in place with no
// packing/blocking. The driver pads tiny row/col tiles to a full microtile, wasting most of its
// work when both output dimensions are far below the microtile; this route computes exactly the
// `m*n` outputs
/// Compiled default for [`small_mn_dim`] (before any env/setter override)
pub const SMALL_MN_DIM_DEFAULT: usize = 16;
static SMALL_MN_DIM: Threshold = Threshold::new("GEMMKIT_SMALL_MN_DIM", SMALL_MN_DIM_DEFAULT);

// Minimum `k` (exclusive) at which the small-`m,n` horizontal route engages its PACK tier: a shape
// that clears the `small_mn_dim` gates but whose operand is strided along `k` (an all-row-major or
// all-col-major small-`m,n` GEMM) copies the failing operand into `k`-contiguous scratch (a `~1/m`
// or `~1/n` tax) and runs the same horizontal dot, instead of falling to the register-tiling
// driver. The zero-copy tier (both operands already unit-stride along `k`) is unaffected and keeps
// firing at `k > small_k_threshold`. Below this `k` a strided shape stays on the driver: the pack
// copy no longer amortizes against the driver gap. The crossover is a **machine** property (pack
// copy cost vs the driver's padded-microtile deficit, and the cache geometry the packed re-reads
// hit), the same class as `small_k_threshold`, so it is a knob. Calibrated on Zen5 (AVX-512): the
// packed route beats the driver at every measured `k` for every small shape (1.1x at `16x16 k=32`,
// up to ~6.8x at `4x4`, never a regression), so the gate sits right at the small-`k` boundary - a
// strided small-`m,n` shape packs as soon as `k` grows past where an in-place shape leaves small_k
/// Compiled default for [`small_mn_pack_min_k`] (before any env/setter override)
pub const SMALL_MN_PACK_MIN_K_DEFAULT: usize = 16;
static SMALL_MN_PACK_MIN_K: Threshold =
    Threshold::new("GEMMKIT_SMALL_MN_PACK_MIN_K", SMALL_MN_PACK_MIN_K_DEFAULT);

// Byte floor below which a bandwidth-bound gemv/gevv stays single-threaded: below it the
// touched data (the matrix for gemv, the output for gevv) is LLC-resident and one core gets
// the full LLC bandwidth, so splitting only loses (fork/join + shared-LLC contention, no DRAM
// to gain). `0` (the default) derives the floor from the LLC size (see
// `crate::cache::gemv_parallel_floor_bytes`); any non-zero value overrides it
/// Compiled default for [`gemv_parallel_bytes`] (before any env/setter override); `0` = auto (LLC-derived)
pub const GEMV_PARALLEL_BYTES_DEFAULT: usize = 0;
static GEMV_PARALLEL_BYTES: Threshold =
    Threshold::new("GEMMKIT_GEMV_PARALLEL_BYTES", GEMV_PARALLEL_BYTES_DEFAULT);

// Maximum workers a bandwidth-bound gemv/gevv may use. `0` (the default) derives a proxy
// from the logical core count (see `parallel::bandwidth_cap`); any non-zero value is a hard
// cap. This is the escape hatch for the fact that the memory-parallel width is a heuristic:
// no physical-core / memory-channel count is exposed, and DRAM saturates at far fewer
// workers than the logical core count. Raise it on a high-bandwidth shared-L2 part (Apple)
/// Compiled default for [`gemv_thread_cap`] (before any env/setter override); `0` = auto (core-derived)
pub const GEMV_THREAD_CAP_DEFAULT: usize = 0;
static GEMV_THREAD_CAP: Threshold =
    Threshold::new("GEMMKIT_GEMV_THREAD_CAP", GEMV_THREAD_CAP_DEFAULT);

// Dynamic-scheduling granularity: the parallel driver aims for this many work
// chunks *per worker*, handed out from a shared cursor on demand, so faster cores
// (heterogeneous big.LITTLE P/E layouts) pull proportionally more. Higher = finer
// load balance and a smaller tail, at the cost of more atomic claims (and, on the
// rare packed-LHS path, more re-packing at chunk edges); lower = coarser balance
// but less overhead
/// Compiled default for [`parallel_oversample`] (before any env/setter override)
pub const PARALLEL_OVERSAMPLE_DEFAULT: usize = 8;
static PARALLEL_OVERSAMPLE: Threshold =
    Threshold::new("GEMMKIT_PARALLEL_OVERSAMPLE", PARALLEL_OVERSAMPLE_DEFAULT);

// Auto worker-count ramp granularity (units of linear problem dimension per
// worker): the auto `Rayon(0)` path targets `cbrt(m*n*k).div_ceil(this)` workers
// `0` (the default) means *auto*: derive the stride from the core count (see
// [`thread_dim_stride`]); any non-zero env/setter value overrides verbatim
/// Compiled default for [`thread_dim_stride`] (before any env/setter override); `0` = auto (core-derived)
pub const THREAD_DIM_STRIDE_DEFAULT: usize = 0;
static THREAD_DIM_STRIDE: Threshold =
    Threshold::new("GEMMKIT_THREAD_DIM_STRIDE", THREAD_DIM_STRIDE_DEFAULT);

// Worker count for a threaded wasm build (`wasm_threads` feature). wasm has no
// `available_parallelism`, so the deployer sets the parallel width here instead: it caps
// `auto_threads` and sizes gemmkit's wasm rayon pool. Off-target builds stay serial via the
// `RAYON_USABLE` guard regardless
/// Compiled default for [`wasm_threads`] (before any env/setter override)
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub const WASM_THREADS_DEFAULT: usize = 8;
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
static WASM_THREADS: Threshold = Threshold::new("GEMMKIT_WASM_THREADS", WASM_THREADS_DEFAULT);

// Minimum `m*n*k` for the shared-LHS A-pack to engage (on top of the runtime
// `n_mc < n_threads` redundancy guard in the driver). The shared pre-pass removes
// redundant per-worker packs but adds a fork-join barrier per depth slice; it pays
// only once the problem is large enough to amortize that barrier, so small/mid
// sizes regress and it is gated above the crossover
//
// The crossover is a **machine** property, not a tile property
/// Compiled default for [`shared_lhs_mnk`], ignoring any env override. Public for the same reason as
/// [`SMALL_K_THRESHOLD_DEFAULT`]: a calibration tool can read the shipped default directly instead of
/// mirroring this arch-/width-split value
#[cfg(target_arch = "aarch64")]
pub const SHARED_LHS_MNK_DEFAULT: usize = 50_000_000;
/// See the aarch64 variant
#[cfg(all(not(target_arch = "aarch64"), target_pointer_width = "64"))]
pub const SHARED_LHS_MNK_DEFAULT: usize = 8_000_000_000;
// A 32-bit `usize` cannot hold the 8e9 literal above; the 32-bit default is
// `usize::MAX` instead, which disables the pre-pass
/// See the aarch64 variant
#[cfg(all(not(target_arch = "aarch64"), not(target_pointer_width = "64")))]
pub const SHARED_LHS_MNK_DEFAULT: usize = usize::MAX;
static SHARED_LHS_MNK: Threshold = Threshold::new("GEMMKIT_SHARED_LHS_MNK", SHARED_LHS_MNK_DEFAULT);

// Largest `k` for which an axpy-shape gemv register-blocks the output (holds the output panel in
// registers across the whole k-sweep). Above it the many in-place matrix column-streams exceed the
// hardware prefetcher window and the plain column-outer form wins. Calibrated on Zen5:
// register-blocking wins by `k <= 16`, is a wash near `k = 32`, regresses by `k ~ 48`
/// Compiled default for [`k_stream_max`], ignoring any env override. Public for the same reason as
/// [`SMALL_K_THRESHOLD_DEFAULT`]: a calibration tool reads it directly instead of hard-coding `32`
pub const K_STREAM_MAX_DEFAULT: usize = 32;
static K_STREAM_MAX: Threshold = Threshold::new("GEMMKIT_K_STREAM_MAX", K_STREAM_MAX_DEFAULT);

// aarch64 batched-GEMM crossover: a batch element splits across the machine (`SequentialInternal`)
// rather than running one-per-worker cache-hot once its per-batch-worker byte share
// `elem_bytes / batch` exceeds this. Only the aarch64 `resolve_batch` reads it; the non-aarch64
// value is inert (the knob and its env var still exist there, but nothing consults them). Calibrated
// on M4 Max (shared cluster-L2 + high unified bandwidth): ~128 KiB separates the DRAM/L2-bandwidth-
// scaling elements (split across the cluster) from the small cache-hot ones (one-per-worker)
/// Compiled default for [`seq_internal_bytes_per_worker`], ignoring any env override. Public so a
/// calibration tool can read the shipped default as its baseline. Only consulted on aarch64 (inert
/// on other targets), but the default is not `#[cfg]`-split: there is nothing arch-specific to
/// calibrate for a value no other target reads
pub const SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT: usize = 128 * 1024;
static SEQ_INTERNAL_BYTES_PER_WORKER: Threshold = Threshold::new(
    "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
    SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT,
);

// Packed-LHS dynamic-scheduling split target: `packed_block_grain` splits each row-block into
// power-of-two column sub-chunks until there are at least `this * n_threads` chunks, trading pack
// reuse for load balance. Distinct from `PARALLEL_OVERSAMPLE` (the general-path grain, default 8);
// this is the packed path's swept optimum: splitting harder re-packs A too often and regresses
/// Compiled default for [`packed_oversample`] (before any env/setter override)
pub const PACKED_OVERSAMPLE_DEFAULT: usize = 2;
static PACKED_OVERSAMPLE: Threshold =
    Threshold::new("GEMMKIT_PACKED_OVERSAMPLE", PACKED_OVERSAMPLE_DEFAULT);

// MC cap, in microtile rows: the A macro-panel is bounded to `this * MR` rows (BLIS keeps MC a
// small multiple of MR). A calibration point, not an invariant: it currently binds on every
// measured topology, overriding the L2-capacity-derived MC, so a larger L2 does not move MC today
/// Compiled default for [`mc_reg_panels`] (before any env/setter override)
pub const MC_REG_PANELS_DEFAULT: usize = 8;
static MC_REG_PANELS: Threshold = Threshold::new("GEMMKIT_MC_REG_PANELS", MC_REG_PANELS_DEFAULT);

// NC cap, in microtile columns, when the machine reports no L3 (e.g. Apple Silicon): the no-L3
// column block is `min(this * NR, N)`: full-`N` up to this ceiling. With no L3 the BLIS model
// keeps NC large (B streams from DRAM; the A panel is what fits the last-level cache); the cap
// bounds the shared packed-B panel. Dead where an L3 exists. `512` (= 2048 cols at NR = 4) keeps
// typical widths full-`N`
/// Compiled default for [`nc_no_l3_panels`] (before any env/setter override)
pub const NC_NO_L3_PANELS_DEFAULT: usize = 512;
static NC_NO_L3_PANELS: Threshold =
    Threshold::new("GEMMKIT_NC_NO_L3_PANELS", NC_NO_L3_PANELS_DEFAULT);

// Small-matrix shortcut gate: a shape with both `m` and `n` at or below this skips the full BLIS
// blocking model and just keeps A/B panels in L2. One knob bounds both dimensions. The prepack
// paths derive their branch-dodging sentinel as `this + 1`, so raising the gate never makes a
// prepack take the tiny branch
/// Compiled default for [`tiny_block_dim`] (before any env/setter override)
pub const TINY_BLOCK_DIM_DEFAULT: usize = 64;
static TINY_BLOCK_DIM: Threshold = Threshold::new("GEMMKIT_TINY_BLOCK_DIM", TINY_BLOCK_DIM_DEFAULT);

// Tiny-branch `kc` ceiling: in the small-matrix shortcut the depth block is `k` clamped to this
/// Compiled default for [`kc`] (before any env/setter override)
pub const KC_DEFAULT: usize = 512;
static KC: Threshold = Threshold::new("GEMMKIT_KC", KC_DEFAULT);

// Main-model `kc` floor: the L1-fit depth estimate is raised to at least this before the
// last-panel rebalance, so a small L1 never starves the microkernel depth-walk
/// Compiled default for [`kc_min`] (before any env/setter override)
pub const KC_MIN_DEFAULT: usize = 512;
static KC_MIN: Threshold = Threshold::new("GEMMKIT_KC_MIN", KC_MIN_DEFAULT);

// Deep-contraction engage gate, in bytes. A narrow-output family (`f16`/`bf16`, `OUT_IS_ACC =
// false`) runs the whole contraction as one depth panel (`kc = k`) so the narrow output rounds
// once; at large `k` its single RHS micropanel (`nr * k * sizeof(N)` bytes) outgrows L2 and every
// microtile call streams it from L3/DRAM. When that micropanel size exceeds this gate the dispatch
// switches to the f32-output twin (`OUT_IS_ACC = true`), which re-blocks the contraction at the
// cache-model `kc` (multi-slice, panels L2-resident) into an f32 scratch and then narrows once
// `0` (the default) means *auto*: derive the gate from half the detected L2 (see
// `crate::cache::deep_k_engage_bytes`, where the measured cliff sits); any non-zero value is the
// byte threshold verbatim. Bit-identical to the single panel for the common `beta in {0, 1}` (the
// twin continues the same ascending-k accumulation across slices); accurate to tolerance otherwise
/// Compiled default for [`deep_kc_bytes`] (before any env/setter override); `0` = auto (L2-derived)
pub const DEEP_KC_BYTES_DEFAULT: usize = 0;
static DEEP_KC_BYTES: Threshold = Threshold::new("GEMMKIT_DEEP_KC_BYTES", DEEP_KC_BYTES_DEFAULT);

// Strip length for the cache-blocked transpose in the strided packing paths (real and complex):
// the source is walked along its contiguous dimension in strips of this many depth steps and each
// strip scattered into the panel, turning a per-element strided gather into blocked copies. One
// knob backs both packers (identical strip geometry)
/// Compiled default for [`pack_transpose_tile`] (before any env/setter override)
pub const PACK_TRANSPOSE_TILE_DEFAULT: usize = 16;
static PACK_TRANSPOSE_TILE: Threshold =
    Threshold::new("GEMMKIT_PACK_TRANSPOSE_TILE", PACK_TRANSPOSE_TILE_DEFAULT);

// Below this `m*n*k`, an auto-selected VNNI i8 kernel hands a *multi-threaded* problem to the widen
// fallback (VNNI's mandatory RHS-pack barrier outweighs its compute saving on a small parallel
// problem). Bit-identical to VNNI (exact i32), so the swap never perturbs results. Calibrated on
// Zen5: VNNI wins serial at every size and parallel from `n ~ 1024` up. Inert unless the x86 VNNI
// auto path is selected
/// Compiled default for [`i8_vnni_min_par_mnk`] (before any env/setter override)
#[cfg(feature = "int8")]
pub const I8_VNNI_MIN_PAR_MNK_DEFAULT: usize = 768 * 768 * 768;
#[cfg(feature = "int8")]
static I8_VNNI_MIN_PAR_MNK: Threshold =
    Threshold::new("GEMMKIT_I8_VNNI_MIN_PAR_MNK", I8_VNNI_MIN_PAR_MNK_DEFAULT);

/// Get the serial/parallel work gate (`m*n*k` threshold)
pub fn parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.get()
}
/// Override the serial/parallel work gate
pub fn set_parallel_threshold(v: usize) {
    PARALLEL_THRESHOLD.set(v);
}

/// Get the RHS-packing gate (on `m`)
pub fn rhs_pack_threshold() -> usize {
    RHS_PACK_THRESHOLD.get()
}
/// Override the RHS-packing gate
pub fn set_rhs_pack_threshold(v: usize) {
    RHS_PACK_THRESHOLD.set(v);
}

/// Get the LHS-packing gate (per-worker column reuse)
pub fn lhs_pack_threshold() -> usize {
    LHS_PACK_THRESHOLD.get()
}
/// Override the LHS-packing gate
pub fn set_lhs_pack_threshold(v: usize) {
    LHS_PACK_THRESHOLD.set(v);
}

/// Get the LHS-packing depth-stride gate, in bytes: a column-major A whose
/// `csa * sizeof(Lhs)` reaches this is packed to avoid a TLB/cache-hostile strided
/// read, independent of the reuse gate above. `0` (the default) means *auto*: the
/// driver derives the gate from the OS page size
pub fn lhs_pack_stride() -> usize {
    LHS_PACK_STRIDE.get()
}
/// Override the LHS-packing depth-stride gate (bytes); `0` restores auto
pub fn set_lhs_pack_stride(v: usize) {
    LHS_PACK_STRIDE.set(v);
}

/// Get the gemv special-path cap on `min(m, n)`
pub fn gemv_threshold() -> usize {
    GEMV_THRESHOLD.get()
}
/// Override the gemv special-path cap
pub fn set_gemv_threshold(v: usize) {
    GEMV_THRESHOLD.set(v);
}

/// Get the small-`k` route threshold (`k` at/below this takes the generic small-`k` path)
pub fn small_k_threshold() -> usize {
    SMALL_K_THRESHOLD.get()
}
/// Override the small-`k` route threshold
pub fn set_small_k_threshold(v: usize) {
    SMALL_K_THRESHOLD.set(v);
}

/// Get the small-matrix horizontal route dimension cap: a shape with both `m` and `n` at or
/// below this (and `k` above the small-`k` threshold) takes the horizontal inner-product path.
/// `0` disables the route
pub fn small_mn_dim() -> usize {
    SMALL_MN_DIM.get()
}
/// Override the small-matrix horizontal route dimension cap (`0` disables the route)
pub fn set_small_mn_dim(v: usize) {
    SMALL_MN_DIM.set(v);
}

/// Get the small-`m,n` horizontal PACK-tier `k` gate: a small-`m,n` shape whose operand is strided
/// along `k` is copied into `k`-contiguous scratch and run through the horizontal dot only when
/// `k` exceeds this (else it stays on the register-tiling driver). The zero-copy tier (both operands
/// already unit-stride along `k`) ignores this knob and gates on `small_k_threshold`
pub fn small_mn_pack_min_k() -> usize {
    SMALL_MN_PACK_MIN_K.get()
}
/// Override the small-`m,n` horizontal PACK-tier `k` gate
pub fn set_small_mn_pack_min_k(v: usize) {
    SMALL_MN_PACK_MIN_K.set(v);
}

/// Get the gemv/gevv parallelism byte floor. `0` means *auto*: derive it from the LLC size
/// (see `crate::cache::gemv_parallel_floor_bytes`); any non-zero value is the floor verbatim
pub fn gemv_parallel_bytes() -> usize {
    GEMV_PARALLEL_BYTES.get()
}
/// Override the gemv/gevv parallelism byte floor (`0` restores the LLC-derived auto value)
pub fn set_gemv_parallel_bytes(v: usize) {
    GEMV_PARALLEL_BYTES.set(v);
}

/// Get the gemv/gevv worker cap. `0` means *auto*: derive a bandwidth proxy from the
/// core count (see `crate::parallel::bandwidth_cap`); any non-zero value is a hard cap
pub fn gemv_thread_cap() -> usize {
    GEMV_THREAD_CAP.get()
}
/// Override the gemv/gevv worker cap (`0` restores the core-derived auto proxy)
pub fn set_gemv_thread_cap(v: usize) {
    GEMV_THREAD_CAP.set(v);
}

/// Get the parallel dynamic-scheduling oversample factor (chunks per worker).
/// Always `>= 1` so the scheduler can never receive a zero grain
pub fn parallel_oversample() -> usize {
    PARALLEL_OVERSAMPLE.get().max(1)
}
/// Override the parallel dynamic-scheduling oversample factor
pub fn set_parallel_oversample(v: usize) {
    PARALLEL_OVERSAMPLE.set(v);
}

/// Get the shared-LHS A-pack workload gate (`m*n*k` threshold): the shared
/// pre-pack engages at or above it; below, each worker packs its own A
pub fn shared_lhs_mnk() -> usize {
    SHARED_LHS_MNK.get()
}
/// Override the shared-LHS A-pack workload gate (`m*n*k`)
pub fn set_shared_lhs_mnk(v: usize) {
    SHARED_LHS_MNK.set(v);
}

/// Get the axpy-gemv output register-blocking `k` ceiling (register-block when `k <= this`)
pub fn k_stream_max() -> usize {
    K_STREAM_MAX.get()
}
/// Override the axpy-gemv output register-blocking `k` ceiling
pub fn set_k_stream_max(v: usize) {
    K_STREAM_MAX.set(v);
}

/// Get the aarch64 batched-GEMM `SequentialInternal` byte crossover (per batch-worker share)
pub fn seq_internal_bytes_per_worker() -> usize {
    SEQ_INTERNAL_BYTES_PER_WORKER.get()
}
/// Override the aarch64 batched-GEMM `SequentialInternal` byte crossover
pub fn set_seq_internal_bytes_per_worker(v: usize) {
    SEQ_INTERNAL_BYTES_PER_WORKER.set(v);
}

/// Get the packed-LHS dynamic-scheduling split target (chunks per worker). Always `>= 1` so the
/// grain computation can never target zero chunks
pub fn packed_oversample() -> usize {
    PACKED_OVERSAMPLE.get().max(1)
}
/// Override the packed-LHS dynamic-scheduling split target
pub fn set_packed_oversample(v: usize) {
    PACKED_OVERSAMPLE.set(v);
}

/// Get the MC cap in microtile rows (the A macro-panel is bounded to `this * MR` rows). Always
/// `>= 1` so `MC` stays a positive multiple of `MR`
pub fn mc_reg_panels() -> usize {
    MC_REG_PANELS.get().max(1)
}
/// Override the MC cap in microtile rows
pub fn set_mc_reg_panels(v: usize) {
    MC_REG_PANELS.set(v);
}

/// Get the no-L3 NC cap in microtile columns (`NC <= this * NR`; only consulted when the machine
/// reports no L3)
pub fn nc_no_l3_panels() -> usize {
    NC_NO_L3_PANELS.get()
}
/// Override the no-L3 NC cap in microtile columns
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

/// Get the tiny-branch `kc` ceiling (small-matrix depth block = `k` clamped to this). Always `>= 1`
/// so the clamp's upper bound never falls below its lower bound
pub fn kc() -> usize {
    KC.get().max(1)
}
/// Override the tiny-branch `kc` ceiling
pub fn set_kc(v: usize) {
    KC.set(v);
}

/// Get the main-model `kc` floor (the L1-fit depth block is raised to at least this)
pub fn kc_min() -> usize {
    KC_MIN.get()
}
/// Override the main-model `kc` floor
pub fn set_kc_min(v: usize) {
    KC_MIN.set(v);
}

/// Get the deep-contraction engage gate, in bytes: a narrow-output family switches to its
/// f32-output multi-slice twin once its single RHS micropanel (`nr * k * sizeof(N)`) exceeds this.
/// `0` (the default) means *auto*: derive the gate from the detected L2 (see
/// `crate::cache::deep_k_engage_bytes`); any non-zero value is the byte threshold verbatim
pub fn deep_kc_bytes() -> usize {
    DEEP_KC_BYTES.get()
}
/// Override the deep-contraction engage gate (bytes); `0` restores the L2-derived auto value
pub fn set_deep_kc_bytes(v: usize) {
    DEEP_KC_BYTES.set(v);
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

/// Get the i8 VNNI small-parallel fallback gate (`m*n*k`): below it an auto-selected VNNI kernel
/// hands a multi-threaded problem to the widen fallback. Bit-identical to VNNI
#[cfg(feature = "int8")]
pub fn i8_vnni_min_par_mnk() -> usize {
    I8_VNNI_MIN_PAR_MNK.get()
}
/// Override the i8 VNNI small-parallel fallback gate
#[cfg(feature = "int8")]
pub fn set_i8_vnni_min_par_mnk(v: usize) {
    I8_VNNI_MIN_PAR_MNK.set(v);
}

/// Get the auto worker-count ramp granularity (units of linear problem dimension
/// per worker). `0` (the default) derives the stride from the machine's core count
/// (see `auto_thread_dim_stride`); any non-zero env/setter value is used verbatim.
/// Always `>= 1` so the `cbrt(mnk).div_ceil(stride)` ramp cannot divide by zero
pub fn thread_dim_stride() -> usize {
    match THREAD_DIM_STRIDE.get() {
        0 => auto_thread_dim_stride(),
        v => v.max(1),
    }
}
/// Override the auto worker-count ramp granularity (`0` restores the core-derived
/// auto value)
pub fn set_thread_dim_stride(v: usize) {
    THREAD_DIM_STRIDE.set(v);
}

/// As [`thread_dim_stride`] but deriving the auto value from an already-sampled core
/// count: `resolve` needs the count anyway for its worker cap, so it samples
/// `available_parallelism` once and shares it, instead of paying a 2nd
/// affinity/cgroup query inside the stride derivation on every auto-parallel call
#[cfg(feature = "parallel")]
pub(crate) fn thread_dim_stride_for(cores: usize) -> usize {
    match THREAD_DIM_STRIDE.get() {
        0 => auto_stride_for(cores),
        v => v.max(1),
    }
}

/// Get the worker count for a threaded wasm build (default 8). See `WASM_THREADS`.
/// Only exists on `wasm32` with the `wasm_threads` feature, where the runtime cannot
/// report a core count; elsewhere `available_parallelism` is used instead
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn wasm_threads() -> usize {
    WASM_THREADS.get().max(1)
}
/// Override the threaded-wasm worker count (clamped to `>= 1`)
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
pub fn set_wasm_threads(v: usize) {
    WASM_THREADS.set(v.max(1));
}

/// Core-count-derived auto ramp granularity. The ramp saturates all `cores` workers at
/// the linear size `cbrt(mnk) == stride * cores`, so the stride sets how fast a problem
/// ramps to full width. This is an *empirical calibration, not a derivation*: it is fit
/// to 2 measured points: a low/mid-core part that benefits from a fast ramp (small
/// stride) and a higher-core part that wants a slow one (large stride), as
/// `stride = clamp(cores^2/16, 16, 64)`. The real driver is memory-domain topology
/// (cross-domain traffic favors a slower ramp), which cannot be robustly detected, so core
/// count is only a proxy and the interpolation between the 2 anchors is unvalidated.
/// The `16` floor keeps small machines from ramping *more* aggressively than measured (a
/// bare `cores^2/16` gives `1` at 4 cores); the `64` ceiling keeps large ones no more
/// aggressive than the legacy default. Recomputed, not memoized (affinity can change at
/// runtime); `resolve` samples the count once per call and routes it through
/// [`thread_dim_stride_for`], so the hot path pays a single query. Override
/// `GEMMKIT_THREAD_DIM_STRIDE` on any topology this 2-point fit misses
#[cfg(feature = "std")]
fn auto_thread_dim_stride() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    auto_stride_for(cores)
}

/// The `cores -> stride` fit shared by the self-querying getter above and the
/// caller-sampled [`thread_dim_stride_for`]
#[cfg(any(feature = "std", feature = "parallel"))]
fn auto_stride_for(cores: usize) -> usize {
    (cores * cores / 16).clamp(16, 64)
}
/// Without `std` there is no `available_parallelism`; keep the legacy constant
#[cfg(not(feature = "std"))]
fn auto_thread_dim_stride() -> usize {
    64
}
