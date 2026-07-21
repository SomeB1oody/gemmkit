//! Parallelism control and job splitting (layer L5)
//!
//! The driver flattens its per-panel work into a 1-D list of (row-block, column-tile)
//! jobs, and workers pull contiguous chunks from a shared [`JobCursor`] on demand: one
//! flat work-gate, no nested tree of splits. Demand-driven pulling lets a faster core
//! (a heterogeneous big.LITTLE P/E layout) absorb proportionally more chunks instead of
//! every core getting an equal slice sized for the slowest. Worker count scales with
//! the workload instead of jumping straight to every core. Blocking and job order are
//! both independent of the thread count, and independent of which worker ends up
//! computing a given tile, so the result is reproducible for a fixed config regardless
//! of how many threads ran it. (Serial and parallel also happen to be bitwise-equal
//! today, because both run the same kernel - the contract this module promises is
//! reproducibility under a fixed config, not bitwise serial-vs-parallel identity)

use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
use crate::tuning;

/// Threading strategy for a GEMM call
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// Run on the calling thread only
    Serial,
    /// Use rayon, capped at `n` workers (further capped by the core and job counts).
    /// `Rayon(0)` picks the worker count automatically
    Rayon(usize),
}

impl Default for Parallelism {
    fn default() -> Self {
        Parallelism::Rayon(0)
    }
}

// Memoized: `available_parallelism` walks affinity AND cgroup-quota state on Linux,
// measured at ~8 us per call on the reference Zen5 box - longer than half of an entire
// 128^3 GEMM - and every `Rayon` resolve consults it, so recomputing per call taxed
// exactly the small problems that can least afford it. The price of caching is that an
// affinity change after the first parallel call is not observed; that is rare enough
// (and recoverable via an explicit `Rayon(n)`) to be the right trade
#[cfg(feature = "parallel")]
fn auto_threads() -> usize {
    use std::sync::OnceLock;
    static AUTO_THREADS: OnceLock<usize> = OnceLock::new();
    *AUTO_THREADS.get_or_init(|| {
        match std::thread::available_parallelism() {
            Ok(n) => n.get(),
            // No `available_parallelism` on bare wasm; fall back to the wasm worker count
            // tunable, reachable only when `RAYON_USABLE` gates the parallel path open
            #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
            Err(_) => crate::tuning::wasm_threads(),
            #[cfg(not(all(target_arch = "wasm32", feature = "wasm_threads")))]
            Err(_) => 1,
        }
    })
}

/// A rayon pool sized by [`crate::tuning::wasm_threads`], since wasm has no
/// `available_parallelism` for rayon's global pool to auto-size from (which would
/// otherwise leave a threaded wasm build running on a single worker). Built lazily
/// on first use, reached only when [`RAYON_USABLE`]
#[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
fn wasm_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(crate::tuning::wasm_threads())
            .build()
            .expect("gemmkit: failed to build the wasm rayon thread pool")
    })
}

/// The auto value for `tuning::full_width_mnk` (a `0` knob): the `m*n*k` above which the auto
/// path leaves the largest private pool tier for the full machine width. Arch-split, each side
/// calibrated on its own reference machine:
/// * x86: between 448^3 (89.9M, where the Zen5 physical-core tier still wins by 7%) and 512^3
///   (134M, where full SMT width wins by 5%)
/// * aarch64: between 224^3 (11.2M, where the M4 Max width/2 tier of 7 still wins by 2.2x) and
///   256^3 (16.8M, where full width, engaging the E-cores, wins by 3%). With no SMT there is no
///   late SMT tax to defer, so the crossover sits an order of magnitude lower
#[cfg(all(
    feature = "parallel",
    not(target_arch = "wasm32"),
    not(target_arch = "aarch64")
))]
const FULL_WIDTH_MNK_AUTO: usize = 110_000_000;
#[cfg(all(feature = "parallel", target_arch = "aarch64"))]
const FULL_WIDTH_MNK_AUTO: usize = 14_000_000;

/// The active exact-fit pool tier sizes, ascending, as a fixed `[size; 3]` array plus a length
/// (no heap). Derived on every call so the `pool_classes` knob stays sweepable in-process (its
/// getter is a cached atomic, cheap). `pool_classes` (clamped to 3) picks how many tiers halve
/// down from half the machine width: 1 tier is width/2, 2 adds width/4, 3 adds width/8. Any tier
/// below 2 workers, or not strictly below the full width, is dropped. An empty result (length 0)
/// means the tiers are disabled or the machine is too narrow, and the caller uses the plain
/// work-ramp width and the ambient pool
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
fn class_sizes() -> ([usize; 3], usize) {
    let n = tuning::pool_classes().min(3);
    let cores = auto_threads();
    let mut out = [0usize; 3];
    let mut count = 0;
    if n == 0 {
        return (out, 0);
    }
    // Ascending: the smallest tier divides the width by 2^n, up through 2^1 (half width)
    let mut div = 1usize << n;
    while div >= 2 {
        let size = cores / div;
        if size >= 2 && size < cores {
            out[count] = size;
            count += 1;
        }
        div /= 2;
    }
    (out, count)
}

/// The persistent exact-fit rayon pool for tier `size`, built lazily on first use, or `None`
/// if it failed to build. There are at most 3 tier sizes (half, quarter, eighth of the machine
/// width), each keyed to its own static `OnceLock` slot by halving level (width/2 -> 0,
/// width/4 -> 1, width/8 -> 2), so a pool is built once and its warm threads are reused across
/// calls rather than rebuilt per call (~12 us/thread to build, ~free to drop). A build failure
/// caches `None` and the caller falls back to the ambient pool, so a perf feature never panics
/// the process
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
fn class_pool(size: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOLS: [OnceLock<Option<rayon::ThreadPool>>; 3] =
        [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let cores = auto_threads();
    // Map the tier size to its stable halving slot; the sizes class_sizes emits are always
    // width/2, width/4, or width/8, so a given size maps to the same slot on every call
    let slot = if size >= cores / 2 {
        0
    } else if size >= cores / 4 {
        1
    } else {
        2
    };
    POOLS[slot]
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(size)
                .thread_name(|i| format!("gemmkit-pool-{i}"))
                .build()
                .ok()
        })
        .as_ref()
}

/// Whether rayon can spawn extra worker threads at runtime. `false` only for a wasm
/// build that has not opted into threading (baseline `wasm32-wasip1` has no thread
/// runtime), where `parallel` degrades to the serial loop instead of trapping. `true`
/// for every non-wasm target, for the `wasm_threads` opt-in (a threaded wasm runtime:
/// `wasm32-wasip1-threads`, or a browser with `SharedArrayBuffer`), and for
/// `target_feature = "atomics"` (settable only via nightly `-Zbuild-std`, which is why
/// `wasm_threads` exists as a stable-toolchain alternative). Every input is known at
/// compile time, hence a `const`: there is no safe runtime probe, since spawning a
/// thread to test would itself panic on a threadless wasm target
#[cfg(feature = "parallel")]
const RAYON_USABLE: bool = cfg!(any(
    not(target_arch = "wasm32"),
    feature = "wasm_threads",
    target_feature = "atomics",
));

impl Parallelism {
    /// Worker count for a compute-bound problem of total work `mnk = m*n*k`, given
    /// `n_jobs` available (row-block, column-tile) jobs to split it into
    #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
    pub(crate) fn resolve(self, mnk: usize, n_jobs: usize) -> usize {
        let n_jobs = n_jobs.max(1);
        match self {
            Parallelism::Serial => 1,
            #[cfg(not(feature = "parallel"))]
            Parallelism::Rayon(_) => 1,
            #[cfg(feature = "parallel")]
            Parallelism::Rayon(req) => {
                // Wasm without the threading opt-in
                if !RAYON_USABLE {
                    return 1;
                }
                let gate = tuning::parallel_threshold();
                if mnk < gate {
                    return 1;
                }
                // An explicit count is honored as-is, only capped by the core and job
                // counts so `Rayon(huge)` cannot over-subscribe or over-allocate pack
                // scratch; the heuristic ramp below applies only to `Rayon(0)`, so a
                // test or the scaling diagnostic gets exactly the width it asked for
                if req != 0 {
                    return req.min(auto_threads()).min(n_jobs).max(1);
                }
                // Auto: scale the worker count with the total work, one worker per
                // `par_mnk_per_worker` block of `m*n*k` (default 2e6, the measured
                // per-worker floor under which fork/join overhead eats the gain)
                // Work-based rather than dimension-based: the measured optimum
                // tracks total flops, not linear size (128^3 runs fastest serial
                // while 384^3 already wants every hardware thread), a curve no
                // linear-dimension stride can fit
                let cores = auto_threads();
                let want = mnk / tuning::par_mnk_per_worker().max(1);
                // With the exact-fit size-class pools active, snap the width to a pool tier
                // instead of taking the raw ramp value: `for_each_worker` runs a tier-fitting
                // width inside a same-width private pool, whose fork-join sees no idle slack
                // Calibrated on the Zen5 9950X (32 HW threads, 16 physical cores): 96^3 through
                // 288^3 want the width/4 tier of 8; 320^3 through 448^3 want the width/2
                // physical-core tier of 16, which beats full SMT width by 7-11% by staying on the
                // physical cores; only from ~110M `m*n*k` up does full width finally win. The
                // M4 Max (aarch64) mirrors that shape at smaller scale: 128^3 through 224^3
                // want its single width/2 tier of 7, and the lower aarch64 auto gate hands
                // 256^3 and up to full width. Below that gate the width never drops below the
                // smallest tier - an exact-fit pool wakes whole either way, so a sub-tier width
                // is pointless above the serial gate
                // The 3/2 tier margin is measured: 256^3 (want 8.4) is still fastest at 8, 288^3
                // (want 11.9) ties, 320^3 (want 16.4) wants 16. Wasm keeps the plain ramp (its
                // dedicated pool has no tiers), so this reduces to the old formula there
                #[cfg(not(target_arch = "wasm32"))]
                let w = {
                    let (tiers, n_tiers) = class_sizes();
                    if n_tiers == 0 {
                        want
                    } else {
                        let full = match tuning::full_width_mnk() {
                            0 => FULL_WIDTH_MNK_AUTO,
                            v => v,
                        };
                        if mnk >= full {
                            cores
                        } else {
                            let tiers = &tiers[..n_tiers];
                            tiers
                                .iter()
                                .copied()
                                .find(|&t| want <= (3 * t) / 2)
                                .unwrap_or(tiers[n_tiers - 1])
                        }
                    }
                };
                #[cfg(target_arch = "wasm32")]
                let w = want;
                w.min(cores).min(n_jobs).max(1)
            }
        }
    }

    /// Worker count for a bandwidth-bound shape (gemv/gevv) touching `bytes_touched`
    /// bytes over `rows` partitionable output rows
    ///
    /// Unlike [`Parallelism::resolve`], whose `cbrt(mnk)` ramp models compute, this gates on
    /// memory: below an LLC-derived byte floor the data is cache-resident and stays serial,
    /// since splitting only adds fork/join and shared-cache contention with no DRAM to gain.
    /// Above the floor, auto steps straight to the topology bandwidth cap instead of ramping
    /// through it: a handful of workers is the worst point on a bandwidth-bound scaling curve,
    /// so ramping through that dip is worse than jumping past it. The floor gate runs before
    /// the request check, like `resolve`'s work gate, so an explicit `Rayon(n)` also stays
    /// serial below it and is honored (capped by cores and rows) above it
    #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
    pub(crate) fn resolve_bandwidth(self, bytes_touched: usize, rows: usize) -> usize {
        let rows = rows.max(1);
        match self {
            Parallelism::Serial => 1,
            #[cfg(not(feature = "parallel"))]
            Parallelism::Rayon(_) => 1,
            #[cfg(feature = "parallel")]
            Parallelism::Rayon(req) => {
                // Wasm without the threading opt-in
                if !RAYON_USABLE {
                    return 1;
                }
                // Below the LLC-resident floor: one core already gets full LLC bandwidth, so
                // splitting only adds fork/join and shared-cache contention with no DRAM to gain
                if bytes_touched < crate::cache::gemv_parallel_floor_bytes() {
                    return 1;
                }
                // An explicit count is honored as-is, capped by the core and row counts like
                // `resolve`, so a test or the scaling diagnostic gets an exact forced width
                if req != 0 {
                    return req.min(auto_threads()).min(rows).max(1);
                }
                // Auto: jump straight to the cap rather than ramp through it. A handful of
                // workers is the worst point on a bandwidth-bound curve (fork/join and
                // shared-cache contention, before the cap's aggregate DRAM bandwidth pays
                // off), so neither serial nor a slow ramp beats the cap here
                let cores = auto_threads();
                bandwidth_cap(cores).min(cores).min(rows).max(1)
            }
        }
    }
}

/// Schedule for a batched GEMM (many independent products) across workers, produced by
/// [`Parallelism::resolve_batch`]. Without the `parallel` feature only `Serial` is ever produced
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) enum BatchPlan {
    /// Run every element on the calling thread, one after another
    Serial,
    /// Split **across the batch**: `n` workers each run whole GEMMs serially and cache-hot,
    /// so the batch pays one fork/join total instead of one per element. Each element still
    /// runs on a single worker, so the result is bit-identical across worker counts
    BatchParallel(usize),
    /// Loop the batch on the calling thread, giving each element the engine's full worker
    /// count in turn. Chosen for fewer elements than there are workers, when those elements
    /// are large and DRAM-bound enough to scale across cores on their own, so splitting one
    /// element beats confining it to a single core. Only used for `m, n > 1` shapes (driver /
    /// small_k / small_mn), whose per-element route is serial==parallel bit-identical under
    /// the current thread-independent blocking; excludes gemv, held only to reproducibility
    SequentialInternal,
}

impl Parallelism {
    /// Pick the batched schedule for `batch` products of shape `m x k x n` (`sizeof` bytes per
    /// element). The batch is independent elements, not one big GEMM, so this does not use the
    /// `cbrt(mnk)` compute ramp: it hands whole elements to workers once the total work justifies
    /// forking at all
    ///
    /// With `batch >= budget` there are enough elements to keep every worker busy on its own,
    /// running whole GEMMs serially and cache-hot. With fewer elements than workers, the spare
    /// workers would otherwise idle, so the choice is between one element per worker
    /// (`BatchParallel`) and splitting each element across every worker in turn
    /// (`SequentialInternal`): a cache-resident element (A/B/C fit L2) saturates one core's L2 and
    /// scales poorly if split, so `BatchParallel` wins there; a larger, DRAM-bound element scales
    /// with aggregate core bandwidth, so splitting wins instead. `SequentialInternal` is offered
    /// only for `m, n > 1` shapes, whose driver / small_k / small_mn route already reduces each
    /// output within one worker and so agrees bit-for-bit between serial and parallel under the
    /// current thread-independent blocking; gemv is excluded, since the library holds it only to
    /// reproducibility, not bitwise serial/parallel agreement
    #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
    pub(crate) fn resolve_batch(
        self,
        m: usize,
        k: usize,
        n: usize,
        sizeof: usize,
        batch: usize,
    ) -> BatchPlan {
        let batch = batch.max(1);
        let elem_mnk = m.saturating_mul(k).saturating_mul(n);
        match self {
            Parallelism::Serial => BatchPlan::Serial,
            #[cfg(not(feature = "parallel"))]
            Parallelism::Rayon(_) => BatchPlan::Serial,
            #[cfg(feature = "parallel")]
            Parallelism::Rayon(req) => {
                if !RAYON_USABLE {
                    return BatchPlan::Serial;
                }
                // Cheap total-work gate before probing the core count, so a trivially small
                // batch never pays the `available_parallelism` cost
                if elem_mnk.saturating_mul(batch) < tuning::parallel_threshold() {
                    return BatchPlan::Serial;
                }
                let budget = if req != 0 {
                    req.min(auto_threads())
                } else {
                    auto_threads()
                };
                if budget <= 1 {
                    return BatchPlan::Serial;
                }
                if batch >= budget {
                    // Enough independent elements to keep every worker busy on its own
                    return BatchPlan::BatchParallel(budget);
                }
                // Fewer elements than workers: choose between splitting each element across
                // the machine in turn (`SequentialInternal`) and running `batch` elements
                // one-per-worker, cache-hot (`BatchParallel`). Only `m, n > 1` shapes may
                // split, since their route reduces each output within one worker and stays
                // reproducible; a gemv-shaped element always gets one core
                let elem_bytes = m
                    .saturating_mul(k)
                    .saturating_add(k.saturating_mul(n))
                    .saturating_add(m.saturating_mul(n))
                    .saturating_mul(sizeof);
                // x86 has a private per-core L2, so a cache-resident element does not scale
                // internally: split only once the element spills its per-core L2 share
                #[cfg(not(target_arch = "aarch64"))]
                let split_wins = elem_bytes > crate::cache::topology().l2.effective_bytes().max(1);
                // Apple's shared cluster-L2 scales even an L2-resident element across the
                // cluster's cores, so plain residency is the wrong test; the crossover is
                // 2-D instead, since `BatchParallel(batch)` wastes `budget - batch` cores and
                // only wins once `batch` is large enough that each worker's share
                // (`elem_bytes / batch`) drops below a small threshold. Calibrated on M4 Max
                // to split above a ~128 KiB share: SequentialInternal wins for 512^3 at
                // batch <= 8, 384^3 and 256^3 at batch <= 4, one-per-worker wins the small
                // remainder (256^3 at batch = 8). The 1-D share rule is approximate (it also
                // splits 384^3 at batch = 8, a ~1.06x miss), but the prior `elem_bytes >
                // l2.effective` test (3.2 MiB) missed every win: 512^3 ran BatchParallel(2)
                // at 0.28x of the per-element split
                #[cfg(target_arch = "aarch64")]
                let split_wins =
                    elem_bytes > batch.saturating_mul(tuning::seq_internal_bytes_per_worker());
                if m > 1 && n > 1 && split_wins {
                    BatchPlan::SequentialInternal
                } else {
                    BatchPlan::BatchParallel(batch)
                }
            }
        }
    }

    /// Worker count for a heterogeneous batch of `count` independent products totaling
    /// `total_mnk` work. Simpler than [`resolve_batch`]: elements vary in size, so there is no
    /// uniform cache-residency test to run, just an assignment of whole GEMMs to workers (each
    /// run serially) once the total work clears the gate. Every element runs on one worker, so
    /// the batch is bit-identical across worker counts. Returns `1` for the serial fallback
    #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
    pub(crate) fn resolve_batch_flat(self, total_mnk: usize, count: usize) -> usize {
        let count = count.max(1);
        match self {
            Parallelism::Serial => 1,
            #[cfg(not(feature = "parallel"))]
            Parallelism::Rayon(_) => 1,
            #[cfg(feature = "parallel")]
            Parallelism::Rayon(req) => {
                if !RAYON_USABLE || total_mnk < tuning::parallel_threshold() {
                    return 1;
                }
                let budget = if req != 0 {
                    req.min(auto_threads())
                } else {
                    auto_threads()
                };
                budget.min(count).max(1)
            }
        }
    }
}

/// Worker cap for a bandwidth-bound shape, from the `GEMMKIT_GEMV_THREAD_CAP` knob (`0`
/// picks this auto proxy, non-zero passes through verbatim). DRAM saturates at far fewer
/// workers than the logical core count: only a handful of physical cores are needed to
/// saturate the memory controllers, and SMT siblings share a core's load/store units and
/// memory ports besides. Neither the physical-core nor the memory-channel count is exposed
/// (`l2.shared_by` is the GEMM-worker cluster size, `1` on x86/Neoverse), so this proxy is
/// an arch-dependent fraction of the logical count, floored at 2
///
/// * x86: a quarter (halved for SMT, halved again since roughly half the physical cores
///   suffice to saturate DDR). Calibrated on Zen5, where a bandwidth-bound gemv plateaus
///   around a quarter of the 32 logical cores
/// * aarch64 (Apple/ARM, no SMT): a half, dropping the SMT-halving factor. Calibrated on
///   M4 Max (10P+4E, ~245 GB/s aggregate): a bandwidth-bound gemv climbs to about 8 of 14
///   workers and then declines as the E-cores add contention rather than bandwidth, so
///   half the logical count (7) sits on the broad t=4-8 plateau. A higher-bandwidth part
///   wants more workers; raise the knob for it
#[cfg(feature = "parallel")]
fn bandwidth_cap(cores: usize) -> usize {
    match tuning::gemv_thread_cap() {
        0 => {
            #[cfg(target_arch = "aarch64")]
            let auto = cores / 2;
            #[cfg(not(target_arch = "aarch64"))]
            let auto = cores / 4;
            auto.max(2)
        }
        v => v.max(1),
    }
}

/// `Send + Sync` wrapper around a raw pointer, so worker closures can capture shared
/// matrix pointers across the rayon boundary. Soundness rests on the caller: workers
/// write disjoint output tiles and private packing scratch and only read shared inputs,
/// and the safe API checks that `C` does not alias `A`/`B`. Shared by the driver and
/// [`crate::special`] so the one unsafe Send/Sync justification lives in a single place
#[derive(Copy, Clone)]
pub(crate) struct Ptr<T>(pub(crate) *mut T);
// SAFETY: see the type doc above - every access is disjoint by construction
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

/// A shared, lock-free cursor handing out contiguous job ranges on demand: the dynamic
/// analogue of a static `n_jobs / n_threads` split
///
/// Build a fresh cursor per parallel region: it counts through `0..n_jobs` once and is
/// exhausted afterward
pub(crate) struct JobCursor {
    next: AtomicUsize,
    n_jobs: usize,
    grain: usize,
}

impl JobCursor {
    /// A cursor over `[0, n_jobs)` handing out chunks of `grain`, clamped to `>= 1`
    /// since a zero grain would never advance the cursor and so spin forever
    #[inline]
    pub(crate) fn new(n_jobs: usize, grain: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            n_jobs,
            grain: grain.max(1),
        }
    }

    /// Atomically claims the next `[start, end)` chunk, or `None` once the job space
    /// is exhausted
    #[inline]
    pub(crate) fn next_chunk(&self) -> Option<(usize, usize)> {
        let start = self.next.fetch_add(self.grain, Ordering::Relaxed);
        if start >= self.n_jobs {
            None
        } else {
            Some((start, (start + self.grain).min(self.n_jobs)))
        }
    }
}

/// Chunk size for a [`JobCursor`], aiming for `parallel_oversample` chunks per worker
/// so a faster core can pull proportionally more while chunks stay coarse enough to
/// amortize the atomic claim. Always `>= 1`; a single worker (serial, or the
/// `parallel` feature off) takes the whole job space in one chunk
#[inline]
pub(crate) fn job_grain(n_jobs: usize, n_threads: usize) -> usize {
    if n_threads <= 1 {
        return n_jobs.max(1);
    }
    let oversample = crate::tuning::parallel_oversample();

    (n_jobs / n_threads.saturating_mul(oversample)).max(1)
}

/// Job-cursor grain for the packed-LHS path, where the natural chunk is a whole row-block
/// (`n_nt` jobs), so its A panel packs once and is reused across the block's column tiles.
/// That yields only `n_mc` chunks, so when `n_mc` is a small non-multiple of `n_threads` the
/// `ceil(n_mc / n_threads)` rounding leaves some workers doing an extra whole block while the
/// rest idle at the join
///
/// Each block splits into the fewest power-of-two column sub-chunks needed to reach
/// `packed_oversample() * n_threads` chunks, but only by a divisor of `n_nt`, so a chunk never
/// straddles a row-block boundary. A non-power-of-two `n_nt` (a tail column panel, or an
/// L3-derived `nc/nr`) would otherwise leave `n_nt % splits != 0`, and the demand-driven
/// [`JobCursor`] would then hand workers cross-block chunks that each re-pack A; the back-off
/// falls to whole-block grain there instead of straddling. Each split block is packed by up to
/// `splits` workers, a bounded and deliberate trade of pack reuse for balance. The split target
/// (`packed_oversample`, default 2) is an empirically swept optimum: splitting harder re-packs
/// too often and regresses
#[inline]
pub(crate) fn packed_block_grain(n_nt: usize, n_mc: usize, n_threads: usize) -> usize {
    let target = crate::tuning::packed_oversample().saturating_mul(n_threads);
    let mut splits = 1usize;
    while n_mc * splits < target && n_nt / (splits * 2) >= 1 {
        splits *= 2;
    }
    while splits > 1 && !n_nt.is_multiple_of(splits) {
        splits /= 2;
    }
    (n_nt / splits).max(1)
}

/// Runs `f(tid)` for every worker `tid` in `0..n_threads`, in parallel once `n_threads > 1`
#[cfg(feature = "parallel")]
pub(crate) fn for_each_worker<F>(n_threads: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    if n_threads <= 1 {
        f(0);
        return;
    }
    // Wasm without the threading opt-in
    if !RAYON_USABLE {
        for tid in 0..n_threads {
            f(tid);
        }
        return;
    }
    use rayon::prelude::*;
    // Threaded wasm: use gemmkit's own explicitly-sized pool, not rayon's global one
    #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
    {
        wasm_pool().install(|| (0..n_threads).into_par_iter().for_each(f));
    }
    // Bare wasm reached here only via `target_feature = "atomics"` (no gemmkit pool): the
    // ambient global pool, as before
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm_threads")))]
    {
        (0..n_threads).into_par_iter().for_each(f);
    }
    // Native: route through an exact-fit private size-class pool when one fits, else the
    // ambient global pool. rayon's fork-join tax scales with a pool's SLACK (its width minus
    // the active workers), not the worker count: measured on the Zen5 9950X, 256^3 at w=8 runs
    // 1978 GFLOP/s in an exact 8-wide pool vs 1152 in a 16-wide pool vs 820 in the default
    // 32-wide global pool. The tier pools are persistent (built once at ~12 us/thread, reused
    // warm), not a fresh exact-width pool per call, which would hop between pools and abandon
    // warm threads
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Already on some rayon pool's worker (a nested gemm, or the caller installed its own
        // pool): run in the current pool rather than nesting into a private one
        if rayon::current_thread_index().is_some() {
            (0..n_threads).into_par_iter().for_each(f);
            return;
        }
        // Smallest active tier that still holds all n_threads workers, installed so its
        // fork-join sees no idle slack. A tier found but whose pool failed to build, or no
        // fitting tier at all, falls through to the ambient global pool
        let (tiers, n_tiers) = class_sizes();
        for &size in &tiers[..n_tiers] {
            if size >= n_threads {
                if let Some(pool) = class_pool(size) {
                    pool.install(|| (0..n_threads).into_par_iter().for_each(f));
                    return;
                }
                break;
            }
        }
        (0..n_threads).into_par_iter().for_each(f);
    }
}

/// Serial fallback used when the `parallel` feature is off
#[cfg(not(feature = "parallel"))]
pub(crate) fn for_each_worker<F>(n_threads: usize, f: F)
where
    F: Fn(usize),
{
    for tid in 0..n_threads {
        f(tid);
    }
}

// Unit tests for job splitting and the job cursor
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// A cursor's chunks must tile `[0, n_jobs)` exactly: adjacent, disjoint, and
    /// covering, for any grain and any `n_jobs` including the empty range
    #[test]
    fn cursor_tiles_range_exactly() {
        for &n_jobs in &[0usize, 1, 2, 7, 100, 1000] {
            for &grain in &[1usize, 3, 8, 64, 1000, 100_000] {
                let cur = JobCursor::new(n_jobs, grain);
                let mut seen = Vec::new();
                while let Some((s, e)) = cur.next_chunk() {
                    assert!(
                        s < e && e <= n_jobs,
                        "chunk [{s}, {e}) escapes [0, {n_jobs})"
                    );
                    seen.extend(s..e);
                }
                assert_eq!(
                    seen,
                    (0..n_jobs).collect::<Vec<_>>(),
                    "n_jobs={n_jobs} grain={grain}"
                );
            }
        }
    }

    /// A zero grain is clamped to 1, so the cursor always terminates rather than spins
    #[test]
    fn zero_grain_clamped_and_terminates() {
        let cur = JobCursor::new(5, 0);
        let mut n = 0;
        while let Some((s, e)) = cur.next_chunk() {
            assert_eq!(e - s, 1);
            n += 1;
        }
        assert_eq!(n, 5);
    }

    /// Under real concurrent pulls the cursor still partitions `[0, n_jobs)`
    /// bijectively: every index goes to exactly one puller, none skipped or
    /// duplicated. This is the soundness property the parallel driver relies on
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn cursor_partition_is_bijective_under_threads() {
        use std::sync::Mutex;
        let n_jobs = 10_000usize;
        let cur = JobCursor::new(n_jobs, 7);
        let collected = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    while let Some((s, e)) = cur.next_chunk() {
                        local.extend(s..e);
                    }
                    collected.lock().unwrap().extend(local);
                });
            }
        });
        let mut all = collected.into_inner().unwrap();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..n_jobs).collect::<Vec<_>>(),
            "indices must partition [0, n_jobs)"
        );
    }

    /// `job_grain` never returns 0 and never panics, even for the adversarial
    /// oversample the `saturating_mul` guards against
    #[test]
    fn job_grain_is_robust() {
        assert_eq!(job_grain(100, 1), 100); // a single worker takes the whole space
        assert_eq!(job_grain(0, 8), 1); // grain is never zero
        let g = job_grain(10_000, 8);
        assert!((1..=10_000).contains(&g));
    }

    /// `packed_block_grain` must always return a divisor of `n_nt` (so cursor chunks
    /// never straddle a row-block boundary), for any `n_nt` whether or not it is a power
    /// of two, and split enough to balance when `n_nt` permits. Guards the straddle/
    /// re-pack regression on tail panels and non-power-of-two L3 `nc/nr`
    #[test]
    fn packed_block_grain_divides_and_balances() {
        for &n_nt in &[1usize, 2, 3, 4, 96, 127, 128, 192, 500, 512] {
            for &n_mc in &[1usize, 7, 14, 16, 32, 100] {
                for &n_threads in &[2usize, 8, 14, 32] {
                    let g = packed_block_grain(n_nt, n_mc, n_threads);
                    assert!(g >= 1 && g <= n_nt, "grain {g} out of (0, {n_nt}]");
                    // The defining invariant: chunks tile each row-block exactly
                    assert_eq!(n_nt % g, 0, "grain {g} does not divide n_nt {n_nt}");
                    // A power-of-two `n_nt` (the common full-panel case) can always
                    // balance to more than 2*n_threads chunks
                    if n_nt.is_power_of_two() && n_nt >= 2 {
                        let chunks = n_mc * (n_nt / g);
                        assert!(
                            chunks >= 2 * n_threads || g == 1,
                            "n_nt={n_nt} n_mc={n_mc} thr={n_threads}: {chunks} chunks underfills"
                        );
                    }
                }
            }
        }
    }

    // Serializes the size-class-pool knob tests below: they mutate the process-global
    // `pool_classes` / `full_width_mnk` knobs, so 2 running concurrently in this binary could
    // interleave their set/restore. Recovers a poisoned lock so 1 panic does not cascade
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    static POOL_KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// With the size-class pools disabled (`pool_classes` = 0), the auto width reproduces the
    /// legacy work-ramp formula exactly (`want.min(cores).min(n_jobs).max(1)`), so turning the
    /// feature off is behavior-preserving. Derived from the live knobs, never a hard-coded width
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn resolve_matches_legacy_formula_with_tiers_off() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = crate::tuning::pool_classes();
        crate::tuning::set_pool_classes(0);
        let cores = auto_threads();
        let per = crate::tuning::par_mnk_per_worker().max(1);
        let gate = crate::tuning::parallel_threshold();
        let n_jobs = 1_000_000usize;
        // A spread of sizes straddling the gate and the whole ramp, plus a huge value
        for &mnk in &[
            gate,
            gate * 2,
            gate * 37,
            per * (cores * 3 + 1),
            usize::MAX / 4,
        ] {
            let want = mnk / per;
            let expect = if mnk < gate {
                1
            } else {
                want.min(cores).min(n_jobs).max(1)
            };
            assert_eq!(
                Parallelism::Rayon(0).resolve(mnk, n_jobs),
                expect,
                "tiers-off mnk={mnk}"
            );
        }
        crate::tuning::set_pool_classes(prev);
    }

    /// With tiers active, every auto width lands in {1, the active tier sizes, cores}: above the
    /// serial gate the width snaps to a pool tier (never a sub-tier value) or, in the full-width
    /// regime, to the full machine width. Trivially passes on a machine too narrow to form a tier
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn resolve_auto_width_lands_on_a_tier_or_cores() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cores = auto_threads();
        if cores < 4 {
            return; // too narrow to form any tier (width/2 would be < 2); nothing to assert
        }
        let prev_pc = crate::tuning::pool_classes();
        let per = crate::tuning::par_mnk_per_worker().max(1);
        let n_jobs = 1_000_000usize;
        for &pc in &[1usize, 2, 3] {
            crate::tuning::set_pool_classes(pc);
            let (tiers, n_tiers) = class_sizes();
            let mut allowed: Vec<usize> = vec![1usize, cores];
            allowed.extend_from_slice(&tiers[..n_tiers]);
            // Sweep the ramp from below the gate through the full-width regime (whether the
            // sweep top clears the arch-split auto full-width gate depends on the machine;
            // widths on both sides of it are in `allowed` either way)
            for want in 0..=(cores * 2 + 2) {
                let mnk = (want * per).max(1);
                let w = Parallelism::Rayon(0).resolve(mnk, n_jobs);
                assert!(
                    allowed.contains(&w),
                    "pc={pc} want={want} mnk={mnk} -> width {w} not in {allowed:?}"
                );
            }
        }
        crate::tuning::set_pool_classes(prev_pc);
    }

    /// A huge `m*n*k` (well past the full-width gate) always routes to the full machine width,
    /// whether or not the tiers are active
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn resolve_huge_mnk_routes_to_full_width() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = crate::tuning::pool_classes();
        crate::tuning::set_pool_classes(2); // tiers active, so the full-width gate is exercised
        let cores = auto_threads();
        let n_jobs = 1_000_000usize;
        let w = Parallelism::Rayon(0).resolve(usize::MAX / 4, n_jobs);
        assert_eq!(w, cores.min(n_jobs).max(1), "huge mnk must take full width");
        crate::tuning::set_pool_classes(prev);
    }
}
