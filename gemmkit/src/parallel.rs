//! Parallelism control and job splitting (layer L2)
//!
//! The driver flattens its per-panel work into a 1-D list of row-block and column-tile
//! jobs. Workers pull contiguous chunks from a shared [`JobCursor`] on demand, using one
//! flat work gate instead of a nested tree of splits
//!
//! Demand-driven pulling lets a faster core absorb more chunks than a slower core. This
//! helps a heterogeneous P-core and E-core layout, where an equal fixed slice would size
//! every core for the slowest one. Worker count scales with the workload instead of
//! jumping straight to every core
//!
//! Blocking and job order do not depend on the thread count. They also do not depend on
//! which worker computes a given tile. The result is reproducible for a fixed config,
//! regardless of how many threads ran it. Serial and parallel also produce bitwise-equal
//! output today, because both run the same kernel. The contract this module keeps is
//! reproducibility under a fixed config, not bitwise serial-versus-parallel identity

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

// Memoized: `available_parallelism` walks affinity and cgroup-quota state on Linux, and every
// `Rayon` resolve calls it, so recomputing it each time taxes small problems most. Caching
// means a later affinity change goes unseen until an explicit `Rayon(n)` call
#[cfg(feature = "parallel")]
fn auto_threads() -> usize {
    use std::sync::OnceLock;
    static AUTO_THREADS: OnceLock<usize> = OnceLock::new();
    *AUTO_THREADS.get_or_init(|| {
        match std::thread::available_parallelism() {
            Ok(n) => n.get(),
            // No `available_parallelism` exists on bare wasm. Fall back to the wasm
            // worker count tunable, reached only when `RAYON_USABLE` opens the path
            #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
            Err(_) => crate::tuning::wasm_threads(),
            #[cfg(not(all(target_arch = "wasm32", feature = "wasm_threads")))]
            Err(_) => 1,
        }
    })
}

/// The rayon pool for a threaded wasm build. Wasm has no `available_parallelism`, so
/// this sizes the pool from [`crate::tuning::wasm_threads`] instead of letting rayon
/// auto-size its global pool. Built lazily on first use, reached only when
/// [`RAYON_USABLE`] is true
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

/// The auto value for `tuning::full_width_mnk`, a `0` knob
///
/// This sets the `m*n*k` threshold above which the auto path leaves the largest private
/// pool tier for the full machine width
///
/// The value is arch-split. On x86, SMT means the extra logical threads help only once
/// the work is large enough to hide their overhead, so the threshold sits higher. On
/// aarch64 there is no SMT tax to defer, so the threshold sits an order of magnitude lower.
/// This lets the full width, including the E-cores, engage sooner
#[cfg(all(
    feature = "parallel",
    not(target_arch = "wasm32"),
    not(target_arch = "aarch64")
))]
const FULL_WIDTH_MNK_AUTO: usize = 110_000_000;
#[cfg(all(feature = "parallel", target_arch = "aarch64"))]
const FULL_WIDTH_MNK_AUTO: usize = 14_000_000;

/// The auto value for `tuning::gemv_tier_step`, a `0` knob: the factor in touched bytes
/// between 2 rungs of the bandwidth-bound worker ladder in [`bandwidth_cap`]
///
/// Not arch-split, because the only shipped multi-tier default is the one this value
/// fits. A machine whose bandwidth scales differently at a different byte scale can
/// sweep the knob to match it
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
const GEMV_TIER_STEP_AUTO: usize = 8;

/// The active exact-fit pool tier sizes, ascending, as a fixed `[size; 3]` array plus a
/// length. Derived on every call, since `pool_classes` is a cheap cached atomic and this
/// keeps the knob sweepable at runtime
///
/// `pool_classes` (clamped to 3) sets how many tiers halve down from half the machine
/// width: 1 tier gives width/2, 2 adds width/4, 3 adds width/8. A tier below 2 workers, or
/// not strictly below the full width, is dropped. An empty result means the tiers are off
/// or the machine is too narrow, and the caller falls back to the plain work-ramp width
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

/// The persistent exact-fit rayon pool for tier `size`, built lazily on first use, or
/// `None` if the build failed
///
/// There are at most 3 tier sizes: half, quarter, and eighth of the machine width. Each
/// maps to its own static `OnceLock` slot by halving level: width/2 to 0, width/4 to 1,
/// width/8 to 2. A pool builds once, and its threads stay warm across calls instead of
/// rebuilding each time. A build failure caches `None`, so the caller falls back to the
/// ambient pool and this never panics the process
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
fn class_pool(size: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOLS: [OnceLock<Option<rayon::ThreadPool>>; 3] =
        [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let cores = auto_threads();
    // Map the tier size to its stable halving slot. The sizes class_sizes emits are
    // always width/2, width/4, or width/8, so a given size maps to the same slot
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

/// Whether rayon can spawn extra worker threads at runtime
///
/// This is `false` only for a wasm build that has not opted into threading, since the
/// baseline `wasm32-wasip1` target has no thread runtime. There `parallel` degrades to
/// the serial loop instead of trapping
///
/// This is `true` for every non-wasm target and for the `wasm_threads` opt-in, a threaded
/// wasm runtime such as `wasm32-wasip1-threads`, or a browser with `SharedArrayBuffer`. It
/// is also `true` for `target_feature = "atomics"`, settable only via nightly `-Zbuild-std`.
/// That is why `wasm_threads` exists as a stable-toolchain alternative
///
/// Every input is known at compile time, so this is a `const`. No safe runtime probe
/// exists, because spawning a thread to test would itself panic on a threadless wasm
/// target
#[cfg(feature = "parallel")]
const RAYON_USABLE: bool = cfg!(any(
    not(target_arch = "wasm32"),
    feature = "wasm_threads",
    target_feature = "atomics",
));

impl Parallelism {
    /// Worker count for a compute-bound problem of total work `mnk = m*n*k`, given
    /// `n_jobs` available row-block and column-tile jobs to split it into
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
                // An explicit count is capped by the core and job counts, so
                // `Rayon(huge)` cannot over-subscribe or over-allocate pack scratch
                if req != 0 {
                    return req.min(auto_threads()).min(n_jobs).max(1);
                }
                // Auto: scale the worker count with total work, 1 worker per
                // `par_mnk_per_worker` block of `m*n*k`, the floor below which fork/join
                // overhead outweighs the gain. This is work-based, not dimension-based,
                // since the optimal count tracks total flops, not linear size
                let cores = auto_threads();
                let want = mnk / tuning::par_mnk_per_worker().max(1);
                // With the exact-fit pools active, snap the width to a pool tier instead of the raw
                // ramp value. So `for_each_worker` runs in a same-width private pool with no idle
                // slack. Below the smallest tier, width stays there since a smaller one gains
                // nothing. Wasm keeps the plain ramp, since it has no tiers
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
                                // Round up to the tier that comfortably covers `want`, so a
                                // `want` just above a tier still uses it instead of jumping up
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

    /// Worker count for a bandwidth-bound shape, such as gemv or gevv, touching
    /// `bytes_touched` bytes over `rows` partitionable output rows
    ///
    /// Unlike [`Parallelism::resolve`], whose work-based ramp models compute, this gates on
    /// memory. Below a cache-derived byte floor the data fits one core's private cache,
    /// which that core saturates alone, so splitting only adds fork/join and cache
    /// contention
    ///
    /// Above the floor, auto steps straight to the [`bandwidth_cap`] width for those bytes
    /// instead of ramping up to it. A handful of workers is the worst point on a
    /// bandwidth-bound scaling curve, so a direct jump beats ramping through that dip. The
    /// width itself still climbs with size, but in pool-tier steps
    ///
    /// The floor gate runs before the request check, like `resolve`'s work gate. An
    /// explicit `Rayon(n)` also stays serial below the floor, and is capped by the core
    /// and row counts above it
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
                // Below the floor the data fits one core's private cache, so splitting only
                // adds fork/join and cache contention, with no bandwidth to gain
                if bytes_touched < crate::cache::gemv_parallel_floor_bytes() {
                    return 1;
                }
                // An explicit count is capped by the core and row counts, like `resolve`
                if req != 0 {
                    return req.min(auto_threads()).min(rows).max(1);
                }
                // Auto: jump straight to the width for these bytes, rather than ramp up to
                // it. A handful of workers is the worst point on a bandwidth-bound curve
                let cores = auto_threads();
                bandwidth_cap(cores, bytes_touched)
                    .min(cores)
                    .min(rows)
                    .max(1)
            }
        }
    }
}

/// Schedule for a batched GEMM, many independent products run across workers, produced
/// by [`Parallelism::resolve_batch`]. Without the `parallel` feature only `Serial` is
/// produced
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) enum BatchPlan {
    /// Run every element on the calling thread, one after another
    Serial,
    /// Split across the batch: `n` workers each run whole GEMMs serially and cache-hot,
    /// so the batch pays one fork/join total instead of one per element. Each element
    /// still runs on a single worker, so the result is bit-identical across worker counts
    BatchParallel(usize),
    /// Loop the batch on the calling thread, giving each element the full worker count in
    /// turn. Chosen when there are fewer elements than workers and each element is large
    /// and DRAM-bound enough to scale across cores on its own
    ///
    /// Used only for `m, n > 1` shapes, such as the driver, `small_k`, and `small_mn`
    /// routes. Their per-element route is bit-identical between serial and parallel under
    /// the current thread-independent blocking. This excludes gemv, which the library
    /// holds only to reproducibility, not bitwise agreement
    SequentialInternal,
}

impl Parallelism {
    /// Pick the batched schedule for `batch` products of shape `m x k x n`, `sizeof` bytes
    /// per element
    ///
    /// The batch is independent elements, not one big GEMM, so this does not use the
    /// work-based `m*n*k` compute ramp. It hands whole elements to workers once the total
    /// work justifies forking at all
    ///
    /// With `batch >= budget` there are enough elements to keep every worker busy on its
    /// own, running whole GEMMs serially and cache-hot. With fewer elements than workers,
    /// the spare workers would otherwise idle. The choice is then between one element per
    /// worker (`BatchParallel`) and splitting each element across every worker in turn
    /// (`SequentialInternal`)
    ///
    /// A cache-resident element, where A, B, and C fit the private cache, saturates one
    /// core's cache and scales poorly if split, so `BatchParallel` wins there. A larger,
    /// DRAM-bound element scales with aggregate core bandwidth, so splitting wins instead
    ///
    /// `SequentialInternal` is offered only for `m, n > 1` shapes. Their route already
    /// reduces each output within one worker, so it agrees bit-for-bit between serial and
    /// parallel under the current thread-independent blocking. Gemv is excluded, since the
    /// library holds it only to reproducibility, not bitwise serial-parallel agreement
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
                // the machine (`SequentialInternal`) and running one element per worker,
                // cache-hot (`BatchParallel`). Only `m, n > 1` shapes may split
                let elem_bytes = m
                    .saturating_mul(k)
                    .saturating_add(k.saturating_mul(n))
                    .saturating_add(m.saturating_mul(n))
                    .saturating_mul(sizeof);
                // x86 has a private per-core L2, so a cache-resident element does not scale
                // internally. Split only once the element spills its per-core L2 share
                #[cfg(not(target_arch = "aarch64"))]
                let split_wins = elem_bytes > crate::cache::topology().l2.effective_bytes().max(1);
                // A cluster-shared L2 scales even an L2-resident element across the
                // cluster's cores, so plain residency is the wrong test. The crossover is
                // 2-D instead. `BatchParallel(batch)` wastes `budget - batch` cores and wins
                // only once each worker's share, `elem_bytes / batch`, drops below a
                // threshold
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
    /// `total_mnk` work
    ///
    /// Simpler than [`resolve_batch`], since elements vary in size, so there is no
    /// uniform cache-residency test to run. Once the total work clears the gate, this
    /// assigns whole GEMMs to workers, each run serially. Every element runs on one
    /// worker, so the batch is bit-identical across worker counts. Returns `1` for the
    /// serial fallback
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

/// Worker width for a bandwidth-bound shape touching `bytes` bytes, from the
/// `GEMMKIT_GEMV_THREAD_CAP` knob. A `0` knob picks the auto ladder below, a non-zero
/// knob passes through as a flat width
///
/// A gemv saturates its bandwidth well below the logical core count. Past that point,
/// extra workers only add fork/join and cross-cache contention, so the auto width tops
/// out at half the logical count. Neither the physical-core nor the memory-channel count
/// is exposed, so a fraction of the logical count is the available proxy
///
/// The width is a ladder over bytes, not one fraction. The fastest width shifts with how
/// much of the touched data stays resident in a shared cache. The rungs are the
/// [`class_sizes`] pool tiers themselves, rather than a private set of fractions. An auto
/// gemv width always has an exact-fit pool waiting in [`for_each_worker`], and never pays
/// rayon's slack tax. `gemv_tier_step` sets how many bytes apart the rungs sit, starting
/// from the serial floor. With the tiers off, or on a machine too narrow to form one,
/// there are no rungs to climb. The width then falls back to the flat half
///
/// A machine with only one default tier makes the ladder degenerate to that single width.
/// This is deliberate, since a 2nd tier there would also change the compute path's
/// tier snapping
#[cfg(feature = "parallel")]
fn bandwidth_cap(cores: usize, bytes: usize) -> usize {
    // An explicit width is a per-machine override. It pins the ladder flat rather than
    // capping it, so the knob means what it did before the ladder existed
    let knob = tuning::gemv_thread_cap();
    if knob != 0 {
        return knob.max(1);
    }
    let flat = (cores / 2).max(2);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = bytes;
        flat
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (tiers, n_tiers) = class_sizes();
        if n_tiers == 0 {
            return flat;
        }
        let step = match tuning::gemv_tier_step() {
            0 => GEMV_TIER_STEP_AUTO,
            v => v.max(1),
        };
        // Climb one tier per `step` in touched bytes above the serial floor. Rung `i` owns
        // `[floor * step^i, floor * step^(i+1))`, and the top tier owns everything above. A
        // `step` of 1, or a saturated bound on a huge floor, lands on the top tier
        let mut rung = 0;
        let mut bound = crate::cache::gemv_parallel_floor_bytes();
        while rung + 1 < n_tiers {
            bound = bound.saturating_mul(step);
            if bytes < bound {
                break;
            }
            rung += 1;
        }
        tiers[rung]
    }
}

/// `Send + Sync` wrapper around a raw pointer, so worker closures can capture shared
/// matrix pointers across the rayon boundary
///
/// Soundness rests on the caller. Workers write disjoint output tiles and private packing
/// scratch, and only read shared inputs. The safe API also checks that `C` does not alias
/// `A` or `B`. The driver and [`crate::special`] share this type, so the one unsafe
/// Send/Sync justification lives in a single place
#[derive(Copy, Clone)]
pub(crate) struct Ptr<T>(pub(crate) *mut T);
// SAFETY: see the type doc above. Every access is disjoint by construction
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
    /// A cursor over `[0, n_jobs)` handing out chunks of `grain`, clamped to at least 1.
    /// A zero grain would never advance the cursor, so it would spin forever
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
///
/// A faster core can then pull proportionally more chunks, while each chunk stays coarse
/// enough to amortize the atomic claim
///
/// Always at least 1. A single worker, serial or with the `parallel` feature off, takes
/// the whole job space in one chunk
#[inline]
pub(crate) fn job_grain(n_jobs: usize, n_threads: usize) -> usize {
    if n_threads <= 1 {
        return n_jobs.max(1);
    }
    let oversample = crate::tuning::parallel_oversample();

    (n_jobs / n_threads.saturating_mul(oversample)).max(1)
}

/// Job-cursor grain for the packed-LHS path, where the natural chunk is a whole row-block
/// (`n_nt` jobs)
///
/// Its A panel packs once and is reused across the block's column tiles. This yields only
/// `n_mc` chunks. When `n_mc` is a small non-multiple of `n_threads`, the
/// `ceil(n_mc / n_threads)` rounding gives some workers an extra whole block. The rest then
/// idle at the join
///
/// Each block splits into the fewest power-of-two column sub-chunks needed to reach
/// `packed_oversample() * n_threads` chunks. It splits only by a divisor of `n_nt`, so a
/// chunk never straddles a row-block boundary. A non-power-of-two `n_nt`, such as a tail
/// column panel or an L3-derived `nc/nr`, would otherwise leave `n_nt % splits != 0`. The
/// demand-driven [`JobCursor`] would then hand workers cross-block chunks that each
/// re-pack A, so the back-off falls to whole-block grain instead of straddling
///
/// Each split block is packed by up to `splits` workers, a bounded trade of pack reuse
/// for balance. Splitting harder than the target re-packs too often and regresses
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
    // Bare wasm reaches here only via `target_feature = "atomics"`, with no gemmkit pool,
    // so it uses the ambient global pool
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm_threads")))]
    {
        (0..n_threads).into_par_iter().for_each(f);
    }
    // Native: route through an exact-fit private size-class pool when one fits, else the
    // ambient global pool. Rayon's fork-join tax scales with a pool's slack, its width
    // minus the active workers, not the worker count itself
    //
    // The tier pools are persistent, built once and reused warm. A fresh exact-width pool
    // per call would instead hop between pools and abandon warm threads
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Already on a rayon pool's worker, from a nested gemm or a caller-installed pool. Run
        // in the current pool rather than nesting into a private one
        if rayon::current_thread_index().is_some() {
            (0..n_threads).into_par_iter().for_each(f);
            return;
        }
        // Smallest active tier that still holds all n_threads workers, installed so its
        // fork-join sees no idle slack. A missing or failed-to-build pool falls through
        // to the ambient global pool
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

    /// A zero grain is clamped to 1, so the cursor always terminates instead of spinning
    /// forever
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

    /// Under real concurrent pulls the cursor still partitions `[0, n_jobs)` bijectively.
    /// Every index goes to exactly one puller, none skipped or duplicated. This is the
    /// soundness property the parallel driver relies on
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

    /// `packed_block_grain` must always return a divisor of `n_nt`, for any `n_nt`
    /// whether or not it is a power of two. This makes sure cursor chunks never straddle
    /// a row-block boundary. It must also split enough to balance when `n_nt` permits
    ///
    /// This guards against a regression that straddles blocks and re-packs A on tail
    /// panels and non-power-of-two L3 `nc/nr` values
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

    // Serializes the size-class-pool knob tests below. They mutate the process-global
    // `pool_classes`, `full_width_mnk`, and gemv-ladder knobs, so 2 tests running
    // concurrently could interleave their set and restore. Recovers a poisoned lock so
    // 1 panic does not cascade
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    static POOL_KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// With the size-class pools disabled (`pool_classes` = 0), the auto width reproduces
    /// the plain work-ramp formula exactly, `want.min(cores).min(n_jobs).max(1)`, so
    /// turning the feature off is behavior-preserving. Derived from the live knobs, never
    /// a hard-coded width
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

    /// With tiers active, every auto width lands in 1, an active tier size, or cores.
    /// Above the serial gate the width snaps to a pool tier, never a sub-tier value, or
    /// to the full machine width in the full-width regime. This trivially passes on a
    /// machine too narrow to form a tier
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn resolve_auto_width_lands_on_a_tier_or_cores() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cores = auto_threads();
        if cores < 4 {
            return; // too narrow to form any tier, since width/2 would be under 2
        }
        let prev_pc = crate::tuning::pool_classes();
        let per = crate::tuning::par_mnk_per_worker().max(1);
        let n_jobs = 1_000_000usize;
        for &pc in &[1usize, 2, 3] {
            crate::tuning::set_pool_classes(pc);
            let (tiers, n_tiers) = class_sizes();
            let mut allowed: Vec<usize> = vec![1usize, cores];
            allowed.extend_from_slice(&tiers[..n_tiers]);
            // Sweep the ramp from below the gate through the full-width regime. Whether the
            // sweep clears the arch-split full-width gate depends on the machine, but
            // widths on both sides of it are in `allowed` either way
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

    /// A huge `m*n*k`, well past the full-width gate, always routes to the full machine
    /// width, whether or not the tiers are active
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

    /// The auto bandwidth width climbs the pool-tier ladder with touched bytes
    ///
    /// It stays serial below the floor and takes the smallest tier at the floor itself.
    /// From there it steps up one tier per `gemv_tier_step` factor, reaching the top tier
    /// for everything beyond
    ///
    /// Derived from the live tiers and floor, never a hard-coded width. Skipped on a
    /// machine too narrow to form the 2 tiers a ladder needs
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn resolve_bandwidth_climbs_the_tier_ladder() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_pc = crate::tuning::pool_classes();
        crate::tuning::set_pool_classes(2);
        let (tiers, n_tiers) = class_sizes();
        if n_tiers < 2 {
            crate::tuning::set_pool_classes(prev_pc);
            return; // no ladder to climb on this machine
        }
        let prev_step = crate::tuning::gemv_tier_step();
        let step = 4;
        crate::tuning::set_gemv_tier_step(step);
        let floor = crate::cache::gemv_parallel_floor_bytes();
        // Rows never bind here, so the width is the ladder's alone
        let at = |bytes: usize| Parallelism::Rayon(0).resolve_bandwidth(bytes, usize::MAX);

        assert_eq!(
            at(floor.saturating_sub(1)),
            1,
            "below the floor stays serial"
        );
        assert_eq!(at(floor), tiers[0], "the floor takes the smallest tier");
        let first_step = floor.saturating_mul(step);
        assert_eq!(at(first_step - 1), tiers[0], "just under the first step");
        assert_eq!(at(first_step), tiers[1], "one step up takes the next tier");
        assert_eq!(
            at(usize::MAX / 2),
            tiers[n_tiers - 1],
            "past the ladder takes the top tier"
        );

        // Monotone: more bytes never buy fewer workers
        let mut prev_w = 0;
        let mut bytes = floor;
        for _ in 0..8 {
            let w = at(bytes);
            assert!(
                w >= prev_w,
                "width fell from {prev_w} to {w} at {bytes} bytes"
            );
            prev_w = w;
            bytes = bytes.saturating_mul(2);
        }

        crate::tuning::set_gemv_tier_step(prev_step);
        crate::tuning::set_pool_classes(prev_pc);
    }

    /// A non-zero `gemv_thread_cap` is a manual override, so it pins the width flat and
    /// the size ladder never runs. The same width comes back at every size above the
    /// floor
    #[test]
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn gemv_thread_cap_knob_pins_the_width_flat() {
        let _lock = POOL_KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pinned = 3;
        if auto_threads() < pinned {
            return; // the width would clamp to the core count, not the knob
        }
        let prev = crate::tuning::gemv_thread_cap();
        crate::tuning::set_gemv_thread_cap(pinned);
        let floor = crate::cache::gemv_parallel_floor_bytes();
        for bytes in [floor, floor.saturating_mul(64), usize::MAX / 2] {
            assert_eq!(
                Parallelism::Rayon(0).resolve_bandwidth(bytes, usize::MAX),
                pinned,
                "the pinned width must hold at {bytes} bytes"
            );
        }
        crate::tuning::set_gemv_thread_cap(prev);
    }
}
