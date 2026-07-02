//! Parallelism control and job splitting (layer L5).
//!
//! The driver flattens its inner work into a 1-D list of jobs (column strips) and
//! workers pull contiguous chunks from a shared [`JobCursor`] *on demand* — a
//! single work-gate, no nested 2×2 tree. Demand-driven pulling means faster cores
//! (heterogeneous big.LITTLE P/E layouts) absorb proportionally more work instead
//! of every core getting an equal indivisible slice bounded by the slowest. Thread
//! count *scales with workload* rather than jumping straight to all cores. The math
//! is identical for serial and parallel runs and independent of *which* worker
//! computes a given tile, so output is **reproducible** regardless of thread count:
//! a fixed config gives the same result. (That the two are also bitwise-equal today
//! follows from serial and parallel running the same kernel — the promised contract
//! is reproducibility under a fixed config, not bitwise serial-vs-parallel identity.)

use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
use crate::tuning;

/// How to parallelize a GEMM call.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Parallelism {
    /// Single-threaded.
    Serial,
    /// rayon with at most `n` threads; `Rayon(0)` auto-detects.
    Rayon(usize),
}

impl Default for Parallelism {
    fn default() -> Self {
        Parallelism::Rayon(0)
    }
}

#[cfg(feature = "parallel")]
fn auto_threads() -> usize {
    match std::thread::available_parallelism() {
        Ok(n) => n.get(),
        // wasm has no `available_parallelism`
        // so fall back to the tunable wasm worker count — reached only under `RAYON_USABLE`
        #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
        Err(_) => crate::tuning::wasm_threads(),
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm_threads")))]
        Err(_) => 1,
    }
}

/// gemmkit's own rayon pool for threaded wasm, sized by [`crate::tuning::wasm_threads`].
/// rayon's *global* pool auto-sizes from `available_parallelism` (unsupported on wasm), so it
/// would run single-threaded; this explicitly-sized pool is what makes a threaded wasm build
/// actually parallel. Built lazily on first use (only reached when `RAYON_USABLE`).
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

/// Whether rayon can spawn workers at runtime. `false` on a wasm build that hasn't opted into
/// threading, so `parallel` degrades to the serial loop there instead of trapping (baseline
/// `wasm32-wasip1` has no thread runtime). `true` for: non-wasm; the `wasm_threads` opt-in
/// (a threaded wasm runtime — `wasm32-wasip1-threads` / browser + SharedArrayBuffer); or
/// `target_feature = "atomics"` (only settable on nightly `-Zbuild-std`, which is why
/// `wasm_threads` exists). All compile-time, so a `const` — there is no safe runtime probe
/// (spawning to test would panic on threadless wasm).
#[cfg(feature = "parallel")]
const RAYON_USABLE: bool = cfg!(any(
    not(target_arch = "wasm32"),
    feature = "wasm_threads",
    target_feature = "atomics",
));

impl Parallelism {
    /// Resolve the number of job partitions (workers) for a problem of the given
    /// total work `m*n*k` and `n_jobs` available jobs.
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
                // Explicit count: honor it, capped by cores and jobs so
                // `Rayon(huge)` can't over-subscribe or over-allocate pack regions
                // Only auto (below) is heuristic, so `Rayon(n)` gives the exact
                // width the tests and scaling diagnostic ask for.
                if req != 0 {
                    return req.min(auto_threads()).min(n_jobs).max(1);
                }
                // Auto: ramp workers with the linear dimension
                // contention grows with the worker count.
                // The stride is core-count-derived by default and tunable via
                // `GEMMKIT_THREAD_DIM_STRIDE`
                let dim = (mnk as f64).cbrt() as usize; // ≈ n for square problems
                let want = dim.div_ceil(tuning::thread_dim_stride());
                want.min(auto_threads()).min(n_jobs).max(1)
            }
        }
    }

    /// Resolve the worker count for a **bandwidth-bound** shape (gemv / gevv) touching
    /// `bytes_touched` bytes over `rows` partitionable output rows.
    ///
    /// Unlike [`Parallelism::resolve`] — whose `cbrt(mnk)` ramp models *compute* — this gates
    /// on memory: below an LLC-derived byte floor the touched data is cache-resident, so it
    /// stays serial (splitting only loses — the scaling curve dips at a few workers on
    /// fork/join and shared-cache contention, with no DRAM to gain). Above the floor the auto
    /// count steps straight to the topology bandwidth cap: a *few* workers is the worst point
    /// on the curve, so a ramp through it is worse than jumping to the cap. The floor gate
    /// precedes the request (like [`resolve`]'s work gate), so below it even an explicit
    /// `Rayon(n)` stays serial; above it `Rayon(n)` is honored (capped by cores and rows).
    #[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
    pub(crate) fn resolve_bandwidth(self, bytes_touched: usize, rows: usize) -> usize {
        let rows = rows.max(1);
        match self {
            Parallelism::Serial => 1,
            #[cfg(not(feature = "parallel"))]
            Parallelism::Rayon(_) => 1,
            #[cfg(feature = "parallel")]
            Parallelism::Rayon(req) => {
                // Wasm without the threading opt-in.
                if !RAYON_USABLE {
                    return 1;
                }
                // Cache-resident / small: one core already gets the full LLC bandwidth, so
                // splitting only loses (fork/join + shared-cache contention, no DRAM to gain).
                if bytes_touched < crate::cache::gemv_parallel_floor_bytes() {
                    return 1;
                }
                // Explicit count: honor it, capped by cores and rows (like `resolve`), so a
                // forced width is exact for the tests and the scaling diagnostic.
                if req != 0 {
                    return req.min(auto_threads()).min(rows).max(1);
                }
                // Auto: step straight to the cap. A *few* workers is the worst choice for a
                // bandwidth-bound shape — the scaling curve dips there (fork/join + shared-cache
                // contention) before the cap's aggregate DRAM bandwidth pays off — so a ramp
                // through the dip beats neither serial (below the floor) nor the cap.
                bandwidth_cap().min(auto_threads()).min(rows).max(1)
            }
        }
    }
}

/// How to schedule a batched GEMM (many independent products) across workers, chosen by
/// [`Parallelism::resolve_batch`]. Without the `parallel` feature only `Serial` is ever produced.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) enum BatchPlan {
    /// Run the whole batch on the calling thread, each element serially.
    Serial,
    /// Parallelize **across the batch**: `n` workers each run whole GEMMs serially and
    /// cache-hot, so the batch pays one fork/join instead of one per element. Every element runs
    /// serially on one worker, so the batch stays bit-identical across worker counts.
    BatchParallel(usize),
    /// Loop the batch sequentially, giving **each** element the full engine parallelism in turn.
    /// For few (`< budget`) but large, DRAM-bound elements that scale across cores on their own,
    /// where a per-element split beats confining each element to one core. Splitting an element
    /// across workers relies on its per-element route being serial==parallel bit-identical, which
    /// the `m, n > 1` routes (driver / small_k / small_mn, all output-tile-partitioned) are under
    /// the current thread-independent blocking; gemv (held only to reproducibility) is excluded.
    SequentialInternal,
}

impl Parallelism {
    /// Pick the batched schedule for `batch` products of shape `m×k×n` (`sizeof` bytes/element).
    /// The batch is embarrassingly parallel over independent elements — not a single big GEMM — so
    /// this does **not** use the `cbrt(mnk)` compute ramp: it hands whole elements to workers once
    /// the *total* work justifies a fork.
    ///
    /// With enough elements to fill the workers (`batch >= budget`), each worker runs whole GEMMs
    /// serially and cache-hot. With fewer elements than workers, the spare workers would idle, so
    /// the choice is between running each element on one core (batch-parallel) and splitting each
    /// element across all cores in turn (`SequentialInternal`): a **cache-resident** element (its
    /// A/B/C fit L2) saturates one core's L2 and scales poorly, so batch-parallel wins; a larger,
    /// **DRAM-bound** element scales with aggregate core bandwidth, so the per-element split wins.
    /// `SequentialInternal` splits an element across workers, so the batch inherits that element's
    /// serial==parallel behavior. It is used only for `m, n > 1` shapes — whose driver / small_k /
    /// small_mn route reduces each output within one worker, so serial and parallel agree
    /// bit-for-bit under the current thread-independent blocking — never gemv, which the library
    /// holds only to reproducibility.
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
                // Cheap total-work gate first (before probing the core count): a trivially small
                // batch stays serial, so a tiny batch never pays the `available_parallelism` cost.
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
                    // Enough independent elements to keep every worker busy on its own.
                    return BatchPlan::BatchParallel(budget);
                }
                // Fewer elements than workers. A DRAM-bound element (working set spills L2) scales
                // across cores, so hand it the whole machine in turn — but only for `m, n > 1`
                // (whose route reduces each output within one worker, so splitting it across
                // workers keeps the batch reproducible); a gemv-shaped element stays
                // one-core-per-element.
                let elem_bytes = (m.saturating_mul(k) + k.saturating_mul(n) + m.saturating_mul(n))
                    .saturating_mul(sizeof);
                let l2 = crate::cache::topology().l2.effective_bytes().max(1);
                if m > 1 && n > 1 && elem_bytes > l2 {
                    BatchPlan::SequentialInternal
                } else {
                    BatchPlan::BatchParallel(batch)
                }
            }
        }
    }

    /// Worker count for a **heterogeneous** batch of `count` independent products with total work
    /// `total_mnk`. Simpler than [`resolve_batch`]: elements vary in size, so there is no uniform
    /// cache-residency test — just assign whole GEMMs to workers (each run serially) once the total
    /// work clears the gate. Every element runs on one worker, so the batch is bit-identical across
    /// worker counts. Returns `1` for the serial fallback.
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

/// Maximum worker count for a bandwidth-bound shape, from the `GEMMKIT_GEMV_THREAD_CAP`
/// knob (`0` ⇒ this auto proxy; non-zero ⇒ verbatim). DRAM saturates at far fewer workers
/// than the logical core count: SMT siblings share a core's load/store units and memory
/// ports, and only a handful of physical cores saturate the memory controllers. No
/// physical-core / memory-channel count is exposed (`l2.shared_by` is the GEMM-worker
/// cluster size, `1` on x86/Neoverse), so quarter the logical count as a documented proxy
/// (÷2 for SMT, ÷2 because roughly half the physical cores saturate DDR), floored at 2.
/// Calibrated on Zen5, where a bandwidth-bound gemv plateaus around a quarter of the 32
/// logical cores. A high-bandwidth shared-L2 part (Apple) wants more — raise the knob.
#[cfg(feature = "parallel")]
fn bandwidth_cap() -> usize {
    match tuning::gemv_thread_cap() {
        0 => (auto_threads() / 4).max(2),
        v => v.max(1),
    }
}

/// `Send + Sync` raw-pointer shim so worker closures can capture shared matrices. Soundness
/// rests on the caller's invariants: workers write disjoint output tiles / private packing
/// buffers and only read shared inputs, and the safe API validates that `C` does not alias
/// `A`/`B`. Shared by the driver and the [`crate::special`] paths so the single unsafe
/// Send/Sync justification lives in one place.
#[derive(Copy, Clone)]
pub(crate) struct Ptr<T>(pub(crate) *mut T);
// SAFETY: see the type comment — access is disjoint by construction.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

/// A shared, lock-free cursor that hands out contiguous job ranges on demand — the
/// *dynamic* analogue of a static `n_jobs / n_threads` split.
///
/// Build a **fresh** cursor per parallel region — it counts `0..n_jobs` once and is
/// exhausted thereafter.
pub(crate) struct JobCursor {
    next: AtomicUsize,
    n_jobs: usize,
    grain: usize,
}

impl JobCursor {
    /// A cursor over `[0, n_jobs)` handing out chunks of `grain` (clamped to `>= 1`,
    /// since a zero grain would never advance the cursor and so spin forever).
    #[inline]
    pub(crate) fn new(n_jobs: usize, grain: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            n_jobs,
            grain: grain.max(1),
        }
    }

    /// Atomically claim the next `[start, end)` chunk, or `None` once the job space
    /// is exhausted.
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

/// Chunk size for a [`JobCursor`]: aim for `parallel_oversample` chunks per worker
/// so faster cores can pull proportionally more, while keeping chunks coarse enough
/// to amortize the atomic claim. Always `>= 1`. A single worker (serial / feature
/// off) takes the whole job space in one chunk.
#[inline]
pub(crate) fn job_grain(n_jobs: usize, n_threads: usize) -> usize {
    if n_threads <= 1 {
        return n_jobs.max(1);
    }
    let oversample = crate::tuning::parallel_oversample();

    (n_jobs / n_threads.saturating_mul(oversample)).max(1)
}

/// Job-cursor grain for the **packed-LHS** path, where the natural chunk is a whole
/// row-block (`n_nt` jobs) so its A panel packs once and is reused across the block's
/// column tiles. That yields only `n_mc` chunks, so when `n_mc` is a small non-multiple
/// of `n_threads` the `ceil(n_mc / n_threads)` rounding imbalances the workers — some do
/// an extra whole block while the rest idle at the join.
///
/// We split each block into the fewest power-of-two column sub-chunks needed to exceed
/// `2 * n_threads` chunks — but **only by a divisor of `n_nt`**, so a chunk never
/// straddles a row-block boundary. (A non-power-of-two `n_nt` — a tail column panel, or
/// an L3-derived `nc/nr` — would otherwise leave `n_nt % splits != 0`, and the
/// demand-driven [`JobCursor`] would hand workers cross-block chunks that re-pack A; the
/// back-off falls to whole-block grain there rather than straddle.) Each split block is
/// then packed by up to `splits` workers — a bounded, deliberate trade of pack reuse for
/// balance. The `2 *` split target is an empirically swept optimum; splitting harder
/// re-packs too often and regresses.
#[inline]
pub(crate) fn packed_block_grain(n_nt: usize, n_mc: usize, n_threads: usize) -> usize {
    let mut splits = 1usize;
    while n_mc * splits < 2 * n_threads && n_nt / (splits * 2) >= 1 {
        splits *= 2;
    }
    while splits > 1 && !n_nt.is_multiple_of(splits) {
        splits /= 2;
    }
    (n_nt / splits).max(1)
}

/// Run `f(tid)` for every worker `tid` in `0..n_threads`, in parallel when the
/// `parallel` feature is on and `n_threads > 1`.
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
    // Threaded wasm
    #[cfg(all(target_arch = "wasm32", feature = "wasm_threads"))]
    {
        wasm_pool().install(|| (0..n_threads).into_par_iter().for_each(f));
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm_threads")))]
    {
        (0..n_threads).into_par_iter().for_each(f);
    }
}

/// Serial fallback when the `parallel` feature is off.
#[cfg(not(feature = "parallel"))]
pub(crate) fn for_each_worker<F>(n_threads: usize, f: F)
where
    F: Fn(usize),
{
    for tid in 0..n_threads {
        f(tid);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// A cursor's chunks must tile `[0, n_jobs)` exactly — adjacent, disjoint, and
    /// covering — for any grain and any `n_jobs` (incl. the empty range).
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

    /// A zero grain is clamped to 1, so the cursor always terminates (never spins).
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
    /// bijectively — every index handed to exactly one puller, none skipped or
    /// duplicated. This is the soundness property the parallel driver relies on.
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
    /// oversample the `saturating_mul` guards against.
    #[test]
    fn job_grain_is_robust() {
        assert_eq!(job_grain(100, 1), 100); // single worker takes the whole space
        assert_eq!(job_grain(0, 8), 1); // never zero
        let g = job_grain(10_000, 8);
        assert!((1..=10_000).contains(&g));
    }

    /// `packed_block_grain` must always return a **divisor of `n_nt`** (so cursor chunks
    /// never straddle a row-block boundary) for *any* `n_nt` — power of two or not — and
    /// split enough to balance when `n_nt` permits. Guards the straddle/re-pack
    /// regression on tail panels and non-power-of-two L3 `nc/nr`.
    #[test]
    fn packed_block_grain_divides_and_balances() {
        for &n_nt in &[1usize, 2, 3, 4, 96, 127, 128, 192, 500, 512] {
            for &n_mc in &[1usize, 7, 14, 16, 32, 100] {
                for &n_threads in &[2usize, 8, 14, 32] {
                    let g = packed_block_grain(n_nt, n_mc, n_threads);
                    assert!(g >= 1 && g <= n_nt, "grain {g} out of (0, {n_nt}]");
                    // The defining invariant: chunks tile each row-block exactly.
                    assert_eq!(n_nt % g, 0, "grain {g} does not divide n_nt {n_nt}");
                    // When `n_nt` has a power-of-two factor large enough, balance to
                    // > 2*n_threads chunks; powers of two (the common full-panel case)
                    // always can.
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
}
