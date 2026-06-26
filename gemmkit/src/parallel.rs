//! Parallelism control and job splitting (layer L5).
//!
//! The driver flattens its inner work into a 1-D list of jobs (column strips) and
//! workers pull contiguous chunks from a shared [`JobCursor`] *on demand* — a
//! single work-gate, no nested 2×2 tree. Demand-driven pulling means faster cores
//! (heterogeneous big.LITTLE P/E layouts) absorb proportionally more work instead
//! of every core getting an equal indivisible slice bounded by the slowest. Thread
//! count *scales with workload* rather than jumping straight to all cores. The math
//! is identical for serial and parallel runs and independent of *which* worker
//! computes a given tile, so output is bit-identical regardless of thread count.

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
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

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
                let gate = tuning::parallel_threshold();
                if mnk < gate {
                    return 1;
                }
                let max = if req == 0 { auto_threads() } else { req };
                // Half the gate, so the gate behaves as a clean serial→2-thread
                // step: just above it we get two workers, then it scales up.
                let work_per_thread = (gate / 2).max(1);
                let want = (mnk / work_per_thread).max(1);
                want.min(max).min(n_jobs).max(1)
            }
        }
    }
}

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

/// Run `f(tid)` for every worker `tid` in `0..n_threads`, in parallel when the
/// `parallel` feature is on and `n_threads > 1`.
#[cfg(feature = "parallel")]
pub(crate) fn for_each_worker<F>(n_threads: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    if n_threads <= 1 {
        f(0);
    } else {
        use rayon::prelude::*;
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
}
