//! Parallelism control and job splitting (layer L5).
//!
//! The driver flattens its inner work into a 1-D list of jobs (column strips)
//! and hands a contiguous slice to each worker — a single work-gate, no nested
//! 2×2 tree. Thread count *scales with workload* rather than jumping straight to
//! all cores. The math is identical for serial and parallel runs, so output is
//! bit-identical regardless of thread count.

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

/// The `[start, end)` job range for worker `tid` of `n_threads`, balanced so the
/// first `n_jobs % n_threads` workers get one extra job.
#[inline]
pub(crate) fn job_range(n_jobs: usize, tid: usize, n_threads: usize) -> (usize, usize) {
    let base = n_jobs / n_threads;
    let rem = n_jobs % n_threads;
    let start = tid * base + tid.min(rem);
    let len = base + if tid < rem { 1 } else { 0 };
    (start, start + len)
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
