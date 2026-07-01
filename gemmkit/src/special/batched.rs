//! Batched GEMM: many independent products `C_b = α·A_b·B_b + β·C_b` in one call.
//!
//! This is an **orchestration layer**, not a new microkernel: each element re-dispatches through
//! the full single-GEMM engine ([`crate::dispatch::execute`]), so a batched call composes with
//! the driver / small_k / gemv / horizontal routes automatically. The win over a naive loop of
//! `gemm()` calls comes from assigning **whole GEMMs to workers** — each element runs serially on
//! one core, cache-hot — so the batch pays a single fork/join instead of one per element (the
//! right model for the motivating workload of many tiny matrices).
//!
//! The schedule is chosen by [`Parallelism::resolve_batch`]: batch-level parallelism when there
//! is enough total work and enough elements to keep workers busy; a sequential loop with
//! per-element internal parallelism for the few-but-large DRAM-bound regime; serial otherwise.
//! Elements are independent, so the batch is **reproducible** across worker counts. The serial and
//! batch-parallel schedules run each element serially (no element split across workers), so they
//! are bit-identical across worker counts; the few-but-large schedule splits an element across
//! workers and so is gated to `m, n > 1` shapes, whose route reduces each output within one worker
//! (serial and parallel agree bit-for-bit under the current thread-independent blocking).

use crate::dispatch::{self, GemmScalar, Task};
use crate::parallel::{self, BatchPlan, JobCursor, Parallelism, Ptr};
use crate::workspace::{self, Workspace};

/// Run a strided-batched GEMM: element `bi` uses `A + bi·a_bs`, `B + bi·b_bs`, `C + bi·c_bs`, each
/// with the shared single-element shape `(m, k, n)` and strides. `alpha == 0` / `k == 0` / `m,n
/// == 0` degeneracy is handled per element by [`crate::dispatch::execute`].
///
/// # Safety
/// Every element's pointers must be valid for the region implied by its strides and sizes; the
/// `batch` C regions must be pairwise disjoint and none may alias any A/B (the safe API validates
/// this).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run<T: GemmScalar>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    a_bs: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    b_bs: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    c_bs: isize,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if batch == 0 {
            return;
        }
        // Element `bi`'s task (base pointers advanced by the batch strides).
        let make = |bi: usize| Task {
            m,
            k,
            n,
            alpha,
            a: a.offset(bi as isize * a_bs),
            rsa,
            csa,
            b: b.offset(bi as isize * b_bs),
            rsb,
            csb,
            beta,
            c: c.offset(bi as isize * c_bs),
            rsc,
            csc,
        };

        match par.resolve_batch(m, k, n, core::mem::size_of::<T>(), batch) {
            // Serial: whole batch on this thread, each element single-threaded on the passed ws.
            BatchPlan::Serial => {
                for bi in 0..batch {
                    dispatch::execute(make(bi), Parallelism::Serial, ws);
                }
            }
            // Few but large, DRAM-bound elements: loop the batch, giving each element the full
            // engine parallelism in turn (only reached for `m, n > 1`, whose route reduces each
            // output within one worker, so splitting it stays reproducible). Uses the passed `ws`
            // sequentially (each element's internal driver carves per-worker regions from it — no
            // thread-local re-borrow).
            BatchPlan::SequentialInternal => {
                for bi in 0..batch {
                    dispatch::execute(make(bi), par, ws);
                }
            }
            // Batch-level parallelism: workers pull disjoint element ranges from a shared cursor,
            // and every element runs *serially* on one worker — so no element is split across
            // workers and the batch is bit-identical to the serial run for any worker count,
            // independent of whether a route is serial==parallel bit-identical. Each worker packs
            // through its own persistent batched pool (reused across calls). That pool is distinct
            // from the plain `with_thread_pool` one so a worker running inline while
            // `gemm_batched`'s outer `with_thread_pool` holds the plain pool cannot re-borrow it.
            BatchPlan::BatchParallel(n_threads) => {
                let (ap, bp, cp) = (Ptr(a as *mut T), Ptr(b as *mut T), Ptr(c));
                let cur = JobCursor::new(batch, parallel::job_grain(batch, n_threads));
                parallel::for_each_worker(n_threads, |_tid| {
                    let (ap, bp, cp) = (ap, bp, cp);
                    workspace::with_batch_pool(|wsb| {
                        while let Some((s, e)) = cur.next_chunk() {
                            for bi in s..e {
                                let task = Task {
                                    m,
                                    k,
                                    n,
                                    alpha,
                                    a: (ap.0 as *const T).offset(bi as isize * a_bs),
                                    rsa,
                                    csa,
                                    b: (bp.0 as *const T).offset(bi as isize * b_bs),
                                    rsb,
                                    csb,
                                    beta,
                                    c: cp.0.offset(bi as isize * c_bs),
                                    rsc,
                                    csc,
                                };
                                dispatch::execute(task, Parallelism::Serial, wsb);
                            }
                        }
                    });
                });
            }
        }
    }
}
