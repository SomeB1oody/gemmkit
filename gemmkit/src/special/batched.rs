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

use crate::dispatch::{self, GemmProblem, GemmScalar, Task};
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
            // through the re-entrancy-safe thread-local pool; the calling thread, which still holds
            // that pool via `gemm_batched`'s outer `with_thread_pool`, takes the fresh-scratch
            // fallback for the share it runs inline while the other workers reuse theirs.
            BatchPlan::BatchParallel(n_threads) => {
                let (ap, bp, cp) = (Ptr(a as *mut T), Ptr(b as *mut T), Ptr(c));
                let cur = JobCursor::new(batch, parallel::job_grain(batch, n_threads));
                parallel::for_each_worker(n_threads, |_tid| {
                    let (ap, bp, cp) = (ap, bp, cp);
                    workspace::with_thread_pool(|wsb| {
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

/// Run a **heterogeneous** batch — a slice of independent GEMM problems, each with its own shape
/// and pointers (the pointer-array / grouped form). Parallelizes across the problems: each runs
/// serially on one worker, cache-hot, so the batch pays one fork/join and is bit-identical across
/// worker counts. Elements vary in size, so it uses the flat [`Parallelism::resolve_batch_flat`]
/// policy (no uniform cache-residency test) and never the per-element-internal split. Takes the
/// public [`GemmProblem`] slice directly (no `Vec<Task>` copy) and derives each [`Task`] in place.
///
/// # Safety
/// Each problem's pointers must be valid for its shape/strides; the `problems` output regions must
/// be pairwise disjoint and none may alias any input (the safe API validates this; the unchecked
/// one makes the caller promise it).
pub(crate) unsafe fn run_ptr<T: GemmScalar>(
    problems: &[GemmProblem<T>],
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if problems.is_empty() {
            return;
        }
        // Saturating sum (each term is already clamped) so the parallelism gate can't overflow.
        let total_mnk = problems.iter().fold(0usize, |acc, p| {
            acc.saturating_add(p.m.saturating_mul(p.k).saturating_mul(p.n))
        });
        let n_threads = par.resolve_batch_flat(total_mnk, problems.len());
        if n_threads <= 1 {
            for p in problems {
                dispatch::execute(p.task(), Parallelism::Serial, ws);
            }
            return;
        }
        // Workers pull disjoint problem ranges from a shared cursor; each output region is disjoint
        // by validation, so the shared read-only problem list plus disjoint writes are race-free.
        let base = Ptr(problems.as_ptr() as *mut GemmProblem<T>);
        let cur = JobCursor::new(
            problems.len(),
            parallel::job_grain(problems.len(), n_threads),
        );
        parallel::for_each_worker(n_threads, |_tid| {
            // Capture the whole `Ptr` (Send + Sync), not its raw-pointer field.
            let bp = base;
            workspace::with_thread_pool(|wsb| {
                while let Some((s, e)) = cur.next_chunk() {
                    for pi in s..e {
                        let task = (*(bp.0 as *const GemmProblem<T>).add(pi)).task();
                        dispatch::execute(task, Parallelism::Serial, wsb);
                    }
                }
            });
        });
    }
}
