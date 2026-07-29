//! Batched GEMM: `batch` independent products `C_b <- alpha*A_b*B_b + beta*C_b` in 1 call
//!
//! This is not a new compute strategy. Every element re-enters the ordinary single-GEMM path
//! through [`crate::dispatch::execute`]. A batched call therefore gets the same driver,
//! small_k, small_mn, or gemv routing that each element's shape would get on its own. The
//! gain over a plain loop of `gemm()` calls is in how work reaches workers. Whole elements go
//! to workers, instead of splitting 1 GEMM across all of them, so each element runs serially
//! on 1 core, cache-hot. The batch then pays a single fork/join instead of one per element.
//! This is the shape that wins for many small matrices
//!
//! [`Parallelism::resolve_batch`] picks the schedule per call. It splits across the batch once
//! there is enough total work and enough elements to keep every worker busy. It loops the
//! batch on 1 thread and hands each element the engine's full worker count in turn. This
//! applies when elements are few but large enough to be worth splitting. Otherwise it runs
//! serially. Since every element is independent, the batch result never depends on the worker
//! count
//!
//! The serial and batch-parallel schedules always run an element whole on 1 worker. So those 2
//! schedules agree bit-for-bit with each other at any worker count. The few-large schedule
//! instead splits a single element's own work across workers. It is offered only for `m, n >
//! 1` shapes. That route already reduces every output cell within 1 worker, regardless of how
//! the driver tiles it

#[cfg(feature = "epilogue")]
use crate::dispatch::FusedScalar;
use crate::dispatch::{self, GemmProblem, GemmScalar, Task};
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::FusedEpi;
use crate::parallel::{self, BatchPlan, JobCursor, Parallelism, Ptr};
use crate::workspace::{self, Workspace};

/// Shared driver behind the 2 strided-batched entry points ([`run`], [`run_fused`]). It
/// resolves the [`BatchPlan`] once from the common per-element shape. For every element `bi`
/// in `0..batch`, it builds that element's [`Task`]. The shared `a`/`b`/`c` base pointers
/// advance by `bi * {a,b,c}_bs`. It then hands the task to `exec`, together with the
/// schedule's per-element [`Parallelism`] and a workspace. `exec` is the only difference
/// between [`run`] and [`run_fused`]. It wraps [`crate::dispatch::execute`] or
/// [`crate::dispatch::execute_fused`]. The `Task` construction, the schedule choice, the work
/// partition, and the reproducibility contract all live here once, and both entry points
/// inherit them identically
///
/// # Safety
/// Every element's pointers must be valid for the region its strides and `m`/`k`/`n` imply.
/// The `batch` output regions must be pairwise disjoint, and none may alias any A/B input.
/// `exec` must run its `Task` as one complete GEMM call. Under `BatchPlan::BatchParallel`, it
/// runs on the workspace it is given. Under `Serial` or `SequentialInternal`, it runs with the
/// passed [`Parallelism`] on the shared `ws`
#[allow(clippy::too_many_arguments)]
unsafe fn schedule<T: GemmScalar>(
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
    exec: impl Fn(Task<T>, Parallelism, &mut Workspace) + Copy + Send + Sync,
) {
    unsafe {
        if batch == 0 {
            return;
        }
        // `Ptr` makes the bases `Send + Sync` so the `BatchParallel` closures below can capture
        // them. Every schedule, not only the parallel one, builds its tasks through this `make`
        let (ap, bp, cp) = (Ptr(a as *mut T), Ptr(b as *mut T), Ptr(c));
        let make = move |bi: usize| {
            // Re-bind the `Ptr` values themselves: edition-2021 field capture would otherwise
            // reach into `.0` and capture a bare pointer, which is not `Send + Sync`
            let (ap, bp, cp) = (ap, bp, cp);
            Task {
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
            }
        };

        match par.resolve_batch(m, k, n, core::mem::size_of::<T>(), batch) {
            // Whole batch, 1 element at a time, on the calling thread
            BatchPlan::Serial => {
                for bi in 0..batch {
                    exec(make(bi), Parallelism::Serial, ws);
                }
            }
            // Few large elements, more workers than the batch. Loop the batch on this thread and
            // give each element the engine's full parallelism in turn. This is only reached for
            // `m, n > 1` shapes, whose route already reduces each output within 1 worker, so
            // splitting one stays reproducible. `ws` is reused across elements serially, since
            // each element's own driver call carves its per-worker regions out of it, with no
            // thread-local re-borrow
            BatchPlan::SequentialInternal => {
                for bi in 0..batch {
                    exec(make(bi), par, ws);
                }
            }
            // Batch-level split. Workers claim disjoint element ranges from a shared cursor and
            // run each element whole and serially, so no single element crosses a worker
            // boundary. The batch then matches the serial result at any worker count, whatever
            // a given route's own serial/parallel agreement is. Each worker packs through its
            // own re-entrancy-safe thread-local pool. If the calling thread also runs a share
            // inline while it already holds that pool, for example when called through a
            // pool-wrapped entry point like `gemm_batched`, that share falls back to fresh
            // scratch while the other workers reuse theirs
            BatchPlan::BatchParallel(n_threads) => {
                let cur = JobCursor::new(batch, parallel::job_grain(batch, n_threads));
                parallel::for_each_worker(n_threads, |_tid| {
                    workspace::with_thread_pool(|wsb| {
                        while let Some((s, e)) = cur.next_chunk() {
                            for bi in s..e {
                                exec(make(bi), Parallelism::Serial, wsb);
                            }
                        }
                    });
                });
            }
        }
    }
}

/// Run a strided-batched GEMM. Element `bi` reads and writes `A + bi*a_bs`, `B + bi*b_bs`,
/// and `C + bi*c_bs`. The whole batch shares 1 shape `(m, k, n)` and 1 set of strides. Each
/// element's `alpha == 0`, `k == 0`, or `m,n == 0` degeneracy is handled individually by
/// [`crate::dispatch::execute`]
///
/// # Safety
/// Every element's pointers must be valid for the region its strides and sizes imply. The
/// `batch` output regions must be pairwise disjoint, and none may alias any A/B input. The
/// safe API validates this
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
    // Forward each element's task straight to `dispatch::execute`, no epilogue
    unsafe {
        schedule(
            batch,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            a_bs,
            b,
            rsb,
            csb,
            b_bs,
            beta,
            c,
            rsc,
            csc,
            c_bs,
            par,
            ws,
            |task, par, ws| dispatch::execute(task, par, ws),
        );
    }
}

/// Run a strided-batched GEMM with a fused epilogue. Element `bi` reads and writes
/// `A + bi*a_bs`, `B + bi*b_bs`, and `C + bi*c_bs`. The whole batch shares 1 shape
/// `(m, k, n)` and 1 set of strides. Every element applies the same `epi`, meaning 1 bias
/// vector and 1 activation shared across the whole batch. So
/// `C_bi <- act(alpha*A_bi*B_bi + beta*C_bi + bias)`
///
/// This mirrors [`run`] exactly, with [`crate::dispatch::execute`] replaced by
/// [`crate::dispatch::execute_fused`] in every schedule arm. Element `bi`'s output therefore
/// matches a standalone `gemm_fused` call on that element bit-for-bit. For `f32`/`f64`, that
/// in turn matches plain `gemm()` followed by the same map, for every shape. For `f16`/`bf16`,
/// the epilogue runs in `f32` before the single narrowing round at the store. Per-element
/// `alpha == 0`, `k == 0`, or `m,n == 0` degeneracy is handled by `execute_fused`. `epi` is
/// `Copy`, captured into the parallel workers exactly like the base pointers
///
/// Scheduling and reproducibility match [`run`]. The fused routes reuse the same kernels, so
/// [`Parallelism::resolve_batch`]'s policy carries over unchanged. Every element is
/// independent, so the batch result never depends on the worker count. Serial and
/// batch-parallel run each element whole on 1 worker, so they agree bit-for-bit with each
/// other at any worker count. The few-large schedule instead splits a single element's own
/// work across workers, and is offered only for `m, n > 1` shapes
///
/// # Safety
/// As [`run`]. `epi`'s bias pointer must also be valid for the element's `m` (`PerRow`) or
/// `n` (`PerCol`), and must not alias any `C` region. The safe API validates this
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_fused<T: FusedScalar>(
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
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    // Forward each element's task, plus the shared `epi`, to `dispatch::execute_fused`
    unsafe {
        schedule(
            batch,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            a_bs,
            b,
            rsb,
            csb,
            b_bs,
            beta,
            c,
            rsc,
            csc,
            c_bs,
            par,
            ws,
            move |task, par, ws| dispatch::execute_fused(task, epi, par, ws),
        );
    }
}

/// Run a heterogeneous batch: a slice of independent GEMM problems, each carrying its own
/// shape and pointers. This is the pointer-array, or grouped, form, unlike [`run`]'s single
/// shared shape. Problems are parallelized directly, 1 whole problem per worker, cache-hot,
/// so the batch pays 1 fork/join and each worker's share is independent of the others. Since
/// problems can differ in size, this uses the flat [`Parallelism::resolve_batch_flat`]
/// policy, a total-work gate rather than a per-element cache-residency test. It never splits
/// a single problem's own work across workers, so the result never depends on the worker
/// count. It reads each [`Task`] straight out of the `problems` slice, with no intermediate
/// `Vec<Task>` copy
///
/// # Safety
/// Each problem's pointers must be valid for its own shape and strides. The `problems` output
/// regions must be pairwise disjoint, and none may alias any input. The safe API validates
/// this. The unchecked entry point instead takes the caller's word for it
pub(crate) unsafe fn run_ptr<T: GemmScalar>(
    problems: &[GemmProblem<T>],
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if problems.is_empty() {
            return;
        }
        // Sum with saturation. Each term is already saturated, so a huge batch cannot overflow
        // the gate
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
        // Workers claim disjoint problem ranges from a shared cursor. Validation already
        // guarantees the output regions are pairwise disjoint, so concurrent writes through the
        // shared read-only slice are race-free
        let base = Ptr(problems.as_ptr() as *mut GemmProblem<T>);
        let cur = JobCursor::new(
            problems.len(),
            parallel::job_grain(problems.len(), n_threads),
        );
        parallel::for_each_worker(n_threads, |_tid| {
            // Re-bind the `Ptr` wrapper itself, not its raw-pointer field, to keep it Send + Sync
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
