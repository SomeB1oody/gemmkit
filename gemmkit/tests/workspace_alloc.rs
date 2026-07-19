//! Reusing a [`Workspace`] performs zero heap allocations once it has grown to a call's
//! needed size
//!
//! A counting global allocator wraps the system allocator to verify that the
//! 2nd and later `gemm_with` calls of the same size allocate nothing

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm_with};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, ns) }
    }
}

#[global_allocator]
static GA: Counting = Counting;

/// Both allocation-accounting phases live in this **single** test on purpose: `ALLOCS` is a
/// process-global counter, so a 2nd `#[test]` in this binary would let libtest record that
/// test's result on the main thread (an allocation) concurrently with a measured window here,
/// forging a false positive. One test => no inter-test concurrency, the only robust design for a
/// global-allocation assertion
#[test]
fn workspace_allocation_behavior() {
    // Phase 1: reusing one workspace across same-size calls is zero-alloc
    {
        let (m, k, n) = (96usize, 80, 64);
        let a = vec![0.5f32; m * k];
        let b = vec![0.25f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let mut ws = Workspace::new();

        let call = |ws: &mut Workspace, c: &mut [f32]| {
            gemm_with(
                ws,
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(c, m, n),
                Parallelism::Serial,
            );
        };

        // Warm up: grows the workspace, primes the dispatch OnceLocks and the cache
        // topology - all the one-time allocations happen here
        call(&mut ws, &mut c);
        call(&mut ws, &mut c);

        // Now steady state: every further call must allocate nothing
        for iter in 0..50 {
            let before = ALLOCS.load(Ordering::Relaxed);
            call(&mut ws, &mut c);
            let after = ALLOCS.load(Ordering::Relaxed);
            assert_eq!(
                after,
                before,
                "iteration {iter}: gemm_with allocated {} times",
                after - before
            );
        }
    }

    // Phase 2: a GEMM whose A is read in place (column-major, `m` an `mr` multiple) and whose
    // B is small enough never to pack must reserve no packing scratch at all, so a fresh, empty
    // `Workspace` handed to it stays un-grown. Measured on a fresh workspace so that if the
    // driver did reserve an A/B region, it would have to grow from zero and allocate: the
    // assertion that it does not is the allocation-behavior win. `m = 256` is a multiple of
    // every dispatched `mr` (<= 32); `n = 96` (> the small-m,n dim) keeps the shape on the
    // general driver while its per-worker column reuse stays under the LHS-pack gate; `k = 64`
    // (> the small-k gate) avoids the in-place small-k route. Phase 1 already primed every
    // one-time global, so nothing here allocates
    {
        let (m, k, n) = (256usize, 64, 96);
        let a = vec![0.5f32; m * k];
        let b = vec![0.25f32; k * n];
        let mut c = vec![0.0f32; m * n];

        let call = |ws: &mut Workspace, c: &mut [f32]| {
            gemm_with(
                ws,
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(c, m, n),
                Parallelism::Serial,
            );
        };

        // A throwaway warm workspace primes this shape's own path (its blocking / job setup),
        // so the measured fresh-workspace call below allocates nothing
        let mut warm = Workspace::new();
        call(&mut warm, &mut c);
        call(&mut warm, &mut c);

        // Fresh, empty workspace: a no-pack GEMM must touch no scratch, so it must not
        // allocate (a reserved-but-unused region would grow it from zero here)
        let mut ws = Workspace::new();
        let before = ALLOCS.load(Ordering::Relaxed);
        call(&mut ws, &mut c);
        let after = ALLOCS.load(Ordering::Relaxed);
        assert_eq!(
            after,
            before,
            "no-pack serial GEMM grew the workspace ({} allocations)",
            after - before
        );
    }
}
