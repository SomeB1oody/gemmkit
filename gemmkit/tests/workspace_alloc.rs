//! §7.4: reusing a `Workspace` performs **zero heap allocations** after warmup.
//!
//! A counting global allocator wraps the system allocator; we assert that the
//! second and later `gemm_with` calls of the same size allocate nothing.

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

#[test]
fn gemm_with_reuses_workspace_zero_alloc() {
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
    // topology — all the one-time allocations happen here.
    call(&mut ws, &mut c);
    call(&mut ws, &mut c);

    // Now steady state: every further call must allocate nothing.
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
