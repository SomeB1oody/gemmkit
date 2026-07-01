//! Batched GEMM (`gemm_batched`) behavior: it must reproduce a loop of single `gemm()` calls
//! bit-for-bit (same route per element), stay serial==parallel bit-identical, and reject the
//! invalid batch layouts its new validation is responsible for.

use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_batched, gemm_batched_with};

/// Deterministic `f32` fill (xorshift, non-constant so reductions are non-trivial).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Build `batch` contiguously-packed column-major elements of shape `m×k` (stride `m*k`).
fn packed(batch: usize, r: usize, c: usize, seed: u64) -> Vec<f32> {
    fill(batch * r * c, seed)
}

/// `gemm_batched` must equal a loop of single `gemm(Serial)` calls **bit-for-bit** — each element
/// takes the same route, and no element is split across workers. Contiguously-packed column-major
/// elements; `c_init` seeds the `beta` term.
fn assert_batched_matches_loop(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    par: Parallelism,
) {
    let a = packed(batch, m, k, 1);
    let b = packed(batch, k, n, 2);
    let c_init = packed(batch, m, n, 3);

    // Reference: independent single-GEMM per element on its own slice window.
    let mut c_ref = c_init.clone();
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        gemm(
            alpha,
            MatRef::from_col_major(&a[ao..ao + m * k], m, k),
            MatRef::from_col_major(&b[bo..bo + k * n], k, n),
            beta,
            MatMut::from_col_major(&mut c_ref[co..co + m * n], m, n),
            Parallelism::Serial,
        );
    }

    let mut c_bat = c_init.clone();
    gemm_batched(
        batch,
        alpha,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        beta,
        MatMut::new(&mut c_bat, m, n, 1, m as isize),
        (m * n) as isize,
        par,
    );

    assert_eq!(
        c_ref, c_bat,
        "batch={batch} {m}x{k}x{n} alpha={alpha} beta={beta} par={par:?}: batched != gemm() loop"
    );
}

#[test]
fn batched_matches_gemm_loop_many_small() {
    // batch not a multiple of the worker count; non-square, beta != 0, alpha != 1. Small elements
    // exercise the batch-parallel schedule (and the horizontal route inside each element).
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        assert_batched_matches_loop(200, 8, 8, 8, 1.0, 0.0, par);
        assert_batched_matches_loop(101, 6, 40, 5, 2.5, -0.5, par);
        assert_batched_matches_loop(37, 16, 100, 4, -1.0, 1.0, par);
        assert_batched_matches_loop(1, 12, 12, 12, 1.0, 0.0, par); // degenerate single-element batch
    }
}

#[test]
fn batched_matches_gemm_loop_few_large() {
    // Few but large elements: batch-parallel assigns one element per worker (fewer than all
    // cores when batch < cores), each run serially — still bit-identical to the gemm() loop.
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        assert_batched_matches_loop(3, 200, 150, 180, 1.0, 0.0, par);
        assert_batched_matches_loop(2, 128, 96, 160, 0.75, 0.25, par);
    }
}

/// A few large, **DRAM-bound** elements (working set spills L2) select the sequential-internal
/// schedule on a multi-core host: each element runs with the parallel engine. That splits an
/// element across workers, so compare to a serial gemm() loop with a tolerance (the driver's
/// serial and parallel reductions agree to a tight bound) rather than bit-for-bit. Correct
/// whichever schedule the host's core count selects.
#[test]
fn batched_dram_bound_few_large_correct() {
    let (batch, m, k, n) = (3usize, 384usize, 384usize, 384usize); // ~1.7 MB/element > L2
    let a = packed(batch, m, k, 51);
    let b = packed(batch, k, n, 52);
    let c_init = packed(batch, m, n, 53);
    let (alpha, beta) = (1.25f32, -0.5f32);

    let mut c_ref = c_init.clone();
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        gemm(
            alpha,
            MatRef::from_col_major(&a[ao..ao + m * k], m, k),
            MatRef::from_col_major(&b[bo..bo + k * n], k, n),
            beta,
            MatMut::from_col_major(&mut c_ref[co..co + m * n], m, n),
            Parallelism::Serial,
        );
    }

    let mut c_bat = c_init.clone();
    gemm_batched(
        batch,
        alpha,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        beta,
        MatMut::new(&mut c_bat, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Rayon(0),
    );
    for (got, exp) in c_bat.iter().zip(&c_ref) {
        let tol = 1e-3 * got.abs().max(exp.abs()) + 1e-4;
        assert!(
            (got - exp).abs() <= tol,
            "DRAM-bound few-large batched diverged: {got} vs {exp}"
        );
    }
}

/// `gemm_batched` must not panic when called from **inside** a rayon worker: the batch-parallel
/// schedule runs its own `for_each` there, whose inline work executes on the calling worker. Its
/// workers pack through a *separate* per-thread pool from the one the outer `with_thread_pool`
/// holds, so the inline re-entry cannot hit a `BorrowMutError`.
#[cfg(all(feature = "parallel", not(miri)))]
#[test]
fn batched_from_inside_rayon_worker_does_not_panic() {
    use rayon::prelude::*;
    let (batch, m, k, n) = (256usize, 12, 48, 9);
    let a = packed(batch, m, k, 41);
    let b = packed(batch, k, n, 42);
    // Several outer tasks each invoke gemm_batched from a rayon worker, on a shape whose total
    // work selects the batch-parallel schedule.
    let sums: Vec<f32> = (0..8u32)
        .into_par_iter()
        .map(|_| {
            let mut c = vec![0.0f32; batch * m * n];
            gemm_batched(
                batch,
                1.0,
                MatRef::new(&a, m, k, 1, m as isize),
                (m * k) as isize,
                MatRef::new(&b, k, n, 1, k as isize),
                (k * n) as isize,
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize),
                (m * n) as isize,
                Parallelism::Rayon(0),
            );
            c.iter().sum()
        })
        .collect();
    // All outer tasks computed the same product, so their checksums agree.
    for s in &sums {
        assert_eq!(
            *s, sums[0],
            "nested gemm_batched produced inconsistent results"
        );
    }
}

#[test]
fn batched_serial_equals_parallel_bit_identical() {
    let (batch, m, k, n) = (300, 12, 64, 9);
    let a = packed(batch, m, k, 11);
    let b = packed(batch, k, n, 12);
    let run = |par| {
        let mut c = packed(batch, m, n, 13);
        gemm_batched(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.5,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            par,
        );
        c
    };
    assert_eq!(
        run(Parallelism::Serial),
        run(Parallelism::Rayon(0)),
        "batched serial must equal parallel bit-for-bit"
    );
}

#[test]
fn batched_broadcast_a_shared_across_elements() {
    // a_batch_stride = 0 shares one A across the batch; each element has its own B and C.
    let (batch, m, k, n) = (16, 8, 20, 6);
    let a = fill(m * k, 21); // one element's worth
    let b = packed(batch, k, n, 22);
    let c_init = packed(batch, m, n, 23);

    let mut c_ref = c_init.clone();
    for bi in 0..batch {
        let (bo, co) = (bi * k * n, bi * m * n);
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b[bo..bo + k * n], k, n),
            0.0,
            MatMut::from_col_major(&mut c_ref[co..co + m * n], m, n),
            Parallelism::Serial,
        );
    }

    let mut c_bat = c_init.clone();
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        0, // broadcast A
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c_bat, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Rayon(0),
    );
    assert_eq!(c_ref, c_bat, "broadcast-A batched != gemm() loop");
}

#[test]
fn batched_zero_batch_is_noop() {
    // A zero-length batch is a pure no-op even with mismatched / placeholder element shapes: the
    // views are never dereferenced, so validation (shape/bounds) must be skipped, not panic.
    let a = fill(4 * 3, 1);
    let b = fill(5 * 2, 2); // B.rows (5) != A.cols (3) on purpose
    let mut c = vec![7.0f32; 4 * 2];
    let before = c.clone();
    gemm_batched(
        0,
        1.0,
        MatRef::new(&a, 4, 3, 1, 4),
        12,
        MatRef::new(&b, 5, 2, 1, 5),
        10,
        0.0,
        MatMut::new(&mut c, 4, 2, 1, 4),
        8,
        Parallelism::Rayon(0),
    );
    assert_eq!(
        c, before,
        "batch=0 must be a no-op regardless of element shapes"
    );
}

#[test]
fn batched_reuses_workspace() {
    // The `_with` form threads a caller workspace through the serial/few-large paths.
    let (batch, m, k, n) = (4, 64, 64, 64);
    let a = packed(batch, m, k, 31);
    let b = packed(batch, k, n, 32);
    let mut c = packed(batch, m, n, 33);
    let mut ws = Workspace::new();
    gemm_batched_with(
        &mut ws,
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
    // A second call reuses the now-sized workspace with no reallocation (correctness is enough
    // here; the no-realloc property is covered by the workspace tests).
    gemm_batched_with(
        &mut ws,
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
}

// ---- validation: the batched API must reject invalid layouts ----

#[test]
#[should_panic(expected = "stay disjoint")]
fn batched_rejects_overlapping_c() {
    // C batch stride below the element extent (m*n) → element 1 overwrites element 0's tail.
    let (batch, m, n) = (2usize, 4usize, 4usize);
    let a = packed(batch, m, m, 1);
    let b = packed(batch, m, n, 2);
    let mut c = vec![0.0f32; 2 * m * n];
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, m, 1, m as isize),
        (m * m) as isize,
        MatRef::new(&b, m, n, 1, m as isize),
        (m * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n - 1) as isize, // < element extent m*n
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "aliases itself")]
fn batched_rejects_self_aliasing_c() {
    // A broadcast column stride (cs = 0) makes distinct C columns share memory.
    let (batch, m, n) = (2usize, 4usize, 4usize);
    let a = packed(batch, m, m, 1);
    let b = packed(batch, m, n, 2);
    let mut c = vec![0.0f32; batch * m];
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, m, 1, m as isize),
        (m * m) as isize,
        MatRef::new(&b, m, n, 1, m as isize),
        (m * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, 0), // cs = 0
        m as isize,
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "needs")]
fn batched_rejects_last_element_out_of_bounds() {
    // The slice fits batch-1 elements but not the last one's extent.
    let (batch, m, k, n) = (4usize, 8usize, 8usize, 8usize);
    let a = vec![0.0f32; batch * m * k - 1]; // one element short
    let b = packed(batch, k, n, 2);
    let mut c = packed(batch, m, n, 3);
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "non-negative")]
fn batched_rejects_negative_batch_stride() {
    let (batch, m, k, n) = (3usize, 4usize, 4usize, 4usize);
    let a = packed(batch, m, k, 1);
    let b = packed(batch, k, n, 2);
    let mut c = packed(batch, m, n, 3);
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        -((m * k) as isize), // negative
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "!= B.rows")]
fn batched_rejects_shape_mismatch() {
    let (batch, m, k, n) = (2usize, 4usize, 4usize, 4usize);
    let a = packed(batch, m, k, 1);
    let b = packed(batch, k, n, 2);
    let mut c = packed(batch, m, n, 3);
    gemm_batched(
        batch,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k + 1, n, 1, (k + 1) as isize), // B.rows != A.cols
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
}
