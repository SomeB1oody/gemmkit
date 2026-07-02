//! Batched GEMM (`gemm_batched`) behavior: it must reproduce a loop of single `gemm()` calls
//! bit-for-bit (same route per element), stay serial==parallel bit-identical, and reject the
//! invalid batch layouts its new validation is responsible for.

use gemmkit::{
    BatchProblem, GemmProblem, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_batched,
    gemm_batched_ptr_unchecked, gemm_batched_slice, gemm_batched_unchecked, gemm_batched_with,
};

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

/// The raw `gemm_batched_unchecked` (pointers + strides, no checks) must equal a loop of single
/// `gemm(Serial)` calls bit-for-bit — the ndarray/FFI-facing entry, tested through its raw
/// signature. Contiguously-packed column-major elements.
#[test]
fn batched_unchecked_matches_gemm_loop() {
    let (batch, m, k, n) = (37usize, 16, 40, 5);
    let (alpha, beta) = (2.5f32, -0.5);
    let a = packed(batch, m, k, 1);
    let b = packed(batch, k, n, 2);
    let c_init = packed(batch, m, n, 3);

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

    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = c_init.clone();
        // SAFETY: contiguously-packed column-major elements — every element view is in bounds, the
        // C regions are disjoint (batch stride m*n == element extent) and don't alias A/B.
        unsafe {
            gemm_batched_unchecked(
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                1,
                m as isize,
                (m * k) as isize,
                b.as_ptr(),
                1,
                k as isize,
                (k * n) as isize,
                beta,
                c.as_mut_ptr(),
                1,
                m as isize,
                (m * n) as isize,
                par,
            );
        }
        assert_eq!(
            c_ref, c,
            "gemm_batched_unchecked != gemm() loop (par={par:?})"
        );
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

/// `gemm_batched` must not panic when called from **inside** a rayon worker. The batch-parallel
/// schedule blocks the calling worker in its own `for_each`, and rayon may work-steal *another*
/// nested `gemm_batched` onto that worker while it already holds the thread-local pool; the pool
/// accessor is re-entrancy-safe (hands out a fresh scratch that one time) so this can't
/// `BorrowMutError`. Enough outer tasks to queue past the core count and actually force the steal.
#[cfg(all(feature = "parallel", not(miri)))]
#[test]
fn batched_from_inside_rayon_worker_does_not_panic() {
    use rayon::prelude::*;
    let (batch, m, k, n) = (256usize, 12, 48, 9); // total work selects the batch-parallel schedule
    let a = packed(batch, m, k, 41);
    let b = packed(batch, k, n, 42);
    let sums: Vec<f32> = (0..256u32)
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
    // Every outer task computed the same product on the same batch-parallel schedule (each element
    // driver-serial on one worker), so the results — and thus the checksums — are bit-identical.
    for &s in &sums {
        assert_eq!(
            s, sums[0],
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

// ---- heterogeneous pointer-array batched: gemm_batched_slice / gemm_batched_ptr_unchecked ----

/// One heterogeneous element: `(m, k, n, alpha, beta, a, b, c_init)`, column-major.
type HeteroElem = (usize, usize, usize, f32, f32, Vec<f32>, Vec<f32>, Vec<f32>);

/// Three heterogeneous column-major elements (different shapes, alpha/beta).
fn hetero_case(seed: u64) -> [HeteroElem; 3] {
    let mk = |m, k, n, s| {
        (
            m,
            k,
            n,
            fill(m * k, s),
            fill(k * n, s + 1),
            fill(m * n, s + 2),
        )
    };
    let (m0, k0, n0, a0, b0, c0) = mk(5, 7, 3, seed);
    let (m1, k1, n1, a1, b1, c1) = mk(8, 4, 6, seed + 10);
    let (m2, k2, n2, a2, b2, c2) = mk(2, 9, 4, seed + 20);
    [
        (m0, k0, n0, 1.0, 0.0, a0, b0, c0),
        (m1, k1, n1, 2.5, -0.5, a1, b1, c1),
        (m2, k2, n2, -1.0, 1.0, a2, b2, c2),
    ]
}

/// The checked pointer-array form (`gemm_batched_slice`) must equal a loop of single `gemm(par)`
/// calls bit-for-bit — heterogeneous shapes, per-element alpha/beta, both Serial and Rayon.
#[test]
fn batched_slice_matches_loop() {
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let case = hetero_case(1);
        // Reference: one gemm() per element (C seeded from the shared init).
        let mut refs: Vec<Vec<f32>> = case.iter().map(|e| e.7.clone()).collect();
        for (e, cref) in case.iter().zip(refs.iter_mut()) {
            let (m, k, n, alpha, beta, a, b, _) = e;
            gemm(
                *alpha,
                MatRef::from_col_major(a, *m, *k),
                MatRef::from_col_major(b, *k, *n),
                *beta,
                MatMut::from_col_major(cref, *m, *n),
                par,
            );
        }
        // Batched: distinct MatMut per element (disjoint by construction).
        let mut c: Vec<Vec<f32>> = case.iter().map(|e| e.7.clone()).collect();
        {
            let mut probs: Vec<BatchProblem<'_, f32>> = c
                .iter_mut()
                .zip(case.iter())
                .map(|(ci, e)| {
                    let (m, k, n, alpha, beta, a, b, _) = e;
                    BatchProblem {
                        alpha: *alpha,
                        a: MatRef::from_col_major(a, *m, *k),
                        b: MatRef::from_col_major(b, *k, *n),
                        beta: *beta,
                        c: MatMut::from_col_major(ci, *m, *n),
                    }
                })
                .collect();
            gemm_batched_slice(&mut probs, par);
        }
        for (ci, cref) in c.iter().zip(&refs) {
            assert_eq!(ci, cref, "gemm_batched_slice != gemm() loop (par={par:?})");
        }
    }
}

/// The unchecked pointer-array form must equal a loop of single `gemm(par)` calls bit-for-bit.
#[test]
fn batched_ptr_unchecked_matches_loop() {
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let case = hetero_case(2);
        let mut refs: Vec<Vec<f32>> = case.iter().map(|e| e.7.clone()).collect();
        for (e, cref) in case.iter().zip(refs.iter_mut()) {
            let (m, k, n, alpha, beta, a, b, _) = e;
            gemm(
                *alpha,
                MatRef::from_col_major(a, *m, *k),
                MatRef::from_col_major(b, *k, *n),
                *beta,
                MatMut::from_col_major(cref, *m, *n),
                par,
            );
        }
        let mut c: Vec<Vec<f32>> = case.iter().map(|e| e.7.clone()).collect();
        let problems: Vec<GemmProblem<f32>> = c
            .iter_mut()
            .zip(case.iter())
            .map(|(ci, e)| {
                let (m, k, n, alpha, beta, a, b, _) = e;
                GemmProblem {
                    m: *m,
                    k: *k,
                    n: *n,
                    alpha: *alpha,
                    a: a.as_ptr(),
                    rsa: 1,
                    csa: *m as isize,
                    b: b.as_ptr(),
                    rsb: 1,
                    csb: *k as isize,
                    beta: *beta,
                    c: ci.as_mut_ptr(),
                    rsc: 1,
                    csc: *m as isize,
                }
            })
            .collect();
        // SAFETY: each element's pointers are valid for its shape; the C buffers are distinct Vecs
        // (pairwise disjoint) and don't alias the A/B inputs.
        unsafe { gemm_batched_ptr_unchecked(&problems, par) };
        for (ci, cref) in c.iter().zip(&refs) {
            assert_eq!(
                ci, cref,
                "gemm_batched_ptr_unchecked != gemm() loop (par={par:?})"
            );
        }
    }
}

/// The heterogeneous batch runs each element serially on one worker, so serial == parallel
/// bit-for-bit. Enough elements/work to actually split.
#[test]
fn batched_slice_serial_equals_parallel() {
    let run = |par| {
        // 40 elements, each 16×64×9 (routes through the horizontal kernel — element-serial).
        let (m, k, n) = (16usize, 64usize, 9usize);
        let a: Vec<Vec<f32>> = (0..40).map(|s| fill(m * k, s as u64 + 1)).collect();
        let b: Vec<Vec<f32>> = (0..40).map(|s| fill(k * n, s as u64 + 100)).collect();
        let mut c: Vec<Vec<f32>> = (0..40).map(|_| vec![0.0f32; m * n]).collect();
        {
            let mut probs: Vec<BatchProblem<'_, f32>> = c
                .iter_mut()
                .enumerate()
                .map(|(i, ci)| BatchProblem {
                    alpha: 1.0,
                    a: MatRef::from_col_major(&a[i], m, k),
                    b: MatRef::from_col_major(&b[i], k, n),
                    beta: 0.0,
                    c: MatMut::from_col_major(ci, m, n),
                })
                .collect();
            gemm_batched_slice(&mut probs, par);
        }
        c.concat()
    };
    assert_eq!(
        run(Parallelism::Serial),
        run(Parallelism::Rayon(0)),
        "heterogeneous batch serial must equal parallel bit-for-bit"
    );
}

#[test]
#[should_panic(expected = "!= B.rows")]
fn batched_slice_rejects_shape_mismatch() {
    let a = fill(4 * 3, 1);
    let b = fill(5 * 2, 2); // B.rows (5) != A.cols (3)
    let mut c = vec![0.0f32; 4 * 2];
    let mut probs = [BatchProblem {
        alpha: 1.0,
        a: MatRef::from_col_major(&a, 4, 3),
        b: MatRef::from_col_major(&b, 5, 2),
        beta: 0.0,
        c: MatMut::from_col_major(&mut c, 4, 2),
    }];
    gemm_batched_slice(&mut probs, Parallelism::Serial);
}
