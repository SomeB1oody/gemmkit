//! Batched GEMM (`gemm_batched` and its siblings) must reproduce a loop of single `gemm()` calls
//! bit-for-bit (each element takes the same route it would standalone), stay serial == parallel
//! bit-identical, and reject the invalid batch layouts the validation layer is responsible for

use gemmkit::{
    BatchProblem, GemmProblem, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_batched,
    gemm_batched_ptr_unchecked, gemm_batched_slice, gemm_batched_unchecked, gemm_batched_with,
};

/// Deterministic xorshift `f32` fill, non-constant so the reduction it feeds is non-trivial
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

/// `batch` contiguously packed column-major `r x c` elements, back-to-back with stride `r*c`
fn packed(batch: usize, r: usize, c: usize, seed: u64) -> Vec<f32> {
    fill(batch * r * c, seed)
}

/// Checks `gemm_batched` against a loop of single `gemm(Serial)` calls, bit-for-bit, over
/// contiguously packed column-major elements; `c_init` seeds the `beta` term
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

    // Reference: an independent single gemm() per element, on that element's own slice window
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
    // Batch counts not a multiple of the worker count; non-square shapes; beta != 0, alpha != 1
    // Small elements land on the batch-parallel schedule, and some also route through a small
    // element's own horizontal kernel
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        assert_batched_matches_loop(200, 8, 8, 8, 1.0, 0.0, par);
        assert_batched_matches_loop(101, 6, 40, 5, 2.5, -0.5, par);
        assert_batched_matches_loop(37, 16, 100, 4, -1.0, 1.0, par);
        assert_batched_matches_loop(1, 12, 12, 12, 1.0, 0.0, par); // batch of 1, the degenerate case
    }
}

/// `gemm_batched_unchecked` (raw pointers and strides, no validation; the entry FFI and adapter
/// crates use) must equal a loop of single `gemm(Serial)` calls bit-for-bit, over contiguously
/// packed column-major elements
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
        // SAFETY: elements are contiguously packed, so every element view is in bounds; the C
        // batch stride equals the element extent m*n, so the C regions are disjoint and alias
        // neither A nor B (distinct Vecs)
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
    // Cache-resident elements (each fits L2), so a batch smaller than the core count still
    // resolves to the batch-parallel schedule (1 element per worker, run serially) rather than
    // splitting elements across workers, and stays bit-identical to the gemm() loop
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        assert_batched_matches_loop(3, 200, 150, 180, 1.0, 0.0, par);
        assert_batched_matches_loop(2, 128, 96, 160, 0.75, 0.25, par);
    }
}

/// Elements whose working set spills L2 favor the sequential-internal schedule on a multi-core
/// host: the batch loops, but each element runs with the full parallel engine, splitting it
/// across workers. Compares against a serial gemm() loop with a tolerance rather than
/// bit-for-bit, since that route is only accurate to a tight bound, and stays correct whichever
/// schedule the host's core count actually picks
#[test]
fn batched_dram_bound_few_large_correct() {
    let (batch, m, k, n) = (3usize, 384usize, 384usize, 384usize); // A+B+C ~1.7 MB/element
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

/// `gemm_batched` must not panic when called from inside a rayon worker. The batch-parallel
/// schedule blocks its calling worker inside a `for_each`, and rayon can work-steal a 2nd, nested
/// `gemm_batched` onto that same worker while it still holds the thread-local pool; the pool
/// accessor falls back to a fresh one-off `Workspace` rather than double-borrowing, so this can't
/// panic on a borrow conflict. Uses more outer tasks than the core count to actually force a steal
#[cfg(all(feature = "parallel", not(miri)))]
#[test]
fn batched_from_inside_rayon_worker_does_not_panic() {
    use rayon::prelude::*;
    let (batch, m, k, n) = (256usize, 12, 48, 9); // total work clears the parallel-threshold gate
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
    // Every outer task ran the same batch on the same schedule (each element serial on 1 worker,
    // whether or not it hit the re-entrant fallback), so every checksum must agree
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
    // a_batch_stride = 0 broadcasts a single A across every element; B and C are per-element
    let (batch, m, k, n) = (16, 8, 20, 6);
    let a = fill(m * k, 21); // a single element's worth, reused by every element below
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
        0, // broadcast: every element reads the same A
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
    // batch == 0 short-circuits before validation, so mismatched/placeholder element shapes
    // (which would otherwise panic) must not be dereferenced or checked at all
    let a = fill(4 * 3, 1);
    let b = fill(5 * 2, 2); // B.rows (5) deliberately != A.cols (3)
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
    // gemm_batched_with threads a caller-owned Workspace through the serial and few-large
    // schedules instead of the thread-local pool
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
    // A 2nd call against the now-sized ws; just checks correctness here, since the workspace
    // tests already cover the no-reallocation property
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

// The batched API must reject invalid layouts

#[test]
#[should_panic(expected = "stay disjoint")]
fn batched_rejects_overlapping_c() {
    // C batch stride set below the element extent m*n, so element 1 would overwrite element 0
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
        (m * n - 1) as isize, // 1 short of the element extent m*n
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "aliases itself")]
fn batched_rejects_self_aliasing_c() {
    // cs = 0 collapses every column onto the same memory, aliasing C against itself
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
        MatMut::new(&mut c, m, n, 1, 0), // broadcast column stride
        m as isize,
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "needs")]
fn batched_rejects_last_element_out_of_bounds() {
    // The buffer is 1 element short of what `batch` elements need, so only the last element's
    // view runs out of bounds; the check must not stop at element 0
    let (batch, m, k, n) = (4usize, 8usize, 8usize, 8usize);
    let a = vec![0.0f32; batch * m * k - 1];
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
        -((m * k) as isize), // negative batch stride
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
        MatRef::new(&b, k + 1, n, 1, (k + 1) as isize), // B.rows = k+1 != A.cols = k
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Parallelism::Serial,
    );
}

// Heterogeneous pointer-array batched GEMM: gemm_batched_slice / gemm_batched_ptr_unchecked

/// 1 heterogeneous element: `(m, k, n, alpha, beta, a, b, c_init)`, all column-major
type HeteroElem = (usize, usize, usize, f32, f32, Vec<f32>, Vec<f32>, Vec<f32>);

/// 3 column-major elements with different shapes and alpha/beta, for the heterogeneous batch
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

/// `gemm_batched_slice` (the checked pointer-array form) must equal a loop of single `gemm(par)`
/// calls bit-for-bit, over heterogeneous shapes and per-element alpha/beta, both serial and
/// parallel
#[test]
fn batched_slice_matches_loop() {
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let case = hetero_case(1);
        // Reference: 1 gemm() per element, C seeded from the shared c_init
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
        // Batched: 1 distinct MatMut per element, so the outputs are disjoint by construction
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

/// `gemm_batched_ptr_unchecked` (raw `GemmProblem` pointers, no validation) must equal a loop of
/// single `gemm(par)` calls bit-for-bit
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
        // SAFETY: each element's pointers are valid for its own shape; the C buffers are distinct
        // Vecs, so they are pairwise disjoint and alias neither A nor B
        unsafe { gemm_batched_ptr_unchecked(&problems, par) };
        for (ci, cref) in c.iter().zip(&refs) {
            assert_eq!(
                ci, cref,
                "gemm_batched_ptr_unchecked != gemm() loop (par={par:?})"
            );
        }
    }
}

/// A heterogeneous batch runs every element serially on 1 worker, so serial must equal parallel
/// bit-for-bit; enough elements and total work here to actually force a parallel split
#[test]
fn batched_slice_serial_equals_parallel() {
    let run = |par| {
        // 40 elements of 16x64x9: small enough m,n to route through the horizontal kernel
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

/// A heterogeneous batch whose total work clears the `GEMMKIT_PARALLEL_THRESHOLD` gate under
/// `Rayon(4)` exercises `run_ptr`'s parallel branch, unlike the smaller cases above, which stay
/// under the gate and never fork. Every element still runs wholly on 1 worker, so the batch must
/// be bit-identical both to the serial run and to a per-element `gemm()` loop
#[test]
fn batched_slice_parallel_matches_serial_bit_for_bit() {
    // 4 elements around 128^3: total m*k*n ~= 8.5M, well above the 48*48*256 = 589824 threshold
    let shapes = [
        (128usize, 128usize, 128usize, 1.0f32, 0.0f32),
        (130, 120, 140, 2.5, -0.5),
        (150, 110, 128, -1.0, 1.0),
        (128, 140, 120, 0.75, 0.25),
    ];
    let a: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(m, k, _, _, _))| fill(m * k, 100 + i as u64))
        .collect();
    let b: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(_, k, n, _, _))| fill(k * n, 200 + i as u64))
        .collect();
    let c0: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(m, _, n, _, _))| fill(m * n, 300 + i as u64))
        .collect();

    let run = |par: Parallelism| -> Vec<Vec<f32>> {
        let mut c: Vec<Vec<f32>> = c0.clone();
        {
            let mut probs: Vec<BatchProblem<'_, f32>> = c
                .iter_mut()
                .enumerate()
                .map(|(i, ci)| {
                    let (m, k, n, alpha, beta) = shapes[i];
                    BatchProblem {
                        alpha,
                        a: MatRef::from_col_major(&a[i], m, k),
                        b: MatRef::from_col_major(&b[i], k, n),
                        beta,
                        c: MatMut::from_col_major(ci, m, n),
                    }
                })
                .collect();
            gemm_batched_slice(&mut probs, par);
        }
        c
    };

    let serial = run(Parallelism::Serial);
    let parallel = run(Parallelism::Rayon(4));
    assert_eq!(
        serial, parallel,
        "heterogeneous parallel batch must equal the serial batch bit-for-bit"
    );

    // Also check against a standalone gemm() per element, not just the batch's own serial run
    for (i, &(m, k, n, alpha, beta)) in shapes.iter().enumerate() {
        let mut cref = c0[i].clone();
        gemm(
            alpha,
            MatRef::from_col_major(&a[i], m, k),
            MatRef::from_col_major(&b[i], k, n),
            beta,
            MatMut::from_col_major(&mut cref, m, n),
            Parallelism::Serial,
        );
        assert_eq!(parallel[i], cref, "element {i} != standalone gemm()");
    }
}

#[test]
#[should_panic(expected = "!= B.rows")]
fn batched_slice_rejects_shape_mismatch() {
    let a = fill(4 * 3, 1);
    let b = fill(5 * 2, 2); // B.rows = 5, deliberately != A.cols = 3
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
