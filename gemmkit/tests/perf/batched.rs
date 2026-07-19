//! `gemm_batched` throughput vs a naive loop of individual `gemm()` calls, serial and
//! parallel, across a size/batch grid plus a dedicated "few but large" tail

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm, gemm_batched};

// perf_batched: gemm_batched(auto) vs the naive-loop baselines

/// 1 `perf_batched` row: `gemm_batched` (auto worker count) against 2 baselines, a serial
/// loop of `gemm(Serial)` calls and a parallel loop of `gemm(Rayon(0))` calls (each call
/// internally re-parallelized), over `batch` contiguously packed column-major `m x k` /
/// `k x n` elements. There is no batched entry point in the `gemm` crate or
/// `matrixmultiply`, so these 2 loops are the only baselines available. All 3 arms are
/// reported as **total** GFLOP/s (`2 * batch * m * k * n`); passing `m * batch` as the `m`
/// argument to `measure` is what folds the batch count into its `2*m*k*n` formula
fn bench_batched(batch: usize, m: usize, k: usize, n: usize) {
    let a = fill(batch * m * k, 1);
    let b = fill(batch * k * n, 2);
    let mut c = vec![0.0f32; batch * m * n];

    let batched = measure(m * batch, k, n, || {
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
    });
    let mut naive = |par| {
        measure(m * batch, k, n, || {
            for bi in 0..batch {
                let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
                gemm(
                    1.0,
                    MatRef::from_col_major(&a[ao..ao + m * k], m, k),
                    MatRef::from_col_major(&b[bo..bo + k * n], k, n),
                    0.0,
                    MatMut::from_col_major(&mut c[co..co + m * n], m, n),
                    par,
                );
            }
        })
    };
    let loop_ser = naive(Parallelism::Serial);
    let loop_par = naive(Parallelism::Rayon(0));
    println!(
        "  b={batch:<6} {m}×{k}×{n:<5}  batched={:8.1} (±{:>2.0}%)  loop_ser={:8.1} ({:.2}×)  loop_par={:8.1} ({:.2}×)",
        batched.median,
        batched.spread_pct(),
        loop_ser.median,
        batched.median / loop_ser.median.max(1e-9),
        loop_par.median,
        batched.median / loop_par.median.max(1e-9),
    );
}

/// Batched GEMM throughput across a size/batch grid, plus a "few but large" tail where the
/// batch count is smaller than the core count. `gemm_batched`'s auto schedule hands whole
/// elements to workers (each element runs serially, cache-hot) instead of splitting 1
/// element's work across every worker the way the naive parallel loop does per call, so its
/// win there is the fork/join it avoids paying once per element. Once the batch is smaller
/// than the core count, the schedule instead picks between running 1 element per worker
/// (idle cores left over) and splitting a single element's own work across the whole
/// machine in turn, based on whether the element's footprint fits a worker's share of
/// cache: the "few but large" sizes below are chosen to land on both sides of that split
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_batched() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nbatched GEMM (total GFLOP/s) — gemm_batched(auto) vs naive serial/parallel gemm() loops:"
    );
    for &(m, k, n) in &[
        (4usize, 4usize, 4usize),
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 64),
        (4, 4, 1024),
    ] {
        for &batch in &[64usize, 1024, 16384] {
            bench_batched(batch, m, k, n);
        }
    }
    // batch < cores: 256^3 stays cache-resident per element and is left 1-per-worker;
    // 512^3 is large enough that splitting its own work across the machine wins instead
    println!("  few-but-large (batch < cores):");
    for &batch in &[4usize, 8] {
        bench_batched(batch, 256, 256, 256);
    }
    for &batch in &[2usize, 4] {
        bench_batched(batch, 512, 512, 512);
    }
}
