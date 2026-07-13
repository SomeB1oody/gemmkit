//! Batched GEMM vs naive gemm() loops.

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm, gemm_batched};

// ---- batched GEMM: perf_batched ----

/// One `perf_batched` row: `gemm_batched` (auto) vs the two honest baselines — a naive serial loop
/// of `gemm(Serial)` and a naive parallel loop of `gemm(Rayon(0))` (per-element internal
/// parallelism) — over `batch` contiguously-packed column-major `m×k · k×n` elements, as **total**
/// GFLOP/s (`2·batch·m·k·n`). There is no batched entry in the `gemm` crate / `matrixmultiply`, so
/// the naive loops are the only honest baselines. Passing `m·batch` to `measure` folds the batch
/// into its `2·m·k·n` count so the reported figure is whole-batch throughput.
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

/// Batched GEMM throughput: `gemm_batched` (auto) vs a naive serial / naive parallel loop of
/// single `gemm()` calls, as total GFLOP/s. The batch-parallel schedule assigns whole GEMMs to
/// workers (each element serial, cache-hot), so the win over the naive parallel loop is avoiding
/// its per-element fork/join. The few-but-large tail (batch < cores) uses at most `batch` workers.
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
    // Few but large: fewer elements than cores, so batch-parallel uses only `batch` workers.
    // A cache-resident element (256³) that scales poorly favors batch-parallelism; a bigger,
    // DRAM-touching element (512³) is where a per-element internal split would use more cores.
    println!("  few-but-large (batch < cores):");
    for &batch in &[4usize, 8] {
        bench_batched(batch, 256, 256, 256);
    }
    for &batch in &[2usize, 4] {
        bench_batched(batch, 512, 512, 512);
    }
}
