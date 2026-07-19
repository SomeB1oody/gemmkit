//! `gemm_batched` throughput vs a naive loop of single `gemm` calls, over many small-to-medium GEMMs.
//! `#[ignore]`d (a benchmark, not a correctness gate); run with:
//!   cargo test -p gemmkit-faer --release --test perf_batched -- --ignored --nocapture
#![cfg(not(miri))]

use std::time::Instant;

use faer::{Mat, MatRef};
use gemmkit::Parallelism;

use gemmkit_faer::{gemm, gemm_batched};

/// Held for the duration of `perf_batched`, so no other test in this file that also locks it can
/// run at the same time and corrupt every GFLOP/s figure
static BENCH_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Column-major `Mat<f64>` fill from a `seed`-derived xorshift64 stream, values in `[-0.5, 0.5)`
fn rand_mat(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    Mat::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

const REPS: usize = 9;
const BATCH_SECS: f64 = 0.07;

/// Median GFLOP/s over `REPS` repetitions of an auto-calibrated iteration count (mirrors the core
/// crate's bench harness, which a test crate cannot import). `flops` is the whole call's total
/// float ops, e.g. `2*batch*m*k*n` for a batched GEMM
fn measure<F: FnMut()>(flops: f64, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    f();
    let one = t0.elapsed().as_secs_f64().max(1e-9);
    let iters = ((BATCH_SECS / one).ceil() as usize).clamp(1, 200_000);
    let mut g: Vec<f64> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let secs = t.elapsed().as_secs_f64() / iters as f64;
        g.push(flops / secs / 1e9);
    }
    g.sort_by(f64::total_cmp);
    g[REPS / 2]
}

/// Prints 1 result row: `gemm_batched` vs a naive loop of `gemm`, over `batch` independent `m x k`
/// by `k x n` products, as total GFLOP/s under both Serial and auto (`Rayon(0)`) parallelism
fn bench(batch: usize, m: usize, k: usize, n: usize) {
    let a: Vec<Mat<f64>> = (0..batch).map(|e| rand_mat(m, k, 1 + e as u64)).collect();
    let b: Vec<Mat<f64>> = (0..batch)
        .map(|e| rand_mat(k, n, 1000 + e as u64))
        .collect();
    let mut c: Vec<Mat<f64>> = (0..batch).map(|_| Mat::zeros(m, n)).collect();
    let ab: Vec<(MatRef<'_, f64>, MatRef<'_, f64>)> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x.as_dyn_stride(), y.as_dyn_stride()))
        .collect();
    let flops = 2.0 * batch as f64 * m as f64 * k as f64 * n as f64;

    let run = |par: Parallelism, c: &mut Vec<Mat<f64>>| {
        let batched = measure(flops, || {
            let mut cv: Vec<_> = c.iter_mut().map(|m| m.as_dyn_stride_mut()).collect();
            gemm_batched(1.0, &ab, 0.0, &mut cv, par);
        });
        let naive = measure(flops, || {
            for e in 0..batch {
                gemm(1.0, ab[e].0, ab[e].1, 0.0, c[e].as_dyn_stride_mut(), par);
            }
        });
        (batched, naive)
    };

    let (bs, ns) = run(Parallelism::Serial, &mut c);
    let (bp, np) = run(Parallelism::Rayon(0), &mut c);
    println!(
        "  b={batch:<5} {m}x{k}x{n:<5}  serial: batched={bs:7.1} naive={ns:7.1} ({:.2}x)   auto: batched={bp:7.1} naive={np:7.1} ({:.2}x)",
        bs / ns.max(1e-9),
        bp / np.max(1e-9),
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_batched() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nfaer batched (total GFLOP/s) - gemm_batched vs naive loop of gemm():");
    bench(256, 16, 16, 16);
    bench(64, 48, 48, 48);
    bench(8, 256, 256, 256);
}
