//! nalgebra batched throughput: `gemm_batched` (1 call over a slice of per-element matrix triples)
//! vs the honest baseline of a naive loop of single adapter `gemm` calls, for many small GEMMs.
//! `#[ignore]` benchmark (not a correctness gate); run with:
//!   cargo test -p gemmkit-nalgebra --release --test perf_batched -- --ignored --nocapture
#![cfg(not(miri))]

use std::time::Instant;

use gemmkit::Parallelism;
use nalgebra::DMatrix;

use gemmkit_nalgebra::{gemm, gemm_batched};

/// Serializes the core-saturating bench so the default multi-threaded harness cannot run it
/// concurrently with itself (which would make every GFLOP/s figure meaningless)
static BENCH_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Deterministic column-major `DMatrix<f64>` fill (xorshift), values in `[-0.5, 0.5)`
fn rand_mat(r: usize, c: usize, seed: u64) -> DMatrix<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    DMatrix::from_fn(r, c, |_, _| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

const REPS: usize = 9;
const BATCH_SECS: f64 = 0.07;

/// Robust median GFLOP/s over `REPS` auto-calibrated batches (copied from the core bench harness,
/// which is not importable across crates). `flops` is the whole-batch `2*batch*m*k*n`
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

/// 1 row: `gemm_batched` vs a naive loop of `gemm`, over `batch` `mxk * kxn` products, for both
/// Serial and auto (`Rayon(0)`) parallelism, as total GFLOP/s
fn bench(batch: usize, m: usize, k: usize, n: usize) {
    let a: Vec<DMatrix<f64>> = (0..batch).map(|e| rand_mat(m, k, 1 + e as u64)).collect();
    let b: Vec<DMatrix<f64>> = (0..batch)
        .map(|e| rand_mat(k, n, 1000 + e as u64))
        .collect();
    let mut c: Vec<DMatrix<f64>> = (0..batch).map(|_| DMatrix::zeros(m, n)).collect();
    let ab: Vec<(&DMatrix<f64>, &DMatrix<f64>)> = a.iter().zip(&b).collect();
    let flops = 2.0 * batch as f64 * m as f64 * k as f64 * n as f64;

    let run = |par: Parallelism, c: &mut Vec<DMatrix<f64>>| {
        let batched = measure(flops, || gemm_batched(1.0, &ab, 0.0, c, par));
        let naive = measure(flops, || {
            for e in 0..batch {
                gemm(1.0, &a[e], &b[e], 0.0, &mut c[e], par);
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
    println!("\nnalgebra batched (total GFLOP/s) - gemm_batched vs naive loop of gemm():");
    bench(256, 16, 16, 16);
    bench(64, 48, 48, 48);
    bench(8, 256, 256, 256);
}
