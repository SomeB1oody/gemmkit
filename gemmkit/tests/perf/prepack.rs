//! Prepacked-RHS/LHS reuse, gather-pack probe, shared-LHS gate sweep

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// Prepacked-RHS reuse

/// Per-call throughput of a reused prepacked B (`gemm_packed_b`) vs plain `gemm`
/// (which re-reads / re-packs B every call) for a fixed `(k, n)` B and a varying
/// `m` (the activation batch). `b_row_major` is the strided case: plain gemm reads
/// B with a large K-stride each call and, below `m > 2048`, never packs it, so the
/// contiguous prepacked panel should win per call. `colB` is the control. The win
/// is the per-call speedup (the one-time pack amortizes away over many calls)
fn bench_prepack(k: usize, n: usize, m: usize, b_row_major: bool, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let (brs, bcs) = if b_row_major {
        (n as isize, 1)
    } else {
        (1, k as isize)
    };
    let mut c = vec![0.0f32; m * n];

    let s_plain = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::new(&b, k, n, brs, bcs),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    let packed = gemmkit::prepack_rhs(MatRef::new(&b, k, n, brs, bcs));
    let s_packed = measure(m, k, n, || {
        gemmkit::gemm_packed_b(
            1.0,
            MatRef::from_col_major(&a, m, k),
            &packed,
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });

    let layout = if b_row_major { "rowB" } else { "colB" };
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  m={m:<5} k={k} n={n} {layout} {mode}  plain={:7.1} (±{:>2.0}%)  packed={:7.1} (±{:>2.0}%)  ({:.0}% of plain)",
        s_plain.median,
        s_plain.spread_pct(),
        s_packed.median,
        s_packed.spread_pct(),
        100.0 * s_packed.median / s_plain.median.max(1e-9)
    );
}

/// Pack-path probe: isolate the gather-pack cost. Row-major A packs via the strided
/// gather; col-major A at these sizes packs via the fast `copy_nonoverlapping`
/// contiguous path. Same FLOPs otherwise, so the row/col gap is an upper bound on
/// what a faster gather-pack could recover. Small `n` keeps A-packing unamortized
fn bench_pack_probe(m: usize, k: usize, n: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let row = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::new(&a, m, k, k as isize, 1),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    let col = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  m={m:<5} k={k} n={n:<4} {mode}  rowA(gather)={:7.1} (±{:>2.0}%)  colA(copy)={:7.1}  (gather {:.0}% of copy)",
        row.median,
        row.spread_pct(),
        col.median,
        100.0 * row.median / col.median.max(1e-9)
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_pack_probe() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nB3 probe — gather-pack overhead (rowA gather vs colA copy):");
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &(m, k, n) in &[
            (2048usize, 2048, 64),
            (2048, 2048, 128),
            (4096, 2048, 64),
            (2048, 2048, 256),
        ] {
            bench_pack_probe(m, k, n, par);
        }
    }
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_prepack() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nprepacked-RHS reuse — per-call GFLOP/s, plain gemm vs gemm_packed_b (k=n=1024):");
    for &brm in &[true, false] {
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            for &m in &[128usize, 512, 1024, 2048] {
                bench_prepack(1024, 1024, m, brm, par);
            }
        }
    }
}

/// Per-call throughput of a reused prepacked A (`gemm_packed_a`) vs plain `gemm`
/// (which re-packs A every call) for a fixed `(m, k)` A and varying `n`. The
/// packed-LHS path drives the product transposed, so C is row-major (its supported
/// orientation) and A plays the transposed RHS. `a_col_major` is the strided case:
/// after the transpose the driver reads A with a large K-stride (`= m`) and, below
/// the pack gate (transposed `m = n > 2048`), never packs it, so the contiguous
/// prepacked panel should win per call. Row-major A is the contiguous control
fn bench_prepack_lhs(m: usize, k: usize, n: usize, a_col_major: bool, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let (ars, acs) = if a_col_major {
        (1, m as isize)
    } else {
        (k as isize, 1)
    };
    // Row-major C: the supported orientation for the prepacked-LHS path
    let mut c = vec![0.0f32; m * n];

    let s_plain = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::new(&a, m, k, ars, acs),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_row_major(&mut c, m, n),
            par,
        );
    });
    let packed = gemmkit::prepack_lhs(MatRef::new(&a, m, k, ars, acs));
    let s_packed = measure(m, k, n, || {
        gemmkit::gemm_packed_a(
            1.0,
            &packed,
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_row_major(&mut c, m, n),
            par,
        );
    });

    let layout = if a_col_major { "colA" } else { "rowA" };
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  n={n:<5} m={m} k={k} {layout} {mode}  plain={:7.1} (±{:>2.0}%)  packed={:7.1} (±{:>2.0}%)  ({:.0}% of plain)",
        s_plain.median,
        s_plain.spread_pct(),
        s_packed.median,
        s_packed.spread_pct(),
        100.0 * s_packed.median / s_plain.median.max(1e-9)
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_prepack_lhs() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nprepacked-LHS reuse — per-call GFLOP/s, plain gemm vs gemm_packed_a (m=k=1024):");
    for &acm in &[true, false] {
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            // n past 2048 crosses the RHS-pack gate, where plain gemm re-packs the
            // (fixed) A every call: the case prepacking should win most
            for &n in &[128usize, 512, 1024, 2048, 4096, 6144] {
                bench_prepack_lhs(1024, 1024, n, acm, par);
            }
        }
    }
}

// Shared-LHS A-pack gate calibration

/// Force the shared-LHS A-pack gate **on vs off back-to-back in one process** (via
/// the runtime setter, so the same buffers/thread-pool are reused and machine drift
/// cancels) and report the parallel throughput of each. The gate only changes
/// behavior on the packed-A path: a row-major A (`rsa != 1`) always packs, so every
/// size exercises the pre-pass; a column-major A packs only once its K-walk stride
/// trips the TLB gate (large `m`), so its crossover sits higher. The `on % of off`
/// column is the signal: above 100% the shared pre-pass wins, below it regresses
fn bench_shared_lhs(s: usize, row_major_a: bool) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let (ars, acs) = if row_major_a {
        (k as isize, 1)
    } else {
        (1, m as isize)
    };
    let par = Parallelism::Rayon(0);

    let prev = gemmkit::tuning::shared_lhs_mnk();
    gemmkit::tuning::set_shared_lhs_mnk(1); // force the shared pre-pass on
    let on = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::new(&a, m, k, ars, acs),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    gemmkit::tuning::set_shared_lhs_mnk(usize::MAX - 1); // force it off
    let off = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::new(&a, m, k, ars, acs),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    gemmkit::tuning::set_shared_lhs_mnk(prev);

    let layout = if row_major_a { "rowA" } else { "colA" };
    println!(
        "  n={s:<5} {layout}  shared-on={:7.1} (±{:>2.0}%)  off={:7.1} (±{:>2.0}%)  (on {:.0}% of off)",
        on.median,
        on.spread_pct(),
        off.median,
        off.spread_pct(),
        100.0 * on.median / off.median.max(1e-9)
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_shared_lhs() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nshared-LHS A-pack gate sweep (parallel, f32 col-major C) — forced on vs off:");
    for &rma in &[false, true] {
        for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
            bench_shared_lhs(s, rma);
        }
    }
}
