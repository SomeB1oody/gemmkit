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

// Prepack buffer setup cost (half-type dead zero-fill probe)

/// Median wall time of `reps` calls to `f`, in microseconds (after a short warmup).
/// For timing an operation whose cost is not a GFLOP count (here, the prepack buffer
/// setup), so the GFLOP/s-oriented `measure` does not apply
#[cfg(feature = "half")]
fn time_us<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    use std::time::Instant;
    for _ in 0..3 {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    samples[reps / 2]
}

/// Isolated cost of the prepack buffer setup. `prepack_rhs` allocates `ceil(n/nr)*nr*k_pad`
/// elements, then packs every one of them. For `f32` the zero-init specializes to
/// `alloc_zeroed` (no write pass), but the `half` types (`f16`/`bf16`) lack std's `IsZero`
/// specialization, so a plain `vec![ZERO; ..]` runs a dead `O(k*n)` write the pack immediately
/// overwrites. This times prepack alone (no compute) so that dead pass is visible; `f32` is the
/// control (its zero-init is already free, so its number must not move)
#[cfg(feature = "half")]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_prepack_alloc() {
    use half::{bf16, f16};
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let (k, n) = (4096usize, 4096usize);
    println!("\nprepack buffer setup cost — prepack_rhs alone (k=n=4096), median us/call:");

    let bf: Vec<bf16> = (0..k * n)
        .map(|i| bf16::from_f32(i as f32 * 0.001))
        .collect();
    let t_bf = time_us(21, || {
        let p = gemmkit::prepack_rhs(MatRef::from_col_major(&bf, k, n));
        core::hint::black_box(&p);
    });
    let hf: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(i as f32 * 0.001))
        .collect();
    let t_hf = time_us(21, || {
        let p = gemmkit::prepack_rhs(MatRef::from_col_major(&hf, k, n));
        core::hint::black_box(&p);
    });
    let f: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
    let t_f = time_us(21, || {
        let p = gemmkit::prepack_rhs(MatRef::from_col_major(&f, k, n));
        core::hint::black_box(&p);
    });
    println!("  bf16 {t_bf:8.1} us   f16 {t_hf:8.1} us   f32 (control) {t_f:8.1} us");
}

// Prepacked-i8-RHS reuse

/// Per-call throughput of a reused prepacked `i8` B (`gemm_i8_packed_b`) vs plain `gemm_i8`
/// (which re-packs B every call - for the VNNI `vpdpbusd` kernel the k-quad-interleaved RHS pack
/// is *mandatory* on every call, the quad layout can't be read in place) for a fixed `(k, n)` B
/// and a small activation batch `m`. At small `m` the `O(k*n)` pack dominates the `O(m*k*n)`
/// compute, so a reused prepacked panel should win per call (the one-time pack amortizes away over
/// a stream of activation batches)
#[cfg(all(feature = "int8", not(target_family = "wasm")))]
fn bench_prepack_i8(k: usize, n: usize, m: usize, par: Parallelism) {
    let a: Vec<i8> = (0..m * k).map(|i| (i % 17) as i8 - 8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 13) as i8 - 6).collect();
    let mut c = vec![0i32; m * n];

    let s_plain = measure(m, k, n, || {
        gemmkit::gemm_i8(
            1,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    let packed = gemmkit::prepack_rhs_i8(MatRef::from_col_major(&b, k, n));
    let s_packed = measure(m, k, n, || {
        gemmkit::gemm_i8_packed_b(
            1,
            MatRef::from_col_major(&a, m, k),
            &packed,
            0,
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
        "  m={m:<4} k={k} n={n} {mode}  plain={:7.1} (±{:>2.0}%)  packed={:7.1} (±{:>2.0}%)  ({:.0}% of plain)",
        s_plain.median,
        s_plain.spread_pct(),
        s_packed.median,
        s_packed.spread_pct(),
        100.0 * s_packed.median / s_plain.median.max(1e-9)
    );
}

#[cfg(all(feature = "int8", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_prepack_i8() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nprepacked-i8-RHS reuse — per-call GFLOP/s, plain gemm_i8 vs gemm_i8_packed_b (fixed B):"
    );
    for &(k, n) in &[(1024usize, 1024usize), (2048, 2048)] {
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            for &m in &[8usize, 64] {
                bench_prepack_i8(k, n, m, par);
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
