//! Prepack-and-reuse throughput (RHS and LHS), the isolated cost of building a prepacked
//! buffer, and a forced on/off sweep of the shared-LHS A-pack gate

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// Prepacked-RHS reuse: perf_prepack

/// Per-call throughput of a reused prepacked B ([`gemmkit::gemm_packed_b`], packed once
/// outside the timed closure) against plain [`gemm`] (which re-reads, and past
/// `rhs_pack_threshold`, re-packs B on every call), for a fixed `(k, n)` B and varying
/// activation batch `m`. `b_row_major` selects the strided case: row-major B has a
/// non-unit depth stride, and for every `m` swept here (up to 2048, at or below the
/// default 2048 packing threshold) plain `gemm` never crosses that threshold and so never
/// packs it either, leaving it to re-read that stride on every call, which the prepacked
/// contiguous panel should beat. `colB` is the already-contiguous control. Either way the
/// reported number is the per-call speedup; the pack's own one-time cost is not counted
/// against either arm and only amortizes across many calls the loop here does not model
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

// Prepack buffer setup cost (isolated, no compute)

/// Median wall time of `reps` calls to `f`, in microseconds, after a short warmup. For timing
/// an operation that has no natural flop count (here, building a prepacked buffer), where the
/// GFLOP/s-oriented [`measure`] does not apply
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

/// Isolated cost of `prepack_rhs` alone, no compute, for `bf16`/`f16` against the `f32`
/// control. The pack buffer is sized with `Vec::with_capacity` + `set_len` rather than a
/// zero-initializing allocation, specifically so the `half` types are not left paying a dead
/// `O(k*n)` write before the pack loop below overwrites every slot anyway; this is the
/// regression guard for that choice. `f32`'s number should never move (its own path was
/// already allocation-light), and `bf16`/`f16` should track it rather than showing any
/// leftover zero-fill tax
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

// Prepacked-i8-RHS reuse: perf_prepack_i8

/// Per-call throughput of a reused prepacked `i8` B ([`gemmkit::gemm_i8_packed_b`]) against
/// plain [`gemmkit::gemm_i8`] (which re-packs B on every call), for a fixed `(k, n)` B and a
/// small activation batch `m`. Plain `gemm_i8` has no unpacked-read option to fall back to
/// here: whichever kernel this box auto-selects packs its RHS unconditionally (the VNNI
/// `vpdpbusd` kernel's k-quad-interleaved layout in particular cannot be read in place at
/// all), so every plain call pays a fresh `O(k*n)` pack. At the small `m` values below that
/// pack cost dominates the `O(m*k*n)` compute, which is exactly where reusing 1 prepacked
/// panel across a stream of activation batches should win per call
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

/// Per-call throughput of a reused prepacked A ([`gemmkit::gemm_packed_a`]) against plain
/// [`gemm`] (which re-packs A on every call), for a fixed `(m, k)` A and varying `n`. The
/// packed-LHS path always drives the transposed product internally, so `C` must be row-major
/// (`gemm_packed_a`'s supported orientation) with `A` playing the role of the transposed
/// product's RHS. `a_col_major` is the strided case: with column-major `A`, the transposed
/// product's depth-walk stride is `m`, and below the RHS-pack gate (here, once the swept `n`
/// exceeds 2048) plain `gemm` leaves that stride unpacked, so the prepacked contiguous panel
/// should win most clearly right past that crossover. Row-major `A` is the already-contiguous
/// control
fn bench_prepack_lhs(m: usize, k: usize, n: usize, a_col_major: bool, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let (ars, acs) = if a_col_major {
        (1, m as isize)
    } else {
        (k as isize, 1)
    };
    // gemm_packed_a requires row-major-ish C: the orientation its transposed-product path supports
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

// Shared-LHS A-pack gate calibration: perf_shared_lhs

/// Forces the shared-LHS A-pack gate on and then off, back-to-back in 1 process (via the
/// runtime setter, so both runs reuse the same buffers/thread pool and machine drift
/// cancels), and reports the parallel throughput of each. The gate only matters on the
/// packed-A path: row-major `A` (non-unit row stride) always packs regardless of the gate, so
/// every size here exercises the shared pre-pass either way; column-major `A` packs only once
/// its depth-walk stride (`= m` elements, growing with `s`) trips the separate TLB-driven
/// `lhs_pack_stride` gate, so its crossover sits at a larger `s`. The "on % of off" column is
/// the signal: above 100% the shared pre-pass is winning, below it is a regression
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
