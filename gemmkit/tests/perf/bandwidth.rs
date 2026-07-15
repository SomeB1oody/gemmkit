//! Bandwidth-bound shapes: STREAM ceilings, gemv, gevv, small-k, gemv scaling

use crate::harness::{BENCH_GUARD, Stat, fill, measure_gbps};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// Bandwidth-bound shapes: gemv, gevv, and the STREAM ceiling they are judged against
//
// gemv (matrix*vector), gevv (rank-k / outer product), and skinny/low-k GEMM have
// arithmetic intensity ~= O(1): each input byte feeds only a few flops, so the ceiling
// is *memory bandwidth*, not compute. The metric is therefore achieved GB/s as a
// fraction of what the machine's DRAM can sustain, not GFLOP/s. STREAM Triad is that
// ceiling: a single-core triad for the serial arm, an aggregate multi-core triad for
// the parallel arm (one core's bandwidth is far below aggregate DRAM bandwidth, so the
// serial ceiling would be the wrong yardstick for a threaded run)

/// STREAM array length in f32 elements: 64 Mi (256 MiB), much larger than any
/// last-level cache, so the kernels stream from DRAM and cannot be served from cache
const STREAM_LEN: usize = 64 * 1024 * 1024;

/// Single-core STREAM Triad `a[i] = b[i] + alpha*c[i]` in GB/s (3*N*4 B moved). The scalar
/// bandwidth ceiling the *serial* gemv/gevv arms are measured against. `black_box`
/// around the output keeps the optimizer from eliding the streaming loop
fn stream_triad_serial() -> Stat {
    let n = STREAM_LEN;
    let b = fill(n, 1);
    let c = fill(n, 2);
    let mut a = vec![0.0f32; n];
    let alpha = 1.5f32;
    measure_gbps(3 * n * 4, || {
        for i in 0..n {
            a[i] = b[i] + alpha * c[i];
        }
        std::hint::black_box(a.as_ptr());
    })
}

/// Single-core STREAM Copy `dst[i] = src[i]` in GB/s (2*N*4 B moved): the read+write
/// bandwidth reference next to the triad
fn stream_copy_serial() -> Stat {
    let n = STREAM_LEN;
    let src = fill(n, 3);
    let mut dst = vec![0.0f32; n];
    measure_gbps(2 * n * 4, || {
        dst.copy_from_slice(&src);
        std::hint::black_box(dst.as_ptr());
    })
}

/// Aggregate STREAM Triad across `threads` `std::thread::scope` chunks (rayon is an
/// optional dep, not a dev-dep, so it can't be used here). This is the fair ceiling for
/// the *parallel* gemv/gevv arms: a single core saturates only a fraction of DRAM
/// bandwidth, so the whole-machine triad is what a threaded output-partitioned gemv is
/// really racing. The array is large enough (256 MiB) that per-call thread-spawn cost is
/// a small fraction of the ~10 ms streaming time
fn stream_triad_parallel(threads: usize) -> Stat {
    let n = STREAM_LEN;
    let b = fill(n, 1);
    let c = fill(n, 2);
    let mut a = vec![0.0f32; n];
    let alpha = 1.5f32;
    let threads = threads.max(1);
    let chunk = n.div_ceil(threads);
    measure_gbps(3 * n * 4, || {
        std::thread::scope(|s| {
            for (ai, (bi, ci)) in a
                .chunks_mut(chunk)
                .zip(b.chunks(chunk).zip(c.chunks(chunk)))
            {
                s.spawn(move || {
                    for i in 0..ai.len() {
                        ai[i] = bi[i] + alpha * ci[i];
                    }
                    std::hint::black_box(ai.as_ptr());
                });
            }
        });
        std::hint::black_box(a.as_ptr());
    })
}

/// The best aggregate Triad bandwidth over a few thread counts, with the thread count
/// that reached it. DRAM bandwidth typically saturates well below the logical core count
/// and can even regress past it (memory-controller contention), so the *peak* (not the
/// all-cores figure) is the honest ceiling for the parallel arm
fn stream_triad_parallel_peak(avail: usize) -> (usize, Stat) {
    let mut best: Option<(usize, Stat)> = None;
    for &t in &[2usize, 4, 8, 16, 32] {
        if t > avail {
            break;
        }
        let s = stream_triad_parallel(t);
        if best
            .as_ref()
            .is_none_or(|(_, prev): &(usize, Stat)| s.median > prev.median)
        {
            best = Some((t, s));
        }
    }
    best.unwrap_or_else(|| (1, stream_triad_serial()))
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_stream() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nSTREAM bandwidth ceiling (f32, {} MiB arrays):",
        STREAM_LEN * 4 / (1024 * 1024)
    );
    let copy = stream_copy_serial();
    let triad = stream_triad_serial();
    println!(
        "  serial   copy ={:7.1} GB/s (±{:>2.0}%)   triad={:7.1} GB/s (±{:>2.0}%)",
        copy.median,
        copy.spread_pct(),
        triad.median,
        triad.spread_pct()
    );
    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    for &t in &[2usize, 4, 8, 16, 32] {
        if t > avail {
            break;
        }
        let s = stream_triad_parallel(t);
        println!(
            "  {t:3} thr triad={:7.1} GB/s (±{:>2.0}%)",
            s.median,
            s.spread_pct()
        );
    }
}

/// External-library GB/s for a column-major f32 `C(mxn) = A(mxk)*B(kxn)` (`beta = 0`): the
/// same-shape baseline the gemv/gevv rows compare against: the `gemm` crate, plus (serial only)
/// `matrixmultiply`. Native-only (both are wasm-excluded dev-deps). Returns `(gemm, mm)`
#[cfg(not(target_family = "wasm"))]
fn extern_baselines(
    m: usize,
    k: usize,
    n: usize,
    bytes: usize,
    par: Parallelism,
) -> (f64, Option<f64>) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let gpar = if matches!(par, Parallelism::Serial) {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(0)
    };
    let g = measure_gbps(bytes, || unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            m as isize,
            1,
            false,
            a.as_ptr(),
            m as isize,
            1,
            b.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gpar,
        );
    });
    // matrixmultiply is single-threaded as used here, so only compare it on the serial arm
    let mm = matches!(par, Parallelism::Serial).then(|| {
        measure_gbps(bytes, || unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                1,
                m as isize,
            );
        })
        .median
    });
    (g.median, mm)
}

/// Format the external-baseline tail (`gemm=... (kit ...x)  mm=... (kit ...x)`) for a `kit`-GB/s
/// gemmkit result on the given shape. Empty on wasm (no external crate there)
#[allow(unused_variables)]
fn baseline_tail(m: usize, k: usize, n: usize, bytes: usize, par: Parallelism, kit: f64) -> String {
    #[cfg(not(target_family = "wasm"))]
    {
        let (g, mm) = extern_baselines(m, k, n, bytes, par);
        let mm_s = mm
            .map(|v| format!("  mm={v:6.1} (kit {:.2}×)", kit / v.max(1e-9)))
            .unwrap_or_default();
        format!("  gemm={g:6.1} (kit {:.2}×){mm_s}", kit / g.max(1e-9))
    }
    #[cfg(target_family = "wasm")]
    {
        String::new()
    }
}

/// gemv `C(mx1) = A(mxk)*x` through the public [`gemm`], reported as GB/s of the minimum
/// traffic `(m*k + k + m)*4` (A read once, x once, C written once) against the STREAM
/// `ceiling`, plus the `gemm`-crate / `matrixmultiply` GB/s on the same shape (`kit x` is
/// gemmkit's speedup over each). `k` spans fits-L2 -> DRAM; column-major A/C hit the axpy
/// form of the gemv path
fn bench_gemv(m: usize, k: usize, par: Parallelism, ceiling: f64) {
    let a = fill(m * k, 1);
    let x = fill(k, 2);
    let mut c = vec![0.0f32; m];
    let bytes = (m * k + k + m) * 4;
    let st = measure_gbps(bytes, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&x, k, 1),
            0.0,
            MatMut::from_col_major(&mut c, m, 1),
            par,
        );
    });
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    let tail = baseline_tail(m, k, 1, bytes, par, st.median);
    println!(
        "  m={m:<8} k={k:<5} {mode}  kit={:7.1} GB/s (±{:>2.0}%)  {:3.0}% ceil{tail}",
        st.median,
        st.spread_pct(),
        100.0 * st.median / ceiling.max(1e-9),
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gemv() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\ngemv (C[m×1] = A[m×k]·x) — GB/s vs STREAM Triad ceiling:");
    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let ser_ceiling = stream_triad_serial().median;
    let (peak_thr, par_ceiling) = stream_triad_parallel_peak(avail);
    let par_ceiling = par_ceiling.median;
    println!(
        "  serial ceiling {ser_ceiling:.1} GB/s;  parallel ceiling {par_ceiling:.1} GB/s @ {peak_thr} thr"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let ceiling = if matches!(par, Parallelism::Serial) {
            ser_ceiling
        } else {
            par_ceiling
        };
        // Fits-L2 -> DRAM sweep: `out` (m*4 B) stays within L2 here, so the axpy form's
        // per-column re-read of `out` is cache-cheap and A's DRAM read dominates
        for &k in &[64usize, 256, 1024] {
            for &m in &[1024usize, 8192, 65536] {
                bench_gemv(m, k, par, ceiling);
            }
        }
        // `out` spills-cache sweep: at these `m`, `out` spills L2 (4 MiB) through past L3
        // (64 MiB), so the axpy form's `k` re-reads of `out` become real DRAM traffic: the
        // regime output register-blocking is meant to fix. `k` is kept small so A
        // stays <= 512 MiB
        for &(m, k) in &[
            (1_048_576usize, 64usize),
            (4_194_304, 16),
            (8_388_608, 8),
            (16_777_216, 8),
            (16_777_216, 16), // out spills L3 with moderate k: probes A-stream prefetch thrash
        ] {
            bench_gemv(m, k, par, ceiling);
        }
    }
}

/// Investigation: for a gemv shape, measure the dedicated gemv special path vs the general
/// driver (reached by disabling the gemv path) vs the `gemm` crate. Confirms the special
/// path is the right choice (the driver, which packs A into micropanels for a compute-bound
/// kernel, is far slower on a bandwidth-bound gemv) and tracks the remaining gap to `gemm`
#[cfg(not(target_family = "wasm"))]
fn bench_gemv_paths(m: usize, k: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let x = fill(k, 2);
    let mut c = vec![0.0f32; m];
    let bytes = (m * k + k + m) * 4;
    let mut run = || {
        measure_gbps(bytes, || {
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&x, k, 1),
                0.0,
                MatMut::from_col_major(&mut c, m, 1),
                par,
            )
        })
    };
    let prev = gemmkit::tuning::gemv_threshold();
    gemmkit::tuning::set_gemv_threshold(usize::MAX - 1); // gemv path on
    let axpy = run();
    gemmkit::tuning::set_gemv_threshold(0); // gemv path off -> general driver (packs A)
    let driver = run();
    gemmkit::tuning::set_gemv_threshold(prev);
    let (g, _) = extern_baselines(m, k, 1, bytes, par);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  m={m:<8} k={k:<5} {mode}  axpy={:7.1}  driver={:7.1} ({:.2}× axpy)  gemm={:7.1} ({:.2}× axpy)",
        axpy.median,
        driver.median,
        driver.median / axpy.median.max(1e-9),
        g,
        g / axpy.median.max(1e-9),
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gemv_paths() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\ngemv path investigation — axpy special path vs general driver vs gemm crate:");
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &(m, k) in &[
            (4_194_304usize, 16usize),
            (4_194_304, 32),
            (2_097_152, 48),
            (2_097_152, 64),
            (1_048_576, 96),
        ] {
            bench_gemv_paths(m, k, par);
        }
    }
}

/// Register-block calibration for the axpy gemv: sweeps `K_STREAM_MAX` (the cap on `k` for which an
/// *engaged* output register-block runs) over an `m x k` grid to map *where* register-blocking pays.
/// The result is **output-size-dependent**: it loses while the output stays L3-resident (the
/// matrix-stream prefetcher thrash dominates the cheap re-reads) and wins once the output spills
/// toward DRAM, which is why the engage gate is a fraction of L3, not L2 (see
/// `special::gemv::output_register_block`). With the L3/2 gate, moderate-`m` rows read the same at
/// every cap (register-block gated off), and only the huge-`m` rows show the cap effect. Calibrated
/// on Zen5; re-run on any new target (e.g. aarch64) before retuning the gate or the cap
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_k_stream() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nK_STREAM_MAX calibration — axpy gemv GB/s vs register-block cap (serial):");
    // Restore K_STREAM_MAX on scope exit *including an unwinding panic*: it is a process-global
    // that later benches in this binary read, so a trailing statement would leak the last cap if a
    // large-shape iteration panics (e.g. OOM). RAII, not a trailing `set`
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            gemmkit::tuning::set_k_stream_max(self.0);
        }
    }
    let _restore = Restore(gemmkit::tuning::k_stream_max());
    // Output spills L2 at every `m` below; `k` straddles the candidate caps. `cap >= k` => the
    // output register-blocks; `cap < k` => the plain column-outer form. So (32 vs 16) changes only
    // the k=24/32 rows, (16 vs 8) only the k=16 row; k=8 register-blocks under all 3. The huge
    // `m` rows are the regime the path exists for: output spills LLC (DRAM-bound), where holding it in
    // registers avoids re-streaming it per column. `k` is bounded there to cap the matrix alloc
    for &(m, ks) in &[
        (300_000usize, &[8usize, 16, 24, 32, 48][..]),
        (1_048_576, &[8, 16, 24, 32, 48][..]),
        (4_194_304, &[8, 16, 24, 32][..]),
        (8_388_608, &[8, 16, 32][..]),
        (16_777_216, &[8, 16][..]), // k bounded so the matrix stays <= ~1 GiB
    ] {
        for &k in ks {
            let a = fill(m * k, 1);
            let x = fill(k, 2);
            let bytes = (m * k + k + m) * 4;
            let mut row = format!("  m={m:<9} k={k:<3} ");
            for &cap in &[32usize, 16, 8] {
                gemmkit::tuning::set_k_stream_max(cap);
                let mut y = vec![0.0f32; m];
                let s = measure_gbps(bytes, || {
                    gemm(
                        1.0,
                        MatRef::from_col_major(&a, m, k),
                        MatRef::from_col_major(&x, k, 1),
                        0.0,
                        MatMut::from_col_major(&mut y, m, 1),
                        Parallelism::Serial,
                    );
                });
                row.push_str(&format!("cap={cap:<2}={:6.1}  ", s.median));
            }
            println!("{row}");
        }
    }
}

/// gevv / skinny GEMM `C(mxn) = A(mxk)*B(kxn)` at small `k`, reported as GB/s of the
/// minimum traffic `(m*k + k*n + m*n)*4` (beta = 0, so C is write-only) against the STREAM
/// `ceiling`, plus the `gemm`-crate / `matrixmultiply` GB/s on the same shape (`kit x` is
/// gemmkit's speedup). At tiny `k` the `m*n` C write dominates, so this is
/// write-bandwidth-bound
fn bench_gevv(m: usize, n: usize, k: usize, par: Parallelism, ceiling: f64) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let bytes = (m * k + k * n + m * n) * 4;
    let st = measure_gbps(bytes, || {
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
    let tail = baseline_tail(m, k, n, bytes, par, st.median);
    println!(
        "  m={m:<5} n={n:<5} k={k} {mode}  kit={:7.1} GB/s (±{:>2.0}%)  {:3.0}% ceil{tail}",
        st.median,
        st.spread_pct(),
        100.0 * st.median / ceiling.max(1e-9),
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gevv() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\ngevv / skinny GEMM (C[m×n] = A[m×k]·B[k×n], small k) — GB/s vs STREAM Triad ceiling:"
    );
    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let ser_ceiling = stream_triad_serial().median;
    let (peak_thr, par_ceiling) = stream_triad_parallel_peak(avail);
    let par_ceiling = par_ceiling.median;
    println!(
        "  serial ceiling {ser_ceiling:.1} GB/s;  parallel ceiling {par_ceiling:.1} GB/s @ {peak_thr} thr"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let ceiling = if matches!(par, Parallelism::Serial) {
            ser_ceiling
        } else {
            par_ceiling
        };
        for &(m, n) in &[(4096usize, 4096usize), (8192, 2048)] {
            for &k in &[1usize, 2, 4] {
                bench_gevv(m, n, k, par, ceiling);
            }
        }
    }
}

/// Forced thread-scaling of a bandwidth-bound gemv: GB/s at `Rayon(t)` for a ladder of `t`,
/// plus what the auto `Rayon(0)` path actually picks. On a DRAM-bound shape the curve should
/// climb to the STREAM ceiling within a few threads and then *plateau or dip* (more workers
/// add no bandwidth and eventually cost sync), which is exactly what the bandwidth cap is
/// there to hold: the auto row should sit on the plateau, not past it
fn bench_gemv_scaling(m: usize, k: usize, ceiling: f64, avail: usize) {
    let a = fill(m * k, 1);
    let x = fill(k, 2);
    let mut c = vec![0.0f32; m];
    let bytes = (m * k + k + m) * 4;
    let mut run = |par| {
        measure_gbps(bytes, || {
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&x, k, 1),
                0.0,
                MatMut::from_col_major(&mut c, m, 1),
                par,
            );
        })
    };
    println!("  m={m} k={k}  (A={} MiB):", m * k * 4 / (1024 * 1024));
    for &t in &[1usize, 2, 4, 8, 16, 32] {
        if t > avail {
            break;
        }
        let par = if t == 1 {
            Parallelism::Serial
        } else {
            Parallelism::Rayon(t)
        };
        let st = run(par);
        println!(
            "    t={t:<3} {:7.1} GB/s (±{:>2.0}%)  {:3.0}% ceil",
            st.median,
            st.spread_pct(),
            100.0 * st.median / ceiling.max(1e-9)
        );
    }
    let st = run(Parallelism::Rayon(0));
    println!(
        "    auto  {:7.1} GB/s (±{:>2.0}%)  {:3.0}% ceil",
        st.median,
        st.spread_pct(),
        100.0 * st.median / ceiling.max(1e-9)
    );
}

/// Force the small-`k` route on vs off back-to-back (via the threshold setter, so the same
/// buffers/pool are reused and machine drift cancels) for a skinny GEMM, sweeping `k` to
/// find where the register-tiling driver catches up. `on % of off` above 100% means the
/// in-place small-`k` route still beats the driver; the calibrated threshold sits at the
/// last `k` where it does
fn bench_small_k_crossover(m: usize, n: usize, k: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let bytes = (m * k + k * n + m * n) * 4;
    let mut run = |v: usize| {
        gemmkit::tuning::set_small_k_threshold(v);
        measure_gbps(bytes, || {
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    };
    let prev = gemmkit::tuning::small_k_threshold();
    let on = run(k); // k <= threshold -> small-k route
    let off = run(0); // threshold 0 -> driver route
    gemmkit::tuning::set_small_k_threshold(prev);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  m={m:<5} n={n:<5} k={k:<3} {mode}  small_k={:7.1} (±{:>2.0}%)  driver={:7.1} (±{:>2.0}%)  (small_k {:.0}% of driver)",
        on.median,
        on.spread_pct(),
        off.median,
        off.spread_pct(),
        100.0 * on.median / off.median.max(1e-9)
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_small_k() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nsmall-k route crossover (skinny GEMM) — in-place small_k vs register-tiling driver:"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &(m, n) in &[(4096usize, 4096usize), (8192, 2048)] {
            for &k in &[2usize, 4, 8, 16, 32, 64] {
                bench_small_k_crossover(m, n, k, par);
            }
        }
    }
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gemv_scaling() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\ngemv thread-scaling (forced Rayon(t) vs auto) — GB/s vs parallel STREAM ceiling:");
    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let (peak_thr, par_ceiling) = stream_triad_parallel_peak(avail);
    let ceiling = par_ceiling.median;
    println!("  parallel ceiling {ceiling:.1} GB/s @ {peak_thr} thr; cores={avail}");
    // A DRAM-bound axpy shape (out fits cache) and a register-blocked out-spills-L3 shape
    bench_gemv_scaling(65536, 1024, ceiling, avail);
    bench_gemv_scaling(16_777_216, 8, ceiling, avail);
    // Cache-resident shapes (A fits L3): the ceiling here is aggregate cache bandwidth, which
    // keeps scaling with cores well past the DRAM-saturation count, so the auto row (capped
    // by the DRAM proxy) is expected to sit far below the forced-high-t peak
    bench_gemv_scaling(1024, 1024, ceiling, avail);
    bench_gemv_scaling(8192, 64, ceiling, avail);
}
