//! Bandwidth-bound shapes: STREAM ceilings, gemv (axpy/dot/mixed), gevv, and the small-k /
//! small-k-vs-driver crossover, each read as a fraction of the machine's own DRAM bandwidth
//! rather than as raw GFLOP/s

use crate::harness::{BENCH_GUARD, Stat, fill, measure_gbps};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// gemv (matrix*vector), gevv (rank-k / outer product), and any GEMM with small k move O(1)
// bytes per flop: each element of A or B feeds only 1 or a few multiply-adds before it is
// never touched again, so these shapes never get to reuse anything out of cache the way a
// square, compute-bound GEMM does. The ceiling on their throughput is therefore how fast the
// machine can move bytes, not how fast it can multiply, so every bench below reports achieved
// GB/s against a STREAM Triad ceiling instead of GFLOP/s against a peak-flop ceiling. The
// serial arm is judged against a single-core Triad; the parallel arm against an aggregate
// multi-core Triad, since 1 core's bandwidth share is far below what the whole memory
// subsystem can sustain and using the serial number there would flatter every parallel result

/// STREAM array length, in `f32` elements: 64 Mi (256 MiB), far larger than any last-level
/// cache on the reference hardware, so every STREAM kernel below is forced to stream from
/// DRAM rather than being served out of a cache that happens to be big enough to hide it
const STREAM_LEN: usize = 64 * 1024 * 1024;

/// Single-core STREAM Triad, `a[i] = b[i] + alpha*c[i]`, reported in GB/s over the 3*N*4-byte
/// traffic (`b` and `c` each read once, `a` written once). This is the scalar ceiling the
/// *serial* gemv/gevv benches are measured against. `black_box` on the output pointer stops
/// the optimizer from proving the loop's result is unused and eliding it
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

/// Single-core STREAM Copy, `dst[i] = src[i]`, reported in GB/s over the 2*N*4-byte traffic:
/// a simpler read+write reference to compare the triad number against
fn stream_copy_serial() -> Stat {
    let n = STREAM_LEN;
    let src = fill(n, 3);
    let mut dst = vec![0.0f32; n];
    measure_gbps(2 * n * 4, || {
        dst.copy_from_slice(&src);
        std::hint::black_box(dst.as_ptr());
    })
}

/// Aggregate STREAM Triad across `threads` OS threads, each computing 1 disjoint chunk inside
/// a `std::thread::scope`: raw threads rather than gemmkit's own `Rayon` parallelism (or the
/// `rayon` dev-dependency used elsewhere in this test suite), so the ceiling stays independent
/// of the very machinery it is meant to judge. This is the fair reference for the *parallel*
/// gemv/gevv benches: a single core saturates only a slice of DRAM bandwidth, so a
/// threaded, output-partitioned gemv is really racing the whole-machine triad, not the
/// single-core one. The array is large enough (256 MiB) that thread-spawn cost is negligible
/// next to the several-millisecond streaming time it takes to sweep it
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

/// The best aggregate Triad bandwidth found over a small thread-count ladder, with the count
/// that reached it. DRAM bandwidth typically saturates well before the logical core count and
/// can even regress past it (memory-controller contention from too many concurrent streams),
/// so sweeping for the peak is the honest parallel ceiling, not just reading off the
/// all-cores figure
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

/// External-library GB/s for a column-major f32 `C(mxn) = A(mxk)*B(kxn)` with `beta = 0`, on
/// the same traffic byte count the caller is scoring gemmkit against: the `gemm` crate, plus
/// (serial only) `matrixmultiply`. Native-only: both are dev-deps excluded from wasm builds.
/// Returns `(gemm GB/s, matrixmultiply GB/s or None)`
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
    // matrixmultiply has no parallel entry point in how it is called here, so it is only a
    // fair comparison on the serial arm
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

/// Formats the external-baseline tail (`gemm=... (kit ...x)  mm=... (kit ...x)`) that most
/// GB/s rows below append to their own `kit=...` figure. Empty on wasm, where there is no
/// external crate to compare against
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

/// gemv `C(mx1) = A(mxk)*x` through the public [`gemm`] entry point, column-major A (the
/// axpy layout), reported as GB/s of the minimum traffic `(m*k + k + m)*4` (A read once, `x`
/// read once, `C` written once) against the STREAM `ceiling`, plus the `gemm`-crate and
/// `matrixmultiply` GB/s on the same shape. `k` sweeps from L2-resident up to DRAM-sized so
/// both cache-hot and cache-cold regimes are covered
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
        // The output (m*4 bytes) stays well under L2 for every m here, so the axpy form's
        // per-column output re-read costs nothing extra and A's DRAM stream is what this
        // sweep is actually measuring as k climbs from L2-resident to DRAM-sized
        for &k in &[64usize, 256, 1024] {
            for &m in &[1024usize, 8192, 65536] {
                bench_gemv(m, k, par, ceiling);
            }
        }
        // Output-size sweep at fixed small k: the smallest m here already spills the private
        // L2 (1 MiB / 262144 f32 elements), and the sweep climbs on through the point where
        // the output also outgrows the shared L3, which is exactly the regime output
        // register-blocking exists to help (see `special::gemv::output_register_block`)
        // `k` is kept small throughout so A itself stays well under DRAM-filling size
        for &(m, k) in &[
            (1_048_576usize, 64usize),
            (4_194_304, 16),
            (8_388_608, 8),
            (16_777_216, 8),
            (16_777_216, 16), // same output size as above, larger k: prefetch-thrash regression check
        ] {
            bench_gemv(m, k, par, ceiling);
        }
    }
}

/// gemv `C(mx1) = A(mxk)*x` with **row-major** A, which routes to the dot-layout path
/// ([`gemmkit`]'s `special::gemv::dot_rows`) instead of the axpy one: row-major A makes the
/// matrix rows contiguous over `k`, and the unit-stride `x`/`C` complete the dot path's gate.
/// Reported as GB/s of the same minimum traffic `(m*k + k + m)*4` as [`bench_gemv`] against
/// the STREAM `ceiling`, plus a derived GFLOP/s figure (`2*m*k` flops, same elapsed time) that
/// makes the dot path's single-FMA-chain-per-row latency wall visible on cache-resident
/// shapes. The `gemm`-crate / `matrixmultiply` baselines still run with column-major A (their
/// only supported layout here), but the traffic and flop count line up with this shape either
/// way, so the comparison stays meaningful
fn bench_gemv_dot(m: usize, k: usize, par: Parallelism, ceiling: f64) {
    let a = fill(m * k, 1);
    let x = fill(k, 2);
    let mut c = vec![0.0f32; m];
    let bytes = (m * k + k + m) * 4;
    let st = measure_gbps(bytes, || {
        gemm(
            1.0,
            MatRef::from_row_major(&a, m, k),
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
    // Same elapsed time as the GB/s figure, so this just re-expresses it through the
    // known flop/byte ratio rather than timing the call a 2nd time
    let gflops = st.median * (2.0 * m as f64 * k as f64) / bytes as f64;
    let tail = baseline_tail(m, k, 1, bytes, par, st.median);
    println!(
        "  m={m:<6} k={k:<6} {mode}  kit={:7.1} GB/s (±{:>2.0}%)  {:7.1} GFLOP/s  {:3.0}% ceil{tail}",
        st.median,
        st.spread_pct(),
        gflops,
        100.0 * st.median / ceiling.max(1e-9),
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gemv_dot() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\ngemv dot-layout (C[m×1] = A[m×k]·x, row-major A -> dot_rows) — GB/s + GFLOP/s vs STREAM ceiling:"
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
        // An L2-resident pair and an L3-resident pair (both latency-bound, where the dot
        // path's 1-FMA-chain-per-row cost is visible), a few-long-rows shape (probes how
        // much sharing 1 vector read across many rows is worth), and a DRAM-bound pair
        // this route should leave unchanged (bandwidth already hides the arithmetic there)
        for &(m, k) in &[
            (512usize, 512usize), // ~1 MiB: L2-resident
            (2048, 2048),         // 16 MiB: L3-resident
            (64, 65536),          // few, very long rows
            (8192, 8192),         // 256 MiB: DRAM-bound regression guard
        ] {
            bench_gemv_dot(m, k, par, ceiling);
        }
    }
}

/// f16/bf16 gemv `C(mx1) = A(mxk)*x` (or the `m == 1` transposed form) through the dedicated
/// mixed-precision gemv route, reported as GB/s of the **narrow** minimum traffic
/// `(m*k + k + m)*sizeof(T)` plus a derived GFLOP/s figure. Measures the same buffers twice,
/// back to back, with the widen-gemv route forced on and then off (via the threshold setter,
/// so both runs share machine state and any drift cancels): `driver=` is what a general-driver
/// fallback would have delivered on this exact shape, so the ratio is the honest gain from
/// having a dedicated narrow gemv path at all. `row_major_a` selects the dot layout
/// (`dot_rows_mixed`) vs the axpy layout (`axpy_mixed`); both widen every load to `f32`,
/// accumulate there, and round to the narrow type once at the store
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_gemv_mixed<T: gemmkit::GemmScalar>(
    label: &str,
    m: usize,
    k: usize,
    row_major_a: bool,
    par: Parallelism,
    to_t: impl Fn(f32) -> T,
) {
    let a: Vec<T> = fill(m * k, 1).iter().map(|&v| to_t(v)).collect();
    let x: Vec<T> = fill(k, 2).iter().map(|&v| to_t(v)).collect();
    let mut c = vec![T::ZERO; m];
    let bytes = (m * k + k + m) * core::mem::size_of::<T>();
    let (alpha, beta) = (to_t(1.0), to_t(0.0));
    let mut run = || {
        measure_gbps(bytes, || {
            let a_ref = if row_major_a {
                MatRef::from_row_major(&a, m, k)
            } else {
                MatRef::from_col_major(&a, m, k)
            };
            gemm(
                alpha,
                a_ref,
                MatRef::from_col_major(&x, k, 1),
                beta,
                MatMut::from_col_major(&mut c, m, 1),
                par,
            );
        })
    };
    let prev = gemmkit::tuning::gemv_threshold();
    gemmkit::tuning::set_gemv_threshold(usize::MAX - 1); // widen gemv route on
    let gv = run();
    gemmkit::tuning::set_gemv_threshold(0); // gemv route off -> falls to the general driver
    let drv = run();
    gemmkit::tuning::set_gemv_threshold(prev);
    let gflops = gv.median * (2.0 * m as f64 * k as f64) / bytes as f64;
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    let layout = if row_major_a { "dot " } else { "axpy" };
    println!(
        "  {label} {layout} m={m:<7} k={k:<6} {mode}  gemv={:7.1} GB/s (±{:>2.0}%)  {:8.1} GFLOP/s   driver={:7.1}  ({:.2}× driver)",
        gv.median,
        gv.spread_pct(),
        gflops,
        drv.median,
        gv.median / drv.median.max(1e-9),
    );
}

#[cfg(all(feature = "half", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_gemv_mixed() {
    use gemmkit::{bf16, f16};
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nf16/bf16 gemv (C[m×1] = A[m×k]·x) — widen gemv route (dot/axpy) vs general driver:"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &row_major_a in &[true, false] {
            for &(m, k) in &[(4096usize, 4096usize), (65536, 1024), (1024, 65536)] {
                bench_gemv_mixed::<f16>("f16 ", m, k, row_major_a, par, f16::from_f32);
                bench_gemv_mixed::<bf16>("bf16", m, k, row_major_a, par, bf16::from_f32);
            }
        }
    }
}

/// For a gemv-shaped call, compares the dedicated gemv special path against the general
/// driver (reached by forcing the gemv threshold down to 0) and against the `gemm` crate, all
/// 3 on the same buffers. Confirms the special path earns its keep: the driver packs A into
/// micropanels for a compute-bound kernel, which is the wrong tool for a bandwidth-bound
/// shape, and this shows exactly how much that costs; it also tracks the remaining gap to
/// `gemm`'s own dedicated path
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

/// Calibration sweep for `K_STREAM_MAX`, the depth ceiling that gates the axpy-gemv output
/// register-blocking strategy (see `special::gemv::output_register_block`): for each `(m, k)`
/// pair, GB/s is measured 3 times with the cap forced to 32/16/8 in turn, so `cap >= k` takes
/// the register-blocked strategy and `cap < k` falls back to the plain column-outer one on
/// the exact same buffers. Register-blocking also needs the output to clear a separate byte
/// gate (a fraction of the last-level cache, not swept here), so a row where the output stays
/// small enough to be cache-resident reads identically at every cap regardless of `k`, and
/// only once the output grows past that gate does the cap actually change which strategy
/// runs. `k` is kept small on the largest `m` rows so the A matrix itself stays a manageable
/// size. Calibrated on Zen5; re-run on any new target (e.g. aarch64) before retuning either
/// the gate or this cap, since both are machine properties
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_k_stream() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nK_STREAM_MAX calibration — axpy gemv GB/s vs register-block cap (serial):");
    // Restored on scope exit, including an unwinding panic from a later assertion in this
    // process: K_STREAM_MAX is a global every other bench in this binary also reads, so a
    // trailing `set_k_stream_max` at the end of this function would never run if a large
    // shape panics partway through, leaking a stale cap into whatever runs next. RAII instead
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            gemmkit::tuning::set_k_stream_max(self.0);
        }
    }
    let _restore = Restore(gemmkit::tuning::k_stream_max());
    // Every m/k pair below is chosen so the sweep crosses through the byte gate as well as
    // the cap: the smaller m rows keep the output well under the gate (all 3 caps behave
    // identically there, since register-blocking never engages regardless of k), while the
    // largest m rows push the output past it, where the cap sweep actually changes behavior
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

/// gevv / skinny GEMM `C(mxn) = A(mxk)*B(kxn)` at small `k`, reported as GB/s of the minimum
/// traffic `(m*k + k*n + m*n)*4` against the STREAM `ceiling`, plus the `gemm`-crate and
/// `matrixmultiply` GB/s on the same shape. `beta = 0` makes `C` write-only, and at these tiny
/// `k` values the `m*n` write dominates the traffic total, so this is really a write-bandwidth
/// probe more than a read one
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
/// plus what the auto `Rayon(0)` path actually picks on this same shape. On a DRAM-bound
/// shape the curve should climb toward the STREAM ceiling within a few threads and then
/// plateau (or even dip, once sync overhead outweighs any bandwidth still on the table) since
/// more workers cannot conjure more DRAM bandwidth; the bandwidth-aware worker cap exists
/// exactly to keep the auto row sitting on that plateau instead of past it
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

/// Forces the small-`k` in-place route on and off, back-to-back, via the threshold setter (so
/// both runs reuse the same buffers and machine drift cancels), and sweeps `k` to find where
/// the register-tiling driver catches up to it. A value above 100% in the "small_k % of
/// driver" figure means the in-place route still wins at that `k`; the shipped threshold sits
/// at the largest `k` where that is still true
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
    // A DRAM-bound shape (the output fits comfortably in cache, so A's own streaming is what
    // limits it) and a register-blocked shape whose output itself spills the last-level cache
    bench_gemv_scaling(65536, 1024, ceiling, avail);
    bench_gemv_scaling(16_777_216, 8, ceiling, avail);
    // Cache-resident shapes (A fits comfortably in the shared last-level cache): the relevant
    // ceiling here is aggregate cache bandwidth, not DRAM, which keeps climbing with core
    // count well past where DRAM would have saturated, so the DRAM-bandwidth-capped auto row
    // is expected to land far below the forced-high-thread-count peak on these 2
    bench_gemv_scaling(1024, 1024, ceiling, avail);
    bench_gemv_scaling(8192, 64, ceiling, avail);
}
