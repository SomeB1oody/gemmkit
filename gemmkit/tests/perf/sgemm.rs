//! f32 sgemm vs gemm crate / matrixmultiply, thread-scaling, per-call latency

use crate::harness::{BENCH_GUARD, fill, measure};
// Driver-level imports are used only by the equal-ISA bench; gate them to its
// architectures so other targets stay warning-clean
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use crate::harness::{NATIVE_LABEL, NATIVE_MR, NATIVE_NR, NativeTok};
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use gemmkit::Workspace;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use gemmkit::driver;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use gemmkit::kernel::FloatGemm;
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// gemmkit best-ISA vs the `gemm` crate + `matrixmultiply`: external crates that do
// not build for wasm, so this bench (and its `perf_sgemm` caller) is gated off wasm
#[cfg(not(target_family = "wasm"))]
fn bench_one(s: usize, parallel: bool) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];

    let par = if parallel {
        Parallelism::Rayon(0)
    } else {
        Parallelism::Serial
    };
    let s_kit = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });

    let gpar = if parallel {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    let s_gemm = measure(m, k, n, || unsafe {
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

    let mode = if parallel { "par" } else { "ser" };
    print!(
        "  n={s:<5} {mode}  gemmkit={:7.1} (±{:>2.0}%)  gemm={:7.1} (±{:>2.0}%)  ({:.0}% of gemm)",
        s_kit.median,
        s_kit.spread_pct(),
        s_gemm.median,
        s_gemm.spread_pct(),
        100.0 * s_kit.median / s_gemm.median.max(1e-9)
    );
    if !parallel {
        let s_mm = measure(m, k, n, || unsafe {
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
        });
        print!(
            "  mm={:7.1}  ({:.2}x mm)",
            s_mm.median,
            s_kit.median / s_mm.median.max(1e-9)
        );
    }
    println!();
}

/// Equal-ISA comparison: gemmkit's native single-ISA path (forced via the
/// driver) vs gemm's default (the same ISA on stable). Single-threaded,
/// column-major
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn bench_native_equal_isa(s: usize) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let mut ws = Workspace::new();

    let s_kit = measure(m, k, n, || unsafe {
        driver::run::<FloatGemm<f32>, NativeTok, NATIVE_MR, NATIVE_NR>(
            NativeTok::default(),
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
            Parallelism::Serial,
            &mut ws,
        );
    });
    let s_gemm = measure(m, k, n, || unsafe {
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
            gemm::Parallelism::None,
        );
    });
    let label = NATIVE_LABEL;
    println!(
        "  n={s:<5} ser  gemmkit-{label}={:7.1} (±{:>2.0}%)  gemm-{label}={:7.1}  ({:.0}% of gemm)",
        s_kit.median,
        s_kit.spread_pct(),
        s_gemm.median,
        100.0 * s_kit.median / s_gemm.median
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_sgemm() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nsgemm GFLOP/s (f32, column-major) — gemmkit best-ISA vs gemm default:");
    for &s in &[256usize, 512, 1024, 2048] {
        bench_one(s, false);
    }
    for &s in &[512usize, 1024, 2048, 4096] {
        bench_one(s, true);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        println!("\nequal-ISA (gemmkit vs gemm, same single ISA), single-threaded:");
        for &s in &[256usize, 512, 1024, 2048] {
            bench_native_equal_isa(s);
        }
    }
}

/// Per-call latency probe for shapes at and just above the parallel work gate
/// (`parallel_threshold`, default 48*48*256): a few-MFLOP product runs tens of microseconds,
/// so the fixed per-call resolve cost (knob reads, core-count queries, worker-ramp math) is a
/// visible fraction here and nowhere else. Serial is the control: its resolve path exits
/// before any core-count query, so a resolve-cost change moves only the parallel rows
#[cfg(not(target_family = "wasm"))]
fn bench_call_latency(m: usize, k: usize, n: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let st = measure(m, k, n, || {
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
    // Median microseconds per call, derived from the median throughput
    let us = 2.0 * m as f64 * k as f64 * n as f64 / st.median / 1e3;
    println!(
        "  {m:>4}x{k:<4}x{n:<4} {mode}  {:7.1} GFLOP/s (±{:>2.0}%)  {us:7.2} us/call",
        st.median,
        st.spread_pct()
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_call_latency() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nper-call latency at the parallel gate (fixed resolve cost visibility):");
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        bench_call_latency(48, 256, 48, par);
        bench_call_latency(96, 256, 96, par);
        bench_call_latency(128, 256, 128, par);
    }
}

// Parallel thread-scaling diagnostic (the mid-size-parallel gap)

/// The (MR, NR) tile the default `gemm()` dispatch uses on this target, used
/// only to *estimate* the per-region job count (the parallel work granularity).
/// Assumes the best available x86 ISA is AVX-512; if the box only has AVX2 the
/// real tile is 16x6 and the printed job estimate is a lower bound
#[cfg(not(target_family = "wasm"))]
fn native_default_tile() -> (usize, usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        (32, 12)
    }
    #[cfg(target_arch = "aarch64")]
    {
        (16, 4)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        (4, 4)
    }
}

/// Print gemmkit's parallel *self*-scaling (and gemm's, for reference) at a fixed
/// size across thread counts, showing *where* scaling breaks: poor speedup
/// already at 2-4 threads => per-call fork/join + atomics overhead dominates the
/// tiny work; a plateau after 8-16 => memory bandwidth or job starvation (compare
/// against the printed ~jobs/region). Throughput is the median of `REPS`
/// calibrated batches; the spread column flags differences smaller than the noise
#[cfg(not(target_family = "wasm"))]
fn bench_scaling(s: usize) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];

    let (mr, nr) = native_default_tile();
    let blk = gemmkit::topology().blocking(mr, nr, 4, m, n, k);
    let mc = blk.mc.next_multiple_of(mr).max(mr);
    let nc = blk.nc.next_multiple_of(nr).max(nr);
    let n_jobs = m.div_ceil(mc) * n.min(nc).div_ceil(nr);
    println!(
        "\n  n={s}  kc={} mc={} nc={}  ~{} jobs/region (tile {mr}x{nr}):",
        blk.kc, mc, nc, n_jobs
    );
    println!("    thr |   gemmkit  spd  eff% | spread |     gemm  spd");

    let base = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            Parallelism::Serial,
        );
    });
    let gbase = measure(m, k, n, || unsafe {
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
            gemm::Parallelism::None,
        );
    });

    // The t=1 row is the serial `base`/`gbase` already measured (Rayon(1) resolves
    // to the same single-worker path), so reuse them instead of re-measuring
    println!(
        "      1 | {:9.1}  1.0x 100% | {:5.0}% | {:8.1}  1.0x",
        base.median,
        base.spread_pct(),
        gbase.median
    );

    let avail = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    for &t in &[2usize, 4, 8, 16, 32] {
        let sk = measure(m, k, n, || {
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Rayon(t),
            );
        });
        let sg = measure(m, k, n, || unsafe {
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
                gemm::Parallelism::Rayon(t),
            );
        });
        let spd = sk.median / base.median.max(1e-9);
        // Effective workers = what resolve() actually grants (capped by cores and
        // the per-region job count), not the requested t, else eff% reads low
        // where n_jobs throttles below t and masquerades as a bandwidth wall
        let workers = t.min(avail).min(n_jobs).max(1);
        println!(
            "    {t:3} | {:9.1} {:4.1}x {:3.0}% | {:5.0}% | {:8.1} {:4.1}x",
            sk.median,
            spd,
            100.0 * spd / workers as f64,
            sk.spread_pct(),
            sg.median,
            sg.median / gbase.median.max(1e-9)
        );
    }

    // Auto row: the forced-t curve above never exercises the default `Rayon(0)`
    // path production uses, so this is the only line that shows what the auto ramp
    // actually selects and delivers. `auto_w` mirrors `resolve`'s auto branch
    // (cbrt(mnk).div_ceil(stride), capped) for sizes above the serial gate
    let auto_w = (((m * k * n) as f64).cbrt() as usize)
        .div_ceil(gemmkit::tuning::thread_dim_stride())
        .min(avail)
        .min(n_jobs)
        .max(1);
    let sk = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            Parallelism::Rayon(0),
        );
    });
    let spd = sk.median / base.median.max(1e-9);
    println!(
        "   auto | {:9.1} {:4.1}x {:3.0}% | {:5.0}% | picks {auto_w} workers",
        sk.median,
        spd,
        100.0 * spd / auto_w as f64,
        sk.spread_pct()
    );
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_scaling() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nparallel thread-scaling (f32 col-major) — gemmkit default ISA vs gemm:");
    for &s in &[256usize, 512, 1024, 2048] {
        bench_scaling(s);
    }
}
