//! f32 sgemm vs the `gemm` crate / `matrixmultiply`, an equal-ISA variant of that same
//! comparison, per-call latency at the parallel work gate, and a thread-scaling diagnostic

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

// gemmkit's best auto-selected ISA against the `gemm` crate and `matrixmultiply`: both are
// dev-deps excluded from wasm builds, so this bench (and `perf_sgemm`, its caller) is too
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

/// gemmkit vs `gemm` with both pinned to the same single ISA, so the comparison isolates
/// kernel/scheduling quality from any ISA-selection gap between the 2. Serial, column-major.
/// gemmkit is driven directly through [`driver::run`] with the harness's `NativeTok`/tile
/// (bypassing dispatch's own auto-select); `gemm`'s `Parallelism::None` already runs its own
/// default (auto-selected) ISA, which on x86 matches gemmkit's forced token only because
/// `gemm`'s AVX-512 path needs its `nightly` feature, not enabled here (see `harness.rs`)
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

/// Per-call latency at and just above the parallel work gate (`parallel_threshold`, default
/// 48*48*256): a product this size takes only tens of microseconds, so the per-call resolve
/// cost fixed overhead (reading tuning knobs, querying the core count, computing the worker
/// ramp) is a real fraction of the total here, unlike at any larger size where it disappears
/// into the noise. The serial arm is the control: its resolve path returns before ever
/// querying the core count, so a change in resolve cost only ever moves the parallel rows
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
    // Derived from the median GFLOP/s rather than timed separately, so it shares the same
    // calibrated batches instead of paying its own measurement noise
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

// Parallel thread-scaling diagnostic (locating where mid-size parallel throughput breaks down)

/// A rough estimate of the (MR, NR) microtile the default `gemm()` dispatch would use on this
/// target, for sizing the printed "jobs/region" estimate below (not for driving any call
/// here). Assumes the best available x86 ISA is AVX-512; on an AVX2-only box the real tile is
/// the smaller 16x6: `mc`'s cap scales directly with `mr` (`tuning::mc_reg_panels`), and the
/// per-region job count divides directly by both `mr` and `nr`, so the smaller tile splits the
/// same problem into more, finer jobs than this bigger assumed tile predicts, making the
/// printed count a lower bound rather than the true figure there
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

/// Prints gemmkit's own parallel scaling (and `gemm`'s, for reference) at a fixed size across
/// a thread-count ladder, to show *where* scaling stalls: a poor speedup already by 2-4
/// threads points at per-call fork/join and atomic-cursor overhead dominating a small
/// workload; a plateau only after 8-16 threads points instead at memory bandwidth or job
/// starvation (compare the plateau point against the printed jobs/region estimate). Each
/// figure is the median of `REPS` calibrated batches; the spread column exists so a
/// difference smaller than the batch-to-batch noise is not mistaken for a real effect
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

    // Rayon(1) resolves to the same single-worker code path as Parallelism::Serial, so the
    // t=1 row below is just this already-measured serial baseline, not a fresh measurement
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
        // The efficiency denominator is what resolve() actually grants (capped by the core
        // count and the per-region job count), not the raw requested `t`; using `t` itself
        // would read artificially low wherever `n_jobs` throttles the grant below `t` and
        // make that throttling look like a bandwidth wall instead of what it is
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

    // The forced-t ladder above never exercises the auto Rayon(0) path production code
    // actually takes, so this row is the only one that shows what the auto ramp picks and
    // delivers on this shape. `auto_w` mirrors resolve()'s own auto-worker formula
    // (mnk / par_mnk_per_worker, capped by cores and jobs) purely for the printed estimate
    let auto_w = ((m * k * n) / gemmkit::tuning::par_mnk_per_worker().max(1))
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
