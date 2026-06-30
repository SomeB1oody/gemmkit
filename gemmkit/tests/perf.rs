//! Performance suite — `#[ignore]` benchmarks (not correctness gates), run manually.
//!
//! * **Native cross-library benchmarks**: gemmkit vs the `gemm` crate / `matrixmultiply`.
//!   These depend on those dev-deps, which **do not build for wasm** (they are
//!   `cfg(all(not(miri), not(target_family = "wasm")))` dev-deps — see `Cargo.toml`), so
//!   each bench that calls them is individually gated `cfg(not(target_family = "wasm"))`.
//! * **The wasm `simd128` benchmark** (`perf_simd128`): simd128 vs the scalar token,
//!   mirroring the native `NativeTok` + `bench_native_equal_isa` pattern with the *scalar
//!   token* as the reference (no external crate on wasm). The shared harness
//!   (`fill`/`measure`/`gflops`/`Stat`) is `std`-only, so it serves both worlds and the
//!   file compiles on wasm. (Correctness of the simd128 path is gated separately by
//!   `isa_simd128` in `tests/correctness.rs`; this is the throughput sanity print.)
//!
//! The whole file compiles away under Miri. The benchmarks each saturate every core, so
//! they must not run concurrently — they take a shared `BENCH_GUARD` lock, so even the
//! default multi-threaded harness serializes them and `--test-threads=1` is optional.
//! Run them with:
//!   cargo test -p gemmkit --release --test perf -- --ignored --nocapture
//! Run the wasm benchmark (compile-time `+simd128`) under a wasm runtime:
//!   RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//!     cargo test -p gemmkit --release --target wasm32-wasip1 \
//!       --no-default-features --features std --test perf -- --ignored --nocapture
#![cfg(not(miri))]

use std::time::Instant;

// `driver` / `FloatGemm` / `Workspace` drive the low-level single-ISA benches. Those
// exist in every config *except* wasm-without-simd128 (there only the public-API tail
// benches survive), so gate the imports to match — `any(not(wasm32), simd128)` — to stay
// warning-clean in the scalar-fallback wasm build.
#[cfg(any(not(target_arch = "wasm32"), target_feature = "simd128"))]
use gemmkit::Workspace;
#[cfg(any(not(target_arch = "wasm32"), target_feature = "simd128"))]
use gemmkit::driver;
#[cfg(any(not(target_arch = "wasm32"), target_feature = "simd128"))]
use gemmkit::kernel::FloatGemm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::Fma;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use gemmkit::simd::{ScalarTok, Simd128};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Serializes the two core-saturating `#[ignore]` benches so the default
/// multi-threaded test harness can't run them concurrently (which would make every
/// GFLOP/s figure meaningless). Poisoning is ignored — a panicking bench must not
/// wedge the other.
static BENCH_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Reps and per-batch target for the robust estimator below.
const REPS: usize = 9;
const BATCH_SECS: f64 = 0.07;

/// A throughput sample: median GFLOP/s plus the min/max so run-to-run spread is
/// *visible* and tuning decisions are not made on noise.
struct Stat {
    median: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn spread_pct(&self) -> f64 {
        100.0 * (self.max - self.min) / self.median.max(1e-9)
    }
}

/// Robust throughput estimate: warm up, auto-calibrate the batch size to
/// ~`BATCH_SECS`, then report the median GFLOP/s (and spread) over `REPS`
/// batches. Far steadier than a single fixed-iter timing.
fn measure<F: FnMut()>(m: usize, k: usize, n: usize, mut f: F) -> Stat {
    for _ in 0..3 {
        f();
    } // warmup + thread-pool spin-up
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
        g.push(gflops(m, k, n, secs));
    }
    g.sort_by(f64::total_cmp);
    Stat {
        median: g[REPS / 2],
        min: g[0],
        max: g[REPS - 1],
    }
}

fn gflops(m: usize, k: usize, n: usize, secs: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / secs / 1e9
}

/// f16 GEMM throughput: gemmkit (f32-accumulate mixed kernel) vs the `gemm` crate
/// (same f16-in-f32-acc convention), reported as a ratio. f16 FLOPs counted like f32.
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_f16(s: usize, parallel: bool) {
    use gemmkit::f16;
    let (m, k, n) = (s, s, s);
    let to16 = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    let a = to16(&fill(m * k, 1));
    let b = to16(&fill(k * n, 2));
    let mut c = vec![f16::from_f32(0.0); m * n];

    let par = if parallel {
        Parallelism::Rayon(0)
    } else {
        Parallelism::Serial
    };
    let s_kit = measure(m, k, n, || {
        gemm(
            f16::from_f32(1.0),
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            f16::from_f32(0.0),
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
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            false,
            false,
            false,
            gpar,
        );
    });
    let mode = if parallel { "par" } else { "ser" };
    println!(
        "  n={s:<5} {mode}  gemmkit={:7.1} (±{:>2.0}%)  gemm={:7.1} (±{:>2.0}%)  ({:.0}% of gemm)",
        s_kit.median,
        s_kit.spread_pct(),
        s_gemm.median,
        s_gemm.spread_pct(),
        100.0 * s_kit.median / s_gemm.median.max(1e-9)
    );
}

#[cfg(all(feature = "half", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_f16() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nf16 GFLOP/s (column-major) — gemmkit mixed kernel vs gemm crate:");
    for &s in &[256usize, 512, 1024, 2048] {
        bench_f16(s, false);
    }
    for &s in &[512usize, 1024, 2048] {
        bench_f16(s, true);
    }
}

/// i8 -> i32 GEMM throughput (no `gemm`-crate baseline — it lacks i8 in 0.18). Just
/// confirms the widen-and-multiply kernel is SIMD-accelerated, not scalar-bound.
#[cfg(all(feature = "int8", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_i8() {
    use gemmkit::{MatMut, MatRef};
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\ni8->i32 GFLOP/s (column-major) — gemmkit widen+i32 kernel:");
    for &par in &[false, true] {
        for &s in &[256usize, 512, 1024, 2048] {
            let (m, k, n) = (s, s, s);
            let a: Vec<i8> = (0..m * k).map(|i| (i % 17) as i8 - 8).collect();
            let b: Vec<i8> = (0..k * n).map(|i| (i % 13) as i8 - 6).collect();
            let mut c = vec![0i32; m * n];
            let p = if par {
                Parallelism::Rayon(0)
            } else {
                Parallelism::Serial
            };
            let st = measure(m, k, n, || {
                gemmkit::gemm_i8(
                    1,
                    MatRef::from_col_major(&a, m, k),
                    MatRef::from_col_major(&b, k, n),
                    0,
                    MatMut::from_col_major(&mut c, m, n),
                    p,
                );
            });
            let mode = if par { "par" } else { "ser" };
            println!(
                "  n={s:<5} {mode}  gemmkit={:7.1} (±{:>2.0}%)",
                st.median,
                st.spread_pct()
            );
        }
    }
}

/// Complex (c32) GEMM throughput: gemmkit (`gemm_cplx`, no conj) vs the `gemm` crate
/// (native c32). Complex FLOPs counted as 4× the real count (a complex mul-add is
/// ~4 real mul + 4 real add), the convention both report.
#[cfg(all(feature = "complex", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_complex() {
    use gemmkit::Complex;
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nc32 GFLOP/s (column-major, 4 flop/mul-add) — gemmkit vs gemm crate:");
    for &par in &[false, true] {
        for &s in &[256usize, 512, 1024] {
            let (m, k, n) = (s, s, s);
            let mk = |seed: u64, n: usize| {
                let mut z = seed | 1;
                (0..n)
                    .map(|_| {
                        z ^= z << 13;
                        z ^= z >> 7;
                        z ^= z << 17;
                        Complex::new((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5, 0.25)
                    })
                    .collect::<Vec<_>>()
            };
            let a = mk(1, m * k);
            let b = mk(2, k * n);
            let mut c = vec![Complex::new(0.0f32, 0.0); m * n];
            let p = if par {
                Parallelism::Rayon(0)
            } else {
                Parallelism::Serial
            };
            let gp = if par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // 4x for the complex flop convention.
            let cflop = |secs: f64| 4.0 * 2.0 * (m * k * n) as f64 / secs / 1e9;
            let sk = measure(m, k, n, || {
                gemmkit::gemm_cplx(
                    Complex::new(1.0f32, 0.0),
                    MatRef::from_col_major(&a, m, k),
                    false,
                    MatRef::from_col_major(&b, k, n),
                    false,
                    Complex::new(0.0f32, 0.0),
                    MatMut::from_col_major(&mut c, m, n),
                    p,
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
                    Complex::new(0.0f32, 0.0),
                    Complex::new(1.0f32, 0.0),
                    false,
                    false,
                    false,
                    gp,
                );
            });
            // `measure` already divides by 2*m*n*k; rescale to the complex flop count.
            let (kit, gem) = (sk.median * 2.0, sg.median * 2.0);
            let mode = if par { "par" } else { "ser" };
            println!(
                "  n={s:<5} {mode}  gemmkit={:7.1}  gemm={:7.1}  ({:.0}% of gemm)",
                kit,
                gem,
                100.0 * kit / gem.max(1e-9)
            );
            let _ = cflop;
        }
    }
}

// gemmkit best-ISA vs the `gemm` crate + `matrixmultiply` — external crates that do
// not build for wasm, so this bench (and its `perf_sgemm` caller) is gated off wasm.
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

// The native single-ISA token + microkernel tile, matching the production
// dispatch choice for this architecture (see `dispatch.rs`). Used by the
// equal-ISA comparison below so gemmkit and the `gemm` crate run the same ISA.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type NativeTok = Fma;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NATIVE_MR: usize = 2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NATIVE_NR: usize = 6;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NATIVE_LABEL: &str = "FMA";

#[cfg(target_arch = "aarch64")]
type NativeTok = Neon;
#[cfg(target_arch = "aarch64")]
const NATIVE_MR: usize = 4;
#[cfg(target_arch = "aarch64")]
const NATIVE_NR: usize = 4;
#[cfg(target_arch = "aarch64")]
const NATIVE_LABEL: &str = "NEON";

// wasm `simd128` (compile-time feature; no runtime detection)
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
type NativeTok = Simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const NATIVE_MR: usize = 2;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const NATIVE_NR: usize = 4;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const NATIVE_LABEL: &str = "simd128";

/// Equal-ISA comparison: gemmkit's native single-ISA path (forced via the
/// driver) vs gemm's default (the same ISA on stable). Single-threaded,
/// column-major.
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

/// wasm `simd128`, column-major, single-threaded
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn bench_simd128_vs_scalar(s: usize) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let mut ws = Workspace::new();

    let s_simd = measure(m, k, n, || unsafe {
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
    let s_scalar = measure(m, k, n, || unsafe {
        driver::run::<FloatGemm<f32>, ScalarTok, 4, 4>(
            ScalarTok,
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
    let label = NATIVE_LABEL;
    println!(
        "  n={s:<5} ser  gemmkit-{label}={:7.2} (±{:>2.0}%)  scalar={:7.2} (±{:>2.0}%)  ({:.2}×)",
        s_simd.median,
        s_simd.spread_pct(),
        s_scalar.median,
        s_scalar.spread_pct(),
        s_simd.median / s_scalar.median,
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_simd128() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nwasm simd128 GFLOP/s (f32, column-major) — gemmkit simd128 vs scalar token:");
    for &s in &[128usize, 256, 512, 1024] {
        bench_simd128_vs_scalar(s);
    }
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

// ---------------------------------------------------------------------------
// Parallel thread-scaling diagnostic (the mid-size-parallel gap)
// ---------------------------------------------------------------------------

/// The (MR, NR) tile the default `gemm()` dispatch uses on this target — used
/// only to *estimate* the per-region job count (the parallel work granularity).
/// Assumes the best available x86 ISA is AVX-512; if the box only has AVX2 the
/// real tile is 16x6 and the printed job estimate is a lower bound.
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
/// size across thread counts, so we can see *where* scaling breaks: poor speedup
/// already at 2-4 threads => per-call fork/join + atomics overhead dominates the
/// tiny work; a plateau after 8-16 => memory bandwidth or job starvation (compare
/// against the printed ~jobs/region). Throughput is the median of `REPS`
/// calibrated batches; the spread column flags differences smaller than the noise.
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
    // to the same single-worker path), so reuse them instead of re-measuring.
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
        // the per-region job count), not the requested t — else eff% reads low
        // where n_jobs throttles below t and masquerades as a bandwidth wall.
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
    // (cbrt(mnk).div_ceil(stride), capped) for sizes above the serial gate.
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

// ---------------------------------------------------------------------------
// Prepacked-RHS reuse
// ---------------------------------------------------------------------------

/// Per-call throughput of a reused prepacked B (`gemm_packed_b`) vs plain `gemm`
/// (which re-reads / re-packs B every call) for a fixed `(k, n)` B and a varying
/// `m` (the activation batch). `b_row_major` is the strided case: plain gemm reads
/// B with a large K-stride each call and, below `m > 2048`, never packs it, so the
/// contiguous prepacked panel should win per call. `colB` is the control. The win
/// is the per-call speedup (the one-time pack amortizes away over many calls).
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
/// what a faster gather-pack could recover. Small `n` keeps A-packing unamortized.
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
/// the pack gate (transposed `m = n > 2048`), never packs it — so the contiguous
/// prepacked panel should win per call. Row-major A is the contiguous control.
fn bench_prepack_lhs(m: usize, k: usize, n: usize, a_col_major: bool, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let (ars, acs) = if a_col_major {
        (1, m as isize)
    } else {
        (k as isize, 1)
    };
    // Row-major C: the supported orientation for the prepacked-LHS path.
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
            // (fixed) A every call — the case prepacking should win most.
            for &n in &[128usize, 512, 1024, 2048, 4096, 6144] {
                bench_prepack_lhs(1024, 1024, n, acm, par);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared-LHS A-pack gate calibration
// ---------------------------------------------------------------------------

/// Force the shared-LHS A-pack gate **on vs off back-to-back in one process** (via
/// the runtime setter, so the same buffers/thread-pool are reused and machine drift
/// cancels) and report the parallel throughput of each. The gate only changes
/// behavior on the packed-A path: a row-major A (`rsa != 1`) always packs, so every
/// size exercises the pre-pass; a column-major A packs only once its K-walk stride
/// trips the TLB gate (large `m`), so its crossover sits higher. The `on % of off`
/// column is the signal — above 100% the shared pre-pass wins, below it regresses.
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
