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
use gemmkit::{MatMut, MatRef, Parallelism, gemm, gemm_batched};

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

/// Byte-for-byte sibling of [`measure`] for **bandwidth-bound** shapes: same warmup,
/// batch calibration, REPS, and median machinery, but each batch reports moved-bytes
/// throughput `bytes / secs / 1e9` (GB/s) instead of GFLOP/s. `measure` has already
/// divided by seconds, so a GFLOP/s `Stat` cannot be post-scaled into GB/s — this is
/// the parallel estimator, not a wrapper. `bytes` is the total traffic of one `f()`.
fn measure_gbps<F: FnMut()>(bytes: usize, mut f: F) -> Stat {
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
        g.push(bytes as f64 / secs / 1e9);
    }
    g.sort_by(f64::total_cmp);
    Stat {
        median: g[REPS / 2],
        min: g[0],
        max: g[REPS - 1],
    }
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

/// Lean perf-neutrality probe (gemmkit only, no external baselines): measures the paths touched by
/// the runtime-knob promotion at a few representative shapes, so a before/after run confirms the
/// hoisted per-call knob read is free (it is one relaxed atomic load per call, never per element).
/// Covers the general driver (mc/nc/kc/kc_min/tiny-block knobs), a tiny shape, gemv register-block
/// (k_stream_max), the packed-LHS path (packed_oversample + the transpose packer's strip knob, both
/// hit by a row-major A), batched GEMM (seq_internal_bytes), and i8 (i8_vnni_min_par_mnk).
#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_knob_neutral() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nknob-indirection neutrality probe (gemmkit only):");

    // General register-tiling driver, serial + parallel.
    for &s in &[256usize, 512, 1024] {
        let a = fill(s * s, 1);
        let b = fill(s * s, 2);
        for &(tag, par) in &[("ser", Parallelism::Serial), ("par", Parallelism::Rayon(0))] {
            let mut c = vec![0.0f32; s * s];
            let st = measure(s, s, s, || {
                gemm(
                    1.0,
                    MatRef::from_col_major(&a, s, s),
                    MatRef::from_col_major(&b, s, s),
                    0.0,
                    MatMut::from_col_major(&mut c, s, s),
                    par,
                );
            });
            println!(
                "  sgemm  {tag} s={s:<5} {:8.1} GFLOP/s (±{:>2.0}%)",
                st.median,
                st.spread_pct()
            );
        }
    }

    // Tiny shape (tiny-block branch + its kc ceiling).
    {
        let (m, k, n) = (48usize, 512usize, 48usize);
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
                Parallelism::Serial,
            );
        });
        println!(
            "  tiny   ser {m}x{k}x{n} {:8.1} GFLOP/s (±{:>2.0}%)",
            st.median,
            st.spread_pct()
        );
    }

    // gemv register-block (k_stream_max), serial, output spilling L2.
    {
        let (m, k) = (65536usize, 64usize);
        let a = fill(m * k, 1);
        let x = fill(k, 2);
        let mut y = vec![0.0f32; m];
        let bytes = (m * k + k + m) * 4;
        let st = measure_gbps(bytes, || {
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&x, k, 1),
                0.0,
                MatMut::from_col_major(&mut y, m, 1),
                Parallelism::Serial,
            );
        });
        println!(
            "  gemv   ser m={m} k={k} {:8.1} GB/s   (±{:>2.0}%)",
            st.median,
            st.spread_pct()
        );
    }

    // Packed-LHS path (row-major A forces both the packed-block grain and the transpose packer).
    {
        let (m, k, n) = (2048usize, 256usize, 256usize);
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);
        let mut c = vec![0.0f32; m * n];
        let st = measure(m, k, n, || {
            gemm(
                1.0,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Rayon(0),
            );
        });
        println!(
            "  packA  par {m}x{k}x{n} {:8.1} GFLOP/s (±{:>2.0}%)",
            st.median,
            st.spread_pct()
        );
    }

    // Batched GEMM (seq_internal_bytes_per_worker; inert on x86 but exercises resolve_batch).
    {
        let (batch, m, k, n) = (64usize, 96usize, 96usize, 96usize);
        let a = fill(batch * m * k, 1);
        let b = fill(batch * k * n, 2);
        let mut c = vec![0.0f32; batch * m * n];
        let st = measure(batch * m, k, n, || {
            gemm_batched(
                batch,
                1.0,
                MatRef::new(&a, m, k, 1, m as isize),
                (m * k) as isize,
                MatRef::new(&b, k, n, 1, k as isize),
                (k * n) as isize,
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize),
                (m * n) as isize,
                Parallelism::Rayon(0),
            );
        });
        println!(
            "  batch  par b={batch} {m}x{k}x{n} {:8.1} GFLOP/s (±{:>2.0}%)",
            st.median,
            st.spread_pct()
        );
    }

    // i8 (i8_vnni_min_par_mnk fallback gate), serial + parallel.
    #[cfg(feature = "int8")]
    {
        let s = 512usize;
        let a: Vec<i8> = (0..s * s).map(|x| (x % 17) as i8 - 8).collect();
        let b: Vec<i8> = (0..s * s).map(|x| (x % 13) as i8 - 6).collect();
        for &(tag, par) in &[("ser", Parallelism::Serial), ("par", Parallelism::Rayon(0))] {
            let mut c = vec![0i32; s * s];
            let st = measure(s, s, s, || {
                gemmkit::gemm_i8(
                    1,
                    MatRef::from_col_major(&a, s, s),
                    MatRef::from_col_major(&b, s, s),
                    0,
                    MatMut::from_col_major(&mut c, s, s),
                    par,
                );
            });
            println!(
                "  i8     {tag} s={s} {:8.1} GFLOP/s (±{:>2.0}%)",
                st.median,
                st.spread_pct()
            );
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

// ---------------------------------------------------------------------------
// Bandwidth-bound shapes: gemv, gevv, and the STREAM ceiling they are judged against
// ---------------------------------------------------------------------------
//
// gemv (matrix·vector), gevv (rank-k / outer product), and skinny/low-k GEMM have
// arithmetic intensity ≈ O(1): each input byte feeds only a few flops, so the ceiling
// is *memory bandwidth*, not compute. The metric is therefore achieved GB/s as a
// fraction of what the machine's DRAM can sustain, not GFLOP/s. STREAM Triad is that
// ceiling — a single-core triad for the serial arm, an aggregate multi-core triad for
// the parallel arm (one core's bandwidth is far below aggregate DRAM bandwidth, so the
// serial ceiling would be the wrong yardstick for a threaded run).

/// STREAM array length in f32 elements: 64 Mi (256 MiB) ≫ any last-level cache, so the
/// kernels stream from DRAM and cannot be served from cache.
const STREAM_LEN: usize = 64 * 1024 * 1024;

/// Single-core STREAM Triad `a[i] = b[i] + α·c[i]` in GB/s (3·N·4 B moved). The scalar
/// bandwidth ceiling the *serial* gemv/gevv arms are measured against. `black_box`
/// around the output keeps the optimizer from eliding the streaming loop.
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

/// Single-core STREAM Copy `dst[i] = src[i]` in GB/s (2·N·4 B moved) — the read+write
/// bandwidth reference next to the triad.
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
/// a small fraction of the ~10 ms streaming time.
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
/// and can even regress past it (memory-controller contention), so the *peak* — not the
/// all-cores figure — is the honest ceiling for the parallel arm.
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

/// External-library GB/s for a column-major f32 `C(m×n) = A(m×k)·B(k×n)` (`beta = 0`) — the
/// same-shape baseline the gemv/gevv rows compare against: the `gemm` crate, plus (serial only)
/// `matrixmultiply`. Native-only (both are wasm-excluded dev-deps). Returns `(gemm, mm)`.
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
    // matrixmultiply is single-threaded as used here, so only compare it on the serial arm.
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

/// Format the external-baseline tail (`gemm=… (kit …×)  mm=… (kit …×)`) for a `kit`-GB/s
/// gemmkit result on the given shape. Empty on wasm (no external crate there).
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

/// gemv `C(m×1) = A(m×k)·x` through the public [`gemm`], reported as GB/s of the minimum
/// traffic `(m*k + k + m)*4` (A read once, x once, C written once) against the STREAM
/// `ceiling`, plus the `gemm`-crate / `matrixmultiply` GB/s on the same shape (`kit ×` is
/// gemmkit's speedup over each). `k` spans fits-L2 → DRAM; column-major A/C hit the axpy
/// form of the gemv path.
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
        // Fits-L2 → DRAM sweep: `out` (m·4 B) stays within L2 here, so the axpy form's
        // per-column re-read of `out` is cache-cheap and A's DRAM read dominates.
        for &k in &[64usize, 256, 1024] {
            for &m in &[1024usize, 8192, 65536] {
                bench_gemv(m, k, par, ceiling);
            }
        }
        // `out ∤ cache` sweep: at these `m`, `out` spills L2 (4 MiB) through past L3
        // (64 MiB), so the axpy form's `k` re-reads of `out` become real DRAM traffic —
        // the regime output register-blocking is meant to fix. `k` is kept small so A
        // stays ≤ 512 MiB.
        for &(m, k) in &[
            (1_048_576usize, 64usize),
            (4_194_304, 16),
            (8_388_608, 8),
            (16_777_216, 8),
            (16_777_216, 16), // out ∤ L3 with moderate k: probes A-stream prefetch thrash
        ] {
            bench_gemv(m, k, par, ceiling);
        }
    }
}

/// Investigation: for a gemv shape, measure the dedicated gemv special path vs the general
/// driver (reached by disabling the gemv path) vs the `gemm` crate. Confirms the special
/// path is the right choice — the driver, which packs A into micropanels for a compute-bound
/// kernel, is far slower on a bandwidth-bound gemv — and tracks the remaining gap to `gemm`.
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

/// gevv / skinny GEMM `C(m×n) = A(m×k)·B(k×n)` at small `k`, reported as GB/s of the
/// minimum traffic `(m*k + k*n + m*n)*4` (beta = 0, so C is write-only) against the STREAM
/// `ceiling`, plus the `gemm`-crate / `matrixmultiply` GB/s on the same shape (`kit ×` is
/// gemmkit's speedup). At tiny `k` the `m*n` C write dominates, so this is
/// write-bandwidth-bound.
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
/// there to hold: the auto row should sit on the plateau, not past it.
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
/// last `k` where it does.
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
    // A DRAM-bound axpy shape (out fits cache) and a register-blocked out-∤-L3 shape.
    bench_gemv_scaling(65536, 1024, ceiling, avail);
    bench_gemv_scaling(16_777_216, 8, ceiling, avail);
    // Cache-resident shapes (A fits L3): the ceiling here is aggregate cache bandwidth, which
    // keeps scaling with cores well past the DRAM-saturation count — so the auto row (capped
    // by the DRAM proxy) is expected to sit far below the forced-high-t peak.
    bench_gemv_scaling(1024, 1024, ceiling, avail);
    bench_gemv_scaling(8192, 64, ceiling, avail);
}

// ---- small-matrix horizontal (inner-product) route: perf_small_mn ----

/// Force the horizontal / small_k / driver route for a `gemm` call by pinning the two gates,
/// run `f`, then restore. `small_mn_dim = MAX` + `small_k_threshold = 0` sends every small-m,n
/// shape to the horizontal path (its gate needs `k > small_k_threshold`, so drop the latter to
/// 0); `small_mn_dim = 0` + `small_k_threshold = MAX` forces small_k; both `0` forces the driver.
#[cfg(not(target_family = "wasm"))]
fn with_route<R>(small_mn: usize, small_k: usize, f: impl FnOnce() -> R) -> R {
    let (pm, pk) = (
        gemmkit::tuning::small_mn_dim(),
        gemmkit::tuning::small_k_threshold(),
    );
    gemmkit::tuning::set_small_mn_dim(small_mn);
    gemmkit::tuning::set_small_k_threshold(small_k);
    let r = f();
    gemmkit::tuning::set_small_mn_dim(pm);
    gemmkit::tuning::set_small_k_threshold(pk);
    r
}

/// `gemm`-crate / `matrixmultiply` GFLOP/s for a small-`m,n` f32 `C = A·B` in the horizontal
/// path's target layout (`row_major_a` ? row-major A : col-major A; col-major B; col-major C).
/// Native-only. Returns `(gemm, Option<mm>)`; mm is serial-only.
#[cfg(not(target_family = "wasm"))]
fn extern_gflops_small(
    m: usize,
    k: usize,
    n: usize,
    row_major_a: bool,
    par: Parallelism,
) -> (f64, Option<f64>) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    // A: row-major (col_stride 1, row_stride k) or col-major (col_stride m, row_stride 1).
    let (a_cs, a_rs) = if row_major_a {
        (1isize, k as isize)
    } else {
        (m as isize, 1isize)
    };
    let gpar = if matches!(par, Parallelism::Serial) {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(0)
    };
    let g = measure(m, k, n, || unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            m as isize,
            1, // dst col-major
            false,
            a.as_ptr(),
            a_cs,
            a_rs,
            b.as_ptr(),
            k as isize,
            1, // rhs col-major
            0.0,
            1.0,
            false,
            false,
            false,
            gpar,
        );
    });
    let mm = matches!(par, Parallelism::Serial).then(|| {
        measure(m, k, n, || unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                a_rs,
                a_cs, // lhs (row_stride, col_stride)
                b.as_ptr(),
                1,
                k as isize, // rhs col-major (row_stride 1, col_stride k)
                0.0,
                c.as_mut_ptr(),
                1,
                m as isize, // dst col-major
            );
        })
        .median
    });
    (g.median, mm)
}

/// One `perf_small_mn` row: the horizontal path vs the small_k route vs the register-tiling
/// driver on a small-`m,n` / long-`k` shape, plus the `gemm`-crate and `matrixmultiply`
/// baselines — all GFLOP/s, back-to-back over the same buffers so drift cancels. `row_major_a`
/// selects the horizontal path's contiguous-`k` fast-path layout (row-major A, col-major B) vs
/// col-major A (its strided fallback).
#[cfg(not(target_family = "wasm"))]
fn bench_small_mn(m: usize, n: usize, k: usize, row_major_a: bool, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let mut run = || {
        measure(m, k, n, || {
            let av = if row_major_a {
                MatRef::from_row_major(&a, m, k)
            } else {
                MatRef::from_col_major(&a, m, k)
            };
            gemm(
                1.0,
                av,
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    };
    let horiz = with_route(usize::MAX, 0, &mut run);
    let smallk = with_route(0, usize::MAX, &mut run);
    let driver = with_route(0, 0, &mut run);
    let (g, mm) = extern_gflops_small(m, k, n, row_major_a, par);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    let mm_s = mm
        .map(|v| format!("  mm={v:6.1} ({:.2}×)", horiz.median / v.max(1e-9)))
        .unwrap_or_default();
    println!(
        "  {m:>2}×{n:<2} k={k:<5} {mode}  horiz={:7.1}  small_k={:7.1}  driver={:7.1} ({:.2}× h)  gemm={:6.1} ({:.2}× h){mm_s}",
        horiz.median,
        smallk.median,
        driver.median,
        horiz.median / driver.median.max(1e-9),
        g,
        horiz.median / g.max(1e-9),
    );
}

/// `perf_small_mn` row for **f16** (f32-accumulate mixed horizontal kernel): the horizontal path
/// vs the register-tiling driver, plus the `gemm` crate (same f16-in-f32-acc convention), all
/// GFLOP/s in the fast-path layout (row-major A, col-major B). Confirms the widen-load horizontal
/// path beats the driver's padded microtile the same way f32 does.
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_small_mn_f16(m: usize, n: usize, k: usize, par: Parallelism) {
    use gemmkit::f16;
    let to16 = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    let a = to16(&fill(m * k, 1));
    let b = to16(&fill(k * n, 2));
    let mut c = vec![f16::from_f32(0.0); m * n];
    let mut run = || {
        measure(m, k, n, || {
            gemm(
                f16::from_f32(1.0),
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                f16::from_f32(0.0),
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    };
    let horiz = with_route(usize::MAX, 0, &mut run);
    let driver = with_route(0, 0, &mut run);
    let gpar = if matches!(par, Parallelism::Serial) {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(0)
    };
    let g = measure(m, k, n, || unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            m as isize,
            1, // dst col-major
            false,
            a.as_ptr(),
            1,
            k as isize, // lhs row-major (col_stride 1, row_stride k)
            b.as_ptr(),
            k as isize,
            1, // rhs col-major
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            false,
            false,
            false,
            gpar,
        );
    });
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  {m:>2}×{n:<2} k={k:<5} {mode}  horiz={:7.1}  driver={:7.1} ({:.2}× h)  gemm={:6.1} ({:.2}× h)",
        horiz.median,
        driver.median,
        horiz.median / driver.median.max(1e-9),
        g.median,
        horiz.median / g.median.max(1e-9),
    );
}

/// `perf_small_mn` row for **bf16**. On x86 the driver takes the `vdpbf16ps` VNNI dot path while
/// the horizontal route widens bf16→f32 like f16 does, so the `×h` ratio measures the widen
/// route against the VNNI driver (a different, faster kernel than the f16 widen driver). No
/// `gemm`-crate bf16 support, so it is horiz-vs-driver only.
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_small_mn_bf16(m: usize, n: usize, k: usize, par: Parallelism) {
    use gemmkit::bf16;
    let to16 = |v: &[f32]| v.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>();
    let a = to16(&fill(m * k, 1));
    let b = to16(&fill(k * n, 2));
    let mut c = vec![bf16::from_f32(0.0); m * n];
    let mut run = || {
        measure(m, k, n, || {
            gemm(
                bf16::from_f32(1.0),
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                bf16::from_f32(0.0),
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    };
    let horiz = with_route(usize::MAX, 0, &mut run);
    let driver = with_route(0, 0, &mut run);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  {m:>2}×{n:<2} k={k:<5} {mode}  horiz={:7.1}  driver={:7.1} ({:.2}× h)",
        horiz.median,
        driver.median,
        horiz.median / driver.median.max(1e-9),
    );
}

/// Small-matrix horizontal (inner-product) route: small `m,n`, long `k`. Sweeps the output
/// dimensions against the contraction, forcing each of the three gemmkit routes (horizontal /
/// small_k / driver) plus the `gemm`-crate and `matrixmultiply` baselines. The crossover — where
/// the driver catches up as `m,n` grow — is visible in the `×h` (driver-over-horizontal) ratio.
#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_small_mn() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nsmall-m,n horizontal route (C[m×n]=A·B, small m,n, long k) — GFLOP/s, row-major A + col-major B (fast-path layout):"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &s in &[2usize, 4, 8, 16, 32] {
            for &k in &[64usize, 256, 1024, 4096] {
                bench_small_mn(s, s, k, true, par);
            }
        }
        // A couple of non-square small shapes.
        for &(m, n) in &[(2usize, 8usize), (8, 2), (4, 16), (16, 4)] {
            for &k in &[256usize, 4096] {
                bench_small_mn(m, n, k, true, par);
            }
        }
    }
    // The route needs A rows / B cols unit-stride along `k`; a col-major A (strided along `k`)
    // would force a scalar dot that loses to the driver's packed microkernel, so the dispatch
    // gate excludes it and those shapes stay on the driver.

    #[cfg(feature = "half")]
    {
        println!("\n  f16 (f32-accumulate mixed horizontal kernel):");
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            for &s in &[4usize, 8, 16, 32] {
                for &k in &[256usize, 4096] {
                    bench_small_mn_f16(s, s, k, par);
                }
            }
        }
        println!("\n  bf16 (widen horizontal path vs vdpbf16ps VNNI driver on x86):");
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            for &s in &[4usize, 8, 16, 32] {
                for &k in &[256usize, 4096] {
                    bench_small_mn_bf16(s, s, k, par);
                }
            }
        }
    }
}

// ---- batched GEMM: perf_batched ----

/// One `perf_batched` row: `gemm_batched` (auto) vs the two honest baselines — a naive serial loop
/// of `gemm(Serial)` and a naive parallel loop of `gemm(Rayon(0))` (per-element internal
/// parallelism) — over `batch` contiguously-packed column-major `m×k · k×n` elements, as **total**
/// GFLOP/s (`2·batch·m·k·n`). There is no batched entry in the `gemm` crate / `matrixmultiply`, so
/// the naive loops are the only honest baselines. Passing `m·batch` to `measure` folds the batch
/// into its `2·m·k·n` count so the reported figure is whole-batch throughput.
fn bench_batched(batch: usize, m: usize, k: usize, n: usize) {
    let a = fill(batch * m * k, 1);
    let b = fill(batch * k * n, 2);
    let mut c = vec![0.0f32; batch * m * n];

    let batched = measure(m * batch, k, n, || {
        gemm_batched(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            Parallelism::Rayon(0),
        );
    });
    let mut naive = |par| {
        measure(m * batch, k, n, || {
            for bi in 0..batch {
                let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
                gemm(
                    1.0,
                    MatRef::from_col_major(&a[ao..ao + m * k], m, k),
                    MatRef::from_col_major(&b[bo..bo + k * n], k, n),
                    0.0,
                    MatMut::from_col_major(&mut c[co..co + m * n], m, n),
                    par,
                );
            }
        })
    };
    let loop_ser = naive(Parallelism::Serial);
    let loop_par = naive(Parallelism::Rayon(0));
    println!(
        "  b={batch:<6} {m}×{k}×{n:<5}  batched={:8.1} (±{:>2.0}%)  loop_ser={:8.1} ({:.2}×)  loop_par={:8.1} ({:.2}×)",
        batched.median,
        batched.spread_pct(),
        loop_ser.median,
        batched.median / loop_ser.median.max(1e-9),
        loop_par.median,
        batched.median / loop_par.median.max(1e-9),
    );
}

/// Batched GEMM throughput: `gemm_batched` (auto) vs a naive serial / naive parallel loop of
/// single `gemm()` calls, as total GFLOP/s. The batch-parallel schedule assigns whole GEMMs to
/// workers (each element serial, cache-hot), so the win over the naive parallel loop is avoiding
/// its per-element fork/join. The few-but-large tail (batch < cores) uses at most `batch` workers.
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_batched() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nbatched GEMM (total GFLOP/s) — gemm_batched(auto) vs naive serial/parallel gemm() loops:"
    );
    for &(m, k, n) in &[
        (4usize, 4usize, 4usize),
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 64),
        (4, 4, 1024),
    ] {
        for &batch in &[64usize, 1024, 16384] {
            bench_batched(batch, m, k, n);
        }
    }
    // Few but large: fewer elements than cores, so batch-parallel uses only `batch` workers.
    // A cache-resident element (256³) that scales poorly favors batch-parallelism; a bigger,
    // DRAM-touching element (512³) is where a per-element internal split would use more cores.
    println!("  few-but-large (batch < cores):");
    for &batch in &[4usize, 8] {
        bench_batched(batch, 256, 256, 256);
    }
    for &batch in &[2usize, 4] {
        bench_batched(batch, 512, 512, 512);
    }
}
