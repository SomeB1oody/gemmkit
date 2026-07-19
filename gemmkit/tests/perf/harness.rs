//! Shared plumbing for the perf suite: the cross-file `BENCH_GUARD` serialization lock,
//! deterministic input generation, the GFLOP/s and GB/s throughput estimators, and each
//! target's native ISA token for the equal-ISA comparisons in `sgemm.rs`/`simd128.rs`

use std::time::Instant;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::Fma;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use gemmkit::simd::Simd128;

/// Mutex every `#[ignore]` bench in this test binary locks before running. The default
/// multi-threaded test harness would otherwise execute several core-saturating benches
/// concurrently, making every GFLOP/s or GB/s figure meaningless. Poisoning is ignored: a
/// panicking bench must not wedge the rest of the suite
pub(crate) static BENCH_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn fill(n: usize, seed: u64) -> Vec<f32> {
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

/// Repetitions per throughput sample: the estimator reports the median of this many
/// calibrated batches
const REPS: usize = 9;
/// Target wall time per calibrated batch, in seconds: `measure`/`measure_gbps` scale the
/// inner iteration count so each of the `REPS` batches takes about this long
const BATCH_SECS: f64 = 0.07;

/// 1 throughput sample from `measure`/`measure_gbps`: the median rate across `REPS`
/// batches, plus the min/max so run-to-run spread is visible instead of hidden behind a
/// single number
pub(crate) struct Stat {
    /// Median GFLOP/s (or GB/s) across the `REPS` batches
    pub(crate) median: f64,
    /// Minimum observed batch rate
    pub(crate) min: f64,
    /// Maximum observed batch rate
    pub(crate) max: f64,
}

impl Stat {
    pub(crate) fn spread_pct(&self) -> f64 {
        100.0 * (self.max - self.min) / self.median.max(1e-9)
    }
}

/// GFLOP/s estimator for a GEMM-shaped `m x k x n` workload: 3 warmup calls, 1 timed call
/// to calibrate how many iterations fill `BATCH_SECS`, then `REPS` such batches, each
/// converted to GFLOP/s and reported as the batch median (with min/max for spread).
/// Steadier than a single fixed-iteration timing, since scheduler jitter and cache-state
/// noise average out over many short batches instead of landing in one long one
pub(crate) fn measure<F: FnMut()>(m: usize, k: usize, n: usize, mut f: F) -> Stat {
    for _ in 0..3 {
        f();
    } // warmup + rayon thread-pool spin-up
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

/// GB/s estimator for a bandwidth-bound shape: the same warmup/calibration/`REPS`
/// machinery as [`measure`], but each batch converts elapsed time into `bytes / secs / 1e9`
/// instead of a flop count. Not a thin wrapper over `measure`: that function has already
/// folded the flop formula into its result by the time it returns a `Stat`, so there is no
/// way to rescale a GFLOP/s figure into GB/s after the fact. `bytes` is the total traffic
/// moved by one call to `f`
pub(crate) fn measure_gbps<F: FnMut()>(bytes: usize, mut f: F) -> Stat {
    for _ in 0..3 {
        f();
    } // warmup + rayon thread-pool spin-up
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

// Per-target native SIMD token, plus the (MR, NR) microtile it drives `driver::run` with in
// the equal-ISA benches. `NATIVE_MR` is expressed in whole SIMD registers (`driver::run`'s
// `MR_REG` parameter), not output rows: the effective row tile is `NATIVE_MR * S::LANES`,
// which is how these numbers line up with the `mr` each target's production dispatch table
// carries (e.g. x86 FMA: 2 registers * 8 f32 lanes = the same 16-row tile `select_f32` uses)
// On x86 the token itself is pinned to `Fma`, not the best ISA this CPU may support: this
// crate's `gemm` dev-dependency has no `nightly` feature enabled, so its AVX-512 path (gated
// behind unstable `stdarch` features) is unreachable here, and forcing gemmkit down to
// AVX2+FMA is what makes the comparison against `gemm`'s stable-Rust default apples-to-apples
// aarch64 has only 1 SIMD ISA (NEON) to begin with, so no such gap exists there
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) type NativeTok = Fma;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const NATIVE_MR: usize = 2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const NATIVE_NR: usize = 6;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const NATIVE_LABEL: &str = "FMA";

#[cfg(target_arch = "aarch64")]
pub(crate) type NativeTok = Neon;
#[cfg(target_arch = "aarch64")]
pub(crate) const NATIVE_MR: usize = 4;
#[cfg(target_arch = "aarch64")]
pub(crate) const NATIVE_NR: usize = 4;
#[cfg(target_arch = "aarch64")]
pub(crate) const NATIVE_LABEL: &str = "NEON";

// wasm `simd128` is a compile-time target feature (`-C target-feature=+simd128`), not
// something runtime-detected, so there is no auto-select ladder to match here the way the
// x86 comment above describes: this token is simply the only SIMD ISA the build has
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) type NativeTok = Simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_MR: usize = 2;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_NR: usize = 4;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_LABEL: &str = "simd128";
