//! Shared bench harness: BENCH_GUARD, fill/measure/Stat, native-ISA token.

use std::time::Instant;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::Fma;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use gemmkit::simd::Simd128;

/// Serializes the two core-saturating `#[ignore]` benches so the default
/// multi-threaded test harness can't run them concurrently (which would make every
/// GFLOP/s figure meaningless). Poisoning is ignored — a panicking bench must not
/// wedge the other.
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

/// Reps and per-batch target for the robust estimator below.
const REPS: usize = 9;
const BATCH_SECS: f64 = 0.07;

/// A throughput sample: median GFLOP/s plus the min/max so run-to-run spread is
/// *visible* and tuning decisions are not made on noise.
pub(crate) struct Stat {
    pub(crate) median: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl Stat {
    pub(crate) fn spread_pct(&self) -> f64 {
        100.0 * (self.max - self.min) / self.median.max(1e-9)
    }
}

/// Robust throughput estimate: warm up, auto-calibrate the batch size to
/// ~`BATCH_SECS`, then report the median GFLOP/s (and spread) over `REPS`
/// batches. Far steadier than a single fixed-iter timing.
pub(crate) fn measure<F: FnMut()>(m: usize, k: usize, n: usize, mut f: F) -> Stat {
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
pub(crate) fn measure_gbps<F: FnMut()>(bytes: usize, mut f: F) -> Stat {
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

// The native single-ISA token + microkernel tile, matching the production
// dispatch choice for this architecture (see `dispatch.rs`). Used by the
// equal-ISA comparison below so gemmkit and the `gemm` crate run the same ISA.
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

// wasm `simd128` (compile-time feature; no runtime detection)
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) type NativeTok = Simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_MR: usize = 2;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_NR: usize = 4;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) const NATIVE_LABEL: &str = "simd128";
