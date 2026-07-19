//! wasm `simd128` throughput vs the scalar token, the wasm target's substitute for the
//! native-vs-`gemm`-crate comparison the other perf benches run (no external GEMM crate
//! builds for wasm)

use crate::harness::{BENCH_GUARD, NATIVE_LABEL, NATIVE_MR, NATIVE_NR, NativeTok, fill, measure};
use gemmkit::kernel::FloatGemm;
use gemmkit::simd::ScalarTok;
use gemmkit::{Parallelism, Workspace, driver};

/// `simd128` vs `ScalarTok`, both driven directly through `driver::run` (bypassing
/// dispatch, so each token's own microtile geometry is forced rather than auto-selected):
/// single-threaded, column-major, square `m = k = n = s`. `ScalarTok` always uses a 4x4
/// tile; `simd128` uses `NATIVE_MR`/`NATIVE_NR` from the harness
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
