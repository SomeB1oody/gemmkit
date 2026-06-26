//! Quick performance comparison vs the `gemm` crate and `matrixmultiply`.
//! Ignored by default (it is a benchmark, not a correctness gate). Run with:
//!   cargo test -p gemmkit --release --test perf -- --ignored --nocapture

use std::time::Instant;

use gemmkit::driver;
use gemmkit::kernel::FloatGemm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::Fma;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm};

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

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    } // warmup
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

fn gflops(m: usize, k: usize, n: usize, secs: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / secs / 1e9
}

fn bench_one(s: usize, parallel: bool) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let iters = if s <= 512 { 20 } else { 5 };

    let par = if parallel {
        Parallelism::Rayon(0)
    } else {
        Parallelism::Serial
    };
    let t_kit = time(iters, || {
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
    let t_gemm = time(iters, || unsafe {
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

    let mm = if !parallel {
        Some(time(iters, || unsafe {
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
        }))
    } else {
        None
    };

    let g_kit = gflops(m, k, n, t_kit);
    let g_gemm = gflops(m, k, n, t_gemm);
    let mode = if parallel { "par" } else { "ser" };
    print!(
        "  n={s:<5} {mode}  gemmkit={g_kit:7.1}  gemm={g_gemm:7.1}  ({:.0}% of gemm)",
        100.0 * g_kit / g_gemm
    );
    if let Some(t_mm) = mm {
        let g_mm = gflops(m, k, n, t_mm);
        print!("  mm={g_mm:7.1}  ({:.2}x mm)", g_kit / g_mm);
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
const NATIVE_MR: usize = 3;
#[cfg(target_arch = "aarch64")]
const NATIVE_NR: usize = 8;
#[cfg(target_arch = "aarch64")]
const NATIVE_LABEL: &str = "NEON";

/// Equal-ISA comparison: gemmkit's native single-ISA path (forced via the
/// driver) vs gemm's default (the same ISA on stable). Single-threaded,
/// column-major.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn bench_native_equal_isa(s: usize) {
    let (m, k, n) = (s, s, s);
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let iters = if s <= 512 { 20 } else { 5 };
    let mut ws = Workspace::new();

    let t_kit = time(iters, || unsafe {
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
    let t_gemm = time(iters, || unsafe {
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
    let g_kit = gflops(m, k, n, t_kit);
    let g_gemm = gflops(m, k, n, t_gemm);
    let label = NATIVE_LABEL;
    println!(
        "  n={s:<5} ser  gemmkit-{label}={g_kit:7.1}  gemm-{label}={g_gemm:7.1}  ({:.0}% of gemm)",
        100.0 * g_kit / g_gemm
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_sgemm() {
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
