//! Fused-epilogue overhead: `gemm_fused` against plain `gemm` on the same shape and the
//! same ISA, so the ratio isolates what the epilogue seam costs
//!
//! Unlike the rest of this suite these benches assert, because the failure they guard
//! against is not a few percent of throughput. A fused kernel is a separate
//! monomorphization of the same generic body, so an epilogue can degrade the kernel itself
//! rather than just the store, if the compiler stops keeping the accumulator tile in
//! registers across the `kc` loop. That is invisible to every correctness test and to any
//! bench that measures fused paths alone, so the bound below is sized for that class of
//! collapse, not to police small regressions
//!
//! Self-relative by construction: both sides are gemmkit on the current machine and ISA, so
//! there is no absolute figure to go stale on other hardware. The sweep covers a large
//! square, a shallow-`k` shape (where the epilogue is least amortized), and a tall/skinny one

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, gemm, gemm_fused};

/// Ratio of fused to plain time above which the epilogue is judged to have broken the
/// kernel rather than added work to the store: set well above normal jitter, so it must
/// not fire on a loaded CI runner, and well below the multi-x slowdown a real collapse
/// produces
const MAX_FUSED_RATIO: f64 = 2.0;

/// `(m, k, n)` shapes: a large square, a shallow `k` (the epilogue's worst amortization),
/// and a tall/skinny one
const SHAPES: &[(usize, usize, usize)] = &[(512, 512, 512), (1024, 64, 1024), (4096, 128, 64)];

fn bench(par: Parallelism, tag: &str) {
    for &(m, k, n) in SHAPES {
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);
        let bias = fill(m, 3);
        let mut c = vec![0.0f32; m * n];

        let plain = measure(m, k, n, || {
            gemm(
                1.0,
                MatRef::new(&a, m, k, 1, m as isize),
                MatRef::new(&b, k, n, 1, k as isize),
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize),
                par,
            );
        });
        // Bias plus activation together: the epilogue with the most live state, and the
        // combination a real inference layer asks for
        let fused = measure(m, k, n, || {
            gemm_fused(
                1.0,
                MatRef::new(&a, m, k, 1, m as isize),
                MatRef::new(&b, k, n, 1, k as isize),
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize),
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
        });

        // measure() reports GFLOP/s, so the time ratio is the inverted rate ratio
        let ratio = plain.median / fused.median;
        println!(
            "{tag} {m:>5}x{k:<5}x{n:<5} plain {:8.1} GF/s (spread {:4.1}%)  \
             fused {:8.1} GF/s (spread {:4.1}%)  ratio x{ratio:.2}",
            plain.median,
            plain.spread_pct(),
            fused.median,
            fused.spread_pct(),
        );
        assert!(
            ratio < MAX_FUSED_RATIO,
            "{tag} {m}x{k}x{n}: fused/plain time ratio x{ratio:.2} exceeds \
             x{MAX_FUSED_RATIO:.2}; the epilogue is degrading the kernel, not just the \
             store (see the module comment)"
        );
    }
}

/// Fused vs plain across every shape in `SHAPES`, serial: asserts the ratio stays under
/// `MAX_FUSED_RATIO`
#[test]
#[ignore = "benchmark"]
fn perf_fused_overhead_serial() {
    let _g = BENCH_GUARD.lock();
    bench(Parallelism::Serial, "ser");
}

/// Fused vs plain across every shape in `SHAPES`, parallel: asserts the ratio stays under
/// `MAX_FUSED_RATIO`
#[test]
#[ignore = "benchmark"]
fn perf_fused_overhead_parallel() {
    let _g = BENCH_GUARD.lock();
    bench(Parallelism::Rayon(0), "par");
}
