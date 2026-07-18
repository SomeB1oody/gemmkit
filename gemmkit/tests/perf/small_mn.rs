//! Small-m,n horizontal (inner-product) route benches

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// small-matrix horizontal (inner-product) route: perf_small_mn

/// Force the horizontal / small_k / driver route for a `gemm` call by pinning the 2 gates,
/// run `f`, then restore. `small_mn_dim = MAX` + `small_k_threshold = 0` sends every small-m,n
/// shape to the horizontal path (its gate needs `k > small_k_threshold`, so drop the latter to
/// 0); `small_mn_dim = 0` + `small_k_threshold = MAX` forces small_k; both `0` forces the driver
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

/// `gemm`-crate / `matrixmultiply` GFLOP/s for a small-`m,n` f32 `C = A*B` in the horizontal
/// path's target layout (`row_major_a` ? row-major A : col-major A; col-major B; col-major C).
/// Native-only. Returns `(gemm, Option<mm>)`; mm is serial-only
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
    // A: row-major (col_stride 1, row_stride k) or col-major (col_stride m, row_stride 1)
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
/// baselines: all GFLOP/s, back-to-back over the same buffers so drift cancels. `row_major_a`
/// selects the horizontal path's contiguous-`k` fast-path layout (row-major A, col-major B) vs
/// col-major A (its strided fallback)
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

/// Layout-coverage probe: the horizontal path requires `csa == 1 && rsb == 1` (row-major A,
/// col-major B) after orientation, so an all-row-major or all-col-major small-`m,n` shape misses
/// that predicate. At default settings (`small_mn_pack_min_k`) a long-`k` miss now takes the pack
/// tier instead of falling to the driver / small_k, so this measures how close the pack tier's
/// `all_rm`/`all_cm` rate lands to the eligible layout's horizontal rate (the ceiling), the residual
/// gap being the `~1/m`/`~1/n` copy tax the pack tier could not amortize away
#[cfg(not(target_family = "wasm"))]
fn bench_small_mn_layouts(m: usize, n: usize, k: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    // Eligible: row-major A, col-major B (the horizontal route)
    let horiz = with_route(usize::MAX, 0, || {
        measure(m, k, n, || {
            gemm(
                1.0,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    });
    // All-row-major: B fails `rsb == 1`, so at defaults it takes the pack tier, not small_k / driver
    let all_rm = measure(m, k, n, || {
        gemm(
            1.0,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_row_major(&b, k, n),
            0.0,
            MatMut::from_row_major(&mut c, m, n),
            par,
        );
    });
    // All-col-major: A fails `csa == 1` post-swap
    let all_cm = measure(m, k, n, || {
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
        "  {m:>2}x{n:<2} k={k:<6} {mode}  horiz={:7.1}  all_rm={:7.1} ({:.2}x h)  all_cm={:7.1} ({:.2}x h)",
        horiz.median,
        all_rm.median,
        horiz.median / all_rm.median.max(1e-9),
        all_cm.median,
        horiz.median / all_cm.median.max(1e-9),
    );
}

/// Pack-tier crossover probe: for an **ineligible** (all-col-major, so A fails `csa == 1`) small
/// shape, compare the new packed-horizontal route (pack A into `k`-contiguous scratch, then the
/// horizontal dot) against the current fallback (the register-tiling driver) and the small_k route
/// across `k`. This is what sets `small_mn_pack_min_k`: the packed route must beat the driver at
/// (and above) the gate. `packed` forces the pack tier (`small_mn_dim = MAX`, `pack_min_k = 0`,
/// `small_k = 0`); `driver` forces the driver (`small_mn_dim = 0`, `small_k = 0`); `small_k` forces
/// the in-place small_k route (`small_mn_dim = 0`, `small_k = MAX`). The `xd` ratio is packed/driver
#[cfg(not(target_family = "wasm"))]
fn bench_small_mn_pack_crossover(m: usize, n: usize, k: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    // All-col-major: A fails `csa == 1` post-orientation, so it takes the pack tier when enabled
    let mut run = || {
        measure(m, k, n, || {
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
    // pack tier: small_mn_dim = MAX + small_k = 0 route small-m,n here; pack_min_k = 0 clears the
    // pack-tier k gate for every measured k
    let pmnk = gemmkit::tuning::small_mn_pack_min_k();
    gemmkit::tuning::set_small_mn_pack_min_k(0);
    let packed = with_route(usize::MAX, 0, &mut run);
    gemmkit::tuning::set_small_mn_pack_min_k(pmnk);
    let driver = with_route(0, 0, &mut run);
    let smallk = with_route(0, usize::MAX, &mut run);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  {m:>2}x{n:<2} k={k:<6} {mode}  packed={:7.1}  driver={:7.1} ({:.2}x d)  small_k={:7.1}",
        packed.median,
        driver.median,
        packed.median / driver.median.max(1e-9),
        smallk.median,
    );
}

/// Small-`m,n` pack-tier probe, two views of the same tier. View 1 (`bench_small_mn_layouts`, at
/// shipped defaults): how close the pack tier's ineligible-layout rate (`all_rm` = pack-B,
/// `all_cm` = pack-A) lands to the eligible horizontal rate (the ceiling), the residual gap being
/// the copy tax. View 2 (`bench_small_mn_pack_crossover`, forced routes): the packed-horizontal
/// route vs the driver / small_k swept across `k` for ineligible (all-col-major) shapes, which is
/// what sets `small_mn_pack_min_k` (the gate sits where `packed` overtakes the driver). Both live
/// in one probe because they measure the same tier from opposite ends: recovery-at-default and the
/// crossover that fixes the gate
#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_small_mn_pack() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!(
        "\nsmall-m,n pack tier — view 1: recovery vs the eligible horizontal ceiling (defaults):"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &(m, n, k) in &[(4usize, 4usize, 65536usize), (8, 8, 65536), (16, 16, 16384)] {
            bench_small_mn_layouts(m, n, k, par);
        }
    }
    println!(
        "\nsmall-m,n pack tier — view 2: crossover (packed-horizontal vs driver, ineligible all-col-major):"
    );
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        for &(m, n) in &[(4usize, 4usize), (8, 8), (16, 16)] {
            for &k in &[32usize, 64, 256, 1024, 4096, 16384] {
                bench_small_mn_pack_crossover(m, n, k, par);
            }
        }
    }
}

/// `perf_small_mn` row for **f16** (f32-accumulate mixed horizontal kernel): the horizontal path
/// vs the register-tiling driver, plus the `gemm` crate (same f16-in-f32-acc convention), all
/// GFLOP/s in the fast-path layout (row-major A, col-major B). Confirms the widen-load horizontal
/// path beats the driver's padded microtile the same way f32 does
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
/// the horizontal route widens bf16->f32 like f16 does, so the `xh` ratio measures the widen
/// route against the VNNI driver (a different, faster kernel than the f16 widen driver). No
/// `gemm`-crate bf16 support, so it is horiz-vs-driver only
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

/// `perf_small_mn` row for **i8 -> i32** (widen horizontal dot vs the register-tiling driver). On
/// this box the driver is the `vpdpbusd` VNNI dot kernel serial (the widen kernel small-parallel),
/// so the `xh` ratio measures the widen horizontal route against whichever driver kernel is picked.
/// The horizontal path widens `i8 -> i32` on load and reduces in `i32`; no `gemm`-crate i8 baseline
/// (0.18 lacks it), so it is horiz-vs-small_k-vs-driver only. Fast-path layout: row-major A,
/// col-major B, col-major C32
#[cfg(all(feature = "int8", not(target_family = "wasm")))]
fn bench_small_mn_i8(m: usize, n: usize, k: usize, par: Parallelism) {
    let a: Vec<i8> = (0..m * k).map(|i| (i % 17) as i8 - 8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 13) as i8 - 6).collect();
    let mut c = vec![0i32; m * n];
    let mut run = || {
        measure(m, k, n, || {
            gemmkit::gemm_i8(
                1,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0,
                MatMut::from_col_major(&mut c, m, n),
                par,
            );
        })
    };
    let horiz = with_route(usize::MAX, 0, &mut run);
    let smallk = with_route(0, usize::MAX, &mut run);
    let driver = with_route(0, 0, &mut run);
    let mode = if matches!(par, Parallelism::Serial) {
        "ser"
    } else {
        "par"
    };
    println!(
        "  {m:>2}×{n:<2} k={k:<5} {mode}  horiz={:7.1}  small_k={:7.1}  driver={:7.1} ({:.2}× h)",
        horiz.median,
        smallk.median,
        driver.median,
        horiz.median / driver.median.max(1e-9),
    );
}

/// Small-matrix horizontal (inner-product) route: small `m,n`, long `k`. Sweeps the output
/// dimensions against the contraction, forcing each of the 3 gemmkit routes (horizontal /
/// small_k / driver) plus the `gemm`-crate and `matrixmultiply` baselines. The crossover (where
/// the driver catches up as `m,n` grow) is visible in the `xh` (driver-over-horizontal) ratio
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
        // A couple of non-square small shapes
        for &(m, n) in &[(2usize, 8usize), (8, 2), (4, 16), (16, 4)] {
            for &k in &[256usize, 4096] {
                bench_small_mn(m, n, k, true, par);
            }
        }
    }
    // The route needs A rows / B cols unit-stride along `k`; a col-major A (strided along `k`)
    // would force a scalar dot that loses to the driver's packed microkernel, so the dispatch
    // gate excludes it and those shapes stay on the driver

    #[cfg(feature = "int8")]
    {
        println!("\n  i8 -> i32 (widen horizontal dot vs vpdpbusd/widen driver):");
        for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
            for &(m, n, k) in &[
                (4usize, 4usize, 65536usize),
                (8, 8, 65536),
                (16, 16, 65536),
                (16, 16, 4096),
            ] {
                bench_small_mn_i8(m, n, k, par);
            }
        }
    }

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
