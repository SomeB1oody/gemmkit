//! Small-`m,n`, long-`k` shapes: the horizontal inner-product route against the small-`k`
//! route and the register-tiling driver, its pack tier for layouts that miss the route's
//! unit-stride predicate, and the f16/bf16/i8 twins of the same comparison

use crate::harness::{BENCH_GUARD, fill, measure};
use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// Small-matrix horizontal (inner-product) route: perf_small_mn

/// Runs `f` with the horizontal / small-`k` / driver route forced via the 2 gates that
/// together decide it, then restores both. `small_mn_dim = MAX` plus `small_k_threshold = 0`
/// routes every small-`m,n` shape to the horizontal path (it also needs `k` above the small-k
/// floor, so dropping that floor to 0 clears it unconditionally); `small_mn_dim = 0` plus
/// `small_k_threshold = MAX` forces the small-k route instead; `0` for both forces the
/// general driver
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

/// `gemm`-crate / `matrixmultiply` GFLOP/s for a small-`m,n` f32 `C = A*B`, laid out the way
/// the horizontal route's fast path wants it (`row_major_a` selects row-major vs column-major
/// A; B and C are always column-major). Native-only. Returns `(gemm, Option<matrixmultiply>)`;
/// the latter is `None` on the parallel arm, since `matrixmultiply` is only compared serially
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

/// 1 `perf_small_mn` row: the horizontal path, the small-`k` route, and the register-tiling
/// driver, forced in turn on the same buffers, plus the `gemm`-crate and `matrixmultiply`
/// baselines, all reported as GFLOP/s so drift across the back-to-back runs cancels.
/// `row_major_a` selects the horizontal route's contiguous-`k` fast-path layout (row-major A,
/// column-major B) versus column-major A, its strided fallback
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

/// Layout-coverage probe: the horizontal route's zero-copy gate needs `csa == 1 && rsb == 1`
/// (column-major-along-`k` A rows and column-major-along-`k` B columns, i.e. row-major A
/// with column-major B), so an all-row-major or all-col-major small-`m,n` shape misses it.
/// At the shipped defaults, a long-`k` miss like that no longer falls all the way back to the
/// driver / small-k route: it takes the pack tier instead (copy just the failing operand into
/// `k`-contiguous scratch, then run the same horizontal kernel). This measures how close that
/// recovery gets to the zero-copy horizontal rate (the ceiling neither ineligible layout can
/// beat), the remaining gap being the pack tier's own `~1/m` or `~1/n` amortized copy cost
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
    // All-row-major: B fails rsb == 1, so at defaults it takes the pack tier (pack B), not
    // the small_k route or the driver
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
    // All-col-major: A fails csa == 1 (pack tier packs A instead)
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

/// Pack-tier crossover probe: for an **ineligible**, all-col-major shape (A fails
/// `csa == 1`), compares the pack tier (copy A into `k`-contiguous scratch, then the
/// horizontal dot) against the register-tiling driver and the small-k route across a `k`
/// sweep. This is the data behind `small_mn_pack_min_k`: the shipped gate must sit at (or
/// past) the `k` where `packed` first overtakes `driver`, never before it. `packed` forces
/// the pack tier by pairing `small_mn_dim = MAX` with `pack_min_k = 0` (clearing the pack
/// tier's own `k` gate so every measured `k` takes it) and `small_k = 0`; `driver` and
/// `small_k` force the other 2 routes the same way [`with_route`] always does. The printed
/// `xd` ratio is packed/driver
#[cfg(not(target_family = "wasm"))]
fn bench_small_mn_pack_crossover(m: usize, n: usize, k: usize, par: Parallelism) {
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    // All-col-major: A fails csa == 1 post-orientation, so it takes the pack tier when enabled
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

/// Small-`m,n` pack-tier coverage, from 2 angles on the same tier. View 1
/// ([`bench_small_mn_layouts`], shipped defaults): how close the pack tier's recovery on an
/// ineligible layout lands to the eligible horizontal ceiling. View 2
/// ([`bench_small_mn_pack_crossover`], every route forced in turn): where the pack tier
/// itself first beats the driver as `k` grows, for an ineligible shape, which is what
/// `small_mn_pack_min_k` is calibrated against. 1 test, since both views are needed together
/// to make the case for wherever the gate ends up: recovery at the shipped default, and the
/// crossover data that justifies it
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

/// `perf_small_mn` row for **f16** (the f32-accumulate mixed horizontal kernel): the
/// horizontal route against the register-tiling driver, plus the `gemm` crate (same
/// f16-in-f32-accumulate convention), in the fast-path layout (row-major A, column-major B).
/// Confirms the widen-load horizontal path beats the driver's padded microtile the same way
/// it does for f32
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

/// `perf_small_mn` row for **bf16**. On x86 the driver auto-selects the `vdpbf16ps` dot
/// kernel, while the horizontal route still widens bf16 to f32 the same way it does for f16,
/// so the printed `xh` ratio here compares the widen route against a different, faster driver
/// kernel than the f16 row above does. No `gemm`-crate bf16 support to compare against, so
/// this is horizontal-vs-driver only
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

/// `perf_small_mn` row for **i8 -> i32** (the widen horizontal dot against the register-tiling
/// driver). On a VNNI-capable box the driver here is `vpdpbusd` in the serial arm; every
/// parallel row's `m*n*k` stays well under the `i8_vnni_min_par_mnk` gate, so the parallel arm
/// always falls back to the plain widen kernel instead. The printed `xh` ratio therefore
/// compares against whichever of the 2 the driver picked for that row. The horizontal path
/// itself always widens `i8 -> i32` on load and reduces in `i32`, with no `gemm`-crate i8
/// baseline to compare against (0.18 has none). Fast-path layout: row-major A, column-major B,
/// column-major C (i32)
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
/// dimensions against the contraction depth, forcing each of gemmkit's 3 routes (horizontal /
/// small-k / driver) in turn, plus the `gemm`-crate and `matrixmultiply` baselines. The
/// crossover where the driver catches up to the horizontal route as `m,n` grow shows up
/// directly in the `xh` (driver-over-horizontal) ratio
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
    // The route needs A's rows and B's columns unit-stride along k; a column-major A (strided
    // along k) would force a scalar dot that loses to the driver's packed microkernel, so the
    // dispatch gate excludes it and those shapes stay on the driver instead

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
