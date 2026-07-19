//! Per-element-type throughput (f16, bf16, i8, c32) against whichever external baseline
//! actually supports that type: the `gemm` crate has f16 and c32, but neither bf16 nor i8, so
//! those 2 report gemmkit-only figures

#[cfg(feature = "half")]
use crate::harness::fill;
use crate::harness::{BENCH_GUARD, measure};
use gemmkit::Parallelism;
#[cfg(feature = "half")]
use gemmkit::gemm;
#[cfg(any(feature = "half", feature = "complex"))]
use gemmkit::{MatMut, MatRef};

/// f16 GEMM throughput: gemmkit's mixed kernel (f16 in, f32 accumulate, rounded back to f16 on
/// store) against the `gemm` crate's f16 support, which uses the same f16-in-f32-accumulate
/// convention. Flops are counted the same way as f32 (`2*m*k*n`) for both
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

/// f32 vs f16 vs bf16 GFLOP/s across a `k` sweep at fixed, small `m = n = 512`, all serial.
/// f32 always runs the driver's ordinary cache-blocked path, re-blocking at the analytic `kc`
/// regardless of how large `k` gets. f16/bf16 instead start out on the narrow-output
/// single-slice route (`OUT_IS_ACC = false`: the entire contraction as 1 depth panel, so the
/// narrow result rounds only once at the end), which on this target auto-switches itself to
/// an `f32`-output multi-slice twin (re-blocked at the same analytic `kc` f32 uses) once the
/// resulting RHS micropanel (`nr * k * sizeof(N)`) outgrows the deep-contraction engage gate.
/// On AVX-512 (`nr = 12`, gate = half the detected L2) that crossover sits around `k ~
/// 21800`, so the last 2 points below (32768, 65536) are already running the multi-slice
/// twin, not the single-slice route the first 3 exercise. The sweep is therefore really 2
/// questions in 1: does single-slice throughput hold up cleanly all the way to its own
/// hand-off point, and is the hand-off itself smooth rather than a visible step
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_narrow_k_sweep(m: usize, k: usize, n: usize) {
    use gemmkit::{bf16, f16};
    let af = fill(m * k, 1);
    let bf = fill(k * n, 2);
    let mut cf = vec![0.0f32; m * n];
    let s_f32 = measure(m, k, n, || {
        gemm(
            1.0f32,
            MatRef::from_col_major(&af, m, k),
            MatRef::from_col_major(&bf, k, n),
            0.0,
            MatMut::from_col_major(&mut cf, m, n),
            Parallelism::Serial,
        );
    });
    let a16: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = bf.iter().map(|&x| f16::from_f32(x)).collect();
    let mut c16 = vec![f16::from_f32(0.0); m * n];
    let s_f16 = measure(m, k, n, || {
        gemm(
            f16::from_f32(1.0),
            MatRef::from_col_major(&a16, m, k),
            MatRef::from_col_major(&b16, k, n),
            f16::from_f32(0.0),
            MatMut::from_col_major(&mut c16, m, n),
            Parallelism::Serial,
        );
    });
    let ab: Vec<bf16> = af.iter().map(|&x| bf16::from_f32(x)).collect();
    let bb: Vec<bf16> = bf.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut cb = vec![bf16::from_f32(0.0); m * n];
    let s_bf16 = measure(m, k, n, || {
        gemm(
            bf16::from_f32(1.0),
            MatRef::from_col_major(&ab, m, k),
            MatRef::from_col_major(&bb, k, n),
            bf16::from_f32(0.0),
            MatMut::from_col_major(&mut cb, m, n),
            Parallelism::Serial,
        );
    });
    println!(
        "  m=n={m:<4} k={k:<6} ser  f32={:7.1} (±{:>2.0}%)  f16={:7.1} (±{:>2.0}%)  bf16={:7.1} (±{:>2.0}%)",
        s_f32.median,
        s_f32.spread_pct(),
        s_f16.median,
        s_f16.spread_pct(),
        s_bf16.median,
        s_bf16.spread_pct()
    );
}

#[cfg(all(feature = "half", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_narrow_k_sweep() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nnarrow-output deep-k probe (single-slice kc = k vs the f32 control):");
    for &k in &[512usize, 4096, 16384, 32768, 65536] {
        bench_narrow_k_sweep(512, k, 512);
    }
}

/// bf16-in, f32-accumulate GEMM throughput: no baseline here since the `gemm` crate (0.18) has
/// no bf16 support to compare against. On an AVX-512-BF16 box the driver auto-selects the
/// `vdpbf16ps` dot kernel, whose LHS/RHS packs both go through `pack_kgroup_panels`;
/// row-major A (contiguous rows) plus col-major B (contiguous columns) is the layout where
/// both operands hit that packer's fast, straight-copy path rather than its strided one, so
/// this exercises the kernel at its best. Flops counted the same way as f32. A gemmkit-only
/// sibling of [`bench_f16`]: same shape, no crate comparison to print
#[cfg(all(feature = "half", not(target_family = "wasm")))]
fn bench_bf16(s: usize, parallel: bool) {
    use gemmkit::bf16;
    let (m, k, n) = (s, s, s);
    let to16 = |v: &[f32]| v.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>();
    let a = to16(&fill(m * k, 1));
    let b = to16(&fill(k * n, 2));
    let mut c = vec![bf16::from_f32(0.0); m * n];

    let par = if parallel {
        Parallelism::Rayon(0)
    } else {
        Parallelism::Serial
    };
    let st = measure(m, k, n, || {
        gemm(
            bf16::from_f32(1.0),
            MatRef::from_row_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            bf16::from_f32(0.0),
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
    });
    let mode = if parallel { "par" } else { "ser" };
    println!(
        "  n={s:<5} {mode}  gemmkit={:7.1} (±{:>2.0}%)",
        st.median,
        st.spread_pct()
    );
}

#[cfg(all(feature = "half", not(target_family = "wasm")))]
#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn perf_bf16() {
    let _guard = BENCH_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    println!("\nbf16->f32 GFLOP/s (row-major A, col-major B) — gemmkit vdpbf16ps dot kernel:");
    for &par in &[false, true] {
        for &s in &[384usize, 1024] {
            bench_bf16(s, par);
        }
    }
}

/// i8-in, i32-accumulate GEMM throughput: no baseline here either (0.18 has no i8 support).
/// On a VNNI-capable box `gemm_i8`'s auto-select mostly runs the `vpdpbusd` dot kernel here
/// (every serial call, and every parallel call whose `m*n*k` clears `i8_vnni_min_par_mnk`);
/// only a small parallel problem falls back to the plain widen-and-multiply kernel. Either
/// way this just confirms i8 throughput is SIMD-accelerated rather than scalar-bound
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

/// Complex (c32) GEMM throughput: gemmkit's [`gemmkit::gemm_cplx`] (no conjugation on either
/// operand) against the `gemm` crate's native c32 support. A complex multiply-add is
/// conventionally counted as 4 real flops (4 real multiplies + 4 real adds), so both sides
/// scale their raw `2*m*k*n` timing by that factor rather than reporting the real-valued count
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
            // Unused: the reported figures below come straight from `measure`'s own count,
            // rescaled by 2x (see the comment at the print site)
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
            // measure() already assumes 2*m*n*k flops; a complex mul-add is 4x that, so
            // doubling its GFLOP/s figure (not calling the unused cflop() above) gets there
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
