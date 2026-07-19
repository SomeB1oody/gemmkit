//! Public `cargo bench` surface for gemmkit (criterion)
//!
//! A small, curated set of statistically-rigorous benches over the crate's headline
//! entry points. This is the regression-tracking harness (`--save-baseline`); the
//! exhaustive investigation suite lives in `tests/perf/` (median-of-9 sampling,
//! `#[ignore]` tests) and is not duplicated here. GFLOP/s is throughput over median
//! time, counting `2*m*n*k` flops for a real GEMM and `8*m*n*k` for complex (4 real
//! multiplies + 4 real adds per complex multiply-add)
//!
//! Groups:
//!   sgemm      square f32, gemmkit vs the `gemm` crate and `matrixmultiply`, serial
//!              + parallel
//!   dtypes     gemmkit-only 1024^3 throughput by element type, serial + parallel:
//!              f32 always, f16 / bf16 / i8 / c32 behind their cargo features
//!   gemv       n=1 at m=k=4096, the dot layout (row-major A) and the axpy layout
//!              (col-major A) vs the `gemm` crate, serial + parallel
//!   prepacked  fixed-weight inference (small m, fixed B): plain gemm vs a reused
//!              prepacked B, f32 and (behind `int8`) the i8 twin
//!   batched    `gemm_batched` vs a naive per-call loop, auto parallelism
//!
//! Run (default features cover sgemm, dtypes f32, gemv, prepacked f32, batched):
//!   cargo bench -p gemmkit
//! Unlock the f16/bf16/i8/c32 dtypes and the i8 prepacked arm:
//!   cargo bench -p gemmkit --all-features
//! Track regressions against a saved baseline:
//!   cargo bench -p gemmkit --all-features -- --save-baseline main
//!   cargo bench -p gemmkit --all-features -- --baseline main
//! Smoke every bench once, no measurement (fast):
//!   cargo bench -p gemmkit -- --test
//!
//! Excluded under Miri: `criterion`/`gemm`/`matrixmultiply` are `cfg(not(miri))`
//! dev-dependencies (see `Cargo.toml`)
#![cfg(not(miri))]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use gemmkit::{MatMut, MatRef, Parallelism, gemm, gemm_batched};

/// Criterion config shared by every `criterion_group!` below: a lighter sample size and
/// shorter measurement/warm-up window than criterion's defaults, so the full 5-group
/// suite finishes in a few minutes. `criterion_group!` appends `.configure_from_args()`,
/// so a CLI flag (`--sample-size`, `--save-baseline`, `--test`, a filter) still overrides
/// these
fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_secs(1))
}

/// Deterministic f32 fill in [0, 1): each element is hashed from its index alone (not a
/// chained RNG), so the same `n` always reproduces the same values
fn fill(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0)
        .collect()
}

fn par_of(label: &str) -> Parallelism {
    if label == "serial" {
        Parallelism::Serial
    } else {
        Parallelism::Rayon(0)
    }
}

/// Square f32 sgemm at 5 sizes, gemmkit vs the `gemm` crate (faer's backend) and
/// `matrixmultiply` (ndarray's default backend), serial + parallel; `matrixmultiply` is
/// only entered in the serial group
fn bench_sgemm(c: &mut Criterion) {
    let sizes = [128usize, 256, 512, 1024, 2048];
    for par_label in ["serial", "parallel"] {
        let par = par_of(par_label);
        let mut group = c.benchmark_group(format!("sgemm_square_{par_label}"));
        for &s in &sizes {
            let (m, k, n) = (s, s, s);
            let a = fill(m * k);
            let b = fill(k * n);
            let mut cc = vec![0.0f32; m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));

            group.bench_with_input(BenchmarkId::new("gemmkit", s), &s, |bch, _| {
                bch.iter(|| {
                    gemm(
                        1.0,
                        MatRef::from_col_major(&a, m, k),
                        MatRef::from_col_major(&b, k, n),
                        0.0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });

            group.bench_with_input(BenchmarkId::new("gemm_crate", s), &s, |bch, _| {
                let parallelism = if par_label == "serial" {
                    gemm::Parallelism::None
                } else {
                    gemm::Parallelism::Rayon(0)
                };
                bch.iter(|| {
                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            cc.as_mut_ptr(),
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
                            parallelism,
                        );
                    }
                    black_box(&cc);
                });
            });

            if par_label == "serial" {
                group.bench_with_input(BenchmarkId::new("matrixmultiply", s), &s, |bch, _| {
                    bch.iter(|| {
                        unsafe {
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
                                cc.as_mut_ptr(),
                                1,
                                m as isize,
                            );
                        }
                        black_box(&cc);
                    });
                });
            }
        }
        group.finish();
    }
}

/// gemmkit-only element-type throughput at a single representative 1024^3 shape, serial
/// and parallel. f32 is always present; f16/bf16 (behind `half`), i8 (behind `int8`), and
/// c32 (behind `complex`) each add a bench function only when their feature is enabled, so
/// `--all-features` shows more entries in the group. Flops are counted like
/// `tests/perf/dtypes.rs`: `2*m*n*k` for the real types, `8*m*n*k` for complex
fn bench_dtypes(c: &mut Criterion) {
    let s = 1024usize;
    let (m, k, n) = (s, s, s);
    let af = fill(m * k);
    let bf = fill(k * n);
    for par_label in ["serial", "parallel"] {
        let par = par_of(par_label);
        let mut group = c.benchmark_group(format!("dtypes_{par_label}"));

        // f32: the always-on baseline entry
        {
            let mut cc = vec![0.0f32; m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_function("f32", |bch| {
                bch.iter(|| {
                    gemm(
                        1.0,
                        MatRef::from_col_major(&af, m, k),
                        MatRef::from_col_major(&bf, k, n),
                        0.0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
        }

        // f16 (col-major, widened to f32 for accumulation) and bf16 (row-major A +
        // col-major B: on the vdpbf16ps dot kernel this lets A's pack read k contiguously
        // in place instead of gathering it with a strided walk)
        #[cfg(feature = "half")]
        {
            use gemmkit::{bf16, f16};
            let a16: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
            let b16: Vec<f16> = bf.iter().map(|&x| f16::from_f32(x)).collect();
            let mut c16 = vec![f16::from_f32(0.0); m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_function("f16", |bch| {
                bch.iter(|| {
                    gemm(
                        f16::from_f32(1.0),
                        MatRef::from_col_major(&a16, m, k),
                        MatRef::from_col_major(&b16, k, n),
                        f16::from_f32(0.0),
                        MatMut::from_col_major(&mut c16, m, n),
                        par,
                    );
                    black_box(&c16);
                });
            });

            let ab: Vec<bf16> = af.iter().map(|&x| bf16::from_f32(x)).collect();
            let bb: Vec<bf16> = bf.iter().map(|&x| bf16::from_f32(x)).collect();
            let mut cb = vec![bf16::from_f32(0.0); m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_function("bf16", |bch| {
                bch.iter(|| {
                    gemm(
                        bf16::from_f32(1.0),
                        MatRef::from_row_major(&ab, m, k),
                        MatRef::from_col_major(&bb, k, n),
                        bf16::from_f32(0.0),
                        MatMut::from_col_major(&mut cb, m, n),
                        par,
                    );
                    black_box(&cb);
                });
            });
        }

        // i8 -> i32, col-major operands
        #[cfg(feature = "int8")]
        {
            let ai: Vec<i8> = (0..m * k).map(|i| (i % 17) as i8 - 8).collect();
            let bi: Vec<i8> = (0..k * n).map(|i| (i % 13) as i8 - 6).collect();
            let mut ci = vec![0i32; m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_function("i8", |bch| {
                bch.iter(|| {
                    gemmkit::gemm_i8(
                        1,
                        MatRef::from_col_major(&ai, m, k),
                        MatRef::from_col_major(&bi, k, n),
                        0,
                        MatMut::from_col_major(&mut ci, m, n),
                        par,
                    );
                    black_box(&ci);
                });
            });
        }

        // c32, col-major operands, no conjugation; 8 flops per multiply-add
        #[cfg(feature = "complex")]
        {
            use gemmkit::Complex;
            let mkc = |seed: u64, len: usize| {
                let mut z = seed | 1;
                (0..len)
                    .map(|_| {
                        z ^= z << 13;
                        z ^= z >> 7;
                        z ^= z << 17;
                        Complex::new((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5, 0.25)
                    })
                    .collect::<Vec<_>>()
            };
            let ac = mkc(1, m * k);
            let bc = mkc(2, k * n);
            let mut cc = vec![Complex::new(0.0f32, 0.0); m * n];
            group.throughput(Throughput::Elements((8 * m * n * k) as u64));
            group.bench_function("c32", |bch| {
                bch.iter(|| {
                    gemmkit::gemm_cplx(
                        Complex::new(1.0f32, 0.0),
                        MatRef::from_col_major(&ac, m, k),
                        false,
                        MatRef::from_col_major(&bc, k, n),
                        false,
                        Complex::new(0.0f32, 0.0),
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
        }

        group.finish();
    }
}

/// f32 gemv (n == 1) at m = k = 4096, serial + parallel: the dot layout (row-major A,
/// which routes to `dot_rows`) and the axpy layout (col-major A, which routes to the
/// register-blocked or plain axpy path), each vs the `gemm` crate on the same m,k,1
/// shape. Flops = `2*m*k`
fn bench_gemv(c: &mut Criterion) {
    let (m, k, n) = (4096usize, 4096usize, 1usize);
    let a = fill(m * k);
    let x = fill(k * n);
    for par_label in ["serial", "parallel"] {
        let par = par_of(par_label);
        let mut group = c.benchmark_group(format!("gemv_{par_label}"));
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        let mut cc = vec![0.0f32; m * n];

        group.bench_function("gemmkit_dot", |bch| {
            bch.iter(|| {
                gemm(
                    1.0,
                    MatRef::from_row_major(&a, m, k),
                    MatRef::from_col_major(&x, k, n),
                    0.0,
                    MatMut::from_col_major(&mut cc, m, n),
                    par,
                );
                black_box(&cc);
            });
        });

        group.bench_function("gemmkit_axpy", |bch| {
            bch.iter(|| {
                gemm(
                    1.0,
                    MatRef::from_col_major(&a, m, k),
                    MatRef::from_col_major(&x, k, n),
                    0.0,
                    MatMut::from_col_major(&mut cc, m, n),
                    par,
                );
                black_box(&cc);
            });
        });

        let gpar = if par_label == "serial" {
            gemm::Parallelism::None
        } else {
            gemm::Parallelism::Rayon(0)
        };
        group.bench_function("gemm_crate", |bch| {
            bch.iter(|| {
                unsafe {
                    gemm::gemm(
                        m,
                        n,
                        k,
                        cc.as_mut_ptr(),
                        m as isize,
                        1,
                        false,
                        a.as_ptr(),
                        m as isize,
                        1,
                        x.as_ptr(),
                        k as isize,
                        1,
                        0.0,
                        1.0,
                        false,
                        false,
                        false,
                        gpar,
                    );
                }
                black_box(&cc);
            });
        });

        group.finish();
    }
}

/// Fixed-weight inference pattern (serial): a small activation batch `m` against a
/// fixed `k = n = 2048` weight matrix B, plain gemm vs `gemm_packed_b` / `gemm_i8_packed_b`
/// reusing a B packed once outside the timed loop. `m` (8, 64) sits well below the
/// engine's RHS-pack threshold, so plain f32 gemm reads row-major B in place with a
/// large per-column stride every call; the prepacked panel is contiguous instead, which
/// is where its win comes from. The i8 arm (behind `int8`) is the stronger case: the
/// VNNI `vpdpbusd` kernel always packs its RHS into a k-quad-interleaved layout, so
/// plain `gemm_i8` pays that repack on every call regardless of B's layout, while the
/// prepacked entry pays it once
fn bench_prepacked(c: &mut Criterion) {
    let (k, n) = (2048usize, 2048usize);
    let par = Parallelism::Serial;
    let mut group = c.benchmark_group("prepacked");

    // f32, row-major B: plain gemm vs a reused prepacked B
    {
        let b = fill(k * n);
        let packed = gemmkit::prepack_rhs(MatRef::from_row_major(&b, k, n));
        for &m in &[8usize, 64] {
            let a = fill(m * k);
            let mut cc = vec![0.0f32; m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_with_input(BenchmarkId::new("f32_plain", m), &m, |bch, _| {
                bch.iter(|| {
                    gemm(
                        1.0,
                        MatRef::from_col_major(&a, m, k),
                        MatRef::from_row_major(&b, k, n),
                        0.0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
            group.bench_with_input(BenchmarkId::new("f32_packed_b", m), &m, |bch, _| {
                bch.iter(|| {
                    gemmkit::gemm_packed_b(
                        1.0,
                        MatRef::from_col_major(&a, m, k),
                        &packed,
                        0.0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
        }
    }

    // i8, col-major B: plain gemm_i8 vs a reused prepacked B (the VNNI repack runs on
    // every plain call regardless of B's layout)
    #[cfg(feature = "int8")]
    {
        let b: Vec<i8> = (0..k * n).map(|i| (i % 13) as i8 - 6).collect();
        let packed = gemmkit::prepack_rhs_i8(MatRef::from_col_major(&b, k, n));
        for &m in &[8usize, 64] {
            let a: Vec<i8> = (0..m * k).map(|i| (i % 17) as i8 - 8).collect();
            let mut cc = vec![0i32; m * n];
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_with_input(BenchmarkId::new("i8_plain", m), &m, |bch, _| {
                bch.iter(|| {
                    gemmkit::gemm_i8(
                        1,
                        MatRef::from_col_major(&a, m, k),
                        MatRef::from_col_major(&b, k, n),
                        0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
            group.bench_with_input(BenchmarkId::new("i8_packed_b", m), &m, |bch, _| {
                bch.iter(|| {
                    gemmkit::gemm_i8_packed_b(
                        1,
                        MatRef::from_col_major(&a, m, k),
                        &packed,
                        0,
                        MatMut::from_col_major(&mut cc, m, n),
                        par,
                    );
                    black_box(&cc);
                });
            });
        }
    }

    group.finish();
}

/// Batched f32 GEMM: `gemm_batched` (1 call, parallelized across the batch) vs a naive
/// loop that issues `batch` separate `gemm(Rayon(0))` calls, over the same
/// contiguously-packed operands. Total throughput `2*batch*m*n*k`. 2 regimes: many
/// small cubes (batch = 64 of 48^3, where the naive loop pays a fork/join per element)
/// and a few larger cubes (batch = 8 of 256^3)
fn bench_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched");
    for &(batch, s) in &[(64usize, 48usize), (8, 256)] {
        let (m, k, n) = (s, s, s);
        let a = fill(batch * m * k);
        let b = fill(batch * k * n);
        let mut cc = vec![0.0f32; batch * m * n];
        group.throughput(Throughput::Elements((2 * batch * m * n * k) as u64));
        let id = format!("b{batch}_{s}cube");

        group.bench_with_input(BenchmarkId::new("batched", &id), &id, |bch, _| {
            bch.iter(|| {
                gemm_batched(
                    batch,
                    1.0,
                    MatRef::new(&a, m, k, 1, m as isize),
                    (m * k) as isize,
                    MatRef::new(&b, k, n, 1, k as isize),
                    (k * n) as isize,
                    0.0,
                    MatMut::new(&mut cc, m, n, 1, m as isize),
                    (m * n) as isize,
                    Parallelism::Rayon(0),
                );
                black_box(&cc);
            });
        });

        group.bench_with_input(BenchmarkId::new("naive_loop", &id), &id, |bch, _| {
            bch.iter(|| {
                for bi in 0..batch {
                    let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
                    gemm(
                        1.0,
                        MatRef::from_col_major(&a[ao..ao + m * k], m, k),
                        MatRef::from_col_major(&b[bo..bo + k * n], k, n),
                        0.0,
                        MatMut::from_col_major(&mut cc[co..co + m * n], m, n),
                        Parallelism::Rayon(0),
                    );
                }
                black_box(&cc);
            });
        });
    }
    group.finish();
}

criterion_group! { name = sgemm; config = config(); targets = bench_sgemm }
criterion_group! { name = dtypes; config = config(); targets = bench_dtypes }
criterion_group! { name = gemv; config = config(); targets = bench_gemv }
criterion_group! { name = prepacked; config = config(); targets = bench_prepacked }
criterion_group! { name = batched; config = config(); targets = bench_batched }
criterion_main!(sgemm, dtypes, gemv, prepacked, batched);
