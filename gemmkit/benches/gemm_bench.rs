//! Benchmark gemmkit against the `gemm` crate (faer's backend) and
//! `matrixmultiply` (ndarray's current backend) across a range of square sizes.
//!
//! Run: `cargo bench -p gemmkit`. GFLOP/s = 2·m·n·k / median_time.
//!
//! Built only when *not* under Miri: `criterion`/`gemm`/`matrixmultiply` are
//! `cfg(not(miri))` dev-dependencies (see `Cargo.toml`).
#![cfg(not(miri))]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use gemmkit::{MatMut, MatRef, Parallelism, gemm};

fn fill(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0)
        .collect()
}

fn bench_square(c: &mut Criterion) {
    let sizes = [128usize, 256, 512, 1024, 2048];
    for par_label in ["serial", "parallel"] {
        let par = if par_label == "serial" {
            Parallelism::Serial
        } else {
            Parallelism::Rayon(0)
        };
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

criterion_group!(benches, bench_square);
criterion_main!(benches);
