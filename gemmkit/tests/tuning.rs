//! Tuning-surface behavior. Isolated in its own test binary because it mutates
//! process-global thresholds; each test touches a *different* knob so they do not
//! interfere when the harness runs them concurrently.

use gemmkit::{MatMut, MatRef, Parallelism, gemm, tuning};

/// `usize::MAX` must not collide with the internal "unset" sentinel: setting the
/// maximum should take effect (clamped to `usize::MAX - 1`), not be ignored.
#[test]
fn max_value_threshold_takes_effect() {
    tuning::set_parallel_threshold(usize::MAX);
    let got = tuning::parallel_threshold();
    assert_ne!(
        got,
        48 * 48 * 256,
        "usize::MAX was silently dropped to the default"
    );
    assert_eq!(
        got,
        usize::MAX - 1,
        "should clamp to the largest usable value"
    );
}

/// Both the packed and the in-place (unpacked) RHS paths must be correct,
/// including partial column tiles (n not a multiple of NR). Toggle the gate to
/// force each mode and compare to a naive reference.
#[test]
fn rhs_packing_both_modes_correct() {
    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j]; // row-major logical
                }
                c[i * n + j] = s;
            }
        }
        c
    }
    let col = |v: &[f64], r: usize, c: usize| {
        let mut o = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = v[i * c + j];
            }
        }
        o
    };
    for &force in &[0usize, usize::MAX] {
        tuning::set_rhs_pack_threshold(force); // 0 = always pack, MAX = never pack
        for &(m, k, n) in &[(33, 17, 19), (64, 40, 13), (128, 65, 11), (40, 33, 28)] {
            let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
            let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
            let cref = naive(&a, &b, m, k, n);
            let (ac, bc) = (col(&a, m, k), col(&b, k, n));
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&ac, m, k),
                MatRef::from_col_major(&bc, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            for i in 0..m {
                for j in 0..n {
                    let got = cc[j * m + i];
                    let exp = cref[i * n + j];
                    assert!(
                        (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                        "force={force} {m}x{k}x{n} ({i},{j}): {got} vs {exp}"
                    );
                }
            }
        }
    }
}

/// `gemv_threshold` is a live knob: setting it to 0 disables the dedicated gemv
/// path, which then falls through to the general driver and stays correct.
#[test]
fn gemv_threshold_disables_path_but_stays_correct() {
    tuning::set_gemv_threshold(0);
    // m == 1 row-vector times 5x4 matrix.
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0]; // 1x5
    let bm = [
        1.0f64, 0.0, 1.0, 0.0, // row 0
        0.0, 1.0, 0.0, 1.0, // row 1
        2.0, 0.0, 0.0, 0.0, // row 2
        0.0, 3.0, 0.0, 0.0, // row 3
        1.0, 1.0, 1.0, 1.0, // row 4
    ]; // 5x4 row-major
    let mut c = [0.0f64; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 1, 5),
        MatRef::from_row_major(&bm, 5, 4),
        0.0,
        MatMut::from_row_major(&mut c, 1, 4),
        Parallelism::Serial,
    );
    // Reference: c[j] = sum_k a[k]*B[k,j].
    let mut expect = [0.0f64; 4];
    for j in 0..4 {
        for k in 0..5 {
            expect[j] += a[k] * bm[k * 4 + j];
        }
    }
    for j in 0..4 {
        assert!(
            (c[j] - expect[j]).abs() < 1e-12,
            "c[{j}]={} expect {}",
            c[j],
            expect[j]
        );
    }
}

/// Both LHS paths must be correct under parallelism: packed (forced by a zero-byte
/// stride gate, so every column-major A packs) and read-in-place (gate disabled).
/// Exercises the dynamic scheduler's whole-row-block ("packed") grain plus partial
/// row/column tiles, against a naive reference.
#[test]
fn lhs_packing_both_modes_correct() {
    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }
    let col = |v: &[f64], r: usize, c: usize| {
        let mut o = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = v[i * c + j];
            }
        }
        o
    };
    // 1 = always pack a column-major A (csa*sizeof >= 1); MAX = never via stride.
    // (0 would mean "auto" — derive from page size — so it is not an extreme here.)
    for &stride in &[1usize, usize::MAX] {
        tuning::set_lhs_pack_stride(stride);
        for &(m, k, n) in &[(97, 64, 80), (160, 48, 133), (200, 96, 175), (33, 17, 19)] {
            let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
            let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
            let cref = naive(&a, &b, m, k, n);
            let (ac, bc) = (col(&a, m, k), col(&b, k, n));
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&ac, m, k),
                MatRef::from_col_major(&bc, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            for i in 0..m {
                for j in 0..n {
                    let got = cc[j * m + i];
                    let exp = cref[i * n + j];
                    assert!(
                        (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                        "stride={stride} {m}x{k}x{n} ({i},{j}): {got} vs {exp}"
                    );
                }
            }
        }
    }
}

/// `parallel_oversample` is a live knob: 0 (clamped to 1), 1, and an adversarially
/// huge value must each yield a correct parallel result with no panic — the latter
/// proves the grain computation's saturating multiply guards against overflow.
#[test]
fn parallel_oversample_extremes_stay_correct() {
    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }
    let col = |v: &[f64], r: usize, c: usize| {
        let mut o = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = v[i * c + j];
            }
        }
        o
    };
    let (m, k, n) = (96usize, 80, 64);
    let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
    let cref = naive(&a, &b, m, k, n);
    let (ac, bc) = (col(&a, m, k), col(&b, k, n));
    for &ov in &[0usize, 1, usize::MAX] {
        tuning::set_parallel_oversample(ov);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&ac, m, k),
            MatRef::from_col_major(&bc, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        for i in 0..m {
            for j in 0..n {
                let got = cc[j * m + i];
                let exp = cref[i * n + j];
                assert!(
                    (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                    "oversample={ov} ({i},{j}): {got} vs {exp}"
                );
            }
        }
    }
}
