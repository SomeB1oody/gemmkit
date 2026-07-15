//! Naive dense references and the tolerance/exact result gates

use crate::common::{CplxElem, RealElem};

// dense materialization + references

pub(crate) fn dense_real<T: RealElem>(
    buf: &[T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize].to_f64();
        }
    }
    out
}

pub(crate) fn dense_i32_from_i8(
    buf: &[i8],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
) -> Vec<i32> {
    let mut out = vec![0i32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize] as i32;
        }
    }
    out
}

pub(crate) fn dense_i32(buf: &[i32], rows: usize, cols: usize, rs: isize, cs: isize) -> Vec<i32> {
    let mut out = vec![0i32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize];
        }
    }
    out
}

/// Materialize a complex view row-major in f64, applying `conj` to the imaginary part
/// (mirrors `api.rs::gemm_cplx`, which conjugates the *operand* before the product)
pub(crate) fn dense_cplx<T: CplxElem>(
    buf: &[T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
    conj: bool,
) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let (re, im) = buf[(i as isize * rs + j as isize * cs) as usize].parts();
            out[i * cols + j] = (re, if conj { -im } else { im });
        }
    }
    out
}

pub(crate) fn frob(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// f64 reference `C <- beta*C0 + alpha*A*B` with the `beta == 0` "C not read" rule
/// (`reference` in tests/correctness/common.rs), so a NaN-seeded C (the beta==0 fuzz) never taints it
pub(crate) fn ref_gemm_real(
    da: &[f64],
    db: &[f64],
    dc0: &[f64],
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += da[i * k + p] * db[p * n + j];
            }
            let base = if beta == 0.0 {
                0.0
            } else {
                beta * dc0[i * n + j]
            };
            out[i * n + j] = base + alpha * acc;
        }
    }
    out
}

/// Exact wrapping-i32 reference; `i32` accumulation is associative mod 2^32, so every
/// blocking/threading/ISA schedule reproduces it bit-for-bit (the `assert_eq!` bar)
pub(crate) fn ref_gemm_i8(
    da: &[i32],
    db: &[i32],
    dc0: &[i32],
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    beta: i32,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc = acc.wrapping_add(da[i * k + p].wrapping_mul(db[p * n + j]));
            }
            let base = if beta == 0 {
                0
            } else {
                beta.wrapping_mul(dc0[i * n + j])
            };
            out[i * n + j] = base.wrapping_add(alpha.wrapping_mul(acc));
        }
    }
    out
}

/// f64 complex reference (conj already baked into `da`/`db`) with the beta==0 rule:
/// `ref_cplx` in the suite has no such rule, so this closes the NaN-C false positive
pub(crate) fn ref_gemm_cplx(
    da: &[(f64, f64)],
    db: &[(f64, f64)],
    dc0: &[(f64, f64)],
    m: usize,
    k: usize,
    n: usize,
    alpha: (f64, f64),
    beta: (f64, f64),
) -> Vec<(f64, f64)> {
    let cmul = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let beta_zero = beta == (0.0, 0.0);
    let mut out = vec![(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = (0.0f64, 0.0f64);
            for p in 0..k {
                let pr = cmul(da[i * k + p], db[p * n + j]);
                acc = (acc.0 + pr.0, acc.1 + pr.1);
            }
            let term = cmul(alpha, acc);
            let base = if beta_zero {
                (0.0, 0.0)
            } else {
                cmul(beta, dc0[i * n + j])
            };
            out[i * n + j] = (base.0 + term.0, base.1 + term.1);
        }
    }
    out
}

// tolerance / exact gates (panic == the libFuzzer report channel)

/// Relative-Frobenius gate. The denominator is the `8*k*EPS` gate of
/// `assert_accurate` (tests/correctness/common.rs), `||A||*||B||`, augmented with the
/// `|alpha|` product scale and the `|beta|*||C0||` epilogue scale. The suite's gate
/// omits both because its inputs are always `k >= 1` with O(1) alpha/non-empty A*B; the
/// fuzzer reaches `k == 0` (empty operands, `||A||*||B|| == 0`) and `|alpha| = 2.5`,
/// where the epilogue/product rounding (correct in the type but higher-precision in the
/// f64 reference) would otherwise blow up a zero denominator. `denom` is precomputed by
/// the caller (with the `||C0||` term dropped when `beta == 0`, so a NaN-seeded C0 can't
/// taint it)
pub(crate) fn real_gate<T: RealElem>(
    cbuf: &[T],
    rsc: isize,
    csc: isize,
    m: usize,
    n: usize,
    cref: &[f64],
    denom: f64,
    k: usize,
    ctx: &str,
) {
    let mut diff2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let g = cbuf[(i as isize * rsc + j as isize * csc) as usize].to_f64();
            let r = cref[i * n + j];
            if !g.is_finite() {
                panic!("{ctx}: non-finite output at ({i},{j}) (m={m},k={k},n={n})");
            }
            let d = g - r;
            diff2 += d * d;
        }
    }
    let rel = diff2.sqrt() / denom;
    let tol = 8.0 * (k.max(1) as f64) * T::EPS;
    if !(rel <= tol) {
        panic!("{ctx}: rel err {rel:e} > tol {tol:e} (m={m},k={k},n={n})");
    }
}

/// The gate denominator: `|alpha|*||A||*||B|| + |beta|*||C0|| + tiny`. `nc0` must be `0`
/// when `beta == 0` (the C0 term is dropped and C0 may be NaN-seeded)
pub(crate) fn real_denom(alpha_f: f64, na: f64, nb: f64, beta_f: f64, nc0: f64) -> f64 {
    alpha_f.abs() * na * nb + beta_f.abs() * nc0 + 1e-30
}

pub(crate) fn cplx_gate<T: CplxElem>(
    cbuf: &[T],
    rsc: isize,
    csc: isize,
    m: usize,
    n: usize,
    cref: &[(f64, f64)],
    k: usize,
    ctx: &str,
) {
    let mut diff2 = 0.0;
    let mut ref2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let (gr, gi) = cbuf[(i as isize * rsc + j as isize * csc) as usize].parts();
            let (rr, ri) = cref[i * n + j];
            if !(gr.is_finite() && gi.is_finite()) {
                panic!("{ctx}: non-finite output at ({i},{j}) (m={m},k={k},n={n})");
            }
            diff2 += (gr - rr).powi(2) + (gi - ri).powi(2);
            ref2 += rr * rr + ri * ri;
        }
    }
    let rel = diff2.sqrt() / (ref2.sqrt() + 1e-30);
    let tol = 16.0 * (k.max(1) as f64) * T::EPS;
    if !(rel <= tol) {
        panic!("{ctx}: rel err {rel:e} > tol {tol:e} (m={m},k={k},n={n})");
    }
}

pub(crate) fn i8_gate(
    cbuf: &[i32],
    rsc: isize,
    csc: isize,
    m: usize,
    n: usize,
    cref: &[i32],
    ctx: &str,
) {
    for i in 0..m {
        for j in 0..n {
            let g = cbuf[(i as isize * rsc + j as isize * csc) as usize];
            let r = cref[i * n + j];
            if g != r {
                panic!("{ctx}: i8 mismatch at ({i},{j}): got {g}, ref {r}");
            }
        }
    }
}
