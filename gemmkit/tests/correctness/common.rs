//! Shared test harness: element traits, random fills, views, references, accuracy gates.
//!
//! The proptest-free numeric oracle core (the `Elem`/`CElem` traits, `rand_vec`, `Mat`,
//! `reference`, `ref_cplx`, `rand_cplx`, and the Frobenius accuracy gates) is single-sourced
//! in `tests/oracle_common/mod.rs` and shared with the property suite; this module adds the
//! correctness-only pieces (the fixed `Layout` views, `run_case`, and the exact i8 reference).

use gemmkit::{MatMut, MatRef, Parallelism, gemm};

#[path = "../oracle_common/mod.rs"]
mod oracle_common;
// Re-export the oracle core so consumers keep importing everything through `crate::common::*`.
// The local `assert_accurate` below shadows the oracle's `denom_extra`-taking one on purpose.
pub use oracle_common::*;

/// Relative Frobenius error gate: `||C - Cref|| / (||A||*||B|| + tiny) <= 8*k*eps`.
///
/// Thin wrapper over [`oracle_common::assert_accurate`] fixing `denom_extra = 0.0`: the
/// correctness suite keeps tiny dims on `beta == 0`, so `||A||*||B||` alone bounds the output
/// and no `|beta|*||C0||` term is needed (the property suite adds it).
#[allow(clippy::too_many_arguments)]
pub fn assert_accurate<T: Elem>(
    got: &[T],
    got_rs: isize,
    got_cs: isize,
    m: usize,
    n: usize,
    cref: &[f64],
    a: &Mat<T>,
    b: &Mat<T>,
    k: usize,
    ctx: &str,
) {
    oracle_common::assert_accurate(got, got_rs, got_cs, m, n, cref, a, b, k, 0.0, ctx);
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum Layout {
    Row,
    Col,
    /// Padded leading dimension (general strides, both > 1).
    GeneralPad,
}

/// Build a backing buffer + (rs, cs) for `m`, presenting it in `layout`.
pub(crate) fn build_view<T: Elem>(m: &Mat<T>, layout: Layout) -> (Vec<T>, isize, isize) {
    let (r, c) = (m.rows, m.cols);
    match layout {
        Layout::Row => {
            let pad = 0;
            let rs = (c + pad) as isize;
            let mut buf = vec![T::from_f64(0.0); r * (c + pad)];
            for i in 0..r {
                for j in 0..c {
                    buf[i * (c + pad) + j] = m.at(i, j);
                }
            }
            (buf, rs, 1)
        }
        Layout::Col => {
            let cs = r as isize;
            let mut buf = vec![T::from_f64(0.0); r * c];
            for i in 0..r {
                for j in 0..c {
                    buf[j * r + i] = m.at(i, j);
                }
            }
            (buf, 1, cs)
        }
        Layout::GeneralPad => {
            // row-major with padded rows: rs = c+3, cs = 1 -> general but cs==1;
            // make cs=2 too by interleaving a dummy column.
            let cs = 2isize;
            let rs = (2 * c + 5) as isize;
            let total = r * (2 * c + 5);
            let mut buf = vec![T::from_f64(0.0); total];
            for i in 0..r {
                for j in 0..c {
                    buf[i * (2 * c + 5) + j * 2] = m.at(i, j);
                }
            }
            (buf, rs, cs)
        }
    }
}

pub(crate) fn run_case<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: Layout,
    lb: Layout,
    lc: Layout,
    alpha: T,
    beta: T,
    par: Parallelism,
) {
    let a = Mat::<T>::rand(m, k, 0x1111 + (m * 7 + k * 13 + n) as u64);
    let b = Mat::<T>::rand(k, n, 0x2222 + (m + k * 5 + n * 11) as u64);
    let c0 = Mat::<T>::rand(m, n, 0x3333 + (m * 3 + k + n * 2) as u64);

    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, lb);
    let (mut cbuf, rsc, csc) = build_view(&c0, lc);

    let cref = reference(&a, &b, &c0, alpha.to_f64(), beta.to_f64());

    gemm(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    let ctx = format!(
        "T={} {m}x{k}x{n} la={la:?} lb={lb:?} lc={lc:?} a={} b={} par={par:?}",
        core::any::type_name::<T>(),
        alpha.to_f64(),
        beta.to_f64()
    );
    assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, &ctx);
}

/// Deterministic i8 fill in [-100, 100] (kept small so the i32 reference never
/// overflows for the tested k, making the comparison exact).
#[cfg(feature = "int8")]
pub(crate) fn rand_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 24) as i64 % 201 - 100) as i8
        })
        .collect()
}

/// Exact i32 GEMM reference (row-major), accumulated in i64 then range-checked, so
/// the integer kernel must match it **bit-for-bit**.
#[cfg(feature = "int8")]
pub(crate) fn ref_i8(
    a: &[i8],
    b: &[i8],
    c0: &[i32],
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    beta: i32,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for p in 0..k {
                acc += a[i * k + p] as i64 * b[p * n + j] as i64;
            }
            let v = beta as i64 * c0[i * n + j] as i64 + alpha as i64 * acc;
            assert!(
                (i32::MIN as i64..=i32::MAX as i64).contains(&v),
                "reference overflow — tighten test sizes"
            );
            out[i * n + j] = v as i32;
        }
    }
    out
}
