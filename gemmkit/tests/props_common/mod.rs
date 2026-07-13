//! Shared strategies, reference implementations, and accuracy gates for the
//! property-based suite (props_api / props_packed / props_knobs).
//!
//! Included via `mod props_common;` from each top-level `tests/props_*.rs`, so it is
//! only ever compiled inside an already `cfg(all(not(miri), not(target_family =
//! "wasm")))`-gated crate root; it is never itself a test target. Each binary uses a
//! different subset of these helpers, so the module is `#![allow(dead_code)]` (the
//! classic `tests/common` pattern); everything else stays clippy-clean under
//! `-D warnings`.
//!
//! The numeric tolerances (`assert_accurate` = `8·k·EPS`, `assert_cplx_accurate` =
//! `16·k·EPS`, the wrapping-i32 i8 reference) and the reference kernels are ported
//! from `tests/correctness/common.rs` — that file is the canonical source; keep in sync.
#![allow(dead_code)]

use gemmkit::{GemmScalar, Parallelism};
use proptest::prelude::*;

/// Per-property case count with a `PROPTEST_CASES` override. `ProptestConfig::default()`
/// already reads the env var, but an explicit `cases:` field clobbers it, so read it here
/// and fall back to the tuned default. Every property block passes `cases(N)`.
pub fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// element trait (ported from tests/correctness/common.rs `Elem`, without the Miri arms —
// these files are `not(miri)`)
// ---------------------------------------------------------------------------

/// Trait letting the harness be generic over the homogeneous float element types.
pub trait Elem: GemmScalar {
    const EPS: f64;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
    /// Native bit pattern (widened to u64) for the exact run-to-run determinism check.
    fn to_bits_u64(self) -> u64;
}
impl Elem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}
impl Elem for f64 {
    const EPS: f64 = f64::EPSILON;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}
// Narrow types accumulate in f32 and round outputs to 16 bits, so their `EPS` is the
// 16-bit machine epsilon (f16 ≈ 9.8e-4, bf16 ≈ 7.8e-3) — the dominant error is the
// final round (see the f16 `Elem` impl in tests/correctness/common.rs).
#[cfg(feature = "half")]
impl Elem for gemmkit::f16 {
    const EPS: f64 = 9.765625e-4; // 2^-10
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(x: f64) -> Self {
        gemmkit::f16::from_f64(x)
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}
#[cfg(feature = "half")]
impl Elem for gemmkit::bf16 {
    const EPS: f64 = 7.8125e-3; // 2^-7
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(x: f64) -> Self {
        gemmkit::bf16::from_f64(x)
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}

/// Deterministic pseudo-random fill in [-1, 1) (port of `rand_vec` in tests/correctness/common.rs).
pub fn rand_vec<T: Elem>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            T::from_f64(2.0 * u - 1.0)
        })
        .collect()
}

/// Logical matrix in row-major order plus its dimensions, for building views.
pub struct Mat<T> {
    pub v: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}
impl<T: Elem> Mat<T> {
    pub fn rand(rows: usize, cols: usize, seed: u64) -> Self {
        Mat {
            v: rand_vec(rows * cols, seed),
            rows,
            cols,
        }
    }
    pub fn at(&self, i: usize, j: usize) -> T {
        self.v[i * self.cols + j]
    }
}

// ---------------------------------------------------------------------------
// layout / stride strategies (generalizes `build_view` in tests/correctness/common.rs;
// the safe API accepts only non-negative strides — api.rs:107-128)
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub enum PLayout {
    /// rs = cols + pad, cs = 1.
    Row { pad: usize },
    /// rs = 1, cs = rows + pad.
    Col { pad: usize },
    /// Both strides > 1: cs in 2..=4, rs = cols*cs + pad.
    General { cs: usize, pad: usize },
}

pub fn layout() -> impl Strategy<Value = PLayout> {
    prop_oneof![
        3 => (0usize..=7).prop_map(|pad| PLayout::Row { pad }),
        3 => (0usize..=7).prop_map(|pad| PLayout::Col { pad }),
        2 => (2usize..=4, 0usize..=5).prop_map(|(cs, pad)| PLayout::General { cs, pad }),
    ]
}

/// The (rs, cs) an untyped `rows×cols` view takes under `l`.
fn strides_for(rows: usize, cols: usize, l: PLayout) -> (usize, usize) {
    match l {
        PLayout::Row { pad } => (cols + pad, 1),
        PLayout::Col { pad } => (1, rows + pad),
        PLayout::General { cs, pad } => (cols * cs + pad, cs),
    }
}

/// Materialize a row-major logical matrix (`vals`, `rows`×`cols`) into a strided buffer
/// presented in layout `l`; returns `(buf, rs, cs)`. Generic port of
/// tests/correctness/common.rs `build_view` that also serves the i8 (i8/i32) element paths.
pub fn build_view_rowmajor<T: Copy>(
    vals: &[T],
    rows: usize,
    cols: usize,
    zero: T,
    l: PLayout,
) -> (Vec<T>, isize, isize) {
    let (rs, cs) = strides_for(rows, cols, l);
    let need = if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * rs + (cols - 1) * cs + 1
    };
    let mut buf = vec![zero; need];
    for i in 0..rows {
        for j in 0..cols {
            buf[i * rs + j * cs] = vals[i * cols + j];
        }
    }
    (buf, rs as isize, cs as isize)
}

/// Typed convenience wrapper for the float element path.
pub fn build_view<T: Elem>(m: &Mat<T>, l: PLayout) -> (Vec<T>, isize, isize) {
    build_view_rowmajor(&m.v, m.rows, m.cols, T::from_f64(0.0), l)
}

// ---------------------------------------------------------------------------
// dimension / coefficient / parallelism strategies
// ---------------------------------------------------------------------------

/// Dimension distribution — edge-weighted around 0, 1, the AVX-512 f32 tile (MR=32,
/// NR=12), the FMA/NEON tiles (6, 12, 16), and the tiny gate 64 (tuning.rs).
pub fn dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        2 => Just(0usize),
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
    ]
}

/// Like [`dim`] but strictly positive (drops the `0` branch): for the packed paths a
/// `0`-length row/col makes the C's `Col`/`Row` layout stride collapse to `0`, which the
/// orientation guard (`|csc| >= |rsc|` / `<=`) correctly rejects — an empty packed C is
/// not a meaningful test.
pub fn pos_dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
    ]
}

/// `k` with an extra tail across the default kc boundary (KC_DEFAULT = 512, tuning.rs:312).
pub fn kdim() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => dim(),
        1 => proptest::sample::select(&[200usize, 511, 512, 513][..]),
    ]
}

/// `k` like [`kdim`] but strictly positive: a `k == 0` empty contraction makes the
/// relative-Frobenius denominator `||A||·||B|| == 0`, so the accuracy gate is only
/// well-defined for `k >= 1` (the `k == 0` overwrite case is covered separately, with
/// `beta == 0`, where the exact result is `0`).
pub fn kdim_pos() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
        1 => proptest::sample::select(&[200usize, 511, 512, 513][..]),
    ]
}

/// Alpha/beta special values (union of the `correctness_alpha_beta` combos in tests/correctness/common.rs; no
/// NaN/Inf — those inputs have no documented contract).
pub fn coeff() -> impl Strategy<Value = f64> {
    proptest::sample::select(&[0.0f64, 1.0, -1.0, 0.5, -1.5, 2.0, 2.5, 1e-3][..])
}

pub fn par() -> impl Strategy<Value = Parallelism> {
    proptest::sample::select(
        &[
            Parallelism::Serial,
            Parallelism::Rayon(0),
            Parallelism::Rayon(3),
        ][..],
    )
}

// ---------------------------------------------------------------------------
// references + accuracy gates (ported from `reference`/`assert_accurate` in tests/correctness/common.rs)
// ---------------------------------------------------------------------------

/// f64 reference: `C <- beta*C0 + alpha*A*B` (beta==0 overwrites, never reads C0).
pub fn reference<T: Elem>(a: &Mat<T>, b: &Mat<T>, c0: &Mat<T>, alpha: f64, beta: f64) -> Vec<f64> {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a.at(i, p).to_f64() * b.at(p, j).to_f64();
            }
            let base = if beta == 0.0 {
                0.0
            } else {
                beta * c0.at(i, j).to_f64()
            };
            out[i * n + j] = base + alpha * acc;
        }
    }
    out
}

/// Frobenius norm of a logical matrix (over its widened f64 values).
pub fn frob_norm<T: Elem>(m: &Mat<T>) -> f64 {
    m.v.iter().map(|x| x.to_f64().powi(2)).sum::<f64>().sqrt()
}

/// Relative Frobenius error gate. `tol = 8*k*eps` is the tolerance factor of
/// `assert_accurate` in tests/correctness/common.rs (keep in sync); the denominator is the textbook GEMM
/// backward-error magnitude `||A||*||B|| + denom_extra`, where `denom_extra` carries the
/// `|beta|*||C0||` term. the correctness suite keeps tiny dims on `beta == 0` so `||A||*||B||`
/// alone bounds the output there; the property suite draws beta over the full (dim, beta)
/// space, so it must add the `beta*C0` contribution or a tiny `k` with a dominant
/// `beta*C0` term would spuriously fail (`denom_extra == 0` reduces to the canonical gate).
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
    denom_extra: f64,
    ctx: &str,
) {
    let norm = |it: &mut dyn Iterator<Item = f64>| -> f64 { it.map(|x| x * x).sum::<f64>().sqrt() };
    let na = norm(&mut a.v.iter().map(|x| x.to_f64()));
    let nb = norm(&mut b.v.iter().map(|x| x.to_f64()));
    let mut diff2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let g = got[(i as isize * got_rs + j as isize * got_cs) as usize].to_f64();
            let r = cref[i * n + j];
            assert!(g.is_finite(), "{ctx}: non-finite output at ({i},{j})");
            let d = g - r;
            diff2 += d * d;
        }
    }
    let rel = diff2.sqrt() / (na * nb + denom_extra + 1e-30);
    let tol = 8.0 * (k.max(1) as f64) * T::EPS;
    assert!(
        rel <= tol,
        "{ctx}: relative error {rel:.3e} > tol {tol:.3e} (m={m},k={k},n={n})"
    );
}

/// Two strided outputs (same shape/strides) are bit-identical element-for-element.
pub fn bits_identical<T: Elem>(x: &[T], y: &[T]) -> bool {
    x.len() == y.len()
        && x.iter()
            .zip(y)
            .all(|(a, b)| a.to_bits_u64() == b.to_bits_u64())
}

// ---------------------------------------------------------------------------
// integer (i8 -> i32) helpers
// ---------------------------------------------------------------------------

/// Deterministic **full-range** i8 fill (unlike the range-limited tests/correctness/common.rs
/// `rand_i8`, :2011-2022, this spans all of `i8` so accumulation wrap is exercised).
#[cfg(feature = "int8")]
pub fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as i8
        })
        .collect()
}

/// **Wrapping**-i32 GEMM reference (row-major operands). The documented i8 contract is
/// two's-complement wrapping i32 arithmetic (`i8_wraps_on_overflow` in tests/correctness/int8.rs); wrapping add
/// is associative, so the reference is order-independent and the kernel must match it exactly.
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub fn ref_i8_wrapping(
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
            let mut acc: i32 = 0;
            for p in 0..k {
                let prod = a[i * k + p] as i32 * b[p * n + j] as i32;
                acc = acc.wrapping_add(prod);
            }
            out[i * n + j] = beta
                .wrapping_mul(c0[i * n + j])
                .wrapping_add(alpha.wrapping_mul(acc));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// complex (c32 / c64) helpers (ported from the complex helpers in tests/correctness/common.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "complex")]
pub trait CElem: gemmkit::ComplexScalar {
    const EPS: f64;
    fn of(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
#[cfg(feature = "complex")]
impl CElem for gemmkit::c32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn of(re: f64, im: f64) -> Self {
        gemmkit::Complex::new(re as f32, im as f32)
    }
    fn parts(self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
}
#[cfg(feature = "complex")]
impl CElem for gemmkit::c64 {
    const EPS: f64 = f64::EPSILON;
    fn of(re: f64, im: f64) -> Self {
        gemmkit::Complex::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

#[cfg(feature = "complex")]
pub fn rand_cplx<T: CElem>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    (0..n).map(|_| T::of(next(), next())).collect()
}

/// f64 complex reference (column-major operands), with conj of A / B as selected.
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn ref_cplx<T: CElem>(
    a: &[T],
    b: &[T],
    c0: &[T],
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> Vec<(f64, f64)> {
    let cmul = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let (al, be) = (alpha.parts(), beta.parts());
    let mut out = vec![(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = (0.0f64, 0.0f64);
            for p in 0..k {
                let mut av = a[p * m + i].parts(); // column-major A
                let mut bv = b[j * k + p].parts(); // column-major B
                if conj_a {
                    av.1 = -av.1;
                }
                if conj_b {
                    bv.1 = -bv.1;
                }
                let pr = cmul(av, bv);
                acc = (acc.0 + pr.0, acc.1 + pr.1);
            }
            let term = cmul(al, acc);
            let bc = cmul(be, c0[j * m + i].parts());
            out[i * n + j] = (bc.0 + term.0, bc.1 + term.1);
        }
    }
    out
}

/// Relative Frobenius gate for the column-major complex output: `rel <= 16*k*eps`
/// (canonical formula from `assert_cplx_accurate` in tests/correctness/common.rs; keep in sync).
#[cfg(feature = "complex")]
pub fn assert_cplx_accurate<T: CElem>(
    got: &[T],
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
            let (gr, gi) = got[j * m + i].parts();
            let (rr, ri) = cref[i * n + j];
            assert!(
                gr.is_finite() && gi.is_finite(),
                "{ctx}: non-finite ({i},{j})"
            );
            diff2 += (gr - rr).powi(2) + (gi - ri).powi(2);
            ref2 += rr * rr + ri * ri;
        }
    }
    let rel = diff2.sqrt() / (ref2.sqrt() + 1e-30);
    let tol = 16.0 * (k.max(1) as f64) * T::EPS;
    assert!(rel <= tol, "{ctx}: rel err {rel:.3e} > tol {tol:.3e}");
}

/// Column-major complex outputs bit-identical (real+imag native bits).
#[cfg(feature = "complex")]
pub fn cplx_bits_identical<T: CElem>(x: &[T], y: &[T]) -> bool {
    x.len() == y.len()
        && x.iter().zip(y).all(|(a, b)| {
            let (ar, ai) = a.parts();
            let (br, bi) = b.parts();
            ar.to_bits() == br.to_bits() && ai.to_bits() == bi.to_bits()
        })
}
