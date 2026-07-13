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
//! The proptest-free numeric oracle core (element traits, `rand_vec`, `Mat`, the f64
//! `reference`, the complex `ref_cplx`, and the `8·k·EPS` / `16·k·EPS` accuracy gates) is
//! single-sourced in `tests/oracle_common/mod.rs` and re-exported below; this module adds
//! the proptest strategies and the property-suite-only helpers (full-range i8 fill, the
//! wrapping-i32 i8 reference, `frob_norm`, and the bit-identity checks).
#![allow(dead_code)]

use gemmkit::Parallelism;
use proptest::prelude::*;

#[path = "../oracle_common/mod.rs"]
mod oracle_common;
pub use oracle_common::*;

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
// property-suite-only numeric helpers (the oracle core is re-exported above)
// ---------------------------------------------------------------------------

/// Frobenius norm of a logical matrix (over its widened f64 values).
pub fn frob_norm<T: Elem>(m: &Mat<T>) -> f64 {
    m.v.iter().map(|x| x.to_f64().powi(2)).sum::<f64>().sqrt()
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
// complex (c32 / c64) property-suite-only helper (the oracle core is re-exported above)
// ---------------------------------------------------------------------------

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
