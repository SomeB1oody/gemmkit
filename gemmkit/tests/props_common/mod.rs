//! Shared proptest strategies, oracle references, and accuracy gates for the
//! property-based test suite (props_api, props_packed, props_knobs)
//!
//! Pulled in via `mod props_common;` from each `tests/props_*.rs` crate root, all of which
//! are gated `cfg(all(not(miri), not(target_family = "wasm")))`; this module is therefore
//! never compiled under Miri or wasm, and is never itself a test target (cargo only builds
//! top-level `tests/*.rs` and `tests/*/main.rs`). Each including binary uses a different
//! subset of the helpers here, so `#![allow(dead_code)]` applies, same as every other
//! `tests/*_common` module
//!
//! The numeric oracle core (the `Elem`/`CElem` traits, `rand_vec`, `Mat`, the f64
//! `reference`, the complex `ref_cplx`, and the `8*k*eps`/`16*k*eps` accuracy gates) lives
//! once in `tests/oracle_common/mod.rs` and is re-exported below. What this module adds on
//! top is the proptest strategies (dimension, layout, coefficient, parallelism) and the
//! helpers only the property suite needs: the full-range i8 fill, the wrapping-i32 i8
//! reference, `frob_norm`, and the bit-identity checks
#![allow(dead_code)]

use gemmkit::Parallelism;
use proptest::prelude::*;

// File-backed at ../oracle_common/mod.rs; pub-used below for Mat, Elem, CElem, reference,
// ref_cplx, rand_vec/rand_cplx, the accuracy gates, and fast_test (read by cases() below)
#[path = "../oracle_common/mod.rs"]
mod oracle_common;
pub use oracle_common::*;

/// Per-property case count, honoring a `PROPTEST_CASES` override. `ProptestConfig::default()`
/// already resolves that variable into its own `cases` field, but every property block here
/// overwrites that field explicitly with `cases(N)`, which discards whatever
/// `..ProptestConfig::default()` would have produced; this re-reads the override directly so
/// that precedence is not lost
///
/// # Parameters
/// - `default` - tuned case count to use absent any override
///
/// # Returns
/// `u32` - `PROPTEST_CASES` if set and parseable (the Intel SDE CI jobs pin it to 16); else,
///   under `GEMMKIT_FAST_TEST`, `max(default/8, 8)`; else `default`
pub fn cases(default: u32) -> u32 {
    if let Some(n) = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.trim().parse().ok())
    {
        return n;
    }
    if fast_test() {
        return (default / 8).max(8);
    }
    default
}

// layout/stride strategies: PLayout generalizes the Layout enum in
// tests/correctness/common.rs; the checked gemm/gemm_with API only accepts non-negative
// strides, so no variant here ever produces one

#[derive(Copy, Clone, Debug)]
pub enum PLayout {
    /// rs = cols + pad, cs = 1
    Row { pad: usize },
    /// rs = 1, cs = rows + pad
    Col { pad: usize },
    /// General strides: cs drawn from 2..=4 so cs > 1, rs = cols*cs + pad
    General { cs: usize, pad: usize },
}

pub fn layout() -> impl Strategy<Value = PLayout> {
    prop_oneof![
        3 => (0usize..=7).prop_map(|pad| PLayout::Row { pad }),
        3 => (0usize..=7).prop_map(|pad| PLayout::Col { pad }),
        2 => (2usize..=4, 0usize..=5).prop_map(|(cs, pad)| PLayout::General { cs, pad }),
    ]
}

/// (rs, cs) for a `rows x cols` view laid out as `l`
fn strides_for(rows: usize, cols: usize, l: PLayout) -> (usize, usize) {
    match l {
        PLayout::Row { pad } => (cols + pad, 1),
        PLayout::Col { pad } => (1, rows + pad),
        PLayout::General { cs, pad } => (cols * cs + pad, cs),
    }
}

/// Materialize a row-major logical `rows x cols` matrix (`vals`) into a strided buffer laid
/// out as `l`. Generic over `Copy` rather than [`Elem`], so it also serves the i8 element
/// path ([`fill_i8`]/[`ref_i8_wrapping`]), not just the float `Elem` types
///
/// # Parameters
/// - `vals` - source values in row-major order, length `rows*cols`
/// - `rows` - logical row count
/// - `cols` - logical column count
/// - `zero` - fill value for the padding gaps `l` leaves unwritten
/// - `l` - target stride layout
///
/// # Returns
/// `(Vec<T>, isize, isize)` - the strided buffer, its row stride, and its column stride
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

/// Wraps [`build_view_rowmajor`] for `Mat<T: Elem>`, zero-filling padding via `T::from_f64(0.0)`
pub fn build_view<T: Elem>(m: &Mat<T>, l: PLayout) -> (Vec<T>, isize, isize) {
    build_view_rowmajor(&m.v, m.rows, m.cols, T::from_f64(0.0), l)
}

// dimension, coefficient, and parallelism strategies

/// Dimension distribution, weighted toward degenerate cases: heavy on 0 and 1, then
/// boundary triples (V-1, V, V+1) around 12, 16, 32, 48, and the tiny-block shortcut gate 64
/// (`TINY_BLOCK_DIM_DEFAULT` in tuning.rs); 12 and 32 are the AVX-512F f32 microkernel tile's
/// NR/MR, 16 is the FMA/NEON f32 tile's MR. A few extra small values (2, 4, 5, 6, 24) and a
/// broad 2..=96 tail round it out
pub fn dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        2 => Just(0usize),
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
    ]
}

/// Like [`dim`] but never draws 0: in the packed-path tests, C is built through
/// `PLayout::Col`/`PLayout::Row`, and a 0-length row/col there collapses that layout's
/// stride to `pad`, which can trip the packed API's column-major-ish/row-major-ish
/// (`|csc| >= |rsc|` / `|csc| <= |rsc|`) orientation guard on a shape that was never meant
/// to probe that guard. Used for the `m`/`n` draws so a generated C always clears it
pub fn pos_dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
    ]
}

/// `k` like [`dim`] but with an extra tail straddling the default kc block boundary
/// (`KC_DEFAULT` = 512 in tuning.rs): 200, 511, 512, 513
pub fn kdim() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => dim(),
        1 => proptest::sample::select(&[200usize, 511, 512, 513][..]),
    ]
}

/// `k` like [`kdim`] but never 0, so every draw genuinely exercises the A*B contraction: a
/// `k == 0` draw collapses A*B to an empty sum and only tests beta-scaling of C, which the
/// dedicated `k == 0` overwrite properties in props_api already cover (there, k is drawn
/// from [`kdim`] with beta pinned to 0, so the reference result is exactly 0)
pub fn kdim_pos() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => Just(1usize),
        5 => proptest::sample::select(
            &[2usize, 4, 5, 6, 11, 12, 13, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65][..]),
        8 => 2usize..=96,
        1 => proptest::sample::select(&[200usize, 511, 512, 513][..]),
    ]
}

/// Alpha/beta value pool, overlapping the combos `correctness_alpha_beta` in
/// tests/correctness/float.rs sweeps, plus a few extra edge values (-1, 0.5, a near-zero
/// 1e-3); alpha and beta each draw independently from this same set. No NaN/Inf: those have
/// no documented gemm contract
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

// property-suite-only numeric helpers (the oracle core is re-exported above)

/// Frobenius norm of `m`'s row-major values, computed in widened f64
pub fn frob_norm<T: Elem>(m: &Mat<T>) -> f64 {
    m.v.iter().map(|x| x.to_f64().powi(2)).sum::<f64>().sqrt()
}

/// `true` if `x` and `y` are the same length and equal element-for-element by native bit
/// pattern (not IEEE `==`, so -0.0 and 0.0, or differing NaN payloads, compare unequal)
pub fn bits_identical<T: Elem>(x: &[T], y: &[T]) -> bool {
    x.len() == y.len()
        && x.iter()
            .zip(y)
            .all(|(a, b)| a.to_bits_u64() == b.to_bits_u64())
}

// integer (i8 -> i32) property-suite-only helpers

/// Deterministic full-range i8 fill: unlike tests/correctness/common.rs's range-limited
/// `rand_i8` ([-100, 100]), this spans all of i8 ([-128, 127]) so wrapping accumulation on
/// overflow is actually exercised
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

/// Row-major wrapping-i32 GEMM reference: matches the documented i8 contract of
/// two's-complement wrapping i32 arithmetic (see `i8_wraps_on_overflow` in
/// tests/correctness/int8.rs). Wrapping add is associative and commutative, so this
/// reference is order-independent, and the kernel under test must match it exactly
/// regardless of its own summation order
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

// complex (c32 / c64) property-suite-only helper (the oracle core is re-exported above)

/// `true` if `x` and `y` are the same length and equal element-for-element, real and
/// imaginary parts each compared by native bit pattern (not IEEE `==`)
#[cfg(feature = "complex")]
pub fn cplx_bits_identical<T: CElem>(x: &[T], y: &[T]) -> bool {
    x.len() == y.len()
        && x.iter().zip(y).all(|(a, b)| {
            let (ar, ai) = a.parts();
            let (br, bi) = b.parts();
            ar.to_bits() == br.to_bits() && ai.to_bits() == bi.to_bits()
        })
}
