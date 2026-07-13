//! Proptest-free numeric oracle core, single-sourced for both test harnesses.
//!
//! The `Elem` / `CElem` element traits, the deterministic random fills, the row-major
//! `Mat` view, the f64 `reference` GEMM, the complex `ref_cplx`, and the relative-Frobenius
//! accuracy gates live here once. Both `tests/correctness/common.rs` and
//! `tests/props_common/mod.rs` `#[path]`-include this file and `pub use` it, so this is the
//! single source of truth — edit here, never in a copy.
//!
//! Pulled in via `#[path = "../oracle_common/mod.rs"] mod oracle_common;`; it is never a
//! test target of its own (cargo only builds top-level `tests/*.rs` and `tests/*/main.rs`).
//! Each including binary uses a different subset of these helpers, so `dead_code` is allowed.
#![allow(dead_code)]

use gemmkit::GemmScalar;

/// Trait letting the harness be generic over the homogeneous float element types
/// (f32/f64, and — under the `half` feature — f16/bf16).
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
// final round, the f32 accumulation being far more accurate.
// `half`'s hardware `to_f64`/`from_f64` are inline asm on aarch64 (Miri can't interpret
// it), so under `cfg(miri)` the harness routes through `half`'s pure-software `*_const`
// conversions — bit-equivalent, keeping the mixed-precision scalar path exercisable under
// Miri (the gemmkit-internal conversions are handled the same way in src/scalar.rs).
#[cfg(feature = "half")]
impl Elem for gemmkit::f16 {
    const EPS: f64 = 9.765625e-4; // 2^-10
    fn to_f64(self) -> f64 {
        #[cfg(not(miri))]
        {
            self.to_f64()
        }
        #[cfg(miri)]
        {
            self.to_f64_const()
        }
    }
    fn from_f64(x: f64) -> Self {
        #[cfg(not(miri))]
        {
            gemmkit::f16::from_f64(x)
        }
        #[cfg(miri)]
        {
            gemmkit::f16::from_f64_const(x)
        }
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}
#[cfg(feature = "half")]
impl Elem for gemmkit::bf16 {
    const EPS: f64 = 7.8125e-3; // 2^-7
    fn to_f64(self) -> f64 {
        #[cfg(not(miri))]
        {
            self.to_f64()
        }
        #[cfg(miri)]
        {
            self.to_f64_const()
        }
    }
    fn from_f64(x: f64) -> Self {
        #[cfg(not(miri))]
        {
            gemmkit::bf16::from_f64(x)
        }
        #[cfg(miri)]
        {
            gemmkit::bf16::from_f64_const(x)
        }
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
}

/// Deterministic pseudo-random fill in [-1, 1).
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

/// Relative Frobenius error gate: `rel <= 8*k*eps`, where the denominator is the textbook
/// GEMM backward-error magnitude `||A||*||B|| + denom_extra`, and `denom_extra` carries the
/// `|beta|*||C0||` term. The correctness suite keeps tiny dims on `beta == 0` so `||A||*||B||`
/// alone bounds the output there (it passes `denom_extra == 0`, reducing to the canonical
/// gate); the property suite draws beta over the full (dim, beta) space, so it must add the
/// `beta*C0` contribution or a tiny `k` with a dominant `beta*C0` term would spuriously fail.
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

/// A complex element type the complex test harness is generic over.
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

/// Relative Frobenius gate for the column-major complex output: `rel <= 16*k*eps`.
#[cfg(feature = "complex")]
pub fn assert_cplx_accurate<T: CElem>(
    got: &[T],
    m: usize,
    n: usize,
    cref: &[(f64, f64)],
    k: usize,
    ctx: &str,
) {
    // Relative error over the whole matrix (column-major `got`).
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
