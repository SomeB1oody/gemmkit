//! Shared test harness: element traits, random fills, views, references, accuracy gates.

use gemmkit::{MatMut, MatRef, Parallelism, gemm};

/// Trait letting the harness be generic over f32/f64.
pub(crate) trait Elem: gemmkit::GemmScalar {
    const EPS: f64;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}
impl Elem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64) -> Self {
        x as f32
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
}

/// Deterministic pseudo-random fill in [-1, 1).
pub(crate) fn rand_vec<T: Elem>(n: usize, seed: u64) -> Vec<T> {
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
pub(crate) struct Mat<T> {
    pub(crate) v: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}
impl<T: Elem> Mat<T> {
    pub(crate) fn rand(rows: usize, cols: usize, seed: u64) -> Self {
        Mat {
            v: rand_vec(rows * cols, seed),
            rows,
            cols,
        }
    }
    pub(crate) fn at(&self, i: usize, j: usize) -> T {
        self.v[i * self.cols + j]
    }
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

/// f64 reference: `C <- beta*C0 + alpha*A*B`.
pub(crate) fn reference<T: Elem>(
    a: &Mat<T>,
    b: &Mat<T>,
    c0: &Mat<T>,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
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

/// Relative Frobenius error gate: ||C - Cref|| / (||A||*||B|| + tiny) <= 8*k*eps.
pub(crate) fn assert_accurate<T: Elem>(
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
    let rel = diff2.sqrt() / (na * nb + 1e-30);
    let tol = 8.0 * (k.max(1) as f64) * T::EPS;
    assert!(
        rel <= tol,
        "{ctx}: relative error {rel:.3e} > tol {tol:.3e} (m={m},k={k},n={n})"
    );
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

/// A complex element type the complex test harness is generic over.
#[cfg(feature = "complex")]
pub(crate) trait CElem: gemmkit::ComplexScalar {
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
pub(crate) fn rand_cplx<T: CElem>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    (0..n).map(|_| T::of(next(), next())).collect()
}

/// f64 complex reference (column-major), with conj of A / B as selected.
#[cfg(feature = "complex")]
pub(crate) fn ref_cplx<T: CElem>(
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

#[cfg(feature = "complex")]
pub(crate) fn assert_cplx_accurate<T: CElem>(
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
