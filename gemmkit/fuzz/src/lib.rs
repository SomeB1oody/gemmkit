//! Shared harness for the gemmkit fuzz targets.
//!
//! Everything the five targets need lives here so the target files stay thin
//! (`fuzz_target!(|p| run_x(p))`) and the differential-oracle logic is in one
//! testable place. Four targets feed *valid-by-construction* problems and treat
//! **any** panic as a bug; `fuzz_api_validation` instead drives adversarial
//! geometry into the checked APIs and accepts documented `gemmkit:` panics.
//!
//! Numerical bars mirror `gemmkit/tests/correctness.rs`: the `8·k·EPS` /
//! `16·k·EPS` relative-Frobenius gates, the `beta == 0` "C is not read" rule, and
//! the exact wrapping-i32 reference for `i8`. Each `Plan` carries already-bounded,
//! resolved values (manual `Arbitrary`), so a crash artifact decoded with
//! `cargo fuzz fmt` is directly translatable into a stable regression test.

use arbitrary::{Arbitrary, Result, Unstructured};
use gemmkit::{
    BatchProblem, ComplexScalar, Complex, GemmScalar, MatMut, MatRef, Parallelism, Workspace,
    bf16, c32, c64, f16, gemm, gemm_batched, gemm_batched_slice, gemm_cplx, gemm_i8, gemm_packed_a,
    gemm_packed_b, gemm_with, prepack_lhs, prepack_rhs, tuning,
};

// ---------------------------------------------------------------------------
// element tables and generators
// ---------------------------------------------------------------------------

/// alpha/beta values for the float/complex gates, mirroring the sets exercised by
/// `tests/correctness.rs` (e.g. `correctness_f16_layouts`). `0.0` first so the
/// `beta == 0` "C not read" contract is well-represented.
pub const AB_TABLE: [f64; 6] = [0.0, 1.0, -1.0, 0.5, 0.75, 2.5];

/// Integer alpha/beta: `gemm_i8` takes `i32`, so the float table's `0.5`/`0.75`
/// would truncate to `0` and collapse half of it — use a dedicated integer table.
pub const I8_AB_TABLE: [i32; 6] = [0, 1, -1, 2, 3, -2];

/// Distinctive `i32` fill for the gap slots of an `i8`-GEMM output buffer.
const I32_CANARY: i32 = 0x0BAD_F00Du32 as i32;

/// xorshift matching `rand_vec` (`tests/correctness.rs`), used to fill each operand
/// from a single 8-byte plan seed so `-max_len` never starves per-element entropy.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    #[inline]
    fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A full-range `i8` (used directly for `i8` operands).
    #[inline]
    pub fn next_i8(&mut self) -> i8 {
        (self.step() >> 24) as u8 as i8
    }
    /// `i8 / 64.0` — magnitude ≤ 2 and exactly representable in every float type
    /// (denominator `2^6`), so the tolerance gate stays meaningful.
    #[inline]
    pub fn next_quant(&mut self) -> f64 {
        self.next_i8() as f64 / 64.0
    }
}

// ---------------------------------------------------------------------------
// element traits: numeric conversion + gap canary
// ---------------------------------------------------------------------------

/// Bit-pattern sentinel written into the non-view "gap" slots of an output buffer;
/// the driver must never touch those, so a changed sentinel is a stray write.
pub trait Canary: Copy {
    const SENTINEL: Self;
    fn is_sentinel(self) -> bool;
}
impl Canary for f32 {
    const SENTINEL: f32 = f32::from_bits(0x7FC0_ABCD);
    fn is_sentinel(self) -> bool {
        self.to_bits() == 0x7FC0_ABCD
    }
}
impl Canary for f64 {
    const SENTINEL: f64 = f64::from_bits(0x7FF8_0000_0000_ABCD);
    fn is_sentinel(self) -> bool {
        self.to_bits() == 0x7FF8_0000_0000_ABCD
    }
}
impl Canary for f16 {
    const SENTINEL: f16 = f16::from_bits(0x7E01);
    fn is_sentinel(self) -> bool {
        self.to_bits() == 0x7E01
    }
}
impl Canary for bf16 {
    const SENTINEL: bf16 = bf16::from_bits(0x7FC1);
    fn is_sentinel(self) -> bool {
        self.to_bits() == 0x7FC1
    }
}
impl Canary for i32 {
    const SENTINEL: i32 = I32_CANARY;
    fn is_sentinel(self) -> bool {
        self == I32_CANARY
    }
}
impl Canary for c32 {
    const SENTINEL: c32 = Complex::new(<f32 as Canary>::SENTINEL, <f32 as Canary>::SENTINEL);
    fn is_sentinel(self) -> bool {
        self.re.is_sentinel() && self.im.is_sentinel()
    }
}
impl Canary for c64 {
    const SENTINEL: c64 = Complex::new(<f64 as Canary>::SENTINEL, <f64 as Canary>::SENTINEL);
    fn is_sentinel(self) -> bool {
        self.re.is_sentinel() && self.im.is_sentinel()
    }
}

/// A real GEMM element (f32/f64/f16/bf16): construction, f64 view, and its EPS.
pub trait RealElem: GemmScalar + Canary {
    const EPS: f64;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}
impl RealElem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl RealElem for f64 {
    const EPS: f64 = f64::EPSILON;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}
// Narrow types accumulate in f32 and round outputs to 16 bits, so EPS is the 16-bit
// machine epsilon (tests/correctness.rs:55,79).
impl RealElem for f16 {
    const EPS: f64 = 9.765625e-4; // 2^-10
    fn from_f64(x: f64) -> Self {
        f16::from_f64(x)
    }
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
}
impl RealElem for bf16 {
    const EPS: f64 = 7.8125e-3; // 2^-7
    fn from_f64(x: f64) -> Self {
        bf16::from_f64(x)
    }
    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }
}

/// A complex GEMM element (c32/c64).
pub trait CplxElem: ComplexScalar + Canary {
    const EPS: f64;
    fn make(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
impl CplxElem for c32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn make(re: f64, im: f64) -> Self {
        Complex::new(re as f32, im as f32)
    }
    fn parts(self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
}
impl CplxElem for c64 {
    const EPS: f64 = f64::EPSILON;
    fn make(re: f64, im: f64) -> Self {
        Complex::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

// ---------------------------------------------------------------------------
// layout plans and operand construction
// ---------------------------------------------------------------------------

/// Generalizes the Row/Col/GeneralPad layouts of `tests/correctness.rs` to an
/// interleave (`il`) × trailing-pad (`pad`) family. `BroadcastRow` (`rs = 0`) is a
/// self-aliasing view legal only for the read-only operands A/B.
#[derive(Debug, Clone, Copy)]
pub enum LayoutPlan {
    RowIsh { il: usize, pad: usize },
    ColIsh { il: usize, pad: usize },
    BroadcastRow,
}

impl LayoutPlan {
    /// `(rs, cs)` for a `rows × cols` view. Mirrors the extent formula of
    /// `api.rs::extent` (all strides here are non-negative).
    pub fn strides(self, rows: usize, cols: usize) -> (isize, isize) {
        match self {
            LayoutPlan::RowIsh { il, pad } => ((cols * il + pad) as isize, il as isize),
            LayoutPlan::ColIsh { il, pad } => (il as isize, (rows * il + pad) as isize),
            LayoutPlan::BroadcastRow => (0, 1),
        }
    }
    fn arbitrary_general(u: &mut Unstructured, allow_broadcast: bool) -> Result<Self> {
        let hi: u8 = if allow_broadcast { 2 } else { 1 };
        let il = u.int_in_range(1usize..=3)?;
        let pad = u.int_in_range(0usize..=4)?;
        Ok(match u.int_in_range(0u8..=hi)? {
            0 => LayoutPlan::RowIsh { il, pad },
            1 => LayoutPlan::ColIsh { il, pad },
            _ => LayoutPlan::BroadcastRow,
        })
    }
}

/// Highest slice offset (exclusive) of a non-negative-stride view — mirror of
/// `api.rs::extent` for the strides this harness builds (never negative/overflowing).
pub fn extent_of(rows: usize, cols: usize, rs: isize, cs: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    ((rows - 1) as isize * rs + (cols - 1) as isize * cs) as usize + 1
}

/// Allocate exactly the extent a `rows × cols` view needs, fill its view slots
/// through the strides, and return `(buf, rs, cs)`. Gap slots keep `fill`.
pub fn build_operand<T: Copy>(
    rows: usize,
    cols: usize,
    lp: LayoutPlan,
    fill: T,
    mut genf: impl FnMut() -> T,
) -> (Vec<T>, isize, isize) {
    let (rs, cs) = lp.strides(rows, cols);
    let extent = extent_of(rows, cols, rs, cs);
    let mut buf = vec![fill; extent];
    for i in 0..rows {
        for j in 0..cols {
            buf[(i as isize * rs + j as isize * cs) as usize] = genf();
        }
    }
    (buf, rs, cs)
}

/// Assert the driver never wrote a slot outside the `rows × cols` view — the cheapest
/// detector for the stride/epilogue out-of-bounds-write class the layouts probe.
pub fn assert_no_gap_writes<T: Canary>(
    buf: &[T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
    ctx: &str,
) {
    let extent = extent_of(rows, cols, rs, cs);
    let mut is_view = vec![false; extent];
    for i in 0..rows {
        for j in 0..cols {
            is_view[(i as isize * rs + j as isize * cs) as usize] = true;
        }
    }
    for (idx, slot) in buf.iter().enumerate() {
        if idx < extent && !is_view[idx] && !slot.is_sentinel() {
            panic!("{ctx}: gap slot {idx} overwritten (out-of-view write; strides {rs},{cs})");
        }
    }
}

// ---------------------------------------------------------------------------
// dense materialization + references
// ---------------------------------------------------------------------------

fn dense_real<T: RealElem>(buf: &[T], rows: usize, cols: usize, rs: isize, cs: isize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize].to_f64();
        }
    }
    out
}

fn dense_i32_from_i8(buf: &[i8], rows: usize, cols: usize, rs: isize, cs: isize) -> Vec<i32> {
    let mut out = vec![0i32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize] as i32;
        }
    }
    out
}

fn dense_i32(buf: &[i32], rows: usize, cols: usize, rs: isize, cs: isize) -> Vec<i32> {
    let mut out = vec![0i32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = buf[(i as isize * rs + j as isize * cs) as usize];
        }
    }
    out
}

/// Materialize a complex view row-major in f64, applying `conj` to the imaginary part
/// (mirrors `api.rs::gemm_cplx`, which conjugates the *operand* before the product).
fn dense_cplx<T: CplxElem>(
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

fn frob(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// f64 reference `C <- beta·C0 + alpha·A·B` with the `beta == 0` "C not read" rule
/// (tests/correctness.rs:195-199), so a NaN-seeded C (the beta==0 fuzz) never taints it.
fn ref_gemm_real(
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
            let base = if beta == 0.0 { 0.0 } else { beta * dc0[i * n + j] };
            out[i * n + j] = base + alpha * acc;
        }
    }
    out
}

/// Exact wrapping-i32 reference; `i32` accumulation is associative mod 2^32, so every
/// blocking/threading/ISA schedule reproduces it bit-for-bit (the `assert_eq!` bar).
fn ref_gemm_i8(
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

/// f64 complex reference (conj already baked into `da`/`db`) with the beta==0 rule —
/// `ref_cplx` in the suite has no such rule, so this closes the NaN-C false positive.
fn ref_gemm_cplx(
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

// ---------------------------------------------------------------------------
// tolerance / exact gates (panic == the libFuzzer report channel)
// ---------------------------------------------------------------------------

/// Relative-Frobenius gate. The denominator is the `8·k·EPS` gate of
/// `assert_accurate` (tests/correctness.rs:232-237), `||A||·||B||`, augmented with the
/// `|alpha|` product scale and the `|beta|·||C0||` epilogue scale. The suite's gate
/// omits both because its inputs are always `k >= 1` with O(1) alpha/non-empty A·B; the
/// fuzzer reaches `k == 0` (empty operands, `||A||·||B|| == 0`) and `|alpha| = 2.5`,
/// where the epilogue/product rounding — correct in the type but higher-precision in the
/// f64 reference — would otherwise blow up a zero denominator. `denom` is precomputed by
/// the caller (with the `||C0||` term dropped when `beta == 0`, so a NaN-seeded C0 can't
/// taint it).
fn real_gate<T: RealElem>(
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

/// The gate denominator: `|alpha|·||A||·||B|| + |beta|·||C0|| + tiny`. `nc0` must be `0`
/// when `beta == 0` (the C0 term is dropped and C0 may be NaN-seeded).
fn real_denom(alpha_f: f64, na: f64, nb: f64, beta_f: f64, nc0: f64) -> f64 {
    alpha_f.abs() * na * nb + beta_f.abs() * nc0 + 1e-30
}

fn cplx_gate<T: CplxElem>(
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

fn i8_gate(
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

// ---------------------------------------------------------------------------
// generic differential drivers (shared across targets)
// ---------------------------------------------------------------------------

/// A moderately sized fixed problem run through a caller `Workspace` before the plan
/// problem, so the reuse path sees a shape change (grow or shrink) — the
/// `workspace_alloc.rs` axis the thread-local pool alone never exercises.
fn warm_ws<T: RealElem>(ws: &mut Workspace) {
    let (m, k, n) = (16usize, 16usize, 16usize);
    let mut rr = Rng::new(0xA11CE);
    let a: Vec<T> = (0..m * k).map(|_| T::from_f64(rr.next_quant())).collect();
    let b: Vec<T> = (0..k * n).map(|_| T::from_f64(rr.next_quant())).collect();
    let mut c: Vec<T> = vec![T::ZERO; m * n];
    gemm_with(
        ws,
        T::ONE,
        MatRef::new(&a, m, k, k as isize, 1),
        MatRef::new(&b, k, n, n as isize, 1),
        T::ZERO,
        MatMut::new(&mut c, m, n, n as isize, 1),
        Parallelism::Serial,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn differential_gemm_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    nan_c: bool,
    par: Parallelism,
    ws_reuse: bool,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    let seed_nan = nan_c && beta_f == 0.0;
    let (mut cbuf, rsc, csc) = build_operand(m, n, lc, T::SENTINEL, || {
        if seed_nan {
            T::from_f64(f64::NAN)
        } else {
            T::from_f64(rc.next_quant())
        }
    });

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    let a = MatRef::new(&abuf, m, k, rsa, csa);
    let b = MatRef::new(&bbuf, k, n, rsb, csb);
    if ws_reuse {
        let mut ws = Workspace::new();
        warm_ws::<T>(&mut ws);
        gemm_with(
            &mut ws,
            alpha,
            a,
            b,
            beta,
            MatMut::new(&mut cbuf, m, n, rsc, csc),
            par,
        );
    } else {
        gemm(alpha, a, b, beta, MatMut::new(&mut cbuf, m, n, rsc, csc), par);
    }

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

#[allow(clippy::too_many_arguments)]
pub fn differential_gemm_i8(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha: i32,
    beta: i32,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand::<i8>(m, k, la, 0, || ra.next_i8());
    let (bbuf, rsb, csb) = build_operand::<i8>(k, n, lb, 0, || rb.next_i8());
    // C0 elements in [-128, 127] so the i32 epilogue stays in a sane magnitude.
    let (mut cbuf, rsc, csc) =
        build_operand::<i32>(m, n, lc, i32::SENTINEL, || rc.next_i8() as i32);

    let da = dense_i32_from_i8(&abuf, m, k, rsa, csa);
    let db = dense_i32_from_i8(&bbuf, k, n, rsb, csb);
    let dc0 = dense_i32(&cbuf, m, n, rsc, csc);
    let cref = ref_gemm_i8(&da, &db, &dc0, m, k, n, alpha, beta);

    gemm_i8(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    i8_gate(&cbuf, rsc, csc, m, n, &cref, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

#[allow(clippy::too_many_arguments)]
fn differential_gemm_cplx<T: CplxElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha: (f64, f64),
    beta: (f64, f64),
    conj_a: bool,
    conj_b: bool,
    nan_c: bool,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let al = T::make(alpha.0, alpha.1);
    let be = T::make(beta.0, beta.1);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) =
        build_operand::<T>(m, k, la, T::ZERO, || T::make(ra.next_quant(), ra.next_quant()));
    let (bbuf, rsb, csb) =
        build_operand::<T>(k, n, lb, T::ZERO, || T::make(rb.next_quant(), rb.next_quant()));
    let seed_nan = nan_c && beta == (0.0, 0.0);
    let (mut cbuf, rsc, csc) = build_operand::<T>(m, n, lc, T::SENTINEL, || {
        if seed_nan {
            T::make(f64::NAN, f64::NAN)
        } else {
            T::make(rc.next_quant(), rc.next_quant())
        }
    });

    let da = dense_cplx(&abuf, m, k, rsa, csa, conj_a);
    let db = dense_cplx(&bbuf, k, n, rsb, csb, conj_b);
    let dc0 = dense_cplx(&cbuf, m, n, rsc, csc, false);
    let cref = ref_gemm_cplx(&da, &db, &dc0, m, k, n, alpha, beta);

    gemm_cplx(
        al,
        MatRef::new(&abuf, m, k, rsa, csa),
        conj_a,
        MatRef::new(&bbuf, k, n, rsb, csb),
        conj_b,
        be,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    cplx_gate::<T>(&cbuf, rsc, csc, m, n, &cref, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Prepacked-RHS round trip: `prepack_rhs(B)` then `gemm_packed_b` with column-major-ish
/// C (the orientation the API requires). Gate is tolerance, not bitwise, per the API's
/// tiny/gemv last-ULP allowance.
#[allow(clippy::too_many_arguments)]
pub fn differential_packed_b_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    // Column-major-ish C (|csc| >= |rsc|); dims are >= 1 so the invariant holds.
    let lc = LayoutPlan::ColIsh { il: 1, pad: 1 };
    let (mut cbuf, rsc, csc) = build_operand(m, n, lc, T::SENTINEL, || T::from_f64(rc.next_quant()));

    let packed = prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
    if packed.rows() != k || packed.cols() != n {
        panic!("{ctx}: prepack_rhs echo mismatch: rows {} cols {}", packed.rows(), packed.cols());
    }

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    gemm_packed_b(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        &packed,
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Prepacked-LHS round trip: `prepack_lhs(A)` then `gemm_packed_a` with row-major-ish C.
#[allow(clippy::too_many_arguments)]
pub fn differential_packed_a_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    // Row-major-ish C (|csc| <= |rsc|); dims are >= 1 so the invariant holds.
    let lc = LayoutPlan::RowIsh { il: 1, pad: 1 };
    let (mut cbuf, rsc, csc) = build_operand(m, n, lc, T::SENTINEL, || T::from_f64(rc.next_quant()));

    let packed = prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
    if packed.rows() != m || packed.cols() != k {
        panic!("{ctx}: prepack_lhs echo mismatch: rows {} cols {}", packed.rows(), packed.cols());
    }

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    gemm_packed_a(
        alpha,
        &packed,
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Strided-batched GEMM over one big buffer per operand (batch strides valid by
/// construction) plus a `gemm_batched_slice` cross-check over per-element buffers.
#[allow(clippy::too_many_arguments)]
pub fn differential_batched_real<T: RealElem>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    a_broadcast: bool,
    b_broadcast: bool,
    a_bs_pad: usize,
    b_bs_pad: usize,
    c_bs_pad: usize,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let (rsa, csa) = la.strides(m, k);
    let (rsb, csb) = lb.strides(k, n);
    let (rsc, csc) = lc.strides(m, n);
    let ea = extent_of(m, k, rsa, csa);
    let eb = extent_of(k, n, rsb, csb);
    let ec = extent_of(m, n, rsc, csc);
    // Batch strides: 0 broadcasts a read-only operand; C must clear one element extent.
    let a_bs = if a_broadcast { 0 } else { ea + a_bs_pad };
    let b_bs = if b_broadcast { 0 } else { eb + b_bs_pad };
    let c_bs = ec + c_bs_pad;

    let a_len = if batch <= 1 { ea } else { (batch - 1) * a_bs + ea };
    let b_len = if batch <= 1 { eb } else { (batch - 1) * b_bs + eb };
    let c_len = if batch <= 1 { ec } else { (batch - 1) * c_bs + ec };

    let mut ra = Rng::new(seed ^ 0x0A);
    let mut rb = Rng::new(seed ^ 0x0B);
    let mut rc = Rng::new(seed ^ 0x0C);
    let mut abuf = vec![T::ZERO; a_len];
    let mut bbuf = vec![T::ZERO; b_len];
    let mut cbuf = vec![T::SENTINEL; c_len];
    for e in 0..batch {
        let base = e * a_bs;
        for i in 0..m {
            for j in 0..k {
                abuf[base + (i as isize * rsa + j as isize * csa) as usize] =
                    T::from_f64(ra.next_quant());
            }
        }
        let base = e * b_bs;
        for i in 0..k {
            for j in 0..n {
                bbuf[base + (i as isize * rsb + j as isize * csb) as usize] =
                    T::from_f64(rb.next_quant());
            }
        }
        let base = e * c_bs;
        for i in 0..m {
            for j in 0..n {
                cbuf[base + (i as isize * rsc + j as isize * csc) as usize] =
                    T::from_f64(rc.next_quant());
            }
        }
    }

    // Per-element references before the call.
    let mut refs: Vec<(Vec<f64>, f64)> = Vec::with_capacity(batch);
    for e in 0..batch {
        let da = dense_real(&abuf[e * a_bs..], m, k, rsa, csa);
        let db = dense_real(&bbuf[e * b_bs..], k, n, rsb, csb);
        let dc0 = dense_real(&cbuf[e * c_bs..], m, n, rsc, csc);
        let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
        let denom = real_denom(alpha_f, frob(&da), frob(&db), beta_f, nc0);
        let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);
        refs.push((cref, denom));
    }

    gemm_batched(
        batch,
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        a_bs as isize,
        MatRef::new(&bbuf, k, n, rsb, csb),
        b_bs as isize,
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        c_bs as isize,
        par,
    );

    for e in 0..batch {
        let (cref, denom) = &refs[e];
        let slot = &cbuf[e * c_bs..e * c_bs + ec];
        real_gate::<T>(slot, rsc, csc, m, n, cref, *denom, k, ctx);
    }
    // Whole-buffer canary: element gaps AND inter-element gaps must be untouched.
    assert_batched_no_gap_writes(&cbuf, batch, m, n, rsc, csc, c_bs, ctx);

    // Second entry point: pointer-array batched over per-element buffers.
    if batch >= 1 {
        batched_slice_real::<T>(batch, m, k, n, alpha_f, beta_f, par, seed, ctx);
    }
}

fn assert_batched_no_gap_writes<T: Canary>(
    buf: &[T],
    batch: usize,
    m: usize,
    n: usize,
    rsc: isize,
    csc: isize,
    c_bs: usize,
    ctx: &str,
) {
    let ec = extent_of(m, n, rsc, csc);
    let mut is_view = vec![false; buf.len()];
    for e in 0..batch {
        let base = e * c_bs;
        for i in 0..m {
            for j in 0..n {
                is_view[base + (i as isize * rsc + j as isize * csc) as usize] = true;
            }
        }
    }
    for (idx, slot) in buf.iter().enumerate() {
        if !is_view[idx] && !slot.is_sentinel() {
            panic!("{ctx}: batched gap slot {idx} overwritten (ec={ec}, c_bs={c_bs})");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn batched_slice_real<T: RealElem>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let la = LayoutPlan::RowIsh { il: 1, pad: 0 };
    let lb = LayoutPlan::RowIsh { il: 1, pad: 0 };
    let lc = LayoutPlan::RowIsh { il: 1, pad: 1 };
    let (rsa, csa) = la.strides(m, k);
    let (rsb, csb) = lb.strides(k, n);
    let (rsc, csc) = lc.strides(m, n);

    let mut a_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    let mut b_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    let mut c_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    for e in 0..batch {
        let mut ra = Rng::new(seed ^ (e as u64).wrapping_mul(0x9E37) ^ 0x51CE);
        let (ab, _, _) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
        let (bb, _, _) = build_operand(k, n, lb, T::ZERO, || T::from_f64(ra.next_quant()));
        let (cb, _, _) = build_operand(m, n, lc, T::SENTINEL, || T::from_f64(ra.next_quant()));
        a_bufs.push(ab);
        b_bufs.push(bb);
        c_bufs.push(cb);
    }

    let mut refs: Vec<(Vec<f64>, f64)> = Vec::with_capacity(batch);
    for e in 0..batch {
        let da = dense_real(&a_bufs[e], m, k, rsa, csa);
        let db = dense_real(&b_bufs[e], k, n, rsb, csb);
        let dc0 = dense_real(&c_bufs[e], m, n, rsc, csc);
        let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
        let denom = real_denom(alpha_f, frob(&da), frob(&db), beta_f, nc0);
        let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);
        refs.push((cref, denom));
    }

    let mut problems: Vec<BatchProblem<T>> = Vec::with_capacity(batch);
    for ((ab, bb), cb) in a_bufs.iter().zip(b_bufs.iter()).zip(c_bufs.iter_mut()) {
        problems.push(BatchProblem {
            alpha,
            a: MatRef::new(ab, m, k, rsa, csa),
            b: MatRef::new(bb, k, n, rsb, csb),
            beta,
            c: MatMut::new(cb, m, n, rsc, csc),
        });
    }
    gemm_batched_slice(&mut problems, par);
    drop(problems);

    for e in 0..batch {
        let (cref, denom) = &refs[e];
        real_gate::<T>(&c_bufs[e], rsc, csc, m, n, cref, *denom, k, ctx);
        assert_no_gap_writes(&c_bufs[e], m, n, rsc, csc, ctx);
    }
}

// ---------------------------------------------------------------------------
// small Arbitrary helpers
// ---------------------------------------------------------------------------

fn arb_par(u: &mut Unstructured) -> Result<Parallelism> {
    // Serial weighted 2x for exec/s; explicit threads capped at 2 (the 32-thread
    // Zen5 auto pool would tank throughput per exec).
    Ok(*u.choose(&[
        Parallelism::Serial,
        Parallelism::Serial,
        Parallelism::Rayon(1),
        Parallelism::Rayon(2),
    ])?)
}

fn arb_par_knobs(u: &mut Unstructured) -> Result<Parallelism> {
    // Knobs additionally exercises Rayon(0) (auto), where the parallel-threshold /
    // thread-dim-stride interplay lives; still Serial-weighted for throughput.
    Ok(*u.choose(&[
        Parallelism::Serial,
        Parallelism::Serial,
        Parallelism::Serial,
        Parallelism::Rayon(1),
        Parallelism::Rayon(2),
        Parallelism::Rayon(0),
    ])?)
}

fn ab_index(u: &mut Unstructured) -> Result<usize> {
    Ok(u.int_in_range(0usize..=5)?)
}

// ===========================================================================
// fuzz_gemm
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub enum TypeTag {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    C32,
    C64,
}

#[derive(Debug)]
pub struct GemmPlan {
    pub ty: TypeTag,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub lc: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub alpha_im_i: usize,
    pub beta_im_i: usize,
    pub nan_c: bool,
    pub conj_a: bool,
    pub conj_b: bool,
    pub ws_reuse: bool,
    pub par: Parallelism,
    pub a_seed: u64,
    pub b_seed: u64,
    pub c_seed: u64,
}

impl<'a> Arbitrary<'a> for GemmPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ty = match u.int_in_range(0u8..=6)? {
            0 => TypeTag::F32,
            1 => TypeTag::F64,
            2 => TypeTag::F16,
            3 => TypeTag::Bf16,
            4 => TypeTag::I8,
            5 => TypeTag::C32,
            _ => TypeTag::C64,
        };
        Ok(GemmPlan {
            ty,
            // m,n cross the AVX-512 f32 tile edges (MR=32, NR=12); k crosses the
            // bf16/VNNI DEPTH_MULTIPLE padding and partial-depth panels.
            m: u.int_in_range(0usize..=48)?,
            k: u.int_in_range(0usize..=130)?,
            n: u.int_in_range(0usize..=48)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            lc: LayoutPlan::arbitrary_general(u, false)?, // self-aliasing C is a documented panic
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            alpha_im_i: ab_index(u)?,
            beta_im_i: ab_index(u)?,
            nan_c: bool::arbitrary(u)?,
            conj_a: bool::arbitrary(u)?,
            conj_b: bool::arbitrary(u)?,
            ws_reuse: bool::arbitrary(u)?,
            par: arb_par(u)?,
            a_seed: u64::arbitrary(u)?,
            b_seed: u64::arbitrary(u)?,
            c_seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_gemm(p: GemmPlan) {
    let ctx = "fuzz_gemm";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    match p.ty {
        TypeTag::F32 => differential_gemm_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed, p.b_seed,
            p.c_seed, ctx,
        ),
        TypeTag::F64 => differential_gemm_real::<f64>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed, p.b_seed,
            p.c_seed, ctx,
        ),
        TypeTag::F16 => differential_gemm_real::<f16>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed, p.b_seed,
            p.c_seed, ctx,
        ),
        TypeTag::Bf16 => differential_gemm_real::<bf16>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed, p.b_seed,
            p.c_seed, ctx,
        ),
        TypeTag::I8 => differential_gemm_i8(
            p.m, p.k, p.n, p.la, p.lb, p.lc, I8_AB_TABLE[p.alpha_i], I8_AB_TABLE[p.beta_i], p.par,
            p.a_seed, p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::C32 => differential_gemm_cplx::<c32>(
            p.m, p.k, p.n, p.la, p.lb, p.lc,
            (af, AB_TABLE[p.alpha_im_i]), (bf, AB_TABLE[p.beta_im_i]),
            p.conj_a, p.conj_b, p.nan_c, p.par, p.a_seed, p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::C64 => differential_gemm_cplx::<c64>(
            p.m, p.k, p.n, p.la, p.lb, p.lc,
            (af, AB_TABLE[p.alpha_im_i]), (bf, AB_TABLE[p.beta_im_i]),
            p.conj_a, p.conj_b, p.nan_c, p.par, p.a_seed, p.b_seed, p.c_seed, ctx,
        ),
    }
}

// ===========================================================================
// fuzz_knobs
// ===========================================================================

/// Every `set_*` compiled on x86_64 + `std,parallel,complex,half,int8`
/// (`gemmkit/src/tuning.rs`); `set_wasm_threads` is wasm-gated and excluded.
pub const KNOB_SETTERS: &[(&str, fn(usize))] = &[
    ("parallel_threshold", tuning::set_parallel_threshold),
    ("rhs_pack_threshold", tuning::set_rhs_pack_threshold),
    ("lhs_pack_threshold", tuning::set_lhs_pack_threshold),
    ("lhs_pack_stride", tuning::set_lhs_pack_stride),
    ("gemv_threshold", tuning::set_gemv_threshold),
    ("small_k_threshold", tuning::set_small_k_threshold),
    ("small_mn_dim", tuning::set_small_mn_dim),
    ("gemv_parallel_bytes", tuning::set_gemv_parallel_bytes),
    ("gemv_thread_cap", tuning::set_gemv_thread_cap),
    ("parallel_oversample", tuning::set_parallel_oversample),
    ("thread_dim_stride", tuning::set_thread_dim_stride),
    ("shared_lhs_mnk", tuning::set_shared_lhs_mnk),
    ("k_stream_max", tuning::set_k_stream_max),
    ("seq_internal_bytes_per_worker", tuning::set_seq_internal_bytes_per_worker),
    ("packed_oversample", tuning::set_packed_oversample),
    ("mc_reg_panels", tuning::set_mc_reg_panels),
    ("nc_no_l3_panels", tuning::set_nc_no_l3_panels),
    ("tiny_block_dim", tuning::set_tiny_block_dim),
    ("kc", tuning::set_kc),
    ("kc_min", tuning::set_kc_min),
    ("pack_transpose_tile", tuning::set_pack_transpose_tile),
    ("i8_vnni_min_par_mnk", tuning::set_i8_vnni_min_par_mnk),
];

pub const N_KNOBS: usize = 22;

/// The knob-value classes from the brief. Setters store unconditionally and clamp
/// `usize::MAX` to `MAX-1` (the UNSET sentinel), so `MAX` exercises the clamp too.
pub fn knob_value(u: &mut Unstructured) -> Result<usize> {
    Ok(match u.int_in_range(0u8..=8)? {
        0 => 0, // 0 = auto convention on several knobs
        1 => 1,
        2 => u.int_in_range(2usize..=17)?, // small
        3 => u.int_in_range(31usize..=65)?, // dim-boundary (tile/tiny edges)
        4 => 4096,        // page-ish (lhs_pack_stride is in bytes)
        5 => 1usize << 33, // > i32/f32-index range
        6 => 1usize << 48, // huge
        7 => usize::MAX - 1,
        _ => usize::MAX, // clamps to UNSET-1 in the setter
    })
}

#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    PlainF32,
    Gemv,
    SmallMn,
    PrepackB,
    PrepackA,
    I8,
    Batched,
}

#[derive(Debug)]
pub struct KnobsPlan {
    pub values: [usize; N_KNOBS],
    pub scenario: Scenario,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for KnobsPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let mut values = [0usize; N_KNOBS];
        for v in values.iter_mut() {
            *v = knob_value(u)?;
        }
        let scenario = match u.int_in_range(0u8..=6)? {
            0 => Scenario::PlainF32,
            1 => Scenario::Gemv,
            2 => Scenario::SmallMn,
            3 => Scenario::PrepackB,
            4 => Scenario::PrepackA,
            5 => Scenario::I8,
            _ => Scenario::Batched,
        };
        Ok(KnobsPlan {
            values,
            scenario,
            m: u.int_in_range(1usize..=24)?,
            k: u.int_in_range(1usize..=24)?,
            n: u.int_in_range(1usize..=24)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par_knobs(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_knobs(p: KnobsPlan) {
    // Set all 22 knobs every input: setters store unconditionally, so each exec fully
    // overwrites the knob set — no state leaks between libFuzzer execs, making every
    // crash artifact self-contained/replayable.
    for (i, (_, setter)) in KNOB_SETTERS.iter().enumerate() {
        setter(p.values[i]);
    }
    let ctx = "fuzz_knobs";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    let (s1, s2, s3) = (p.seed ^ 0x11, p.seed ^ 0x22, p.seed ^ 0x33);
    let lc_row = LayoutPlan::RowIsh { il: 1, pad: 1 };
    match p.scenario {
        Scenario::PlainF32 => differential_gemm_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, lc_row, af, bf, false, p.par, false, s1, s2, s3, ctx,
        ),
        Scenario::Gemv => differential_gemm_real::<f32>(
            p.m, p.k, 1, p.la, p.lb, lc_row, af, bf, false, p.par, false, s1, s2, s3, ctx,
        ),
        Scenario::SmallMn => differential_gemm_real::<f32>(
            p.m.min(8),
            p.k.max(32),
            p.n.min(8),
            p.la,
            p.lb,
            lc_row,
            af,
            bf,
            false,
            p.par,
            false,
            s1,
            s2,
            s3,
            ctx,
        ),
        Scenario::PrepackB => differential_packed_b_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
        ),
        Scenario::PrepackA => differential_packed_a_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
        ),
        Scenario::I8 => differential_gemm_i8(
            p.m, p.k, p.n, p.la, p.lb, lc_row, I8_AB_TABLE[p.alpha_i], I8_AB_TABLE[p.beta_i], p.par,
            s1, s2, s3, ctx,
        ),
        Scenario::Batched => differential_batched_real::<f32>(
            3, p.m, p.k, p.n, p.la, p.lb, lc_row, false, false, 0, 0, 0, af, bf, p.par, p.seed, ctx,
        ),
    }
}

// ===========================================================================
// fuzz_batched
// ===========================================================================

#[derive(Debug)]
pub struct BatchedPlan {
    pub ty64: bool, // false => f32, true => f64
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub lc: LayoutPlan,
    pub a_broadcast: bool,
    pub b_broadcast: bool,
    pub a_bs_pad: usize,
    pub b_bs_pad: usize,
    pub c_bs_pad: usize,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for BatchedPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(BatchedPlan {
            ty64: bool::arbitrary(u)?,
            batch: u.int_in_range(0usize..=4)?, // 0 is the documented no-op
            m: u.int_in_range(1usize..=24)?,
            k: u.int_in_range(1usize..=24)?,
            n: u.int_in_range(1usize..=24)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            lc: LayoutPlan::arbitrary_general(u, false)?,
            a_broadcast: bool::arbitrary(u)?,
            b_broadcast: bool::arbitrary(u)?,
            a_bs_pad: u.int_in_range(0usize..=8)?,
            b_bs_pad: u.int_in_range(0usize..=8)?,
            c_bs_pad: u.int_in_range(0usize..=8)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_batched(p: BatchedPlan) {
    let ctx = "fuzz_batched";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    if p.ty64 {
        differential_batched_real::<f64>(
            p.batch, p.m, p.k, p.n, p.la, p.lb, p.lc, p.a_broadcast, p.b_broadcast, p.a_bs_pad,
            p.b_bs_pad, p.c_bs_pad, af, bf, p.par, p.seed, ctx,
        );
    } else {
        differential_batched_real::<f32>(
            p.batch, p.m, p.k, p.n, p.la, p.lb, p.lc, p.a_broadcast, p.b_broadcast, p.a_bs_pad,
            p.b_bs_pad, p.c_bs_pad, af, bf, p.par, p.seed, ctx,
        );
    }
}

// ===========================================================================
// fuzz_prepack
// ===========================================================================

#[derive(Debug)]
pub struct PrepackPlan {
    pub ty: TypeTag, // F32, F64, or Bf16
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for PrepackPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ty = match u.int_in_range(0u8..=2)? {
            0 => TypeTag::F32,
            1 => TypeTag::F64,
            _ => TypeTag::Bf16, // exercises the DEPTH_MULTIPLE single-slice depth-pad
        };
        Ok(PrepackPlan {
            ty,
            // dims from 1..=48 (crossing tile edges); 0 excluded so the orientation
            // invariant of the packed C holds for empty views too.
            m: u.int_in_range(1usize..=48)?,
            k: u.int_in_range(1usize..=48)?,
            n: u.int_in_range(1usize..=48)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_prepack(p: PrepackPlan) {
    let ctx = "fuzz_prepack";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    let (s1, s2, s3) = (p.seed ^ 0x11, p.seed ^ 0x22, p.seed ^ 0x33);
    macro_rules! both {
        ($t:ty) => {{
            differential_packed_b_real::<$t>(
                p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
            );
            differential_packed_a_real::<$t>(
                p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
            );
        }};
    }
    match p.ty {
        TypeTag::F32 => both!(f32),
        TypeTag::F64 => both!(f64),
        _ => both!(bf16),
    }
}

// ===========================================================================
// fuzz_api_validation
// ===========================================================================

/// The dim class table from the brief; `2^33` targets the extent isize-mul overflow,
/// `usize::MAX/2` / `usize::MAX` the "too large to address" reject.
#[derive(Debug, Clone, Copy)]
pub enum DimClass {
    Zero,
    One,
    Small(usize),
    P31,
    P32p1,
    P33,
    HalfMax,
    Max,
}
impl DimClass {
    pub fn get(self) -> usize {
        match self {
            DimClass::Zero => 0,
            DimClass::One => 1,
            DimClass::Small(s) => s,
            DimClass::P31 => 1usize << 31,
            DimClass::P32p1 => (1usize << 32) + 1,
            DimClass::P33 => 1usize << 33,
            DimClass::HalfMax => usize::MAX / 2,
            DimClass::Max => usize::MAX,
        }
    }
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        Ok(match u.int_in_range(0u8..=7)? {
            0 => DimClass::Zero,
            1 => DimClass::One,
            2 => DimClass::Small(u.int_in_range(2usize..=17)?),
            3 => DimClass::P31,
            4 => DimClass::P32p1,
            5 => DimClass::P33,
            6 => DimClass::HalfMax,
            _ => DimClass::Max,
        })
    }
}

/// Adversarial isize stride table; `isize::MIN/MAX` and `±2^33` drive the checked-mul
/// overflow inside `extent()`.
#[derive(Debug, Clone, Copy)]
pub enum StrideClass {
    Zero,
    P1,
    N1,
    PSmall(isize),
    NSmall(isize),
    P31,
    N31,
    P33,
    N33,
    IMin,
    IMax,
}
impl StrideClass {
    pub fn get(self) -> isize {
        match self {
            StrideClass::Zero => 0,
            StrideClass::P1 => 1,
            StrideClass::N1 => -1,
            StrideClass::PSmall(s) => s,
            StrideClass::NSmall(s) => -s,
            StrideClass::P31 => 1isize << 31,
            StrideClass::N31 => -(1isize << 31),
            StrideClass::P33 => 1isize << 33,
            StrideClass::N33 => -(1isize << 33),
            StrideClass::IMin => isize::MIN,
            StrideClass::IMax => isize::MAX,
        }
    }
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        Ok(match u.int_in_range(0u8..=10)? {
            0 => StrideClass::Zero,
            1 => StrideClass::P1,
            2 => StrideClass::N1,
            3 => StrideClass::PSmall(u.int_in_range(2isize..=17)?),
            4 => StrideClass::NSmall(u.int_in_range(2isize..=17)?),
            5 => StrideClass::P31,
            6 => StrideClass::N31,
            7 => StrideClass::P33,
            8 => StrideClass::N33,
            9 => StrideClass::IMin,
            _ => StrideClass::IMax,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EntryKind {
    Gemm,
    GemmI8,
    GemmCplx,
    Batched,
    PrepackB,
    PrepackA,
}

#[derive(Debug)]
pub struct ValidationPlan {
    pub entry: EntryKind,
    pub len_a: usize,
    pub len_b: usize,
    pub len_c: usize,
    pub m: DimClass,
    pub k: DimClass,
    pub n: DimClass,
    pub mc: DimClass, // C rows (independent, to exercise the shape assert)
    pub nc: DimClass, // C cols
    pub rsa: StrideClass,
    pub csa: StrideClass,
    pub rsb: StrideClass,
    pub csb: StrideClass,
    pub rsc: StrideClass,
    pub csc: StrideClass,
    pub batch: DimClass,
    pub a_bs: StrideClass,
    pub b_bs: StrideClass,
    pub c_bs: StrideClass,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub conj_a: bool,
    pub conj_b: bool,
    pub par: Parallelism,
}

impl<'a> Arbitrary<'a> for ValidationPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let entry = match u.int_in_range(0u8..=5)? {
            0 => EntryKind::Gemm,
            1 => EntryKind::GemmI8,
            2 => EntryKind::GemmCplx,
            3 => EntryKind::Batched,
            4 => EntryKind::PrepackB,
            _ => EntryKind::PrepackA,
        };
        Ok(ValidationPlan {
            entry,
            len_a: u.int_in_range(0usize..=8192)?,
            len_b: u.int_in_range(0usize..=8192)?,
            len_c: u.int_in_range(0usize..=8192)?,
            m: DimClass::arbitrary(u)?,
            k: DimClass::arbitrary(u)?,
            n: DimClass::arbitrary(u)?,
            mc: DimClass::arbitrary(u)?,
            nc: DimClass::arbitrary(u)?,
            rsa: StrideClass::arbitrary(u)?,
            csa: StrideClass::arbitrary(u)?,
            rsb: StrideClass::arbitrary(u)?,
            csb: StrideClass::arbitrary(u)?,
            rsc: StrideClass::arbitrary(u)?,
            csc: StrideClass::arbitrary(u)?,
            batch: DimClass::arbitrary(u)?,
            a_bs: StrideClass::arbitrary(u)?,
            b_bs: StrideClass::arbitrary(u)?,
            c_bs: StrideClass::arbitrary(u)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            conj_a: bool::arbitrary(u)?,
            conj_b: bool::arbitrary(u)?,
            par: arb_par(u)?,
        })
    }
}

/// Skip-cap on accepted-and-expensive plans: a bounded, valid geometry must not turn
/// into billions of MACs or a k-proportional gigabyte alloc under ASan (those would be
/// false-positive timeouts/OOMs, not memory-unsafety). 2^24 elements/MACs.
const WORK_CAP: usize = 1 << 24;

/// Mirror of `api.rs::extent`: highest slice offset (exclusive), or `None` for a
/// negative-stride or too-large-to-address view (both of which validation rejects).
fn mirror_extent(rows: usize, cols: usize, rs: isize, cs: isize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let mut lo: isize = 0;
    let mut hi: isize = 0;
    for &(dim, s) in &[(rows, rs), (cols, cs)] {
        let e = isize::try_from(dim).ok()?.checked_sub(1)?.checked_mul(s)?;
        if e < 0 {
            lo = lo.checked_add(e)?;
        } else {
            hi = hi.checked_add(e)?;
        }
    }
    if lo < 0 {
        None
    } else {
        (hi as usize).checked_add(1)
    }
}

/// Mirror of `api.rs::self_aliases`.
fn mirror_self_aliases(rows: usize, cols: usize, rs: isize, cs: isize) -> bool {
    if rows == 0 || cols == 0 {
        return false;
    }
    let r = (rows > 1).then_some((rs.unsigned_abs(), rows));
    let c = (cols > 1).then_some((cs.unsigned_abs(), cols));
    match (r, c) {
        (None, None) => false,
        (Some((s, _)), None) | (None, Some((s, _))) => s == 0,
        (Some(a), Some(b)) => {
            let (sm, big) = if a.0 <= b.0 { (a, b.0) } else { (b, a.0) };
            sm.0 == 0 || big < sm.0.saturating_mul(sm.1)
        }
    }
}

fn in_bounds(rows: usize, cols: usize, rs: isize, cs: isize, len: usize) -> bool {
    matches!(mirror_extent(rows, cols, rs, cs), Some(need) if need <= len)
}

fn sat3(a: usize, b: usize, c: usize) -> usize {
    a.saturating_mul(b).saturating_mul(c)
}

/// The raw driver behind `fuzz_api_validation`. May panic with a documented
/// `gemmkit:` message (accepted by the target's `catch_unwind`) or run cleanly; the
/// WORK_CAP guard only skips plans that WOULD fully validate and then do unbounded work.
pub fn drive_validation(p: &ValidationPlan) {
    let (m, k, n) = (p.m.get(), p.k.get(), p.n.get());
    let (mc, nc) = (p.mc.get(), p.nc.get());
    let (rsa, csa) = (p.rsa.get(), p.csa.get());
    let (rsb, csb) = (p.rsb.get(), p.csb.get());
    let (rsc, csc) = (p.rsc.get(), p.csc.get());
    let alpha = AB_TABLE[p.alpha_i] as f32;
    let beta = AB_TABLE[p.beta_i] as f32;

    match p.entry {
        EntryKind::Gemm | EntryKind::GemmI8 | EntryKind::GemmCplx => {
            // Would this geometry fully pass validation? If so, cap the compute.
            let would_pass = in_bounds(m, k, rsa, csa, p.len_a)
                && in_bounds(k, n, rsb, csb, p.len_b)
                && in_bounds(mc, nc, rsc, csc, p.len_c)
                && mc == m
                && nc == n
                && !mirror_self_aliases(mc, nc, rsc, csc);
            if would_pass && sat3(m, n, k) > WORK_CAP {
                return;
            }
            match p.entry {
                EntryKind::Gemm => {
                    let a = vec![0.0f32; p.len_a];
                    let b = vec![0.0f32; p.len_b];
                    let mut c = vec![0.0f32; p.len_c];
                    gemm(
                        alpha,
                        MatRef::new(&a, m, k, rsa, csa),
                        MatRef::new(&b, k, n, rsb, csb),
                        beta,
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
                EntryKind::GemmI8 => {
                    let a = vec![0i8; p.len_a];
                    let b = vec![0i8; p.len_b];
                    let mut c = vec![0i32; p.len_c];
                    gemm_i8(
                        I8_AB_TABLE[p.alpha_i],
                        MatRef::new(&a, m, k, rsa, csa),
                        MatRef::new(&b, k, n, rsb, csb),
                        I8_AB_TABLE[p.beta_i],
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
                _ => {
                    let a = vec![c32::ZERO; p.len_a];
                    let b = vec![c32::ZERO; p.len_b];
                    let mut c = vec![c32::ZERO; p.len_c];
                    gemm_cplx(
                        c32::make(alpha as f64, 0.0),
                        MatRef::new(&a, m, k, rsa, csa),
                        p.conj_a,
                        MatRef::new(&b, k, n, rsb, csb),
                        p.conj_b,
                        c32::make(beta as f64, 0.0),
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
            }
        }
        EntryKind::Batched => {
            let batch = p.batch.get();
            let a_bs = p.a_bs.get();
            let b_bs = p.b_bs.get();
            let c_bs = p.c_bs.get();
            // Batched bounds mirror (extent + (batch-1)*bs). A batch stride < 0 with
            // batch>1 is a documented reject, so it is not "would_pass".
            let batched_ok = |rows, cols, rs, cs, bs: isize, len: usize| -> bool {
                let Some(e) = mirror_extent(rows, cols, rs, cs) else {
                    return false;
                };
                if batch <= 1 {
                    return e <= len;
                }
                if bs < 0 {
                    return false;
                }
                let last = (batch - 1).saturating_mul(bs as usize);
                last.saturating_add(e) <= len
            };
            let ec = mirror_extent(mc, nc, rsc, csc);
            let would_pass = batch != 0
                && batched_ok(m, k, rsa, csa, a_bs, p.len_a)
                && batched_ok(k, n, rsb, csb, b_bs, p.len_b)
                && batched_ok(mc, nc, rsc, csc, c_bs, p.len_c)
                && mc == m
                && nc == n
                && !mirror_self_aliases(mc, nc, rsc, csc)
                && (batch <= 1
                    || (c_bs >= 0 && ec.map(|e| (c_bs as usize) >= e).unwrap_or(false)));
            // The batch LOOP count is itself unbounded work even when each element is
            // empty (m*n == 0 zeroes the product), so cap the raw batch too — else
            // gemm_batched spins over `batch` no-op elements and libFuzzer times out.
            if would_pass
                && (batch > WORK_CAP || batch.saturating_mul(sat3(m, n, k)) > WORK_CAP)
            {
                return;
            }
            let a = vec![0.0f32; p.len_a];
            let b = vec![0.0f32; p.len_b];
            let mut c = vec![0.0f32; p.len_c];
            gemm_batched(
                batch,
                alpha,
                MatRef::new(&a, m, k, rsa, csa),
                a_bs,
                MatRef::new(&b, k, n, rsb, csb),
                b_bs,
                beta,
                MatMut::new(&mut c, mc, nc, rsc, csc),
                c_bs,
                p.par,
            );
        }
        EntryKind::PrepackB => {
            // Skip only the "representable but huge" middle band: a would-pass pack
            // whose ~n*k element count fits usize yet exceeds the work cap would OOM
            // on correct behavior. Everything else is fast to run: empty operands
            // short-circuit in prepack, and a pack size that overflows usize is a
            // documented "too large" reject — both stay fuzzed.
            let would_pass = in_bounds(k, n, rsb, csb, p.len_b);
            let expensive = n != 0
                && k != 0
                && n.checked_mul(k).is_some()
                && (n > WORK_CAP || k > WORK_CAP || n * k > WORK_CAP);
            if would_pass && expensive {
                return;
            }
            let b = vec![0.0f32; p.len_b];
            let _ = prepack_rhs(MatRef::new(&b, k, n, rsb, csb));
        }
        EntryKind::PrepackA => {
            let would_pass = in_bounds(m, k, rsa, csa, p.len_a);
            let expensive = m != 0
                && k != 0
                && m.checked_mul(k).is_some()
                && (m > WORK_CAP || k > WORK_CAP || m * k > WORK_CAP);
            if would_pass && expensive {
                return;
            }
            let a = vec![0.0f32; p.len_a];
            let _ = prepack_lhs(MatRef::new(&a, m, k, rsa, csa));
        }
    }
}
