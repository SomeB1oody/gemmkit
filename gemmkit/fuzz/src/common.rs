//! RNG, element/canary traits, and strided-layout operand construction shared by all targets.

use arbitrary::{Result, Unstructured};
use gemmkit::{Complex, ComplexScalar, GemmScalar, Parallelism, bf16, c32, c64, f16};

// ---------------------------------------------------------------------------
// element tables and generators
// ---------------------------------------------------------------------------

/// alpha/beta values for the float/complex gates, mirroring the sets exercised by
/// `tests/correctness/mixed.rs` (e.g. `correctness_f16_layouts`). `0.0` first so the
/// `beta == 0` "C not read" contract is well-represented.
pub(crate) const AB_TABLE: [f64; 6] = [0.0, 1.0, -1.0, 0.5, 0.75, 2.5];

/// Integer alpha/beta: `gemm_i8` takes `i32`, so the float table's `0.5`/`0.75`
/// would truncate to `0` and collapse half of it — use a dedicated integer table.
pub(crate) const I8_AB_TABLE: [i32; 6] = [0, 1, -1, 2, 3, -2];

/// Distinctive `i32` fill for the gap slots of an `i8`-GEMM output buffer.
const I32_CANARY: i32 = 0x0BAD_F00Du32 as i32;

/// xorshift matching `rand_vec` (`tests/correctness/common.rs`), used to fill each operand
/// from a single 8-byte plan seed so `-max_len` never starves per-element entropy.
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
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
    pub(crate) fn next_i8(&mut self) -> i8 {
        (self.step() >> 24) as u8 as i8
    }
    /// `i8 / 64.0` — magnitude ≤ 2 and exactly representable in every float type
    /// (denominator `2^6`), so the tolerance gate stays meaningful.
    #[inline]
    pub(crate) fn next_quant(&mut self) -> f64 {
        self.next_i8() as f64 / 64.0
    }
}

// ---------------------------------------------------------------------------
// element traits: numeric conversion + gap canary
// ---------------------------------------------------------------------------

/// Bit-pattern sentinel written into the non-view "gap" slots of an output buffer;
/// the driver must never touch those, so a changed sentinel is a stray write.
pub(crate) trait Canary: Copy {
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
pub(crate) trait RealElem: GemmScalar + Canary {
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
// machine epsilon (`Elem::EPS` in tests/correctness/common.rs).
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
pub(crate) trait CplxElem: ComplexScalar + Canary {
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

/// Generalizes the Row/Col/GeneralPad layouts of `tests/correctness/common.rs` to an
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
    pub(crate) fn arbitrary_general(u: &mut Unstructured, allow_broadcast: bool) -> Result<Self> {
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
pub(crate) fn extent_of(rows: usize, cols: usize, rs: isize, cs: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    ((rows - 1) as isize * rs + (cols - 1) as isize * cs) as usize + 1
}

/// Allocate exactly the extent a `rows × cols` view needs, fill its view slots
/// through the strides, and return `(buf, rs, cs)`. Gap slots keep `fill`.
pub(crate) fn build_operand<T: Copy>(
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
pub(crate) fn assert_no_gap_writes<T: Canary>(
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
// small Arbitrary helpers
// ---------------------------------------------------------------------------

pub(crate) fn arb_par(u: &mut Unstructured) -> Result<Parallelism> {
    // Serial weighted 2x for exec/s; explicit threads capped at 2 (the 32-thread
    // Zen5 auto pool would tank throughput per exec).
    Ok(*u.choose(&[
        Parallelism::Serial,
        Parallelism::Serial,
        Parallelism::Rayon(1),
        Parallelism::Rayon(2),
    ])?)
}

pub(crate) fn arb_par_knobs(u: &mut Unstructured) -> Result<Parallelism> {
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

pub(crate) fn ab_index(u: &mut Unstructured) -> Result<usize> {
    Ok(u.int_in_range(0usize..=5)?)
}
