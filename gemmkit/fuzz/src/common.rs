//! RNG, per-element numeric/canary traits, and strided-layout operand construction shared by every fuzz target

use arbitrary::{Result, Unstructured};
use gemmkit::{Complex, ComplexScalar, GemmScalar, Parallelism, bf16, c32, c64, f16};

// element value tables and the per-operand fill RNG

/// alpha/beta values for the float/complex gates: the same magnitude family the
/// correctness suite sweeps (`tests/correctness/mixed.rs`), plus a couple of extra
/// corners. Index 0 is `0.0`, so `ab_index` can land on the `beta == 0` "C not read" case
pub(crate) const AB_TABLE: [f64; 6] = [0.0, 1.0, -1.0, 0.5, 0.75, 2.5];

/// Integer twin of `AB_TABLE`: `gemm_i8` takes `i32` alpha/beta, so truncating the
/// float table would fold `0.0`/`0.5`/`0.75` into the same `0` - this table keeps
/// its entries distinct
pub(crate) const I8_AB_TABLE: [i32; 6] = [0, 1, -1, 2, 3, -2];

/// Sentinel fill for the gap slots of an i8-GEMM output buffer (C is i32)
const I32_CANARY: i32 = 0x0BAD_F00Du32 as i32;

/// xorshift matching the correctness suite's `rand_vec` (`tests/oracle_common/mod.rs`);
/// seeded once per operand from a plan-supplied `u64` and iterated internally, so a large
/// operand does not have to spend libFuzzer's `-max_len` byte budget 1 byte per element
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
    /// Full-range i8 draw: used directly for i8 operands and as the source for next_quant
    #[inline]
    pub(crate) fn next_i8(&mut self) -> i8 {
        (self.step() >> 24) as u8 as i8
    }
    /// `i8 / 64.0`: magnitude <= 2 and, since the denominator is `2^6`, exactly
    /// representable in every float element type - no extra input-rounding error to
    /// fold into the tolerance gate
    #[inline]
    pub(crate) fn next_quant(&mut self) -> f64 {
        self.next_i8() as f64 / 64.0
    }
}

// per-element traits: f64 <-> element conversion, plus the gap-canary sentinel

/// Bit-pattern sentinel for the non-view "gap" slots of an output buffer: the GEMM
/// call must never touch them, so a slot that no longer reads as the sentinel is a
/// stray write
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

/// A real GEMM element (f32/f64/f16/bf16): f64 construction/view plus its gate EPS
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
// f16/bf16 accumulate in f32 and round the output to 16 bits, so EPS is the 16-bit
// machine epsilon (matches `Elem::EPS` in tests/oracle_common/mod.rs)
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

/// A complex GEMM element (c32/c64): re/im construction and extraction, plus its gate EPS
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

// strided layout plans and operand-buffer construction

/// A stride family covering `tests/correctness/common.rs`'s Row/Col/GeneralPad layouts,
/// parameterized by an interleave (`il`) and a trailing pad (`pad`). `BroadcastRow`
/// (`rs = 0`) self-aliases, so callers only allow it for the read-only operands A/B,
/// never for C
#[derive(Debug, Clone, Copy)]
pub enum LayoutPlan {
    RowIsh { il: usize, pad: usize },
    ColIsh { il: usize, pad: usize },
    BroadcastRow,
}

impl LayoutPlan {
    /// `(rs, cs)` for a `rows x cols` view in this layout; every case is non-negative,
    /// matching what the checked API (`api.rs::extent`) accepts
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

/// Highest slice offset (exclusive) of a `rows x cols` view: the same formula as
/// `api.rs::extent`, simplified for the non-negative, non-overflowing strides this
/// harness ever builds
pub(crate) fn extent_of(rows: usize, cols: usize, rs: isize, cs: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    ((rows - 1) as isize * rs + (cols - 1) as isize * cs) as usize + 1
}

/// Build a `rows x cols` operand: allocate exactly the extent this layout needs, fill
/// its view slots via `genf` through the strides, and return `(buf, rs, cs)`. The gap
/// slots outside the view keep `fill` untouched
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

/// Panics if the GEMM call wrote a slot outside the `rows x cols` view: the cheapest
/// detector for the stride/epilogue out-of-bounds-write class these layouts probe
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

// small Arbitrary-decoding helpers shared by every plan

pub(crate) fn arb_par(u: &mut Unstructured) -> Result<Parallelism> {
    // Serial weighted 2x for exec/s; explicit thread counts capped at Rayon(2)
    Ok(*u.choose(&[
        Parallelism::Serial,
        Parallelism::Serial,
        Parallelism::Rayon(1),
        Parallelism::Rayon(2),
    ])?)
}

pub(crate) fn arb_par_knobs(u: &mut Unstructured) -> Result<Parallelism> {
    // Also exercises Rayon(0) (auto-detect), the only path where thread_dim_stride
    // takes effect (parallel_threshold gates every Rayon variant); still Serial-weighted
    // for throughput
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
