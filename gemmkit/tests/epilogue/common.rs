//! Shared harness for the fused-epilogue suite: the deterministic RNG ([`Rng`]), the
//! real-float element trait [`Flt`] (`f32`/`f64`), the reference epilogue map
//! [`ref_apply`] (a scalar mirror of `FusedEpi::apply`), C-layout helpers, and the
//! [`check_fused`] oracle, which runs one fused case against plain `gemm` plus that
//! scalar map and asserts the 2 agree bit-for-bit

use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm};

// GEMMKIT_FAST_TEST is single-sourced via the same #[path] include the oracle suite
// uses, so both binaries read the identical knob; re-exported for crate::common::* callers
#[path = "../fast_test_common/mod.rs"]
mod fast_test_common;
pub(crate) use fast_test_common::fast_test;

/// Deterministic xorshift* RNG (no external dependency, reproducible run to run)
pub(crate) struct Rng(u64);
impl Rng {
    /// Seeds the generator; the low bit is forced to 1 so the state is never all-zero
    /// (xorshift's fixed point)
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    /// One xorshift* step: xorshift the state, then scramble it with an odd multiplier
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F491_4F6CDD1D)
    }
    /// A value in `[-1, 1)`
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The real-float element under test (`f32`/`f64`): deterministic construction, a bitwise
/// compare, and a name for assertion messages. `Float<Acc = Self>` picks out exactly the real
/// floats (the narrow types accumulate in `f32`, so `Acc != Self` there), giving [`ref_apply`]
/// the `+`/`*`/`>`/`<` it needs
pub(crate) trait Flt:
    gemmkit::FusedScalar + gemmkit::Float<Acc = Self> + PartialOrd
{
    fn of(x: f64) -> Self;
    fn bits(self) -> u64;
    fn name() -> &'static str;
}
impl Flt for f32 {
    fn of(x: f64) -> Self {
        x as f32
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn name() -> &'static str {
        "f32"
    }
}
impl Flt for f64 {
    fn of(x: f64) -> Self {
        x
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn name() -> &'static str {
        "f64"
    }
}

/// Reference scalar epilogue: `act(v + bias)`, in the same order `FusedEpi::apply` uses, so it
/// agrees bitwise with both the fused kernel's vector and scratch application paths
pub(crate) fn ref_apply<T: Flt>(v: T, bias: Option<T>, act: &Option<Activation<T>>) -> T {
    let v = match bias {
        Some(b) => v + b,
        None => v,
    };
    match act {
        None => v,
        Some(Activation::Relu) => {
            if v > T::ZERO {
                v
            } else {
                T::ZERO
            }
        }
        Some(Activation::LeakyRelu(s)) => {
            let hi = if v > T::ZERO { v } else { T::ZERO };
            let lo = if v < T::ZERO { v } else { T::ZERO };
            hi + *s * lo
        }
    }
}

/// A strided C layout to test
#[derive(Copy, Clone)]
pub(crate) enum Layout {
    /// Column-major, unit row stride
    Col,
    /// Row-major (unit column stride): forces the driver's orientation swap
    Row,
    /// Column-major with a padded column stride: strided C, forces the scratch store path at
    /// tile edges
    ColPadded,
}

/// `(rsc, csc, buffer length)` for one `m x n` C under `layout`
pub(crate) fn c_strides(layout: Layout, m: usize, n: usize) -> (isize, isize, usize) {
    match layout {
        Layout::Col => (1, m as isize, m * n),
        Layout::Row => (n as isize, 1, m * n),
        Layout::ColPadded => {
            let rs = 1;
            let cs = (m + 3) as isize;
            (rs, cs, (m + 3) * n)
        }
    }
}

/// A length-`m * n` column-major buffer of RNG values
pub(crate) fn make<T: Flt>(rng: &mut Rng, m: usize, n: usize) -> Vec<T> {
    (0..m * n).map(|_| T::of(rng.unit() * 2.0)).collect()
}

/// Runs 1 fused case through [`gemm_fused_with_layout`] and its plain-`gemm`-then-[`ref_apply`]
/// oracle over identical inputs, then asserts every C element's bits match. `bias_kind` selects
/// `0` = none, `1` = `Bias::PerRow`, `2` = `Bias::PerCol`; both the fused call and the oracle
/// read the bias back in the user `(row, col)` frame, so `layout` can force the driver's
/// orientation swap or scratch store path without moving which cell a bias term lands on
pub(crate) fn check_fused<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    layout: Layout,
    bias_kind: u8, // 0 = none, 1 = per-row, 2 = per-col
    act: Option<Activation<T>>,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k); // col-major m x k
    let b = make::<T>(rng, k, n); // col-major k x n
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen, 1);

    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias = match bias_kind {
        1 => Some(Bias::PerRow(&bias_row)),
        2 => Some(Bias::PerCol(&bias_col)),
        _ => None,
    };

    // the call under test
    let mut c_fused = c0.clone();
    let mut ws = Workspace::new();
    gemm_fused_with_layout::<T>(
        &mut ws,
        alpha,
        &a,
        m,
        k,
        &b,
        k,
        n,
        beta,
        &mut c_fused,
        rsc,
        csc,
        bias,
        act.clone_like(),
        par,
    );

    // oracle: plain gemm, then ref_apply in the user frame
    let mut c_ref = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_ref, m, n, rsc, csc);
        gemm(alpha, ar, br, beta, cm, par);
    }
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let bterm = match bias_kind {
                1 => Some(bias_row[i]),
                2 => Some(bias_col[j]),
                _ => None,
            };
            c_ref[idx] = ref_apply(c_ref[idx], bterm, &act);
        }
    }

    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            assert_eq!(
                c_fused[idx].bits(),
                c_ref[idx].bits(),
                "{} {tag}: fused != gemm-then-map at ({i},{j}) [m={m} k={k} n={n}]",
                T::name(),
            );
        }
    }
}

/// Wraps raw column-major LHS/RHS/C storage in `MatRef`/`MatMut` views and forwards to
/// `gemm_fused_with`, so `check_fused` can drive it with borrowed bias slices. `_k2` is unused;
/// `k` alone sizes both operands' contraction dimension
pub(crate) fn gemm_fused_with_layout<T: Flt>(
    ws: &mut Workspace,
    alpha: T,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    _k2: usize,
    n: usize,
    beta: T,
    c: &mut [T],
    rsc: isize,
    csc: isize,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let ar = MatRef::new(a, m, k, 1, m as isize);
    let br = MatRef::new(b, k, n, 1, k as isize);
    let cm = MatMut::new(c, m, n, rsc, csc);
    gemmkit::gemm_fused_with(ws, alpha, ar, br, beta, cm, bias, act, par);
}

/// `Activation<T>` derives neither `Copy` nor `Clone` (unlike `Bias`), so `Option<Activation<T>>`
/// needs this explicit per-variant copy instead
pub(crate) trait CloneLike<T> {
    fn clone_like(&self) -> Option<Activation<T>>;
}
impl<T: Copy> CloneLike<T> for Option<Activation<T>> {
    fn clone_like(&self) -> Option<Activation<T>> {
        match self {
            None => None,
            Some(Activation::Relu) => Some(Activation::Relu),
            Some(Activation::LeakyRelu(s)) => Some(Activation::LeakyRelu(*s)),
        }
    }
}
