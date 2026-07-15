//! Shared harness for the fused-epilogue suite: the deterministic RNG, the real-float element
//! trait [`Flt`], the exact reference epilogue map [`ref_apply`] (a byte-for-byte mirror of
//! `FusedEpi::apply`), C-layout helpers, and the core [`check_fused`] oracle: plain `gemm`
//! followed by that scalar map, compared **bitwise** against the fused result

use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm};

/// Deterministic xorshift* RNG (no external dep, reproducible across runs)
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F491_4F6CDD1D)
    }
    /// A value in roughly `[-1, 1)`
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The real-float element under test (`f32`/`f64`): construction, bit compare, and the exact
/// reference epilogue map (a byte-for-byte mirror of `FusedEpi::apply`). The `Float + PartialOrd`
/// bounds (which `FusedScalar` no longer implies, now that it also covers the narrow floats) give
/// the reference map its `+`/`*` arithmetic and `ReLU` comparisons
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

/// The reference scalar map, an exact mirror of `FusedEpi::apply`: `act(v + bias)`. Same ops,
/// same order, so it agrees bitwise with the fused vector *and* scratch paths
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
    Col,
    Row,
    /// Column-major with a padded row stride (strided C, forces the scratch path at edges)
    ColPadded,
}

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

/// Build an `m x n` matrix (col-major storage) of RNG values
pub(crate) fn make<T: Flt>(rng: &mut Rng, m: usize, n: usize) -> Vec<T> {
    (0..m * n).map(|_| T::of(rng.unit() * 2.0)).collect()
}

/// Run one fused case and its `gemm`+map oracle; assert bitwise-equal C. `bias`/`act` are
/// applied in the user frame; the reference reads back `gemm`'s output and maps it
pub(crate) fn check_fused<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    layout: Layout,
    bias_kind: u8, // 0 none, 1 per-row, 2 per-col
    act: Option<Activation<T>>,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k); // col-major mxk
    let b = make::<T>(rng, k, n); // col-major kxn
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen, 1);

    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias = match bias_kind {
        1 => Some(Bias::PerRow(&bias_row)),
        2 => Some(Bias::PerCol(&bias_col)),
        _ => None,
    };

    // fused
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

    // oracle: plain gemm then the scalar map (user frame)
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

/// Small shim so `check_fused` can borrow the bias slices with the right lifetime and drive
/// `gemm_fused_with` over raw col-major LHS/RHS storage
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

/// `Option<Activation<T>>` is not `Clone` (T need not be), so clone it explicitly
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
