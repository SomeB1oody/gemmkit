//! `gemm_fused` tests for the narrow floats (`f16`/`bf16`, feature `half`)
//!
//! A narrow fused call is not bitwise-equal to `gemm()` followed by a separate narrow map: the
//! bias vector and `LeakyRelu` slope are the narrow type, widened **exactly** to `f32`, and the
//! whole bias+activation transform runs in `f32` against the `f32` accumulator, narrowed to the
//! output **once**. `gemm`-then-map would instead round to `N` for the plain store and round
//! again after the map, 2 roundings where the fused call has 1, so the fused result is *more*
//! precise, not identical. What each test locks down:
//!
//! * (a) the vector fast path and the scalar/scratch path agree bit-for-bit on the same input;
//! * (b) at `k = 1` (an exact `f32` product), the fused output matches a single-rounding
//!   reference bitwise, and that reference is shown to differ from the 2-rounding alternative;
//! * (c) the general-driver shape matches an f64 oracle within a per-element tolerance, across
//!   the full bias x activation x beta sweep;
//! * (d) the `small_mn` / small-`k` special routes match the oracle and reproduce serial ==
//!   Rayon(4) bit-for-bit;
//! * (e) `bias = None, act = None` delegates to plain `gemm` bit-for-bit (the zero-cost path);
//! * (f) `ReLU` maps a NaN accumulator to `+0.0`, every ISA and store path;
//! * (g) at deep `k`, the fused route stays single-panel while a plain narrow `gemm` re-blocks
//!   through its f32-output twin, and both stay accurate and serial == parallel
//!
//! All shapes are platform-independent: deterministic LCG fills, self-computed references

use crate::common::Rng;
use gemmkit::{
    Activation, Bias, MatMut, MatRef, NarrowFloat, Parallelism, Workspace, gemm, gemm_fused_with,
    tuning,
};

/// Guards the process-global `deep_kc_bytes` knob against [`fused_mixed_deep_k`] and
/// [`fused_mixed_identity_delegates`] racing each other under the harness's concurrent test
/// threads. `fused_mixed_deep_k` forces the knob to `1` so its plain-`gemm` half engages the
/// f32-output twin; `fused_mixed_identity_delegates` runs an unrelated plain narrow `gemm` (via
/// the fused entry's zero-cost delegation) and asserts it bit-for-bit against a fused result,
/// which only holds while that call stays on the single-panel route. Without the lock, a forced
/// knob value could leak into the identity test mid-run and flip its plain `gemm` onto the twin,
/// which is only tolerance-equal (not bitwise) for a general `beta`. Poison-tolerant so one
/// panicked test does not wedge every later lock acquisition
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// per-type test harness

/// One narrow float type under test: exact widen to `f32`, bit-pattern compare, a NaN value, and
/// the type's own rounding error, so every test function is written once and instantiated for
/// both `f16` and `bf16`
trait Narrow: NarrowFloat + gemmkit::FusedScalar {
    fn of(x: f64) -> Self;
    /// Widen to `f32`: exact, since every narrow value fits `f32`'s range and mantissa
    fn f32(self) -> f32;
    fn bits(self) -> u16;
    fn nan() -> Self;
    /// This type's machine epsilon: the tolerance gates scale by it since the dominant error
    /// source is the single narrowing round, not the `f32` accumulation
    const EPS: f64;
    fn name() -> &'static str;
}
impl Narrow for gemmkit::f16 {
    fn of(x: f64) -> Self {
        gemmkit::f16::from_f64(x)
    }
    fn f32(self) -> f32 {
        self.widen()
    }
    fn bits(self) -> u16 {
        self.to_bits()
    }
    fn nan() -> Self {
        gemmkit::f16::NAN
    }
    const EPS: f64 = 9.765625e-4; // 2^-10: f16 has a 10-bit mantissa
    fn name() -> &'static str {
        "f16"
    }
}
impl Narrow for gemmkit::bf16 {
    fn of(x: f64) -> Self {
        gemmkit::bf16::from_f64(x)
    }
    fn f32(self) -> f32 {
        self.widen()
    }
    fn bits(self) -> u16 {
        self.to_bits()
    }
    fn nan() -> Self {
        gemmkit::bf16::NAN
    }
    const EPS: f64 = 7.8125e-3; // 2^-7: bf16 has a 7-bit mantissa
    fn name() -> &'static str {
        "bf16"
    }
}

/// A `Copy` descriptor for the activation a reference computation should apply. `gemmkit::Activation`
/// is not `Clone`, so this stands in for it wherever a value needs to be reused across several calls
#[derive(Copy, Clone)]
enum ActK {
    None,
    Relu,
    Leaky(f64),
}
impl ActK {
    fn make<N: Narrow>(self) -> Option<Activation<N>> {
        match self {
            ActK::None => None,
            ActK::Relu => Some(Activation::Relu),
            ActK::Leaky(s) => Some(Activation::LeakyRelu(N::of(s))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            ActK::None => "none",
            ActK::Relu => "relu",
            ActK::Leaky(_) => "leaky",
        }
    }
}

/// Which bias axis a reference computation should apply, or none
#[derive(Copy, Clone, PartialEq)]
enum BiasK {
    None,
    Row,
    Col,
}

/// Mirrors `FusedEpi::<N>::apply` op-for-op: bias add, then activation, then `N::narrow`, all in
/// `f32` with a single narrowing at the end. `bias` is already widened to `f32`
fn ref_apply_narrow<N: Narrow>(v: f32, bias: f32, act: ActK) -> N {
    let v = v + bias;
    let v = match act {
        ActK::None => v,
        ActK::Relu => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        ActK::Leaky(s) => {
            let hi = if v > 0.0 { v } else { 0.0 };
            let lo = if v < 0.0 { v } else { 0.0 };
            hi + (s as f32) * lo
        }
    };
    N::narrow(v)
}

/// Build an `m x n` col-major narrow matrix of LCG values scaled by `scale`
fn make<N: Narrow>(rng: &mut Rng, m: usize, n: usize, scale: f64) -> Vec<N> {
    (0..m * n).map(|_| N::of(rng.unit() * scale)).collect()
}

/// f64 reference for `C <- act(alpha*A*B + beta*C0 + bias)`, un-narrowed, `out[i + j*m]` in
/// logical (row, col) order. Every input reaches `f64` through the exact `f32` widen, and the
/// whole reduction runs in `f64`, so this has strictly more precision than the fused call it
/// gates: [`assert_close`] absorbs that gap with a tolerance rather than a bitwise compare
fn reference_f64<N: Narrow>(
    m: usize,
    k: usize,
    n: usize,
    alpha: N,
    a: &[N],
    rsa: isize,
    csa: isize,
    b: &[N],
    rsb: isize,
    csb: isize,
    beta: N,
    c0: &[N],
    rsc: isize,
    csc: isize,
    bias_kind: BiasK,
    bias_row: &[N],
    bias_col: &[N],
    act: ActK,
) -> Vec<f64> {
    let alpha = alpha.f32() as f64;
    let beta = beta.f32() as f64;
    let mut out = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0f64;
            for p in 0..k {
                let av = a[(i as isize * rsa + p as isize * csa) as usize].f32() as f64;
                let bv = b[(p as isize * rsb + j as isize * csb) as usize].f32() as f64;
                acc += av * bv;
            }
            let base = if beta == 0.0 {
                0.0
            } else {
                beta * c0[(i as isize * rsc + j as isize * csc) as usize].f32() as f64
            };
            let mut v = alpha * acc + base;
            v += match bias_kind {
                BiasK::None => 0.0,
                BiasK::Row => bias_row[i].f32() as f64,
                BiasK::Col => bias_col[j].f32() as f64,
            };
            v = match act {
                ActK::None => v,
                ActK::Relu => {
                    if v > 0.0 {
                        v
                    } else {
                        0.0
                    }
                }
                ActK::Leaky(s) => {
                    let hi = if v > 0.0 { v } else { 0.0 };
                    let lo = if v < 0.0 { v } else { 0.0 };
                    hi + s * lo
                }
            };
            out[i + j * m] = v;
        }
    }
    out
}

/// Per-element accuracy gate against the un-narrowed f64 reference `cref`: for each `(i, j)`,
/// with `r = cref[i + j*m]` and `g = got[..].to_f64()` read back through `got`'s strides, asserts
/// `|g - r| <= (2*eps_N + 8*k*f32_eps)*(1 + |r|)`, where `eps_N = N::EPS` and `f32_eps = 2^-23`
///
/// The bound has 2 terms: the fused path's `f32` accumulation carries a relative error of order
/// `k*f32_eps`, and its single narrowing to `N` adds up to about 1 `N`-ulp on top. Scaling that
/// sum by `(1 + |r|)` turns it into an absolute bound that stays meaningful near `r = 0`. A real
/// bug (a dropped product term, a bias on the wrong axis, the wrong activation slope) produces an
/// `O(1)`-relative error on the affected elements, orders of magnitude above this gate, so the
/// assert catches it instead of averaging it away
fn assert_close<N: Narrow>(
    got: &[N],
    rsc: isize,
    csc: isize,
    m: usize,
    n: usize,
    cref: &[f64],
    k: usize,
    ctx: &str,
) {
    let f32_eps = f32::EPSILON as f64; // 2^-23
    for j in 0..n {
        for i in 0..m {
            let g = got[(i as isize * rsc + j as isize * csc) as usize].f32() as f64;
            let r = cref[i + j * m];
            assert!(g.is_finite(), "{ctx}: non-finite output at ({i},{j})");
            let tol_e = (2.0 * N::EPS + 8.0 * (k as f64) * f32_eps) * (1.0 + r.abs());
            assert!(
                (g - r).abs() <= tol_e,
                "{}: {ctx} abs error {:.3e} > tol {tol_e:.3e} at ({i},{j}) \
                 (got {g:.6e}, ref {r:.6e}, m={m},k={k},n={n})",
                N::name(),
                (g - r).abs(),
            );
        }
    }
}

// a. vector == scalar

/// The fused vector fast path (unit-stride col-major C, `apply_reg` + the family's own
/// `store_out`) and the scratch/scalar path (a strided C, `rsc != 1`, so every tile fails the
/// vector gate and drains through `apply` instead) must land on the identical bits
///
/// `beta` is restricted to `0` and `1` deliberately: those are the 2 states where the mixed
/// kernel's `beta*C + A*B` combine is bit-identical between the 2 paths (`beta == 0` never reads
/// `C`; `beta == 1` is a plain add on both), so `apply`/`apply_reg` receive the same `f32` value
/// and the test isolates the epilogue transform itself. For a general `beta` the fast path fuses
/// the combine as a single `mul_add` while the scratch path (byte-for-byte the pre-epilogue mixed
/// kernel) does not, so the 2 accumulators would only converge after the narrowing absorbs a
/// sub-narrow difference: an ISA-dependent coincidence, not a guarantee. The pre-narrow precision
/// contract for a general `beta` is covered separately by tests (c)/(d)
fn vector_scalar_bitwise<N: Narrow>() {
    let mut rng = Rng::new(0x11CE_A501);
    let (m, k, n) = (64usize, 96usize, 48usize);
    let alpha = N::of(1.3);
    let a = make::<N>(&mut rng, m, k, 1.0); // col-major m x k
    let b = make::<N>(&mut rng, k, n, 1.0); // col-major k x n
    let c0 = make::<N>(&mut rng, m, n, 1.0); // logical col-major m x n
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();

    for beta in [N::of(0.0), N::of(1.0)] {
        // rsc = 1: takes the vector fast path
        let mut c_vec = c0.clone();
        // rsc = 2, csc = 2m over a 2x-larger buffer: |csc| >= |rsc| keeps the no-swap
        // orientation, while rsc != 1 forces every tile onto the scratch/scalar path
        let mut c_str = vec![N::of(0.0); 2 * m * n];
        for j in 0..n {
            for i in 0..m {
                c_str[i * 2 + j * 2 * m] = c0[i + j * m];
            }
        }

        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_vec, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            Some(Activation::LeakyRelu(N::of(0.25))),
            Parallelism::Serial,
        );
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_str, m, n, 2, 2 * m as isize),
            Some(Bias::PerRow(&bias_row)),
            Some(Activation::LeakyRelu(N::of(0.25))),
            Parallelism::Serial,
        );

        for j in 0..n {
            for i in 0..m {
                assert_eq!(
                    c_vec[i + j * m].bits(),
                    c_str[i * 2 + j * 2 * m].bits(),
                    "{}: vector != scalar at ({i},{j}) [beta={:#06x}]",
                    N::name(),
                    beta.bits(),
                );
            }
        }
    }
}

#[test]
fn fused_mixed_vector_scalar_bitwise() {
    vector_scalar_bitwise::<gemmkit::f16>();
    vector_scalar_bitwise::<gemmkit::bf16>();
}

// b. pre-narrow lock

/// At `k = 1, alpha = 1, beta = 0` the accumulator is a single `f32` product `a*b`: exact in
/// `f32` since a narrow x narrow product fits the 23-bit mantissa, and there is no summation
/// order to worry about with only 1 term. For each of the 3 activations (`None`, `ReLU`,
/// `LeakyReLU(0.25)`) the fused output must equal the **single-rounding** reference
/// `narrow(act(a*b + bias.widen()))` bitwise, and that reference must differ from the 2-rounding
/// alternative `narrow(act(narrow(a*b).widen() + bias.widen()))` on at least one element: this
/// locks the pre-narrow semantic through the activation, and rules out the test being vacuous
/// (both references trivially agreeing because the case never arises). `a, b` are drawn from
/// `[1, 2)` so `a*b` lands in `[1, 4)` and `bias` from about `[-2, 2]`; `a*b + bias` then stays
/// positive on most elements, where `ReLU`/`LeakyReLU` are the identity, so the sub-narrow bits
/// that make the `None` case diverge survive the activation there too
fn pre_narrow_semantics<N: Narrow>() {
    let mut rng = Rng::new(0x9E27_B1A5);
    let (m, k, n) = (32usize, 1usize, 24usize);
    // [1, 2): a*b then lands in [1, 4), needing up to 2x N's mantissa bits, so narrowing the
    // product before adding the bias would already have thrown away the bits this test targets
    let a: Vec<N> = (0..m * k)
        .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
        .collect();
    let b: Vec<N> = (0..k * n)
        .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
        .collect();
    // Comparable magnitude to a*b, so adding it does not wash out those sub-narrow bits
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();

    for act in [ActK::None, ActK::Relu, ActK::Leaky(0.25)] {
        let mut c = vec![N::of(0.0); m * n]; // col-major; beta = 0, so this is never read
        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            N::of(1.0),
            MatRef::new(&a, m, k, 1, m as isize), // k = 1, below small_k_threshold: small_k route
            MatRef::new(&b, k, n, 1, k as isize),
            N::of(0.0),
            MatMut::new(&mut c, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            act.make::<N>(),
            Parallelism::Serial,
        );

        let mut differs = 0usize;
        for j in 0..n {
            for i in 0..m {
                // k == 1, col-major: the single f32 product A[i,0]*B[0,j], exact
                let ab = a[i].f32() * b[j].f32();
                let biasw = bias_row[i].f32();
                let one_round: N = ref_apply_narrow::<N>(ab, biasw, act);
                // Narrow the product first, then add bias and narrow again
                let two_round: N = ref_apply_narrow::<N>(N::narrow(ab).f32(), biasw, act);
                assert_eq!(
                    c[i + j * m].bits(),
                    one_round.bits(),
                    "{}: fused != single-rounding reference at ({i},{j}) [act={}]",
                    N::name(),
                    act.name(),
                );
                if one_round.bits() != two_round.bits() {
                    differs += 1;
                }
            }
        }
        assert!(
            differs > 0,
            "{}: pre-narrow test vacuous — single- and two-rounding never diverged [act={}]",
            N::name(),
            act.name(),
        );
    }
}

#[test]
fn fused_mixed_pre_narrow_semantics() {
    pre_narrow_semantics::<gemmkit::f16>();
    pre_narrow_semantics::<gemmkit::bf16>();
}

// c. f64 oracle sweep

/// A general-driver shape swept over every `bias x act x beta` combination, each checked against
/// the f64 oracle within [`assert_close`]'s tolerance
fn matches_reference<N: Narrow>() {
    let mut rng = Rng::new(0xC0FF_EE12);
    let (m, k, n) = (96usize, 128usize, 64usize);
    let a = make::<N>(&mut rng, m, k, 1.0);
    let b = make::<N>(&mut rng, k, n, 1.0);
    let c0 = make::<N>(&mut rng, m, n, 1.0);
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
    let bias_col: Vec<N> = (0..n).map(|_| N::of(rng.unit() * 2.0)).collect();
    let alpha = N::of(1.0);

    for beta in [N::of(0.0), N::of(0.7)] {
        for bias_kind in [BiasK::None, BiasK::Row, BiasK::Col] {
            for act in [ActK::None, ActK::Relu, ActK::Leaky(0.1)] {
                let bias = match bias_kind {
                    BiasK::None => None,
                    BiasK::Row => Some(Bias::PerRow(&bias_row)),
                    BiasK::Col => Some(Bias::PerCol(&bias_col)),
                };
                let mut c = c0.clone();
                let mut ws = Workspace::new();
                gemm_fused_with(
                    &mut ws,
                    alpha,
                    MatRef::new(&a, m, k, 1, m as isize),
                    MatRef::new(&b, k, n, 1, k as isize),
                    beta,
                    MatMut::new(&mut c, m, n, 1, m as isize),
                    bias,
                    act.make::<N>(),
                    Parallelism::Serial,
                );
                let cref = reference_f64::<N>(
                    m, k, n, alpha, &a, 1, m as isize, &b, 1, k as isize, beta, &c0, 1, m as isize,
                    bias_kind, &bias_row, &bias_col, act,
                );
                assert_close::<N>(&c, 1, m as isize, m, n, &cref, k, "matrix");
            }
        }
    }
}

#[test]
fn fused_mixed_matches_reference() {
    matches_reference::<gemmkit::f16>();
    matches_reference::<gemmkit::bf16>();
}

// d. special routes

/// One fused config, run over the small-`m,n` or small-`k` route: accuracy vs the f64 oracle,
/// plus serial == Rayon(4) bitwise. The bitwise part holds regardless of thread count because
/// both routes partition the output by tile with no cross-thread reduction (each worker computes
/// a disjoint range of complete elements), and the fused epilogue only changes that tile's final
/// store, not how the range is split
fn special_route<N: Narrow>(
    m: usize,
    k: usize,
    n: usize,
    a: &[N],
    rsa: isize,
    csa: isize,
    b: &[N],
    rsb: isize,
    csb: isize,
    tag: &str,
) {
    let mut rng = Rng::new(0x5EC1_A177 ^ (m as u64) << 8 ^ n as u64);
    let alpha = N::of(1.0);
    let beta = N::of(0.7);
    let c0 = make::<N>(&mut rng, m, n, 1.0);
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
    let act = ActK::Leaky(0.25);

    let run = |par: Parallelism| -> Vec<N> {
        let mut c = c0.clone();
        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(a, m, k, rsa, csa),
            MatRef::new(b, k, n, rsb, csb),
            beta,
            MatMut::new(&mut c, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            act.make::<N>(),
            par,
        );
        c
    };

    let c_ser = run(Parallelism::Serial);
    let c_par = run(Parallelism::Rayon(4));
    for idx in 0..m * n {
        assert_eq!(
            c_ser[idx].bits(),
            c_par[idx].bits(),
            "{}: {tag} serial != parallel at {idx}",
            N::name(),
        );
    }

    let cref = reference_f64::<N>(
        m,
        k,
        n,
        alpha,
        a,
        rsa,
        csa,
        b,
        rsb,
        csb,
        beta,
        &c0,
        1,
        m as isize,
        BiasK::Row,
        &bias_row,
        &[],
        act,
    );
    assert_close::<N>(&c_ser, 1, m as isize, m, n, &cref, k, tag);
}

fn special_routes<N: Narrow>() {
    // small_mn route: m, n <= small_mn_dim and k > small_k_threshold, with A row-major
    // (csa == 1) and B col-major (rsb == 1) so both stream contiguously along k
    {
        let (m, k, n) = (8usize, 2048usize, 8usize);
        let mut rng = Rng::new(0x5A11_3E00);
        let a = make::<N>(&mut rng, m, k, 1.0); // built col-major...
        let b = make::<N>(&mut rng, k, n, 1.0);
        // ...then transposed into a row-major buffer so csa == 1
        let mut a_row = vec![N::of(0.0); m * k];
        for i in 0..m {
            for p in 0..k {
                a_row[i * k + p] = a[i + p * m];
            }
        }
        special_route::<N>(
            m, k, n, &a_row, k as isize, 1, &b, 1, k as isize, "small_mn",
        );
    }
    // small_k route: k <= small_k_threshold, m past small_mn_dim so small_mn does not also claim
    // it; A col-major (rsa == 1)
    {
        let (m, k, n) = (100usize, 4usize, 80usize);
        let mut rng = Rng::new(0x5A11_C400);
        let a = make::<N>(&mut rng, m, k, 1.0);
        let b = make::<N>(&mut rng, k, n, 1.0);
        special_route::<N>(m, k, n, &a, 1, m as isize, &b, 1, k as isize, "small_k");
    }
}

#[test]
fn fused_mixed_special_routes() {
    special_routes::<gemmkit::f16>();
    special_routes::<gemmkit::bf16>();
}

// e. identity delegates

/// `bias = None, act = None` makes `gemm_fused_with` delegate straight to plain `gemm`, so the
/// 2 calls must agree bit-for-bit at every thread count
fn identity_delegates<N: Narrow>() {
    let mut rng = Rng::new(0x1DE7_17FF);
    let (m, k, n) = (48usize, 40usize, 33usize);
    let a = make::<N>(&mut rng, m, k, 1.0);
    let b = make::<N>(&mut rng, k, n, 1.0);
    let c0 = make::<N>(&mut rng, m, n, 1.0);

    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        let mut c_fused = c0.clone();
        let mut c_ref = c0.clone();
        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            N::of(0.9),
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            N::of(0.5),
            MatMut::new(&mut c_fused, m, n, 1, m as isize),
            None,
            None,
            par,
        );
        gemm(
            N::of(0.9),
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            N::of(0.5),
            MatMut::new(&mut c_ref, m, n, 1, m as isize),
            par,
        );
        for idx in 0..m * n {
            assert_eq!(
                c_fused[idx].bits(),
                c_ref[idx].bits(),
                "{}: identity-fused != gemm at {idx}",
                N::name(),
            );
        }
    }
}

#[test]
fn fused_mixed_identity_delegates() {
    // Holds KNOB_LOCK: the plain `gemm` call inside `identity_delegates` must stay on the
    // single-panel route for its bitwise assert against the fused result to hold, which a
    // concurrent `set_deep_kc_bytes(1)` from `fused_mixed_deep_k` would break (see KNOB_LOCK)
    let _g = knob_guard();
    identity_delegates::<gemmkit::f16>();
    identity_delegates::<gemmkit::bf16>();
}

// f. NaN -> ReLU -> 0

/// A `NaN` in every A row's 1st depth column poisons that row's whole `k`-reduction to `NaN`
/// (NaN propagates through every FMA it touches), so with `beta = 0` every output element is
/// `ReLU(NaN)`. `v > 0.0` is false for NaN on every comparison path, so both the vector and
/// scalar `Act::Relu` arms fall to the zero branch: the output must be exactly `N::ZERO`
/// bit-for-bit, on every ISA
fn nan_relu<N: Narrow>() {
    let mut rng = Rng::new(0x4A11_DEAD);
    let (m, k, n) = (64usize, 3usize, 48usize);
    let mut a = make::<N>(&mut rng, m, k, 1.0); // col-major m x k
    let b = make::<N>(&mut rng, k, n, 1.0);
    for i in 0..m {
        a[i] = N::nan(); // A[i, 0], the depth-0 element of row i
    }

    let mut c = vec![N::of(0.0); m * n];
    let mut ws = Workspace::new();
    gemm_fused_with(
        &mut ws,
        N::of(1.0),
        MatRef::new(&a, m, k, 1, m as isize),
        MatRef::new(&b, k, n, 1, k as isize),
        N::of(0.0),
        MatMut::new(&mut c, m, n, 1, m as isize),
        None,
        Some(Activation::Relu),
        Parallelism::Serial,
    );

    let zero = N::of(0.0).bits();
    for j in 0..n {
        for i in 0..m {
            assert_eq!(
                c[i + j * m].bits(),
                zero,
                "{}: ReLU(NaN) must be +0.0 at ({i},{j})",
                N::name(),
            );
        }
    }
}

#[test]
fn fused_mixed_nan_relu() {
    nan_relu::<gemmkit::f16>();
    nan_relu::<gemmkit::bf16>();
}

// g. fused x deep-k interaction

/// Forcing `deep_kc_bytes` down to `1` drops the engage gate to essentially nothing, so a plain
/// narrow `gemm` at this shape re-blocks through its f32-output twin (`MixedGemmF32` /
/// `Bf16DotGemmF32`). `gemm_fused` never takes that branch (its mixed dispatch has no deep-k
/// route at all), so it stays on the single-panel path regardless of the knob. This locks both
/// halves of that split at the same `k`: the fused entry stays accurate against the f64 oracle
/// and reproduces serial == parallel bit-for-bit on its single panel, and the plain path does the
/// same on its twin. Accuracy, not a bitwise compare against a plain-then-map oracle, is the
/// right gate for the fused half: its bias+activation applies pre-narrow, so it is intentionally
/// more precise than `gemm`-then-map, same as tests (c)/(d)
fn deep_k<N: Narrow>() {
    // m, n past small_mn_dim and k past small_k_threshold, so neither special route claims this
    // shape and both halves below reach the general driver
    let (m, k, n) = (48usize, 96usize, 40usize);
    let mut rng = Rng::new(0xDEE9_CA5E);
    let alpha = N::of(1.0);
    let beta = N::of(0.7);
    let a = make::<N>(&mut rng, m, k, 1.0);
    let b = make::<N>(&mut rng, k, n, 1.0);
    let c0 = make::<N>(&mut rng, m, n, 1.0);
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
    let act = ActK::Leaky(0.25);

    // Held for the whole knob-mutated window; `fused_mixed_identity_delegates` blocks on the
    // same lock until this test restores the knob
    let _g = knob_guard();
    let restore = tuning::deep_kc_bytes();
    tuning::set_deep_kc_bytes(1); // drops the engage gate low enough for any k here to cross it

    // The fused half stays single-panel: serial == parallel bitwise, accurate vs the oracle
    let run_fused = |par: Parallelism| -> Vec<N> {
        let mut c = c0.clone();
        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            act.make::<N>(),
            par,
        );
        c
    };
    let f_ser = run_fused(Parallelism::Serial);
    let f_par = run_fused(Parallelism::Rayon(4));
    for idx in 0..m * n {
        assert_eq!(
            f_ser[idx].bits(),
            f_par[idx].bits(),
            "{}: fused deep-k serial != parallel at {idx}",
            N::name(),
        );
    }
    let cref_f = reference_f64::<N>(
        m,
        k,
        n,
        alpha,
        &a,
        1,
        m as isize,
        &b,
        1,
        k as isize,
        beta,
        &c0,
        1,
        m as isize,
        BiasK::Row,
        &bias_row,
        &[],
        act,
    );
    assert_close::<N>(&f_ser, 1, m as isize, m, n, &cref_f, k, "fused-deep-k");

    // The plain half re-blocks through the f32-output twin: serial == parallel bitwise, accurate
    let run_plain = |par: Parallelism| -> Vec<N> {
        let mut c = c0.clone();
        gemm(
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            MatRef::new(&b, k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c, m, n, 1, m as isize),
            par,
        );
        c
    };
    let p_ser = run_plain(Parallelism::Serial);
    let p_par = run_plain(Parallelism::Rayon(4));
    for idx in 0..m * n {
        assert_eq!(
            p_ser[idx].bits(),
            p_par[idx].bits(),
            "{}: plain deep-k twin serial != parallel at {idx}",
            N::name(),
        );
    }
    let cref_p = reference_f64::<N>(
        m,
        k,
        n,
        alpha,
        &a,
        1,
        m as isize,
        &b,
        1,
        k as isize,
        beta,
        &c0,
        1,
        m as isize,
        BiasK::None,
        &bias_row,
        &[],
        ActK::None,
    );
    assert_close::<N>(&p_ser, 1, m as isize, m, n, &cref_p, k, "plain-twin-deep-k");

    tuning::set_deep_kc_bytes(restore);
}

#[test]
fn fused_mixed_deep_k() {
    deep_k::<gemmkit::f16>();
    deep_k::<gemmkit::bf16>();
}
