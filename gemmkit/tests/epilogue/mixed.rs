//! Fused-epilogue mixed-precision (`f16`/`bf16`) suite (spec section 10, Phase 2)
//!
//! `gemm_fused` accepts the narrow floats under the `half` feature. Their contract differs from
//! `f32`/`f64`: the bias vector and `LeakyRelu` slope are the narrow type, widened **exactly** to
//! `f32`, and the epilogue applies in `f32` to the accumulator **before** the single
//! round-to-nearest-even narrowing to the output. That is *more* precise than `gemm()` then a
//! separate narrow map (which rounds to `N`, widens, then rounds again), so it is **not**
//! bitwise-equal to `gemm`-then-map. What *is* bitwise here:
//!
//! * within one fused run, the vector fast path and the scalar/scratch path agree bit-for-bit
//!   (both compute `act(bias(v))` in `f32` and round once): test (a);
//! * the pre-narrow semantic is locked exactly against a single-rounding reference, crafted so the
//!   2-rounding alternative differs: test (b);
//! * serial == parallel through the special routes: test (d);
//! * the zero-cost identity case equals plain `gemm`: test (e)
//!
//! Accuracy against an f64 oracle is checked with a strict **per-element** absolute-tolerance gate
//! (tests c/d). All shapes are platform-independent (deterministic LCG fills, self-computed
//! references)

use crate::common::Rng;
use gemmkit::{
    Activation, Bias, MatMut, MatRef, NarrowFloat, Parallelism, Workspace, gemm, gemm_fused_with,
    tuning,
};

/// Serializes the `GEMMKIT_DEEP_KC_BYTES`-mutating deep-k test ([`fused_mixed_deep_k`]) against the
/// one other test in this binary that runs a plain narrow `gemm` at a twin-eligible shape and
/// asserts an exact single-panel result ([`fused_mixed_identity_delegates`]). The knob is
/// process-global and the harness runs these tests concurrently, so without a shared lock the deep-k
/// test's `set(1)` could flip that plain gemm onto the f32-output twin mid-run (only tolerance-equal
/// for a general `beta`) and break its bitwise assert. Poison-tolerant so one panic does not cascade
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// harness

/// The narrow float under test (`f16`/`bf16`): exact widen/narrow, bit compare, NaN, tolerance
trait Narrow: NarrowFloat + gemmkit::FusedScalar {
    fn of(x: f64) -> Self;
    /// Widen exactly to `f32` (a subset of `f32`)
    fn f32(self) -> f32;
    fn bits(self) -> u16;
    fn nan() -> Self;
    /// Machine epsilon of the 16-bit output (the dominant error is the final round)
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
    const EPS: f64 = 9.765625e-4; // 2^-10
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
    const EPS: f64 = 7.8125e-3; // 2^-7
    fn name() -> &'static str {
        "bf16"
    }
}

/// The activation kind used by the self-computed references (the public `Activation` is not
/// `Clone`, so carry a small copyable descriptor and rebuild it per call)
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

/// Bias kind: none / per-row / per-col
#[derive(Copy, Clone, PartialEq)]
enum BiasK {
    None,
    Row,
    Col,
}

/// The exact scalar mirror of `FusedEpi::<N>::apply`: `narrow(act(v + bias.widen()))`, all in
/// `f32`, a single narrowing at the end. `bias` is the already-widened `f32` bias term
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

/// Build an `m x n` narrow matrix (col-major storage) of LCG values scaled by `scale`
fn make<N: Narrow>(rng: &mut Rng, m: usize, n: usize, scale: f64) -> Vec<N> {
    (0..m * n).map(|_| N::of(rng.unit() * scale)).collect()
}

/// f64 reference for `C <- act(alpha*A*B + beta*C0 + bias)`: inputs/alpha/beta all in f64 (via the
/// exact `f32` widening), bias/act in f64, returned un-narrowed in `[i + j*m]` logical order. The
/// caller gates `got.to_f64()` against this within the mixed relative tolerance
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

/// **Per-element** accuracy gate against the f64 reference `cref` (the un-narrowed oracle value).
/// For each element `let r = cref[..]` and `got` widened to `f64`, assert
/// `|got - r| <= (2*eps_n + 8*k*f32_eps)*(1 + |r|)`, with `eps_n = N::EPS` (2^-10 / 2^-7) and
/// `f32_eps = 2^-23`
///
/// Rationale: the fused path accumulates in `f32` (relative error `O(k*f32_eps)`) and rounds once
/// to `N` (<= 1 `N`-ulp from `r`, including the double-rounding vs the f64 reference), so the sum of
/// those 2, scaled by `(1 + |r|)`, bounds a *correct* element. Structural regressions (a dropped
/// product term, a bias applied to the wrong row, a wrong activation slope) produce `O(1)`-relative
/// per-element errors, orders of magnitude above this gate, so it FAILS on them. That is the point
/// of moving off the old relative-Frobenius form, whose `||A||*||B||` denominator was so large that
/// even an all-zeros output scored ~0.08, far below the `16*k*EPS` tolerance, and could not fail.
/// `got` is read back through its strides
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
    let f32_eps = f32::EPSILON as f64; // exactly 2^-23
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

/// The fused vector fast path (unit-stride col-major C) and the scratch/scalar path (a strided C,
/// which fails the `rsc == 1` gate for *every* tile) must agree **bit-for-bit**: proving
/// `apply == apply_reg + store_out` end-to-end
///
/// `beta` is `0` and `1` (not an arbitrary `Other`) deliberately: those are the 2 states where
/// the mixed kernel's `beta*C + A*B` combine is **bit-identical** between the 2 paths (`beta == 0`
/// reads no C; `beta == 1` is a plain add on both), so the `f32` value handed to `apply` /
/// `apply_reg` is identical and the test isolates the epilogue transform structurally on every
/// ISA. (For a general `beta` the fast path fuses the combine, `mul_add`, while the scratch path,
/// preserved byte-for-byte from the pre-epilogue mixed kernel, does not, so the 2 would only
/// agree after the final narrowing absorbs a sub-narrow f32 difference: an ISA-dependent tie. The
/// pre-narrow precision contract is covered separately in tests b/c/d)
fn vector_scalar_bitwise<N: Narrow>() {
    let mut rng = Rng::new(0x11CE_A501);
    let (m, k, n) = (64usize, 96usize, 48usize);
    let alpha = N::of(1.3);
    let a = make::<N>(&mut rng, m, k, 1.0); // col-major mxk
    let b = make::<N>(&mut rng, k, n, 1.0); // col-major kxn
    let c0 = make::<N>(&mut rng, m, n, 1.0); // logical col-major mxn
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();

    for beta in [N::of(0.0), N::of(1.0)] {
        // Unit-stride col-major C (rsc = 1): the vector fast path
        let mut c_vec = c0.clone();
        // Strided C (rsc = 2, csc = 2m) over a 2x-larger buffer: |csc| >= |rsc| so no orientation
        // swap, and `rsc != 1` forces the scratch/scalar path for every tile
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

/// `k = 1, alpha = 1, beta = 0`: the accumulator is a single `f32` product `a*b`, exact in `f32`
/// (a narrow x narrow product fits the 23-bit mantissa), so it is order-independent. For each of the
/// 3 activations (`None`, `ReLU`, `LeakyReLU(0.25)`) the fused output must equal the
/// **single-rounding** reference `narrow(act(a*b + bias.widen()))` **bitwise**, and that reference
/// must DIFFER from the 2-rounding alternative `narrow(act(narrow(a*b).widen() + bias.widen()))`
/// for at least one element: locking the pre-narrow semantic **through the activation** (and
/// proving the test is not vacuous). The activations pass the divergence guard because most
/// `a*b + bias` values are positive (`a, b in [1, 2) => a*b in [1, 4)`, `bias in [-2, 2]`), where
/// `ReLU`/`LeakyReLU` are the identity, so the sub-narrow bits that make the `None` case diverge
/// survive the activation on those elements
fn pre_narrow_semantics<N: Narrow>() {
    let mut rng = Rng::new(0x9E27_B1A5);
    let (m, k, n) = (32usize, 1usize, 24usize);
    // Values in [1, 2): products land in [1, 4) with up to 2*mantissa bits, beyond `N`'s
    // mantissa, so narrowing the product first loses bits a well-chosen bias keeps significant
    let a: Vec<N> = (0..m * k)
        .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
        .collect();
    let b: Vec<N> = (0..k * n)
        .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
        .collect();
    // A per-row bias of comparable magnitude, so `a*b + bias` preserves the sub-narrow bits
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();

    for act in [ActK::None, ActK::Relu, ActK::Leaky(0.25)] {
        let mut c = vec![N::of(0.0); m * n]; // col-major, beta = 0 (unread)
        let mut ws = Workspace::new();
        gemm_fused_with(
            &mut ws,
            N::of(1.0),
            MatRef::new(&a, m, k, 1, m as isize), // col-major A (rsa = 1) -> small_k in-place
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
                // The single f32 product A[i,0]*B[0,j] (k == 1, col-major), exact
                let ab = a[i].f32() * b[j].f32();
                let biasw = bias_row[i].f32();
                let one_round: N = ref_apply_narrow::<N>(ab, biasw, act);
                // Two-rounding alternative: narrow the product first, then add bias and narrow
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

/// General driver shape, full `bias x act x beta` sweep, against the f64 oracle within tolerance
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

/// One fused config over the small-`m,n` and small-`k` special routes: accuracy vs the f64 oracle,
/// plus serial == Rayon(4) **bitwise** (the routes partition the output with no cross-thread
/// reduction, and the fused epilogue is a per-range pass)
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
    // small_mn: (8, 2048, 8), row-major A (csa == 1), col-major B (rsb == 1)
    {
        let (m, k, n) = (8usize, 2048usize, 8usize);
        let mut rng = Rng::new(0x5A11_3E00);
        let a = make::<N>(&mut rng, m, k, 1.0); // logical, stored col-major here...
        let b = make::<N>(&mut rng, k, n, 1.0);
        // ...but present A row-major: rebuild a row-major buffer so csa == 1
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
    // small_k: (100, 4, 80), col-major A (rsa == 1)
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

/// `bias None + act None` delegates to plain `gemm`, bit-for-bit (the zero-cost identity path)
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
    // Holds KNOB_LOCK: this runs a plain narrow `gemm` at a twin-eligible shape (m=48,k=40) and
    // asserts it equals the single-panel fused-identity result bitwise, which the deep-k test's
    // `GEMMKIT_DEEP_KC_BYTES` mutation would break if it raced (see KNOB_LOCK)
    let _g = knob_guard();
    identity_delegates::<gemmkit::f16>();
    identity_delegates::<gemmkit::bf16>();
}

// f. NaN -> ReLU -> 0

/// A `NaN` in the first depth column of every A row poisons each output's `k`-reduction to `NaN`,
/// which `ReLU` must map to `+0.0` on every ISA and store path: bit-for-bit `N::ZERO`. With
/// `beta = 0` the whole output is affected; the reference (`ReLU(NaN) = +0`) agrees
fn nan_relu<N: Narrow>() {
    let mut rng = Rng::new(0x4A11_DEAD);
    let (m, k, n) = (64usize, 3usize, 48usize);
    let mut a = make::<N>(&mut rng, m, k, 1.0); // col-major mxk
    let b = make::<N>(&mut rng, k, n, 1.0);
    for i in 0..m {
        a[i] = N::nan(); // A[i, 0] at i + 0*m
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

/// Past the deep-contraction engage gate (forced low via `GEMMKIT_DEEP_KC_BYTES = 1`), a plain narrow
/// `gemm` re-blocks through the f32-output twin (`MixedGemmF32` / `Bf16DotGemmF32`), while `gemm_fused`
/// deliberately stays single-panel (its dispatch has **no** deep-k branch, documented in
/// `dispatch::mixed`). This locks that split: at the same deep `k`, the fused entry must still be
/// accurate against the f64 oracle and reproduce serial==parallel bit-for-bit, and the plain path
/// (now the twin) must too. Accuracy - not a bit compare vs plain-then-map - is the fused oracle here:
/// the mixed epilogue applies pre-narrow, so fused is *more* precise than gemm-then-map (see the module
/// doc), the same discipline as tests c/d
fn deep_k<N: Narrow>() {
    // General-driver shape: m,n past `small_mn_dim` and k past `small_k_threshold`, so no special
    // route claims it. The plain path then reaches the deep-k engage gate; the fused path the driver
    // k stays small for GEMMKIT_FAST_TEST (the forced gate engages the twin at any k > small_k)
    let (m, k, n) = (48usize, 96usize, 40usize);
    let mut rng = Rng::new(0xDEE9_CA5E);
    let alpha = N::of(1.0);
    let beta = N::of(0.7);
    let a = make::<N>(&mut rng, m, k, 1.0);
    let b = make::<N>(&mut rng, k, n, 1.0);
    let c0 = make::<N>(&mut rng, m, n, 1.0);
    let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
    let act = ActK::Leaky(0.25);

    // Hold the lock for the whole knob window: `fused_mixed_identity_delegates` waits it out
    let _g = knob_guard();
    let restore = tuning::deep_kc_bytes();
    tuning::set_deep_kc_bytes(1); // engage the twin for the plain path at any k > small_k

    // FUSED at deep k: stays single-panel. Serial == parallel bitwise, and accurate vs the oracle
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

    // PLAIN at deep k: re-blocks through the f32-output twin. Serial == parallel bitwise, accurate
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
