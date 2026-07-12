//! Fused-epilogue suite (spec §10): the determinism/precision contract for `gemm_fused`
//! (bias + ReLU/LeakyReLU) and `gemm_i8_requant`.
//!
//! Every comparison is **bitwise** (raw bit patterns, so NaN/−0 are exercised). The oracle
//! is "plain GEMM, then the exact scalar map": a fused GEMM must equal it bit-for-bit,
//! because the epilogue is applied to the very value the plain store would have written and
//! blocking is epilogue-independent. All shapes are platform-independent; no machine numbers.
//!
//! The `gemm`/`gemm_fused` oracle holds only where `gemm` uses the general driver (the same
//! path `gemm_fused` always takes), so the tests use driver shapes (m,n > 16, k > 16, not
//! gemv). The requantize oracle is stronger — the `i32` accumulation is exact and
//! ISA-independent, so it holds bitwise under every `GEMMKIT_REQUIRE_ISA` pin.

// The index loops walk C and the bias vectors at different (strided) offsets, so explicit
// indices read clearer than zipped iterators here.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_fused};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift* RNG (no external dep, reproducible across runs).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F491_4F6CDD1D)
    }
    /// A value in roughly `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The real-float element under test (`f32`/`f64`): construction, bit compare, and the exact
/// reference epilogue map (a byte-for-byte mirror of `FusedEpi::apply`).
trait Flt: gemmkit::FusedScalar {
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
/// same order, so it agrees bitwise with the fused vector *and* scratch paths.
fn ref_apply<T: Flt>(v: T, bias: Option<T>, act: &Option<Activation<T>>) -> T {
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

/// A strided C layout to test.
#[derive(Copy, Clone)]
enum Layout {
    Col,
    Row,
    /// Column-major with a padded row stride (strided C, forces the scratch path at edges).
    ColPadded,
}

fn c_strides(layout: Layout, m: usize, n: usize) -> (isize, isize, usize) {
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

/// Build an `m × n` matrix (col-major storage) of RNG values.
fn make<T: Flt>(rng: &mut Rng, m: usize, n: usize) -> Vec<T> {
    (0..m * n).map(|_| T::of(rng.unit() * 2.0)).collect()
}

// ---------------------------------------------------------------------------
// test 2 (the core oracle): fused == gemm-then-map, bitwise
// ---------------------------------------------------------------------------

/// Run one fused case and its `gemm`+map oracle; assert bitwise-equal C. `bias`/`act` are
/// applied in the user frame; the reference reads back `gemm`'s output and maps it.
fn check_fused<T: Flt>(
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
    let a = make::<T>(rng, m, k); // col-major m×k
    let b = make::<T>(rng, k, n); // col-major k×n
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen, 1);

    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias = match bias_kind {
        1 => Some(Bias::PerRow(&bias_row)),
        2 => Some(Bias::PerCol(&bias_col)),
        _ => None,
    };

    // --- fused ---
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

    // --- oracle: plain gemm then the scalar map (user frame) ---
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
/// `gemm_fused_with` over raw col-major LHS/RHS storage.
fn gemm_fused_with_layout<T: Flt>(
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

/// `Option<Activation<T>>` is not `Clone` (T need not be), so clone it explicitly.
trait CloneLike<T> {
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

fn fused_matrix<T: Flt>(par: Parallelism) {
    let mut rng = Rng::new(0xE91109E1);
    let shapes = [
        (17usize, 17usize, 17usize), // just above small_mn/small_k
        (33, 40, 24),                // rectangular, tile edges
        (64, 64, 64),
        (48, 96, 129), // row/col edges vs tiles
    ];
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.1))),
    ];
    for &(m, k, n) in &shapes {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            check_fused::<T>(
                                &mut rng,
                                m,
                                k,
                                n,
                                alpha,
                                beta,
                                layout,
                                bias_kind,
                                act.clone_like(),
                                par,
                                "matrix",
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fused_eq_gemm_then_map_serial() {
    fused_matrix::<f32>(Parallelism::Serial);
    fused_matrix::<f64>(Parallelism::Serial);
}

#[test]
fn fused_eq_gemm_then_map_parallel() {
    fused_matrix::<f32>(Parallelism::Rayon(8));
    fused_matrix::<f64>(Parallelism::Rayon(8));
}

// ---------------------------------------------------------------------------
// test 1: identity delegation + run_epilogue plumbing
// ---------------------------------------------------------------------------

/// `gemm_fused(None, None)` must delegate to plain `gemm`, bit-for-bit — the zero-cost
/// identity case never even reaches a fused monomorphization.
#[test]
fn identity_delegates_to_gemm() {
    let mut rng = Rng::new(42);
    for &(m, k, n) in &[(17usize, 20usize, 19usize), (64, 33, 48)] {
        for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
            for par in [Parallelism::Serial, Parallelism::Rayon(8)] {
                let a = make::<f32>(&mut rng, m, k);
                let b = make::<f32>(&mut rng, k, n);
                let (rsc, csc, clen) = c_strides(layout, m, n);
                let c0 = make::<f32>(&mut rng, clen, 1);

                let mut c_fused = c0.clone();
                let mut c_ref = c0.clone();
                {
                    let ar = MatRef::new(&a, m, k, 1, m as isize);
                    let br = MatRef::new(&b, k, n, 1, k as isize);
                    let cm = MatMut::new(&mut c_fused, m, n, rsc, csc);
                    gemm_fused(1.0f32, ar, br, 0.5, cm, None, None, par);
                }
                {
                    let ar = MatRef::new(&a, m, k, 1, m as isize);
                    let br = MatRef::new(&b, k, n, 1, k as isize);
                    let cm = MatMut::new(&mut c_ref, m, n, rsc, csc);
                    gemm(1.0f32, ar, br, 0.5, cm, par);
                }
                for (x, y) in c_fused.iter().zip(c_ref.iter()) {
                    assert_eq!(x.to_bits(), y.to_bits(), "identity-fused != gemm");
                }
            }
        }
    }
}

/// The internal `run_epilogue::<Identity>` path is byte-identical to the plain `run` path
/// (the observational zero-cost-identity proof), across strides and parallelism, for a fixed
/// ISA token (`ScalarTok`, always valid regardless of `GEMMKIT_REQUIRE_ISA`).
#[test]
fn run_epilogue_identity_matches_run() {
    use gemmkit::driver;
    use gemmkit::kernel::{FloatGemm, Identity};
    use gemmkit::simd::ScalarTok;

    let mut rng = Rng::new(7);
    for &(m, k, n) in &[(20usize, 24usize, 18usize), (40, 32, 40)] {
        for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
            let a = make::<f32>(&mut rng, m, k);
            let b = make::<f32>(&mut rng, k, n);
            let c0 = make::<f32>(&mut rng, m * n, 1);
            let mut c_run = c0.clone();
            let mut c_epi = c0.clone();
            let mut ws = Workspace::new();
            // SAFETY: valid col-major A/B/C, disjoint buffers, ScalarTok always runnable.
            unsafe {
                driver::run::<FloatGemm<f32>, ScalarTok, 4, 4>(
                    ScalarTok,
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    1,
                    m as isize,
                    b.as_ptr(),
                    1,
                    k as isize,
                    0.7,
                    c_run.as_mut_ptr(),
                    1,
                    m as isize,
                    par,
                    &mut ws,
                );
                driver::run_epilogue::<FloatGemm<f32>, ScalarTok, Identity, 4, 4>(
                    ScalarTok,
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    1,
                    m as isize,
                    b.as_ptr(),
                    1,
                    k as isize,
                    0.7,
                    c_epi.as_mut_ptr(),
                    1,
                    m as isize,
                    &Identity,
                    par,
                    &mut ws,
                );
            }
            for (x, y) in c_run.iter().zip(c_epi.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "run != run_epilogue::<Identity>");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// test 3: fire-once (multi-panel K) — a per-panel epilogue would diverge from the oracle
// ---------------------------------------------------------------------------

/// A large-`k` driver shape forces `kc < k` (multiple depth slices), so the epilogue must
/// fire exactly once, on the final panel. Sign-mixed data + `beta = 0.7` makes a per-panel
/// ReLU or a re-added bias diverge loudly from the oracle.
#[test]
fn fire_once_multi_panel() {
    let mut rng = Rng::new(0xF11E);
    // k = 4096 is far above any realistic L1-fit kc (~512), so there are several pc slices.
    check_fused::<f32>(
        &mut rng,
        40,
        4096,
        40,
        1.0,
        0.7,
        Layout::Col,
        1, // per-row bias
        Some(Activation::Relu),
        Parallelism::Serial,
        "fire-once/serial",
    );
    check_fused::<f32>(
        &mut rng,
        40,
        4096,
        40,
        0.9,
        0.7,
        Layout::Row,
        2, // per-col bias
        Some(Activation::LeakyRelu(0.1)),
        Parallelism::Rayon(8),
        "fire-once/parallel",
    );
}

// ---------------------------------------------------------------------------
// test 4: bias orientation matrix (col-major vs row-major C)
// ---------------------------------------------------------------------------

/// {PerRow, PerCol} × {col-major, row-major} C. `check_fused` applies the reference bias in
/// the user frame, so a wrong orientation-flip (row-major C swaps m↔n internally) diverges.
#[test]
fn bias_orientation() {
    let mut rng = Rng::new(0xB1A5);
    for bias_kind in [1u8, 2u8] {
        for layout in [Layout::Col, Layout::Row] {
            check_fused::<f32>(
                &mut rng,
                33,
                40,
                21,
                1.0,
                0.0,
                layout,
                bias_kind,
                None,
                Parallelism::Serial,
                "orient",
            );
            check_fused::<f64>(
                &mut rng,
                33,
                40,
                21,
                1.0,
                0.0,
                layout,
                bias_kind,
                None,
                Parallelism::Serial,
                "orient",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// test 5: NaN / -0 semantics through the fused path
// ---------------------------------------------------------------------------

/// A NaN accumulator (from `inf - inf`) must map to 0 under ReLU/LeakyReLU on every ISA, and
/// −0 must map to +0 — verified end-to-end through `gemm_fused` bitwise. `m`/`n` are `>= mr`
/// and `>= NR` with a col-major C so the full tiles take the SIMD fast path, exercising the
/// `max`/`min` NaN-in-`a` contract (`_mm512_max_ps` / `vmaxnmq` / `f32x4_pmax`), not only the
/// scalar `apply`.
#[test]
fn nan_and_neg_zero() {
    nan_and_neg_zero_for::<f32>();
    nan_and_neg_zero_for::<f64>();
}

fn nan_and_neg_zero_for<T: Flt>() {
    let m = 64usize;
    let k = 2usize;
    let n = 64usize;
    let ctx = T::name();
    // A·B with `inf` inputs: product 0 = inf·1 = +inf, product 1 = inf·(-1) = -inf, and
    // +inf + (-inf) = NaN. (Using `inf` inputs — not `MAX` — is robust under FMA, whose
    // exact intermediate product would otherwise keep `MAX·MAX + inf` finite-then-inf.)
    let mut a = vec![T::of(0.0); m * k];
    let mut b = vec![T::of(0.0); k * n];
    for i in 0..m {
        a[i] = T::of(f64::INFINITY); // column 0
        a[m + i] = T::of(f64::INFINITY); // column 1
    }
    for j in 0..n {
        b[k * j] = T::of(1.0); // row 0 => +inf term
        b[k * j + 1] = T::of(-1.0); // row 1 => -inf term  => inf + (-inf) = NaN
    }
    for &act in &[0u8, 1u8] {
        let activation = if act == 1 {
            Some(Activation::Relu)
        } else {
            Some(Activation::LeakyRelu(T::of(0.25)))
        };
        let mut c = vec![T::of(0.0); m * n];
        {
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut c, m, n, 1, m as isize);
            gemm_fused(
                T::of(1.0),
                ar,
                br,
                T::of(0.0),
                cm,
                None,
                activation.clone_like(),
                Parallelism::Serial,
            );
        }
        for &v in &c {
            assert_eq!(
                v.bits(),
                T::of(0.0).bits(),
                "{ctx}: ReLU/Leaky(NaN) must be +0.0"
            );
        }
    }

    // −0 handling: a zero product with a negative slope must yield +0 under LeakyReLU.
    let a2 = vec![T::of(0.0); m * k];
    let b2 = vec![T::of(1.0); k * n];
    let mut c2 = vec![T::of(0.0); m * n];
    {
        let ar = MatRef::new(&a2, m, k, 1, m as isize);
        let br = MatRef::new(&b2, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c2, m, n, 1, m as isize);
        gemm_fused(
            T::of(1.0),
            ar,
            br,
            T::of(0.0),
            cm,
            None,
            Some(Activation::LeakyRelu(T::of(-0.5))),
            Parallelism::Serial,
        );
    }
    for &v in &c2 {
        assert_eq!(
            v.bits(),
            T::of(0.0).bits(),
            "{ctx}: LeakyReLU(0) must be +0.0"
        );
    }
}

// ---------------------------------------------------------------------------
// test 7: degenerate fused cases (k == 0 / alpha == 0 => C <- act(beta*C + bias))
// ---------------------------------------------------------------------------

#[test]
fn fused_degenerate() {
    let mut rng = Rng::new(0xDE6E);
    for &(m, n) in &[(20usize, 24usize)] {
        // k == 0
        let bias: Vec<f32> = (0..m).map(|_| rng.unit() as f32).collect();
        let c0 = make::<f32>(&mut rng, m * n, 1);
        for &(k, alpha) in &[(0usize, 1.0f32), (24usize, 0.0f32)] {
            let a = make::<f32>(&mut rng, m, k.max(1));
            let b = make::<f32>(&mut rng, k.max(1), n);
            let mut c = c0.clone();
            {
                let ar = MatRef::new(&a, m, k, 1, m as isize);
                let br = MatRef::new(&b, k, n, 1, k as isize);
                let cm = MatMut::new(&mut c, m, n, 1, m as isize);
                gemm_fused(
                    alpha,
                    ar,
                    br,
                    0.5,
                    cm,
                    Some(Bias::PerRow(&bias)),
                    Some(Activation::Relu),
                    Parallelism::Serial,
                );
            }
            // Reference: C = ReLU(0.5*C0 + bias[i]).
            for j in 0..n {
                for i in 0..m {
                    let idx = i + j * m;
                    let want = ref_apply(0.5f32 * c0[idx], Some(bias[i]), &Some(Activation::Relu));
                    assert_eq!(c[idx].to_bits(), want.to_bits(), "degenerate fused");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// test 8: validation panics
// ---------------------------------------------------------------------------

mod validation {
    use super::*;

    fn base() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![1.0f32; 4 * 4],
            vec![1.0f32; 4 * 4],
            vec![0.0f32; 4 * 4],
        )
    }

    #[test]
    #[should_panic(expected = "bias length")]
    fn bias_wrong_length() {
        let (a, b, mut c) = base();
        let bias = vec![0.0f32; 3]; // should be 4 (PerRow, m == 4)
        gemm_fused(
            1.0,
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::from_col_major(&mut c, 4, 4),
            Some(Bias::PerRow(&bias)),
            None,
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "LeakyRelu slope must be finite")]
    fn leaky_slope_not_finite() {
        let (a, b, mut c) = base();
        gemm_fused(
            1.0,
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::from_col_major(&mut c, 4, 4),
            None,
            Some(Activation::LeakyRelu(f32::INFINITY)),
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "bias slice overlaps C")]
    fn bias_overlaps_c() {
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let mut buf = vec![0.0f32; 16];
        // A bias slice aliasing C's storage. It is raw-derived (its lifetime is not tied to
        // `buf`), so `&mut buf` still type-checks; `gemm_fused` panics on the overlap check
        // before any element is read or written, so no aliased access occurs.
        let bias: &[f32] = unsafe { core::slice::from_raw_parts(buf.as_ptr(), 4) };
        gemm_fused(
            1.0,
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::from_col_major(&mut buf, 4, 4),
            Some(Bias::PerRow(bias)),
            None,
            Parallelism::Serial,
        );
    }
}

// ---------------------------------------------------------------------------
// checked/unchecked twin equivalence
// ---------------------------------------------------------------------------

/// `gemm_fused` and `gemm_fused_unchecked` are **parallel** entry points — the checked twin
/// does not delegate to the unchecked one, so a divergence in the `Bias`/`Act` translation
/// would go silently undetected. Exercise the unchecked fn against the checked twin on a
/// driver-shaped problem (m,n,k > 16), bit-for-bit, across both `BiasDim` arms, `has_bias =
/// false`, and every activation arm.
#[test]
fn fused_unchecked_matches_checked() {
    use gemmkit::{BiasDim, gemm_fused, gemm_fused_unchecked};

    let mut rng = Rng::new(0x0F05_ED12);
    let (m, k, n) = (33usize, 24usize, 40usize);
    let a = make::<f32>(&mut rng, m, k); // col-major m×k
    let b = make::<f32>(&mut rng, k, n); // col-major k×n
    let c0 = make::<f32>(&mut rng, m * n, 1); // col-major m×n C
    let bias_row: Vec<f32> = (0..m).map(|_| (rng.unit() * 3.0) as f32).collect();
    let bias_col: Vec<f32> = (0..n).map(|_| (rng.unit() * 3.0) as f32).collect();
    let (alpha, beta) = (0.9f32, 0.7f32);
    let par = Parallelism::Serial;

    let mk_act = |kind: u8| match kind {
        1 => Some(Activation::Relu),
        2 => Some(Activation::LeakyRelu(0.1f32)),
        _ => None,
    };

    // 0 none, 1 per-row, 2 per-col.
    for bias_kind in 0u8..=2 {
        for act_kind in 0u8..=2 {
            let bias_checked = match bias_kind {
                1 => Some(Bias::PerRow(&bias_row)),
                2 => Some(Bias::PerCol(&bias_col)),
                _ => None,
            };
            let (bptr, bdim, has_bias) = match bias_kind {
                1 => (bias_row.as_ptr(), BiasDim::PerRow, true),
                2 => (bias_col.as_ptr(), BiasDim::PerCol, true),
                _ => (core::ptr::null(), BiasDim::PerRow, false),
            };

            let mut c_checked = c0.clone();
            {
                let ar = MatRef::new(&a, m, k, 1, m as isize);
                let br = MatRef::new(&b, k, n, 1, k as isize);
                let cm = MatMut::new(&mut c_checked, m, n, 1, m as isize);
                gemm_fused(alpha, ar, br, beta, cm, bias_checked, mk_act(act_kind), par);
            }

            let mut c_unchecked = c0.clone();
            // SAFETY: every view is a valid in-bounds col-major layout, C aliases neither A/B
            // nor the bias, and the bias slice (when present) is the right length for its axis.
            unsafe {
                gemm_fused_unchecked(
                    m,
                    k,
                    n,
                    alpha,
                    a.as_ptr(),
                    1,
                    m as isize,
                    b.as_ptr(),
                    1,
                    k as isize,
                    beta,
                    c_unchecked.as_mut_ptr(),
                    1,
                    m as isize,
                    bptr,
                    bdim,
                    has_bias,
                    mk_act(act_kind),
                    par,
                );
            }

            for idx in 0..m * n {
                assert_eq!(
                    c_checked[idx].to_bits(),
                    c_unchecked[idx].to_bits(),
                    "fused unchecked != checked at {idx} [bias_kind={bias_kind} act_kind={act_kind}]",
                );
            }
        }
    }
}

// ===========================================================================
// Requantize (i8 -> i8) — bitwise vs gemm_i8-then-map, ties, saturation, bias.
// ===========================================================================

#[cfg(feature = "int8")]
mod requant {
    use super::Rng;
    use gemmkit::{MatMut, MatRef, Parallelism, Requantize, gemm_i8, gemm_i8_requant};

    /// The reference requantize map. The rounding uses the std `round_ties_even` — an
    /// *independent* implementation of the contract, NOT a copy of the kernel's `2^52`
    /// `round_ne_f64` — so a regression in the kernel's rounding is caught here rather than
    /// mirrored. Applied to the exact `i32` accumulator from `gemm_i8`.
    fn ref_requant(acc: i32, bias: i32, scale: f32, zp: i32) -> i8 {
        let scaled = (f64::from(acc.wrapping_add(bias)) * f64::from(scale)).round_ties_even();
        let q = (scaled as i64).saturating_add(i64::from(zp));
        q.clamp(-128, 127) as i8
    }

    fn make_i8(rng: &mut Rng, n: usize) -> Vec<i8> {
        (0..n)
            .map(|_| ((rng.next_u64() % 255) as i64 - 127) as i8)
            .collect()
    }

    /// Bitwise: `gemm_i8_requant` == `gemm_i8` (into i32) then the scalar requant map. Since
    /// the `i32` accumulation is exact and ISA-independent, this holds under any ISA pin.
    fn check_requant(
        rng: &mut Rng,
        m: usize,
        k: usize,
        n: usize,
        scale: f32,
        zp: i32,
        has_bias: bool,
        row_major_c: bool,
        par: Parallelism,
        tag: &str,
    ) {
        let a = make_i8(rng, m * k); // col-major m×k
        let b = make_i8(rng, k * n); // col-major k×n
        let bias: Vec<i32> = if has_bias {
            (0..m)
                .map(|_| (rng.next_u64() % 2001) as i64 as i32 - 1000)
                .collect()
        } else {
            Vec::new()
        };
        let (rsc, csc) = if row_major_c {
            (n as isize, 1isize)
        } else {
            (1isize, m as isize)
        };

        // exact i32 accumulator via gemm_i8
        let mut acc = vec![0i32; m * n];
        {
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut acc, m, n, rsc, csc);
            gemm_i8(1, ar, br, 0, cm, par);
        }

        // fused requantize
        let mut c = vec![0i8; m * n];
        {
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut c, m, n, rsc, csc);
            let req = Requantize {
                scale,
                zero_point: zp,
                bias: if has_bias { Some(&bias) } else { None },
            };
            gemm_i8_requant(ar, br, req, cm, par);
        }

        for j in 0..n {
            for i in 0..m {
                let idx = (i as isize * rsc + j as isize * csc) as usize;
                let bterm = if has_bias { bias[i] } else { 0 };
                let want = ref_requant(acc[idx], bterm, scale, zp);
                assert_eq!(
                    c[idx], want,
                    "{tag}: requant mismatch at ({i},{j}) acc={} [m={m} k={k} n={n}]",
                    acc[idx],
                );
            }
        }
    }

    #[test]
    fn requant_bitwise_matrix() {
        let mut rng = Rng::new(0x9111);
        for &(m, k, n) in &[(17usize, 20usize, 19usize), (32, 40, 24), (48, 300, 33)] {
            for &scale in &[0.003f32, 0.5, 1.0, 7.25] {
                for &zp in &[-128i32, -13, 0, 27, 127] {
                    for has_bias in [false, true] {
                        for row_major in [false, true] {
                            for par in [Parallelism::Serial, Parallelism::Rayon(8)] {
                                check_requant(
                                    &mut rng, m, k, n, scale, zp, has_bias, row_major, par,
                                    "matrix",
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `gemm_i8_requant` and `gemm_i8_requant_unchecked` are **parallel** entry points (the
    /// checked twin does not delegate to the unchecked one), so exercise the unchecked fn
    /// against the checked twin bit-for-bit on a driver-shaped case (m,n,k > 16, with bias).
    #[test]
    fn requant_unchecked_matches_checked() {
        use gemmkit::gemm_i8_requant_unchecked;

        let mut rng = Rng::new(0x5EED_1234);
        let (m, k, n) = (32usize, 40usize, 24usize);
        let a = make_i8(&mut rng, m * k);
        let b = make_i8(&mut rng, k * n);
        let bias: Vec<i32> = (0..m)
            .map(|_| (rng.next_u64() % 2001) as i64 as i32 - 1000)
            .collect();
        let (scale, zp) = (0.5f32, 13i32);
        let (rsc, csc) = (1isize, m as isize);
        let par = Parallelism::Serial;

        let mut c_checked = vec![0i8; m * n];
        {
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut c_checked, m, n, rsc, csc);
            let req = Requantize {
                scale,
                zero_point: zp,
                bias: Some(&bias),
            };
            gemm_i8_requant(ar, br, req, cm, par);
        }

        let mut c_unchecked = vec![0i8; m * n];
        // SAFETY: valid in-bounds col-major layouts; C aliases neither A/B nor the (per-row,
        // length-m) bias.
        unsafe {
            gemm_i8_requant_unchecked(
                m,
                k,
                n,
                a.as_ptr(),
                1,
                m as isize,
                b.as_ptr(),
                1,
                k as isize,
                scale,
                zp,
                bias.as_ptr(),
                true,
                c_unchecked.as_mut_ptr(),
                rsc,
                csc,
                par,
            );
        }

        assert_eq!(
            c_checked, c_unchecked,
            "requant unchecked != checked [m={m} k={k} n={n}]"
        );
    }

    /// Hardcoded round-half-to-even ties, independent of any reference function: each row is a
    /// `1×1` product giving an exact `acc`, and `scale = 0.5` lands `scale·acc` on a half-integer.
    /// A round-half-up/away regression would flip 0.5→1, 2.5→3, etc.
    #[test]
    fn requant_ties_even_exact() {
        let a: [i8; 6] = [1, 3, 5, 7, -1, -3];
        let b: [i8; 1] = [1];
        // scale=0.5: 0.5→0, 1.5→2, 2.5→2, 3.5→4, -0.5→0, -1.5→-2 (ties to even).
        let expect: [i8; 6] = [0, 2, 2, 4, 0, -2];
        let mut c = [0i8; 6];
        gemm_i8_requant(
            MatRef::from_col_major(&a, 6, 1),
            MatRef::from_col_major(&b, 1, 1),
            Requantize {
                scale: 0.5,
                zero_point: 0,
                bias: None,
            },
            MatMut::from_col_major(&mut c, 6, 1),
            Parallelism::Serial,
        );
        assert_eq!(c, expect, "round-half-to-even ties");
    }

    /// Round-half-to-even ties (incl. odd zero-point) and saturation both ends.
    #[test]
    fn requant_ties_and_saturation() {
        let mut rng = Rng::new(0x7135);
        // A large scale drives many outputs to the ±clamp; a range of k exercises exact-tie
        // half-integers as scale*acc lands on x.5.
        for &k in &[1usize, 8, 300, 1000, 5000] {
            check_requant(
                &mut rng,
                20,
                k,
                18,
                0.5,
                3,
                true,
                false,
                Parallelism::Serial,
                "ties",
            );
            check_requant(
                &mut rng,
                20,
                k,
                18,
                100.0,
                120,
                false,
                false,
                Parallelism::Serial,
                "sat+",
            );
            check_requant(
                &mut rng,
                20,
                k,
                18,
                100.0,
                -120,
                true,
                true,
                Parallelism::Rayon(8),
                "sat-",
            );
        }
    }

    /// Small mnk under Rayon(8): exercises the auto-VNNI small-parallel fallback to the widen
    /// `IntGemmQ` (bit-exact-equal), so the fused output still matches the oracle.
    #[test]
    fn requant_small_parallel_fallback() {
        let mut rng = Rng::new(0xFA11);
        check_requant(
            &mut rng,
            24,
            24,
            24,
            0.01,
            5,
            true,
            false,
            Parallelism::Rayon(8),
            "small-par",
        );
    }

    /// Degenerate `k == 0`: C fills with `clamp(zp + round_ne(scale*bias))` (= `zp` without
    /// bias).
    #[test]
    fn requant_degenerate_k0() {
        let m = 12usize;
        let n = 10usize;
        let bias: Vec<i32> = (0..m).map(|i| i as i32 * 40 - 200).collect();
        let a: Vec<i8> = Vec::new();
        let b: Vec<i8> = Vec::new();
        let scale = 0.5f32;
        let zp = 7i32;
        let mut c = vec![99i8; m * n];
        {
            let ar = MatRef::new(&a, m, 0, 1, m as isize);
            let br = MatRef::new(&b, 0, n, 1, 0);
            let cm = MatMut::new(&mut c, m, n, 1, m as isize);
            let req = Requantize {
                scale,
                zero_point: zp,
                bias: Some(&bias),
            };
            gemm_i8_requant(ar, br, req, cm, Parallelism::Serial);
        }
        for j in 0..n {
            for i in 0..m {
                let want = ref_requant(0, bias[i], scale, zp);
                assert_eq!(c[i + j * m], want, "degenerate requant ({i},{j})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "scale")]
    fn requant_bad_scale() {
        let a = vec![0i8; 16];
        let b = vec![0i8; 16];
        let mut c = vec![0i8; 16];
        let req = Requantize {
            scale: 0.0,
            zero_point: 0,
            bias: None,
        };
        gemm_i8_requant(
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            req,
            MatMut::from_col_major(&mut c, 4, 4),
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "zero_point")]
    fn requant_bad_zp() {
        let a = vec![0i8; 16];
        let b = vec![0i8; 16];
        let mut c = vec![0i8; 16];
        let req = Requantize {
            scale: 1.0,
            zero_point: 200,
            bias: None,
        };
        gemm_i8_requant(
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            req,
            MatMut::from_col_major(&mut c, 4, 4),
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "bias length")]
    fn requant_bad_bias_len() {
        let a = vec![0i8; 16];
        let b = vec![0i8; 16];
        let mut c = vec![0i8; 16];
        let bias = vec![0i32; 3];
        let req = Requantize {
            scale: 1.0,
            zero_point: 0,
            bias: Some(&bias),
        };
        gemm_i8_requant(
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            req,
            MatMut::from_col_major(&mut c, 4, 4),
            Parallelism::Serial,
        );
    }
}
