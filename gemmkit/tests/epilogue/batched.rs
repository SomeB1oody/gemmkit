//! Strided-batched fused-epilogue tests (spec §10, Phase 3): `gemm_batched_fused` /
//! `gemm_batched_fused_with`.
//!
//! The headline contract is that the batched-fused call is **bit-identical to a loop of
//! `gemm_fused` calls** — one per element, sharing the single bias vector and single activation.
//! That holds for every element type (`f32`/`f64` and, under `half`, `f16`/`bf16`), since each
//! element re-dispatches through the very same fused engine `gemm_fused` uses. Comparisons are
//! **bitwise** (raw bit patterns). Fills are the deterministic, platform-independent
//! [`crate::common::Rng`]; every reference is self-computed.

use crate::common::Rng;
use gemmkit::{
    Activation, Bias, FusedScalar, MatMut, MatRef, Parallelism, Workspace, gemm_batched,
    gemm_batched_fused, gemm_fused_with,
};

/// The element under test: real floats and (under `half`) narrow floats. `of`/`bits` give
/// deterministic construction and a bitwise compare; `leaky` builds a `LeakyRelu` in `T` (the
/// public `Activation` is not `Clone`, so it is rebuilt per call).
trait BEl: FusedScalar {
    fn of(x: f64) -> Self;
    fn bits(self) -> u64;
    fn name() -> &'static str;
    fn leaky(slope: f64) -> Activation<Self>;
}
impl BEl for f32 {
    fn of(x: f64) -> Self {
        x as f32
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn name() -> &'static str {
        "f32"
    }
    fn leaky(slope: f64) -> Activation<Self> {
        Activation::LeakyRelu(slope as f32)
    }
}
impl BEl for f64 {
    fn of(x: f64) -> Self {
        x
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn name() -> &'static str {
        "f64"
    }
    fn leaky(slope: f64) -> Activation<Self> {
        Activation::LeakyRelu(slope)
    }
}
#[cfg(feature = "half")]
impl BEl for gemmkit::f16 {
    fn of(x: f64) -> Self {
        gemmkit::f16::from_f64(x)
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn name() -> &'static str {
        "f16"
    }
    fn leaky(slope: f64) -> Activation<Self> {
        Activation::LeakyRelu(gemmkit::f16::from_f64(slope))
    }
}
#[cfg(feature = "half")]
impl BEl for gemmkit::bf16 {
    fn of(x: f64) -> Self {
        gemmkit::bf16::from_f64(x)
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn name() -> &'static str {
        "bf16"
    }
    fn leaky(slope: f64) -> Activation<Self> {
        Activation::LeakyRelu(gemmkit::bf16::from_f64(slope))
    }
}

/// A length-`len` column-major buffer of RNG values scaled by `scale`.
fn make_vec<T: BEl>(rng: &mut Rng, len: usize, scale: f64) -> Vec<T> {
    (0..len).map(|_| T::of(rng.unit() * scale)).collect()
}

// ---------------------------------------------------------------------------
// a. batched_fused == loop of gemm_fused, bitwise (PerRow bias + LeakyRelu, beta != 0)
// ---------------------------------------------------------------------------

fn matches_loop<T: BEl>() {
    let (batch, m, k, n) = (5usize, 31usize, 17usize, 23usize);
    let mut rng = Rng::new(0xBA7C_11FE);
    // `batch` contiguously-packed column-major elements (element stride = m*k etc.).
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // ONE shared per-row bias (length m)
    let alpha = T::of(0.9);
    let beta = T::of(0.7);

    // Reference: one gemm_fused per element on its own manually-offset slice window, Serial.
    let mut c_ref = c0.clone();
    let mut ws = Workspace::new();
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a[ao..ao + m * k], m, k, 1, m as isize),
            MatRef::new(&b[bo..bo + k * n], k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_ref[co..co + m * n], m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            Some(T::leaky(0.25)),
            Parallelism::Serial,
        );
    }

    // Each element runs serially under both schedules for these tiny shapes, so the batched result
    // is bit-identical to the Serial loop regardless of which schedule the host selects.
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        let mut c_bat = c0.clone();
        gemm_batched_fused(
            batch,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            beta,
            MatMut::new(&mut c_bat, m, n, 1, m as isize),
            (m * n) as isize,
            Some(Bias::PerRow(&bias_row)),
            Some(T::leaky(0.25)),
            par,
        );
        for idx in 0..batch * m * n {
            assert_eq!(
                c_bat[idx].bits(),
                c_ref[idx].bits(),
                "{}: batched_fused != gemm_fused loop at {idx} (par={par:?})",
                T::name(),
            );
        }
    }
}

#[test]
fn batched_fused_matches_loop() {
    matches_loop::<f32>();
    matches_loop::<f64>();
}

#[cfg(feature = "half")]
#[test]
fn batched_fused_matches_loop_narrow() {
    matches_loop::<gemmkit::f16>();
    matches_loop::<gemmkit::bf16>();
}

// ---------------------------------------------------------------------------
// b. broadcast A (a_batch_stride = 0): PerCol bias + Relu, bitwise vs the gemm_fused loop
// ---------------------------------------------------------------------------

fn broadcast_a<T: BEl>() {
    let (batch, m, k, n) = (6usize, 20usize, 12usize, 14usize);
    let mut rng = Rng::new(0x0AD_C0DE);
    let a = make_vec::<T>(&mut rng, m * k, 1.0); // ONE element of A, shared across the batch
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_col = make_vec::<T>(&mut rng, n, 2.0); // shared per-col bias (length n)
    let alpha = T::of(1.0);
    let beta = T::of(0.5);

    let mut c_ref = c0.clone();
    let mut ws = Workspace::new();
    for bi in 0..batch {
        let (bo, co) = (bi * k * n, bi * m * n);
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize), // same A every element
            MatRef::new(&b[bo..bo + k * n], k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_ref[co..co + m * n], m, n, 1, m as isize),
            Some(Bias::PerCol(&bias_col)),
            Some(Activation::Relu),
            Parallelism::Serial,
        );
    }

    let mut c_bat = c0.clone();
    gemm_batched_fused(
        batch,
        alpha,
        MatRef::new(&a, m, k, 1, m as isize),
        0, // broadcast A across the batch
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        beta,
        MatMut::new(&mut c_bat, m, n, 1, m as isize),
        (m * n) as isize,
        Some(Bias::PerCol(&bias_col)),
        Some(Activation::Relu),
        Parallelism::Rayon(4),
    );

    for idx in 0..batch * m * n {
        assert_eq!(
            c_bat[idx].bits(),
            c_ref[idx].bits(),
            "{}: broadcast-A batched_fused != gemm_fused loop at {idx}",
            T::name(),
        );
    }
}

#[test]
fn batched_fused_broadcast_a() {
    broadcast_a::<f32>();
    broadcast_a::<f64>();
}

// ---------------------------------------------------------------------------
// c. identity (bias None + act None) delegates to plain gemm_batched, bitwise
// ---------------------------------------------------------------------------

fn identity_delegates<T: BEl>() {
    let (batch, m, k, n) = (4usize, 24usize, 20usize, 18usize);
    let mut rng = Rng::new(0x1DE7_17FF);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let (alpha, beta) = (T::of(0.9), T::of(0.5));

    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        let mut c_fused = c0.clone();
        gemm_batched_fused(
            batch,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            beta,
            MatMut::new(&mut c_fused, m, n, 1, m as isize),
            (m * n) as isize,
            None,
            None,
            par,
        );

        let mut c_plain = c0.clone();
        gemm_batched(
            batch,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            beta,
            MatMut::new(&mut c_plain, m, n, 1, m as isize),
            (m * n) as isize,
            par,
        );

        for idx in 0..batch * m * n {
            assert_eq!(
                c_fused[idx].bits(),
                c_plain[idx].bits(),
                "{}: identity batched_fused != gemm_batched at {idx} (par={par:?})",
                T::name(),
            );
        }
    }
}

#[test]
fn batched_fused_identity_delegates() {
    identity_delegates::<f32>();
    identity_delegates::<f64>();
}

// ---------------------------------------------------------------------------
// d. serial == parallel, bitwise (each element is element-serial under either schedule)
// ---------------------------------------------------------------------------

fn parallel_bitwise<T: BEl>() {
    let (batch, m, k, n) = (8usize, 12usize, 24usize, 9usize);
    let mut rng = Rng::new(0x5E71_A11E);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0);
    let (alpha, beta) = (T::of(1.1), T::of(0.5));

    let run = |par: Parallelism| -> Vec<T> {
        let mut c = c0.clone();
        gemm_batched_fused(
            batch,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            beta,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            Some(Bias::PerRow(&bias_row)),
            Some(T::leaky(0.25)),
            par,
        );
        c
    };

    let c_ser = run(Parallelism::Serial);
    let c_par = run(Parallelism::Rayon(4));
    for idx in 0..batch * m * n {
        assert_eq!(
            c_ser[idx].bits(),
            c_par[idx].bits(),
            "{}: batched_fused serial != parallel at {idx}",
            T::name(),
        );
    }
}

#[test]
fn batched_fused_parallel_bitwise() {
    parallel_bitwise::<f32>();
    parallel_bitwise::<f64>();
}

// ---------------------------------------------------------------------------
// e. validation panics
// ---------------------------------------------------------------------------

mod validation {
    use super::*;

    /// `batch` contiguously-packed column-major f32 buffers `(a, b, c)` of one element each.
    fn base(batch: usize, m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![1.0f32; batch * m * k],
            vec![1.0f32; batch * k * n],
            vec![0.0f32; batch * m * n],
        )
    }

    #[test]
    #[should_panic(expected = "bias length")]
    fn bias_wrong_length() {
        let (batch, m, k, n) = (3usize, 4usize, 4usize, 4usize);
        let (a, b, mut c) = base(batch, m, k, n);
        let bias = vec![0.0f32; m - 1]; // PerRow must be length m
        gemm_batched_fused(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            Some(Bias::PerRow(&bias)),
            None,
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "bias slice overlaps C")]
    fn bias_overlaps_c() {
        let (batch, m, k, n) = (3usize, 4usize, 4usize, 4usize);
        let a = vec![1.0f32; batch * m * k];
        let b = vec![1.0f32; batch * k * n];
        let mut buf = vec![0.0f32; batch * m * n];
        // A correctly-sized (length m) PerRow bias aliasing C's backing slice. It is raw-derived
        // (its lifetime is not tied to `buf`), so `&mut buf` still type-checks; the overlap check
        // panics before any element is read or written, so no aliased access occurs.
        let bias: &[f32] = unsafe { core::slice::from_raw_parts(buf.as_ptr(), m) };
        gemm_batched_fused(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut buf, m, n, 1, m as isize),
            (m * n) as isize,
            Some(Bias::PerRow(bias)),
            None,
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "LeakyRelu slope must be finite")]
    fn leaky_slope_not_finite() {
        let (batch, m, k, n) = (3usize, 4usize, 4usize, 4usize);
        let (a, b, mut c) = base(batch, m, k, n);
        gemm_batched_fused(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            None,
            Some(Activation::LeakyRelu(f32::INFINITY)),
            Parallelism::Serial,
        );
    }

    /// The inherited batched disjointness check must still fire through the shared validation
    /// helper on the fused path (a valid bias so validation is actually reached).
    #[test]
    #[should_panic(expected = "stay disjoint")]
    fn c_batch_stride_below_extent() {
        let (batch, m, k, n) = (2usize, 4usize, 4usize, 4usize);
        let (a, b, mut c) = base(batch, m, k, n);
        let bias = vec![0.0f32; m];
        gemm_batched_fused(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n - 1) as isize, // < element extent m*n
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            Parallelism::Serial,
        );
    }
}

// ---------------------------------------------------------------------------
// f. batch == 0 is a no-op that neither touches C nor validates
// ---------------------------------------------------------------------------

/// `batch == 0` returns immediately — before the identity check, the view/bias validation, and any
/// element write — mirroring `gemm_batched_with`. A deliberately wrong-length bias must NOT panic,
/// and the sentinel-filled C must be untouched.
#[test]
fn batched_fused_batch_zero_noop() {
    let (m, k, n) = (4usize, 3usize, 2usize);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![7.0f32; m * n];
    let before = c.clone();
    let wrong_bias = vec![0.0f32; m + 3]; // wrong length on purpose — must not be validated
    gemm_batched_fused(
        0,
        1.0,
        MatRef::new(&a, m, k, 1, m as isize),
        (m * k) as isize,
        MatRef::new(&b, k, n, 1, k as isize),
        (k * n) as isize,
        0.0,
        MatMut::new(&mut c, m, n, 1, m as isize),
        (m * n) as isize,
        Some(Bias::PerRow(&wrong_bias)),
        Some(Activation::Relu),
        Parallelism::Rayon(4),
    );
    assert_eq!(
        c, before,
        "batch=0 fused must be a no-op that skips validation"
    );
}
