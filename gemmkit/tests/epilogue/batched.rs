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

/// Serializes the two threshold-mutating tests in this module, mirroring `tests/tuning.rs`'s
/// `KNOB_LOCK`. `tuning::parallel_threshold` is process-global and the harness runs tests in this
/// binary concurrently; without a shared lock one test's set/restore could interleave with another
/// test's GEMM and flip a route. The plan-coverage tests here hold this via [`KnobGuard`] for their
/// whole body and restore the prior value on drop, so no mutation is observed outside them. (The
/// other epilogue tests do not take the lock: every contract in this suite is bitwise-invariant to
/// scheduling — fused == gemm-then-map and serial == parallel both hold under any plan — so a
/// transiently-lowered threshold cannot perturb their assertions.)
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds [`KNOB_LOCK`] and forces `tuning::parallel_threshold = override_to` for the calling test's
/// duration, saving the prior value (read from the getter) and restoring it on drop. Recovers a
/// poisoned lock so one panicking test does not cascade. The `_lock` field keeps the guard alive for
/// the whole test body.
struct KnobGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: usize,
}
impl KnobGuard {
    fn with_parallel_threshold(override_to: usize) -> Self {
        let lock = KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = gemmkit::tuning::parallel_threshold();
        gemmkit::tuning::set_parallel_threshold(override_to);
        KnobGuard { _lock: lock, saved }
    }
}
impl Drop for KnobGuard {
    fn drop(&mut self) {
        gemmkit::tuning::set_parallel_threshold(self.saved);
    }
}

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
// a'. checked/unchecked twin equivalence: gemm_batched_fused_unchecked(_with)
// ---------------------------------------------------------------------------

/// `gemm_batched_fused` and its raw twins `gemm_batched_fused_unchecked` (pool) / `_with`
/// (caller-owned `Workspace`) are **parallel** entry points — the checked one does not delegate to
/// the raw one, so a divergence in the raw `(ptr, strides, batch strides, bias)` lowering would go
/// undetected. Exercise both raw forms against the checked twin bit-for-bit (PerRow bias +
/// LeakyReLU, contiguously-packed elements).
fn unchecked_matches_checked<T: BEl>() {
    use gemmkit::{BiasDim, gemm_batched_fused_unchecked, gemm_batched_fused_unchecked_with};

    let (batch, m, k, n) = (5usize, 31usize, 17usize, 23usize);
    let mut rng = Rng::new(0xBA7C_0FEE);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // ONE shared per-row bias (length m)
    let alpha = T::of(0.9);
    let beta = T::of(0.7);
    let (a_bs, b_bs, c_bs) = ((m * k) as isize, (k * n) as isize, (m * n) as isize);
    let par = Parallelism::Serial;

    let mut c_checked = c0.clone();
    gemm_batched_fused(
        batch,
        alpha,
        MatRef::new(&a, m, k, 1, m as isize),
        a_bs,
        MatRef::new(&b, k, n, 1, k as isize),
        b_bs,
        beta,
        MatMut::new(&mut c_checked, m, n, 1, m as isize),
        c_bs,
        Some(Bias::PerRow(&bias_row)),
        Some(T::leaky(0.25)),
        par,
    );

    let mut c_pool = c0.clone();
    let mut c_ws = c0.clone();
    let mut ws = Workspace::new();
    // SAFETY: valid in-bounds contiguously-packed col-major elements; the batch C regions are
    // pairwise disjoint and alias neither A/B nor the (per-row, length-m) shared bias.
    unsafe {
        gemm_batched_fused_unchecked(
            batch,
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            1,
            m as isize,
            a_bs,
            b.as_ptr(),
            1,
            k as isize,
            b_bs,
            beta,
            c_pool.as_mut_ptr(),
            1,
            m as isize,
            c_bs,
            bias_row.as_ptr(),
            BiasDim::PerRow,
            true,
            Some(T::leaky(0.25)),
            par,
        );
        gemm_batched_fused_unchecked_with(
            &mut ws,
            batch,
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            1,
            m as isize,
            a_bs,
            b.as_ptr(),
            1,
            k as isize,
            b_bs,
            beta,
            c_ws.as_mut_ptr(),
            1,
            m as isize,
            c_bs,
            bias_row.as_ptr(),
            BiasDim::PerRow,
            true,
            Some(T::leaky(0.25)),
            par,
        );
    }

    for idx in 0..batch * m * n {
        assert_eq!(
            c_checked[idx].bits(),
            c_pool[idx].bits(),
            "{}: batched fused unchecked != checked at {idx}",
            T::name(),
        );
        assert_eq!(
            c_checked[idx].bits(),
            c_ws[idx].bits(),
            "{}: batched fused unchecked_with != checked at {idx}",
            T::name(),
        );
    }
}

#[test]
fn batched_fused_unchecked_matches_checked() {
    unchecked_matches_checked::<f32>();
    unchecked_matches_checked::<f64>();
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

/// Serial ≡ Rayon(4), bitwise. These tiny shapes sit far below the default `parallel_threshold`, so
/// resolve_batch returns `BatchPlan::Serial` for both calls — this covers the **Serial-resolved
/// small-work regime**, not the parallel arms (those are `batched_fused_batch_parallel_bitwise` /
/// `batched_fused_seq_internal_bitwise`, which lower the threshold under a knob guard).
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
// d2. BatchParallel arm: enough elements to fill the workers ⇒ BatchPlan::BatchParallel
// ---------------------------------------------------------------------------

fn batch_parallel_bitwise<T: BEl>() {
    // Under `parallel_threshold = 1` the cheap total-work gate passes for any non-empty batch, so
    // resolve_batch reaches the core-count branch instead of the small-work Serial short-circuit.
    // Plan derivation (feature = "parallel", Rayon(4)): elem_mnk·batch = (13·9·11)·8 = 10296 ≥ 1
    // (gate passes); budget = min(4, auto_threads); batch = 8 ≥ budget ⇒ BatchPlan::BatchParallel(
    // budget). On any host with ≥ 2 usable cores this exercises the BatchParallel arm (its whole
    // point); with only one usable core budget ≤ 1 short-circuits to Serial. Either way the two
    // assertions below are plan-independent: BatchParallel runs every element serially on one worker,
    // so the batch is bit-identical to a Serial loop of gemm_fused and to the Serial batched call. A
    // future policy change that broke this routing is caught by re-deriving the plan against this
    // comment.
    let _guard = KnobGuard::with_parallel_threshold(1);

    let (batch, m, k, n) = (8usize, 13usize, 9usize, 11usize);
    let mut rng = Rng::new(0xB47C_9A2E);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0); // col-major, contiguously packed
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // ONE shared per-row bias
    let (alpha, beta) = (T::of(1.1), T::of(0.7));

    // Reference: a loop of Serial gemm_fused, one per element on its offset window.
    let mut c_loop = c0.clone();
    let mut ws = Workspace::new();
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a[ao..ao + m * k], m, k, 1, m as isize),
            MatRef::new(&b[bo..bo + k * n], k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_loop[co..co + m * n], m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            Some(T::leaky(0.25)),
            Parallelism::Serial,
        );
    }

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

    let c_par = run(Parallelism::Rayon(4)); // BatchParallel(budget) on ≥ 2 cores
    let c_ser = run(Parallelism::Serial); // BatchPlan::Serial
    for idx in 0..batch * m * n {
        assert_eq!(
            c_par[idx].bits(),
            c_loop[idx].bits(),
            "{}: batch-parallel batched_fused != gemm_fused loop at {idx}",
            T::name(),
        );
        assert_eq!(
            c_par[idx].bits(),
            c_ser[idx].bits(),
            "{}: batch-parallel batched_fused != serial batched_fused at {idx}",
            T::name(),
        );
    }
}

#[test]
fn batched_fused_batch_parallel_bitwise() {
    batch_parallel_bitwise::<f32>();
    batch_parallel_bitwise::<f64>();
}

// ---------------------------------------------------------------------------
// d3. SequentialInternal arm (best-effort, platform-dependent coverage): few but large,
//     L2-spilling elements split each element across the machine in turn
// ---------------------------------------------------------------------------

fn seq_internal_bitwise() {
    // Under `parallel_threshold = 1` the work gate passes. Plan derivation (feature = "parallel",
    // Rayon(4)): budget = min(4, auto_threads); batch = 2. With ≥ 3 usable cores batch < budget, so
    // resolve_batch reaches the residency split test. On x86 (private per-core L2) it splits when the
    // element spills that L2: elem_bytes = (mk + kn + mn)·8 = 3·256·256·8 = 1.5 MiB > this Zen5's
    // 1 MiB effective L2 ⇒ BatchPlan::SequentialInternal (each element gets the full engine
    // parallelism in turn). On a host with a larger L2, on aarch64's share-based rule, or with < 3
    // usable cores, the same shape may resolve BatchParallel instead — that changes only which arm is
    // *covered*, never the result. SequentialInternal splits each element across workers, relying on
    // that element's route being serial==parallel bit-identical, so the batch equals a Serial
    // gemm_fused loop under either plan. The assertion is therefore plan-independent; only the
    // coverage is platform-dependent.
    let _guard = KnobGuard::with_parallel_threshold(1);

    let (batch, m, k, n) = (2usize, 256usize, 256usize, 256usize); // 1.5 MiB f64 element
    let mut rng = Rng::new(0x5E90_11A7);
    let a = make_vec::<f64>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<f64>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<f64>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<f64>(&mut rng, m, 2.0);
    let (alpha, beta) = (0.9f64, 0.5f64);

    let mut c_loop = c0.clone();
    let mut ws = Workspace::new();
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a[ao..ao + m * k], m, k, 1, m as isize),
            MatRef::new(&b[bo..bo + k * n], k, n, 1, k as isize),
            beta,
            MatMut::new(&mut c_loop[co..co + m * n], m, n, 1, m as isize),
            Some(Bias::PerRow(&bias_row)),
            Some(<f64 as BEl>::leaky(0.25)),
            Parallelism::Serial,
        );
    }

    let run = |par: Parallelism| -> Vec<f64> {
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
            Some(<f64 as BEl>::leaky(0.25)),
            par,
        );
        c
    };

    let c_par = run(Parallelism::Rayon(4)); // SequentialInternal on this Zen5
    let c_ser = run(Parallelism::Serial); // BatchPlan::Serial
    for idx in 0..batch * m * n {
        assert_eq!(
            c_par[idx].bits(),
            c_loop[idx].bits(),
            "f64: seq-internal batched_fused != gemm_fused loop at {idx}",
        );
        assert_eq!(
            c_par[idx].bits(),
            c_ser[idx].bits(),
            "f64: seq-internal batched_fused != serial batched_fused at {idx}",
        );
    }
}

#[test]
fn batched_fused_seq_internal_bitwise() {
    seq_internal_bitwise();
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
