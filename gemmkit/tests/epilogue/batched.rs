//! Strided-batched fused-epilogue tests: `gemm_batched_fused` / `gemm_batched_fused_with`
//!
//! The headline contract is that the batched-fused call is **bit-identical to a loop of
//! `gemm_fused` calls**, one per element, sharing the single bias vector and single activation.
//! That holds for every element type (`f32`/`f64` and, under `half`, `f16`/`bf16`), since each
//! element re-dispatches through the same fused engine `gemm_fused` uses. Comparisons are
//! **bitwise** (raw bit patterns). Fills are the deterministic, platform-independent
//! [`crate::common::Rng`]; every reference is self-computed

use crate::common::Rng;
use gemmkit::{
    Activation, Bias, FusedScalar, MatMut, MatRef, Parallelism, Workspace, gemm_batched,
    gemm_batched_fused, gemm_fused_with,
};

/// Serializes the 2 threshold-mutating tests in this module ([`batch_parallel_bitwise`] and
/// [`seq_internal_bitwise`]), mirroring `tests/tuning.rs`'s own `KNOB_LOCK`. `tuning::
/// parallel_threshold` is process-global and the harness runs tests in this binary concurrently,
/// so without a shared lock one test's set/restore could interleave with another test's GEMM and
/// flip its route. Those 2 tests hold this lock via [`KnobGuard`] for their whole body and restore
/// the prior value on drop, so no mutation is observed outside them. Every other test in this
/// module skips the lock: their contracts (fused == gemm-then-map, serial == parallel) hold under
/// any plan, so a transiently-lowered threshold elsewhere cannot perturb their assertions
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds [`KNOB_LOCK`] and forces `tuning::parallel_threshold = override_to` for the calling
/// test's duration, saving the prior value (read from the getter) and restoring it on drop.
/// Recovers a poisoned lock so one panicking test does not cascade. The `_lock` field only needs
/// to keep the guard alive for the whole test body; nothing reads it directly
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

/// The batched element under test: `f32`/`f64` and, under `half`, `f16`/`bf16`. `of`/`bits` give
/// deterministic construction and a bitwise compare; `leaky` builds a `LeakyRelu` directly in `T`
/// (`Activation` is not `Clone`, so each call needs its own fresh value rather than a clone)
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

/// A length-`len` column-major buffer of RNG values scaled by `scale`
fn make_vec<T: BEl>(rng: &mut Rng, len: usize, scale: f64) -> Vec<T> {
    (0..len).map(|_| T::of(rng.unit() * scale)).collect()
}

// batched_fused == a loop of gemm_fused calls, bitwise (PerRow bias plus LeakyReLU, beta != 0)

fn matches_loop<T: BEl>() {
    let (batch, m, k, n) = (5usize, 31usize, 17usize, 23usize);
    let mut rng = Rng::new(0xBA7C_11FE);
    // batch contiguously-packed column-major elements: element bi starts at offset bi * m * k etc
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // 1 bias shared across the whole batch
    let alpha = T::of(0.9);
    let beta = T::of(0.7);

    // reference: 1 Serial gemm_fused call per element, on its own offset slice window
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

    // total work (elem_mnk * batch) is far below parallel_threshold, so resolve_batch returns
    // BatchPlan::Serial for both par values below: the batched call is a plain Serial loop either
    // way, matching c_ref exactly
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

// checked/unchecked twin equivalence: gemm_batched_fused_unchecked(_with)

/// `gemm_batched_fused` and its raw twins `gemm_batched_fused_unchecked` (pool) / `_with`
/// (caller-owned `Workspace`) are **parallel** entry points (the checked one does not call either
/// raw one internally), so a divergence in the raw `(ptr, strides, batch strides, bias)` lowering
/// would go undetected. Drives both raw forms against the checked twin bit-for-bit (PerRow bias
/// plus LeakyReLU, contiguously-packed elements)
fn unchecked_matches_checked<T: BEl>() {
    use gemmkit::{BiasDim, gemm_batched_fused_unchecked, gemm_batched_fused_unchecked_with};

    let (batch, m, k, n) = (5usize, 31usize, 17usize, 23usize);
    let mut rng = Rng::new(0xBA7C_0FEE);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0);
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // 1 bias shared across the whole batch
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
    // SAFETY: valid in-bounds contiguously-packed col-major elements; the batch's C regions are
    // pairwise disjoint and alias neither A nor B nor the per-row, length-m shared bias
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

// broadcast A (a_batch_stride == 0): PerCol bias plus ReLU, bitwise vs the gemm_fused loop

fn broadcast_a<T: BEl>() {
    let (batch, m, k, n) = (6usize, 20usize, 12usize, 14usize);
    let mut rng = Rng::new(0x0AD_C0DE);
    let a = make_vec::<T>(&mut rng, m * k, 1.0); // 1 A, reused for every batch element
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_col = make_vec::<T>(&mut rng, n, 2.0); // 1 bias shared across the whole batch
    let alpha = T::of(1.0);
    let beta = T::of(0.5);

    let mut c_ref = c0.clone();
    let mut ws = Workspace::new();
    for bi in 0..batch {
        let (bo, co) = (bi * k * n, bi * m * n);
        gemm_fused_with(
            &mut ws,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize), // the loop reads the same A every iteration
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
        0, // a_batch_stride == 0: reuse the same A for every batch element
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

// identity (bias None, act None) delegates to plain gemm_batched, bitwise

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

// serial == parallel, bitwise, in the Serial-resolved small-work regime

/// Serial == Rayon(4), bitwise. `batch * elem_mnk` here sits far below the default
/// `parallel_threshold`, so `resolve_batch` returns `BatchPlan::Serial` for both calls: this
/// covers the small-work regime specifically, not the parallel arms (those are
/// [`batched_fused_batch_parallel_bitwise`] / [`batched_fused_seq_internal_bitwise`], which lower
/// the threshold under a knob guard)
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

// the BatchParallel arm: enough elements to fill the workers, so resolve_batch picks it

fn batch_parallel_bitwise<T: BEl>() {
    // With parallel_threshold lowered to 1, the cheap total-work gate (elem_mnk * batch =
    // (13*9*11)*8 = 10296 >= 1) passes, so resolve_batch reaches the core-count branch instead of
    // its small-work Serial short-circuit. With Rayon(4), budget = min(4, auto_threads()); batch
    // == 8 >= budget, so it picks BatchPlan::BatchParallel(budget) whenever budget > 1, which
    // needs at least 2 usable cores (otherwise budget <= 1 and resolve_batch falls back to Serial
    // itself). Either way the assertions below hold: BatchParallel still runs each element
    // serially on 1 worker, so the batch stays bit-identical to a Serial gemm_fused loop and to
    // the Serial batched call regardless of which arm actually ran
    let _guard = KnobGuard::with_parallel_threshold(1);

    let (batch, m, k, n) = (8usize, 13usize, 9usize, 11usize);
    let mut rng = Rng::new(0xB47C_9A2E);
    let a = make_vec::<T>(&mut rng, batch * m * k, 1.0); // col-major, contiguously packed
    let b = make_vec::<T>(&mut rng, batch * k * n, 1.0);
    let c0 = make_vec::<T>(&mut rng, batch * m * n, 1.0);
    let bias_row = make_vec::<T>(&mut rng, m, 2.0); // 1 bias shared across the whole batch
    let (alpha, beta) = (T::of(1.1), T::of(0.7));

    // reference: a loop of Serial gemm_fused, 1 call per element on its offset window
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

    let c_par = run(Parallelism::Rayon(4)); // BatchPlan::BatchParallel(budget) on >= 2 cores
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

// the SequentialInternal arm (best-effort, platform-dependent coverage): few but large,
// L2-spilling elements each get split across the machine in turn

fn seq_internal_bitwise() {
    // With parallel_threshold lowered to 1 the work gate passes. With Rayon(4), budget =
    // min(4, auto_threads()); batch == 2, so on a host with >= 3 usable cores budget > batch and
    // resolve_batch reaches the residency split test rather than picking BatchParallel(batch)
    // outright. On x86 (private per-core L2) that test splits once an element spills its L2 share:
    // elem_bytes = (m*k + k*n + m*n)*8 = 3*256*256*8 = 1.5 MiB, sized to exceed the effective L2
    // this test was tuned against, giving BatchPlan::SequentialInternal (each element gets the
    // full engine parallelism in turn). On a host with a bigger L2 share, on aarch64's separate
    // share-based rule, or with < 3 usable cores, the same shape may resolve BatchParallel
    // instead; that only changes which arm gets *covered*, never the assertions below
    // SequentialInternal splits each element across workers, relying on that element's route
    // being serial == parallel bit-identical, so the batch equals the Serial gemm_fused loop
    // under either plan
    let _guard = KnobGuard::with_parallel_threshold(1);

    let (batch, m, k, n) = (2usize, 256usize, 256usize, 256usize); // each element is 1.5 MiB (f64)
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

    let c_par = run(Parallelism::Rayon(4)); // BatchPlan::SequentialInternal, on this test's target
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

// validation panics

mod validation {
    use super::*;

    /// `batch` contiguously-packed column-major f32 elements for A, B, and C, each filled with a
    /// constant (A/B ones, C zeros), the fixture the panic cases below start from
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
        let bias = vec![0.0f32; m - 1]; // PerRow needs length m, not m - 1
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
        // A correctly-sized (length m) PerRow bias built from a raw pointer into buf, so it has
        // no lifetime tied to `buf` and `&mut buf` below still type-checks despite the aliasing;
        // the overlap check panics before any element is read or written
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

    /// The plain-`gemm_batched` disjointness check (the batch's C regions must not overlap) fires
    /// through the fused path's shared validation too; the bias is valid-length so this test
    /// actually reaches that check rather than the bias-length one
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
            (m * n - 1) as isize, // shorter than the m * n element extent: batches would overlap
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            Parallelism::Serial,
        );
    }
}

// batch == 0 is a no-op that neither touches C nor validates

/// `batch == 0` returns immediately, before the identity check, the view/bias validation, or any
/// element write, mirroring plain `gemm_batched_with`'s own `batch == 0` short-circuit. A
/// deliberately wrong-length bias must NOT panic, and the sentinel-filled C must be untouched
#[test]
fn batched_fused_batch_zero_noop() {
    let (m, k, n) = (4usize, 3usize, 2usize);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![7.0f32; m * n];
    let before = c.clone();
    let wrong_bias = vec![0.0f32; m + 3]; // wrong on purpose: must never reach validation
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
