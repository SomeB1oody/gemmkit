//! Property-based tests that mutate the process-global tuning knobs. The only
//! knob-mutating binary: knob state is a per-process `AtomicUsize`, so a separate test
//! binary is a separate process and cannot race `tests/tuning.rs`/`tests/env.rs`.
//! Within this binary the harness runs tests concurrently, so every property holds
//! [`KNOB_LOCK`] for its whole body (mirrors tests/tuning.rs:16-22) and captures/restores
//! every knob it may touch via a per-case RAII [`KnobGuard`] that survives proptest's
//! internal `catch_unwind`. Properties never assert on absolute defaults — only on
//! outputs under knob settings they set themselves. See props_common for shared bars.
#![cfg(all(not(miri), not(target_family = "wasm")))]

mod props_common;

use gemmkit::{
    MatMut, MatRef, Parallelism, gemm, gemm_batched, gemm_packed_b, prepack_rhs, tuning,
};
use props_common::*;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// isolation: shared lock + per-case knob capture/restore
// ---------------------------------------------------------------------------

/// Serializes every test in this binary that mutates a knob (mirrors tests/tuning.rs:16-22).
/// Poison-recovery: a panicking case must not cascade into spurious failures.
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Captures the current value of every knob and restores it on `Drop` — including during
/// proptest's per-case unwind and shrink replays (it is a `Drop` impl). Getters resolve
/// env->cache->default and return the *raw* stored value (so restoring via the setter is a
/// faithful round-trip) for every knob except `thread_dim_stride`, whose getter maps raw `0`
/// to the core-derived auto value; that one is restored to its shipped default `0`
/// (THREAD_DIM_STRIDE_DEFAULT, tuning.rs:213) instead. The `.max(1)`-clamping getters
/// (parallel_oversample/kc/mc_reg_panels/packed_oversample/pack_transpose_tile) restore an
/// idempotently-clamped value — semantically identical.
struct KnobGuard {
    restore: Vec<(fn(usize), usize)>,
}
impl KnobGuard {
    fn capture() -> Self {
        #[allow(unused_mut)]
        let mut restore: Vec<(fn(usize), usize)> =
            KNOBS.iter().map(|&(set, get)| (set, get())).collect();
        // i8 VNNI stays a separate cfg'd append: it is captured/restored but deliberately
        // excluded from the swept `KNOBS` table (int8/f32-inert; exercised by P20).
        #[cfg(feature = "int8")]
        restore.push((
            tuning::set_i8_vnni_min_par_mnk,
            tuning::i8_vnni_min_par_mnk(),
        ));
        Self { restore }
    }
}
impl Drop for KnobGuard {
    fn drop(&mut self) {
        for &(set, v) in &self.restore {
            set(v);
        }
    }
}

/// `thread_dim_stride`'s getter maps a raw stored `0` to the core-derived auto value, so it
/// does not round-trip; its `KNOBS` capture entry restores the shipped default `0`
/// (THREAD_DIM_STRIDE_DEFAULT, tuning.rs:213) instead (see `KnobGuard`).
fn thread_dim_stride_restore() -> usize {
    0
}

/// One swept knob: its setter paired with the capture fn that reads the value the setter
/// round-trips to.
type Knob = (fn(usize), fn() -> usize);

/// The 21 general-path knobs P16 sweeps (i8 VNNI is int8/f32-inert, exercised by P20), each
/// paired with the capture fn that reads the value its setter round-trips to. Order-independent:
/// each is set to an independently-drawn value. Both `KnobGuard::capture` (restore side) and
/// `apply_knobs` (sweep side) drive this single table, so their lengths — and hence
/// [`KNOB_COUNT`] — can never drift apart.
const KNOBS: &[Knob] = &[
    (tuning::set_parallel_threshold, tuning::parallel_threshold),
    (tuning::set_rhs_pack_threshold, tuning::rhs_pack_threshold),
    (tuning::set_lhs_pack_threshold, tuning::lhs_pack_threshold),
    (tuning::set_lhs_pack_stride, tuning::lhs_pack_stride),
    (tuning::set_gemv_threshold, tuning::gemv_threshold),
    (tuning::set_small_k_threshold, tuning::small_k_threshold),
    (tuning::set_small_mn_dim, tuning::small_mn_dim),
    (tuning::set_gemv_parallel_bytes, tuning::gemv_parallel_bytes),
    (tuning::set_gemv_thread_cap, tuning::gemv_thread_cap),
    (tuning::set_parallel_oversample, tuning::parallel_oversample),
    (tuning::set_thread_dim_stride, thread_dim_stride_restore),
    (tuning::set_shared_lhs_mnk, tuning::shared_lhs_mnk),
    (tuning::set_k_stream_max, tuning::k_stream_max),
    (
        tuning::set_seq_internal_bytes_per_worker,
        tuning::seq_internal_bytes_per_worker,
    ),
    (tuning::set_packed_oversample, tuning::packed_oversample),
    (tuning::set_mc_reg_panels, tuning::mc_reg_panels),
    (tuning::set_nc_no_l3_panels, tuning::nc_no_l3_panels),
    (tuning::set_tiny_block_dim, tuning::tiny_block_dim),
    (tuning::set_kc, tuning::kc),
    (tuning::set_kc_min, tuning::kc_min),
    (tuning::set_pack_transpose_tile, tuning::pack_transpose_tile),
];
const KNOB_COUNT: usize = KNOBS.len();

fn apply_knobs(vals: &[usize]) {
    for (&(set, _), &v) in KNOBS.iter().zip(vals) {
        set(v);
    }
}

/// Value pool that lands on the small multipliers (1-3 forces deep multi-block driver
/// paths on tiny inputs) and the saturating-arithmetic extremes (`usize::MAX` re-covers
/// the `blocking()`/`m_iter*mr` saturating-mul class; tests/tuning.rs:43-59, 731-752).
fn knob_val() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => proptest::sample::select(&[0usize, 1, 2, 3, 5, 8][..]),
        2 => 1usize..=64,
        1 => proptest::sample::select(&[1usize << 20, 1usize << 40, usize::MAX][..]),
    ]
}

/// Run a plain f32 gemm into a fresh clone of `cbase`; return the output buffer.
#[allow(clippy::too_many_arguments)]
fn run_gemm(
    m: usize,
    k: usize,
    n: usize,
    abuf: &[f32],
    rsa: isize,
    csa: isize,
    bbuf: &[f32],
    rsb: isize,
    csb: isize,
    cbase: &[f32],
    rsc: isize,
    csc: isize,
    alpha: f32,
    beta: f32,
    par: Parallelism,
) -> Vec<f32> {
    let mut c = cbase.to_vec();
    gemm(
        alpha,
        MatRef::new(abuf, m, k, rsa, csa),
        MatRef::new(bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut c, m, n, rsc, csc),
        par,
    );
    c
}

// ---------------------------------------------------------------------------
// P16 — metamorphic knob sweep: any assignment keeps every path correct (frob vs
// ref64) and deterministic under a fixed config (run-twice BIT). No comparison across
// different assignments (knobs change blocking hence summation order).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(80), ..ProptestConfig::default() })]

    #[test]
    fn prop_knob_metamorphic(
        m in 1usize..=64, k in 1usize..=200, n in 1usize..=64,
        la in layout(), lb in layout(),
        al in coeff(), be in coeff(),
        knobs in prop::collection::vec(knob_val(), KNOB_COUNT..=KNOB_COUNT),
        seed in any::<u64>(),
    ) {
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();
        apply_knobs(&knobs);

        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, la);
        let (bbuf, rsb, csb) = build_view(&b, lb);
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);
        let cref = reference(&a, &b, &c0, al, be);

        // (i) plain gemm, serial and Rayon(0): determinism (run twice) + frob.
        for par in [Parallelism::Serial, Parallelism::Rayon(0)] {
            let c1 = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, par);
            let c2 = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, par);
            prop_assert!(bits_identical(&c1, &c2), "gemm not deterministic under fixed knobs par={:?}", par);
            assert_accurate(&c1, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), "knob gemm");
        }

        // (ii) prepack_rhs + gemm_packed_b (column-major-ish C): exercises the
        // tiny_block_dim().saturating_add(1) prepack sentinel under the same assignment.
        {
            let packed = prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
            let mut pk1 = cbase.clone();
            let mut pk2 = cbase.clone();
            gemm_packed_b(alpha, MatRef::new(&abuf, m, k, rsa, csa), &packed, beta,
                MatMut::new(&mut pk1, m, n, rsc, csc), Parallelism::Rayon(0));
            gemm_packed_b(alpha, MatRef::new(&abuf, m, k, rsa, csa), &packed, beta,
                MatMut::new(&mut pk2, m, n, rsc, csc), Parallelism::Rayon(0));
            prop_assert!(bits_identical(&pk1, &pk2), "gemm_packed_b not deterministic under fixed knobs");
            assert_accurate(&pk1, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), "knob packed_b");
        }

        // (iii) batched (3 contiguous col-major elements): each element frob + determinism.
        {
            let batch = 3usize;
            let (ea, eb, ec) = (m * k, k * n, m * n);
            let ba = rand_vec::<f32>(batch * ea, seed ^ 0x1);
            let bb = rand_vec::<f32>(batch * eb, seed ^ 0x2);
            let bc0 = rand_vec::<f32>(batch * ec, seed ^ 0x3);
            let run_batched = || {
                let mut c = bc0.clone();
                gemm_batched(
                    batch, alpha,
                    MatRef::new(&ba, m, k, 1, m as isize), ea as isize,
                    MatRef::new(&bb, k, n, 1, k as isize), eb as isize,
                    beta,
                    MatMut::new(&mut c, m, n, 1, m as isize), ec as isize,
                    Parallelism::Rayon(0),
                );
                c
            };
            let out1 = run_batched();
            let out2 = run_batched();
            prop_assert!(bits_identical(&out1, &out2), "gemm_batched not deterministic under fixed knobs");
            for bi in 0..batch {
                let ea_a = Mat { v: col_major_to_rowmajor(&ba[bi * ea..(bi + 1) * ea], m, k), rows: m, cols: k };
                let ea_b = Mat { v: col_major_to_rowmajor(&bb[bi * eb..(bi + 1) * eb], k, n), rows: k, cols: n };
                let ea_c0 = Mat { v: col_major_to_rowmajor(&bc0[bi * ec..(bi + 1) * ec], m, n), rows: m, cols: n };
                let eref = reference(&ea_a, &ea_b, &ea_c0, al, be);
                let elem = &out1[bi * ec..(bi + 1) * ec];
                assert_accurate(elem, 1, m as isize, m, n, &eref, &ea_a, &ea_b, k, be.abs() * frob_norm(&ea_c0), "knob batched");
            }
        }
    }
}

/// Transpose a column-major `rows×cols` slice into a row-major `Vec` (for the f64
/// reference, which reads row-major logical matrices).
fn col_major_to_rowmajor(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = v[j * rows + i];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// P16b — prepack under one knob assignment, consume under a *different* one: the
// prepack geometry is baked at pack time and consumed verbatim, so the result must
// still be correct (directly tests the sentinel/geometry decoupling).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_prepack_survives_knob_change(
        m in 1usize..=96, k in 1usize..=200, n in 1usize..=96,
        la in layout(), lb in layout(), al in coeff(), be in coeff(),
        knobs_x in prop::collection::vec(knob_val(), KNOB_COUNT..=KNOB_COUNT),
        knobs_y in prop::collection::vec(knob_val(), KNOB_COUNT..=KNOB_COUNT),
        seed in any::<u64>(),
    ) {
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();

        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, la);
        let (bbuf, rsb, csb) = build_view(&b, lb);
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);
        let cref = reference(&a, &b, &c0, al, be);

        apply_knobs(&knobs_x);
        let packed = prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb)); // geometry baked under X
        apply_knobs(&knobs_y); // mutate knobs AFTER packing

        let mut c1 = cbase.clone();
        let mut c2 = cbase.clone();
        gemm_packed_b(alpha, MatRef::new(&abuf, m, k, rsa, csa), &packed, beta,
            MatMut::new(&mut c1, m, n, rsc, csc), Parallelism::Rayon(0));
        gemm_packed_b(alpha, MatRef::new(&abuf, m, k, rsa, csa), &packed, beta,
            MatMut::new(&mut c2, m, n, rsc, csc), Parallelism::Rayon(0));
        prop_assert!(bits_identical(&c1, &c2), "packed consume not deterministic after knob change");
        assert_accurate(&c1, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), "prepack survives knob change");
    }
}

// ---------------------------------------------------------------------------
// P17 — gemv routing: dedicated gemv path (threshold on) vs the general driver
// (threshold 0) both stay accurate. Routes differ => tolerance, not bitwise.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_route_gemv(
        orient in any::<bool>(), other in 2usize..=200, k in kdim_pos(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(),
        p in par(), seed in any::<u64>(),
    ) {
        let (m, n) = if orient { (1usize, other) } else { (other, 1usize) };
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();

        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, la);
        let (bbuf, rsb, csb) = build_view(&b, lb);
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);
        let cref = reference(&a, &b, &c0, al, be);

        for thr in [usize::MAX - 1, 0usize] {
            tuning::set_gemv_threshold(thr);
            let c = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, p);
            assert_accurate(&c, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), &format!("gemv route thr={thr}"));
        }
    }
}

// ---------------------------------------------------------------------------
// P18 — small-m,n horizontal route (row-major A + col-major B, k > small_k) vs the
// driver, both accurate. small_k_threshold pinned to 16 in-guard so the k-gate is
// deterministic regardless of the arch-split default / ambient env.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_route_small_mn(
        m in 1usize..=16, n in 1usize..=16, k in 17usize..=200,
        al in coeff(), be in coeff(), p in par(), seed in any::<u64>(),
    ) {
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();
        tuning::set_small_k_threshold(16); // deterministic k-gate: every generated k (>=17) clears it

        // Row-major A (csa == 1) + column-major B (rsb == 1) + column-major C: the layout the
        // horizontal route requires (dispatch.rs:483-494); C col-major keeps orient_transpose
        // from swapping it away.
        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, PLayout::Row { pad: 0 });
        let (bbuf, rsb, csb) = build_view(&b, PLayout::Col { pad: 0 });
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);
        let cref = reference(&a, &b, &c0, al, be);

        for smn in [usize::MAX, 0usize] {
            tuning::set_small_mn_dim(smn);
            let c = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, p);
            assert_accurate(&c, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), &format!("small_mn route smn={smn}"));
        }
    }
}

// ---------------------------------------------------------------------------
// P19 — small-k route vs the driver, both accurate. small_mn_dim pinned to 0 in-guard
// so the small-mn gate can't flip mid-property as small_k_threshold is toggled.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_route_small_k(
        m in dim(), n in dim(), k in 1usize..=64,
        la in layout(), lb in layout(), al in coeff(), be in coeff(),
        p in par(), seed in any::<u64>(),
    ) {
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();
        tuning::set_small_mn_dim(0); // the small-mn route stays off so only the small-k gate moves

        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, la);
        let (bbuf, rsb, csb) = build_view(&b, lb);
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);
        let cref = reference(&a, &b, &c0, al, be);

        for skt in [usize::MAX, 0usize] {
            tuning::set_small_k_threshold(skt);
            let c = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, p);
            assert_accurate(&c, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), &format!("small_k route skt={skt}"));
        }
    }
}

// ---------------------------------------------------------------------------
// P20 — i8 VNNI gate (x86 + int8 only; the knob is inert elsewhere, tuning.rs:330-340):
// VNNI (gate 0) and widen (gate MAX) must both equal the wrapping-i32 reference exactly
// and be bit-identical to each other (the two kernels are documented bit-exact).
// ---------------------------------------------------------------------------

#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(32), ..ProptestConfig::default() })]

    #[test]
    fn prop_i8_vnni_gate(
        m in 1usize..=160, k in 1usize..=160, n in 1usize..=160, seed in any::<u64>(),
    ) {
        let _lock = knob_guard();
        let _restore = KnobGuard::capture();

        let a = fill_i8(m * k, seed);
        let b = fill_i8(k * n, seed ^ 0xB);
        let c0 = vec![0i32; m * n];
        let cref = ref_i8_wrapping(&a, &b, &c0, m, k, n, 1, 0);

        // Column-major operands (like tests/tuning.rs:906-946).
        let acol = {
            let mut v = vec![0i8; m * k];
            for i in 0..m { for p in 0..k { v[p * m + i] = a[i * k + p]; } }
            v
        };
        let bcol = {
            let mut v = vec![0i8; k * n];
            for p in 0..k { for j in 0..n { v[j * k + p] = b[p * n + j]; } }
            v
        };
        let run = |gate: usize| {
            tuning::set_i8_vnni_min_par_mnk(gate);
            let mut c = vec![0i32; m * n];
            gemmkit::gemm_i8(
                1,
                MatRef::from_col_major(&acol, m, k),
                MatRef::from_col_major(&bcol, k, n),
                0,
                MatMut::from_col_major(&mut c, m, n),
                Parallelism::Rayon(0),
            );
            c
        };
        let vnni = run(0);
        let widen = run(usize::MAX);
        // Column-major output vs a row-major reference: read at [j*m + i].
        for i in 0..m {
            for j in 0..n {
                prop_assert_eq!(vnni[j * m + i], cref[i * n + j], "VNNI wrong at ({},{})", i, j);
                prop_assert_eq!(widen[j * m + i], cref[i * n + j], "widen wrong at ({},{})", i, j);
            }
        }
        prop_assert_eq!(&vnni, &widen, "VNNI and widen kernels must be bit-identical");
    }
}

// ---------------------------------------------------------------------------
// P21 — mixed (f16) small-m,n horizontal route, parallel tail: the mixed route only ever ran
// serial in the coverage runs. Pinning the gemv/bandwidth floor to 1 lets a small `k` still fork
// under Rayon(2); the route reduces each output within one worker, so serial and parallel must be
// bit-identical. beta == 1 also exercises the epilogue's accumulate arm.
// ---------------------------------------------------------------------------

#[cfg(feature = "half")]
#[test]
fn prop_small_mn_mixed_parallel_bit_identical() {
    use gemmkit::f16;

    let _lock = knob_guard();
    let _restore = KnobGuard::capture();
    // Route to the mixed small_mn path and force its bandwidth-capped worker count above 1:
    // small_mn_dim >= m,n; small_k_threshold < k; and a byte floor of 1 clears resolve_bandwidth.
    tuning::set_small_mn_dim(16);
    tuning::set_small_k_threshold(16);
    tuning::set_gemv_parallel_bytes(1);

    let (m, n, k) = (16usize, 16usize, 64usize);
    // Row-major A (csa == 1) + column-major B (rsb == 1) + column-major C — the layout the
    // horizontal route requires (dispatch.rs mixed gate), and col-major C keeps orient_transpose
    // from swapping it away.
    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32((i % 23) as f32 * 0.03 - 0.3))
        .collect();
    let b: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32((i % 19) as f32 * 0.05 - 0.4))
        .collect();
    let c0: Vec<f16> = (0..m * n)
        .map(|i| f16::from_f32((i % 11) as f32 * 0.1 - 0.5))
        .collect();
    let (alpha, beta) = (f16::from_f32(1.25), f16::from_f32(1.0)); // beta == 1 hits the accumulate arm

    let run = |par: Parallelism| -> Vec<u16> {
        let mut c = c0.clone();
        gemm(
            alpha,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            beta,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
        c.iter().map(|v| v.to_bits()).collect()
    };

    let serial = run(Parallelism::Serial);
    let parallel = run(Parallelism::Rayon(2));
    assert_eq!(
        serial, parallel,
        "mixed small_mn parallel must be bit-identical to serial"
    );
}
