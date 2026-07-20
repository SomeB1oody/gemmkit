//! Property tests over gemmkit's tuning knobs (`tuning::set_*`): every knob assignment
//! must still produce a correct, deterministic result, and every route a knob can steer a
//! problem onto (packed, gemv, small-mn, small-k, i8 VNNI) must agree with the general
//! driver regardless of which knob toggled it. Each knob is a process-global `AtomicUsize`,
//! so mutating one from a different test binary (a separate process) cannot interfere with
//! this one - but the harness runs the properties inside this binary concurrently, and they
//! all touch the same globals. [`KNOB_LOCK`] serializes every property here, and a per-case
//! [`KnobGuard`] captures and restores every knob so a shrink replay or an early panic never
//! leaks a mutated value into the next case. See props_common for the shared strategies,
//! oracle, and accuracy gates
#![cfg(all(not(miri), not(target_family = "wasm")))]

// Shared proptest strategies, oracle references, and accuracy gates
mod props_common;

use gemmkit::{
    MatMut, MatRef, Parallelism, gemm, gemm_batched, gemm_packed_b, prepack_rhs, tuning,
};
use props_common::*;
use proptest::prelude::*;

// Shared lock plus per-case knob capture/restore

/// Serializes every property in this binary that mutates a knob (the same pattern
/// tests/tuning.rs uses for its own knob-touching tests), so no 2 cases can interleave their
/// `set_*` calls. Recovers a poisoned lock so one panicking case cannot cascade into
/// spurious failures elsewhere
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Captures the current value of every knob and restores it on `Drop`, including through
/// proptest's per-case unwind and shrink replays (restoring in `Drop` is what makes that
/// safe). Each knob's getter resolves env -> cache -> default and returns the raw stored
/// value, so feeding it back through the matching setter is a faithful round-trip.
/// The knobs whose getters clamp their result to `.max(1)` (`parallel_oversample`,
/// `kc`, `mc_reg_panels`, `packed_oversample`, `pack_transpose_tile`) restore an
/// already-clamped value, which is semantically identical to the original
struct KnobGuard {
    restore: Vec<(fn(usize), usize)>,
}
impl KnobGuard {
    fn capture() -> Self {
        #[allow(unused_mut)]
        let mut restore: Vec<(fn(usize), usize)> =
            KNOBS.iter().map(|&(_, set, get)| (set, get())).collect();
        // Captured/restored like every other knob, but kept out of the swept KNOBS table
        // (int8-only; exercised separately by P20)
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

/// One swept knob: its canonical `GEMMKIT_*` env name, its setter, and the getter that reads
/// back the value the setter round-trips to. The name backs `knobs_table_covers_every_knob`
/// against `tuning::knob_env_names`; the 2 fns drive the sweep and the restore
type Knob = (&'static str, fn(usize), fn() -> usize);

/// The 24 general-path knobs this suite sweeps: every entry in `tuning::knob_env_names`
/// except i8 VNNI, which is int8/f32-inert and exercised separately by P20. Order-independent:
/// each entry is set to an independently-drawn value per case. Both `KnobGuard::capture`
/// (restore side) and `apply_knobs` (sweep side) iterate this one table, so their lengths -
/// and hence [`KNOB_COUNT`] - can never drift apart. `knobs_table_covers_every_knob` asserts
/// the leading env names cover [`tuning::knob_env_names`] exactly
const KNOBS: &[Knob] = &[
    (
        "GEMMKIT_PARALLEL_THRESHOLD",
        tuning::set_parallel_threshold,
        tuning::parallel_threshold,
    ),
    (
        "GEMMKIT_RHS_PACK_THRESHOLD",
        tuning::set_rhs_pack_threshold,
        tuning::rhs_pack_threshold,
    ),
    (
        "GEMMKIT_LHS_PACK_THRESHOLD",
        tuning::set_lhs_pack_threshold,
        tuning::lhs_pack_threshold,
    ),
    (
        "GEMMKIT_LHS_PACK_STRIDE",
        tuning::set_lhs_pack_stride,
        tuning::lhs_pack_stride,
    ),
    (
        "GEMMKIT_LHS_PACK_SPAN",
        tuning::set_lhs_pack_span,
        tuning::lhs_pack_span,
    ),
    (
        "GEMMKIT_GEMV_THRESHOLD",
        tuning::set_gemv_threshold,
        tuning::gemv_threshold,
    ),
    (
        "GEMMKIT_SMALL_K_THRESHOLD",
        tuning::set_small_k_threshold,
        tuning::small_k_threshold,
    ),
    (
        "GEMMKIT_SMALL_MN_DIM",
        tuning::set_small_mn_dim,
        tuning::small_mn_dim,
    ),
    (
        "GEMMKIT_SMALL_MN_PACK_MIN_K",
        tuning::set_small_mn_pack_min_k,
        tuning::small_mn_pack_min_k,
    ),
    (
        "GEMMKIT_GEMV_PARALLEL_BYTES",
        tuning::set_gemv_parallel_bytes,
        tuning::gemv_parallel_bytes,
    ),
    (
        "GEMMKIT_GEMV_THREAD_CAP",
        tuning::set_gemv_thread_cap,
        tuning::gemv_thread_cap,
    ),
    (
        "GEMMKIT_PARALLEL_OVERSAMPLE",
        tuning::set_parallel_oversample,
        tuning::parallel_oversample,
    ),
    (
        "GEMMKIT_PAR_MNK_PER_WORKER",
        tuning::set_par_mnk_per_worker,
        tuning::par_mnk_per_worker,
    ),
    (
        "GEMMKIT_SHARED_LHS_MNK",
        tuning::set_shared_lhs_mnk,
        tuning::shared_lhs_mnk,
    ),
    (
        "GEMMKIT_K_STREAM_MAX",
        tuning::set_k_stream_max,
        tuning::k_stream_max,
    ),
    (
        "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
        tuning::set_seq_internal_bytes_per_worker,
        tuning::seq_internal_bytes_per_worker,
    ),
    (
        "GEMMKIT_PACKED_OVERSAMPLE",
        tuning::set_packed_oversample,
        tuning::packed_oversample,
    ),
    (
        "GEMMKIT_MC_REG_PANELS",
        tuning::set_mc_reg_panels,
        tuning::mc_reg_panels,
    ),
    (
        "GEMMKIT_NC_NO_L3_PANELS",
        tuning::set_nc_no_l3_panels,
        tuning::nc_no_l3_panels,
    ),
    (
        "GEMMKIT_TINY_BLOCK_DIM",
        tuning::set_tiny_block_dim,
        tuning::tiny_block_dim,
    ),
    ("GEMMKIT_KC", tuning::set_kc, tuning::kc),
    ("GEMMKIT_KC_MIN", tuning::set_kc_min, tuning::kc_min),
    (
        "GEMMKIT_PACK_TRANSPOSE_TILE",
        tuning::set_pack_transpose_tile,
        tuning::pack_transpose_tile,
    ),
    (
        "GEMMKIT_DEEP_KC_BYTES",
        tuning::set_deep_kc_bytes,
        tuning::deep_kc_bytes,
    ),
];
const KNOB_COUNT: usize = KNOBS.len();

/// Set every knob in [`KNOBS`] to its paired value from `vals`
fn apply_knobs(vals: &[usize]) {
    for (&(_, set, _), &v) in KNOBS.iter().zip(vals) {
        set(v);
    }
}

// KNOBS (plus the cfg-gated i8 VNNI knob it deliberately excludes) must exactly cover
// tuning::knob_env_names, the crate's canonical registry: catches a knob added upstream
// that never got mirrored here, which would silently lose its coverage
#[test]
fn knobs_table_covers_every_knob() {
    use std::collections::BTreeSet;
    let canonical: BTreeSet<&str> = tuning::knob_env_names().iter().copied().collect();
    let mut swept: BTreeSet<&str> = KNOBS.iter().map(|&(name, _, _)| name).collect();
    assert_eq!(swept.len(), KNOBS.len(), "KNOBS has a duplicate env name");
    // The one knob legitimately absent from KNOBS: captured/restored by KnobGuard but swept
    // separately by P20, so add it back before comparing against the canonical set
    if cfg!(feature = "int8") {
        swept.insert("GEMMKIT_I8_VNNI_MIN_PAR_MNK");
    }
    assert_eq!(
        swept, canonical,
        "props_knobs KNOBS is out of sync with tuning::knob_env_names()"
    );
}

/// Value pool weighted toward small multipliers (1-3, which force deep multi-block driver
/// paths even on tiny inputs) plus a few huge extremes (`usize::MAX` among them, to hit the
/// saturating-arithmetic branches a knob-derived multiply can take)
fn knob_val() -> impl Strategy<Value = usize> {
    prop_oneof![
        3 => proptest::sample::select(&[0usize, 1, 2, 3, 5, 8][..]),
        2 => 1usize..=64,
        1 => proptest::sample::select(&[1usize << 20, 1usize << 40, usize::MAX][..]),
    ]
}

/// Run a plain f32 gemm into a fresh clone of `cbase`; return the output buffer
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

// P16 metamorphic knob sweep: any single assignment of all 23 knobs keeps every path
// correct (frob vs the f64 reference) and deterministic under a fixed config (run twice,
// compare bit-for-bit). No comparison across 2 different assignments: different knobs
// change the blocking, hence the summation order

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

        // (i) plain gemm, serial and Rayon(0): determinism (run twice) + frob accuracy
        for par in [Parallelism::Serial, Parallelism::Rayon(0)] {
            let c1 = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, par);
            let c2 = run_gemm(m, k, n, &abuf, rsa, csa, &bbuf, rsb, csb, &cbase, rsc, csc, alpha, beta, par);
            prop_assert!(bits_identical(&c1, &c2), "gemm not deterministic under fixed knobs par={:?}", par);
            assert_accurate(&c1, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), "knob gemm");
        }

        // (ii) prepack_rhs + gemm_packed_b (column-major-ish C): also exercises the
        // tiny-branch prepack sentinel (tiny_block_dim() + 1) under the same assignment
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

        // (iii) batched (3 contiguous col-major elements): each element checked for frob
        // accuracy and determinism
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

/// Transpose a column-major `rows x cols` slice into a row-major `Vec` (for the f64
/// reference, which reads row-major logical matrices)
fn col_major_to_rowmajor(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = v[j * rows + i];
        }
    }
    out
}

// P16b: prepack under one knob assignment, then consume under a different one. The
// prepack geometry is baked in at pack time and read back verbatim by the consumer, so the
// result must still be correct regardless of what the knobs did in between - this directly
// tests that decoupling

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

// P17 gemv routing: the dedicated gemv path (threshold on) and the general driver
// (threshold 0) must both stay accurate. The 2 routes differ, so only a tolerance
// comparison applies, not bitwise

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

// P18 small-m,n horizontal route (row-major A + column-major B, k above the small-k
// threshold) vs the general driver, both accurate. small_k_threshold is pinned to 16 inside
// the guard so the k-gate is deterministic regardless of the arch-split default or ambient env

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

        // Row-major A (csa == 1) + column-major B (rsb == 1) + column-major C: the layout
        // small_mn_eligible requires; keeping C column-major also stops orient_transpose
        // from swapping the operands away from that layout
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

// P19 small-k route vs the general driver, both accurate. small_mn_dim is pinned to 0
// inside the guard so the small-mn gate cannot also flip while small_k_threshold is toggled

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

// P20 i8 VNNI gate (x86 + int8 only; the knob only affects the VNNI auto-select path, so
// it is a no-op on any other target/type): the VNNI kernel (gate 0) and the widen fallback
// (gate usize::MAX) must both equal the wrapping-i32 reference exactly and be bit-identical
// to each other, since both compute the exact same wrapping i32 arithmetic

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

        // Column-major operands, matching the layout tests/tuning.rs's own VNNI-gate test uses
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
        // Column-major output vs a row-major reference: read at [j*m + i]
        for i in 0..m {
            for j in 0..n {
                prop_assert_eq!(vnni[j * m + i], cref[i * n + j], "VNNI wrong at ({},{})", i, j);
                prop_assert_eq!(widen[j * m + i], cref[i * n + j], "widen wrong at ({},{})", i, j);
            }
        }
        prop_assert_eq!(&vnni, &widen, "VNNI and widen kernels must be bit-identical");
    }
}

// P21 mixed (f16) small-m,n horizontal route, forced onto a parallel tail: the mixed route
// has otherwise only ever run serial in coverage. Pinning the gemv/bandwidth byte floor to 1
// lets even this small k fork under Rayon(2); the route reduces each output within a single
// worker with no cross-thread combine, so serial and parallel must be bit-identical
// beta == 1 additionally exercises the epilogue's accumulate arm

#[cfg(feature = "half")]
#[test]
fn prop_small_mn_mixed_parallel_bit_identical() {
    use gemmkit::f16;

    let _lock = knob_guard();
    let _restore = KnobGuard::capture();
    // Route onto the mixed small_mn path and force its worker count above 1: small_mn_dim
    // must cover m,n; small_k_threshold must be below k; and a byte floor of 1 clears the
    // gemv/gevv bandwidth gate that would otherwise keep this shape serial
    tuning::set_small_mn_dim(16);
    tuning::set_small_k_threshold(16);
    tuning::set_gemv_parallel_bytes(1);

    let (m, n, k) = (16usize, 16usize, 64usize);
    // Row-major A (csa == 1) + column-major B (rsb == 1) + column-major C: the layout the
    // mixed dispatch gate requires; column-major C also keeps orient_transpose from
    // swapping the operands away from it
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
