//! Property-based tests for the safe public GEMM API: oracle accuracy, run-to-run
//! determinism, serial==parallel, beta==0 overwrite, broadcast inputs, batched, and
//! the documented panic guarantees. Never mutates knobs or env. See props_common for
//! the shared strategies and accuracy bars.
#![cfg(all(not(miri), not(target_family = "wasm")))]

mod props_common;

use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_batched, gemm_with};
use props_common::*;
use proptest::prelude::*;
use std::panic::AssertUnwindSafe;

// ---------------------------------------------------------------------------
// P1/P2/P3/P5 — random oracle: run-to-run BIT determinism + frob vs the f64 reference
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn oracle_float<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: PLayout,
    lb: PLayout,
    lc: PLayout,
    al: f64,
    be: f64,
    p: Parallelism,
    seed: u64,
) {
    let a = Mat::<T>::rand(m, k, seed);
    let b = Mat::<T>::rand(k, n, seed ^ 0xB000);
    let c0 = Mat::<T>::rand(m, n, seed ^ 0xC000);
    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, lb);
    let (cbase, rsc, csc) = build_view(&c0, lc);
    let (alpha, beta) = (T::from_f64(al), T::from_f64(be));
    let cref = reference(&a, &b, &c0, al, be);

    let mut c1 = cbase.clone();
    let mut c2 = cbase.clone();
    gemm(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut c1, m, n, rsc, csc),
        p,
    );
    gemm(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut c2, m, n, rsc, csc),
        p,
    );
    assert!(
        bits_identical(&c1, &c2),
        "run-to-run non-determinism {m}x{k}x{n} par={p:?}"
    );
    let ctx = format!("oracle {m}x{k}x{n} a={al} b={be} par={p:?}");
    assert_accurate(
        &c1,
        rsc,
        csc,
        m,
        n,
        &cref,
        &a,
        &b,
        k,
        be.abs() * frob_norm(&c0),
        &ctx,
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(160), ..ProptestConfig::default() })]

    #[test]
    fn prop_oracle_f32(
        m in dim(), k in kdim_pos(), n in dim(),
        la in layout(), lb in layout(), lc in layout(),
        al in coeff(), be in coeff(), p in par(), seed in any::<u64>(),
    ) {
        oracle_float::<f32>(m, k, n, la, lb, lc, al, be, p, seed);
    }

    #[test]
    fn prop_oracle_f64(
        m in dim(), k in kdim_pos(), n in dim(),
        la in layout(), lb in layout(), lc in layout(),
        al in coeff(), be in coeff(), p in par(), seed in any::<u64>(),
    ) {
        oracle_float::<f64>(m, k, n, la, lb, lc, al, be, p, seed);
    }
}

#[cfg(feature = "half")]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(80), ..ProptestConfig::default() })]

    #[test]
    fn prop_oracle_f16(
        m in dim(), k in kdim_pos(), n in dim(),
        la in layout(), lb in layout(), lc in layout(),
        al in coeff(), be in coeff(), p in par(), seed in any::<u64>(),
    ) {
        oracle_float::<gemmkit::f16>(m, k, n, la, lb, lc, al, be, p, seed);
    }

    #[test]
    fn prop_oracle_bf16(
        m in dim(), k in kdim_pos(), n in dim(),
        la in layout(), lb in layout(), lc in layout(),
        al in coeff(), be in coeff(), p in par(), seed in any::<u64>(),
    ) {
        oracle_float::<gemmkit::bf16>(m, k, n, la, lb, lc, al, be, p, seed);
    }
}

// ---------------------------------------------------------------------------
// P4 — full-range i8 -> i32 vs the wrapping-i32 reference (exact)
// ---------------------------------------------------------------------------

#[cfg(feature = "int8")]
fn fill_i32(n: usize, seed: u64) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as i32
        })
        .collect()
}

#[cfg(feature = "int8")]
fn i8_coeff() -> impl Strategy<Value = i32> {
    proptest::sample::select(&[0i32, 1, -1, 3, -2, 100_000][..])
}

#[cfg(feature = "int8")]
fn i8_par() -> impl Strategy<Value = Parallelism> {
    proptest::sample::select(&[Parallelism::Serial, Parallelism::Rayon(0)][..])
}

#[cfg(feature = "int8")]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(112), ..ProptestConfig::default() })]

    #[test]
    fn prop_oracle_i8(
        m in dim(), k in kdim(), n in dim(),
        la in layout(), lb in layout(), lc in layout(),
        alpha in i8_coeff(), beta in i8_coeff(), p in i8_par(), seed in any::<u64>(),
    ) {
        let a = fill_i8(m * k, seed);
        let b = fill_i8(k * n, seed ^ 0xB);
        let c0 = fill_i32(m * n, seed ^ 0xC);
        let cref = ref_i8_wrapping(&a, &b, &c0, m, k, n, alpha, beta);

        let (abuf, rsa, csa) = build_view_rowmajor(&a, m, k, 0i8, la);
        let (bbuf, rsb, csb) = build_view_rowmajor(&b, k, n, 0i8, lb);
        let (mut cbuf, rsc, csc) = build_view_rowmajor(&c0, m, n, 0i32, lc);

        gemmkit::gemm_i8(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut cbuf, m, n, rsc, csc),
            p,
        );
        for i in 0..m {
            for j in 0..n {
                let got = cbuf[(i as isize * rsc + j as isize * csc) as usize];
                prop_assert_eq!(
                    got, cref[i * n + j],
                    "i8 mismatch at ({},{}) {}x{}x{} a={} b={}", i, j, m, k, n, alpha, beta
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P5 — complex (c32/c64), conj variants: run-to-run BIT + frob(16*k*eps)
// ---------------------------------------------------------------------------

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn oracle_cplx<T: CElem>(
    m: usize,
    k: usize,
    n: usize,
    ca: bool,
    cb: bool,
    ar: f64,
    ai: f64,
    br: f64,
    bi: f64,
    p: Parallelism,
    seed: u64,
) {
    let (alpha, beta) = (T::of(ar, ai), T::of(br, bi));
    let a = rand_cplx::<T>(m * k, seed);
    let b = rand_cplx::<T>(k * n, seed ^ 0xB);
    let c0 = rand_cplx::<T>(m * n, seed ^ 0xC);
    let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, ca, cb);

    let mut c1 = c0.clone();
    let mut c2 = c0.clone();
    gemmkit::gemm_cplx(
        alpha,
        MatRef::from_col_major(&a, m, k),
        ca,
        MatRef::from_col_major(&b, k, n),
        cb,
        beta,
        MatMut::from_col_major(&mut c1, m, n),
        p,
    );
    gemmkit::gemm_cplx(
        alpha,
        MatRef::from_col_major(&a, m, k),
        ca,
        MatRef::from_col_major(&b, k, n),
        cb,
        beta,
        MatMut::from_col_major(&mut c2, m, n),
        p,
    );
    assert!(
        cplx_bits_identical(&c1, &c2),
        "cplx run-to-run non-determinism {m}x{k}x{n} ca={ca} cb={cb} par={p:?}"
    );
    assert_cplx_accurate(&c1, m, n, &cref, k, &format!("cplx oracle {m}x{k}x{n}"));
}

#[cfg(feature = "complex")]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(80), ..ProptestConfig::default() })]

    #[test]
    fn prop_oracle_c32(
        m in dim(), k in kdim_pos(), n in dim(),
        ca in any::<bool>(), cb in any::<bool>(),
        ar in coeff(), ai in coeff(), br in coeff(), bi in coeff(),
        p in par(), seed in any::<u64>(),
    ) {
        oracle_cplx::<gemmkit::c32>(m, k, n, ca, cb, ar, ai, br, bi, p, seed);
    }

    #[test]
    fn prop_oracle_c64(
        m in dim(), k in kdim_pos(), n in dim(),
        ca in any::<bool>(), cb in any::<bool>(),
        ar in coeff(), ai in coeff(), br in coeff(), bi in coeff(),
        p in par(), seed in any::<u64>(),
    ) {
        oracle_cplx::<gemmkit::c64>(m, k, n, ca, cb, ar, ai, br, bi, p, seed);
    }
}

// ---------------------------------------------------------------------------
// P6 — serial == parallel, bit-identical (weighted-large dims so Rayon really splits)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    // Serial and parallel runs are bit-identical **under the current thread-independent
    // blocking** — float add isn't associative, so this holds only because every thread
    // count reduces in the same order, not because the library promises it (see the
    // canonical caveat at tests/correctness.rs:790-798). Kept as the strongest net against
    // an accidental reduction-order divergence; relax to determinism + tolerance only if
    // blocking ever becomes parallelism-dependent (e.g. split-K).
    #[test]
    fn prop_serial_equals_parallel(
        m in 48usize..=200, n in 48usize..=200, k in 32usize..=160,
        t in proptest::sample::select(&[2usize, 4, 8][..]),
        al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (abuf, rsa, csa) = build_view(&a, PLayout::Col { pad: 0 });
        let (bbuf, rsb, csb) = build_view(&b, PLayout::Col { pad: 0 });
        let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 });
        let (alpha, beta) = (al as f32, be as f32);

        let mut c_ser = cbase.clone();
        gemm(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let mut c_par = cbase.clone();
        gemm(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut c_par, m, n, rsc, csc),
            Parallelism::Rayon(t),
        );
        prop_assert!(
            bits_identical(&c_ser, &c_par),
            "serial != parallel({}) for {}x{}x{}", t, m, k, n
        );
    }
}

// ---------------------------------------------------------------------------
// P7 — beta == 0 must overwrite, never read C (NaN-seeded C proves it)
// ---------------------------------------------------------------------------

fn beta_zero_overwrites<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: PLayout,
    lc: PLayout,
    al: f64,
    seed: u64,
) {
    let a = Mat::<T>::rand(m, k, seed);
    let b = Mat::<T>::rand(k, n, seed ^ 0xB);
    // Reference uses a *zeroed* C0 with beta = 0 (mirrors tests/correctness.rs:609-647).
    let zero_c0 = Mat {
        v: vec![T::from_f64(0.0); m * n],
        rows: m,
        cols: n,
    };
    let cref = reference(&a, &b, &zero_c0, al, 0.0);
    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, PLayout::Col { pad: 0 });
    // C seeded all-NaN in whatever layout `lc` picks.
    let nan = Mat {
        v: vec![T::from_f64(f64::NAN); m * n],
        rows: m,
        cols: n,
    };
    let (mut cbuf, rsc, csc) = build_view(&nan, lc);
    gemm(
        T::from_f64(al),
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        T::from_f64(0.0),
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        Parallelism::Serial,
    );
    assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, 0.0, "beta=0 NaN C");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_beta_zero_overwrites_f32(
        m in dim(), k in kdim(), n in dim(),
        la in layout(), lc in layout(), al in coeff(), seed in any::<u64>(),
    ) {
        beta_zero_overwrites::<f32>(m, k, n, la, lc, al, seed);
    }

    #[test]
    fn prop_beta_zero_overwrites_f64(
        m in dim(), k in kdim(), n in dim(),
        la in layout(), lc in layout(), al in coeff(), seed in any::<u64>(),
    ) {
        beta_zero_overwrites::<f64>(m, k, n, la, lc, al, seed);
    }
}

#[cfg(feature = "half")]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    #[test]
    fn prop_beta_zero_overwrites_f16(
        m in dim(), k in kdim(), n in dim(),
        la in layout(), lc in layout(), al in coeff(), seed in any::<u64>(),
    ) {
        beta_zero_overwrites::<gemmkit::f16>(m, k, n, la, lc, al, seed);
    }

    #[test]
    fn prop_beta_zero_overwrites_bf16(
        m in dim(), k in kdim(), n in dim(),
        la in layout(), lc in layout(), al in coeff(), seed in any::<u64>(),
    ) {
        beta_zero_overwrites::<gemmkit::bf16>(m, k, n, la, lc, al, seed);
    }
}

// ---------------------------------------------------------------------------
// P8 — batched GEMM reproduces a loop of single gemm(Serial) calls (bit-identical)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(96), ..ProptestConfig::default() })]

    #[test]
    fn prop_batched_matches_loop(
        batch in 0usize..=4,
        m in 1usize..=48, k in 0usize..=48, n in 1usize..=48,
        a_pad in 0usize..=7, b_pad in 0usize..=7, c_pad in 0usize..=7,
        broadcast in proptest::sample::select(&[0u8, 1, 2][..]), // 0=none, 1=A, 2=B
        al in coeff(), be in coeff(),
        p in proptest::sample::select(&[Parallelism::Serial, Parallelism::Rayon(0)][..]),
        seed in any::<u64>(),
    ) {
        let (alpha, beta) = (al as f32, be as f32);
        // Column-major element extents.
        let (ea, eb, ec) = (m * k, k * n, m * n);
        // A or B may broadcast (batch stride 0, shared across the batch) — legal per api.rs:322-324.
        let (a_bs, b_bs) = match broadcast {
            1 => (0isize, (eb + b_pad) as isize),
            2 => ((ea + a_pad) as isize, 0isize),
            _ => ((ea + a_pad) as isize, (eb + b_pad) as isize),
        };
        let c_bs = (ec + c_pad) as isize; // C batch stride always >= extent (api.rs:455-460)
        let a_slots = if a_bs == 0 { 1 } else { batch };
        let b_slots = if b_bs == 0 { 1 } else { batch };
        let a = rand_vec::<f32>(a_slots.max(1) * (ea + a_pad), seed);
        let b = rand_vec::<f32>(b_slots.max(1) * (eb + b_pad), seed ^ 0xB);
        let c0 = rand_vec::<f32>(batch.max(1) * (ec + c_pad), seed ^ 0xC);

        let a_at = |bi: usize| (bi as isize * a_bs) as usize;
        let b_at = |bi: usize| (bi as isize * b_bs) as usize;
        let c_at = |bi: usize| (bi as isize * c_bs) as usize;

        // Reference: an independent single gemm(Serial) per element on its own window.
        let mut c_ref = c0.clone();
        for bi in 0..batch {
            let (ao, bo, co) = (a_at(bi), b_at(bi), c_at(bi));
            gemm(
                alpha,
                MatRef::from_col_major(&a[ao..ao + ea], m, k),
                MatRef::from_col_major(&b[bo..bo + eb], k, n),
                beta,
                MatMut::from_col_major(&mut c_ref[co..co + ec], m, n),
                Parallelism::Serial,
            );
        }

        let mut c_bat = c0.clone();
        gemm_batched(
            batch,
            alpha,
            MatRef::new(&a, m, k, 1, m as isize),
            a_bs,
            MatRef::new(&b, k, n, 1, k as isize),
            b_bs,
            beta,
            MatMut::new(&mut c_bat, m, n, 1, m as isize),
            c_bs,
            p,
        );
        prop_assert!(
            bits_identical(&c_ref, &c_bat),
            "batched != gemm() loop batch={} {}x{}x{} bcast={} par={:?}", batch, m, k, n, broadcast, p
        );
    }
}

// ---------------------------------------------------------------------------
// P9 — broadcast inputs (rs = 0 on A or cs = 0 on B) drive the full pipeline
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_broadcast_inputs(
        m in dim(), k in kdim_pos(), n in dim(),
        which in any::<bool>(), lc in layout(), al in coeff(), be in coeff(),
        p in par(), seed in any::<u64>(),
    ) {
        // Materialize the logical (broadcast) operand for the reference, but present it to
        // gemm through a compact buffer with a zero stride (legal for read-only A/B,
        // api.rs:193-207 + 269-271).
        let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
        let (cbase, rsc, csc) = build_view(&c0, lc);
        let (alpha, beta) = (al as f32, be as f32);

        let (a, b, abuf, rsa, csa, bbuf, rsb, csb) = if which {
            // Broadcast A: every row equals `base` (k values), rs = 0.
            let base = rand_vec::<f32>(k, seed);
            let full_a = Mat { v: (0..m).flat_map(|_| base.clone()).collect(), rows: m, cols: k };
            let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
            let (bbuf, rsb, csb) = build_view(&b, PLayout::Col { pad: 0 });
            (full_a, b, base, 0isize, 1isize, bbuf, rsb, csb)
        } else {
            // Broadcast B: every column equals `base` (k values), cs = 0.
            let a = Mat::<f32>::rand(m, k, seed);
            let base = rand_vec::<f32>(k, seed ^ 0xB);
            let full_b = Mat { v: (0..k).flat_map(|p2| core::iter::repeat_n(base[p2], n)).collect(), rows: k, cols: n };
            let (abuf, rsa, csa) = build_view(&a, PLayout::Col { pad: 0 });
            (a, full_b, abuf, rsa, csa, base, 1isize, 0isize)
        };

        let cref = reference(&a, &b, &c0, al, be);
        let mut cbuf = cbase.clone();
        gemm(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut cbuf, m, n, rsc, csc),
            p,
        );
        assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, be.abs() * frob_norm(&c0), "broadcast");
    }
}

// ---------------------------------------------------------------------------
// P11 — cross-library oracle: gemmkit vs the `gemm` crate (col-major, beta=0, serial)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    #[test]
    fn prop_matches_gemm_crate(
        m in 1usize..=64, k in 1usize..=64, n in 1usize..=64, seed in any::<u64>(),
    ) {
        let a = Mat::<f32>::rand(m, k, seed);
        let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
        // Column-major buffers (gemm's orientation), zero beta.
        let (abuf, _, _) = build_view(&a, PLayout::Col { pad: 0 });
        let (bbuf, _, _) = build_view(&b, PLayout::Col { pad: 0 });
        let mut c_kit = vec![0.0f32; m * n];
        let mut c_gemm = vec![0.0f32; m * n];
        gemm(
            1.0f32,
            MatRef::from_col_major(&abuf, m, k),
            MatRef::from_col_major(&bbuf, k, n),
            0.0f32,
            MatMut::from_col_major(&mut c_kit, m, n),
            Parallelism::Serial,
        );
        // SAFETY: distinct in-bounds column-major buffers; read_dst=false (beta=0).
        unsafe {
            gemm::gemm(
                m, n, k,
                c_gemm.as_mut_ptr(), m as isize, 1, false,
                abuf.as_ptr(), m as isize, 1,
                bbuf.as_ptr(), k as isize, 1,
                0.0f32, 1.0f32, false, false, false,
                gemm::Parallelism::None,
            );
        }
        // Build a row-major f64 reference from the column-major `gemm` output.
        let mut cref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                cref[i * n + j] = c_gemm[i + j * m] as f64;
            }
        }
        assert_accurate(&c_kit, 1, m as isize, m, n, &cref, &a, &b, k, 0.0, "vs gemm crate");
    }
}

// ---------------------------------------------------------------------------
// P10 — safe-API panic guarantees (adversarial): each class must panic with its
// documented message. `catch_unwind` + downcast; no `set_hook` (it is process-global
// and would race the other concurrently-running tests). `AssertUnwindSafe` because the
// `&mut` C buffers are not `UnwindSafe`.
// ---------------------------------------------------------------------------

fn catch_msg<F: FnOnce()>(f: F) -> Option<String> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => None,
        Err(e) => Some(
            e.downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_default(),
        ),
    }
}

fn assert_panics_with<F: FnOnce()>(f: F, needle: &str, ctx: &str) {
    match catch_msg(f) {
        None => panic!("{ctx}: expected a panic containing {needle:?}, but the call returned"),
        Some(msg) => assert!(
            msg.contains(needle),
            "{ctx}: panic message {msg:?} does not contain {needle:?}"
        ),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    // (a) shape mismatch: A.cols != B.rows.
    #[test]
    fn prop_panic_shape_mismatch(m in 1usize..=8, k in 1usize..=8, n in 1usize..=8, d in 1usize..=4) {
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; (k + d) * n]; // B.rows = k+d != A.cols = k
        let mut c = vec![0.0f32; m * n];
        assert_panics_with(
            || gemm(
                1.0,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_row_major(&b, k + d, n),
                0.0,
                MatMut::from_row_major(&mut c, m, n),
                Parallelism::Serial,
            ),
            "B.rows",
            "shape mismatch",
        );
    }

    // (b) undersized slice for A -> "needs".
    #[test]
    fn prop_panic_undersized(m in 2usize..=8, k in 2usize..=8, n in 1usize..=8, short in 1usize..=3) {
        let full = m * k;
        let a = vec![0.0f32; full.saturating_sub(short).max(1)]; // too small for the m x k row-major view
        let b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        assert_panics_with(
            || gemm(
                1.0,
                MatRef::new(&a, m, k, k as isize, 1),
                MatRef::from_row_major(&b, k, n),
                0.0,
                MatMut::from_row_major(&mut c, m, n),
                Parallelism::Serial,
            ),
            "needs",
            "undersized A",
        );
    }

    // (c) self-aliasing C (rsc = 0 collapses distinct rows) -> "aliases itself".
    #[test]
    fn prop_panic_self_aliasing_c(m in 2usize..=8, k in 1usize..=8, n in 2usize..=8, p in par()) {
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; n]; // one row's worth; rsc = 0 maps every row onto it
        assert_panics_with(
            || gemm(
                1.0,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_row_major(&b, k, n),
                0.0,
                MatMut::new(&mut c, m, n, 0, 1),
                p,
            ),
            "aliases itself",
            "self-aliasing C",
        );
    }

    // (d) negative stride on a safe A view -> "negative strides or is too large".
    #[test]
    fn prop_panic_negative_stride(m in 2usize..=8, k in 2usize..=8, n in 1usize..=8) {
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        assert_panics_with(
            || gemm(
                1.0,
                MatRef::new(&a, m, k, -(k as isize), 1), // negative row stride
                MatRef::from_row_major(&b, k, n),
                0.0,
                MatMut::from_row_major(&mut c, m, n),
                Parallelism::Serial,
            ),
            "negative strides or is too large",
            "negative stride",
        );
    }

    // (e) extent overflow around the addressing boundary. When `(rows-1)*rs` overflows
    // isize, extent() is None and check_view panics "too large to address"; when it fits
    // but the huge need exceeds the 1-element slice, it panics "needs". Both branches are
    // reachable here (small rows + huge rs -> "needs"; big rows -> "too large"); the
    // generator computes which side it lands on.
    #[test]
    fn prop_panic_extent_overflow(
        rows in proptest::sample::select(&[2usize, 3, (1usize << (isize::BITS / 2 + 1)) + 1][..]),
        rs_pow in (isize::BITS / 2)..=(isize::BITS / 2 + 2),
    ) {
        let rs = 1isize << rs_pow;
        let a = vec![0.0f32; 1];
        let b = vec![0.0f32; 1];
        let mut c = vec![0.0f32; 1];
        // Mirror extent(): (rows-1) * rs overflowing isize is the "too large" branch.
        let overflows = isize::try_from(rows)
            .ok()
            .and_then(|r| r.checked_sub(1))
            .and_then(|r| r.checked_mul(rs))
            .is_none();
        let needle = if overflows { "too large to address" } else { "needs" };
        assert_panics_with(
            || gemm(
                1.0,
                MatRef::new(&a, rows, 1, rs, 1),
                MatRef::from_row_major(&b, 1, 1),
                0.0,
                MatMut::new(&mut c, rows, 1, rs, 1),
                Parallelism::Serial,
            ),
            needle,
            "extent overflow",
        );
    }
}

// ---------------------------------------------------------------------------
// Batched-validation panics (adversarial sibling of P10 for gemm_batched)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    // Negative batch stride -> "must be non-negative" (api.rs:169-171).
    #[test]
    fn prop_batched_panic_negative_batch_stride(batch in 2usize..=4, m in 1usize..=6, k in 1usize..=6, n in 1usize..=6) {
        let a = rand_vec::<f32>(batch * m * k, 1);
        let b = rand_vec::<f32>(batch * k * n, 2);
        let mut c = rand_vec::<f32>(batch * m * n, 3);
        assert_panics_with(
            || gemm_batched(
                batch, 1.0,
                MatRef::new(&a, m, k, 1, m as isize), -((m * k) as isize),
                MatRef::new(&b, k, n, 1, k as isize), (k * n) as isize,
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize), (m * n) as isize,
                Parallelism::Serial,
            ),
            "non-negative",
            "batched negative batch stride",
        );
    }

    // C batch stride below the element extent -> "stay disjoint" (api.rs:455-460).
    #[test]
    fn prop_batched_panic_overlapping_c(batch in 2usize..=4, m in 2usize..=6, n in 2usize..=6, deficit in 1usize..=3) {
        let a = rand_vec::<f32>(batch * m * m, 1);
        let b = rand_vec::<f32>(batch * m * n, 2);
        let mut c = vec![0.0f32; batch * m * n];
        let bad = (m * n).saturating_sub(deficit).max(1) as isize;
        assert_panics_with(
            || gemm_batched(
                batch, 1.0,
                MatRef::new(&a, m, m, 1, m as isize), (m * m) as isize,
                MatRef::new(&b, m, n, 1, m as isize), (m * n) as isize,
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize), bad, // < element extent m*n
                Parallelism::Serial,
            ),
            "stay disjoint",
            "batched overlapping C",
        );
    }

    // Per-element shape mismatch -> "!= B.rows" (api.rs:395-399).
    #[test]
    fn prop_batched_panic_shape_mismatch(batch in 1usize..=4, m in 1usize..=6, k in 1usize..=6, n in 1usize..=6, d in 1usize..=3) {
        let a = rand_vec::<f32>(batch * m * k, 1);
        let b = rand_vec::<f32>(batch * (k + d) * n, 2);
        let mut c = rand_vec::<f32>(batch * m * n, 3);
        assert_panics_with(
            || gemm_batched(
                batch, 1.0,
                MatRef::new(&a, m, k, 1, m as isize), (m * k) as isize,
                MatRef::new(&b, k + d, n, 1, (k + d) as isize), ((k + d) * n) as isize,
                0.0,
                MatMut::new(&mut c, m, n, 1, m as isize), (m * n) as isize,
                Parallelism::Serial,
            ),
            "!= B.rows",
            "batched shape mismatch",
        );
    }
}

// ---------------------------------------------------------------------------
// Stateful workspace reuse: a sequence of random-shaped gemm_with() calls through one
// grown-only Workspace, each BIT-compared against a fresh pool-allocating gemm().
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    #[test]
    fn prop_workspace_reuse_stateful(
        ops in prop::collection::vec(
            (dim(), kdim(), dim(), layout(), coeff(), coeff(), par(), any::<u64>()),
            1..=6,
        ),
    ) {
        let mut ws = Workspace::new();
        for (i, (m, k, n, lc, al, be, p, seed)) in ops.into_iter().enumerate() {
            let a = Mat::<f32>::rand(m, k, seed);
            let b = Mat::<f32>::rand(k, n, seed ^ 0xB);
            let c0 = Mat::<f32>::rand(m, n, seed ^ 0xC);
            let (abuf, rsa, csa) = build_view(&a, PLayout::Col { pad: 0 });
            let (bbuf, rsb, csb) = build_view(&b, PLayout::Col { pad: 0 });
            let (cbase, rsc, csc) = build_view(&c0, lc);
            let (alpha, beta) = (al as f32, be as f32);

            let mut c_pool = cbase.clone();
            gemm(
                alpha,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                beta,
                MatMut::new(&mut c_pool, m, n, rsc, csc),
                p,
            );
            let mut c_ws = cbase.clone();
            gemm_with(
                &mut ws,
                alpha,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                beta,
                MatMut::new(&mut c_ws, m, n, rsc, csc),
                p,
            );
            prop_assert!(
                bits_identical(&c_pool, &c_ws),
                "workspace-reuse step {} diverged {}x{}x{}", i, m, k, n
            );
        }
    }
}

/// Regression: on the mixed-precision (f16/bf16) path the depth panel is the whole `k`,
/// and a broadcast (zero-stride) operand passes validation with a logically huge `k` —
/// the pack sizing must fail closed rather than wrap into an undersized workspace the pack
/// writes past. `k = isize::MAX` overflows the pack element-count product for every tile
/// geometry (so the "too large" guard fires on every ISA). The narrower band where the
/// element product fits `usize` but the element→byte conversion overflows is exercised
/// tile-independently by `workspace::region_bytes` unit tests — an end-to-end `k` that hits
/// that band is tile-size-dependent (a smaller tile just requests a huge-but-representable
/// allocation, which is a safe OOM abort, not the checked "too large" panic).
#[cfg(feature = "half")]
#[test]
fn mixed_huge_k_fails_closed() {
    use gemmkit::{bf16, f16};

    let huge = isize::MAX as usize;
    {
        let a = vec![f16::from_f32(1.0); 100];
        let b = vec![f16::from_f32(1.0); 100];
        let mut c = vec![f16::from_f32(0.0); 100 * 100];
        assert_panics_with(
            || {
                gemm(
                    f16::from_f32(1.0),
                    MatRef::new(&a, 100, huge, 1, 0),
                    MatRef::new(&b, huge, 100, 0, 1),
                    f16::from_f32(0.0),
                    MatMut::from_row_major(&mut c, 100, 100),
                    Parallelism::Serial,
                )
            },
            "too large",
            "f16 huge-k",
        );

        let a = vec![bf16::from_f32(1.0); 100];
        let b = vec![bf16::from_f32(1.0); 100];
        let mut c = vec![bf16::from_f32(0.0); 100 * 100];
        assert_panics_with(
            || {
                gemm(
                    bf16::from_f32(1.0),
                    MatRef::new(&a, 100, huge, 1, 0),
                    MatRef::new(&b, huge, 100, 0, 1),
                    bf16::from_f32(0.0),
                    MatMut::from_row_major(&mut c, 100, 100),
                    Parallelism::Serial,
                )
            },
            "too large",
            "bf16 huge-k",
        );
    }
}
