//! Property-based tests for the prepacked-reuse API (prepack_rhs/gemm_packed_b and
//! prepack_lhs/gemm_packed_a): bit-identity to plain gemm on the general regime,
//! accuracy + cross-thread determinism on the documented tiny/gemv exception set, and
//! the orientation-guard panics. Never mutates knobs. See props_common for shared bars.
#![cfg(all(not(miri), not(target_family = "wasm")))]

mod props_common;

use gemmkit::{
    MatMut, MatRef, Parallelism, gemm, gemm_packed_a, gemm_packed_b, prepack_lhs, prepack_rhs,
    tuning,
};
use props_common::*;
use proptest::prelude::*;

/// The frob-vs-BIT split keys off the documented prepack exception set: `m==1 || n==1`
/// (gemv-shaped) or both-tiny (`m,n <= tiny_block_dim()`), read via getters so it tracks
/// ambient env. Plain `gemm` also routes `m,n <= small_mn_dim() && k > small_k_threshold()`
/// to the horizontal path (whose summation order differs from the packed driver); that set
/// is a subset of both-tiny **iff** `small_mn_dim <= tiny_block_dim`, which we assert so a
/// horizontal-routed shape can never fall into the strict BIT branch.
fn assert_packed_env_sane() {
    assert!(
        tuning::small_mn_dim() <= tuning::tiny_block_dim(),
        "props_packed assumes small_mn_dim ({}) <= tiny_block_dim ({}); an ambient \
         GEMMKIT_SMALL_MN_DIM/GEMMKIT_TINY_BLOCK_DIM profile violates that",
        tuning::small_mn_dim(),
        tuning::tiny_block_dim()
    );
}

fn is_tiny_or_gemv(m: usize, n: usize) -> bool {
    let tiny = tuning::tiny_block_dim();
    m == 1 || n == 1 || (m <= tiny && n <= tiny)
}

const PACKED_PARS: [Parallelism; 3] = [
    Parallelism::Serial,
    Parallelism::Rayon(2),
    Parallelism::Rayon(8),
];

// ---------------------------------------------------------------------------
// P12 / P15(rhs) — prepack_rhs matches plain gemm (column-major-ish C)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn prepack_rhs_check<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: PLayout,
    lb: PLayout,
    al: f64,
    be: f64,
    seed: u64,
) {
    assert_packed_env_sane();
    let a = Mat::<T>::rand(m, k, seed);
    let b = Mat::<T>::rand(k, n, seed ^ 0xB);
    let c0 = Mat::<T>::rand(m, n, seed ^ 0xC);
    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, lb);
    let (cbase, rsc, csc) = build_view(&c0, PLayout::Col { pad: 0 }); // column-major-ish C required
    let (alpha, beta) = (T::from_f64(al), T::from_f64(be));

    let packed = prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
    assert_eq!(packed.rows(), k);
    assert_eq!(packed.cols(), n);

    if is_tiny_or_gemv(m, n) {
        // Buffer's own blocking may round differently from plain gemm's tiny shortcut, so
        // check accuracy vs the f64 reference and bit-identity across thread counts.
        let cref = reference(&a, &b, &c0, al, be);
        let mut first: Option<Vec<T>> = None;
        for par in PACKED_PARS {
            let mut c = cbase.clone();
            gemm_packed_b(
                alpha,
                MatRef::new(&abuf, m, k, rsa, csa),
                &packed,
                beta,
                MatMut::new(&mut c, m, n, rsc, csc),
                par,
            );
            assert_accurate(
                &c,
                rsc,
                csc,
                m,
                n,
                &cref,
                &a,
                &b,
                k,
                be.abs() * frob_norm(&c0),
                "prepack_rhs tiny",
            );
            match &first {
                None => first = Some(c),
                Some(f) => assert!(
                    bits_identical(f, &c),
                    "prepack_rhs tiny not deterministic across par {m}x{k}x{n}"
                ),
            }
        }
    } else {
        // General regime: packing only rearranges B's values, so it is bit-identical to a
        // plain gemm() for every thread count.
        let mut c_ref = cbase.clone();
        gemm(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        for par in PACKED_PARS {
            let mut c = cbase.clone();
            gemm_packed_b(
                alpha,
                MatRef::new(&abuf, m, k, rsa, csa),
                &packed,
                beta,
                MatMut::new(&mut c, m, n, rsc, csc),
                par,
            );
            assert!(
                bits_identical(&c_ref, &c),
                "prepack_rhs != gemm {m}x{k}x{n} par={par:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// P13 / P15(lhs) — prepack_lhs matches plain gemm (row-major-ish C)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn prepack_lhs_check<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: PLayout,
    lb: PLayout,
    al: f64,
    be: f64,
    seed: u64,
) {
    assert_packed_env_sane();
    let a = Mat::<T>::rand(m, k, seed);
    let b = Mat::<T>::rand(k, n, seed ^ 0xB);
    let c0 = Mat::<T>::rand(m, n, seed ^ 0xC);
    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, lb);
    let (cbase, rsc, csc) = build_view(&c0, PLayout::Row { pad: 0 }); // row-major-ish C required
    let (alpha, beta) = (T::from_f64(al), T::from_f64(be));

    let packed = prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
    assert_eq!(packed.rows(), m);
    assert_eq!(packed.cols(), k);

    if is_tiny_or_gemv(m, n) {
        let cref = reference(&a, &b, &c0, al, be);
        let mut first: Option<Vec<T>> = None;
        for par in PACKED_PARS {
            let mut c = cbase.clone();
            gemm_packed_a(
                alpha,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                beta,
                MatMut::new(&mut c, m, n, rsc, csc),
                par,
            );
            assert_accurate(
                &c,
                rsc,
                csc,
                m,
                n,
                &cref,
                &a,
                &b,
                k,
                be.abs() * frob_norm(&c0),
                "prepack_lhs tiny",
            );
            match &first {
                None => first = Some(c),
                Some(f) => assert!(
                    bits_identical(f, &c),
                    "prepack_lhs tiny not deterministic across par {m}x{k}x{n}"
                ),
            }
        }
    } else {
        let mut c_ref = cbase.clone();
        gemm(
            alpha,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            beta,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        for par in PACKED_PARS {
            let mut c = cbase.clone();
            gemm_packed_a(
                alpha,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                beta,
                MatMut::new(&mut c, m, n, rsc, csc),
                par,
            );
            assert!(
                bits_identical(&c_ref, &c),
                "prepack_lhs != gemm {m}x{k}x{n} par={par:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(96), ..ProptestConfig::default() })]

    #[test]
    fn prop_prepack_rhs_matches_gemm_f32(
        m in pos_dim(), k in kdim_pos(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_rhs_check::<f32>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_rhs_matches_gemm_f64(
        m in pos_dim(), k in kdim_pos(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_rhs_check::<f64>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_lhs_matches_gemm_f32(
        m in pos_dim(), k in kdim_pos(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_lhs_check::<f32>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_lhs_matches_gemm_f64(
        m in pos_dim(), k in kdim_pos(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_lhs_check::<f64>(m, k, n, la, lb, al, be, seed);
    }
}

/// k drawn with an extra tail across the single-panel (`kc = k`) rule (api.rs:835-839),
/// including `1024`, for the mixed-precision packed paths.
#[cfg(feature = "half")]
fn kdim_mixed() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => kdim_pos(),
        1 => proptest::sample::select(&[513usize, 1024][..]),
    ]
}

#[cfg(feature = "half")]
proptest! {
    #![proptest_config(ProptestConfig { cases: cases(48), ..ProptestConfig::default() })]

    #[test]
    fn prop_prepack_mixed_rhs_f16(
        m in pos_dim(), k in kdim_mixed(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_rhs_check::<gemmkit::f16>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_mixed_rhs_bf16(
        m in pos_dim(), k in kdim_mixed(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_rhs_check::<gemmkit::bf16>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_mixed_lhs_f16(
        m in pos_dim(), k in kdim_mixed(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_lhs_check::<gemmkit::f16>(m, k, n, la, lb, al, be, seed);
    }

    #[test]
    fn prop_prepack_mixed_lhs_bf16(
        m in pos_dim(), k in kdim_mixed(), n in pos_dim(),
        la in layout(), lb in layout(), al in coeff(), be in coeff(), seed in any::<u64>(),
    ) {
        prepack_lhs_check::<gemmkit::bf16>(m, k, n, la, lb, al, be, seed);
    }
}

// ---------------------------------------------------------------------------
// P14 — packed paths reject the wrong C orientation. m,n >= 2 and a strict stride
// inequality (the guards use >=/<=, so square-ish rs==cs C is accepted, not a panic).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    // packed_b requires column-major-ish C; a strictly row-major C must panic.
    #[test]
    fn prop_packed_b_wrong_orientation_panics(m in 2usize..=32, k in 1usize..=16, n in 2usize..=32, pad in 0usize..=4) {
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        // Strictly row-major C: rs = n+pad > 1 = cs (|csc| < |rsc|).
        let mut c = vec![0.0f32; m * (n + pad)];
        let packed = prepack_rhs(MatRef::from_col_major(&b, k, n));
        let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gemm_packed_b(
                1.0,
                MatRef::from_col_major(&a, m, k),
                &packed,
                0.0,
                MatMut::new(&mut c, m, n, (n + pad) as isize, 1),
                Parallelism::Serial,
            );
        }))
        .err()
        .map(panic_string)
        .unwrap_or_default();
        prop_assert!(
            msg.contains("column-major-ish C"),
            "expected column-major-ish C panic, got {:?}", msg
        );
    }

    // packed_a requires row-major-ish C; a strictly column-major C must panic.
    #[test]
    fn prop_packed_a_wrong_orientation_panics(m in 2usize..=32, k in 1usize..=16, n in 2usize..=32, pad in 0usize..=4) {
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        // Strictly column-major C: cs = m+pad > 1 = rs (|csc| > |rsc|).
        let mut c = vec![0.0f32; (m + pad) * n];
        let packed = prepack_lhs(MatRef::from_col_major(&a, m, k));
        let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gemm_packed_a(
                1.0,
                &packed,
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::new(&mut c, m, n, 1, (m + pad) as isize),
                Parallelism::Serial,
            );
        }))
        .err()
        .map(panic_string)
        .unwrap_or_default();
        prop_assert!(
            msg.contains("row-major-ish C"),
            "expected row-major-ish C panic, got {:?}", msg
        );
    }
}

/// Downcast a caught panic payload to its message string.
fn panic_string(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| e.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// Regression: an empty operand must prepack — and round-trip through
/// the consume call as `C <- beta*C` — without running the pack-geometry arithmetic.
#[test]
fn prepack_empty_roundtrips() {
    let packed = prepack_rhs::<f32>(MatRef::new(&[], 0, 4, 1, 1));
    assert_eq!(packed.rows(), 0);
    assert_eq!(packed.cols(), 4);
    let mut c = vec![1.0f32; 3 * 4];
    gemm_packed_b(
        2.0,
        MatRef::new(&[], 3, 0, 1, 1),
        &packed,
        0.5,
        MatMut::from_col_major(&mut c, 3, 4),
        Parallelism::Serial,
    );
    assert!(
        c.iter().all(|&x| x == 0.5),
        "k==0 packed_b must beta-scale C"
    );

    let packed = prepack_lhs::<f32>(MatRef::new(&[], 3, 0, 1, 1));
    assert_eq!(packed.rows(), 3);
    assert_eq!(packed.cols(), 0);
    let mut c = vec![1.0f32; 3 * 4];
    gemm_packed_a(
        2.0,
        &packed,
        MatRef::new(&[], 0, 4, 1, 1),
        0.5,
        MatMut::from_row_major(&mut c, 3, 4),
        Parallelism::Serial,
    );
    assert!(
        c.iter().all(|&x| x == 0.5),
        "k==0 packed_a must beta-scale C"
    );
}

/// Regression: an empty view's extent is 0, so the safe API accepts a
/// huge free dimension (`usize::MAX x 0`); prepacking it must not touch the sizing
/// arithmetic (debug: add-with-overflow panic; release: wrapped geometry).
#[test]
fn prepack_huge_empty_dim_is_ok() {
    let packed = prepack_lhs::<f32>(MatRef::new(&[], usize::MAX, 0, 1, 1));
    assert_eq!(packed.rows(), usize::MAX);
    assert_eq!(packed.cols(), 0);
    let packed = prepack_rhs::<f32>(MatRef::new(&[], 0, usize::MAX, 1, 1));
    assert_eq!(packed.rows(), 0);
    assert_eq!(packed.cols(), usize::MAX);
}

/// Regression: a broadcast (zero-stride) view passes validation with a
/// tiny backing slice, so a logically huge non-empty operand reaches the pack sizing
/// arithmetic — it must fail closed (release would otherwise wrap the product into an
/// undersized allocation the pack then writes past).
#[test]
fn prepack_huge_broadcast_panics() {
    let buf = vec![1.0f32; 4];
    let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prepack_rhs(MatRef::new(&buf, 4, isize::MAX as usize, 1, 0))
    }))
    .err()
    .map(panic_string)
    .unwrap_or_default();
    assert!(
        msg.contains("too large"),
        "RHS: expected a too-large panic, got {msg:?}"
    );

    let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prepack_lhs(MatRef::new(&buf, isize::MAX as usize, 4, 0, 1))
    }))
    .err()
    .map(panic_string)
    .unwrap_or_default();
    assert!(
        msg.contains("too large"),
        "LHS: expected a too-large panic, got {msg:?}"
    );
}
