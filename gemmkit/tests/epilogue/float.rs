//! General-driver `gemm_fused` tests (f32/f64): the core `fused == gemm-then-map` oracle across
//! shapes/coefficients/layouts/bias/activation, `gemm_fused(None, None)` delegating to plain
//! `gemm`, the internal `run_epilogue::<Identity>` vs `run` proof, fire-once behavior across
//! multiple depth panels, bias orientation under row-major C, NaN/-0 handling, the degenerate
//! `k == 0` / `alpha == 0` case, validation panics, and the checked/unchecked twin entry points
//!
//! Every comparison is **bitwise**. The oracle holds for every shape (the fused engine routes
//! through the same kernel plain `gemm` does), but every case here uses a driver shape (`m, n
//! > 16`, `k > 16`, never a gemv shape); the special-path routes are covered in `special`

use crate::common::*;
use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_fused};

// the core oracle: fused == gemm-then-map, bitwise, across the full coefficient/layout/bias/
// activation lattice

fn fused_matrix<T: Flt>(par: Parallelism) {
    let mut rng = Rng::new(0xE91109E1);
    let shapes = [
        (17usize, 17usize, 17usize), // just above the small_mn/small_k thresholds (16)
        (33, 40, 24),                // rectangular, tile edges
        (64, 64, 64),
        (48, 96, 129), // row/col edges vs tiles
    ];
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.1))),
    ];
    // Under GEMMKIT_FAST_TEST, run the full lattice only on shape index 0 (the smallest) and
    // reduce the other 3 shapes to 1 non-trivial combo each: every class stays covered, only the
    // redundant per-shape cross-product is dropped. Off, fast is false and nothing is skipped
    let fast = fast_test();
    let full_lattice = 0usize;
    for (si, &(m, k, n)) in shapes.iter().enumerate() {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            if fast
                                && si != full_lattice
                                && !(beta == T::of(0.7)
                                    && alpha == T::of(0.9)
                                    && matches!(layout, Layout::ColPadded)
                                    && bias_kind == 2
                                    && matches!(act, Some(Activation::LeakyRelu(_))))
                            {
                                continue;
                            }
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

// identity: no bias, no activation

/// `gemm_fused(None, None)` delegates to plain `gemm`, bit-for-bit, across strides and
/// parallelism: the zero-cost identity case never reaches a fused monomorphization
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

/// `driver::run` is defined as `run_epilogue` called with `E = Identity`; this drives both
/// directly and checks they agree bit-for-bit, confirming the zero-cost-identity claim one
/// layer below the public API, for a fixed ISA token (`ScalarTok`, always runnable regardless
/// of `GEMMKIT_REQUIRE_ISA`)
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
            // SAFETY: valid col-major A/B/C views into disjoint buffers, ScalarTok always runnable
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

// fire-once: a large k splits into multiple depth panels; the epilogue must fire only on the last

/// A large-`k` driver shape forces `kc < k` (multiple depth panels), so the epilogue must fire
/// exactly once, on the final panel. Sign-mixed data plus `beta = 0.7` makes a per-panel ReLU or
/// a re-added bias diverge loudly from the oracle
#[test]
fn fire_once_multi_panel() {
    let mut rng = Rng::new(0xF11E);
    // k = 4096 is far above the default kc (512, itself an L1-fit estimate), so this spans
    // several depth panels
    check_fused::<f32>(
        &mut rng,
        40,
        4096,
        40,
        1.0,
        0.7,
        Layout::Col,
        1, // Bias::PerRow
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
        2, // Bias::PerCol
        Some(Activation::LeakyRelu(0.1)),
        Parallelism::Rayon(8),
        "fire-once/parallel",
    );
}

// bias orientation: {PerRow, PerCol} x {col-major, row-major} C

/// `check_fused`'s oracle applies the reference bias in the user frame, so a wrong
/// orientation-flip (row-major C makes the driver compute `C^T = B^T*A^T`, swapping m and n
/// internally) would diverge here
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

// NaN / -0 handling through the fused path

/// A NaN accumulator (from `inf + (-inf)`) must map to +0 under ReLU/LeakyReLU, and a -0 result
/// must map to +0 under LeakyReLU, on every ISA. `m`/`n` are large enough and C is col-major, so
/// the full tiles take the SIMD `apply_reg` fast path, exercising each ISA's own `max`/`min`
/// (`_mm512_max_ps`, `vmaxnmq_f32`, `f32x4_pmax`, ...), not only the scalar `apply` fallback
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
    // Force a NaN accumulator via inf + (-inf): column 0 of A times row 0 of B gives +inf,
    // column 1 times row 1 gives -inf, and the 2 sum to NaN. Using literal infinities (rather
    // than e.g. MAX*MAX) keeps this robust to FMA: an unfused MAX*MAX could overflow to inf on
    // its own before the add, while a fused one might not, changing which path reaches NaN
    let mut a = vec![T::of(0.0); m * k];
    let mut b = vec![T::of(0.0); k * n];
    for i in 0..m {
        a[i] = T::of(f64::INFINITY); // column 0
        a[m + i] = T::of(f64::INFINITY); // column 1
    }
    for j in 0..n {
        b[k * j] = T::of(1.0); // row 0: contributes +inf
        b[k * j + 1] = T::of(-1.0); // row 1: contributes -inf
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

    // A zero A*B product run through a negative-slope LeakyReLU must still yield +0, not the
    // -0 a naive slope*0 multiply could produce
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

// degenerate cases: k == 0 or alpha == 0 collapses the A*B term to C <- act(beta*C + bias)

#[test]
fn fused_degenerate() {
    let mut rng = Rng::new(0xDE6E);
    for &(m, n) in &[(20usize, 24usize)] {
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
            // reference: C = ReLU(beta*C0 + bias[i]), beta = 0.5 here
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

// validation panics

mod validation {
    use super::*;

    /// A 4x4 all-ones A/B and a zeroed C, the common fixture the panic cases below start from
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
        let bias = vec![0.0f32; 3]; // PerRow needs length 4 (m), not 3
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
        // A bias slice built from a raw pointer into buf, so it has no lifetime tied to `buf`
        // and `&mut buf` below still type-checks despite the aliasing. gemm_fused's overlap
        // check panics before any element is read or written, so nothing is actually aliased
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

// checked/unchecked twin equivalence

/// `gemm_fused` and `gemm_fused_unchecked` are **parallel** entry points (the checked one does
/// not call the unchecked one internally), so a divergence in the `Bias`/`Activation` lowering to
/// raw pointers would otherwise go undetected. Drives both on the same driver-shaped problem
/// (`m, n, k > 16`), bit-for-bit, across both `BiasDim` arms, `has_bias = false`, and every
/// activation arm
#[test]
fn fused_unchecked_matches_checked() {
    use gemmkit::{BiasDim, gemm_fused, gemm_fused_unchecked};

    let mut rng = Rng::new(0x0F05_ED12);
    let (m, k, n) = (33usize, 24usize, 40usize);
    let a = make::<f32>(&mut rng, m, k); // col-major m x k
    let b = make::<f32>(&mut rng, k, n); // col-major k x n
    let c0 = make::<f32>(&mut rng, m * n, 1); // col-major m x n C
    let bias_row: Vec<f32> = (0..m).map(|_| (rng.unit() * 3.0) as f32).collect();
    let bias_col: Vec<f32> = (0..n).map(|_| (rng.unit() * 3.0) as f32).collect();
    let (alpha, beta) = (0.9f32, 0.7f32);
    let par = Parallelism::Serial;

    let mk_act = |kind: u8| match kind {
        1 => Some(Activation::Relu),
        2 => Some(Activation::LeakyRelu(0.1f32)),
        _ => None,
    };

    // bias_kind: 0 = none, 1 = per-row, 2 = per-col
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
            // SAFETY: every view is a valid in-bounds col-major layout, C aliases neither A nor
            // B nor the bias, and any present bias slice is the right length for its axis
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

/// `gemm_fused_unchecked_with` takes the same raw lowering as `gemm_fused_unchecked` but drives a
/// caller-owned `Workspace` instead of the pool; drives it against the checked `gemm_fused_with`
/// twin bit-for-bit (per-row bias plus LeakyReLU on a driver shape)
#[test]
fn fused_unchecked_with_matches_checked() {
    use gemmkit::{BiasDim, gemm_fused_unchecked_with, gemm_fused_with};

    let mut rng = Rng::new(0x0F05_ED13);
    let (m, k, n) = (40usize, 33usize, 24usize);
    let a = make::<f64>(&mut rng, m, k); // col-major m x k
    let b = make::<f64>(&mut rng, k, n); // col-major k x n
    let c0 = make::<f64>(&mut rng, m * n, 1); // col-major m x n C
    let bias_row: Vec<f64> = (0..m).map(|_| rng.unit() * 3.0).collect();
    let (alpha, beta) = (0.9f64, 0.7f64);
    let par = Parallelism::Serial;

    let mut c_checked = c0.clone();
    {
        let mut ws = Workspace::new();
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_checked, m, n, 1, m as isize);
        gemm_fused_with(
            &mut ws,
            alpha,
            ar,
            br,
            beta,
            cm,
            Some(Bias::PerRow(&bias_row)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
    }

    let mut c_unchecked = c0.clone();
    let mut ws = Workspace::new();
    // SAFETY: valid in-bounds col-major layouts; C aliases neither A nor B nor the per-row,
    // length-m bias
    unsafe {
        gemm_fused_unchecked_with(
            &mut ws,
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
            bias_row.as_ptr(),
            BiasDim::PerRow,
            true,
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
    }

    for idx in 0..m * n {
        assert_eq!(
            c_checked[idx].to_bits(),
            c_unchecked[idx].to_bits(),
            "fused unchecked_with != checked at {idx}",
        );
    }
}
