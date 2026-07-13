//! Fused-epilogue special-path suite: `gemm_fused` routes gemv (`m == 1` / `n == 1`), small-`m,n`,
//! and small-`k` shapes through the **same** kernels plain `gemm` uses, fusing the epilogue into
//! each route — so the fused result is bit-identical to `gemm`-then-map for those shapes too (the
//! Phase-1 contract). Every comparison is **bitwise**; all shapes are platform-independent
//! (self-computed references, LCG-style fills). The routes are also serial == parallel
//! bit-identical, and the fusion preserves that.

use crate::common::*;
use gemmkit::{Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_fused_with};

/// Drive `gemm_fused` over an explicit A/B/C layout (arbitrary strides, so a route-selecting
/// layout — a row-major A, a column-major B — can be forced) and assert bitwise equality with
/// `gemm`-then-`ref_apply`, across the full bias × activation sweep and `beta ∈ {0, 0.7}`. The
/// reference reads back plain `gemm`'s output and maps it with the exact `FusedEpi` mirror
/// ([`ref_apply`]), so any divergence in the route's store or its fused map is caught bit-for-bit.
fn check_route<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    rsa: isize,
    csa: isize,
    b: &[T],
    rsb: isize,
    csb: isize,
    layout: Layout,
    par: Parallelism,
    tag: &str,
) {
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen, 1);
    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.25))),
    ];
    let alpha = T::of(1.0);

    for &beta in &[T::ZERO, T::of(0.7)] {
        for bias_kind in 0u8..=2 {
            for act in &acts {
                let bias = match bias_kind {
                    1 => Some(Bias::PerRow(&bias_row)),
                    2 => Some(Bias::PerCol(&bias_col)),
                    _ => None,
                };

                let mut c_fused = c0.clone();
                let mut ws = Workspace::new();
                {
                    let ar = MatRef::new(a, m, k, rsa, csa);
                    let br = MatRef::new(b, k, n, rsb, csb);
                    let cm = MatMut::new(&mut c_fused, m, n, rsc, csc);
                    gemm_fused_with(
                        &mut ws,
                        alpha,
                        ar,
                        br,
                        beta,
                        cm,
                        bias,
                        act.clone_like(),
                        par,
                    );
                }

                let mut c_ref = c0.clone();
                {
                    let ar = MatRef::new(a, m, k, rsa, csa);
                    let br = MatRef::new(b, k, n, rsb, csb);
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
                        let want = ref_apply(c_ref[idx], bterm, act);
                        assert_eq!(
                            c_fused[idx].bits(),
                            want.bits(),
                            "{} {tag}: fused != gemm-then-map at ({i},{j}) \
                             [m={m} k={k} n={n} bias_kind={bias_kind} beta={:016x}]",
                            T::name(),
                            beta.bits(),
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// a. gemv route (m == 1 / n == 1)
// ---------------------------------------------------------------------------

/// The gemv route: both orientations of the vector, both C layouts, full bias × activation sweep.
/// `min(m, n) == 1 <= gemv_threshold`, so these hit the dedicated gemv kernel, whose output is
/// fused via a final in-place epilogue pass (in the *user* frame — gemv dispatches before
/// orientation normalization).
fn gemv_bitwise_for<T: Flt>() {
    let mut rng = Rng::new(0x9E_11_0C_A5);
    for &(m, k, n) in &[(1usize, 64usize, 300usize), (300usize, 64usize, 1usize)] {
        let a = make::<T>(&mut rng, m, k); // col-major m×k
        let b = make::<T>(&mut rng, k, n); // col-major k×n
        for layout in [Layout::Col, Layout::Row] {
            check_route::<T>(
                &mut rng,
                m,
                k,
                n,
                &a,
                1,
                m as isize, // col-major A
                &b,
                1,
                k as isize, // col-major B
                layout,
                Parallelism::Serial,
                "gemv",
            );
        }
    }
}

#[test]
fn fused_gemv_bitwise() {
    gemv_bitwise_for::<f32>();
    gemv_bitwise_for::<f64>();
}

// ---------------------------------------------------------------------------
// b. small-m,n horizontal route
// ---------------------------------------------------------------------------

/// The small-`m,n` route: `(8, 2048, 8)` with a row-major A (`csa == 1`) and a column-major B
/// (`rsb == 1`), so the horizontal `small_mn` gate is hit (both dims <= 16, `k` above
/// `small_k_threshold`, `csa == 1 && rsb == 1`). Both C layouts (the row-major C exercises the
/// orientation swap, which flips the bias axis and still routes to `small_mn` since the swapped
/// `csa`/`rsb` stay unit-stride). Full bias × activation sweep, bitwise.
fn small_mn_bitwise_for<T: Flt>() {
    let mut rng = Rng::new(0x5A_11_3E_00);
    let (m, k, n) = (8usize, 2048usize, 8usize);
    let a = make::<T>(&mut rng, m, k);
    let b = make::<T>(&mut rng, k, n);
    for layout in [Layout::Col, Layout::Row] {
        check_route::<T>(
            &mut rng,
            m,
            k,
            n,
            &a,
            k as isize,
            1, // row-major A (csa == 1)
            &b,
            1,
            k as isize, // column-major B (rsb == 1)
            layout,
            Parallelism::Serial,
            "small_mn",
        );
    }
}

#[test]
fn fused_small_mn_bitwise() {
    small_mn_bitwise_for::<f32>();
    small_mn_bitwise_for::<f64>();
}

// ---------------------------------------------------------------------------
// c. small-k route
// ---------------------------------------------------------------------------

/// The small-`k` route: `(200, 4, 160)` with a column-major A (`rsa == 1`, so the in-place route
/// applies for a column-major C; the row-major-C swap makes the oriented `rsa != 1`, deferring to
/// the driver — still fused, still bitwise). Both C layouts, full sweep, bitwise.
fn small_k_bitwise_for<T: Flt>() {
    let mut rng = Rng::new(0x5A_11_C4_00);
    let (m, k, n) = (200usize, 4usize, 160usize);
    let a = make::<T>(&mut rng, m, k);
    let b = make::<T>(&mut rng, k, n);
    for layout in [Layout::Col, Layout::Row] {
        check_route::<T>(
            &mut rng,
            m,
            k,
            n,
            &a,
            1,
            m as isize, // col-major A (rsa == 1)
            &b,
            1,
            k as isize, // col-major B
            layout,
            Parallelism::Serial,
            "small_k",
        );
    }
}

#[test]
fn fused_small_k_bitwise() {
    small_k_bitwise_for::<f32>();
    small_k_bitwise_for::<f64>();
}

// ---------------------------------------------------------------------------
// d. serial == parallel bit-identity through the fused special routes
// ---------------------------------------------------------------------------

/// Run one fused config (PerRow bias + LeakyReLU, `beta = 0.7`) under `Serial` and `Rayon(4)` and
/// assert the two output buffers agree bitwise. The special routes partition the output with no
/// cross-thread reduction, and the fused epilogue is a per-range pass, so the two are bit-identical.
fn par_eq<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    rsa: isize,
    csa: isize,
    b: &[T],
    rsb: isize,
    csb: isize,
    tag: &str,
) {
    let (rsc, csc, clen) = c_strides(Layout::Col, m, n);
    let c0 = make::<T>(rng, clen, 1);
    let bias: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();

    let run = |par: Parallelism| -> Vec<T> {
        let mut c = c0.clone();
        let mut ws = Workspace::new();
        let ar = MatRef::new(a, m, k, rsa, csa);
        let br = MatRef::new(b, k, n, rsb, csb);
        let cm = MatMut::new(&mut c, m, n, rsc, csc);
        gemm_fused_with(
            &mut ws,
            T::of(1.0),
            ar,
            br,
            T::of(0.7),
            cm,
            Some(Bias::PerRow(&bias)),
            Some(Activation::LeakyRelu(T::of(0.25))),
            par,
        );
        c
    };

    let cs = run(Parallelism::Serial);
    let cp = run(Parallelism::Rayon(4));
    for idx in 0..clen {
        assert_eq!(
            cs[idx].bits(),
            cp[idx].bits(),
            "{} {tag}: serial != parallel at {idx}",
            T::name(),
        );
    }
}

fn special_parallel_for<T: Flt>() {
    let mut rng = Rng::new(0x9A_4A_11_E1);
    // gemv
    {
        let (m, k, n) = (1usize, 64usize, 300usize);
        let a = make::<T>(&mut rng, m, k);
        let b = make::<T>(&mut rng, k, n);
        par_eq::<T>(
            &mut rng, m, k, n, &a, 1, m as isize, &b, 1, k as isize, "gemv-par",
        );
    }
    // small_mn (row-major A, col-major B)
    {
        let (m, k, n) = (8usize, 2048usize, 8usize);
        let a = make::<T>(&mut rng, m, k);
        let b = make::<T>(&mut rng, k, n);
        par_eq::<T>(
            &mut rng,
            m,
            k,
            n,
            &a,
            k as isize,
            1,
            &b,
            1,
            k as isize,
            "small_mn-par",
        );
    }
    // small_k (col-major A)
    {
        let (m, k, n) = (200usize, 4usize, 160usize);
        let a = make::<T>(&mut rng, m, k);
        let b = make::<T>(&mut rng, k, n);
        par_eq::<T>(
            &mut rng,
            m,
            k,
            n,
            &a,
            1,
            m as isize,
            &b,
            1,
            k as isize,
            "small_k-par",
        );
    }
}

#[test]
fn fused_special_parallel_bitwise() {
    special_parallel_for::<f32>();
    special_parallel_for::<f64>();
}

// ---------------------------------------------------------------------------
// e. NaN through the special routes: ReLU(NaN) == +0.0, bitwise vs gemm-then-map
// ---------------------------------------------------------------------------

/// A NaN in the first depth column of every A row makes each output's `k`-reduction a NaN, which
/// ReLU must map to `+0.0` — bit-for-bit, and equal to `gemm`-then-map. Exercises the gemv and
/// small_mn routes (whose fused epilogue runs the scalar `apply`, matching `ref_apply`).
fn nan_route<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    mut a: Vec<T>,
    rsa: isize,
    csa: isize,
    b: &[T],
    rsb: isize,
    csb: isize,
    tag: &str,
) {
    // A[i, 0] lives at `i*rsa + 0*csa`; NaN there poisons the whole row's reduction.
    for i in 0..m {
        a[(i as isize * rsa) as usize] = T::of(f64::NAN);
    }
    let (rsc, csc, clen) = c_strides(Layout::Col, m, n);
    let c0 = make::<T>(rng, clen, 1);

    let mut c_fused = c0.clone();
    let mut ws = Workspace::new();
    {
        let ar = MatRef::new(&a, m, k, rsa, csa);
        let br = MatRef::new(b, k, n, rsb, csb);
        let cm = MatMut::new(&mut c_fused, m, n, rsc, csc);
        gemm_fused_with(
            &mut ws,
            T::of(1.0),
            ar,
            br,
            T::ZERO,
            cm,
            None,
            Some(Activation::Relu),
            Parallelism::Serial,
        );
    }

    let mut c_ref = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, rsa, csa);
        let br = MatRef::new(b, k, n, rsb, csb);
        let cm = MatMut::new(&mut c_ref, m, n, rsc, csc);
        gemm(T::of(1.0), ar, br, T::ZERO, cm, Parallelism::Serial);
    }

    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let want = ref_apply(c_ref[idx], None, &Some(Activation::Relu));
            assert_eq!(
                c_fused[idx].bits(),
                want.bits(),
                "{} {tag}: fused != gemm-then-map at ({i},{j})",
                T::name(),
            );
            assert_eq!(
                c_fused[idx].bits(),
                T::of(0.0).bits(),
                "{} {tag}: ReLU(NaN) must be +0.0 at ({i},{j})",
                T::name(),
            );
        }
    }
}

fn special_nan_relu_for<T: Flt>() {
    let mut rng = Rng::new(0x4A_11_DE_AD);
    // gemv (1, 64, 300): col-major A/B.
    {
        let (m, k, n) = (1usize, 64usize, 300usize);
        let a = make::<T>(&mut rng, m, k);
        let b = make::<T>(&mut rng, k, n);
        nan_route::<T>(
            &mut rng, m, k, n, a, 1, m as isize, &b, 1, k as isize, "gemv-nan",
        );
    }
    // small_mn (8, 2048, 8): row-major A, col-major B.
    {
        let (m, k, n) = (8usize, 2048usize, 8usize);
        let a = make::<T>(&mut rng, m, k);
        let b = make::<T>(&mut rng, k, n);
        nan_route::<T>(
            &mut rng,
            m,
            k,
            n,
            a,
            k as isize,
            1,
            &b,
            1,
            k as isize,
            "small_mn-nan",
        );
    }
}

#[test]
fn fused_special_nan_relu() {
    special_nan_relu_for::<f32>();
    special_nan_relu_for::<f64>();
}
