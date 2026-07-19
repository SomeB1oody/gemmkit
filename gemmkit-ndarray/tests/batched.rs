//! Strided-batched ndarray adapter (`gemm_batched` / `dot_batched`, batch on axis 0): the 3-D
//! `Array3` form must reproduce a per-element loop of `dot` / `gemm`, read general-stride and
//! permuted-axes 3-D views straight through without copying, stay serial==parallel reproducible,
//! and (under `epilogue`) match a loop of `gemm_fused` for one shared bias/activation bit-for-bit

use approx::assert_relative_eq;
use ndarray::{Array3, Axis};

use gemmkit::Parallelism;
use gemmkit_ndarray::{dot_batched, gemm_batched};

fn rand3(b: usize, r: usize, c: usize, seed: u64) -> Array3<f64> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Array3::from_shape_fn((b, r, c), |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

#[test]
fn dot_batched_matches_per_element_dot() {
    let (batch, m, k, n) = (5usize, 7, 9, 6);
    let a = rand3(batch, m, k, 11);
    let b = rand3(batch, k, n, 22);
    let got = dot_batched(&a, &b);
    for e in 0..batch {
        let ae = a.index_axis(Axis(0), e);
        let be = b.index_axis(Axis(0), e);
        assert_relative_eq!(
            got.index_axis(Axis(0), e).to_owned(),
            ae.dot(&be),
            max_relative = 1e-10
        );
    }
}

#[test]
fn gemm_batched_matches_per_element_loop() {
    let (batch, m, k, n) = (4usize, 8, 5, 6);
    let a = rand3(batch, m, k, 31);
    let b = rand3(batch, k, n, 32);
    let c0 = rand3(batch, m, n, 33);
    let (alpha, beta) = (0.7f64, 1.3);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c = c0.clone();
        gemm_batched(alpha, &a, &b, beta, &mut c, par);
        for e in 0..batch {
            let ae = a.index_axis(Axis(0), e);
            let be = b.index_axis(Axis(0), e);
            let exp = alpha * &ae.dot(&be) + beta * &c0.index_axis(Axis(0), e).to_owned();
            assert_relative_eq!(
                c.index_axis(Axis(0), e).to_owned(),
                exp,
                max_relative = 1e-10
            );
        }
    }
}

/// Batched over a permuted-axes (non-contiguous) 3-D view: strides read straight through, no copy
#[test]
fn gemm_batched_permuted_axes_view() {
    let (batch, m, k, n) = (3usize, 6, 4, 5);
    let araw = rand3(batch, k, m, 41); // (batch, k, m)
    let a = araw.view().permuted_axes([0, 2, 1]); // (batch, m, k), strided view
    let b = rand3(batch, k, n, 42);
    let got = dot_batched(&a, &b);
    for e in 0..batch {
        let ae = a.index_axis(Axis(0), e);
        let be = b.index_axis(Axis(0), e);
        assert_relative_eq!(
            got.index_axis(Axis(0), e).to_owned(),
            ae.dot(&be),
            max_relative = 1e-10
        );
    }
}

/// `gemm_batched_fused` (one shared `PerRow` bias + `ReLU`) is **bit-identical** to a loop of
/// [`gemm_fused`] calls, one per batch element: gemmkit's batched-fused headline property
#[cfg(feature = "epilogue")]
#[test]
fn batched_fused_matches_loop_of_gemm_fused() {
    use gemmkit_ndarray::{Activation, Bias, gemm_batched_fused, gemm_fused};
    let (batch, m, k, n) = (4usize, 8, 5, 6);
    let a = rand3(batch, m, k, 201);
    let b = rand3(batch, k, n, 202);
    let c0 = rand3(batch, m, n, 203);
    let bias: Vec<f64> = (0..m).map(|i| 0.3 * i as f64 - 1.0).collect();
    let (alpha, beta) = (0.9f64, 1.1);
    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
        let mut c_b = c0.clone();
        gemm_batched_fused(
            alpha,
            &a,
            &b,
            beta,
            &mut c_b,
            Some(Bias::PerRow(&bias)),
            Some(Activation::Relu),
            par,
        );
        for e in 0..batch {
            let ae = a.index_axis(Axis(0), e);
            let be = b.index_axis(Axis(0), e);
            let mut ce = c0.index_axis(Axis(0), e).to_owned();
            gemm_fused(
                alpha,
                &ae,
                &be,
                beta,
                &mut ce,
                Some(Bias::PerRow(&bias)),
                Some(Activation::Relu),
                par,
            );
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        c_b[(e, i, j)].to_bits(),
                        ce[(i, j)].to_bits(),
                        "batched_fused ({e},{i},{j})"
                    );
                }
            }
        }
    }
}
