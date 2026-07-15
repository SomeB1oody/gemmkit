//! Prepacked-operand fused-epilogue tests: the core `packed-fused == packed-then-map` oracle for
//! both `gemm_packed_b_fused` (RHS prepacked, column-major-ish C) and `gemm_packed_a_fused` (LHS
//! prepacked, row-major-ish C), the degenerate cases (`alpha == 0` / `k == 0`), handle reuse across
//! plain + fused calls, the no-swap orientation panics, the checked/allocating twin equivalence,
//! and the narrow (`f16`/`bf16`) pre-narrow contract
//!
//! For `f32`/`f64` every comparison against the plain packed entry then the same scalar map is
//! **bitwise**: the epilogue is store-side only, riding the *same* prepacked kernel `gemm_packed_*`
//! runs (identical blocking / packing / scheduling), applied to the very register the plain store
//! would write. The narrow path applies the epilogue in `f32` before the single narrowing, so it is
//! locked against an `f32` single-rounding reference (a per-element gate for the general shape, and
//! a bitwise `k == 1` case), not the 2-rounding narrow gemm-then-map

use crate::common::*;
use gemmkit::{
    Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm_packed_a, gemm_packed_a_fused,
    gemm_packed_a_fused_with, gemm_packed_b, gemm_packed_b_fused, gemm_packed_b_fused_with,
    prepack_lhs, prepack_rhs,
};

// column-major-ish C layouts for the RHS-packed path (|csc| >= |rsc|)
#[derive(Copy, Clone)]
enum ColC {
    /// Contiguous column-major (rsc = 1, csc = m)
    Col,
    /// Column-major with a padded column stride (rsc = 1, csc = m + 3): a strided C forcing the
    /// scratch path at tile edges
    ColPadded,
}
fn col_c_strides(layout: ColC, m: usize, n: usize) -> (isize, isize, usize) {
    match layout {
        ColC::Col => (1, m as isize, m * n),
        ColC::ColPadded => (1, (m + 3) as isize, (m + 3) * n),
    }
}

// row-major-ish C layouts for the LHS-packed path (|csc| <= |rsc|)
#[derive(Copy, Clone)]
enum RowC {
    /// Contiguous row-major (rsc = n, csc = 1)
    Row,
    /// Row-major with a padded row stride (rsc = n + 3, csc = 1): a strided C
    RowPadded,
}
fn row_c_strides(layout: RowC, m: usize, n: usize) -> (isize, isize, usize) {
    match layout {
        RowC::Row => (n as isize, 1, m * n),
        RowC::RowPadded => ((n + 3) as isize, 1, m * (n + 3)),
    }
}

/// 1 RHS-packed fused case and its `gemm_packed_b`-then-map oracle; assert bitwise-equal C over
/// the whole (possibly strided) output. The same prepacked handle drives both
#[allow(clippy::too_many_arguments)]
fn check_packed_b_fused<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    rsc: isize,
    csc: isize,
    clen: usize,
    bias_kind: u8, // 0 none, 1 per-row, 2 per-col
    act: Option<Activation<T>>,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k.max(1)); // col-major mxk (k.max(1) keeps a real buffer at k == 0)
    let b = make::<T>(rng, k.max(1), n); // col-major kxn
    let c0 = make::<T>(rng, clen, 1);
    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias = match bias_kind {
        1 => Some(Bias::PerRow(&bias_row)),
        2 => Some(Bias::PerCol(&bias_col)),
        _ => None,
    };

    let bv = MatRef::new(&b, k, n, 1, k.max(1) as isize);
    let packed = prepack_rhs(bv);
    let av = MatRef::new(&a, m, k, 1, m as isize);

    // fused
    let mut c_fused = c0.clone();
    let mut ws = Workspace::new();
    gemm_packed_b_fused_with(
        &mut ws,
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_fused, m, n, rsc, csc),
        bias,
        act.clone_like(),
        par,
    );

    // oracle: plain packed (same handle) then the scalar map in the user frame
    let mut c_ref = c0.clone();
    gemm_packed_b(
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_ref, m, n, rsc, csc),
        par,
    );
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let bterm = match bias_kind {
                1 => Some(bias_row[i]),
                2 => Some(bias_col[j]),
                _ => None,
            };
            c_ref[idx] = ref_apply(c_ref[idx], bterm, &act);
        }
    }
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            assert_eq!(
                c_fused[idx].bits(),
                c_ref[idx].bits(),
                "{} {tag}: packed_b_fused != packed_b-then-map at ({i},{j}) [m={m} k={k} n={n}]",
                T::name(),
            );
        }
    }
}

/// 1 LHS-packed fused case and its `gemm_packed_a`-then-map oracle; assert bitwise-equal. Exercises
/// the user-frame bias axis through the packed-A transpose (a wrong flip diverges loudly)
#[allow(clippy::too_many_arguments)]
fn check_packed_a_fused<T: Flt>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    rsc: isize,
    csc: isize,
    clen: usize,
    bias_kind: u8,
    act: Option<Activation<T>>,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k.max(1)); // col-major mxk
    let b = make::<T>(rng, k.max(1), n); // col-major kxn
    let c0 = make::<T>(rng, clen, 1);
    let bias_row: Vec<T> = (0..m).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias_col: Vec<T> = (0..n).map(|_| T::of(rng.unit() * 3.0)).collect();
    let bias = match bias_kind {
        1 => Some(Bias::PerRow(&bias_row)),
        2 => Some(Bias::PerCol(&bias_col)),
        _ => None,
    };

    let av = MatRef::new(&a, m, k, 1, m as isize);
    let packed = prepack_lhs(av);
    let bv = MatRef::new(&b, k, n, 1, k.max(1) as isize);

    let mut c_fused = c0.clone();
    let mut ws = Workspace::new();
    gemm_packed_a_fused_with(
        &mut ws,
        alpha,
        &packed,
        bv,
        beta,
        MatMut::new(&mut c_fused, m, n, rsc, csc),
        bias,
        act.clone_like(),
        par,
    );

    let mut c_ref = c0.clone();
    gemm_packed_a(
        alpha,
        &packed,
        bv,
        beta,
        MatMut::new(&mut c_ref, m, n, rsc, csc),
        par,
    );
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let bterm = match bias_kind {
                1 => Some(bias_row[i]),
                2 => Some(bias_col[j]),
                _ => None,
            };
            c_ref[idx] = ref_apply(c_ref[idx], bterm, &act);
        }
    }
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            assert_eq!(
                c_fused[idx].bits(),
                c_ref[idx].bits(),
                "{} {tag}: packed_a_fused != packed_a-then-map at ({i},{j}) [m={m} k={k} n={n}]",
                T::name(),
            );
        }
    }
}

// core oracle sweep: shapes (incl. non-tile-multiples), beta {0, 1, 0.7}, bias {none/row/col},
// act {none/relu/leaky}, strided C

fn packed_b_matrix<T: Flt>(par: Parallelism) {
    let mut rng = Rng::new(0x9ACB_11B0);
    let shapes = [(200usize, 130usize, 175usize), (65, 64, 64), (40, 200, 129)];
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.1))),
    ];
    for &(m, k, n) in &shapes {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [ColC::Col, ColC::ColPadded] {
                    let (rsc, csc, clen) = col_c_strides(layout, m, n);
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            check_packed_b_fused::<T>(
                                &mut rng,
                                m,
                                k,
                                n,
                                alpha,
                                beta,
                                rsc,
                                csc,
                                clen,
                                bias_kind,
                                act.clone_like(),
                                par,
                                "b/matrix",
                            );
                        }
                    }
                }
            }
        }
    }
}

fn packed_a_matrix<T: Flt>(par: Parallelism) {
    let mut rng = Rng::new(0x5DA1_00A5);
    let shapes = [(200usize, 130usize, 175usize), (65, 64, 64), (40, 200, 129)];
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.1))),
    ];
    for &(m, k, n) in &shapes {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [RowC::Row, RowC::RowPadded] {
                    let (rsc, csc, clen) = row_c_strides(layout, m, n);
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            check_packed_a_fused::<T>(
                                &mut rng,
                                m,
                                k,
                                n,
                                alpha,
                                beta,
                                rsc,
                                csc,
                                clen,
                                bias_kind,
                                act.clone_like(),
                                par,
                                "a/matrix",
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packed_fused_eq_packed_then_map_serial() {
    packed_b_matrix::<f32>(Parallelism::Serial);
    packed_b_matrix::<f64>(Parallelism::Serial);
    packed_a_matrix::<f32>(Parallelism::Serial);
    packed_a_matrix::<f64>(Parallelism::Serial);
}

#[test]
fn packed_fused_eq_packed_then_map_parallel() {
    packed_b_matrix::<f32>(Parallelism::Rayon(8));
    packed_b_matrix::<f64>(Parallelism::Rayon(8));
    packed_a_matrix::<f32>(Parallelism::Rayon(8));
    packed_a_matrix::<f64>(Parallelism::Rayon(8));
}

// degenerate cases: alpha == 0 (k > 0) and k == 0 => C <- act(beta*C + bias), via the same
// packed-then-map oracle (plain packed also beta-scales in the degenerate)

#[test]
fn packed_fused_degenerate() {
    let mut rng = Rng::new(0xDE6E_9AC0);
    let (m, n) = (48usize, 40usize);
    for &(k, alpha) in &[(0usize, 1.0f32), (130usize, 0.0f32)] {
        // RHS-packed, col-major C, per-row bias + ReLU
        let (rsc, csc, clen) = col_c_strides(ColC::ColPadded, m, n);
        check_packed_b_fused::<f32>(
            &mut rng,
            m,
            k,
            n,
            alpha,
            0.5,
            rsc,
            csc,
            clen,
            1,
            Some(Activation::Relu),
            Parallelism::Serial,
            "b/degenerate",
        );
        check_packed_b_fused::<f32>(
            &mut rng,
            m,
            k,
            n,
            alpha,
            0.5,
            rsc,
            csc,
            clen,
            2,
            Some(Activation::LeakyRelu(0.1)),
            Parallelism::Rayon(4),
            "b/degenerate",
        );
        // LHS-packed, row-major C, per-row + per-col bias (exercises the flipped degenerate axis)
        let (rsc, csc, clen) = row_c_strides(RowC::RowPadded, m, n);
        check_packed_a_fused::<f32>(
            &mut rng,
            m,
            k,
            n,
            alpha,
            0.5,
            rsc,
            csc,
            clen,
            1,
            Some(Activation::Relu),
            Parallelism::Serial,
            "a/degenerate",
        );
        check_packed_a_fused::<f64>(
            &mut rng,
            m,
            k,
            n,
            alpha as f64,
            0.5,
            rsc,
            csc,
            clen,
            2,
            None,
            Parallelism::Rayon(4),
            "a/degenerate",
        );
    }
}

// 1 prepacked handle reused across multiple fused calls AND mixed plain + fused calls: each
// result is independent (no shared mutable state; the epilogue is store-side only)

#[test]
fn packed_handle_reused_across_plain_and_fused() {
    let mut rng = Rng::new(0x0AC1_5EED);
    let (m, k, n) = (128usize, 96usize, 100usize);
    let a = make::<f32>(&mut rng, m, k);
    let b = make::<f32>(&mut rng, k, n);
    let c0 = make::<f32>(&mut rng, m * n, 1);
    let bias1: Vec<f32> = (0..m).map(|_| (rng.unit() * 2.0) as f32).collect();
    let bias2: Vec<f32> = (0..n).map(|_| (rng.unit() * 2.0) as f32).collect();
    let (alpha, beta) = (0.9f32, 0.7f32);
    let par = Parallelism::Rayon(4);

    let av = MatRef::new(&a, m, k, 1, m as isize);
    let bv = MatRef::new(&b, k, n, 1, k as isize);
    let packed = prepack_rhs(bv); // ONE handle, reused below

    // 3 products off the SAME handle: fused(PerRow+ReLU), plain, fused(PerCol+Leaky)
    let mut c_f1 = c0.clone();
    gemm_packed_b_fused(
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_f1, m, n, 1, m as isize),
        Some(Bias::PerRow(&bias1)),
        Some(Activation::Relu),
        par,
    );
    let mut c_plain = c0.clone();
    gemm_packed_b(
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_plain, m, n, 1, m as isize),
        par,
    );
    let mut c_f2 = c0.clone();
    gemm_packed_b_fused(
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_f2, m, n, 1, m as isize),
        Some(Bias::PerCol(&bias2)),
        Some(Activation::LeakyRelu(0.25)),
        par,
    );

    // Independent oracles: each fused equals plain-then-its-own-map; plain is untouched by the fused
    for j in 0..n {
        for i in 0..m {
            let idx = i + j * m;
            let base = c_plain[idx];
            let want1 = ref_apply(base, Some(bias1[i]), &Some(Activation::Relu));
            let want2 = ref_apply(base, Some(bias2[j]), &Some(Activation::LeakyRelu(0.25)));
            assert_eq!(c_f1[idx].to_bits(), want1.to_bits(), "reuse f1 ({i},{j})");
            assert_eq!(c_f2[idx].to_bits(), want2.to_bits(), "reuse f2 ({i},{j})");
        }
    }
}

// no-swap orientation panics still fire for the fused entries, with the plain-packed wording

#[test]
#[should_panic(expected = "column-major-ish C")]
fn packed_b_fused_row_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = prepack_rhs(MatRef::from_col_major(&b, k, n));
    gemm_packed_b_fused(
        1.0,
        MatRef::from_col_major(&a, m, k),
        &packed,
        0.0,
        MatMut::from_row_major(&mut c, m, n), // row-major C -> swap -> reject
        None,
        Some(Activation::Relu),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "row-major-ish C")]
fn packed_a_fused_col_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = prepack_lhs(MatRef::from_col_major(&a, m, k));
    gemm_packed_a_fused(
        1.0,
        &packed,
        MatRef::from_col_major(&b, k, n),
        0.0,
        MatMut::from_col_major(&mut c, m, n), // column-major C -> reject
        None,
        Some(Activation::Relu),
        Parallelism::Serial,
    );
}

// _with (caller-owned Workspace) vs the allocating entry: bit-identical

#[test]
fn packed_fused_with_matches_allocating() {
    let mut rng = Rng::new(0x00_1234);
    let (m, k, n) = (129usize, 96usize, 72usize);
    let a = make::<f32>(&mut rng, m, k);
    let b = make::<f32>(&mut rng, k, n);
    let c0 = make::<f32>(&mut rng, m * n, 1);
    let bias: Vec<f32> = (0..m).map(|_| (rng.unit() * 2.0) as f32).collect();
    let (alpha, beta) = (0.9f32, 0.7f32);
    let par = Parallelism::Serial;
    let av = MatRef::new(&a, m, k, 1, m as isize);
    let bv = MatRef::new(&b, k, n, 1, k as isize);

    // RHS-packed
    {
        let packed = prepack_rhs(bv);
        let mut c_alloc = c0.clone();
        gemm_packed_b_fused(
            alpha,
            av,
            &packed,
            beta,
            MatMut::new(&mut c_alloc, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_packed_b_fused_with(
            &mut ws,
            alpha,
            av,
            &packed,
            beta,
            MatMut::new(&mut c_with, m, n, 1, m as isize),
            Some(Bias::PerRow(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        for idx in 0..m * n {
            assert_eq!(
                c_alloc[idx].to_bits(),
                c_with[idx].to_bits(),
                "b_fused_with != b_fused at {idx}"
            );
        }
    }
    // LHS-packed (row-major C)
    {
        let packed = prepack_lhs(av);
        let mut c_alloc = c0.clone();
        gemm_packed_a_fused(
            alpha,
            &packed,
            bv,
            beta,
            MatMut::new(&mut c_alloc, m, n, n as isize, 1),
            Some(Bias::PerRow(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        let mut c_with = c0.clone();
        let mut ws = Workspace::new();
        gemm_packed_a_fused_with(
            &mut ws,
            alpha,
            &packed,
            bv,
            beta,
            MatMut::new(&mut c_with, m, n, n as isize, 1),
            Some(Bias::PerRow(&bias)),
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
        for idx in 0..m * n {
            assert_eq!(
                c_alloc[idx].to_bits(),
                c_with[idx].to_bits(),
                "a_fused_with != a_fused at {idx}"
            );
        }
    }
}

// unchecked twin equivalence: the raw fused packed entries drive the same result as the checked
// ones (the checked entry does not delegate to the unchecked one, so a Bias/Act translation drift
// would go undetected)

#[test]
fn packed_b_fused_unchecked_matches_checked() {
    use gemmkit::{BiasDim, gemm_packed_b_fused_unchecked};
    let mut rng = Rng::new(0x0B_ED12);
    let (m, k, n) = (129usize, 96usize, 72usize);
    let a = make::<f32>(&mut rng, m, k);
    let b = make::<f32>(&mut rng, k, n);
    let c0 = make::<f32>(&mut rng, m * n, 1);
    let bias_col: Vec<f32> = (0..n).map(|_| (rng.unit() * 3.0) as f32).collect();
    let (alpha, beta) = (0.9f32, 0.7f32);
    let par = Parallelism::Serial;
    let av = MatRef::new(&a, m, k, 1, m as isize);
    let packed = prepack_rhs(MatRef::new(&b, k, n, 1, k as isize));

    let mut c_checked = c0.clone();
    gemm_packed_b_fused(
        alpha,
        av,
        &packed,
        beta,
        MatMut::new(&mut c_checked, m, n, 1, m as isize),
        Some(Bias::PerCol(&bias_col)),
        Some(Activation::LeakyRelu(0.1)),
        par,
    );

    let mut c_unchecked = c0.clone();
    // SAFETY: valid in-bounds col-major A/C, C column-major (packed_b orientation), distinct
    // buffers, per-col bias of length n
    unsafe {
        gemm_packed_b_fused_unchecked(
            alpha,
            m,
            a.as_ptr(),
            1,
            m as isize,
            &packed,
            beta,
            c_unchecked.as_mut_ptr(),
            1,
            m as isize,
            bias_col.as_ptr(),
            BiasDim::PerCol,
            true,
            Some(Activation::LeakyRelu(0.1)),
            par,
        );
    }
    for idx in 0..m * n {
        assert_eq!(
            c_checked[idx].to_bits(),
            c_unchecked[idx].to_bits(),
            "b_fused_unchecked != checked at {idx}"
        );
    }
}

#[test]
fn packed_a_fused_unchecked_matches_checked() {
    use gemmkit::{BiasDim, gemm_packed_a_fused_unchecked};
    let mut rng = Rng::new(0x0A_ED34);
    let (m, k, n) = (72usize, 96usize, 129usize);
    let a = make::<f64>(&mut rng, m, k);
    let b = make::<f64>(&mut rng, k, n);
    let c0 = make::<f64>(&mut rng, m * n, 1);
    let bias_row: Vec<f64> = (0..m).map(|_| rng.unit() * 3.0).collect();
    let (alpha, beta) = (0.9f64, 0.7f64);
    let par = Parallelism::Serial;
    let bv = MatRef::new(&b, k, n, 1, k as isize);
    let packed = prepack_lhs(MatRef::new(&a, m, k, 1, m as isize));

    let mut c_checked = c0.clone();
    gemm_packed_a_fused(
        alpha,
        &packed,
        bv,
        beta,
        MatMut::new(&mut c_checked, m, n, n as isize, 1),
        Some(Bias::PerRow(&bias_row)),
        Some(Activation::Relu),
        par,
    );

    let mut c_unchecked = c0.clone();
    // SAFETY: valid in-bounds col-major B, row-major C (packed_a orientation), distinct buffers,
    // per-row bias of length m (user frame; the entry flips the axis internally)
    unsafe {
        gemm_packed_a_fused_unchecked(
            alpha,
            &packed,
            n,
            b.as_ptr(),
            1,
            k as isize,
            beta,
            c_unchecked.as_mut_ptr(),
            n as isize,
            1,
            bias_row.as_ptr(),
            BiasDim::PerRow,
            true,
            Some(Activation::Relu),
            par,
        );
    }
    for idx in 0..m * n {
        assert_eq!(
            c_checked[idx].to_bits(),
            c_unchecked[idx].to_bits(),
            "a_fused_unchecked != checked at {idx}"
        );
    }
}

// narrow (f16/bf16) prepacked-fused: per-element gate vs an f64 reference, plus a k == 1 bitwise
// case vs the f32 single-rounding reference (the pre-narrow contract)

#[cfg(feature = "half")]
mod narrow {
    use crate::common::Rng;
    use gemmkit::{
        Activation, Bias, MatMut, MatRef, NarrowFloat, Parallelism, gemm_packed_a_fused,
        gemm_packed_b_fused, prepack_lhs, prepack_rhs,
    };

    trait Narrow: NarrowFloat + gemmkit::FusedScalar {
        fn of(x: f64) -> Self;
        fn f32(self) -> f32;
        fn bits(self) -> u16;
        const EPS: f64;
        fn name() -> &'static str;
    }
    impl Narrow for gemmkit::f16 {
        fn of(x: f64) -> Self {
            gemmkit::f16::from_f64(x)
        }
        fn f32(self) -> f32 {
            self.widen()
        }
        fn bits(self) -> u16 {
            self.to_bits()
        }
        const EPS: f64 = 9.765625e-4; // 2^-10
        fn name() -> &'static str {
            "f16"
        }
    }
    impl Narrow for gemmkit::bf16 {
        fn of(x: f64) -> Self {
            gemmkit::bf16::from_f64(x)
        }
        fn f32(self) -> f32 {
            self.widen()
        }
        fn bits(self) -> u16 {
            self.to_bits()
        }
        const EPS: f64 = 7.8125e-3; // 2^-7
        fn name() -> &'static str {
            "bf16"
        }
    }

    fn make<N: Narrow>(rng: &mut Rng, m: usize, n: usize) -> Vec<N> {
        (0..m * n).map(|_| N::of(rng.unit())).collect()
    }

    /// f64 reference `C <- relu-or-none(alpha*A*B + beta*C0 + bias_row[i])`, un-narrowed, [i + j*m]
    #[allow(clippy::too_many_arguments)]
    fn reference_f64<N: Narrow>(
        m: usize,
        k: usize,
        n: usize,
        alpha: N,
        a: &[N],
        b: &[N],
        beta: N,
        c0: &[N],
        rsc: isize,
        csc: isize,
        bias_row: &[N],
        relu: bool,
    ) -> Vec<f64> {
        let alpha = alpha.f32() as f64;
        let beta = beta.f32() as f64;
        let mut out = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[i + p * m].f32() as f64 * b[p + j * k].f32() as f64;
                }
                let base = if beta == 0.0 {
                    0.0
                } else {
                    beta * c0[(i as isize * rsc + j as isize * csc) as usize].f32() as f64
                };
                let mut v = alpha * acc + base + bias_row[i].f32() as f64;
                if relu && v <= 0.0 {
                    v = 0.0;
                }
                out[i + j * m] = v;
            }
        }
        out
    }

    /// Per-element gate: `|got - r| <= (2*eps_N + 8*k*f32_eps)*(1 + |r|)` (the established mixed gate)
    #[allow(clippy::too_many_arguments)]
    fn assert_close<N: Narrow>(
        got: &[N],
        rsc: isize,
        csc: isize,
        m: usize,
        n: usize,
        cref: &[f64],
        k: usize,
        ctx: &str,
    ) {
        let f32_eps = f32::EPSILON as f64;
        for j in 0..n {
            for i in 0..m {
                let g = got[(i as isize * rsc + j as isize * csc) as usize].f32() as f64;
                let r = cref[i + j * m];
                assert!(g.is_finite(), "{ctx}: non-finite output at ({i},{j})");
                let tol = (2.0 * N::EPS + 8.0 * (k as f64) * f32_eps) * (1.0 + r.abs());
                assert!(
                    (g - r).abs() <= tol,
                    "{}: {ctx} abs err {:.3e} > tol {tol:.3e} at ({i},{j}) (got {g:.6e}, ref {r:.6e})",
                    N::name(),
                    (g - r).abs(),
                );
            }
        }
    }

    fn gate<N: Narrow>() {
        let mut rng = Rng::new(0xC0FF_AC00);
        let (m, k, n) = (96usize, 128usize, 72usize);
        let a = make::<N>(&mut rng, m, k);
        let b = make::<N>(&mut rng, k, n);
        let c0 = make::<N>(&mut rng, m * n, 1);
        let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
        let (alpha, beta) = (N::of(1.0), N::of(0.7));
        let av = MatRef::new(&a, m, k, 1, m as isize);
        let bv = MatRef::new(&b, k, n, 1, k as isize);

        // RHS-packed, col-major C
        {
            let packed = prepack_rhs(bv);
            let mut c = c0.clone();
            gemm_packed_b_fused(
                alpha,
                av,
                &packed,
                beta,
                MatMut::new(&mut c, m, n, 1, m as isize),
                Some(Bias::PerRow(&bias_row)),
                Some(Activation::Relu),
                Parallelism::Rayon(4),
            );
            let cref = reference_f64::<N>(
                m, k, n, alpha, &a, &b, beta, &c0, 1, m as isize, &bias_row, true,
            );
            assert_close::<N>(&c, 1, m as isize, m, n, &cref, k, "b/gate");
        }
        // LHS-packed, row-major C (bias flip path)
        {
            let packed = prepack_lhs(av);
            let mut c = c0.clone();
            gemm_packed_a_fused(
                alpha,
                &packed,
                bv,
                beta,
                MatMut::new(&mut c, m, n, n as isize, 1),
                Some(Bias::PerRow(&bias_row)),
                Some(Activation::Relu),
                Parallelism::Serial,
            );
            let cref = reference_f64::<N>(
                m, k, n, alpha, &a, &b, beta, &c0, n as isize, 1, &bias_row, true,
            );
            assert_close::<N>(&c, n as isize, 1, m, n, &cref, k, "a/gate");
        }
    }

    #[test]
    fn narrow_packed_fused_gate() {
        gate::<gemmkit::f16>();
        gate::<gemmkit::bf16>();
    }

    /// `k == 1`: the accumulator is a single exact `f32` product, so the fused output must equal the
    /// **single-rounding** reference `narrow(act(alpha*a*b + beta*c0 + bias))` bitwise, for
    /// `beta in {0, 1}` (the exact combine). This locks the pre-narrow semantics through the packed
    /// fused store (the 2-rounding narrow-gemm-then-map would differ)
    fn k1_bitwise<N: Narrow>() {
        let mut rng = Rng::new(0x9E27_ACB1);
        let (m, n) = (32usize, 24usize);
        let k = 1usize;
        // Values in [1, 2) so the product carries sub-narrow bits a matching bias keeps significant
        let a: Vec<N> = (0..m)
            .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
            .collect();
        let b: Vec<N> = (0..n)
            .map(|_| N::of(1.0 + (rng.unit() + 1.0) * 0.5))
            .collect();
        let c0: Vec<N> = (0..m * n).map(|_| N::of(rng.unit())).collect();
        let bias_row: Vec<N> = (0..m).map(|_| N::of(rng.unit() * 2.0)).collect();
        let av = MatRef::new(&a, m, k, 1, m as isize);
        let bv = MatRef::new(&b, k, n, 1, k as isize);
        let alpha = N::of(1.0);

        for beta in [N::of(0.0), N::of(1.0)] {
            // RHS-packed
            let packed = prepack_rhs(bv);
            let mut c = c0.clone();
            gemm_packed_b_fused(
                alpha,
                av,
                &packed,
                beta,
                MatMut::new(&mut c, m, n, 1, m as isize),
                Some(Bias::PerRow(&bias_row)),
                Some(Activation::Relu),
                Parallelism::Serial,
            );
            for j in 0..n {
                for i in 0..m {
                    let ab = a[i].f32() * b[j].f32(); // exact single f32 product
                    let base = if beta.f32() == 0.0 {
                        0.0
                    } else {
                        c0[i + j * m].f32()
                    };
                    let v = ab + base + bias_row[i].f32();
                    let want = N::narrow(if v > 0.0 { v } else { 0.0 });
                    assert_eq!(
                        c[i + j * m].bits(),
                        want.bits(),
                        "{}: k1 packed_b_fused != single-round ref at ({i},{j}) [beta={:#06x}]",
                        N::name(),
                        beta.bits(),
                    );
                }
            }
        }
    }

    #[test]
    fn narrow_packed_fused_k1_bitwise() {
        k1_bitwise::<gemmkit::f16>();
        k1_bitwise::<gemmkit::bf16>();
    }
}
