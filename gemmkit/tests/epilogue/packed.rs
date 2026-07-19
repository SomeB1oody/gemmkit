//! Tests for the prepacked-operand fused entries `gemm_packed_b_fused` (RHS prepacked, requires
//! column-major-ish C) and `gemm_packed_a_fused` (LHS prepacked, requires row-major-ish C):
//! `packed-fused == packed-then-map` over the general shape sweep, the degenerate cases
//! (`alpha == 0` / `k == 0`), reusing one prepacked handle across plain and fused calls, the
//! no-swap orientation panics, the checked/allocating vs `_unchecked`/`_with` equivalence, and
//! (submodule [`narrow`]) the narrow `f16`/`bf16` pre-narrow contract
//!
//! For `f32`/`f64` every comparison against the plain packed entry followed by the same scalar
//! map is bitwise: the fused epilogue only changes the store at the end of the *same* prepacked
//! kernel `gemm_packed_*` already runs (identical blocking, packing, and scheduling), applying to
//! the exact register or scratch value the plain store would otherwise have written. The narrow
//! path instead applies the epilogue in `f32` before its single narrowing, so it is locked against
//! an `f32` single-rounding reference: a per-element tolerance gate over the general shape, and a
//! bitwise check at `k == 1` where the accumulator is a single exact product; not the 2-rounding
//! `narrow-gemm-then-map` a naive oracle would use

use crate::common::*;
use gemmkit::{
    Activation, Bias, MatMut, MatRef, Parallelism, Workspace, gemm_packed_a, gemm_packed_a_fused,
    gemm_packed_a_fused_with, gemm_packed_b, gemm_packed_b_fused, gemm_packed_b_fused_with,
    prepack_lhs, prepack_rhs,
};

// C layouts `gemm_packed_b_fused` accepts: column-major-ish (|csc| >= |rsc|)
#[derive(Copy, Clone)]
enum ColC {
    /// Contiguous column-major: `rsc = 1, csc = m`
    Col,
    /// Column-major with a gap after each column (`rsc = 1, csc = m + 3`): a strided C that
    /// forces the scratch path at tile edges
    ColPadded,
}
fn col_c_strides(layout: ColC, m: usize, n: usize) -> (isize, isize, usize) {
    match layout {
        ColC::Col => (1, m as isize, m * n),
        ColC::ColPadded => (1, (m + 3) as isize, (m + 3) * n),
    }
}

// C layouts `gemm_packed_a_fused` accepts: row-major-ish (|csc| <= |rsc|)
#[derive(Copy, Clone)]
enum RowC {
    /// Contiguous row-major: `rsc = n, csc = 1`
    Row,
    /// Row-major with a gap after each row (`rsc = n + 3, csc = 1`): a strided C
    RowPadded,
}
fn row_c_strides(layout: RowC, m: usize, n: usize) -> (isize, isize, usize) {
    match layout {
        RowC::Row => (n as isize, 1, m * n),
        RowC::RowPadded => ((n + 3) as isize, 1, m * (n + 3)),
    }
}

/// Runs one RHS-packed fused config and its `gemm_packed_b`-then-map oracle off the same
/// prepacked handle, asserting the 2 outputs bitwise-equal over the whole (possibly strided) `C`
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
    bias_kind: u8, // 0 = none, 1 = per-row, 2 = per-col
    act: Option<Activation<T>>,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k.max(1)); // col-major m x k; k.max(1) keeps a real buffer at k == 0
    let b = make::<T>(rng, k.max(1), n); // col-major k x n
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

    // The fused call under test
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

    // The oracle: the same handle through plain gemm_packed_b, then the scalar map by hand
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

/// Runs one LHS-packed fused config and its `gemm_packed_a`-then-map oracle, asserting the 2
/// bitwise-equal. Since the packed-A path drives the transposed product internally, a bias axis
/// flipped the wrong way here would show up as a loud mismatch rather than staying silent
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
    let a = make::<T>(rng, m, k.max(1)); // col-major m x k
    let b = make::<T>(rng, k.max(1), n); // col-major k x n
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

// core oracle sweep: several shapes (including ones that are not multiples of the tile size),
// beta in {0, 1, 0.7}, bias in {none, row, col}, act in {none, relu, leaky}, plain and padded C

fn packed_b_matrix<T: Flt>(par: Parallelism) {
    let mut rng = Rng::new(0x9ACB_11B0);
    let shapes = [(200usize, 130usize, 175usize), (65, 64, 64), (40, 200, 129)];
    let acts: [Option<Activation<T>>; 3] = [
        None,
        Some(Activation::Relu),
        Some(Activation::LeakyRelu(T::of(0.1))),
    ];
    // Under GEMMKIT_FAST_TEST, only the smallest shape (index 1) runs the full lattice; the other
    // 2 shapes each keep just 1 non-trivial combo, since every bias/act/layout class is already
    // covered by the full-lattice shape and only the redundant per-shape cross-product shrinks
    // With the env var unset, `fast` is false and this is byte-for-byte the unshrunk sweep
    let fast = fast_test();
    let full_lattice = 1usize;
    for (si, &(m, k, n)) in shapes.iter().enumerate() {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [ColC::Col, ColC::ColPadded] {
                    let (rsc, csc, clen) = col_c_strides(layout, m, n);
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            if fast
                                && si != full_lattice
                                && !(beta == T::of(0.7)
                                    && alpha == T::of(0.9)
                                    && matches!(layout, ColC::ColPadded)
                                    && bias_kind == 2
                                    && matches!(act, Some(Activation::LeakyRelu(_))))
                            {
                                continue;
                            }
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
    // Same GEMMKIT_FAST_TEST shrink policy as `packed_b_matrix`
    let fast = fast_test();
    let full_lattice = 1usize;
    for (si, &(m, k, n)) in shapes.iter().enumerate() {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [RowC::Row, RowC::RowPadded] {
                    let (rsc, csc, clen) = row_c_strides(layout, m, n);
                    for bias_kind in 0u8..=2 {
                        for act in &acts {
                            if fast
                                && si != full_lattice
                                && !(beta == T::of(0.7)
                                    && alpha == T::of(0.9)
                                    && matches!(layout, RowC::RowPadded)
                                    && bias_kind == 2
                                    && matches!(act, Some(Activation::LeakyRelu(_))))
                            {
                                continue;
                            }
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

// degenerate cases where the A*B term vanishes (alpha == 0 with k > 0, or k == 0): the fused
// entry falls back to C <- act(beta*C + bias) directly, checked against the same
// check_packed_*_fused oracle (plain packed's beta*C, then the scalar map by hand)

#[test]
fn packed_fused_degenerate() {
    let mut rng = Rng::new(0xDE6E_9AC0);
    let (m, n) = (48usize, 40usize);
    for &(k, alpha) in &[(0usize, 1.0f32), (130usize, 0.0f32)] {
        // RHS-packed, padded column-major C, per-row bias + ReLU
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
        // LHS-packed, padded row-major C, per-row then per-col bias: exercises the degenerate
        // path's bias-axis flip in both directions
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

// The same PackedRhs handle drives several fused calls and a plain call in a row: since the
// buffer is read-only and the epilogue is store-side only, each call's output must be independent
// of what ran before it

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
    let packed = prepack_rhs(bv); // 1 handle, reused for every call below

    // 3 calls off that 1 handle: fused (PerRow bias + ReLU), plain, fused (PerCol bias + Leaky)
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

    // Each fused result must equal the plain result mapped by its own bias/act, and the plain
    // result itself must be unaffected by the fused calls sharing its handle
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

// A prepacked operand cannot serve the orientation swap plain gemm would take, so the fused
// entries reject the wrong-orientation C the same way the plain packed entries do

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
        MatMut::from_row_major(&mut c, m, n), // row-major C needs the A/B swap, so this panics
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
        MatMut::from_col_major(&mut c, m, n), // column-major C would keep A as the true LHS: panics
        None,
        Some(Activation::Relu),
        Parallelism::Serial,
    );
}

// The _with entry (caller-owned Workspace) must produce the same bits as the allocating entry
// that hands it a thread-local one internally

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

    // gemm_packed_b_fused
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
    // gemm_packed_a_fused (row-major C, as that path requires)
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

// The checked fused entries build their own Task/FusedEpi rather than delegating to the
// `_unchecked` ones, so a Bias/Activation lowering drift between the 2 call paths would otherwise
// go unnoticed; these tests drive both from the same inputs and compare the output bitwise

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
    // SAFETY: a/c are valid in-bounds col-major buffers matching av/c_unchecked above, c is
    // column-major (the orientation gemm_packed_b requires) and does not alias a, and bias_col
    // has 1 entry per output column as BiasDim::PerCol requires
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
    // SAFETY: b/c are valid in-bounds buffers matching bv/c_unchecked above, c is row-major (the
    // orientation gemm_packed_a requires) and does not alias b, and bias_row has 1 entry per
    // packed-A row (the user frame; the entry flips the axis internally before dispatch)
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

// Narrow (f16/bf16) prepacked-fused tests: a per-element gate against an f64 reference over a
// general shape, plus a bitwise k == 1 case against the f32 single-rounding reference that pins
// down the pre-narrow contract

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
        const EPS: f64 = 9.765625e-4; // 2^-10: f16 has a 10-bit mantissa
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
        const EPS: f64 = 7.8125e-3; // 2^-7: bf16 has a 7-bit mantissa
        fn name() -> &'static str {
            "bf16"
        }
    }

    fn make<N: Narrow>(rng: &mut Rng, m: usize, n: usize) -> Vec<N> {
        (0..m * n).map(|_| N::of(rng.unit())).collect()
    }

    /// f64 reference for `C <- relu-or-none(alpha*A*B + beta*C0 + bias_row[i])`, un-narrowed,
    /// `out[i + j*m]` in logical (row, col) order
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

    /// Per-element accuracy gate against the un-narrowed f64 reference `cref`: asserts
    /// `|got - r| <= (2*eps_N + 8*k*f32_eps)*(1 + |r|)`, the same bound the mixed-suite gate uses
    /// (the `f32`-accumulation error plus 1 narrowing ulp, scaled to stay meaningful near 0)
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

        // gemm_packed_b_fused, column-major C
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
        // gemm_packed_a_fused, row-major C (exercises the bias-axis flip)
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

    /// At `k = 1` the accumulator is a single exact `f32` product, so for `beta in {0, 1}` (an
    /// exact combine on both sides) the fused output must equal the single-rounding reference
    /// `narrow(act(alpha*a*b + beta*c0 + bias))` bitwise. This locks the pre-narrow semantic
    /// through the packed fused store: the 2-rounding `narrow-gemm-then-map` would differ
    fn k1_bitwise<N: Narrow>() {
        let mut rng = Rng::new(0x9E27_ACB1);
        let (m, n) = (32usize, 24usize);
        let k = 1usize;
        // [1, 2): the product carries sub-narrow bits that a comparable bias below keeps significant
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
            // gemm_packed_b_fused
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
                    let ab = a[i].f32() * b[j].f32(); // k == 1: the single f32 product, exact
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
