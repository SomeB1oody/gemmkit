//! Differential drivers: build valid-by-construction operands, run a gemmkit entry
//! point over them, and gate the output against the matching naive reference

use crate::common::{
    Canary, CplxElem, LayoutPlan, RealElem, Rng, assert_no_gap_writes, build_operand, extent_of,
};
use crate::reference::{
    cplx_gate, dense_cplx, dense_i32, dense_i32_from_i8, dense_real, frob, i8_gate, real_denom,
    real_gate, ref_gemm_cplx, ref_gemm_i8, ref_gemm_real,
};
use gemmkit::{
    BatchProblem, MatMut, MatRef, Parallelism, Workspace, gemm, gemm_batched, gemm_batched_slice,
    gemm_cplx, gemm_i8, gemm_i8_packed_b, gemm_packed_a, gemm_packed_b, gemm_with, prepack_lhs,
    prepack_rhs, prepack_rhs_i8,
};

// generic differential drivers (shared across the fuzz_gemm/fuzz_knobs targets)

/// Run a fixed 16x16x16 problem through a caller-owned `Workspace` before the plan's
/// own problem, so the workspace's buffers have to grow or shrink to the plan's shape
/// on reuse: the `tests/workspace_alloc.rs` axis, which a fresh thread-local pool alone
/// never exercises
fn warm_ws<T: RealElem>(ws: &mut Workspace) {
    let (m, k, n) = (16usize, 16usize, 16usize);
    let mut rr = Rng::new(0xA11CE);
    let a: Vec<T> = (0..m * k).map(|_| T::from_f64(rr.next_quant())).collect();
    let b: Vec<T> = (0..k * n).map(|_| T::from_f64(rr.next_quant())).collect();
    let mut c: Vec<T> = vec![T::ZERO; m * n];
    gemm_with(
        ws,
        T::ONE,
        MatRef::new(&a, m, k, k as isize, 1),
        MatRef::new(&b, k, n, n as isize, 1),
        T::ZERO,
        MatMut::new(&mut c, m, n, n as isize, 1),
        Parallelism::Serial,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_gemm_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    nan_c: bool,
    par: Parallelism,
    ws_reuse: bool,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    let seed_nan = nan_c && beta_f == 0.0;
    let (mut cbuf, rsc, csc) = build_operand(m, n, lc, T::SENTINEL, || {
        if seed_nan {
            T::from_f64(f64::NAN)
        } else {
            T::from_f64(rc.next_quant())
        }
    });

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    let a = MatRef::new(&abuf, m, k, rsa, csa);
    let b = MatRef::new(&bbuf, k, n, rsb, csb);
    if ws_reuse {
        let mut ws = Workspace::new();
        warm_ws::<T>(&mut ws);
        gemm_with(
            &mut ws,
            alpha,
            a,
            b,
            beta,
            MatMut::new(&mut cbuf, m, n, rsc, csc),
            par,
        );
    } else {
        gemm(
            alpha,
            a,
            b,
            beta,
            MatMut::new(&mut cbuf, m, n, rsc, csc),
            par,
        );
    }

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_gemm_i8(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha: i32,
    beta: i32,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand::<i8>(m, k, la, 0, || ra.next_i8());
    let (bbuf, rsb, csb) = build_operand::<i8>(k, n, lb, 0, || rb.next_i8());
    // C0 in [-128, 127] (an i8-range i32), same magnitude as A/B, so the epilogue term
    // does not dwarf the product and hide a wrapping bug in one or the other
    let (mut cbuf, rsc, csc) =
        build_operand::<i32>(m, n, lc, i32::SENTINEL, || rc.next_i8() as i32);

    let da = dense_i32_from_i8(&abuf, m, k, rsa, csa);
    let db = dense_i32_from_i8(&bbuf, k, n, rsb, csb);
    let dc0 = dense_i32(&cbuf, m, n, rsc, csc);
    let cref = ref_gemm_i8(&da, &db, &dc0, m, k, n, alpha, beta);

    gemm_i8(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    i8_gate(&cbuf, rsc, csc, m, n, &cref, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Prepacked-i8-RHS round trip: `prepack_rhs_i8(B)`, then `gemm_i8_packed_b` over a
/// column-major-ish C (the only orientation the packed-B entry accepts). Integer GEMM is
/// exact, so unlike the float prepack drivers below this gates EXACTLY: the packed output
/// must equal both the wrapping-i32 reference and a plain `gemm_i8` call bit-for-bit,
/// which is the packed-B API's documented guarantee
#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_prepack_i8(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    alpha: i32,
    beta: i32,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand::<i8>(m, k, la, 0, || ra.next_i8());
    let (bbuf, rsb, csb) = build_operand::<i8>(k, n, lb, 0, || rb.next_i8());
    // ColIsh with il=1 gives (rsc, csc) = (1, m+1), so |csc| >= |rsc| always holds here,
    // satisfying gemm_i8_packed_b's column-major-ish requirement
    let lc = LayoutPlan::ColIsh { il: 1, pad: 1 };
    let (mut cbuf, rsc, csc) =
        build_operand::<i32>(m, n, lc, i32::SENTINEL, || rc.next_i8() as i32);
    // Same C0 (gap slots included) so the plain-gemm_i8 cross-check below starts from
    // the identical buffer the packed call did
    let mut cplain = cbuf.clone();

    let da = dense_i32_from_i8(&abuf, m, k, rsa, csa);
    let db = dense_i32_from_i8(&bbuf, k, n, rsb, csb);
    let dc0 = dense_i32(&cbuf, m, n, rsc, csc);
    let cref = ref_gemm_i8(&da, &db, &dc0, m, k, n, alpha, beta);

    let packed = prepack_rhs_i8(MatRef::new(&bbuf, k, n, rsb, csb));
    if packed.rows() != k || packed.cols() != n {
        panic!(
            "{ctx}: prepack_rhs_i8 echo mismatch: rows {} cols {}",
            packed.rows(),
            packed.cols()
        );
    }

    gemm_i8_packed_b(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        &packed,
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );
    // Same alpha/beta/A/B/C0 through the plain path: the packed result must match it exactly
    gemm_i8(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cplain, m, n, rsc, csc),
        par,
    );

    // Exact vs the independent reference, then exact vs the plain path over the whole
    // buffer (gap slots included, so both must also leave the sentinel untouched there)
    i8_gate(&cbuf, rsc, csc, m, n, &cref, ctx);
    if cbuf != cplain {
        panic!("{ctx}: prepacked i8 output differs from plain gemm_i8 (bit-identity broken)");
    }
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_gemm_cplx<T: CplxElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    alpha: (f64, f64),
    beta: (f64, f64),
    conj_a: bool,
    conj_b: bool,
    nan_c: bool,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let al = T::make(alpha.0, alpha.1);
    let be = T::make(beta.0, beta.1);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand::<T>(m, k, la, T::ZERO, || {
        T::make(ra.next_quant(), ra.next_quant())
    });
    let (bbuf, rsb, csb) = build_operand::<T>(k, n, lb, T::ZERO, || {
        T::make(rb.next_quant(), rb.next_quant())
    });
    let seed_nan = nan_c && beta == (0.0, 0.0);
    let (mut cbuf, rsc, csc) = build_operand::<T>(m, n, lc, T::SENTINEL, || {
        if seed_nan {
            T::make(f64::NAN, f64::NAN)
        } else {
            T::make(rc.next_quant(), rc.next_quant())
        }
    });

    let da = dense_cplx(&abuf, m, k, rsa, csa, conj_a);
    let db = dense_cplx(&bbuf, k, n, rsb, csb, conj_b);
    let dc0 = dense_cplx(&cbuf, m, n, rsc, csc, false);
    let cref = ref_gemm_cplx(&da, &db, &dc0, m, k, n, alpha, beta);

    gemm_cplx(
        al,
        MatRef::new(&abuf, m, k, rsa, csa),
        conj_a,
        MatRef::new(&bbuf, k, n, rsb, csb),
        conj_b,
        be,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    cplx_gate::<T>(&cbuf, rsc, csc, m, n, &cref, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Prepacked-RHS round trip: `prepack_rhs(B)`, then `gemm_packed_b` over a column-major-ish
/// C (the only orientation the packed-B entry accepts). Gates at tolerance, not bit-exact,
/// since the API only promises to reproduce plain `gemm` up to the last ULP for tiny or
/// gemv-shaped products
#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_packed_b_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    // ColIsh with il=1 gives (rsc, csc) = (1, m+1): column-major-ish, as gemm_packed_b requires
    let lc = LayoutPlan::ColIsh { il: 1, pad: 1 };
    let (mut cbuf, rsc, csc) =
        build_operand(m, n, lc, T::SENTINEL, || T::from_f64(rc.next_quant()));

    let packed = prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
    if packed.rows() != k || packed.cols() != n {
        panic!(
            "{ctx}: prepack_rhs echo mismatch: rows {} cols {}",
            packed.rows(),
            packed.cols()
        );
    }

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    gemm_packed_b(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        &packed,
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Prepacked-LHS round trip: `prepack_lhs(A)`, then `gemm_packed_a` over a row-major-ish
/// C (the only orientation the packed-A entry accepts). Same tolerance gate as
/// `differential_packed_b_real`
#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_packed_a_real<T: RealElem>(
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    a_seed: u64,
    b_seed: u64,
    c_seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let mut ra = Rng::new(a_seed);
    let mut rb = Rng::new(b_seed);
    let mut rc = Rng::new(c_seed);
    let (abuf, rsa, csa) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
    let (bbuf, rsb, csb) = build_operand(k, n, lb, T::ZERO, || T::from_f64(rb.next_quant()));
    // RowIsh with il=1 gives (rsc, csc) = (n+1, 1): row-major-ish, as gemm_packed_a requires
    let lc = LayoutPlan::RowIsh { il: 1, pad: 1 };
    let (mut cbuf, rsc, csc) =
        build_operand(m, n, lc, T::SENTINEL, || T::from_f64(rc.next_quant()));

    let packed = prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
    if packed.rows() != m || packed.cols() != k {
        panic!(
            "{ctx}: prepack_lhs echo mismatch: rows {} cols {}",
            packed.rows(),
            packed.cols()
        );
    }

    let da = dense_real(&abuf, m, k, rsa, csa);
    let db = dense_real(&bbuf, k, n, rsb, csb);
    let dc0 = dense_real(&cbuf, m, n, rsc, csc);
    let na = frob(&da);
    let nb = frob(&db);
    let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
    let denom = real_denom(alpha_f, na, nb, beta_f, nc0);
    let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);

    gemm_packed_a(
        alpha,
        &packed,
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    real_gate::<T>(&cbuf, rsc, csc, m, n, &cref, denom, k, ctx);
    assert_no_gap_writes(&cbuf, m, n, rsc, csc, ctx);
}

/// Strided-batched GEMM (`gemm_batched`) over 1 big buffer per operand, sized and
/// strided so every batch element is valid by construction, plus a `gemm_batched_slice`
/// cross-check over separate per-element buffers (`batched_slice_real`, below)
#[allow(clippy::too_many_arguments)]
pub(crate) fn differential_batched_real<T: RealElem>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    la: LayoutPlan,
    lb: LayoutPlan,
    lc: LayoutPlan,
    a_broadcast: bool,
    b_broadcast: bool,
    a_bs_pad: usize,
    b_bs_pad: usize,
    c_bs_pad: usize,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let (rsa, csa) = la.strides(m, k);
    let (rsb, csb) = lb.strides(k, n);
    let (rsc, csc) = lc.strides(m, n);
    let ea = extent_of(m, k, rsa, csa);
    let eb = extent_of(k, n, rsb, csb);
    let ec = extent_of(m, n, rsc, csc);
    // A/B batch stride 0 broadcasts the same element across the batch (read-only, so
    // safe); C never broadcasts, and must clear at least 1 element's extent
    let a_bs = if a_broadcast { 0 } else { ea + a_bs_pad };
    let b_bs = if b_broadcast { 0 } else { eb + b_bs_pad };
    let c_bs = ec + c_bs_pad;

    let a_len = if batch <= 1 {
        ea
    } else {
        (batch - 1) * a_bs + ea
    };
    let b_len = if batch <= 1 {
        eb
    } else {
        (batch - 1) * b_bs + eb
    };
    let c_len = if batch <= 1 {
        ec
    } else {
        (batch - 1) * c_bs + ec
    };

    let mut ra = Rng::new(seed ^ 0x0A);
    let mut rb = Rng::new(seed ^ 0x0B);
    let mut rc = Rng::new(seed ^ 0x0C);
    let mut abuf = vec![T::ZERO; a_len];
    let mut bbuf = vec![T::ZERO; b_len];
    let mut cbuf = vec![T::SENTINEL; c_len];
    for e in 0..batch {
        let base = e * a_bs;
        for i in 0..m {
            for j in 0..k {
                abuf[base + (i as isize * rsa + j as isize * csa) as usize] =
                    T::from_f64(ra.next_quant());
            }
        }
        let base = e * b_bs;
        for i in 0..k {
            for j in 0..n {
                bbuf[base + (i as isize * rsb + j as isize * csb) as usize] =
                    T::from_f64(rb.next_quant());
            }
        }
        let base = e * c_bs;
        for i in 0..m {
            for j in 0..n {
                cbuf[base + (i as isize * rsc + j as isize * csc) as usize] =
                    T::from_f64(rc.next_quant());
            }
        }
    }

    // Compute every element's reference before gemm_batched runs, over the pre-call C0
    let mut refs: Vec<(Vec<f64>, f64)> = Vec::with_capacity(batch);
    for e in 0..batch {
        let da = dense_real(&abuf[e * a_bs..], m, k, rsa, csa);
        let db = dense_real(&bbuf[e * b_bs..], k, n, rsb, csb);
        let dc0 = dense_real(&cbuf[e * c_bs..], m, n, rsc, csc);
        let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
        let denom = real_denom(alpha_f, frob(&da), frob(&db), beta_f, nc0);
        let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);
        refs.push((cref, denom));
    }

    gemm_batched(
        batch,
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        a_bs as isize,
        MatRef::new(&bbuf, k, n, rsb, csb),
        b_bs as isize,
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        c_bs as isize,
        par,
    );

    for e in 0..batch {
        let (cref, denom) = &refs[e];
        let slot = &cbuf[e * c_bs..e * c_bs + ec];
        real_gate::<T>(slot, rsc, csc, m, n, cref, *denom, k, ctx);
    }
    // Whole-buffer check: both the per-element gap slots and the padding between
    // consecutive elements (c_bs_pad) must still read as the sentinel
    assert_batched_no_gap_writes(&cbuf, batch, m, n, rsc, csc, c_bs, ctx);

    // Cross-check the 2nd batched entry point (pointer array over per-element buffers)
    if batch >= 1 {
        batched_slice_real::<T>(batch, m, k, n, alpha_f, beta_f, par, seed, ctx);
    }
}

fn assert_batched_no_gap_writes<T: Canary>(
    buf: &[T],
    batch: usize,
    m: usize,
    n: usize,
    rsc: isize,
    csc: isize,
    c_bs: usize,
    ctx: &str,
) {
    let ec = extent_of(m, n, rsc, csc);
    let mut is_view = vec![false; buf.len()];
    for e in 0..batch {
        let base = e * c_bs;
        for i in 0..m {
            for j in 0..n {
                is_view[base + (i as isize * rsc + j as isize * csc) as usize] = true;
            }
        }
    }
    for (idx, slot) in buf.iter().enumerate() {
        if !is_view[idx] && !slot.is_sentinel() {
            panic!("{ctx}: batched gap slot {idx} overwritten (ec={ec}, c_bs={c_bs})");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn batched_slice_real<T: RealElem>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha_f: f64,
    beta_f: f64,
    par: Parallelism,
    seed: u64,
    ctx: &str,
) {
    let alpha = T::from_f64(alpha_f);
    let beta = T::from_f64(beta_f);
    let la = LayoutPlan::RowIsh { il: 1, pad: 0 };
    let lb = LayoutPlan::RowIsh { il: 1, pad: 0 };
    let lc = LayoutPlan::RowIsh { il: 1, pad: 1 };
    let (rsa, csa) = la.strides(m, k);
    let (rsb, csb) = lb.strides(k, n);
    let (rsc, csc) = lc.strides(m, n);

    let mut a_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    let mut b_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    let mut c_bufs: Vec<Vec<T>> = Vec::with_capacity(batch);
    for e in 0..batch {
        let mut ra = Rng::new(seed ^ (e as u64).wrapping_mul(0x9E37) ^ 0x51CE);
        let (ab, _, _) = build_operand(m, k, la, T::ZERO, || T::from_f64(ra.next_quant()));
        let (bb, _, _) = build_operand(k, n, lb, T::ZERO, || T::from_f64(ra.next_quant()));
        let (cb, _, _) = build_operand(m, n, lc, T::SENTINEL, || T::from_f64(ra.next_quant()));
        a_bufs.push(ab);
        b_bufs.push(bb);
        c_bufs.push(cb);
    }

    let mut refs: Vec<(Vec<f64>, f64)> = Vec::with_capacity(batch);
    for e in 0..batch {
        let da = dense_real(&a_bufs[e], m, k, rsa, csa);
        let db = dense_real(&b_bufs[e], k, n, rsb, csb);
        let dc0 = dense_real(&c_bufs[e], m, n, rsc, csc);
        let nc0 = if beta_f == 0.0 { 0.0 } else { frob(&dc0) };
        let denom = real_denom(alpha_f, frob(&da), frob(&db), beta_f, nc0);
        let cref = ref_gemm_real(&da, &db, &dc0, m, k, n, alpha_f, beta_f);
        refs.push((cref, denom));
    }

    let mut problems: Vec<BatchProblem<T>> = Vec::with_capacity(batch);
    for ((ab, bb), cb) in a_bufs.iter().zip(b_bufs.iter()).zip(c_bufs.iter_mut()) {
        problems.push(BatchProblem {
            alpha,
            a: MatRef::new(ab, m, k, rsa, csa),
            b: MatRef::new(bb, k, n, rsb, csb),
            beta,
            c: MatMut::new(cb, m, n, rsc, csc),
        });
    }
    gemm_batched_slice(&mut problems, par);
    drop(problems);

    for e in 0..batch {
        let (cref, denom) = &refs[e];
        real_gate::<T>(&c_bufs[e], rsc, csc, m, n, cref, *denom, k, ctx);
        assert_no_gap_writes(&c_bufs[e], m, n, rsc, csc, ctx);
    }
}
