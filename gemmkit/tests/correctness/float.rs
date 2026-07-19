//! Real float (f32/f64) GEMM through the public dispatched API: shapes x layouts x
//! alpha/beta, workspace reuse, parallel/serial bit-identity, and gemv shapes

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_unchecked};

/// `m` values (crossed with fixed `k`/`n` lists and a few big squares below) spanning
/// the AVX-512 f32 tile boundaries (`MR=32`, `NR=12`) and general blocking edges
///
/// Under `GEMMKIT_FAST_TEST` this drops to the classes that matter (0, 1, a
/// sub-tile, the `NR`/`MR` boundary and boundary+1, a multi-tile, a large multi-block)
/// and cuts the redundant middle (2, 11, 16, 31, 48, 100); the `k`/`n` lists and the
/// big squares are unaffected either way
fn dims() -> Vec<(usize, usize, usize)> {
    let vals: &[usize] = if fast_test() {
        &[0, 1, 5, 12, 13, 32, 33, 64, 257]
    } else {
        &[0, 1, 2, 5, 11, 12, 13, 16, 31, 32, 33, 48, 64, 100, 257]
    };
    let mut out = Vec::new();
    // A representative cross-section: the full m x k x n cross product is huge
    for &m in vals {
        for &k in &[1usize, 2, 7, 32, 65] {
            for &n in &[1usize, 11, 12, 13, 64] {
                out.push((m, k, n));
            }
        }
    }
    for &s in &[128usize, 200, 384] {
        out.push((s, s, s));
    }
    out
}

/// f32 over the [`dims`] shape spread, row-major A, col-major B, and both C layouts,
/// `alpha=1, beta=0`, against the f64 reference
#[test]
fn correctness_f32_layouts() {
    for (m, k, n) in dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            run_case::<f32>(
                m,
                k,
                n,
                Layout::Row,
                Layout::Col,
                lc,
                1.0,
                0.0,
                Parallelism::Serial,
            );
        }
    }
}

/// f64 twin of [`correctness_f32_layouts`], with A/B layouts swapped (col-major A,
/// row-major B) so the 2 float types don't share identical operand orientations
#[test]
fn correctness_f64_layouts() {
    for (m, k, n) in dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            run_case::<f64>(
                m,
                k,
                n,
                Layout::Col,
                Layout::Row,
                lc,
                1.0,
                0.0,
                Parallelism::Serial,
            );
        }
    }
}

/// General (non-unit, non-trivial) strides on every operand: [`Layout::GeneralPad`]
/// throughout for f32, and mixed with row/col for f64, over a small shape set
#[test]
fn correctness_general_strides() {
    for (m, k, n) in [(7, 9, 5), (32, 32, 32), (33, 17, 19), (64, 64, 64)] {
        run_case::<f32>(
            m,
            k,
            n,
            Layout::GeneralPad,
            Layout::GeneralPad,
            Layout::GeneralPad,
            1.0,
            0.0,
            Parallelism::Serial,
        );
        run_case::<f64>(
            m,
            k,
            n,
            Layout::GeneralPad,
            Layout::Row,
            Layout::Col,
            1.0,
            0.0,
            Parallelism::Serial,
        );
    }
}

/// An alpha/beta sweep (both signs, zero, and 1) crossed with a few shapes, f32
/// row-major throughout and f64 col-major throughout
#[test]
fn correctness_alpha_beta() {
    let combos = [
        (0.0f64, 0.0),
        (0.0, 1.0),
        (0.0, 2.5),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, -1.5),
        (2.0, 0.0),
        (-0.5, 3.0),
    ];
    for (m, k, n) in [(5, 6, 7), (32, 40, 24), (64, 31, 48)] {
        for &(al, be) in &combos {
            run_case::<f32>(
                m,
                k,
                n,
                Layout::Row,
                Layout::Row,
                Layout::Row,
                al as f32,
                be as f32,
                Parallelism::Serial,
            );
            run_case::<f64>(
                m,
                k,
                n,
                Layout::Col,
                Layout::Col,
                Layout::Col,
                al,
                be,
                Parallelism::Serial,
            );
        }
    }
}

/// beta==0 must never read C: seed C with NaN and check the output is finite.
/// `kernel::float` and `kernel::mixed` each have their own `BetaStatus::Zero` branch,
/// so this runs f32/f64 and (under `half`) f16/bf16, not just f32: a branch that
/// loaded C anyway would propagate the NaN (`0*NaN == NaN`) into a failed finite
/// check. `(40,33,28)` sits inside the small-matrix blocking shortcut (both dims
/// under the 64-element default); `(64,16,96)` exceeds it (`n=96 > 64`), covering
/// the general blocking path's `Zero` branch too
#[test]
fn beta_zero_does_not_read_c() {
    fn check<T: Elem>() {
        for (m, k, n) in [(40usize, 33, 28), (64, 16, 96)] {
            let a = Mat::<T>::rand(m, k, 7 + m as u64);
            let b = Mat::<T>::rand(k, n, 9 + n as u64);
            let cref = reference(
                &a,
                &b,
                &Mat {
                    v: vec![T::from_f64(0.0); m * n],
                    rows: m,
                    cols: n,
                },
                1.0,
                0.0,
            );
            let (abuf, rsa, csa) = build_view(&a, Layout::Col);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            let mut cbuf = vec![T::from_f64(f64::NAN); m * n];
            gemm(
                T::from_f64(1.0),
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                T::from_f64(0.0),
                MatMut::from_col_major(&mut cbuf, m, n),
                Parallelism::Serial,
            );
            assert_accurate(&cbuf, 1, m as isize, m, n, &cref, &a, &b, k, "beta=0 NaN C");
        }
    }
    check::<f32>();
    check::<f64>();
    #[cfg(feature = "half")]
    {
        check::<gemmkit::f16>();
        check::<gemmkit::bf16>();
    }
}

/// `gemm_with` (`workspace_alloc.rs` only checks it allocates nothing on reuse, not
/// that the result is correct) must match `gemm` numerically; reuse 1 `Workspace`
/// across 2 calls so both the cold and warm paths are checked
#[test]
fn workspace_reuse_matches_allocating() {
    fn check<T: Elem>() {
        let (m, k, n) = (96, 65, 72);
        let a = Mat::<T>::rand(m, k, 0x7A + m as u64);
        let b = Mat::<T>::rand(k, n, 0x7B + n as u64);
        let c0 = Mat::<T>::rand(m, n, 0x7C + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
        let (al, be) = (T::from_f64(0.5), T::from_f64(0.25));

        let mut c_ref = cbase.clone();
        gemm(
            al,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            be,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let mut ws = Workspace::new();
        for _ in 0..2 {
            let mut c_ws = cbase.clone();
            gemmkit::gemm_with(
                &mut ws,
                al,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                be,
                MatMut::new(&mut c_ws, m, n, rsc, csc),
                Parallelism::Serial,
            );
            assert!(
                c_ref
                    .iter()
                    .zip(&c_ws)
                    .all(|(x, y)| x.to_f64().to_bits() == y.to_f64().to_bits()),
                "gemm_with != gemm"
            );
        }
    }
    check::<f32>();
    check::<f64>();
}

/// Serial and parallel runs land on identical bits, for every thread count, on
/// col-major operands. Float add isn't associative, so this holds only because
/// `blocking()` derives `mc`/`kc`/`nc` from shape and cache topology alone, never the
/// thread count, so every run reduces in the same fixed order; the actual contract
/// (see `driver.rs` and the `accumulate_tile` seam) is reproducibility under a fixed
/// config, not bitwise serial-vs-parallel identity. The exact check stays as the
/// strongest net against an accidental reduction-order divergence (a race, a
/// thread-count-dependent tail bug); relax to determinism + tolerance only if blocking
/// is ever made parallelism-dependent on purpose (e.g. split-K). This caveat applies to
/// every `parallel_equals_serial_*` test in this file
#[test]
fn parallel_equals_serial_bit_identical() {
    for (m, k, n) in [
        (64, 64, 64),
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
    ] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0xABC + m as u64);
            let b = Mat::<f32>::rand(k, n, 0xDEF + n as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x123 + k as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Col);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

            let mut c_serial = cbase.clone();
            let mut c_par = cbase.clone();
            gemm(
                al as f32,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                be as f32,
                MatMut::new(&mut c_serial, m, n, rsc, csc),
                Parallelism::Serial,
            );
            for threads in [2usize, 4, 8, 16] {
                c_par.copy_from_slice(&cbase);
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_par, m, n, rsc, csc),
                    Parallelism::Rayon(threads),
                );
                assert_eq!(
                    c_serial, c_par,
                    "serial != parallel({threads}) for {m}x{k}x{n} a={al} b={be}"
                );
            }
        }
    }
}

/// The row-major-A twin of [`parallel_equals_serial_bit_identical`] (same
/// thread-independent-blocking caveat): `rsa != 1` routes A through per-row-block
/// packing, which schedules jobs via `packed_block_grain`'s whole-row-block grain
/// instead of the column-major case's fine-grain cursor. Sizes are chosen so the
/// row-block count interacts differently with each thread count, reaching both the
/// undivided whole-block grain and its split fallback
#[test]
fn parallel_equals_serial_row_major_a() {
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0xA11 + m as u64);
            let b = Mat::<f32>::rand(k, n, 0xB22 + n as u64);
            let c0 = Mat::<f32>::rand(m, n, 0xC33 + k as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa=k, not 1: triggers packing
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

            let mut c_serial = cbase.clone();
            let mut c_par = cbase.clone();
            gemm(
                al as f32,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                be as f32,
                MatMut::new(&mut c_serial, m, n, rsc, csc),
                Parallelism::Serial,
            );
            for threads in [2usize, 4, 8, 16] {
                c_par.copy_from_slice(&cbase);
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_par, m, n, rsc, csc),
                    Parallelism::Rayon(threads),
                );
                assert_eq!(
                    c_serial, c_par,
                    "row-major A: serial != parallel({threads}) for {m}x{k}x{n} a={al} b={be}"
                );
            }
        }
    }
}

/// `set_shared_lhs_mnk(1)` opens the shared-A-pack gate for any nonzero `m*n*k`, so
/// the shared pre-pack path (1 pack per row-block, read by every worker) runs for
/// every thread count here and must still land on the same bits as serial. All 3
/// shapes stay well under the real default gate (50M on aarch64, 8B elsewhere), so
/// this is otherwise the only coverage this path gets; since bit-identity is expected
/// whether the gate is open or closed, forcing it open cannot disturb concurrently
/// running tests. Row-major A (`rsa != 1`) is what makes A pack in the first place
#[test]
fn shared_lhs_a_bit_identical() {
    let prev = gemmkit::tuning::shared_lhs_mnk();
    gemmkit::tuning::set_shared_lhs_mnk(1); // open the gate for every shape below
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        let a = Mat::<f32>::rand(m, k, 0xA1 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0xB2 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0xC3 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa=k, not 1: triggers packing
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

        let mut c_ser = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            0.4,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        for t in [2usize, 4, 8, 16] {
            let mut c_par = cbase.clone();
            gemm(
                0.9,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                0.4,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(t),
            );
            assert_eq!(
                c_ser, c_par,
                "shared-A: serial != parallel({t}) for {m}x{k}x{n}"
            );
        }
    }
    gemmkit::tuning::set_shared_lhs_mnk(prev);
}

/// `gemm_unchecked` accepts a negative row stride for A (the safe `MatRef` surface
/// cannot express one): point the base at A's last physical row and walk backwards
/// with `rs = -rs`. C is written in ordinary forward row order, so output row `i`
/// ends up holding the product for A's physical row `m-1-i`; the check below reads
/// the f64 reference the same way
#[test]
fn negative_strides_unchecked() {
    let (m, k, n) = (12, 9, 7);
    let a = Mat::<f64>::rand(m, k, 5);
    let b = Mat::<f64>::rand(k, n, 6);
    let cref = reference(
        &a,
        &b,
        &Mat {
            v: vec![0.0; m * n],
            rows: m,
            cols: n,
        },
        1.0,
        0.0,
    );

    let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa == k, csa == 1
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let mut cbuf = vec![0.0f64; m * n];

    unsafe {
        let a_last = abuf.as_ptr().add(((m - 1) as isize * rsa) as usize);
        let c_ptr = cbuf.as_mut_ptr();
        gemm_unchecked(
            m,
            k,
            n,
            1.0,
            a_last,
            -rsa,
            csa,
            bbuf.as_ptr(),
            rsb,
            csb,
            0.0,
            c_ptr,
            n as isize,
            1,
            Parallelism::Serial,
        );
    }
    // C[i,j] holds sum_k A[m-1-i,k]*B[k,j]; index the reference the same way
    for i in 0..m {
        for j in 0..n {
            let got = cbuf[i * n + j];
            let exp = cref[(m - 1 - i) * n + j];
            assert!(
                (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                "neg stride mismatch"
            );
        }
    }
}

// gemv shapes (n == 1 or m == 1, routed through gemm's gemv fast path)

/// `n == 1` (mat*vec) and `m == 1` (vec*mat) shapes, including a `k == 1` degenerate
/// case, across every A/B layout combination, alpha/beta both nonzero
#[test]
fn gemv_shapes() {
    for &(m, k, n) in &[
        (64, 40, 1),
        (1, 40, 64),
        (100, 1, 1),
        (1, 1, 100),
        (255, 129, 1),
    ] {
        for &la in &[Layout::Row, Layout::Col] {
            for &lb in &[Layout::Row, Layout::Col] {
                run_case::<f32>(m, k, n, la, lb, Layout::Col, 1.3, -0.4, Parallelism::Serial);
                run_case::<f64>(m, k, n, la, lb, Layout::Row, 0.5, 2.0, Parallelism::Serial);
            }
        }
    }
}
