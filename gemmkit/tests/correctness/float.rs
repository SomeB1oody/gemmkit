//! Float shapes x layouts x alpha/beta via the public dispatched API, workspace
//! reuse, parallel/serial bit-identity, and gemv shapes

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_unchecked};

fn dims() -> Vec<(usize, usize, usize)> {
    // Edge values around the AVX-512 f32 tile (MR=32, NR=12) and blocking
    let vals = [
        0usize, 1, 2, 5, 11, 12, 13, 16, 31, 32, 33, 48, 64, 100, 257,
    ];
    let mut out = Vec::new();
    // A representative cross-section (full cross product is huge)
    for &m in &vals {
        for &k in &[1usize, 2, 7, 32, 65] {
            for &n in &[1usize, 11, 12, 13, 64] {
                out.push((m, k, n));
            }
        }
    }
    // A few big squares
    for &s in &[128usize, 200, 384] {
        out.push((s, s, s));
    }
    out
}

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

/// beta==0 must not read C, proved by seeding C with NaN. Each kernel family has
/// its own `BetaStatus::Zero` branch (float / mixed / int / complex), so cover every
/// real element type, not just f32: a family that load-then-stored C would propagate
/// the NaN (`0 * NaN == NaN`) and fail the finite check in `assert_accurate`. Sizes
/// hit both the small-matrix branch (40x33x28) and a real tile with partial edges
/// (64x16x96, exercising the strided copy-back `Zero` branch)
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

/// The workspace-reuse entry `gemm_with` must match the pool-allocating `gemm`
/// numerically (it otherwise has only a zero-alloc test). Reuse one `Workspace` across
/// 2 calls to also cover the warm (no-alloc) path
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

/// Serial and parallel runs are bit-identical **under the current thread-independent
/// blocking**: float add isn't associative, so this holds only because every thread
/// count reduces in the same order, not because the library promises it. The *contract*
/// is weaker: reproducible under a fixed config, not bitwise serial-vs-parallel (see the
/// consistency docs and the `accumulate_tile` contract). The exact check is kept as the
/// strongest net against an *accidental* reduction-order divergence (a race, a
/// thread-count-dependent tail bug); relax it to determinism + tolerance only if blocking
/// is ever made parallelism-dependent on purpose (e.g. split-K). (Canonical caveat; the
/// other float `parallel_equals_serial_*` tests share it.)
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

/// Serial == parallel bit-identity (same thread-independent-blocking caveat as
/// `parallel_equals_serial_bit_identical`) with a **row-major A** (`rsa != 1`), which forces per-row-block LHS
/// packing and so exercises the dynamic scheduler's whole-row-block ("packed")
/// grain path under multiple threads: distinct from the column-major case above.
/// Sizes are chosen so the row-block count straddles the thread count (so both the
/// `grain = n_nt` branch and its fine-grain fallback run)
#[test]
fn parallel_equals_serial_row_major_a() {
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0xA11 + m as u64);
            let b = Mat::<f32>::rand(k, n, 0xB22 + n as u64);
            let c0 = Mat::<f32>::rand(m, n, 0xC33 + k as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k != 1 -> packs A
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

/// Shared-LHS A-pack: with the workload gate forced fully open, the shared
/// pre-pack path (one pack per row-block + indexed read) must stay bit-identical
/// to the serial per-worker path, for every thread count. These sizes sit below
/// the default gate, so this is the only coverage of the shared pre-pass; bit-
/// identity holds whether the gate is on or off, so forcing it cannot disturb
/// concurrently-running tests. Row-major A (`rsa != 1`) forces the packed path
#[test]
fn shared_lhs_a_bit_identical() {
    let prev = gemmkit::tuning::shared_lhs_mnk();
    gemmkit::tuning::set_shared_lhs_mnk(1); // force shared-A on for any parallel run
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        let a = Mat::<f32>::rand(m, k, 0xA1 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0xB2 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0xC3 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k != 1 -> packs A
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

/// Negative strides via the unchecked API (reversed-row view of A)
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

    // A laid out row-major; present it with a *negative* row stride by pointing
    // at the last row and walking backwards. C is also stored that way
    let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k, csa = 1
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let mut cbuf = vec![0.0f64; m * n];

    unsafe {
        let a_last = abuf.as_ptr().add(((m - 1) as isize * rsa) as usize);
        let c_ptr = cbuf.as_mut_ptr();
        // Reversed A rows: base = last row, row stride = -rsa, but C in natural
        // order means C rows also reverse: instead reverse A's rows and B stays,
        // producing C with reversed rows; compare against reversed reference
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
    // gemm computed C[i,j] = sum_k A[m-1-i, k] * B[k,j]; compare to reversed ref
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

// gemv shapes

#[test]
fn gemv_shapes() {
    // n == 1 and m == 1, across layouts
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
