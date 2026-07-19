//! Correctness of every route a `tuning::set_*` knob can steer a GEMM onto. Isolated in its
//! own test binary because the knobs are process-global `AtomicUsize`s; since the harness
//! runs the tests in this binary concurrently, every test that sets a knob holds
//! [`KNOB_LOCK`] (via [`knob_guard`]) for its whole body and restores whatever it changed
//! before returning, so no mutation is ever observed by another test

use gemmkit::{
    MatMut, MatRef, Parallelism, gemm, gemm_batched, gemm_packed_b, prepack_rhs, tuning,
};

/// Serializes every test in this binary that mutates a `tuning::set_*` knob. The knobs are
/// process-global and the harness runs tests in this binary concurrently, so without a
/// shared lock one test's set/restore could interleave with another's gemm call: flipping a
/// route (or the serial/parallel gate) mid-run and breaking a bit-identity or consistency
/// assertion. Every knob-touching test holds this for its whole body and restores the knobs
/// it changed before releasing it, so no mutation is ever observed outside the test that made it
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`KNOB_LOCK`] for the calling test's duration. Recovers a poisoned lock so one
/// panicking test does not cascade into spurious failures across the rest
fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Naive `C = A*B` for column-major `f64` operands, returned column-major: the reference
/// the route/knob tests below compare against
fn naive_col(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for p in 0..k {
                s += a[p * m + i] * b[j * k + p];
            }
            c[j * m + i] = s;
        }
    }
    c
}

/// `usize::MAX` must not collide with the crate's internal "unset" sentinel: setting the
/// maximum should actually take effect (clamped to `usize::MAX - 1`), not be silently
/// ignored and left at the default
#[test]
fn max_value_threshold_takes_effect() {
    let _g = knob_guard();
    let prev = tuning::parallel_threshold();
    tuning::set_parallel_threshold(usize::MAX);
    let got = tuning::parallel_threshold();
    tuning::set_parallel_threshold(prev); // restore: this knob gates the general parallel path
    assert_ne!(
        got,
        48 * 48 * 256,
        "usize::MAX was silently dropped to the default"
    );
    assert_eq!(
        got,
        usize::MAX - 1,
        "should clamp to the largest usable value"
    );
}

/// Both the packed and the in-place (unpacked) RHS paths must be correct,
/// including partial column tiles (n not a multiple of NR). Toggle the gate to
/// force each mode and compare to a naive reference
#[test]
fn rhs_packing_both_modes_correct() {
    let _g = knob_guard();
    for &force in &[0usize, usize::MAX] {
        tuning::set_rhs_pack_threshold(force); // 0 = always pack, MAX = never pack
        for &(m, k, n) in &[(33, 17, 19), (64, 40, 13), (128, 65, 11), (40, 33, 28)] {
            let (a, b) = mkmats(m, k, n);
            let cref = naive_col(&a, &b, m, k, n);
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            assert_close(&cc, &cref, &format!("force={force} {m}x{k}x{n}"));
        }
    }
}

/// `gemv_threshold` is a live knob: setting it to 0 disables the dedicated gemv
/// path, which then falls through to the general driver and stays correct
#[test]
fn gemv_threshold_disables_path_but_stays_correct() {
    let _g = knob_guard();
    tuning::set_gemv_threshold(0);
    // m == 1 row-vector times 5x4 matrix
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0]; // 1x5
    let bm = [
        1.0f64, 0.0, 1.0, 0.0, // row 0
        0.0, 1.0, 0.0, 1.0, // row 1
        2.0, 0.0, 0.0, 0.0, // row 2
        0.0, 3.0, 0.0, 0.0, // row 3
        1.0, 1.0, 1.0, 1.0, // row 4
    ]; // 5x4 row-major
    let mut c = [0.0f64; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 1, 5),
        MatRef::from_row_major(&bm, 5, 4),
        0.0,
        MatMut::from_row_major(&mut c, 1, 4),
        Parallelism::Serial,
    );
    // Reference: c[j] = sum_k a[k]*B[k,j]
    let mut expect = [0.0f64; 4];
    for j in 0..4 {
        for k in 0..5 {
            expect[j] += a[k] * bm[k * 4 + j];
        }
    }
    for j in 0..4 {
        assert!(
            (c[j] - expect[j]).abs() < 1e-12,
            "c[{j}]={} expect {}",
            c[j],
            expect[j]
        );
    }
}

/// Both LHS paths must be correct under parallelism: packed (forced by a zero-byte
/// stride gate, so every column-major A packs) and read-in-place (gate disabled).
/// Exercises the dynamic scheduler's packed-panel grain plus partial row/column tiles,
/// against a naive reference
#[test]
fn lhs_packing_both_modes_correct() {
    let _g = knob_guard();
    // 1 = always pack a column-major A (csa*sizeof >= 1); MAX = never via stride
    // (0 would mean "auto" - derive from page size - so it is not an extreme here)
    for &stride in &[1usize, usize::MAX] {
        tuning::set_lhs_pack_stride(stride);
        for &(m, k, n) in &[(97, 64, 80), (160, 48, 133), (200, 96, 175), (33, 17, 19)] {
            let (a, b) = mkmats(m, k, n);
            let cref = naive_col(&a, &b, m, k, n);
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            assert_close(&cc, &cref, &format!("stride={stride} {m}x{k}x{n}"));
        }
    }
}

/// `parallel_oversample` is a live knob: 0 (clamped to 1), 1, and an adversarially
/// huge value must each yield a correct parallel result with no panic - the latter
/// proves the grain computation's saturating multiply guards against overflow
#[test]
fn parallel_oversample_extremes_stay_correct() {
    let _g = knob_guard();
    let (m, k, n) = (96usize, 80, 64);
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    for &ov in &[0usize, 1, usize::MAX] {
        tuning::set_parallel_oversample(ov);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("oversample={ov}"));
    }
}

/// `small_k_threshold` is a live knob: forcing every shape onto the in-place small-`k`
/// route (`usize::MAX`, since `k <= threshold` then always holds) or onto the register-tiling
/// driver (`0`, since `k <= 0` never holds for `k >= 1`) must both stay correct, across `k`
/// values on both sides of the calibrated crossover and with partial tiles
#[test]
fn small_k_threshold_route_correct() {
    let _g = knob_guard();
    let prev = tuning::small_k_threshold();
    // MAX = every k takes the in-place small-k route; 0 = every k takes the driver
    for &force in &[usize::MAX, 0] {
        tuning::set_small_k_threshold(force);
        for &(m, k, n) in &[
            (33, 3, 19),
            (64, 8, 40),
            (128, 16, 50),
            (40, 20, 28),
            (97, 2, 80),
        ] {
            let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
            let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
            let cref = naive_col(&a, &b, m, k, n);
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            for (got, exp) in cc.iter().zip(&cref) {
                assert!(
                    (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                    "force={force} {m}x{k}x{n}: {got} vs {exp}"
                );
            }
        }
    }
    tuning::set_small_k_threshold(prev);
}

/// `gemv_parallel_bytes` is a live knob: the byte floor forced to `1` (parallelize any gemv)
/// or `usize::MAX` (never) must both produce the correct matrix*vector result. (`0` is *auto*,
/// an LLC-derived floor, not an extreme, so it is not exercised here)
#[test]
fn gemv_parallel_bytes_route_correct() {
    let _g = knob_guard();
    let prev = tuning::gemv_parallel_bytes();
    let (m, k, n) = (2000usize, 3usize, 1usize); // n == 1 gemv shape
    let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
    let cref = naive_col(&a, &b, m, k, n);
    for &force in &[1usize, usize::MAX] {
        tuning::set_gemv_parallel_bytes(force);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        for (got, exp) in cc.iter().zip(&cref) {
            assert!(
                (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                "floor={force}: {got} vs {exp}"
            );
        }
    }
    tuning::set_gemv_parallel_bytes(prev);
}

/// `gemv_thread_cap` is a live knob: capping the bandwidth-bound worker count at `1`
/// (single worker) or a large value (many) must leave a parallel gemv both correct against
/// a naive reference and bit-identical to the serial run (output-partitioning is exact)
#[test]
fn gemv_thread_cap_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::gemv_thread_cap();
    // Drop the byte floor so this modest gemv clears it and the cap actually bites (the
    // assertions are correctness/bit-identity, which hold at any worker count, so racing the
    // floor knob with its own test cannot flake). Restored at the end
    let prev_floor = tuning::gemv_parallel_bytes();
    tuning::set_gemv_parallel_bytes(1);
    let (m, k, n) = (200_000usize, 3usize, 1usize);
    let a: Vec<f64> = (0..m * k).map(|x| (x % 31) as f64 * 0.05 - 0.7).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 17) as f64 * 0.1 - 0.8).collect();
    let cref = naive_col(&a, &b, m, k, n);
    let serial = {
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Serial,
        );
        cc
    };
    for &cap in &[1usize, 64] {
        tuning::set_gemv_thread_cap(cap);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        for ((got, exp), ser) in cc.iter().zip(&cref).zip(&serial) {
            assert!(
                (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                "cap={cap}: {got} vs {exp}"
            );
            assert_eq!(
                got, ser,
                "cap={cap}: parallel gemv must equal serial bit-for-bit"
            );
        }
    }
    tuning::set_gemv_thread_cap(prev);
    tuning::set_gemv_parallel_bytes(prev_floor);
}

/// Output-partitioning of gemv and gevv adds no cross-thread reduction, so a parallel run
/// reproduces the serial one exactly; checked here to a tight tolerance rather than requiring
/// bit-for-bit equality
#[test]
fn gemv_gevv_serial_parallel_consistent() {
    let _g = knob_guard();
    // Drop the byte floor so these modest shapes actually split across workers on any machine
    // (the LLC-derived auto floor would keep them serial on a large-cache host)
    let prev_floor = tuning::gemv_parallel_bytes();
    tuning::set_gemv_parallel_bytes(1);
    // gemv: m*k large enough to split across workers; `m` deliberately not a multiple of the
    // register-block width, so the sub-lane scalar tail is exercised
    let (m, k) = (300_007usize, 5usize);
    let a = mkvec(m * k, 1);
    let x = mkvec(k, 2);
    let run_gemv = |par| {
        let mut c = vec![0.0f32; m];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&x, k, 1),
            0.0,
            MatMut::from_col_major(&mut c, m, 1),
            par,
        );
        c
    };
    assert_consistent(
        &run_gemv(Parallelism::Serial),
        &run_gemv(Parallelism::Rayon(0)),
        "gemv",
    );

    // gevv / skinny GEMM: enough C bytes that the ramp gives several workers; dims not
    // multiples of MR/NR so partial tiles are exercised
    let (gm, gk, gn) = (1201usize, 3usize, 1199usize);
    let ga = mkvec(gm * gk, 3);
    let gb = mkvec(gk * gn, 4);
    let run_gevv = |par| {
        let mut c = vec![0.0f32; gm * gn];
        gemm(
            1.0,
            MatRef::from_col_major(&ga, gm, gk),
            MatRef::from_col_major(&gb, gk, gn),
            0.0,
            MatMut::from_col_major(&mut c, gm, gn),
            par,
        );
        c
    };
    assert_consistent(
        &run_gevv(Parallelism::Serial),
        &run_gevv(Parallelism::Rayon(0)),
        "gevv",
    );
    tuning::set_gemv_parallel_bytes(prev_floor);
}

/// Naive `C = A*B` for **row-major A + column-major B**, returned column-major: the small-`m,n`
/// horizontal route's contiguous-along-`k` layout. `A[i,p] = a[i*k+p]`, `B[p,j] = b[j*k+p]`,
/// `C[i,j] = c[j*m+i]`
fn naive_rowa_colb(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            c[j * m + i] = s;
        }
    }
    c
}

/// `small_mn_dim` is a live knob: forcing every small-`m,n` shape onto the horizontal route
/// (`usize::MAX`) or off it (`0`, which disables the route entirely) must both stay correct,
/// across `k` (all above the default `small_k_threshold`, so the route's `k > threshold` gate
/// fires) with partial tiles (`m,n,k` not multiples of the register tile) and a non-trivial
/// `alpha`/`beta`
#[test]
fn small_mn_route_correct() {
    let _g = knob_guard();
    let pmn = tuning::small_mn_dim();
    // MAX = every small-m,n shape takes the horizontal route; 0 = the driver
    for &smn in &[usize::MAX, 0usize] {
        tuning::set_small_mn_dim(smn);
        for &(m, k, n) in &[
            (6, 20, 7),
            (10, 100, 13),
            (3, 50, 5),
            (16, 4096, 16),
            (4, 17, 4),
            (2, 33, 8),
        ] {
            let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
            let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
            let ab = naive_rowa_colb(&a, &b, m, k, n);
            // alpha/beta epilogue over a pre-filled column-major C
            let (alpha, beta) = (2.5f64, -0.5f64);
            let mut cc: Vec<f64> = (0..m * n).map(|x| (x % 7) as f64 * 0.3 - 0.9).collect();
            let cref: Vec<f64> = (0..m * n)
                .map(|idx| alpha * ab[idx] + beta * cc[idx])
                .collect();
            gemm(
                alpha,
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                beta,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            for (got, exp) in cc.iter().zip(&cref) {
                assert!(
                    (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                    "smn={smn} {m}x{k}x{n}: {got} vs {exp}"
                );
            }
        }
    }
    tuning::set_small_mn_dim(pmn);
}

/// The mixed-precision (f16, f32-accumulate) horizontal route must be correct forced on
/// (`usize::MAX`) and off (`0` -> the widen driver), across `k` with partial tiles. Compared to an
/// f32 reference with a tolerance (the narrow output rounds once in the epilogue)
#[cfg(feature = "half")]
#[test]
fn small_mn_mixed_route_correct() {
    use gemmkit::f16;
    let _g = knob_guard();
    let pmn = tuning::small_mn_dim();
    for &smn in &[usize::MAX, 0usize] {
        tuning::set_small_mn_dim(smn);
        for &(m, k, n) in &[(6, 20, 7), (16, 4096, 16), (4, 17, 4), (2, 33, 8)] {
            let af: Vec<f32> = (0..m * k).map(|x| (x % 23) as f32 * 0.1 - 1.0).collect();
            let bf: Vec<f32> = (0..k * n).map(|x| (x % 19) as f32 * 0.2 - 1.5).collect();
            let a: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
            let b: Vec<f16> = bf.iter().map(|&x| f16::from_f32(x)).collect();
            // f32 reference over the widened inputs (row-major A, col-major B -> col-major C)
            let cref: Vec<f32> = {
                let mut c = vec![0.0f32; m * n];
                for j in 0..n {
                    for i in 0..m {
                        let mut s = 0.0f32;
                        for p in 0..k {
                            s += a[i * k + p].to_f32() * b[j * k + p].to_f32();
                        }
                        c[j * m + i] = s;
                    }
                }
                c
            };
            let mut cc = vec![f16::from_f32(0.0); m * n];
            gemm(
                f16::from_f32(1.0),
                MatRef::from_row_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                f16::from_f32(0.0),
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            for (got, exp) in cc.iter().zip(&cref) {
                let g = got.to_f32();
                assert!(
                    (g - exp).abs() <= 1e-2 * (1.0 + exp.abs()),
                    "smn={smn} f16 {m}x{k}x{n}: {g} vs {exp}"
                );
            }
        }
    }
    tuning::set_small_mn_dim(pmn);
}

/// The horizontal route output-partitions disjoint tiles with no cross-thread reduction, so a
/// parallel run must equal the serial run **bit-for-bit**. Force the route on and drop the
/// bandwidth floor so a `16x16` output actually splits across workers
#[test]
fn small_mn_serial_parallel_bit_identical() {
    let _g = knob_guard();
    let pmn = tuning::small_mn_dim();
    let pfloor = tuning::gemv_parallel_bytes();
    tuning::set_small_mn_dim(usize::MAX); // k = 4096 > default small_k_threshold, so it routes here
    tuning::set_gemv_parallel_bytes(1); // clear the LLC floor so the small output splits
    let (m, k, n) = (16usize, 4096usize, 16usize);
    let a = mkvec(m * k, 7);
    let b = mkvec(k * n, 8);
    let run = |par| {
        let mut c = vec![0.0f32; m * n];
        gemm(
            1.0,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut c, m, n),
            par,
        );
        c
    };
    let serial = run(Parallelism::Serial);
    let parallel = run(Parallelism::Rayon(0));
    assert_eq!(
        serial, parallel,
        "horizontal route: parallel must equal serial bit-for-bit"
    );
    tuning::set_small_mn_dim(pmn);
    tuning::set_gemv_parallel_bytes(pfloor);
}

/// The horizontal route's PACK tier (a small-`m,n` shape whose operand is strided along `k`) must be
/// **bit-identical** to the zero-copy eligible layout for the same values: the pre-pack is a pure
/// reorder, so each cell's dot reads the same numbers in the same order and lands on the same bits.
/// This macro builds one such test for a real float (`f32`/`f64`), sweeping pack-A (col-major A) /
/// pack-B (row-major B) / pack-both, tail shapes (`m,n,k` not multiples of the register tile), an
/// alpha/beta sweep, and serial + parallel. All 4 layouts carry identical logical `A`/`B` values, so
/// every result must equal the eligible one to the bit
macro_rules! small_mn_pack_bit_identical {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let _g = knob_guard();
            let pmn = tuning::small_mn_dim();
            let psk = tuning::small_k_threshold();
            let ppk = tuning::small_mn_pack_min_k();
            let pfl = tuning::gemv_parallel_bytes();
            tuning::set_small_mn_dim(usize::MAX); // route every small-m,n shape to the horizontal path
            tuning::set_small_k_threshold(4); // k below 5 would take small_k; every k here is > 4
            tuning::set_small_mn_pack_min_k(0); // pack tier fires for every k > 0
            tuning::set_gemv_parallel_bytes(1); // drop the bandwidth floor so parallel actually forks

            for &(m, k, n) in &[
                (4, 64, 4),
                (16, 4096, 16),
                (5, 40, 7),
                (3, 17, 8),
                (13, 100, 11),
            ] {
                // Logical A (m x k) row-major and B (k x n) col-major (the eligible forms), plus the
                // same values re-laid-out col-major A / row-major B (the ineligible forms)
                let a_rm: Vec<$t> = (0..m * k).map(|x| (x % 23) as $t * 0.1 - 1.0).collect(); // A[i,t] = a_rm[i*k+t]
                let b_cm: Vec<$t> = (0..k * n).map(|x| (x % 19) as $t * 0.2 - 1.5).collect(); // B[t,j] = b_cm[j*k+t]
                let mut a_cm = vec![0.0 as $t; m * k]; // col-major A: a_cm[t*m+i] = A[i,t]
                for i in 0..m {
                    for t in 0..k {
                        a_cm[t * m + i] = a_rm[i * k + t];
                    }
                }
                let mut b_rm = vec![0.0 as $t; k * n]; // row-major B: b_rm[t*n+j] = B[t,j]
                for t in 0..k {
                    for j in 0..n {
                        b_rm[t * n + j] = b_cm[j * k + t];
                    }
                }
                let c0: Vec<$t> = (0..m * n).map(|x| (x % 7) as $t * 0.3 - 0.9).collect();

                for &(alpha, beta) in &[(1.0 as $t, 0.0 as $t), (2.5, -0.5), (1.0, 1.0), (0.0, 2.0)]
                {
                    for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
                        // `arm`/`brm` pick row-major vs col-major A/B, covering the 4 layouts; C
                        // stays col-major so orient_transpose never swaps
                        let run = |arm: bool, brm: bool| -> Vec<<$t as SmnBits>::Bits> {
                            let mut c = c0.clone();
                            let av = if arm {
                                MatRef::from_row_major(&a_rm, m, k)
                            } else {
                                MatRef::from_col_major(&a_cm, m, k)
                            };
                            let bv = if brm {
                                MatRef::from_row_major(&b_rm, k, n)
                            } else {
                                MatRef::from_col_major(&b_cm, k, n)
                            };
                            gemm(
                                alpha,
                                av,
                                bv,
                                beta,
                                MatMut::from_col_major(&mut c, m, n),
                                par,
                            );
                            c.iter().map(|v| v.smn_bits()).collect()
                        };
                        let elig = run(true, false); // row-major A + col-major B: zero-copy
                        let ctx = format!("{m}x{k}x{n} a={alpha} b={beta} {par:?}");
                        assert_eq!(run(false, false), elig, "pack-A != eligible {ctx}");
                        assert_eq!(run(true, true), elig, "pack-B != eligible {ctx}");
                        assert_eq!(run(false, true), elig, "pack-both != eligible {ctx}");
                    }
                }
            }
            tuning::set_small_mn_dim(pmn);
            tuning::set_small_k_threshold(psk);
            tuning::set_small_mn_pack_min_k(ppk);
            tuning::set_gemv_parallel_bytes(pfl);
        }
    };
}

/// Raw-bit view used by the pack-parity tests to compare results exactly (a byte compare, so a `NaN`
/// or a `-0.0` still has to match)
trait SmnBits {
    type Bits: PartialEq + core::fmt::Debug;
    fn smn_bits(&self) -> Self::Bits;
}
impl SmnBits for f32 {
    type Bits = u32;
    fn smn_bits(&self) -> u32 {
        self.to_bits()
    }
}
impl SmnBits for f64 {
    type Bits = u64;
    fn smn_bits(&self) -> u64 {
        self.to_bits()
    }
}

small_mn_pack_bit_identical!(small_mn_pack_bit_identical_f32, f32);
small_mn_pack_bit_identical!(small_mn_pack_bit_identical_f64, f64);

/// Mixed-precision (`f16`, `f32`-accumulate) pack tier: like the real-float parity, the pre-pack is
/// a pure reorder of the narrow inputs, so the widen-load dot reads the same `f16` bits in the same
/// order and narrows to the same output bits. Bit-identical to the eligible layout (`f16` output
/// bits compared exactly), across pack-A / pack-B / pack-both, tails, alpha/beta, serial + parallel
#[cfg(feature = "half")]
#[test]
fn small_mn_pack_bit_identical_f16() {
    use gemmkit::f16;
    let _g = knob_guard();
    let pmn = tuning::small_mn_dim();
    let psk = tuning::small_k_threshold();
    let ppk = tuning::small_mn_pack_min_k();
    let pfl = tuning::gemv_parallel_bytes();
    tuning::set_small_mn_dim(usize::MAX);
    tuning::set_small_k_threshold(4);
    tuning::set_small_mn_pack_min_k(0);
    tuning::set_gemv_parallel_bytes(1);

    let h = |x: usize, m: usize, s: f32| f16::from_f32((x % m) as f32 * s);
    for &(m, k, n) in &[(4, 64, 4), (16, 512, 16), (5, 40, 7), (3, 17, 8)] {
        let a_rm: Vec<f16> = (0..m * k).map(|x| h(x, 23, 0.03)).collect();
        let b_cm: Vec<f16> = (0..k * n).map(|x| h(x, 19, 0.05)).collect();
        let mut a_cm = vec![f16::from_f32(0.0); m * k];
        for i in 0..m {
            for t in 0..k {
                a_cm[t * m + i] = a_rm[i * k + t];
            }
        }
        let mut b_rm = vec![f16::from_f32(0.0); k * n];
        for t in 0..k {
            for j in 0..n {
                b_rm[t * n + j] = b_cm[j * k + t];
            }
        }
        let c0: Vec<f16> = (0..m * n).map(|x| h(x, 11, 0.1)).collect();
        for &(alpha, beta) in &[(1.0f32, 0.0f32), (1.25, 1.0), (0.0, 2.0)] {
            let (alpha, beta) = (f16::from_f32(alpha), f16::from_f32(beta));
            for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
                let run = |arm: bool, brm: bool| -> Vec<u16> {
                    let mut c = c0.clone();
                    let av = if arm {
                        MatRef::from_row_major(&a_rm, m, k)
                    } else {
                        MatRef::from_col_major(&a_cm, m, k)
                    };
                    let bv = if brm {
                        MatRef::from_row_major(&b_rm, k, n)
                    } else {
                        MatRef::from_col_major(&b_cm, k, n)
                    };
                    gemm(
                        alpha,
                        av,
                        bv,
                        beta,
                        MatMut::from_col_major(&mut c, m, n),
                        par,
                    );
                    c.iter().map(|v| v.to_bits()).collect()
                };
                let elig = run(true, false);
                let ctx = format!("f16 {m}x{k}x{n} {par:?}");
                assert_eq!(run(false, false), elig, "pack-A != eligible {ctx}");
                assert_eq!(run(true, true), elig, "pack-B != eligible {ctx}");
                assert_eq!(run(false, true), elig, "pack-both != eligible {ctx}");
            }
        }
    }
    tuning::set_small_mn_dim(pmn);
    tuning::set_small_k_threshold(psk);
    tuning::set_small_mn_pack_min_k(ppk);
    tuning::set_gemv_parallel_bytes(pfl);
}

/// Integer (`i8 -> i32`) pack tier: the pre-pack is a pure reorder of the `i8` inputs, so the widen
/// dot lands on the exact same wrapping-`i32` output. Exactly equal to the eligible layout, across
/// pack-A / pack-B / pack-both, tails, alpha/beta, serial + parallel
#[cfg(feature = "int8")]
#[test]
fn small_mn_pack_exact_i8() {
    use gemmkit::gemm_i8;
    let _g = knob_guard();
    let pmn = tuning::small_mn_dim();
    let psk = tuning::small_k_threshold();
    let ppk = tuning::small_mn_pack_min_k();
    let pfl = tuning::gemv_parallel_bytes();
    tuning::set_small_mn_dim(usize::MAX);
    tuning::set_small_k_threshold(4);
    tuning::set_small_mn_pack_min_k(0);
    tuning::set_gemv_parallel_bytes(1);

    for &(m, k, n) in &[(4, 64, 4), (16, 4096, 16), (5, 40, 7), (3, 17, 8)] {
        let a_rm: Vec<i8> = (0..m * k).map(|x| (x % 17) as i8 - 8).collect();
        let b_cm: Vec<i8> = (0..k * n).map(|x| (x % 13) as i8 - 6).collect();
        let mut a_cm = vec![0i8; m * k];
        for i in 0..m {
            for t in 0..k {
                a_cm[t * m + i] = a_rm[i * k + t];
            }
        }
        let mut b_rm = vec![0i8; k * n];
        for t in 0..k {
            for j in 0..n {
                b_rm[t * n + j] = b_cm[j * k + t];
            }
        }
        let c0: Vec<i32> = (0..m * n).map(|x| (x % 5) as i32 - 2).collect();
        for &(alpha, beta) in &[(1i32, 0i32), (3, -1), (1, 1), (0, 2)] {
            for &par in &[Parallelism::Serial, Parallelism::Rayon(0)] {
                let run = |arm: bool, brm: bool| -> Vec<i32> {
                    let mut c = c0.clone();
                    let av = if arm {
                        MatRef::from_row_major(&a_rm, m, k)
                    } else {
                        MatRef::from_col_major(&a_cm, m, k)
                    };
                    let bv = if brm {
                        MatRef::from_row_major(&b_rm, k, n)
                    } else {
                        MatRef::from_col_major(&b_cm, k, n)
                    };
                    gemm_i8(
                        alpha,
                        av,
                        bv,
                        beta,
                        MatMut::from_col_major(&mut c, m, n),
                        par,
                    );
                    c
                };
                let elig = run(true, false);
                let ctx = format!("i8 {m}x{k}x{n} a={alpha} b={beta} {par:?}");
                assert_eq!(run(false, false), elig, "pack-A != eligible {ctx}");
                assert_eq!(run(true, true), elig, "pack-B != eligible {ctx}");
                assert_eq!(run(false, true), elig, "pack-both != eligible {ctx}");
            }
        }
    }
    tuning::set_small_mn_dim(pmn);
    tuning::set_small_k_threshold(psk);
    tuning::set_small_mn_pack_min_k(ppk);
    tuning::set_gemv_parallel_bytes(pfl);
}

// Remaining calibration knobs: each test below forces an extreme (plus, where relevant, the
// default) and checks correctness

/// Assert `got` matches a naive reference to a tight relative tolerance, with a labelled message
fn assert_close(got: &[f64], expect: &[f64], label: &str) {
    assert_eq!(got.len(), expect.len(), "{label}: length mismatch");
    for (g, e) in got.iter().zip(expect) {
        assert!(
            (g - e).abs() <= 1e-10 * (1.0 + e.abs()),
            "{label}: {g} vs {e}"
        );
    }
}

/// Deterministic column-major `f64` operands for a knob test
fn mkmats(m: usize, k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..m * k).map(|x| (x % 23) as f64 * 0.1 - 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 19) as f64 * 0.2 - 1.5).collect();
    (a, b)
}

/// `k_stream_max` gates the axpy-gemv output register-blocking strategy. Forcing it to `0` (never
/// register-block) or `usize::MAX` (always, even once the output spills L2) must both give the
/// correct gemv result. `n == 1` selects the axpy gemv path
#[test]
fn k_stream_max_route_correct() {
    let _g = knob_guard();
    let prev = tuning::k_stream_max();
    // A large output (spills L2 on typical machines, so the register-block path is actually taken
    // for the non-zero setting) and a small `k` (register-block eligible). Correctness holds either
    // way
    let (m, k, n) = (300_000usize, 5usize, 1usize);
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    for &force in &[0usize, usize::MAX] {
        tuning::set_k_stream_max(force);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Serial,
        );
        assert_close(&cc, &cref, &format!("k_stream_max={force}"));
    }
    tuning::set_k_stream_max(prev);
}

/// `seq_internal_bytes_per_worker` selects the aarch64 batched-GEMM split plan (inert on other
/// targets, but still exercised here for correctness on whatever host runs the test). Forcing it
/// to `1` or `usize::MAX` must leave a batched GEMM correct against a naive per-element reference
#[test]
fn seq_internal_bytes_batched_correct() {
    let _g = knob_guard();
    let prev = tuning::seq_internal_bytes_per_worker();
    let (batch, m, k, n) = (6usize, 40usize, 48usize, 40usize);
    let a: Vec<f64> = (0..batch * m * k)
        .map(|x| (x % 23) as f64 * 0.1 - 1.0)
        .collect();
    let b: Vec<f64> = (0..batch * k * n)
        .map(|x| (x % 19) as f64 * 0.2 - 1.5)
        .collect();
    // Per-element naive reference (each element is an independent col-major product)
    let mut cref = vec![0.0f64; batch * m * n];
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        let e = naive_col(&a[ao..ao + m * k], &b[bo..bo + k * n], m, k, n);
        cref[co..co + m * n].copy_from_slice(&e);
    }
    for &force in &[1usize, usize::MAX] {
        tuning::set_seq_internal_bytes_per_worker(force);
        let mut c = vec![0.0f64; batch * m * n];
        gemm_batched(
            batch,
            1.0,
            MatRef::new(&a, m, k, 1, m as isize),
            (m * k) as isize,
            MatRef::new(&b, k, n, 1, k as isize),
            (k * n) as isize,
            0.0,
            MatMut::new(&mut c, m, n, 1, m as isize),
            (m * n) as isize,
            Parallelism::Rayon(0),
        );
        assert_close(&c, &cref, &format!("seq_internal_bytes={force}"));
    }
    tuning::set_seq_internal_bytes_per_worker(prev);
}

/// `packed_oversample` sets the packed-LHS path's dynamic-scheduling grain. A row-major A
/// (`rsa != 1`) always packs regardless of reuse, so it reliably takes that path. Forcing the
/// knob to `0` (clamped to 1), `1`, or a huge value must each give a correct parallel result
#[test]
fn packed_oversample_extremes_stay_correct() {
    let _g = knob_guard();
    let prev = tuning::packed_oversample();
    // `m` large enough that the row-block count reaches the worker count on typical machines, so
    // `packed_block_grain` is actually consulted; correctness holds regardless
    let (m, k, n) = (4096usize, 64usize, 96usize);
    let (a, b) = mkmats(m, k, n); // A read row-major, B col-major -> col-major C
    let cref = naive_rowa_colb(&a, &b, m, k, n);
    for &ov in &[0usize, 1, usize::MAX] {
        tuning::set_packed_oversample(ov);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("packed_oversample={ov}"));
    }
    tuning::set_packed_oversample(prev);
}

/// `mc_reg_panels` caps the A macro-panel height at `this * MR`. Forcing it to `1` (the
/// smallest possible MC) or a huge value (the cap no longer binds, so MC falls back to the
/// cache-derived estimate) must keep a general GEMM correct
#[test]
fn mc_reg_panels_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::mc_reg_panels();
    for &force in &[1usize, usize::MAX] {
        tuning::set_mc_reg_panels(force);
        for &(m, k, n) in &[(200usize, 96usize, 175usize), (160, 48, 133)] {
            let (a, b) = mkmats(m, k, n);
            let cref = naive_col(&a, &b, m, k, n);
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            assert_close(&cc, &cref, &format!("mc_reg_panels={force} {m}x{k}x{n}"));
        }
    }
    tuning::set_mc_reg_panels(prev);
}

/// `nc_no_l3_panels` caps the no-L3 column block (inert where an L3 exists). Forcing it to `1` or a
/// huge value must keep a general GEMM correct on any machine
#[test]
fn nc_no_l3_panels_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::nc_no_l3_panels();
    for &force in &[1usize, usize::MAX] {
        tuning::set_nc_no_l3_panels(force);
        let (m, k, n) = (160usize, 64usize, 200usize);
        let (a, b) = mkmats(m, k, n);
        let cref = naive_col(&a, &b, m, k, n);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("nc_no_l3_panels={force}"));
    }
    tuning::set_nc_no_l3_panels(prev);
}

/// `tiny_block_dim` gates the small-matrix blocking shortcut. Forcing every shape onto the tiny
/// branch (`usize::MAX`) or off it (`0`) must keep a plain GEMM correct, and, because the prepack
/// paths derive their own branch-dodging sentinel (`gate + 1`) from this knob, a prepacked-B GEMM
/// must stay correct even when the knob would otherwise route the shape into the tiny branch
#[test]
fn tiny_block_dim_route_correct() {
    let _g = knob_guard();
    let prev = tuning::tiny_block_dim();
    for &force in &[usize::MAX, 0usize] {
        tuning::set_tiny_block_dim(force);
        for &(m, k, n) in &[(40usize, 33usize, 28usize), (128, 65, 96)] {
            let (a, b) = mkmats(m, k, n);
            let cref = naive_col(&a, &b, m, k, n);
            let mut cc = vec![0.0f64; m * n];
            gemm(
                1.0,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                0.0,
                MatMut::from_col_major(&mut cc, m, n),
                Parallelism::Rayon(0),
            );
            assert_close(&cc, &cref, &format!("tiny_block_dim={force} {m}x{k}x{n}"));
        }
    }
    // Prepack coupling: with the gate forced huge, a normal `gemm()` on this shape would take the
    // tiny branch, but the prepack sentinel (`gate + 1`) still dodges it so the prepacked geometry
    // is valid
    tuning::set_tiny_block_dim(usize::MAX);
    let (m, k, n) = (32usize, 33usize, 28usize);
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    let packed = prepack_rhs(MatRef::from_col_major(&b, k, n));
    let mut cc = vec![0.0f64; m * n];
    gemm_packed_b(
        1.0,
        MatRef::from_col_major(&a, m, k),
        &packed,
        0.0,
        MatMut::from_col_major(&mut cc, m, n),
        Parallelism::Rayon(0),
    );
    assert_close(&cc, &cref, "tiny_block_dim prepack coupling");
    tuning::set_tiny_block_dim(prev);
}

/// `kc` caps the tiny-branch depth block. On a small-matrix shape (which takes the tiny branch),
/// forcing it to `1` (many depth panels) or a huge value (`kc == k`) must both stay correct
#[test]
fn kc_tiny_ceiling_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::kc();
    let (m, k, n) = (40usize, 200usize, 40usize); // m,n <= default tiny gate -> tiny branch
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    for &force in &[1usize, usize::MAX] {
        tuning::set_kc(force);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("kc={force}"));
    }
    tuning::set_kc(prev);
}

/// `kc_min` floors the main-model depth block. On a general shape, forcing it to `1` (the L1
/// estimate stands) or a huge value (`kc == k`) must both stay correct
#[test]
fn kc_min_floor_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::kc_min();
    let (m, k, n) = (200usize, 300usize, 175usize);
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    for &force in &[1usize, usize::MAX] {
        tuning::set_kc_min(force);
        let mut cc = vec![0.0f64; m * n];
        gemm(
            1.0,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("kc_min={force}"));
    }
    tuning::set_kc_min(prev);
}

/// `pack_transpose_tile` sets the strip length of the cache-blocked transpose in the strided
/// packing path: only the copy order changes, not the packed bytes. A column-major B always takes
/// that strided path when it is prepacked (its column stride is not 1), so a prepacked-B GEMM must
/// be correct at `1`, the default, and a huge strip length
#[test]
fn pack_transpose_tile_stays_correct() {
    let _g = knob_guard();
    let prev = tuning::pack_transpose_tile();
    let (m, k, n) = (200usize, 96usize, 128usize);
    let (a, b) = mkmats(m, k, n);
    let cref = naive_col(&a, &b, m, k, n);
    for &tile in &[1usize, 16, usize::MAX] {
        tuning::set_pack_transpose_tile(tile);
        let packed = prepack_rhs(MatRef::from_col_major(&b, k, n));
        let mut cc = vec![0.0f64; m * n];
        gemm_packed_b(
            1.0,
            MatRef::from_col_major(&a, m, k),
            &packed,
            0.0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        assert_close(&cc, &cref, &format!("pack_transpose_tile={tile}"));
    }
    tuning::set_pack_transpose_tile(prev);
}

/// `i8_vnni_min_par_mnk` gates the VNNI->widen small-parallel fallback. Forcing it to `0` (always
/// keep VNNI) or `usize::MAX` (always fall back to widen for a multi-threaded run) must both give
/// the correct i32 product; the 2 kernels compute exact i32 arithmetic, so they must also agree
/// bit-for-bit
#[cfg(feature = "int8")]
#[test]
fn i8_vnni_min_par_mnk_route_correct() {
    let _g = knob_guard();
    let prev = tuning::i8_vnni_min_par_mnk();
    let (m, k, n) = (128usize, 128usize, 128usize); // above the parallel gate, so Rayon truly splits
    let a: Vec<i8> = (0..m * k).map(|x| ((x % 17) as i8) - 8).collect();
    let b: Vec<i8> = (0..k * n).map(|x| ((x % 13) as i8) - 6).collect();
    let mut cref = vec![0i32; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut s = 0i32;
            for p in 0..k {
                s = s.wrapping_add(a[p * m + i] as i32 * b[j * k + p] as i32);
            }
            cref[j * m + i] = s;
        }
    }
    let run = |force: usize| {
        tuning::set_i8_vnni_min_par_mnk(force);
        let mut cc = vec![0i32; m * n];
        gemmkit::gemm_i8(
            1,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0,
            MatMut::from_col_major(&mut cc, m, n),
            Parallelism::Rayon(0),
        );
        cc
    };
    let vnni = run(0);
    let widen = run(usize::MAX);
    tuning::set_i8_vnni_min_par_mnk(prev);
    assert_eq!(vnni, cref, "i8_vnni_min_par_mnk=0 (VNNI) wrong");
    assert_eq!(widen, cref, "i8_vnni_min_par_mnk=MAX (widen) wrong");
    assert_eq!(
        vnni, widen,
        "VNNI and widen i8 kernels must be bit-identical"
    );
}

/// Assert a serial and a parallel result agree to a tight relative tolerance. Within one route
/// they are bit-identical (output-partitioning reorders nothing); the tolerance only absorbs
/// the last-ULP gap when a raced routing knob lands the 2 runs on different paths
fn assert_consistent(serial: &[f32], parallel: &[f32], what: &str) {
    assert_eq!(serial.len(), parallel.len(), "{what}: length mismatch");
    for (i, (&s, &p)) in serial.iter().zip(parallel).enumerate() {
        let tol = 1e-4 * s.abs().max(p.abs()) + 1e-6;
        assert!(
            (s - p).abs() <= tol,
            "{what}: element {i} diverged beyond tolerance: serial={s} parallel={p} (tol {tol})"
        );
    }
}

/// Small deterministic `f32` fill (a xorshift, so the values are not all equal and the
/// reductions are non-trivial) for the consistency test
fn mkvec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}
