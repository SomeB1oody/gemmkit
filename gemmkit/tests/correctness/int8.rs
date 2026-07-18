//! Integer GEMM (i8 -> i32)

use crate::common::*;
use gemmkit::Parallelism;

#[cfg(feature = "int8")]
#[test]
fn correctness_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [
        (1, 1, 1),
        (3, 4, 5),
        (16, 8, 7),
        (32, 32, 32),
        (33, 17, 19),
        (40, 33, 28),
        (64, 80, 48),
        (65, 64, 64),
        (128, 96, 112),
    ] {
        for &(alpha, beta) in &[(1i32, 0i32), (1, 1), (3, -2), (0, 3)] {
            let a = rand_i8(m * k, 0x100 + (m * 7 + k) as u64);
            let b = rand_i8(k * n, 0x200 + (n * 3 + k) as u64);
            let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();
            let cref = ref_i8(&a, &b, &c0, m, k, n, alpha, beta);

            // Row-major A, column-major B, row-major C32
            let bcol: Vec<i8> = {
                let mut v = vec![0i8; k * n];
                for i in 0..k {
                    for j in 0..n {
                        v[j * k + i] = b[i * n + j];
                    }
                }
                v
            };
            let mut c = c0.clone();
            gemmkit::gemm_i8(
                alpha,
                MatRef::from_row_major(&a, m, k),
                MatRef::new(&bcol, k, n, 1, k as isize),
                beta,
                MatMut::from_row_major(&mut c, m, n),
                Parallelism::Serial,
            );
            assert_eq!(c, cref, "i8 mismatch {m}x{k}x{n} alpha={alpha} beta={beta}");
        }
    }
}

/// The `small_mn` horizontal i8 route (both `m, n` small, long `k`, A rows / B cols unit-stride
/// along `k`) must be **bit-exact** vs the register-tiling driver: `i32` is a ring, so the single
/// fixed-order widen dot and the driver's panel-split accumulation land on the same wrapping `i32`.
/// Straddle the default `small_mn_dim` gate (`m, n` in `1..=17`, so `17` in either axis spills to
/// the driver), tail-sized `k` (`small_k_threshold + 1`, not a multiple of the tile), every
/// alpha/beta sign, and serial + parallel. For each shape BOTH the eligible layout (row-major A +
/// col-major B, which the orientation swap re-orients into `small_mn`'s unit-stride operands) and
/// an ineligible layout (row-major B, whose post-swap A columns are strided along `k`, so the gate
/// rejects it and the driver runs) are checked against the exact `i64` reference. No tuning-knob
/// flip, so the route is selected by the real dispatch and it cannot perturb concurrent tests
#[cfg(feature = "int8")]
#[test]
fn i8_small_mn_matches_reference() {
    use gemmkit::{MatMut, MatRef};
    let kt = gemmkit::tuning::small_k_threshold();
    // Straddle `small_mn_dim` (16): tail sizes (not multiples of the 4x4 register tile) drive the
    // edge-cell dot, `1` the degenerate single row/col, `17` the spill to the driver
    let dims: &[usize] = if fast_test() {
        &[1, 3, 4, 7, 16, 17]
    } else {
        &[1, 2, 3, 4, 5, 7, 8, 13, 16, 17]
    };
    let ks: &[usize] = if fast_test() {
        &[kt + 1]
    } else {
        &[kt + 1, 4096]
    };
    for &m in dims {
        for &n in dims {
            for &k in ks {
                let a = rand_i8(m * k, 0x510 + (m * 131 + n * 7) as u64);
                let b = rand_i8(k * n, 0x620 + (n * 17 + k * 3) as u64);
                // Column-major B (rsb=1, csb=k): the eligible-layout twin of the row-major B
                let bcol: Vec<i8> = {
                    let mut v = vec![0i8; k * n];
                    for p in 0..k {
                        for j in 0..n {
                            v[p + j * k] = b[p * n + j];
                        }
                    }
                    v
                };
                for &(alpha, beta) in &[(1i32, 0i32), (1, 1), (2, 2), (0, 1), (3, -1)] {
                    let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();
                    let cref = ref_i8(&a, &b, &c0, m, k, n, alpha, beta);
                    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                        // Eligible: row-major A + col-major B + row-major C => `small_mn` for
                        // m,n <= 16 (m or n == 17 spills to the driver); both must equal the ref
                        let mut c_h = c0.clone();
                        gemmkit::gemm_i8(
                            alpha,
                            MatRef::from_row_major(&a, m, k),
                            MatRef::new(&bcol, k, n, 1, k as isize),
                            beta,
                            MatMut::from_row_major(&mut c_h, m, n),
                            par,
                        );
                        assert_eq!(
                            c_h, cref,
                            "i8 small_mn (eligible) {m}x{k}x{n} alpha={alpha} beta={beta} {par:?}"
                        );
                        // Ineligible: row-major B => the driver route on the same math
                        let mut c_d = c0.clone();
                        gemmkit::gemm_i8(
                            alpha,
                            MatRef::from_row_major(&a, m, k),
                            MatRef::from_row_major(&b, k, n),
                            beta,
                            MatMut::from_row_major(&mut c_d, m, n),
                            par,
                        );
                        assert_eq!(
                            c_d, cref,
                            "i8 driver (ineligible) {m}x{k}x{n} alpha={alpha} beta={beta} {par:?}"
                        );
                    }
                }
            }
        }
    }
}

/// `gemm_i8_unchecked_with` (raw pointers + a caller-owned `Workspace`) must equal `gemm_i8`.
/// It is the FFI/adapter-facing signature and the missing `_with` sibling for the reuse-workspace path
#[cfg(feature = "int8")]
#[test]
fn i8_unchecked_with_matches_gemm_i8() {
    use gemmkit::{Workspace, gemm_i8_unchecked_with};
    let mut ws = Workspace::new();
    for (m, k, n) in [(3usize, 4, 5), (32, 32, 32), (65, 64, 64)] {
        for &(alpha, beta) in &[(1i32, 0i32), (3, -2)] {
            let a = rand_i8(m * k, 0x100 + (m * 7 + k) as u64);
            let b = rand_i8(k * n, 0x200 + (n * 3 + k) as u64);
            let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();
            let cref = ref_i8(&a, &b, &c0, m, k, n, alpha, beta);

            // Column-major B for the row-major-A / col-major-B / row-major-C layout
            let bcol: Vec<i8> = {
                let mut v = vec![0i8; k * n];
                for i in 0..k {
                    for j in 0..n {
                        v[j * k + i] = b[i * n + j];
                    }
                }
                v
            };
            let mut c = c0.clone();
            // SAFETY: A row-major (rsa=k, csa=1), B col-major (rsb=1, csb=k), C row-major
            // (rsc=n, csc=1); all in bounds, distinct buffers, C doesn't alias A/B
            unsafe {
                gemm_i8_unchecked_with(
                    &mut ws,
                    m,
                    k,
                    n,
                    alpha,
                    a.as_ptr(),
                    k as isize,
                    1,
                    bcol.as_ptr(),
                    1,
                    k as isize,
                    beta,
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                    Parallelism::Serial,
                );
            }
            assert_eq!(
                c, cref,
                "i8_unchecked_with mismatch {m}x{k}x{n} alpha={alpha} beta={beta}"
            );
        }
    }
}

/// The documented i8 contract is *wrapping* i32 arithmetic on overflow. Every other
/// i8 test keeps values in range so the wrap never fires; force it here with a large
/// `alpha` and check against a wrapping-i32 reference (not the range-checked `ref_i8`)
#[cfg(feature = "int8")]
#[test]
fn i8_wraps_on_overflow() {
    use gemmkit::{MatMut, MatRef};
    // 2x2x2, all 127s: each dot = 127*127 + 127*127 = 32258 (fits i32). A large `alpha`
    // then overflows the i32 epilogue; 2x2 stays on the general kernel drain (not the
    // gemv path), exercising the scalar `wrapping_mul`/`wrapping_add`
    let a = [127i8; 4];
    let b = [127i8; 4];
    let c0 = [1_000_000i32, -2_000_000, 3_000_000, -4_000_000];
    let (alpha, beta) = (100_000i32, 1i32);
    let acc: i32 = 127 * 127 + 127 * 127;
    assert!(
        (alpha as i64) * (acc as i64) > i32::MAX as i64,
        "test setup must actually overflow i32"
    );
    let scaled = alpha.wrapping_mul(acc);
    let want: Vec<i32> = c0
        .iter()
        .map(|&c| beta.wrapping_mul(c).wrapping_add(scaled))
        .collect();
    let mut c = c0;
    gemmkit::gemm_i8(
        alpha,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        beta,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
    assert_eq!(
        c.to_vec(),
        want,
        "i8 must wrap (two's complement) on i32 overflow per the documented contract"
    );
}

/// Integer serial == parallel **bit-identity** (integer accumulation is exact, so
/// any thread count must produce identical i32 output). Unlike the float
/// `parallel_equals_serial_*` tests, this is *order-independent* (wrapping i32 add is
/// associative), so it stays a hard guarantee even if blocking ever becomes
/// parallelism-dependent; it does not carry their thread-independent-blocking caveat
#[cfg(feature = "int8")]
#[test]
fn parallel_equals_serial_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [(200, 130, 175), (256, 64, 200), (384, 96, 320)] {
        let a = rand_i8(m * k, 0x300 + m as u64);
        let b = rand_i8(k * n, 0x400 + n as u64);
        let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 5) - 2).collect();
        for &(alpha, beta) in &[(1i32, 0i32), (2, 3)] {
            let mut c_ser = c0.clone();
            gemmkit::gemm_i8(
                alpha,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                beta,
                MatMut::from_col_major(&mut c_ser, m, n),
                Parallelism::Serial,
            );
            for t in [2usize, 4, 8, 16] {
                let mut c_par = c0.clone();
                gemmkit::gemm_i8(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    MatRef::from_col_major(&b, k, n),
                    beta,
                    MatMut::from_col_major(&mut c_par, m, n),
                    Parallelism::Rayon(t),
                );
                assert_eq!(c_ser, c_par, "i8 serial != parallel({t}) for {m}x{k}x{n}");
            }
        }
    }
}

/// Negative strides for the integer path via [`gemmkit::gemm_i8_unchecked`] (the
/// heterogeneous escape hatch: the homogeneous `gemm_unchecked` can't serve
/// `i8 -> i32`). Reversed-row A, compared to the row-reversed exact reference
#[cfg(feature = "int8")]
#[test]
fn i8_negative_strides_unchecked() {
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_i8(m * k, 5); // row-major m x k
    let b = rand_i8(k * n, 6); // row-major k x n
    let c0 = vec![0i32; m * n];
    let cref = ref_i8(&a, &b, &c0, m, k, n, 1, 0);

    let mut c = vec![0i32; m * n];
    unsafe {
        let a_last = a.as_ptr().add((m - 1) * k); // base = last row
        gemmkit::gemm_i8_unchecked(
            m,
            k,
            n,
            1,
            a_last,
            -(k as isize), // reversed rows of A
            1,
            b.as_ptr(),
            n as isize, // row-major B
            1,
            0,
            c.as_mut_ptr(),
            n as isize, // row-major C
            1,
            Parallelism::Serial,
        );
    }
    // Computed C[i,j] = sum_k A[m-1-i,k]*B[k,j]; compare to the reversed reference
    for i in 0..m {
        for j in 0..n {
            assert_eq!(
                c[i * n + j],
                cref[(m - 1 - i) * n + j],
                "i8 neg stride ({i},{j})"
            );
        }
    }
}
