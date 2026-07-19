//! Integer GEMM (i8 -> i32): exact-reference correctness, the `small_mn` route,
//! overflow wrapping, workspace reuse, parallel bit-identity, and negative strides

use crate::common::*;
use gemmkit::Parallelism;

/// A shape and alpha/beta spread on row-major A, col-major B, row-major C, checked
/// against the exact `ref_i8` reference
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

            // Transpose B into column-major, pairing row-major A with col-major B
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

/// The `small_mn` horizontal i8 route (small `m`/`n`, unit-stride operands along `k`) must
/// land on the same wrapping `i32` bits as the general register-tiling driver: wrapping i32
/// addition is associative, so a single fixed-order widen dot and the driver's panel-split
/// accumulation are guaranteed to agree regardless of which sums which first. `m`/`n` straddle
/// the default `small_mn_dim` gate (16, so 17 on either axis spills to the driver), `k` is tail
/// sized (`small_k_threshold + 1`, not a multiple of the SIMD lane width), and every
/// alpha/beta sign and both `Parallelism` variants are covered. Each shape runs both the
/// eligible layout (row-major A, col-major B: the orientation swap lands both operands
/// unit-stride along `k`, the small_mn gate's `csa == 1 && rsb == 1` condition) and an
/// ineligible one (row-major B, which the swap leaves strided along `k`, so the gate rejects
/// it and the driver runs instead), both checked against the exact `i64` reference. No
/// tuning-knob override: the route comes from the real dispatch, so this is safe to run
/// alongside other tests
#[cfg(feature = "int8")]
#[test]
fn i8_small_mn_matches_reference() {
    use gemmkit::{MatMut, MatRef};
    let kt = gemmkit::tuning::small_k_threshold();
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
                // Column-major B (rsb=1): the small_mn-eligible twin of the row-major b below
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
                        // Eligible layout: takes the small_mn route (when m,n <= 16)
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
                        // Ineligible layout: falls through to the general driver instead
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

/// `gemm_i8_unchecked_with` (raw pointers, no bounds/alias checks, reusing a caller-owned
/// `Workspace`) must produce the same output as the checked `gemm_i8`
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

            // Transpose B into column-major, pairing row-major A with col-major B
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
            // SAFETY: a/bcol/c are 3 distinct, correctly sized buffers, so every stride/extent
            // implied by (rsa,csa)/(rsb,csb)/(rsc,csc) stays in bounds and C aliases neither operand
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

/// `gemm_i8` is documented to wrap on i32 overflow, but every other i8 test keeps values
/// small enough that the wrap never fires; force it here with a large `alpha` and check
/// against a `wrapping_mul`/`wrapping_add` reference, not the range-checked `ref_i8`
/// (which panics on overflow)
#[cfg(feature = "int8")]
#[test]
fn i8_wraps_on_overflow() {
    use gemmkit::{MatMut, MatRef};
    // 2x2, k=2, all 127s: each dot is 127*127 + 127*127 = 32258, itself in range; alpha
    // then overflows i32 when scaling it. k is below small_k_threshold, so this runs
    // through special::small_k rather than the packing driver, exercising its wrapping
    // epilogue instead
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

/// Serial and parallel i8 runs land on identical i32 bits for every thread count.
/// Unlike the float `parallel_equals_serial_*` tests, this is a hard guarantee, not one
/// that happens to hold under the current blocking: wrapping i32 addition is associative,
/// so any reduction order lands on the same result
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

/// `gemm_i8_unchecked` accepts a negative stride, unlike the homogeneous
/// `gemm_unchecked`, which cannot serve `i8 -> i32` at all. Point A's base at its last
/// physical row and walk backwards; output row `i` then holds the product for A's
/// physical row `m-1-i`, checked against the exact reference indexed the same way
#[cfg(feature = "int8")]
#[test]
fn i8_negative_strides_unchecked() {
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_i8(m * k, 5);
    let b = rand_i8(k * n, 6);
    let c0 = vec![0i32; m * n];
    let cref = ref_i8(&a, &b, &c0, m, k, n, 1, 0);

    let mut c = vec![0i32; m * n];
    unsafe {
        let a_last = a.as_ptr().add((m - 1) * k);
        gemmkit::gemm_i8_unchecked(
            m,
            k,
            n,
            1,
            a_last,
            -(k as isize),
            1,
            b.as_ptr(),
            n as isize,
            1,
            0,
            c.as_mut_ptr(),
            n as isize,
            1,
            Parallelism::Serial,
        );
    }
    // C[i,j] holds sum_k A[m-1-i,k]*B[k,j]; index the reference the same way
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
