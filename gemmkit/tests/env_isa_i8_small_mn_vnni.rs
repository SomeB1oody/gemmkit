//! `GEMMKIT_REQUIRE_ISA=avx512vnni` small-`m,n` horizontal-route pin: a forced ISA still allows the
//! special paths (the `small_mn` gate lives in `run_typed_int`, orthogonal to `select_i8`), so a
//! tiny-`m,n` / long-`k` i8 shape widens through the horizontal dot instead of the `vpdpbusd`
//! driver. This binary pins VNNI in its own process (so the memoized `select_i8` is the dot kernel)
//! and checks the horizontal route (eligible layout) stays bit-exact vs both the driver (ineligible
//! layout) and the exact `i32` reference. Its own single-test binary so the one `set_var` runs
//! before any i8 dispatch. Skips gracefully when the host lacks `avx512f+bw+vnni`
#![cfg(all(
    feature = "std",
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

use gemmkit::{MatMut, MatRef, Parallelism};

/// Exact wrapping-`i32` GEMM reference (row-major A/B), matching the documented i8 contract
#[allow(clippy::too_many_arguments)]
fn ref_i8(
    a: &[i8],
    b: &[i8],
    c0: &[i32],
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    beta: i32,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32).wrapping_mul(b[p * n + j] as i32));
            }
            out[i * n + j] = beta
                .wrapping_mul(c0[i * n + j])
                .wrapping_add(alpha.wrapping_mul(acc));
        }
    }
    out
}

#[test]
fn avx512vnni_pin_i8_small_mn_matches_driver() {
    // Pin once, before any i8 dispatch. SAFETY: the only test in this binary, so nothing reads the
    // environment concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "avx512vnni");
    }
    if !(is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512f"))
    {
        eprintln!("skipping: host does not report avx512f+bw+vnni");
        return;
    }
    let kt = gemmkit::tuning::small_k_threshold();
    // Small `m,n` (both <= small_mn_dim), long `k`, incl. single-row / single-col and a tail tile
    for &(m, n) in &[
        (4usize, 4usize),
        (8, 8),
        (13, 7),
        (16, 16),
        (16, 1),
        (1, 16),
    ] {
        for &k in &[kt + 1, 4096usize] {
            let a: Vec<i8> = (0..m * k)
                .map(|i| ((i as i32 * 7 + 3) % 201 - 100) as i8)
                .collect();
            let b: Vec<i8> = (0..k * n)
                .map(|i| ((i as i32 * 5 + 9) % 201 - 100) as i8)
                .collect();
            // Column-major B (rsb=1): the eligible-layout twin of the row-major B
            let bcol: Vec<i8> = {
                let mut v = vec![0i8; k * n];
                for p in 0..k {
                    for j in 0..n {
                        v[p + j * k] = b[p * n + j];
                    }
                }
                v
            };
            for &(alpha, beta) in &[(1i32, 0i32), (2, 3), (3, -1), (0, 1)] {
                let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();
                let cref = ref_i8(&a, &b, &c0, m, k, n, alpha, beta);
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    // Eligible: row-major A + col-major B => the horizontal `small_mn` route even
                    // under the VNNI pin (the gate is orthogonal to the forced ISA)
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
                        "vnni-pin i8 small_mn {m}x{k}x{n} alpha={alpha} beta={beta} {par:?}"
                    );
                    // Ineligible: row-major B => the `vpdpbusd` driver on the same math
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
                        "vnni-pin i8 driver {m}x{k}x{n} alpha={alpha} beta={beta} {par:?}"
                    );
                }
            }
        }
    }
}
