//! `GEMMKIT_REQUIRE_ISA=avx512vnni` pin: forces the `vpdpbusd` i8 dot kernel (`IntGemmVnni`,
//! `DEPTH_MULTIPLE = 4`, a `+128` LHS bias) instead of whatever `select_i8` would otherwise pick
//! on this host. A forced pin also disables the small-parallel widen fallback that auto-selected
//! VNNI can take (its mandatory RHS-pack barrier is skipped only when the choice is dynamic), so
//! this exercises the VNNI kernel exactly, for every shape that reaches `select_i8` (the small_mn
//! gate below is a separate, earlier reroute)
//!
//! Both tests want the same `avx512vnni` pin, so both go through [`env_isa_common::pin`] (a
//! single `set_var` under a `Once`, before any dispatch; the shared write also overrides an
//! inherited pin, so a CI job that already exports `GEMMKIT_REQUIRE_ISA` still exercises this
//! route). Both skip when the host lacks `avx512f+bw+vnni`, since forcing the pin on such a host
//! would abort in `select_i8` instead of testing anything
#![cfg(all(
    feature = "std",
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared GEMMKIT_REQUIRE_ISA pin helper; both tests here pin `avx512vnni` with it
mod env_isa_common;
// Shared prepacked-vs-plain i8 bit-parity check, run under whichever ISA is pinned
mod i8_packed_common;

use gemmkit::{MatMut, MatRef, Parallelism};

/// Whether the host reports the full `avx512f+bw+vnni` feature set the pin requires
fn host_has_vnni() -> bool {
    is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512f")
}

// The VNNI kernel's prepacked micropanel path must match plain gemm_i8 bit-for-bit

#[test]
fn avx512vnni_pin_i8_packed_matches_plain() {
    env_isa_common::pin("avx512vnni");
    if !host_has_vnni() {
        eprintln!("skipping: host does not report avx512f+bw+vnni");
        return;
    }
    i8_packed_common::check();
}

/// Exact wrapping-i32 GEMM reference (row-major A/B), matching the documented i8 contract
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

// The small_mn eligibility gate lives in run_typed_int, ahead of and independent of select_i8,
// so a forced ISA still lets a tiny-m,n / long-k shape reroute to the horizontal small_mn kernel
// instead of the pinned VNNI driver. Checks that route (eligible A/B layout) against both the
// pinned driver (ineligible layout, same math) and the exact i32 reference

#[test]
fn avx512vnni_pin_i8_small_mn_matches_driver() {
    env_isa_common::pin("avx512vnni");
    if !host_has_vnni() {
        eprintln!("skipping: host does not report avx512f+bw+vnni");
        return;
    }
    let kt = gemmkit::tuning::small_k_threshold();
    // Small m,n (each <= small_mn_dim), long k; includes single-row/single-column shapes
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
            // Column-major transpose of b (rsb=1), the small_mn-eligible layout
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
                    // Eligible layout: takes the small_mn route despite the VNNI pin
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
                    // Ineligible layout: falls through to the pinned VNNI driver
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
