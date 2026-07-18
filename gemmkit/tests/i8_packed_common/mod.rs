//! Shared prepacked-i8 bit-parity check, driven by the per-ISA pin binaries
//! (`env_isa_i8_packed_*`). Each binary sets `GEMMKIT_REQUIRE_ISA` once (its own process, so the
//! `set_var` runs before any dispatch) then calls [`check`], which asserts `gemm_i8_packed_b` is
//! **bit-identical** to a plain `gemm_i8` across a shape / layout / alpha-beta / thread-count
//! sweep. `k` not a multiple of 4 (the VNNI depth pad) and `n` not a multiple of the panel width
//! are both included, so the pinned widen kernel and the pinned VNNI dot kernel each exercise
//! their own prepacked micropanel layout
#![allow(dead_code)]

use gemmkit::{MatMut, MatRef, Parallelism};

fn rand_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 24) as i64 % 201 - 100) as i8
        })
        .collect()
}

/// Assert `gemm_i8_packed_b` reproduces a plain `gemm_i8` bit-for-bit under the currently pinned
/// ISA. Integer accumulation is exact, so the equality is hard for every shape/layout/thread count
pub fn check() {
    for (m, k, n) in [
        (200usize, 130, 175),
        (65, 64, 64),
        (64, 65, 100), // k not a multiple of 4, n not a multiple of nr
        (33, 17, 19),
        (256, 257, 129),
        (8, 1024, 96), // small m, long k: the fixed-weight inference shape
        (2, 1023, 12),
    ] {
        let a = rand_i8(m * k, 0x51 + (m * 7 + n) as u64);
        let b_rm = rand_i8(k * n, 0x62 + (n * 3 + k) as u64); // logical k x n stored row-major
        let mut b_cm = vec![0i8; k * n];
        for i in 0..k {
            for j in 0..n {
                b_cm[j * k + i] = b_rm[i * n + j];
            }
        }
        let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();

        for lb in 0..2 {
            let (bbuf, rsb, csb): (&[i8], isize, isize) = if lb == 0 {
                (&b_rm, n as isize, 1)
            } else {
                (&b_cm, 1, k as isize)
            };
            let bview = MatRef::new(bbuf, k, n, rsb, csb);
            let packed = gemmkit::prepack_rhs_i8(bview);

            for &(alpha, beta) in &[(1i32, 0i32), (2, 3), (0, 5)] {
                let mut c_ref = c0.clone();
                gemmkit::gemm_i8(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    bview,
                    beta,
                    MatMut::from_col_major(&mut c_ref, m, n),
                    Parallelism::Serial,
                );
                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = c0.clone();
                    gemmkit::gemm_i8_packed_b(
                        alpha,
                        MatRef::from_col_major(&a, m, k),
                        &packed,
                        beta,
                        MatMut::from_col_major(&mut c_pk, m, n),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack_i8 != gemm_i8 for {m}x{k}x{n} lb={lb} a={alpha} b={beta} par={par:?}"
                    );
                }
            }
        }
    }
}
