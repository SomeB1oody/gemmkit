//! `IntGemm::pack_rhs`: the widen (non-VNNI) i8 kernel's RHS packer. On this AVX-512-VNNI box the
//! auto i8 path always picks the `vpdpbusd` dot kernel (which packs via `IntGemmVnni::pack_rhs`), so
//! the widen packer is only reachable when the plain AVX-512 kernel is forced. This binary pins
//! `GEMMKIT_REQUIRE_ISA=avx512` (its own process, so the memoized `select_i8` is the widen kernel)
//! and lowers the RHS-pack threshold to `1` so a small `m` still triggers packing, the more robust
//! route than the `m > 2048` default, since it forces the widen kernel directly rather than relying
//! on the coverage orchestrator's avx512 pin pass, and keeps the matrix tiny/fast. Skips gracefully
//! when the host lacks `avx512f` (the pin would otherwise assert)
#![cfg(all(
    feature = "std",
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

use gemmkit::{MatMut, MatRef, Parallelism, tuning};

#[test]
fn widen_i8_pack_rhs_under_avx512_pin() {
    // Pin once, before any i8 dispatch. SAFETY: the only test in this binary, so nothing reads the
    // environment concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "avx512");
    }
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    // Force RHS packing for the widen kernel even at small `m` (`pack_b = m > threshold`)
    tuning::set_rhs_pack_threshold(1);

    // Driver path (k > small_k gate, m,n > small_mn gate), so RHS packing runs through
    // `IntGemm::pack_rhs`. Values stay in range so the wrapping semantics never fire
    let (m, k, n) = (96usize, 64usize, 64usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i % 13) as i32 - 6) as i8).collect();
    let c0: Vec<i32> = (0..m * n).map(|i| (i % 7) as i32 - 3).collect();
    let (alpha, beta) = (2i32, -1i32);

    let mut c = c0.clone();
    gemmkit::gemm_i8(
        alpha,
        MatRef::from_row_major(&a, m, k),
        MatRef::from_row_major(&b, k, n),
        beta,
        MatMut::from_row_major(&mut c, m, n),
        Parallelism::Serial,
    );

    // Naive wrapping-i32 reference
    let mut expect = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32).wrapping_mul(b[p * n + j] as i32));
            }
            expect[i * n + j] = alpha
                .wrapping_mul(acc)
                .wrapping_add(beta.wrapping_mul(c0[i * n + j]));
        }
    }
    assert_eq!(
        c, expect,
        "widen i8 kernel with RHS packing must match the reference"
    );
}
