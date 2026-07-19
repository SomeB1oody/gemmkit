//! `GEMMKIT_REQUIRE_ISA=avx512bf16` pin: forces the `vdpbf16ps` dot kernel `Bf16DotGemm`, so its
//! multi-slice deep-k twin `Bf16DotGemmF32` gets exercised (`kc` rounded up to `DEPTH_MULTIPLE =
//! 2` there, so a k-pair never straddles a slice boundary). Pins through [`env_isa_common::pin`]
//! (a single `set_var` under a `Once`, before any dispatch; the shared write also overrides an
//! inherited pin). The multi-slice route must be byte-for-byte the single depth panel for
//! `beta in {0, 1}`. Skips when the host lacks `avx512bf16`
#![cfg(all(
    feature = "half",
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared GEMMKIT_REQUIRE_ISA pin helper; this binary's only test pins `avx512bf16` with it
mod env_isa_common;

use gemmkit::{MatMut, MatRef, Parallelism, bf16, gemm, tuning};

fn fill(n: usize, seed: u64) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        })
        .collect()
}

#[test]
fn deep_k_bf16_dot_under_avx512bf16_pin() {
    env_isa_common::pin("avx512bf16");
    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16")) {
        eprintln!("skipping: host does not report avx512f+avx512bf16");
        return;
    }
    let restore = tuning::deep_kc_bytes();

    // 4097 is odd, so both a slice boundary and the final tail land on odd depth: the
    // DEPTH_MULTIPLE rounding must keep every interior slice even and pad only the tail
    for &k in &[4096usize, 4097] {
        let (m, n) = (40usize, 50);
        let a = fill(m * k, 0x1);
        let b = fill(k * n, 0x2);
        let c0 = fill(m * n, 0x3);
        for &beta in &[0.0f32, 1.0] {
            let run = |engage: bool, par| -> Vec<u16> {
                tuning::set_deep_kc_bytes(if engage { 1 } else { usize::MAX });
                let mut c = c0.clone();
                gemm(
                    bf16::from_f32(1.25),
                    MatRef::from_col_major(&a, m, k),
                    MatRef::from_col_major(&b, k, n),
                    bf16::from_f32(beta),
                    MatMut::from_col_major(&mut c, m, n),
                    par,
                );
                c.iter().map(|v| v.to_bits()).collect()
            };
            let single = run(false, Parallelism::Serial);
            let deep_ser = run(true, Parallelism::Serial);
            let deep_par = run(true, Parallelism::Rayon(0));
            assert_eq!(
                deep_ser, single,
                "bf16 dot: deep-k != single panel (k={k} beta={beta})"
            );
            assert_eq!(
                deep_par, deep_ser,
                "bf16 dot: deep-k serial != parallel (k={k} beta={beta})"
            );
        }
    }

    tuning::set_deep_kc_bytes(restore);
}
