//! `GEMMKIT_REQUIRE_ISA=avx512vnni` prepacked-i8 bit-parity pin: forces the `vpdpbusd` dot kernel
//! (k-quad-interleaved RHS prepack, `DEPTH_MULTIPLE = 4`, `+128` LHS bias). Forcing it also means
//! the small-parallel widen fallback is disabled, so this pins the VNNI prepack-and-consume path
//! exactly. Its own single-test binary so the one `set_var` runs before any i8 dispatch. Skips
//! gracefully when the host lacks `avx512f+bw+vnni`
#![cfg(all(
    feature = "std",
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared prepacked-i8 bit-parity check driven by the ISA pin binaries
mod i8_packed_common;

#[test]
fn avx512vnni_pin_i8_packed_matches_plain() {
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
    i8_packed_common::check();
}
