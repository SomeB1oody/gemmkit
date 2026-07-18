//! `GEMMKIT_REQUIRE_ISA=avx512` prepacked-i8 bit-parity pin: forces the **widen** AVX-512 integer
//! kernel (plain-panel RHS prepack, `DEPTH_MULTIPLE = 1`), the path an auto VNNI box never takes.
//! Its own single-test binary so the one `set_var` runs before any i8 dispatch. Skips gracefully
//! when the host lacks `avx512f` (the pin would otherwise assert in `select_i8`)
#![cfg(all(
    feature = "std",
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]

// Shared prepacked-i8 bit-parity check driven by the ISA pin binaries
mod i8_packed_common;

#[test]
fn avx512_pin_i8_packed_matches_plain() {
    // Pin once, before any i8 dispatch. SAFETY: the only test in this binary, so nothing reads the
    // environment concurrently with this write
    unsafe {
        std::env::set_var("GEMMKIT_REQUIRE_ISA", "avx512");
    }
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: host does not report avx512f");
        return;
    }
    i8_packed_common::check();
}
