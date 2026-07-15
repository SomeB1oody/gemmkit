//! `GEMMKIT_REQUIRE_ISA` pin parsing plus the memoized ISA-selection plumbing
//! (`x86_isa_detected!` / `memoized_select!`) shared by every family's `select_*` ladder

/// x86 ISA probe for the `select_*` ladders: the runtime `is_x86_feature_detected!`
/// with `std`, else a compile-time `cfg!(target_feature = ...)`: off `std` there is no
/// runtime CPU detection (`raw-cpuid` is `std`-gated), so a no_std build runs whatever
/// its compile-time target-features guarantee
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! x86_isa_detected {
    ($feat:tt) => {
        is_x86_feature_detected!($feat)
    };
}
#[cfg(all(not(feature = "std"), any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! x86_isa_detected {
    ($feat:tt) => {
        cfg!(target_feature = $feat)
    };
}

/// An explicitly requested kernel, parsed from `GEMMKIT_REQUIRE_ISA`
///
/// The non-`Auto` variants are constructed only by the `std` `forced_isa` (env-var
/// parsing); the no-`std` `forced_isa` always yields `Auto`. The `select_*` ladders
/// still match on every variant, so they must remain in the type, hence the
/// `dead_code` allowance for the no-`std` build rather than `#[cfg]`-ing them out
#[cfg_attr(not(feature = "std"), allow(dead_code))]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum ForcedIsa {
    /// No override: auto-select the best available ISA (the default)
    Auto,
    /// Scalar: the fallback path when no other ISA is available
    Scalar,
    /// FMA: the `fma`-based (AVX2) widen kernel
    Fma,
    /// AVX-512 foundation (`avx512f`): the widen kernel
    Avx512F,
    /// AVX-512 VNNI: the `i8` `vpdpbusd` dot kernel
    Avx512Vnni,
    /// AVX-512 BF16: the `bf16` `vdpbf16ps` dot kernel
    Avx512Bf16,
    /// NEON: the AArch64 kernel
    Neon,
    /// WebAssembly `simd128`. Baseline-by-cfg like `Neon`, but `simd128` is an easily-forgotten
    /// compile-time `-C target-feature=+simd128`; pinning it makes a build **assert** the SIMD
    /// path is live (panics if absent) instead of silently falling back to scalar
    Simd128,
}

/// Parse the `GEMMKIT_REQUIRE_ISA` pin. Unset/empty -> [`ForcedIsa::Auto`]; an
/// unrecognized value is a hard error (catches typos in CI config). Read once,
/// since the selection is memoized in the per-type `OnceLock`
#[cfg(feature = "std")]
pub(super) fn forced_isa() -> ForcedIsa {
    match std::env::var("GEMMKIT_REQUIRE_ISA") {
        Err(_) => ForcedIsa::Auto,
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("auto") {
                ForcedIsa::Auto
            } else if t.eq_ignore_ascii_case("scalar") {
                ForcedIsa::Scalar
            } else if t.eq_ignore_ascii_case("fma") || t.eq_ignore_ascii_case("avx2") {
                ForcedIsa::Fma
            } else if t.eq_ignore_ascii_case("avx512") || t.eq_ignore_ascii_case("avx512f") {
                ForcedIsa::Avx512F
            } else if t.eq_ignore_ascii_case("avx512vnni") || t.eq_ignore_ascii_case("vnni") {
                ForcedIsa::Avx512Vnni
            } else if t.eq_ignore_ascii_case("avx512bf16") || t.eq_ignore_ascii_case("bf16") {
                ForcedIsa::Avx512Bf16
            } else if t.eq_ignore_ascii_case("neon") {
                ForcedIsa::Neon
            } else if t.eq_ignore_ascii_case("simd128") || t.eq_ignore_ascii_case("wasm") {
                ForcedIsa::Simd128
            } else {
                panic!(
                    "GEMMKIT_REQUIRE_ISA: unknown value `{t}` (expected scalar|fma|avx512|avx512vnni|avx512bf16|neon|simd128|auto)"
                )
            }
        }
    }
}
#[cfg(not(feature = "std"))]
pub(super) fn forced_isa() -> ForcedIsa {
    ForcedIsa::Auto
}

/// Emit the memoized dispatch accessor for one element type: a `#[cfg(std)]`
/// `OnceLock<$ty>` plus a `fn $accessor() -> $ty` that runs `$select` once under `std`
/// (feature detection is memoized) and directly on each call without `std`. The optional
/// trailing `$feat` additionally gates the accessor and the `OnceLock` on that feature (the
/// static is always further gated by `std`). Every `dispatched_*` slot shares this shape
macro_rules! memoized_select {
    ($static:ident, $accessor:ident, $ty:ty, $select:ident, $doc:literal) => {
        #[cfg(feature = "std")]
        static $static: OnceLock<$ty> = OnceLock::new();
        #[doc = $doc]
        #[inline]
        fn $accessor() -> $ty {
            #[cfg(feature = "std")]
            {
                *$static.get_or_init($select)
            }
            #[cfg(not(feature = "std"))]
            {
                $select()
            }
        }
    };
    ($static:ident, $accessor:ident, $ty:ty, $select:ident, $doc:literal, $feat:literal) => {
        #[cfg(all(feature = "std", feature = $feat))]
        static $static: OnceLock<$ty> = OnceLock::new();
        #[doc = $doc]
        #[cfg(feature = $feat)]
        #[inline]
        fn $accessor() -> $ty {
            #[cfg(feature = "std")]
            {
                *$static.get_or_init($select)
            }
            #[cfg(not(feature = "std"))]
            {
                $select()
            }
        }
    };
}
