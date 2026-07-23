//! Parses the `GEMMKIT_REQUIRE_ISA` pin and provides the memoized ISA-selection
//! plumbing (`x86_isa_detected!`, `memoized_select!`) every dispatch family's
//! `select_*` ladder is built from

/// x86 feature probe shared by every `select_*` ladder: a real runtime CPUID query via
/// `is_x86_feature_detected!` under `std`, or a compile-time `cfg!(target_feature = ...)`
/// without it, since `is_x86_feature_detected!` itself needs `std`. A `no_std` build
/// therefore only ever sees whatever target-features it was compiled with
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
/// Only the `std` build of [`forced_isa`] ever constructs a non-`Auto` variant (it parses
/// the env var); the `no_std` build always returns `Auto`. Every `select_*` ladder still
/// matches on the full enum regardless of `std`, so every variant has to stay in the type;
/// the `dead_code` allowance below silences the resulting unused-variant warning on a
/// `no_std` build rather than `#[cfg]`-ing the variants out
#[cfg_attr(not(feature = "std"), allow(dead_code))]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum ForcedIsa {
    /// No pin: auto-select the best available ISA (the default)
    Auto,
    /// Scalar fallback, used when no other ISA applies
    Scalar,
    /// AVX2 + FMA widen kernel
    Fma,
    /// AVX-512 Foundation (`avx512f`) widen kernel
    Avx512F,
    /// AVX-512 VNNI: the `i8` `vpdpbusd` dot kernel
    Avx512Vnni,
    /// AVX-512 BF16: the `bf16` `vdpbf16ps` dot kernel
    Avx512Bf16,
    /// NEON, the baseline (and only) SIMD ISA on aarch64
    Neon,
    /// WebAssembly `simd128`. Unlike `Neon` this is not automatically part of a wasm32
    /// build: it needs the compile-time `-C target-feature=+simd128` flag, easy to forget,
    /// so pinning it makes the build **assert** the SIMD path is live (panic if the flag
    /// is absent) instead of silently falling back to scalar
    Simd128,
}

/// Parse `GEMMKIT_REQUIRE_ISA` into a [`ForcedIsa`]. Unset or empty maps to
/// [`ForcedIsa::Auto`]; an unrecognized value panics rather than silently falling back
/// to auto-selection (so a typo in CI config fails loudly). Each dispatch type calls
/// this at most once, since the result it feeds into `select_*` is memoized in that
/// type's `OnceLock`
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
            } else if t.eq_ignore_ascii_case("avx512f") {
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
                    "GEMMKIT_REQUIRE_ISA: unknown value `{t}` (expected scalar|fma|avx512f|avx512vnni|avx512bf16|neon|simd128|auto)"
                )
            }
        }
    }
}
// No env var access without std: every select_* ladder always runs its auto-detect path
#[cfg(not(feature = "std"))]
pub(super) fn forced_isa() -> ForcedIsa {
    ForcedIsa::Auto
}

/// Emits the memoized dispatch accessor for one element type: a `#[cfg(std)]`
/// `OnceLock<$ty>` plus a `fn $accessor() -> $ty` that runs `$select` once via
/// `get_or_init` under `std`, or on every call without `std` (there is no `OnceLock` to
/// cache into). The 6-arg form adds a trailing `$feat` literal that additionally gates the
/// accessor and its static on that crate feature (the static always still needs `std` too).
/// Every dispatch family's `dispatched_*` slot is built through one of these 2 arms
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
