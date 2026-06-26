//! Cache topology and analytical blocking (layer L3).
//!
//! Two facts drive the design:
//!
//! 1. **`#[cfg]` only picks the *sniffing method*, never the *values*.** A VM or
//!    container can mask CPUID or hide `/sys`, and `#[cfg(target_arch)]` cannot
//!    tell an Intel apart from an AMD. So every backend is best-effort and there
//!    is always a runtime fallback chain that cannot fail.
//! 2. The blocking sizes `(MC, KC, NC)` are computed analytically from the cache
//!    geometry using the BLIS model, so they adapt to the machine instead of
//!    being hard-coded per micro-arch.
//!
//! The fallback chain is: platform backend (CPUID on x86) → micro-arch hint →
//! a static default calibrated on the Ryzen 9950X (Zen5). Detection runs once
//! and is memoized.

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
mod cpuid;
#[cfg(target_os = "macos")]
mod sysctl;
#[cfg(target_os = "linux")]
mod sysfs;

#[cfg(feature = "std")]
use std::sync::OnceLock;

/// One cache level.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// Total size in bytes (as reported, before dividing by `shared_by`).
    pub bytes: usize,
    /// Associativity (ways).
    pub assoc: usize,
    /// Cache line size in bytes.
    pub line: usize,
    /// Number of logical cores sharing this level (1 = private).
    pub shared_by: usize,
}

impl Level {
    /// Effective per-core capacity = `bytes / shared_by`.
    #[inline]
    pub fn effective_bytes(&self) -> usize {
        self.bytes / self.shared_by.max(1)
    }
}

/// The data-cache hierarchy used for blocking.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CacheTopology {
    /// L1 data cache.
    pub l1d: Level,
    /// L2 cache.
    pub l2: Level,
    /// L3 cache, if any (some Apple / embedded parts report none).
    pub l3: Option<Level>,
}

/// Blocking parameters for the five-loop driver.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Blocking {
    /// Rows of A / C per L3-resident macro-panel iteration (loop 3).
    pub mc: usize,
    /// Depth per L2/L1-resident panel iteration (loop 4).
    pub kc: usize,
    /// Columns of B / C per L3-resident macro-panel iteration (loop 5).
    pub nc: usize,
}

/// The Zen5 (Ryzen 9950X) calibrated default — the bottom of the fallback chain.
/// L1d 48 KiB / 12-way, L2 1 MiB / 16-way (private), L3 32 MiB / 16-way (per
/// CCD; treated as fully available for the B macro-panel, i.e. `shared_by = 1`).
pub const ZEN5_FALLBACK: CacheTopology = CacheTopology {
    l1d: Level {
        bytes: 48 * 1024,
        assoc: 12,
        line: 64,
        shared_by: 1,
    },
    l2: Level {
        bytes: 1024 * 1024,
        assoc: 16,
        line: 64,
        shared_by: 1,
    },
    l3: Some(Level {
        bytes: 32 * 1024 * 1024,
        assoc: 16,
        line: 64,
        shared_by: 1,
    }),
};

#[cfg(feature = "std")]
static TOPOLOGY: OnceLock<CacheTopology> = OnceLock::new();

/// The detected cache topology (memoized; detection runs at most once).
#[cfg(feature = "std")]
pub fn topology() -> &'static CacheTopology {
    TOPOLOGY.get_or_init(detect)
}

/// The detected cache topology. Without `std` there is no memoization or
/// detection, so the calibrated fallback is returned.
#[cfg(not(feature = "std"))]
pub fn topology() -> &'static CacheTopology {
    &ZEN5_FALLBACK
}

/// The OS memory page size in bytes (memoized). Drives the LHS-packing stride
/// gate.
///
/// `getpagesize` is POSIX/BSD and present on both Linux and macOS
#[cfg(all(unix, feature = "std"))]
pub(crate) fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        unsafe extern "C" {
            fn getpagesize() -> core::ffi::c_int;
        }
        let p = unsafe { getpagesize() } as usize;
        // A real base page is a power of two in a sane range; else fall back.
        if p.is_power_of_two() && (4096..=2 * 1024 * 1024).contains(&p) {
            p
        } else {
            4096
        }
    })
}

/// Page-size fallback for non-unix / no-std builds (assume the common 4 KiB).
#[cfg(not(all(unix, feature = "std")))]
pub(crate) fn page_size() -> usize {
    4096
}

/// Run the fallback chain once. Never panics: any backend that fails or returns
/// implausible values is skipped.
#[cfg(feature = "std")]
fn detect() -> CacheTopology {
    // Miri cannot execute the CPUID instruction; use the calibrated fallback.
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
    if let Some(t) = cpuid::detect().filter(plausible) {
        return t;
    }
    #[cfg(target_os = "linux")]
    if let Some(t) = sysfs::detect().filter(plausible) {
        return t;
    }
    #[cfg(target_os = "macos")]
    if let Some(t) = sysctl::detect().filter(plausible) {
        return t;
    }
    ZEN5_FALLBACK
}

/// Sanity gate so a half-populated CPUID read can't poison blocking.
#[cfg(feature = "std")]
fn plausible(t: &CacheTopology) -> bool {
    let ok = |l: &Level| l.bytes >= 4 * 1024 && l.line >= 16 && l.assoc >= 1;
    ok(&t.l1d) && ok(&t.l2) && t.l3.as_ref().map(ok).unwrap_or(true)
}

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

#[inline]
fn round_down(a: usize, b: usize) -> usize {
    (a / b) * b
}

impl CacheTopology {
    /// Compute `(MC, KC, NC)` analytically (BLIS model) for the given microtile
    /// geometry and problem size.
    ///
    /// * `mr`/`nr`: microkernel tile (in elements).
    /// * `sizeof`: size of one accumulator element in bytes.
    ///
    /// The result is independent of the thread count, so serial and parallel
    /// runs use identical blocking (a prerequisite for bit-identical output).
    pub fn blocking(
        &self,
        mr: usize,
        nr: usize,
        sizeof: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Blocking {
        if m == 0 || n == 0 || k == 0 {
            return Blocking {
                mc: m.max(mr),
                kc: k.max(1),
                nc: n.max(nr),
            };
        }

        let l1 = self.l1d.effective_bytes().max(32 * 1024);
        let l2 = self.l2.effective_bytes();
        let l3 = self.l3.map(|l| l.effective_bytes()).unwrap_or(0);
        let line = self.l1d.line.max(64);
        let l1_assoc = self.l1d.assoc.max(2);
        let l2_assoc = self.l2.assoc.max(2);
        let l3_assoc = self.l3.map(|l| l.assoc).unwrap_or(2).max(2);
        let l1_n_sets = (l1 / (line * l1_assoc)).max(1);

        // Small-matrix shortcut: skip the full model, just keep panels in L2.
        if m <= 64 && n <= 64 {
            let kc = k.clamp(1, 512);
            let mc = ((l2 / sizeof / kc) / mr * mr).max(mr);
            let nc = n.next_multiple_of(nr).max(nr);
            return Blocking { mc, kc, nc };
        }

        // --- KC: A & B micropanels coexist in L1 without self-eviction ---
        let g = gcd(mr * sizeof, line * l1_n_sets);
        let kc_0 = (line * l1_n_sets) / g;
        let c_lhs = (mr * sizeof) / g;
        let c_rhs = (nr * kc_0 * sizeof) / (line * l1_n_sets);
        let kc_mult = (l1_assoc / (c_lhs + c_rhs).max(1)).max(1);
        let mut kc = (kc_0 * kc_mult.next_power_of_two()).max(512).min(k);
        let k_iter = k.div_ceil(kc).max(1);
        kc = k.div_ceil(k_iter).max(1); // rebalance so the last panel isn't tiny

        // --- MC: A macro-panel resides in L2 (reserve one way for B) ---
        let rhs_micropanel = nr * kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel.div_ceil((l2 / l2_assoc).max(1));
        let lhs_l2_assoc = l2_assoc.saturating_sub(1 + rhs_l2_assoc).max(1);
        let mc_from = (lhs_l2_assoc * l2) / (l2_assoc * sizeof * kc).max(1);
        let mut mc = round_down(mc_from, mr).max(mr);
        let m_iter = m.div_ceil(mc).max(1);
        mc = (m.div_ceil(m_iter * mr) * mr).max(mr);
        mc = mc.min(8 * mr); // BLIS hard cap

        // --- NC: B macro-panel resides in L3 (reserve one way for A) ---
        let nc = if l3 == 0 {
            // No L3: a fixed, cache-agnostic column block.
            (128 * nr).min(n.next_multiple_of(nr)).max(nr)
        } else {
            let rhs_l3_assoc = l3_assoc.saturating_sub(1).max(1);
            let rhs_macro_max = (rhs_l3_assoc * l3) / l3_assoc;
            let mut nc = round_down(rhs_macro_max / (sizeof * kc).max(1), nr).max(nr);
            let n_iter = n.div_ceil(nc).max(1);
            nc = (n.div_ceil(n_iter * nr) * nr).max(nr);
            nc
        };

        Blocking { mc, kc, nc }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// The detected page size must be a power of two in a sane range (the
    /// LHS-packing stride gate is derived from it), on whatever host runs the test.
    #[test]
    fn page_size_is_plausible() {
        let p = page_size();
        assert!(p.is_power_of_two(), "page size {p} is not a power of two");
        assert!(
            (4096..=2 * 1024 * 1024).contains(&p),
            "page size {p} out of range"
        );
    }
}
