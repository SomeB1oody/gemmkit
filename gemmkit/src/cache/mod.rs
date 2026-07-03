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

// Every backend is consumed only by the `std`-gated `detect()` below (the no-`std`
// path uses the static fallback in `Machine::current`), so each is gated on `std`
// too — otherwise the module compiles but goes uncalled, tripping `dead_code`.
#[cfg(all(
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]
mod cpuid;
// `not(miri)`: the sysctl backend calls the `sysctlbyname` foreign function
#[cfg(all(feature = "std", target_os = "macos", not(miri)))]
mod sysctl;
// `not(miri)`: the sysfs backend reads `/sys` via `std::fs`
#[cfg(all(feature = "std", target_os = "linux", not(miri)))]
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
    /// Number of concurrent GEMM workers that contend for *per-worker* data at
    /// this level — **not** the raw hardware core-sharing count. It divides
    /// [`Level::effective_bytes`], which feeds the blocking model, so it must
    /// reflect contention for the data the driver actually places at this level.
    /// The driver keeps the *shared* B macro-panel in L3 and *per-worker* A/B
    /// micropanels in L1d, so **L1d and L3 are always `1`** (budget the whole
    /// level to that one panel; dividing L3 by the core count would crater `NC`).
    /// Only **L2** — which holds each worker's private A macro-panel — uses the
    /// physical-core L2-sharing degree: `1` for a private L2 (x86, Neoverse), the
    /// cluster size for a shared L2 (Apple). A backend must therefore *derive*
    /// this value, never store a raw `shared_cpu_list` count.
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

/// Aggregated host facts detected once from the machine: the data-cache hierarchy
/// used for blocking and the OS memory page size used for the LHS-packing stride
/// gate. Detection runs at most once and is memoized behind a single `OnceLock`;
/// [`topology`] and the crate-internal page-size accessor both read through here.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// The data-cache hierarchy used for blocking.
    pub cache: CacheTopology,
    /// The OS memory page size in bytes.
    pub page_size: usize,
}

#[cfg(feature = "std")]
static MACHINE: OnceLock<Machine> = OnceLock::new();

impl Machine {
    /// The detected host facts (memoized; detection runs at most once).
    #[cfg(feature = "std")]
    pub fn current() -> &'static Machine {
        MACHINE.get_or_init(|| Machine {
            cache: detect(),
            page_size: detect_page_size(),
        })
    }

    /// Without `std` there is no memoization or detection, so the calibrated
    /// fallback is returned (Zen5 cache geometry, 4 KiB page).
    #[cfg(not(feature = "std"))]
    pub fn current() -> &'static Machine {
        static FALLBACK: Machine = Machine {
            cache: ZEN5_FALLBACK,
            page_size: 4096,
        };
        &FALLBACK
    }
}

/// The detected cache topology (memoized via [`Machine`]; detection runs once).
pub fn topology() -> &'static CacheTopology {
    &Machine::current().cache
}

/// Detect the OS memory page size. `getpagesize` is POSIX/BSD and present on both
/// Linux and macOS; `std` already links libc so a bare declaration resolves with
/// no extra dependency. Called once from [`Machine::current`].
#[cfg(all(unix, feature = "std", not(miri)))]
fn detect_page_size() -> usize {
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
}

/// Page-size detection fallback for non-unix / Miri `std` builds (assume the common
/// 4 KiB). The no-`std` build skips detection entirely (see [`Machine::current`]).
#[cfg(all(feature = "std", not(all(unix, not(miri)))))]
fn detect_page_size() -> usize {
    4096
}

/// The OS memory page size in bytes (memoized via [`Machine`]). Drives the
/// LHS-packing stride gate.
pub(crate) fn page_size() -> usize {
    Machine::current().page_size
}

/// The LHS-packing depth-stride gate in *bytes*. The `GEMMKIT_LHS_PACK_STRIDE`
/// knob overrides it verbatim; `0` (the default) derives it from the OS page size
/// — half a page, so a column-major A whose K-walk stride (`csa * sizeof`) reaches
/// it is packed to dodge TLB thrash. Centralized here (rather than inlined in the
/// driver) so the `0 => auto` derivation has a single home and a direct test.
pub(crate) fn lhs_pack_stride_bytes() -> usize {
    match crate::tuning::lhs_pack_stride() {
        0 => page_size() / 2,
        v => v,
    }
}

/// The gemv/gevv parallelism byte floor: below this much touched data the problem is
/// LLC-resident and one core already gets the full LLC bandwidth, so splitting only adds
/// fork/join and shared-cache contention with no DRAM to gain. `GEMMKIT_GEMV_PARALLEL_BYTES`
/// overrides it; `0` (the default) derives it from the last-level cache. Centralized here
/// (like [`lhs_pack_stride_bytes`]) as the one home for the `0 => auto` derivation.
///
/// * **With an L3** (x86, Graviton): a quarter of the per-core L3 share (`effective_bytes`).
/// * **No L3** (Apple's shared cluster-L2): *half the full shared L2* — not
///   `effective_bytes`, which divides by the cluster size (`shared_by`) for the per-worker
///   BLIS budget. For the serial-vs-parallel gemv question a single core streams from the
///   *whole* cluster L2 but cannot saturate its bandwidth, so splitting across the cluster
///   still gains once the matrix exceeds ~half of it. Calibrated on M4 Max (16 MiB P-cluster
///   L2): parallel is 0.64× serial at a 4 MiB matrix and 1.12× at 8 MiB, so the ~8 MiB floor
///   (`l2.bytes / 2`) keeps small gemv serial and parallelizes the DRAM-bound ones.
#[cfg(feature = "parallel")]
pub(crate) fn gemv_parallel_floor_bytes() -> usize {
    match crate::tuning::gemv_parallel_bytes() {
        0 => {
            let t = topology();
            match t.l3 {
                Some(l3) => (l3.effective_bytes() / 4).max(1),
                None => (t.l2.bytes / 2).max(1),
            }
        }
        v => v,
    }
}

/// Output-size gate for the axpy-gemv register-block strategy (see
/// [`crate::special`]'s `output_register_block`): it engages only once the output is large enough
/// that the plain column-outer form's per-column output re-reads spill toward DRAM. Register-block
/// *loses* while the output is L3-resident (the many matrix-stream prefetches thrash while the
/// re-reads stay cheap) and *wins* once the output approaches leaving L3. Calibrated on Zen5 (it
/// loses up to a ~16 MiB output and wins from ~32 MiB), i.e. around half the per-core L3 share.
/// No L3 (Apple): fall back to the L2 gate, pending on-device calibration. Sibling of
/// [`gemv_parallel_floor_bytes`] — the one home for cache-derived byte thresholds. Not gated on
/// `parallel`: the serial gemv path uses it too.
pub(crate) fn gemv_regblock_engage_bytes() -> usize {
    let t = topology();
    match t.l3 {
        Some(l3) => (l3.effective_bytes() / 2).max(1),
        None => t.l2.effective_bytes().max(1),
    }
}

/// Run the fallback chain once. Never panics: any backend that fails or returns
/// implausible values is skipped.
#[cfg(feature = "std")]
fn detect() -> CacheTopology {
    // use cpuid as backend
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
    if let Some(t) = cpuid::detect().filter(plausible) {
        return t;
    }
    // the backend reads `/sys`
    #[cfg(all(target_os = "linux", not(miri)))]
    if let Some(t) = sysfs::detect().filter(plausible) {
        return t;
    }
    // the backend calls `sysctlbyname`
    #[cfg(all(target_os = "macos", not(miri)))]
    if let Some(t) = sysctl::detect().filter(plausible) {
        return t;
    }
    ZEN5_FALLBACK
}

/// Sanity gate so a half-populated CPUID read can't poison blocking
#[cfg(feature = "std")]
#[cfg_attr(any(target_family = "wasm", miri), allow(dead_code))]
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
    /// * `sizeof`: size in bytes of one *packed input* element
    ///
    /// The result is independent of the thread count, so serial and parallel
    /// runs use identical blocking — the mechanism behind reproducible output
    /// under a fixed config.
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

        // Runtime blocking knobs, read once (never inside the model's arithmetic below).
        let tiny_dim = crate::tuning::tiny_block_dim();
        let kc_cap = crate::tuning::kc();
        let kc_floor = crate::tuning::kc_min();
        let mc_panels = crate::tuning::mc_reg_panels();
        let nc_panels = crate::tuning::nc_no_l3_panels();

        let l1 = self.l1d.effective_bytes().max(32 * 1024);
        let l2 = self.l2.effective_bytes();
        let l3 = self.l3.map(|l| l.effective_bytes()).unwrap_or(0);
        let line = self.l1d.line.max(64);
        let l1_assoc = self.l1d.assoc.max(2);
        let l2_assoc = self.l2.assoc.max(2);
        let l3_assoc = self.l3.map(|l| l.assoc).unwrap_or(2).max(2);
        let l1_n_sets = (l1 / (line * l1_assoc)).max(1);

        // Small-matrix shortcut: skip the full model, just keep panels in L2.
        if m <= tiny_dim && n <= tiny_dim {
            let kc = k.clamp(1, kc_cap);
            // Cap by the actual row count: there are only `m` rows, so a larger `mc`
            // never splits fewer blocks (`n_mc` stays 1)
            let mc = ((l2 / sizeof / kc) / mr * mr)
                .min(m.next_multiple_of(mr))
                .max(mr);
            let nc = n.next_multiple_of(nr).max(nr);
            return Blocking { mc, kc, nc };
        }

        // --- KC: A & B micropanels coexist in L1 without self-eviction ---
        let g = gcd(mr * sizeof, line * l1_n_sets);
        let kc_0 = (line * l1_n_sets) / g;
        let c_lhs = (mr * sizeof) / g;
        let c_rhs = (nr * kc_0 * sizeof) / (line * l1_n_sets);
        let kc_mult = (l1_assoc / (c_lhs + c_rhs).max(1)).max(1);
        let mut kc = (kc_0 * kc_mult.next_power_of_two()).max(kc_floor).min(k);
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
        mc = mc.min(mc_panels.saturating_mul(mr)); // BLIS hard cap

        // --- NC: B macro-panel resides in L3 (reserve one way for A) ---
        let nc = if l3 == 0 {
            // No L3: full-`N` up to the panel-count cap. Dead where an L3 exists.
            nc_panels
                .saturating_mul(nr)
                .min(n.next_multiple_of(nr))
                .max(nr)
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

    /// The aggregate is the single source of truth: it is memoized (every
    /// `current()` hands back the same instance) and the back-compat accessors
    /// `topology()` / `page_size()` read straight through it.
    #[test]
    fn machine_aggregates_and_memoizes() {
        let m = Machine::current();
        assert!(
            core::ptr::eq(m, Machine::current()),
            "current() must return the one memoized instance"
        );
        assert_eq!(&m.cache, topology(), "topology() must read through Machine");
        assert_eq!(
            m.page_size,
            page_size(),
            "page_size() must read through Machine"
        );
    }

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

    /// The LHS-pack stride gate: the default `0` knob must resolve to *half the page*
    /// (the page-derived auto path that determinism assertions cannot observe), and
    /// any non-zero knob must pass through verbatim as a byte threshold. Guards the
    /// `0 => page_size()/2` derivation and the override branch against regression
    /// (e.g. an inverted match or a changed divisor).
    #[test]
    fn lhs_pack_stride_gate_auto_and_override() {
        // Auto: 0 => exactly half the page (and never zero, so the gate can fire).
        crate::tuning::set_lhs_pack_stride(0);
        let auto = lhs_pack_stride_bytes();
        assert_eq!(auto, page_size() / 2, "auto gate must be half the page");
        assert!(auto > 0, "auto gate must be non-zero");
        // Override: any non-zero value is the byte threshold verbatim.
        crate::tuning::set_lhs_pack_stride(4096);
        assert_eq!(lhs_pack_stride_bytes(), 4096, "override must pass through");
        // Restore the default so concurrent/later tests see auto.
        crate::tuning::set_lhs_pack_stride(0);
    }
}
