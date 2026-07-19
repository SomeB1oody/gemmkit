//! Cache topology and analytical blocking (layer L3)
//!
//! 2 facts shape the design:
//!
//! 1. **`#[cfg]` only picks the *sniffing method*, never the *values*.** A VM or
//!    container can mask CPUID or hide `/sys`, and `#[cfg(target_arch)]` cannot
//!    tell an Intel apart from an AMD. So every backend is best-effort, and there
//!    is always a runtime fallback chain that cannot fail
//! 2. The blocking sizes `(MC, KC, NC)` are computed analytically from the cache
//!    geometry (the BLIS model), so they adapt to the detected machine instead of
//!    being hard-coded per micro-architecture
//!
//! The fallback chain tries, in order, the CPUID backend (x86), the sysfs backend
//! (Linux), the sysctl backend (macOS), and finally the [`ZEN5_FALLBACK`] static
//! default (Ryzen 9950X). The 1st backend that succeeds and passes a plausibility
//! check wins. Detection runs at most once per process and the result is memoized
//! in [`Machine`]

// x86/x86-64 CPUID cache backend; gated on `std` since only the `std`-gated
// `detect()` function consumes it, which would otherwise leave it dead code
#[cfg(all(
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]
mod cpuid;
// macOS sysctl cache backend; also gated `not(miri)`, which does not support
// its `sysctlbyname` FFI call
#[cfg(all(feature = "std", target_os = "macos", not(miri)))]
mod sysctl;
// Linux sysfs cache backend; also gated `not(miri)`, which isolates file reads
// from the host by default and so cannot see the real `/sys` tree
#[cfg(all(feature = "std", target_os = "linux", not(miri)))]
mod sysfs;

#[cfg(feature = "std")]
use std::sync::OnceLock;

/// One cache level
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// Total size in bytes, as reported by the backend, before dividing by `shared_by`
    pub bytes: usize,
    /// Associativity (ways)
    pub assoc: usize,
    /// Cache line size in bytes
    pub line: usize,
    /// Number of concurrent GEMM workers that contend for the *per-worker* data the
    /// driver keeps at this level, not the raw hardware core-sharing count. It divides
    /// [`Level::effective_bytes`], which feeds the blocking model, so it must reflect
    /// contention for the data the driver actually places here. The driver keeps the
    /// *shared* B macro-panel in L3 and *per-worker* A/B micropanels in L1d, so **L1d
    /// and L3 are always `1`**: the whole level is budgeted to that one panel, and
    /// dividing L3 by the core count would crater `NC`. Only **L2**, which holds each
    /// worker's private A macro-panel, uses the physical-core L2-sharing degree: `1`
    /// for a private L2 (x86, Neoverse), the cluster size for a shared L2 (Apple). A
    /// backend must therefore *derive* this value, never store a raw `shared_cpu_list`
    /// count
    pub shared_by: usize,
}

impl Level {
    /// Effective per-worker capacity: `bytes / shared_by`
    #[inline]
    pub fn effective_bytes(&self) -> usize {
        self.bytes / self.shared_by.max(1)
    }
}

/// The data-cache hierarchy used for blocking
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CacheTopology {
    /// L1 data cache
    pub l1d: Level,
    /// L2 cache
    pub l2: Level,
    /// L3 cache, if any (some Apple / embedded parts report none)
    pub l3: Option<Level>,
}

/// Blocking parameters for the 5-loop driver
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Blocking {
    /// Rows of A / C per L3-resident macro-panel iteration (loop 3)
    pub mc: usize,
    /// Depth per L2/L1-resident panel iteration (loop 4)
    pub kc: usize,
    /// Columns of B / C per L3-resident macro-panel iteration (loop 5)
    pub nc: usize,
}

/// The Zen5 (Ryzen 9950X) calibrated default: the bottom of the fallback chain, used
/// when every runtime backend fails or is unavailable. L1d 48 KiB / 12-way, L2 1 MiB
/// / 16-way (private), L3 32 MiB / 16-way (per CCD, treated as fully available for
/// the B macro-panel, i.e. `shared_by = 1`)
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

/// Aggregated host facts, detected once from the running machine: the data-cache
/// hierarchy used for blocking and the OS memory page size used for the LHS-packing
/// stride gate. Detection runs at most once and is memoized behind a single
/// `OnceLock`; [`topology`] and the crate-internal page-size accessor both read
/// through this struct rather than detecting independently
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// The data-cache hierarchy used for blocking
    pub cache: CacheTopology,
    /// The OS memory page size in bytes
    pub page_size: usize,
}

#[cfg(feature = "std")]
static MACHINE: OnceLock<Machine> = OnceLock::new();

impl Machine {
    /// The detected host facts, computed on the 1st call and memoized for every call after
    #[cfg(feature = "std")]
    pub fn current() -> &'static Machine {
        MACHINE.get_or_init(|| Machine {
            cache: detect(),
            page_size: detect_page_size(),
        })
    }

    /// Without `std` there is no `OnceLock` to memoize into and no OS to probe, so this
    /// always returns the calibrated fallback (Zen5 cache geometry, 4 KiB page)
    #[cfg(not(feature = "std"))]
    pub fn current() -> &'static Machine {
        static FALLBACK: Machine = Machine {
            cache: ZEN5_FALLBACK,
            page_size: 4096,
        };
        &FALLBACK
    }
}

/// The detected cache topology, read through the memoized [`Machine`]
pub fn topology() -> &'static CacheTopology {
    &Machine::current().cache
}

/// Detect the OS memory page size via `getpagesize`, called once from [`Machine::current`].
/// `getpagesize` is POSIX/BSD and present on both Linux and macOS, and `std` already links
/// libc, so a bare `extern "C"` declaration resolves with no extra dependency
#[cfg(all(unix, feature = "std", not(miri)))]
fn detect_page_size() -> usize {
    unsafe extern "C" {
        fn getpagesize() -> core::ffi::c_int;
    }
    let p = unsafe { getpagesize() } as usize;
    // Reject an implausible reading (a real base page is a power of 2 in a sane range)
    if p.is_power_of_two() && (4096..=2 * 1024 * 1024).contains(&p) {
        p
    } else {
        4096
    }
}

/// Page-size fallback for non-unix targets and for Miri, which cannot make the
/// `getpagesize` FFI call: assume the common 4 KiB. The no-`std` build skips detection
/// entirely and hard-codes the same value (see [`Machine::current`])
#[cfg(all(feature = "std", not(all(unix, not(miri)))))]
fn detect_page_size() -> usize {
    4096
}

/// The OS memory page size in bytes, read through the memoized [`Machine`]. Drives
/// the LHS-packing stride gate
pub(crate) fn page_size() -> usize {
    Machine::current().page_size
}

/// The LHS-packing depth-stride gate in bytes: a column-major A whose K-walk stride
/// (`csa * sizeof`) reaches this is packed instead, to dodge a TLB/cache-hostile strided
/// read. The `GEMMKIT_LHS_PACK_STRIDE` knob overrides it verbatim; `0` (the default) derives
/// it from the OS page size (half a page). Centralized here, rather than inlined at the
/// driver's call site, so the `0 => auto` derivation has a single home and a direct test
pub(crate) fn lhs_pack_stride_bytes() -> usize {
    match crate::tuning::lhs_pack_stride() {
        0 => page_size() / 2,
        v => v,
    }
}

/// The gemv/gevv parallelism byte floor: below this much touched data the problem is
/// LLC-resident and one core already gets the full LLC bandwidth, so splitting only adds
/// fork/join and shared-cache contention with no DRAM bandwidth to gain. `GEMMKIT_GEMV_PARALLEL_BYTES`
/// overrides it verbatim; `0` (the default) derives it from the last-level cache. Centralized
/// here (like [`lhs_pack_stride_bytes`]) as the one home for the `0 => auto` derivation
///
/// * With an L3 (x86, Graviton): a quarter of the per-core L3 share ([`Level::effective_bytes`])
/// * No L3 (Apple's shared cluster-L2): half the *full* shared L2, not `effective_bytes`,
///   which divides by the cluster size (`shared_by`) for the per-worker BLIS budget. For the
///   serial-vs-parallel gemv question, a single core streams from the whole cluster L2 but
///   cannot saturate its bandwidth alone, so splitting across the cluster still gains once the
///   matrix exceeds about half of it. Calibrated on an M4 Max (16 MiB P-cluster L2): parallel
///   ran 0.64x serial at a 4 MiB matrix and 1.12x at 8 MiB, so the ~8 MiB floor (`l2.bytes / 2`)
///   keeps small gemv serial and parallelizes only the DRAM-bound ones
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

/// Output-size gate for the axpy-gemv register-block strategy (`output_register_block` in
/// [`crate::special`]): it engages only once the output is large enough that the plain
/// column-outer form's per-column output re-reads spill toward DRAM. Register-blocking *loses*
/// while the output is L3-resident (the extra matrix-stream prefetches thrash while the re-reads
/// stay cheap) and *wins* once the output approaches leaving L3. Calibrated on Zen5, where it
/// loses up to a ~16 MiB output and wins from ~32 MiB, i.e. around half the per-core L3 share.
/// No L3 (Apple): falls back to the L2 gate, pending on-device calibration. Sibling of
/// [`gemv_parallel_floor_bytes`] as a cache-derived byte threshold; not gated on `parallel`
/// since the serial gemv path uses it too
pub(crate) fn gemv_regblock_engage_bytes() -> usize {
    let t = topology();
    match t.l3 {
        Some(l3) => (l3.effective_bytes() / 2).max(1),
        None => t.l2.effective_bytes().max(1),
    }
}

/// The deep-contraction engage gate in bytes: a narrow-output family (`OUT_IS_ACC = false`)
/// runs `kc = k` (a single depth panel), so its RHS micropanel is `nr * k * sizeof(N)` bytes;
/// once that outgrows L2, every microtile call streams it, alongside the even larger `mr * k`
/// LHS micropanel, from L3/DRAM instead. The `GEMMKIT_DEEP_KC_BYTES` knob overrides it verbatim;
/// `0` (the default) derives it from half the L2 effective per-worker capacity. Measured on a
/// Zen5 9950X (AVX-512, `nr = 12`, `f16`/`bf16`): the throughput cliff hits at `k = 32768` (a
/// 768 KiB RHS micropanel, about 0.75x the 1 MiB L2), while `k = 16384` (384 KiB) is still near
/// peak, so a full-L2 gate would engage only past `k ~ 43000` and miss the cliff entirely. The
/// `L2/2` gate instead switches to the multi-slice twin at `k = 32768`/`65536` (2.8x / 3.6x
/// faster for `f16`) while leaving `16384` and below on the single panel, where the twin is
/// within noise, so no regression there. Centralized here, like [`lhs_pack_stride_bytes`] and
/// [`gemv_parallel_floor_bytes`], as the one home for the `0 => auto` derivation and its direct
/// test. Consumed only by the `half` dispatch (and the in-module test), so it is gated to match
/// and stay dead-code-free in half-less builds
#[cfg(any(test, feature = "half"))]
pub(crate) fn deep_k_engage_bytes() -> usize {
    match crate::tuning::deep_kc_bytes() {
        0 => (topology().l2.effective_bytes() / 2).max(1),
        v => v,
    }
}

/// Run the fallback chain once, returning the 1st backend's result that both succeeds
/// and passes [`plausible`]. Never panics: a backend that fails or returns implausible
/// values is simply skipped
#[cfg(feature = "std")]
fn detect() -> CacheTopology {
    // Try the CPUID backend
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
    if let Some(t) = cpuid::detect().filter(plausible) {
        return t;
    }
    // Try the sysfs backend
    #[cfg(all(target_os = "linux", not(miri)))]
    if let Some(t) = sysfs::detect().filter(plausible) {
        return t;
    }
    // Try the sysctl backend
    #[cfg(all(target_os = "macos", not(miri)))]
    if let Some(t) = sysctl::detect().filter(plausible) {
        return t;
    }
    ZEN5_FALLBACK
}

/// Sanity gate on a detected topology, so a half-populated or garbled backend read
/// (e.g. a masked CPUID leaf) cannot poison the blocking model: every present level
/// needs a size of at least 4 KiB, a line of at least 16 bytes, and at least 1-way
/// associativity. A missing L3 passes (`None` is a valid topology)
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
    /// Compute `(MC, KC, NC)` analytically (the BLIS model) for the given microkernel
    /// tile and problem size
    ///
    /// # Parameters
    /// - `mr`/`nr` - microkernel tile shape, in elements
    /// - `sizeof` - size in bytes of one packed input element
    /// - `m`/`n`/`k` - problem dimensions
    ///
    /// # Returns
    /// - `Blocking` - the computed `mc`/`kc`/`nc` triple
    ///
    /// # Notes
    /// The result depends only on the cache geometry and the problem shape, never on
    /// the thread count, so serial and parallel runs block identically: the mechanism
    /// behind reproducible output under a fixed config
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

        // Runtime blocking knobs: read once up front, then used as plain values below
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

        // Small-matrix shortcut: skip the full model, just size panels to fit L2
        if m <= tiny_dim && n <= tiny_dim {
            let kc = k.clamp(1, kc_cap);
            // Cap at the rounded-up row count: with only `m` rows total, a larger
            // `mc` cannot split into fewer blocks, so it buys nothing
            let mc = ((l2 / sizeof / kc) / mr * mr)
                .min(m.next_multiple_of(mr))
                .max(mr);
            let nc = n.next_multiple_of(nr).max(nr);
            return Blocking { mc, kc, nc };
        }

        // KC: size the A and B micropanels so both coexist in L1 without evicting each other
        let g = gcd(mr * sizeof, line * l1_n_sets);
        let kc_0 = (line * l1_n_sets) / g;
        let c_lhs = (mr * sizeof) / g;
        let c_rhs = (nr * kc_0 * sizeof) / (line * l1_n_sets);
        let kc_mult = (l1_assoc / (c_lhs + c_rhs).max(1)).max(1);
        let mut kc = (kc_0 * kc_mult.next_power_of_two()).max(kc_floor).min(k);
        let k_iter = k.div_ceil(kc).max(1);
        kc = k.div_ceil(k_iter).max(1); // spread k evenly over k_iter panels, no tiny tail

        // MC: fit the A macro-panel into L2, after reserving the ways the B micropanel
        // needs plus 1 way of headroom
        let rhs_micropanel = nr * kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel.div_ceil((l2 / l2_assoc).max(1));
        let lhs_l2_assoc = l2_assoc.saturating_sub(1 + rhs_l2_assoc).max(1);
        let mc_from = (lhs_l2_assoc * l2) / (l2_assoc * sizeof * kc).max(1);
        let mut mc = round_down(mc_from, mr).max(mr);
        let m_iter = m.div_ceil(mc).max(1);
        mc = (m.div_ceil(m_iter.saturating_mul(mr).max(1)) * mr).max(mr);
        mc = mc.min(mc_panels.saturating_mul(mr)); // hard cap regardless of the model's result

        // NC: fit the B macro-panel into L3, after reserving 1 way for the streamed A data
        let nc = if l3 == 0 {
            // No L3: take the full (rounded-up) N, capped by the panel-count knob. This
            // arm is dead on any machine that does report an L3
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

    /// `Machine` is the single source of truth: `current()` always hands back the same
    /// memoized instance, and the `topology()`/`page_size()` accessors read straight
    /// through it rather than detecting independently
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

    /// The detected page size must be a power of 2 in a sane range, whatever host
    /// runs the test: the LHS-packing stride gate is derived from it, so a garbage
    /// value there would silently break that gate
    #[test]
    fn page_size_is_plausible() {
        let p = page_size();
        assert!(p.is_power_of_two(), "page size {p} is not a power of two");
        assert!(
            (4096..=2 * 1024 * 1024).contains(&p),
            "page size {p} out of range"
        );
    }

    /// The LHS-pack stride gate: the default `0` knob must resolve to exactly half the
    /// page, and any non-zero knob must pass through verbatim as a byte threshold.
    /// Guards the `0 => page_size() / 2` derivation and the override branch against a
    /// regression such as an inverted match or a changed divisor
    #[test]
    fn lhs_pack_stride_gate_auto_and_override() {
        // Auto: 0 => exactly half the page (and never zero, so the gate can fire)
        crate::tuning::set_lhs_pack_stride(0);
        let auto = lhs_pack_stride_bytes();
        assert_eq!(auto, page_size() / 2, "auto gate must be half the page");
        assert!(auto > 0, "auto gate must be non-zero");
        // Override: any non-zero value is the byte threshold verbatim
        crate::tuning::set_lhs_pack_stride(4096);
        assert_eq!(lhs_pack_stride_bytes(), 4096, "override must pass through");
        // Restore the default so concurrent/later tests see auto
        crate::tuning::set_lhs_pack_stride(0);
    }

    /// The deep-contraction engage gate: the default `0` knob must resolve to half the L2
    /// effective per-worker bytes, and any non-zero knob must pass through verbatim. Guards
    /// the `0 => l2 / 2` derivation and the override branch; asserts against the host's own
    /// detected L2, so it holds regardless of which machine runs the test
    #[test]
    fn deep_k_engage_gate_auto_and_override() {
        let restore = crate::tuning::deep_kc_bytes();
        // Auto: 0 => half the L2 effective bytes (and never zero, so the gate can still fire)
        crate::tuning::set_deep_kc_bytes(0);
        let auto = deep_k_engage_bytes();
        assert_eq!(
            auto,
            (topology().l2.effective_bytes() / 2).max(1),
            "auto gate must be half the L2 effective bytes"
        );
        assert!(auto > 0, "auto gate must be non-zero");
        // Override: any non-zero value is the byte threshold verbatim
        crate::tuning::set_deep_kc_bytes(4096);
        assert_eq!(deep_k_engage_bytes(), 4096, "override must pass through");
        crate::tuning::set_deep_kc_bytes(restore);
    }

    /// A degenerate dimension (`m`, `n`, or `k` == 0) short-circuits `blocking` before the BLIS
    /// model runs. The gemm driver itself never calls `blocking` with a 0 dimension (it early-
    /// returns first), so this guard is otherwise unreachable and can only be exercised directly.
    /// Each blocking dim clamps to its microtile floor (or 1 for `kc`), independent of the
    /// detected cache: no tuning knobs are read on this path
    #[test]
    fn blocking_zero_dim_early_return() {
        let t = topology();
        let (mr, nr) = (16usize, 4usize);
        // m == 0
        let b = t.blocking(mr, nr, 4, 0, 8, 8);
        assert_eq!((b.mc, b.kc, b.nc), (mr, 8, 8));
        // n == 0
        let b = t.blocking(mr, nr, 4, 8, 0, 8);
        assert_eq!((b.mc, b.kc, b.nc), (16, 8, nr));
        // k == 0
        let b = t.blocking(mr, nr, 4, 8, 8, 0);
        assert_eq!((b.mc, b.kc, b.nc), (16, 1, 8));
    }

    /// The no-L3 `NC` arm (take the full, rounded-up `N` up to the panel-count cap) is dead on
    /// any machine that reports an L3, so on x86 it only runs here, against a synthetic
    /// `l3: None` topology. `CacheTopology`/`Level`'s fields are public, so a test can build one
    /// directly; this makes the branch coverable platform-independently, while an aarch64 run on
    /// one of Apple's L3-less parts also hits it live through the normal `topology()` path
    #[test]
    fn blocking_no_l3_nc_arm() {
        let topo = CacheTopology {
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
            l3: None,
        };
        let (mr, nr) = (16usize, 4usize);
        let (m, n, k) = (512usize, 512usize, 512usize); // above tiny_block_dim: takes the full model
        let b = topo.blocking(mr, nr, 4, m, n, k);
        // No L3: NC is nc_no_l3_panels * nr, capped by the rounded-up N
        let expect_nc = (crate::tuning::nc_no_l3_panels() * nr)
            .min(n.next_multiple_of(nr))
            .max(nr);
        assert_eq!(b.nc, expect_nc, "no-L3 NC must use the panel-count cap");
        assert!(
            b.mc >= mr && b.mc.is_multiple_of(mr),
            "mc must be a positive mr multiple"
        );
        assert!(b.kc >= 1 && b.kc <= k, "kc must be within [1, k]");
    }
}
