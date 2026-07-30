//! Cache topology and analytical blocking (layer L3)
//!
//! 2 facts shape the design:
//!
//! 1. A `#[cfg]` only picks the sniffing method, never the values. A VM or
//!    container can mask CPUID or hide `/sys`, and `#[cfg(target_arch)]` cannot
//!    tell an Intel apart from an AMD. Every backend is therefore best-effort,
//!    backed by a runtime fallback chain that cannot fail
//! 2. The blocking sizes `(MC, KC, NC)` are computed analytically from the
//!    cache geometry (the BLIS model). They adapt to the detected machine
//!    instead of being hard-coded per microarchitecture
//!
//! The fallback chain tries, in order, the CPUID backend (x86), the sysfs
//! backend (Linux), the sysctl backend (macOS), and finally the
//! [`ZEN5_FALLBACK`] static default. The 1st backend that succeeds and passes
//! a plausibility check wins. Detection runs at most once per process, and
//! the result is memoized in [`Machine`]

// x86/x86-64 CPUID cache backend. Gated on `std` because only the `std`-gated
// `detect()` function uses it, and it would otherwise be dead code
#[cfg(all(
    feature = "std",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]
mod cpuid;
// macOS sysctl cache backend. Also gated `not(miri)` because Miri does not
// support the `sysctlbyname` FFI call
#[cfg(all(feature = "std", target_os = "macos", not(miri)))]
mod sysctl;
// Linux sysfs cache backend. Also gated `not(miri)` because Miri isolates file
// reads from the host by default and cannot see the real `/sys` tree
#[cfg(all(feature = "std", target_os = "linux", not(miri)))]
mod sysfs;

#[cfg(feature = "std")]
use std::sync::OnceLock;

/// One cache level
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Level {
    /// Size in bytes of the cache as reachable from one core, before dividing by
    /// `shared_by`. Every backend must report the per-core-reachable slice, not a
    /// package aggregate. On a multi-die part, this is the local die's L3 that a
    /// core can reach directly. A different die's slice costs a fabric round trip
    /// close to DRAM
    pub bytes: usize,
    /// Associativity (ways)
    pub assoc: usize,
    /// Cache line size in bytes
    pub line: usize,
    /// Number of concurrent GEMM workers that contend for the per-worker data
    /// the driver keeps at this level, not the raw hardware core-sharing count.
    /// It divides [`Level::effective_bytes`], which feeds the blocking model,
    /// so it must reflect contention for the data the driver places here
    ///
    /// The driver keeps the shared B macro-panel in L3 and the per-worker A/B
    /// micropanels in L1d, so L1d and L3 are always `1`. The whole level is
    /// budgeted to that one panel, and dividing L3 by the core count would
    /// make `NC` wrong. Only L2, which holds each worker's private A
    /// macro-panel, uses the physical-core L2-sharing degree. This is `1` for
    /// a private L2, or the cluster size for a shared L2. A backend must
    /// derive this value, never store a raw `shared_cpu_list`
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
    /// L3 cache, if any (some embedded or shared-cluster-L2 designs report none)
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

/// The Zen5 (Ryzen 9950X) cache geometry, used as the bottom of the fallback
/// chain when every runtime backend fails or is unavailable. L1d is 48 KiB,
/// 12-way. L2 is 1 MiB, 16-way, private. L3 is 32 MiB, 16-way, per CCD,
/// treated as fully available for the B macro-panel (`shared_by = 1`)
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

/// Aggregated host facts, detected once from the running machine. This holds
/// the data-cache hierarchy used for blocking, and the OS memory page size
/// used for the LHS-packing stride gate. Detection runs at most once and is
/// memoized behind a single `OnceLock`. [`topology`] and the crate-internal
/// page-size accessor both read through this struct instead of detecting
/// independently
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
    /// The detected host facts, computed on the 1st call and memoized for
    /// every call after
    #[cfg(feature = "std")]
    pub fn current() -> &'static Machine {
        MACHINE.get_or_init(|| Machine {
            cache: detect(),
            page_size: detect_page_size(),
        })
    }

    /// Without `std` there is no `OnceLock` to memoize into and no OS to
    /// probe. This always returns the fixed fallback (Zen5 cache geometry,
    /// 4 KiB page)
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

/// Detect the OS memory page size using `getpagesize`, called once from
/// [`Machine::current`]. `getpagesize` is POSIX/BSD and present on both
/// Linux and macOS, and `std` already links libc. A bare `extern "C"`
/// declaration resolves with no extra dependency
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

/// Page-size fallback for non-unix targets and for Miri, which cannot make
/// the `getpagesize` FFI call: assume the common 4 KiB. The no-`std` build
/// skips detection entirely and hard-codes the same value (see
/// [`Machine::current`])
#[cfg(all(feature = "std", not(all(unix, not(miri)))))]
fn detect_page_size() -> usize {
    4096
}

/// The OS memory page size in bytes, read through the memoized [`Machine`].
/// Drives the LHS-packing stride gate
pub(crate) fn page_size() -> usize {
    Machine::current().page_size
}

/// The LHS-packing depth-stride gate in bytes. The driver packs a
/// column-major A instead of reading it strided, once its K-walk stride
/// (`csa * sizeof`) reaches this. This avoids a TLB- and cache-hostile
/// strided read. The `GEMMKIT_LHS_PACK_STRIDE` knob overrides it verbatim.
/// The default, `0`, derives it from the OS page size as half a page
pub(crate) fn lhs_pack_stride_bytes() -> usize {
    match crate::tuning::lhs_pack_stride() {
        0 => page_size() / 2,
        v => v,
    }
}

/// The LHS-packing address-span gate in bytes. The stride gate above only
/// forces a pack when the whole depth-slice walk (`csa * sizeof * kc`) also
/// reaches this much address range. A page-scale stride over a span that
/// stays cache-resident re-walks warm lines and costs less than the pack it
/// would otherwise trigger. The `GEMMKIT_LHS_PACK_SPAN` knob overrides it
/// verbatim. The default, `0`, is a flat 4 MiB rather than a value derived
/// from the detected topology. This is because the crossover tracks TLB
/// reach more than cache size
pub(crate) fn lhs_pack_span_bytes() -> usize {
    match crate::tuning::lhs_pack_span() {
        0 => 4 << 20,
        v => v,
    }
}

/// The gemv/gevv parallelism byte floor. Below this many touched bytes the
/// matrix fits one core's private cache, which that core saturates alone. A
/// split then only adds fork/join and shared-cache contention with no
/// bandwidth to gain. Above it the matrix spills to the shared last level,
/// whose bandwidth one core cannot saturate, so a split pays
///
/// The `GEMMKIT_GEMV_PARALLEL_BYTES` knob overrides the floor verbatim. The
/// default, `0`, derives it from the detected cache
///
/// With an L3, the floor is the private L2 size (`l2.bytes`), where the
/// matrix stops fitting private cache and spills to the shared L3
///
/// Without an L3, the cache is a cluster L2 shared across cores instead. The
/// floor is then a fraction of the full shared L2, not
/// [`Level::effective_bytes`], which divides by the cluster size
/// (`shared_by`) for the per-worker BLIS budget. A single core streams from
/// the whole cluster L2 but cannot saturate its bandwidth alone. A split
/// across the cluster gains once the matrix grows past the sizes a single
/// core handles cheaply. On aarch64 the fraction is 1/8 of `l2.bytes`.
/// Elsewhere it is 1/2
#[cfg(feature = "parallel")]
pub(crate) fn gemv_parallel_floor_bytes() -> usize {
    match crate::tuning::gemv_parallel_bytes() {
        0 => {
            let t = topology();
            #[cfg(target_arch = "aarch64")]
            const NO_L3_DIV: usize = 8;
            #[cfg(not(target_arch = "aarch64"))]
            const NO_L3_DIV: usize = 2;
            match t.l3 {
                Some(_) => t.l2.bytes.max(1),
                None => (t.l2.bytes / NO_L3_DIV).max(1),
            }
        }
        v => v,
    }
}

/// The output-size gate for the axpy-gemv register-block strategy
/// (`output_register_block` in [`crate::special`]). The strategy engages
/// only once the output grows large enough that the plain column-outer
/// form's per-column output re-reads spill toward DRAM
///
/// Register-blocking loses while the output stays resident in L3, because
/// its extra matrix-stream prefetches thrash while the re-reads stay cheap
/// anyway. It wins once the output outgrows L3, using the full
/// per-core-reachable L3 as the gate ([`Level::effective_bytes`]). Without
/// an L3, the gate falls back to the L2 effective bytes. It shares its
/// cache-derived design with [`gemv_parallel_floor_bytes`], but does not
/// gate on the `parallel` feature, because the serial gemv path also uses it
pub(crate) fn gemv_regblock_engage_bytes() -> usize {
    let t = topology();
    match t.l3 {
        Some(l3) => l3.effective_bytes().max(1),
        None => t.l2.effective_bytes().max(1),
    }
}

/// The deep-contraction engage gate in bytes. A narrow-output family
/// (`OUT_IS_ACC = false`) runs `kc = k`, a single depth panel, so its RHS
/// micropanel is `nr * k * sizeof(N)` bytes. Once that micropanel outgrows
/// L2, every microtile call streams it, alongside the even larger `mr * k`
/// LHS micropanel, from L3 or DRAM instead
///
/// Past this gate the driver switches to a multi-slice twin that keeps each
/// slice inside L2. The `GEMMKIT_DEEP_KC_BYTES` knob overrides the gate
/// verbatim. The default, `0`, derives it from half the L2 effective
/// per-worker capacity, because the RHS micropanel does not have L2 to
/// itself
///
/// This gate serves only the `half` dispatch and the in-module test, so the
/// function compiles only under test or the `half` feature
#[cfg(any(test, feature = "half"))]
pub(crate) fn deep_k_engage_bytes() -> usize {
    match crate::tuning::deep_kc_bytes() {
        0 => (topology().l2.effective_bytes() / 2).max(1),
        v => v,
    }
}

/// The C-tile prefetch engage gate in bytes. The driver issues a T0
/// prefetch of each output microtile just ahead of its microkernel call. It
/// fires only once the call's working set, meaning the A, B, and C bytes
/// together, exceeds this. Past the gate the output streams from beyond the
/// LLC, where the hardware prefetchers no longer hide the tile's
/// read-modify-write latency. Below it the tiles stay cache-resident and
/// the hint is pure overhead
///
/// The `GEMMKIT_PREFETCH_MIN_BYTES` knob overrides the gate verbatim. The
/// default, `0`, is the per-core-reachable last-level cache
/// ([`Level::effective_bytes`]): L3 where present, L2 otherwise
///
/// On a target without the x86 prefetch emission, the gate still resolves,
/// but the prefetch instruction lowers to nothing, so behavior is unchanged
/// there
pub(crate) fn prefetch_ws_bytes() -> usize {
    match crate::tuning::prefetch_min_bytes() {
        0 => {
            let t = topology();
            match t.l3 {
                Some(l3) => l3.effective_bytes().max(1),
                None => t.l2.effective_bytes().max(1),
            }
        }
        v => v,
    }
}

/// Run the fallback chain once, returning the 1st backend's result that
/// both succeeds and passes [`plausible`]. Never panics: a backend that
/// fails or returns implausible values is simply skipped
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

/// Sanity gate on a detected topology. A half-populated or garbled backend
/// read (e.g. a masked CPUID leaf) cannot corrupt the blocking model.
/// Every present level needs a size of at least 4 KiB, a line of at least
/// 16 bytes, and at least 1-way associativity. A missing L3 passes
/// (`None` is a valid topology)
#[cfg(feature = "std")]
#[cfg_attr(any(target_family = "wasm", miri), allow(dead_code))]
fn plausible(t: &CacheTopology) -> bool {
    let ok = |l: &Level| l.bytes >= 4 * 1024 && l.line >= 16 && l.assoc >= 1;
    ok(&t.l1d) && ok(&t.l2) && t.l3.as_ref().map(ok).unwrap_or(true)
}

// Euclidean greatest common divisor, always at least 1
#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

// Round a down to the nearest multiple of b
#[inline]
fn round_down(a: usize, b: usize) -> usize {
    (a / b) * b
}

impl CacheTopology {
    /// Compute `(MC, KC, NC)` analytically (the BLIS model) for the given
    /// microkernel tile and problem size
    ///
    /// # Parameters
    ///
    /// - `mr` - microkernel row tile size, in elements
    /// - `nr` - microkernel column tile size, in elements
    /// - `sizeof` - size in bytes of one packed input element
    /// - `m` - row count of A and C
    /// - `n` - column count of B and C
    /// - `k` - shared inner dimension of A and B
    ///
    /// # Returns
    ///
    /// - `Blocking` - the computed `mc`, `kc`, and `nc` triple
    ///
    /// # Notes
    ///
    /// The result depends only on the cache geometry and the problem shape,
    /// never on the thread count. Serial and parallel runs therefore block
    /// identically, which is the mechanism behind reproducible output under
    /// a fixed config
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
        // The tiny-branch ceiling resolves against the packed element size, so ask for the
        // depth this element gets rather than reading the raw knob
        let kc_cap = crate::tuning::tiny_kc(sizeof);
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

    /// `Machine` is the single source of truth. `current()` always hands
    /// back the same memoized instance, and the `topology()`/`page_size()`
    /// accessors read straight through it instead of detecting independently
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

    /// The detected page size must be a power of 2 in a sane range, on
    /// whatever host runs the test. The LHS-packing stride gate derives
    /// from it, so a bad value here would silently break that gate
    #[test]
    fn page_size_is_plausible() {
        let p = page_size();
        assert!(p.is_power_of_two(), "page size {p} is not a power of two");
        assert!(
            (4096..=2 * 1024 * 1024).contains(&p),
            "page size {p} out of range"
        );
    }

    /// The LHS-pack stride gate. The default `0` knob must resolve to
    /// exactly half the page, and any non-zero knob must pass through
    /// verbatim as a byte threshold. This guards the `0 => page_size() / 2`
    /// derivation and the override branch against a regression such as an
    /// inverted match or a changed divisor
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

    /// The deep-contraction engage gate. The default `0` knob must resolve
    /// to half the L2 effective per-worker bytes, and any non-zero knob
    /// must pass through verbatim. This guards the `0 => l2 / 2` derivation
    /// and the override branch. The assertion reads the host's own detected
    /// L2, so it holds regardless of which machine runs the test
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

    /// The C-tile prefetch engage gate. The default `0` knob must resolve
    /// to the per-core-reachable last-level cache (L3 where present, L2
    /// otherwise), and any non-zero knob must pass through verbatim. This
    /// guards the `0 => LLC` derivation and the override branch. The
    /// assertion reads the host's own detected topology, so it holds
    /// regardless of which machine runs the test
    #[test]
    fn prefetch_ws_gate_auto_and_override() {
        let restore = crate::tuning::prefetch_min_bytes();
        // Auto: 0 => the per-core-reachable LLC (and never zero, so the gate can still fire)
        crate::tuning::set_prefetch_min_bytes(0);
        let auto = prefetch_ws_bytes();
        let t = topology();
        let expect = match t.l3 {
            Some(l3) => l3.effective_bytes().max(1),
            None => t.l2.effective_bytes().max(1),
        };
        assert_eq!(auto, expect, "auto gate must be the per-core-reachable LLC");
        assert!(auto > 0, "auto gate must be non-zero");
        // Override: any non-zero value is the byte threshold verbatim
        crate::tuning::set_prefetch_min_bytes(4096);
        assert_eq!(prefetch_ws_bytes(), 4096, "override must pass through");
        crate::tuning::set_prefetch_min_bytes(restore);
    }

    /// A degenerate dimension (`m`, `n`, or `k` equal to `0`) short-circuits
    /// `blocking` before the BLIS model runs. The gemm driver itself never
    /// calls `blocking` with a `0` dimension, because it early-returns
    /// first. This guard is otherwise unreachable, and only a direct call
    /// can exercise it. Each blocking dimension clamps to its microtile
    /// floor, or to `1` for `kc`, independent of the detected cache. No
    /// tuning knobs are read on this path
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

    /// The small-matrix shortcut resolves its depth ceiling against the packed element size.
    /// A narrow element (f16, i8) divides the byte budget and takes a deeper block, which
    /// holds the packed panel bytes fixed. This is what stops 1 calibrated number from
    /// under-blocking i8 by 4x. The checks are ratios across element sizes, so an ambient
    /// `GEMMKIT_KC` cannot make them vacuous. Integer division can drop up to 1 element of
    /// depth, so the byte totals match to within 1 element
    #[test]
    fn tiny_shortcut_deepens_the_block_for_a_narrow_element() {
        let t = topology();
        let tiny = crate::tuning::tiny_block_dim();
        // k far past any ceiling, so the ceiling binds instead of k
        let (m, n, k) = (tiny, tiny, 1usize << 22);
        let f32_kc = t.blocking(16, 4, 4, m, n, k).kc;
        let budget = f32_kc * 4;
        for sizeof in [1usize, 2] {
            let bytes = t.blocking(16, 4, sizeof, m, n, k).kc * sizeof;
            assert!(
                bytes <= budget && bytes + sizeof > budget,
                "sizeof {sizeof}: {bytes} panel bytes against a {budget} budget"
            );
        }
    }

    /// Whether a *wide* element also divides the ceiling is a machine property, so the divisor
    /// carries an arch-split cap ([`crate::tuning::KC_SIZEOF_DIV_CAP`]). Either way the depth
    /// stays between the 2 ends of that choice, and never rises as the element widens. The
    /// branch below then pins whichever policy this target measured. It reads the cap at run
    /// time rather than through `#[cfg]`, so both arms compile on every target
    #[test]
    fn tiny_shortcut_bounds_the_block_for_a_wide_element() {
        let t = topology();
        let tiny = crate::tuning::tiny_block_dim();
        let (m, n, k) = (tiny, tiny, 1usize << 22);
        let f32_kc = t.blocking(16, 4, 4, m, n, k).kc;
        let mut prev = f32_kc;
        for sizeof in [8usize, 16] {
            let kc = t.blocking(16, 4, sizeof, m, n, k).kc;
            assert!(
                kc <= f32_kc && kc >= f32_kc * 4 / sizeof,
                "sizeof {sizeof}: depth {kc} outside [{}, {f32_kc}]",
                f32_kc * 4 / sizeof
            );
            assert!(kc <= prev, "sizeof {sizeof}: depth rose with the width");
            prev = kc;
        }
        let kc16 = t.blocking(16, 4, 16, m, n, k).kc;
        if crate::tuning::KC_SIZEOF_DIV_CAP <= 4 {
            // Hold a wide element at the whole ceiling. Dividing it multiplies the slice
            // count, which cost the parallel path 21 to 51 percent on c64 on an M4 Max
            assert_eq!(kc16, f32_kc, "a wide element must keep the whole ceiling");
        } else {
            // Divide at every width, where a private L2 makes panel residency bind first
            assert!(
                kc16 * 16 <= f32_kc * 4 && kc16 * 16 + 16 > f32_kc * 4,
                "a wide element must hold the panel bytes"
            );
        }
    }

    /// The no-L3 `NC` arm (take the full, rounded-up `N` up to the
    /// panel-count cap) is dead on any machine that reports an L3. On x86 it
    /// only runs here, against a synthetic `l3: None` topology.
    /// `CacheTopology` and `Level` expose public fields, so a test can build
    /// one directly, which makes the branch coverable on any platform. An
    /// aarch64 machine that reports no L3 also hits this branch live,
    /// through the normal `topology()` path
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
        // Above tiny_block_dim, so this takes the full model
        let (m, n, k) = (512usize, 512usize, 512usize);
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
