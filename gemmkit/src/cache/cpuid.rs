//! CPUID cache backend (x86 / x86-64)
//!
//! Cache geometry comes straight from the CPUID instruction, so this works the
//! same in a container or most VMs as on bare metal; only a hypervisor that
//! masks the relevant leaves makes [`detect`] return `None`, in which case the
//! caller falls through to the next backend. Both vendors expose a per-cache
//! topology leaf (Intel CPUID.04h, AMD 0x8000_001D), walked sub-leaf by
//! sub-leaf; it describes each cache as reachable from the executing core, so
//! on a multi-die part (a 2-CCD Ryzen) the L3 figure is the one complex a core
//! can actually hit, not the package total - the semantic every consumer wants
//! (see [`super::Level::bytes`]). AMD parts or hypervisors without that leaf
//! fall back to the legacy L1 (0x8000_0005) and L2/L3 (0x8000_0006) leaves,
//! which only know a package-total L3 and a coarse associativity encoding

use super::{CacheTopology, Level};
use raw_cpuid::{Associativity, CacheType, CpuId, CpuIdReader};

/// Best-effort cache topology from CPUID; `None` when both leaf families are
/// unreadable
pub fn detect() -> Option<CacheTopology> {
    let cpuid = CpuId::new();
    // The topology leaf first: raw-cpuid selects CPUID.04h or 0x8000_001D by
    // vendor, and both report per-core-reachable caches. The legacy AMD leaves
    // are only a fallback, at the cost of a package-total L3 figure on
    // multi-CCD parts and a guessed associativity
    detect_topology_leaf(&cpuid).or_else(|| detect_amd_legacy(&cpuid))
}

// Map raw_cpuid's associativity encoding to a plain way count
fn assoc_num(a: Associativity) -> usize {
    match a {
        Associativity::DirectMapped => 1,
        Associativity::NWay(n) => n as usize,
        // Not a real way count, just a value large enough that the blocking model
        // treats the cache as effectively unconstrained by associativity
        Associativity::FullyAssociative => 64,
        // Disabled, or Unknown ("see leaf 0x8000_001d"): no ways figure to report,
        // so fall back to a plausible default
        _ => 8,
    }
}

// AMD legacy leaves: the L1 (0x8000_0005) and L2/L3 (0x8000_0006) leaves must both
// decode; L3 is reported inside the L2/L3 leaf and treated as absent below when its
// size field reads 0, not by the leaf itself being unreadable. Only reached when the
// topology leaf is unavailable: the L3 size here is the package total, which
// overstates what one core can hit on a multi-CCD part, and 16-way associativity is
// not even encodable (it reads back as `Unknown`)
fn detect_amd_legacy<R: CpuIdReader>(cpuid: &CpuId<R>) -> Option<CacheTopology> {
    let l1 = cpuid.get_l1_cache_and_tlb_info()?;
    let l23 = cpuid.get_l2_l3_cache_and_tlb_info()?;

    let l1d = Level {
        bytes: l1.dcache_size() as usize * 1024,
        assoc: assoc_num(l1.dcache_associativity()),
        line: l1.dcache_line_size() as usize,
        shared_by: 1,
    };

    let l2 = Level {
        bytes: l23.l2cache_size() as usize * 1024,
        assoc: assoc_num(l23.l2cache_associativity()),
        line: l23.l2cache_line_size() as usize,
        shared_by: 1,
    };

    // The L2/L3 leaf's EDX field reports L3 size in units of 512 KiB, unlike the
    // ECX field's plain-KiB L2 size
    let l3_bytes = l23.l3cache_size() as usize * 512 * 1024;
    let l3 = (l3_bytes > 0).then(|| Level {
        bytes: l3_bytes,
        assoc: assoc_num(l23.l3cache_associativity()),
        line: l23.l3cache_line_size() as usize,
        shared_by: 1,
    });

    Some(CacheTopology { l1d, l2, l3 })
}

// The per-cache topology leaf (Intel CPUID.04h, AMD 0x8000_001D - raw-cpuid picks
// by vendor): the sub-leaf iterator stops on its own at the Null terminator, so this
// just keeps the 1st Data-or-Unified entry seen at each of levels 1-3. L3 is
// optional; L1d and L2 are not, so a topology missing either comes back as None. That
// None also covers `get_cache_parameters` itself returning None, which is what happens
// on an AMD part whose 0x8000_001D is masked: raw-cpuid never falls back to the Intel
// leaf for AMD, so the caller then tries the legacy leaves
fn detect_topology_leaf<R: CpuIdReader>(cpuid: &CpuId<R>) -> Option<CacheTopology> {
    let params = cpuid.get_cache_parameters()?;
    let mut l1d = None;
    let mut l2 = None;
    let mut l3 = None;

    for c in params {
        if !matches!(c.cache_type(), CacheType::Data | CacheType::Unified) {
            continue;
        }
        let bytes =
            c.associativity() * c.coherency_line_size() * c.sets() * c.physical_line_partitions();
        let level = Level {
            bytes,
            assoc: c.associativity(),
            line: c.coherency_line_size(),
            shared_by: 1,
        };
        match c.level() {
            1 => l1d = Some(level),
            2 => l2 = Some(level),
            3 => l3 = Some(level),
            _ => {}
        }
    }

    Some(CacheTopology {
        l1d: l1d?,
        l2: l2?,
        l3,
    })
}

// Canned-CPUID-reader tests for the Intel and AMD detection paths
#[cfg(test)]
mod tests {
    use raw_cpuid::{CpuId, CpuIdResult};

    /// Build a CPUID leaf-04h sub-leaf `CpuIdResult` from decoded fields, following the
    /// real register layout: `eax` packs the cache type into bits 0-4 and the level into
    /// bits 5-7; `ebx` packs line size, physical partitions, and ways, each biased by 1;
    /// `ecx` is the set count, also biased by 1. The cache size those fields describe is
    /// `ways * line * sets * partitions`
    fn leaf04(ctype: u32, level: u32, line: u32, parts: u32, ways: u32, sets: u32) -> CpuIdResult {
        CpuIdResult {
            eax: ctype | (level << 5),
            ebx: (line - 1) | ((parts - 1) << 12) | ((ways - 1) << 22),
            ecx: sets - 1,
            edx: 0,
        }
    }

    /// Feed `detect_topology_leaf` a mock leaf-04h walk (L1d 48 KiB/12-way, an L1
    /// *instruction* cache that the `Data | Unified` filter must skip, L2 1 MiB/8-way
    /// unified, L3 32 MiB/16-way unified) and check the resulting sizes and way counts.
    /// `detect_topology_leaf` is generic over the CPUID reader, so a mock exercises the
    /// Intel flavor of the leaf on any host, including the AMD dev box
    #[test]
    fn detect_topology_leaf_from_canned_intel_leaf04() {
        let reader = |eax: u32, ecx: u32| -> CpuIdResult {
            match (eax, ecx) {
                // Leaf 0: report a max basic leaf of 4 or higher, and spell out
                // "GenuineIntel" across ebx/edx/ecx in that order
                (0x0, _) => CpuIdResult {
                    eax: 0x16,
                    ebx: 0x756e_6547, // "Genu"
                    ecx: 0x6c65_746e, // "ntel"
                    edx: 0x4965_6e69, // "ineI"
                },
                // Leaf 4 sub-leaves in order: L1d, L1i, L2, L3, then anything else
                // falls through to the Null (type 0) terminator below
                (0x4, 0) => leaf04(1, 1, 64, 1, 12, 64), // Data,  L1: 12*64*64      = 48 KiB
                (0x4, 1) => leaf04(2, 1, 64, 1, 8, 64),  // Instruction, L1 (skipped by the filter)
                (0x4, 2) => leaf04(3, 2, 64, 1, 8, 2048), // Unified, L2: 8*64*2048  = 1 MiB
                (0x4, 3) => leaf04(3, 3, 64, 1, 16, 32768), // Unified, L3: 16*64*32768 = 32 MiB
                (0x4, _) => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
                _ => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
            }
        };
        let cpuid = CpuId::with_cpuid_fn(reader);
        let t = super::detect_topology_leaf(&cpuid).expect("canned Intel leaf-04h must detect");

        assert_eq!(t.l1d.bytes, 48 * 1024, "L1d size");
        assert_eq!(t.l1d.assoc, 12, "L1d ways");
        assert_eq!(t.l1d.line, 64, "L1d line");
        assert_eq!(t.l1d.shared_by, 1, "L1d shared_by is fixed at 1");
        assert_eq!(t.l2.bytes, 1024 * 1024, "L2 size");
        assert_eq!(t.l2.assoc, 8, "L2 ways");
        let l3 = t.l3.expect("L3 present");
        assert_eq!(l3.bytes, 32 * 1024 * 1024, "L3 size");
        assert_eq!(l3.assoc, 16, "L3 ways");
    }

    /// Feed `detect_amd_legacy` a mock AMD leaf pair whose associativity nibbles decode
    /// to `DirectMapped` (L1d) and `FullyAssociative` (L2 and L3), encodings a real Zen
    /// part does not emit for those fields, and check that `assoc_num` folds them to
    /// `1` and `64` respectively. A real AMD host never reaches this path (the topology
    /// leaf wins), so the mock is the only coverage for the legacy decode and for the
    /// `DirectMapped` and `FullyAssociative` arms
    #[test]
    fn detect_amd_legacy_exotic_associativities() {
        let reader = |eax: u32, _ecx: u32| -> CpuIdResult {
            match eax {
                // Leaf 0: spell "AuthenticAMD" across ebx/edx/ecx (max basic leaf value,
                // eax, is not read by this path)
                0x0 => CpuIdResult {
                    eax: 0x10,
                    ebx: 0x6874_7541, // "Auth"
                    ecx: 0x444d_4163, // "cAMD"
                    edx: 0x6974_6e65, // "enti"
                },
                // Extended-function max leaf: must be >= 0x8000_0006 so the L1/L2/L3
                // leaves below are considered valid
                0x8000_0000 => CpuIdResult {
                    eax: 0x8000_0008,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
                // L1 cache/TLB leaf: ecx = size(KiB) << 24 | assoc << 16 | line;
                // assoc 0x01 decodes to DirectMapped
                0x8000_0005 => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: (64 << 24) | (0x01 << 16) | 64,
                    edx: 0,
                },
                // L2/L3 cache leaf: ecx = l2size(KiB) << 16 | l2assoc << 12 | l2line;
                // edx = l3size(*512 KiB) << 18 | l3assoc << 12 | l3line;
                // assoc 0xF decodes to FullyAssociative on both
                0x8000_0006 => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: (512 << 16) | (0xF << 12) | 64,
                    edx: (16 << 18) | (0xF << 12) | 64,
                },
                _ => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
            }
        };
        let cpuid = CpuId::with_cpuid_fn(reader);
        let t = super::detect_amd_legacy(&cpuid).expect("canned AMD leaves must detect");

        assert_eq!(t.l1d.bytes, 64 * 1024, "L1d size");
        assert_eq!(t.l1d.assoc, 1, "DirectMapped L1d folds to assoc 1");
        assert_eq!(t.l2.bytes, 512 * 1024, "L2 size");
        assert_eq!(t.l2.assoc, 64, "FullyAssociative L2 folds to assoc 64");
        let l3 = t.l3.expect("L3 present");
        assert_eq!(l3.bytes, 16 * 512 * 1024, "L3 size (units of 512 KiB)");
        assert_eq!(l3.assoc, 64, "FullyAssociative L3 folds to assoc 64");
    }

    /// Feed `detect_topology_leaf` the AMD flavor of the leaf (0x8000_001D, selected by
    /// raw-cpuid on an "AuthenticAMD" vendor with the extended max leaf high enough),
    /// canned as the 9950X reports it: L3 32 MiB/16-way, the one-CCD slice a core can
    /// hit. The mock also serves the legacy L2/L3 leaf with the 64 MiB package total,
    /// so the asserts prove the topology leaf is the one being decoded - the exact
    /// misread this backend used to ship (64 MiB, associativity guessed as 8, from the
    /// legacy leaf) can never come back silently
    #[test]
    fn detect_topology_leaf_from_canned_amd_leaf1d() {
        let reader = |eax: u32, ecx: u32| -> CpuIdResult {
            match (eax, ecx) {
                // Leaf 0: spell "AuthenticAMD" across ebx/edx/ecx
                (0x0, _) => CpuIdResult {
                    eax: 0x10,
                    ebx: 0x6874_7541, // "Auth"
                    ecx: 0x444d_4163, // "cAMD"
                    edx: 0x6974_6e65, // "enti"
                },
                // Extended-function max leaf: must be >= 0x8000_001D so raw-cpuid
                // routes get_cache_parameters to the AMD topology leaf
                (0x8000_0000, _) => CpuIdResult {
                    eax: 0x8000_0020,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
                // The legacy L2/L3 leaf, reporting the 64 MiB package-total L3
                // (128 * 512 KiB): decoding this instead of the topology leaf below
                // is exactly the regression the asserts rule out
                (0x8000_0006, _) => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: (1024 << 16) | (0x6 << 12) | 64,
                    edx: (128 << 18) | (0x9 << 12) | 64,
                },
                // Topology leaf sub-leaves, 9950X geometry: L1d, L1i (skipped by the
                // Data | Unified filter), L2, the per-CCD L3, then the Null terminator
                (0x8000_001D, 0) => leaf04(1, 1, 64, 1, 12, 64), // Data, L1: 48 KiB
                (0x8000_001D, 1) => leaf04(2, 1, 64, 1, 8, 64),  // Instruction, L1
                (0x8000_001D, 2) => leaf04(3, 2, 64, 1, 16, 1024), // Unified, L2: 1 MiB
                (0x8000_001D, 3) => leaf04(3, 3, 64, 1, 16, 32768), // Unified, L3: 32 MiB
                _ => CpuIdResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                },
            }
        };
        let cpuid = CpuId::with_cpuid_fn(reader);
        let t = super::detect_topology_leaf(&cpuid).expect("canned AMD leaf-1Dh must detect");

        assert_eq!(t.l1d.bytes, 48 * 1024, "L1d size");
        assert_eq!(t.l1d.assoc, 12, "L1d ways");
        assert_eq!(t.l2.bytes, 1024 * 1024, "L2 size");
        assert_eq!(t.l2.assoc, 16, "L2 ways");
        let l3 = t.l3.expect("L3 present");
        assert_eq!(
            l3.bytes,
            32 * 1024 * 1024,
            "L3 is the per-CCD slice, not the total"
        );
        assert_eq!(l3.assoc, 16, "L3 ways read exactly, not guessed");
        assert_eq!(l3.shared_by, 1, "L3 shared_by is fixed at 1");
    }
}
