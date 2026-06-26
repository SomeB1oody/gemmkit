//! CPUID cache backend (x86 / x86-64).
//!
//! CPUID is an *instruction*, so this works regardless of OS — including in
//! containers and most VMs (unless the hypervisor masks the leaves, in which
//! case [`detect`] returns `None` and the caller falls through the chain).
//! Intel exposes the deterministic cache leaf (CPUID.04h); AMD uses the legacy
//! L1 (0x80000005) and L2/L3 (0x80000006) leaves.

use super::{CacheTopology, Level};
use raw_cpuid::{Associativity, CacheType, CpuId, CpuIdReader};

/// Best-effort cache topology from CPUID; `None` if the leaves are unavailable.
pub fn detect() -> Option<CacheTopology> {
    let cpuid = CpuId::new();
    let vi = cpuid.get_vendor_info();
    let vendor = vi.as_ref().map(|v| v.as_str()).unwrap_or("");

    if vendor.contains("AMD") {
        detect_amd(&cpuid)
    } else {
        // Intel deterministic leaf, with the AMD leaves as a secondary attempt.
        detect_intel(&cpuid).or_else(|| detect_amd(&cpuid))
    }
}

fn assoc_num(a: Associativity) -> usize {
    match a {
        Associativity::DirectMapped => 1,
        Associativity::NWay(n) => n as usize,
        Associativity::FullyAssociative => 64,
        // Unknown / Disabled / "see leaf 0x8000001D" → a safe default the
        // blocking model tolerates.
        _ => 8,
    }
}

fn detect_amd<R: CpuIdReader>(cpuid: &CpuId<R>) -> Option<CacheTopology> {
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

    // L3 size is reported in units of 512 KiB.
    let l3_bytes = l23.l3cache_size() as usize * 512 * 1024;
    let l3 = (l3_bytes > 0).then(|| Level {
        bytes: l3_bytes,
        assoc: assoc_num(l23.l3cache_associativity()),
        line: l23.l3cache_line_size() as usize,
        shared_by: 1,
    });

    Some(CacheTopology { l1d, l2, l3 })
}

fn detect_intel<R: CpuIdReader>(cpuid: &CpuId<R>) -> Option<CacheTopology> {
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
