//! Linux `sysfs` cache backend (placeholder).
//!
//! On x86 Linux the CPUID backend already covers detection, so this is a
//! reserved seam: it will read `/sys/devices/system/cpu/cpu0/cache/index*/` once
//! aarch64 Linux support lands (where CPUID does not exist). v1 returns `None`
//! so the fallback chain proceeds.

use super::CacheTopology;

pub fn detect() -> Option<CacheTopology> {
    None
}
