//! macOS `sysctl` cache backend (placeholder).
//!
//! Reserved seam for Apple Silicon: it will read `hw.perflevelN.l1dcachesize`
//! etc. once the NEON ISA token lands. v1 returns `None` so the fallback chain
//! proceeds.

use super::CacheTopology;

pub fn detect() -> Option<CacheTopology> {
    None
}
