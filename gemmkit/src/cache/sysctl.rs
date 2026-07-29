//! macOS `sysctl` cache backend (Apple Silicon and Intel Macs)
//!
//! Apple Silicon has no CPUID instruction, so this is the primary topology source
//! there. On an Intel Mac the CPUID backend runs first, and this backend covers
//! only what CPUID misses. Values come from `sysctlbyname` through a small
//! hand-written `extern "C"` declaration, so the crate does not need `libc` for
//! this one call
//!
//! Apple Silicon exposes per-performance-level keys (`hw.perflevel0.*`, where
//! `perflevel0` names the P-cores). The older flat keys, such as `hw.l1dcachesize`,
//! are read as a fallback, which is what an Intel Mac actually has. `sysctl` has no
//! associativity key, so this backend assumes typical values. The BLIS blocking
//! model only needs an approximate way count, and clamps every level to at least 2

use core::ffi::{c_char, c_int, c_void};

use super::{CacheTopology, Level};

unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int;
}

/// Reads one integer-valued sysctl by its NUL-terminated name. Returns `None` when
/// the key does not exist. The kernel may report the value as a 4-byte or an 8-byte
/// integer. Reading into a zeroed `u64` on a little-endian target means a short
/// 4-byte write still lands as the correct value in the low bytes. The high bytes
/// then stay 0
fn sysctl_u64(name: &[u8]) -> Option<u64> {
    debug_assert_eq!(name.last().copied(), Some(0), "name must be NUL-terminated");
    let mut val: u64 = 0;
    let mut len = core::mem::size_of::<u64>();
    // SAFETY: name is a valid NUL-terminated C string. val and len are valid,
    // properly sized out-parameters. A null newp makes this a read-only call
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr() as *const c_char,
            (&mut val as *mut u64).cast::<c_void>(),
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len > 0).then_some(val)
}

/// Reads a best-effort cache topology from `sysctl`
///
/// # Returns
///
/// - `Option<CacheTopology>` - `None` when the L1d or L2 size key is missing, so
///   the caller falls through to the next backend
pub fn detect() -> Option<CacheTopology> {
    let line = sysctl_u64(b"hw.cachelinesize\0").unwrap_or(64) as usize;

    // Prefers the P-core (perflevel0) view. An Intel Mac has no perflevel0 keys,
    // so the flat key answers there instead
    let l1 =
        sysctl_u64(b"hw.perflevel0.l1dcachesize\0").or_else(|| sysctl_u64(b"hw.l1dcachesize\0"))?;
    let l2 =
        sysctl_u64(b"hw.perflevel0.l2cachesize\0").or_else(|| sysctl_u64(b"hw.l2cachesize\0"))?;
    // Apple Silicon has no conventional per-core L3, since its system-level cache
    // is not exposed through this key. This treats a missing or zero reading as
    // none
    let l3 = sysctl_u64(b"hw.perflevel0.l3cachesize\0")
        .or_else(|| sysctl_u64(b"hw.l3cachesize\0"))
        .filter(|&b| b > 0);

    // `cpusperl2` counts the cores in one P-cluster sharing an L2. Dividing the raw
    // L2 size by it gives the per-worker budget the BLIS model needs (see
    // Level::shared_by). L1d has no such sharing, and this defaults to 1, meaning a
    // private L2, when the key is absent, as on Intel
    let l2_shared = sysctl_u64(b"hw.perflevel0.cpusperl2\0")
        .filter(|&c| c > 0)
        .unwrap_or(1) as usize;

    // No associativity key exists in sysctl, so this fills in typical values
    let lvl = |bytes: u64, assoc: usize, shared_by: usize| Level {
        bytes: bytes as usize,
        assoc,
        line,
        shared_by,
    };

    Some(CacheTopology {
        l1d: lvl(l1, 8, 1),
        l2: lvl(l2, 8, l2_shared),
        l3: l3.map(|b| lvl(b, 16, 1)),
    })
}
