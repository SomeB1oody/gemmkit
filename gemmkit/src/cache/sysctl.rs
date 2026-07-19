//! macOS `sysctl` cache backend (Apple Silicon + Intel Macs)
//!
//! Apple Silicon has no CPUID instruction, so this is the *primary* topology
//! source there; on an Intel Mac the CPUID backend runs first and this only
//! covers whatever it misses. Values come from `sysctlbyname` through a tiny
//! hand-written `extern "C"` declaration, so the crate does not need to pull in
//! `libc` just for this one call
//!
//! Apple Silicon exposes per-performance-level keys (`hw.perflevel0.*`, where
//! `perflevel0` names the P-cores); the older flat keys (`hw.l1dcachesize`, and
//! so on) are read as a fallback, which is what an Intel Mac actually has.
//! `sysctl` has no associativity key at all, so this backend just assumes typical
//! values: the BLIS blocking model only needs an approximate way count and
//! clamps every level to at least 2 anyway

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

/// Read one integer-valued sysctl by its (NUL-terminated) name. `None` when the
/// key does not exist. The kernel may report the value as a 4-byte or an 8-byte
/// integer; reading into a zeroed `u64` and relying on every macOS target being
/// little-endian means a short 4-byte write still lands as the correct value in
/// the low bytes, with the high bytes staying 0
fn sysctl_u64(name: &[u8]) -> Option<u64> {
    debug_assert_eq!(name.last().copied(), Some(0), "name must be NUL-terminated");
    let mut val: u64 = 0;
    let mut len = core::mem::size_of::<u64>();
    // SAFETY: name is a valid NUL-terminated C string; val and len are valid,
    // properly sized out-parameters; a null newp makes this a read-only call
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

/// Best-effort cache topology from `sysctl`; `None` when the L1d or L2 size key
/// is missing (the caller then falls through to the next backend)
pub fn detect() -> Option<CacheTopology> {
    let line = sysctl_u64(b"hw.cachelinesize\0").unwrap_or(64) as usize;

    // Prefer the P-core (perflevel0) view; an Intel Mac has no perflevel0 keys
    // at all, so the flat key is what actually answers there
    let l1 =
        sysctl_u64(b"hw.perflevel0.l1dcachesize\0").or_else(|| sysctl_u64(b"hw.l1dcachesize\0"))?;
    let l2 =
        sysctl_u64(b"hw.perflevel0.l2cachesize\0").or_else(|| sysctl_u64(b"hw.l2cachesize\0"))?;
    // Apple Silicon has no conventional per-core L3 (its system-level cache is
    // not exposed through this key), so treat a missing or zero reading as none
    let l3 = sysctl_u64(b"hw.perflevel0.l3cachesize\0")
        .or_else(|| sysctl_u64(b"hw.l3cachesize\0"))
        .filter(|&b| b > 0);

    // `cpusperl2` counts the cores in one P-cluster sharing an L2 (e.g. 5 on an
    // M-series P-cluster); dividing the raw L2 size by it gives the per-worker
    // budget the BLIS model needs (see Level::shared_by). L1d has no such
    // sharing. Default to 1 (private L2) when the key is absent, as on Intel
    let l2_shared = sysctl_u64(b"hw.perflevel0.cpusperl2\0")
        .filter(|&c| c > 0)
        .unwrap_or(1) as usize;

    // No associativity key exists in sysctl at all; fill in typical values
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
