//! macOS `sysctl` cache backend (Apple Silicon + Intel Macs).
//!
//! On Apple Silicon there is no CPUID, so this is the *primary* topology source;
//! on Intel Macs the CPUID backend runs first and this is a fallback. Values are
//! read through `sysctlbyname` via a tiny `extern "C"` block so the crate keeps
//! its dependency surface minimal (no `libc`).
//!
//! Apple Silicon exposes per-performance-level keys (`hw.perflevel0.*`, where
//! `perflevel0` is the P-cores); the older flat keys (`hw.l1dcachesize`, …) are
//! used as a fallback for Intel Macs. Associativity is not exposed by `sysctl`,
//! so conservative typical values are assumed — the BLIS blocking model only
//! needs them approximately and clamps with `.max(2)`.

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

/// Read one integer sysctl by (NUL-terminated) name. `None` if the key is
/// absent. Values may be 4- or 8-byte; we read into a zeroed `u64` and rely on
/// little-endian (every macOS target is) so a short 4-byte write still leaves
/// the correct value in the low bytes.
fn sysctl_u64(name: &[u8]) -> Option<u64> {
    debug_assert_eq!(name.last().copied(), Some(0), "name must be NUL-terminated");
    let mut val: u64 = 0;
    let mut len = core::mem::size_of::<u64>();
    // SAFETY: `name` is a valid NUL-terminated C string; `&mut val`/`&mut len`
    // are valid for `len` bytes; passing a null `newp` makes this a pure read.
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

/// Best-effort cache topology from `sysctl`; `None` if the required L1/L2 keys
/// are unavailable (the caller then falls through the chain).
pub fn detect() -> Option<CacheTopology> {
    let line = sysctl_u64(b"hw.cachelinesize\0").unwrap_or(64) as usize;

    // Prefer the P-core (`perflevel0`) view; fall back to the flat Intel keys.
    let l1 =
        sysctl_u64(b"hw.perflevel0.l1dcachesize\0").or_else(|| sysctl_u64(b"hw.l1dcachesize\0"))?;
    let l2 =
        sysctl_u64(b"hw.perflevel0.l2cachesize\0").or_else(|| sysctl_u64(b"hw.l2cachesize\0"))?;
    // Apple Silicon reports no conventional L3 (the system-level cache is not
    // exposed as L3): treat L3 as absent unless a key is present and non-zero.
    let l3 = sysctl_u64(b"hw.perflevel0.l3cachesize\0")
        .or_else(|| sysctl_u64(b"hw.l3cachesize\0"))
        .filter(|&b| b > 0);

    // On Apple Silicon the L2 is shared by a whole core cluster
    // (`hw.perflevel0.cpusperl2`, e.g. 5 on an M-series P-cluster). Dividing by
    // it gives the realistic per-core L2 budget the BLIS model should block for;
    // L1d is per-core. Default to 1 (private) when the key is absent (Intel).
    let l2_shared = sysctl_u64(b"hw.perflevel0.cpusperl2\0")
        .filter(|&c| c > 0)
        .unwrap_or(1) as usize;

    // `sysctl` exposes no associativity; use conservative typical values.
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
