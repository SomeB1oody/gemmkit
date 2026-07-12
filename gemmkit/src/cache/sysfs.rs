//! Linux `sysfs` cache backend.
//!
//! Reads `/sys/devices/system/cpu/cpu0/cache/index*/`. On x86 Linux the CPUID
//! backend covers detection and runs first, so this is only a fallback there (a
//! VM that masks CPUID); it becomes the **primary** source on aarch64-Linux,
//! where there is no CPUID instruction. Pure `std::fs` — no FFI, no dependency.
//!
//! Per the [`Level::shared_by`] contract, `shared_by` is *derived*, never a raw
//! `shared_cpu_list` count: L1d and L3 are always `1`, and L2 is the number of
//! **physical** cores sharing the L2 (the raw L2 sharing count divided by the
//! SMT degree read from L1d's sharing list). On a private-L2 x86/Neoverse part
//! this yields all-`1`, agreeing with the CPUID backend.

use super::{CacheTopology, Level};

/// Read one sysfs cache field (e.g. `size`), trimmed. `None` if absent/unreadable.
fn read_field(dir: &str, field: &str) -> Option<String> {
    std::fs::read_to_string(format!("{dir}/{field}"))
        .ok()
        .map(|s| s.trim().to_string())
}

/// Parse a sysfs cache `size` like `48K`, `1024K`, `32M`, `2G`, or a bare byte
/// count, into bytes. `None` on an empty / unparseable value or on overflow.
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let last = *s.as_bytes().last()?; // empty -> None
    let (num, mult): (&str, usize) = match last {
        b'K' | b'k' => (&s[..s.len() - 1], 1024),
        b'M' | b'm' => (&s[..s.len() - 1], 1024 * 1024),
        b'G' | b'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    num.trim().parse::<usize>().ok()?.checked_mul(mult)
}

/// Count the CPUs named by a sysfs `shared_cpu_list` such as `"0,16"` or
/// `"0-7,16-23"` (comma-separated `a` or `a-b` ranges). At least `1`.
fn count_cpu_list(s: &str) -> usize {
    let mut n = 0usize;
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>())
                    && b >= a
                {
                    n = n.saturating_add((b - a).saturating_add(1));
                }
            }
            None if part.parse::<usize>().is_ok() => n += 1,
            None => {}
        }
    }
    n.max(1)
}

/// Build a fully-populated [`Level`] (with `shared_by` provisionally `1`) plus the
/// raw `shared_cpu_list` count from one `index*` directory, or `None` if any
/// geometry field is missing or zero (so the entry is skipped, not poisoned).
fn read_level(dir: &str) -> Option<(Level, usize)> {
    let bytes = parse_size(&read_field(dir, "size")?)?;
    let assoc = read_field(dir, "ways_of_associativity")?
        .parse::<usize>()
        .ok()?;
    let line = read_field(dir, "coherency_line_size")?
        .parse::<usize>()
        .ok()?;
    let shared = read_field(dir, "shared_cpu_list")
        .map(|s| count_cpu_list(&s))
        .unwrap_or(1);
    (bytes > 0 && assoc > 0 && line > 0).then_some((
        Level {
            bytes,
            assoc,
            line,
            shared_by: 1,
        },
        shared,
    ))
}

/// Best-effort cache topology from `sysfs`; `None` if the required L1d/L2 indexes
/// are absent or half-populated (the caller then falls through the chain).
pub fn detect() -> Option<CacheTopology> {
    const BASE: &str = "/sys/devices/system/cpu/cpu0/cache";
    // (level, shared_count) for the first matching index of each level.
    let mut l1d: Option<(Level, usize)> = None;
    let mut l2: Option<(Level, usize)> = None;
    let mut l3: Option<(Level, usize)> = None;

    // Cache indexes are contiguous `index0, index1, ...`; stop at the first gap.
    for i in 0..64 {
        let dir = format!("{BASE}/index{i}");
        let Some(level) = read_field(&dir, "level").and_then(|s| s.parse::<usize>().ok()) else {
            break;
        };
        let ty = read_field(&dir, "type").unwrap_or_default();
        let Some(entry) = read_level(&dir) else {
            continue;
        };
        match level {
            // L1 *data* (or unified): never the instruction cache.
            1 if ty == "Data" || ty == "Unified" => {
                l1d.get_or_insert(entry);
            }
            2 => {
                l2.get_or_insert(entry);
            }
            3 => {
                l3.get_or_insert(entry);
            }
            _ => {}
        }
    }

    let (mut l1d, l1d_shared) = l1d?;
    let (mut l2, l2_shared) = l2?;
    // `shared_by` derivation (see the module / `Level::shared_by` docs): L1d and
    // L3 budget their whole level to one panel (=1); L2 holds a private per-worker
    // A panel, so it is the *physical*-core L2-sharing degree = raw L2 sharing
    // count / SMT degree (the latter read from L1d, which siblings share).
    let smt = l1d_shared.max(1);
    l1d.shared_by = 1;
    l2.shared_by = (l2_shared / smt).max(1);
    let l3 = l3.map(|(mut l, _)| {
        l.shared_by = 1;
        l
    });

    Some(CacheTopology { l1d, l2, l3 })
}

#[cfg(all(
    test,
    target_os = "linux",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]
mod tests {
    /// On an x86 Linux host both backends read the same physical caches, so the
    /// `sysfs` parser must agree with CPUID on the **L1d/L2** sizes and line size
    /// — the per-core levels both report identically. (L3 is deliberately *not*
    /// compared: on a multi-CCD AMD part the CPUID leaf reports the whole-package
    /// L3 while `sysfs` reports the per-CCD slice `cpu0` actually sees, so they
    /// legitimately differ; neither feeds a hardcoded constant here.) This
    /// validates the parser on x86, where CPUID is the oracle, without needing
    /// aarch64 hardware. Either backend may return `None` in a container that
    /// masks CPUID or hides `/sys`, so the test skips rather than failing then.
    #[test]
    fn sysfs_agrees_with_cpuid() {
        let (Some(sf), Some(cp)) = (super::detect(), super::super::cpuid::detect()) else {
            eprintln!("skipping: a backend returned None (CPUID masked or /sys hidden)");
            return;
        };
        assert_eq!(sf.l1d.bytes, cp.l1d.bytes, "L1d size mismatch");
        assert_eq!(sf.l2.bytes, cp.l2.bytes, "L2 size mismatch");
        assert_eq!(sf.l1d.line, cp.l1d.line, "L1d line mismatch");
        assert_eq!(sf.l2.line, cp.l2.line, "L2 line mismatch");
        // L1d and L3 are hard-set to 1 by the backend.
        // L2 is *derived* (physical-core L2-sharing degree):
        // 1 on a private-L2 part (mainstream Core/Zen, Neoverse)
        // but the cluster size on a shared-L2 part (Intel Atom / E-core modules, Apple),
        // so it is micro-arch dependent
        assert_eq!(sf.l1d.shared_by, 1, "L1d shared_by must be 1");
        assert!(
            sf.l2.shared_by >= 1,
            "L2 shared_by must be a positive count"
        );
        if let Some(s3) = sf.l3 {
            assert_eq!(s3.shared_by, 1, "L3 shared_by must be 1");
        }
    }

    /// `parse_size` over every suffix arm (K/M/G, both cases, and the bare-byte fall-through)
    /// plus the empty / non-numeric / overflow rejections. This host's sysfs only ever emits
    /// `K` sizes, so the `M`/`G`/bare arms need a synthetic input to be exercised.
    #[test]
    fn parse_size_all_arms() {
        assert_eq!(super::parse_size("48K"), Some(48 * 1024));
        assert_eq!(super::parse_size("1024k"), Some(1024 * 1024));
        assert_eq!(super::parse_size("32M"), Some(32 * 1024 * 1024));
        assert_eq!(super::parse_size("8m"), Some(8 * 1024 * 1024));
        assert_eq!(super::parse_size("2G"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(super::parse_size("1g"), Some(1024 * 1024 * 1024));
        assert_eq!(super::parse_size("123"), Some(123)); // bare byte count
        assert_eq!(super::parse_size("  64K  "), Some(64 * 1024)); // trimmed
        assert_eq!(super::parse_size(""), None); // empty -> None
        assert_eq!(super::parse_size("K"), None); // no digits
        assert_eq!(super::parse_size("notanumber"), None);
        // Overflow on the multiply is rejected (returns None, never panics/wraps).
        assert_eq!(super::parse_size(&format!("{}G", usize::MAX)), None);
    }

    /// `count_cpu_list` over single ids, `a-b` ranges, multiple comma-separated parts, and the
    /// empty / malformed inputs that must clamp to at least `1`. This host's `shared_cpu_list`
    /// is a single private-core id, so the multi-range arms need synthetic inputs.
    #[test]
    fn count_cpu_list_all_arms() {
        assert_eq!(super::count_cpu_list("0-7,16-23"), 16); // two ranges
        assert_eq!(super::count_cpu_list("0-7"), 8); // one range
        assert_eq!(super::count_cpu_list("0,16"), 2); // two singles
        assert_eq!(super::count_cpu_list("3"), 1); // one single
        assert_eq!(super::count_cpu_list(""), 1); // empty -> clamp to 1
        assert_eq!(super::count_cpu_list("  "), 1); // whitespace -> clamp to 1
        assert_eq!(super::count_cpu_list("5-2"), 1); // reversed range ignored -> clamp to 1
        assert_eq!(super::count_cpu_list("0-3,,8"), 5); // empty part skipped: 4 + 1
    }
}
