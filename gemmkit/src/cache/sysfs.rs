//! Linux `sysfs` cache backend
//!
//! Reads the per-level directories under `/sys/devices/system/cpu/cpu0/cache/`.
//! On x86 Linux the CPUID backend runs first and covers detection in the normal
//! case, so this only fires as a fallback (a VM masking CPUID); on aarch64-Linux,
//! which has no CPUID instruction, this is the **primary** source. Implemented
//! with plain `std::fs` reads: no FFI, no extra dependency
//!
//! Per the [`Level::shared_by`] contract, `shared_by` is *derived* rather than
//! copied straight from a `shared_cpu_list` count: L1d and L3 are always `1`,
//! and L2 is the number of **physical** cores sharing it (the raw L2
//! `shared_cpu_list` count divided by the SMT degree read off L1d's own
//! `shared_cpu_list`). On a private-L2 part (mainstream x86, Neoverse) this
//! divides out to `1` everywhere, matching what the CPUID backend reports

use super::{CacheTopology, Level};

/// Read and trim one sysfs cache field (e.g. `size`) from `dir`. `None` if the
/// file is missing or unreadable
fn read_field(dir: &str, field: &str) -> Option<String> {
    std::fs::read_to_string(format!("{dir}/{field}"))
        .ok()
        .map(|s| s.trim().to_string())
}

/// Parse a sysfs cache `size` value, such as `48K`, `1024K`, `32M`, `2G`, or a
/// bare byte count with no suffix, into a byte count. `None` if `s` is empty,
/// the numeric part does not parse, or the multiply overflows `usize`
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let last = *s.as_bytes().last()?; // empty string has no last byte
    let (num, mult): (&str, usize) = match last {
        b'K' | b'k' => (&s[..s.len() - 1], 1024),
        b'M' | b'm' => (&s[..s.len() - 1], 1024 * 1024),
        b'G' | b'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    num.trim().parse::<usize>().ok()?.checked_mul(mult)
}

/// Count the CPU ids named by a sysfs `shared_cpu_list`, a comma-separated list
/// of single ids and/or `a-b` ranges such as `"0,16"` or `"0-7,16-23"`. A
/// malformed or reversed range contributes 0; the total is clamped to at
/// least 1, since a real cache is always shared by at least the reading CPU
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

/// Read one `index*` directory into a [`Level`] (`shared_by` set to a
/// placeholder `1`, fixed up by the caller) paired with the raw
/// `shared_cpu_list` count. `None` if `size`, the associativity, or the line
/// size is missing or non-positive, so a partially-populated directory is
/// skipped rather than turned into a bogus `Level`
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

/// Best-effort cache topology from `sysfs`; `None` when no `index*` directory
/// yields a usable L1d or L2 (the caller then falls through to the next backend)
pub fn detect() -> Option<CacheTopology> {
    const BASE: &str = "/sys/devices/system/cpu/cpu0/cache";
    // Level plus its raw shared_cpu_list count, kept for the 1st index seen at
    // each level (a 2nd index at the same level, if any, is ignored)
    let mut l1d: Option<(Level, usize)> = None;
    let mut l2: Option<(Level, usize)> = None;
    let mut l3: Option<(Level, usize)> = None;

    // index0, index1, ... are contiguous on every real cpu0/cache tree, so a
    // missing `level` file marks the end of the list
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
            // Only Data or Unified counts as L1d; the L1 instruction cache is
            // also reported at level 1 but must not be picked up here
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
    // Fix up the placeholder shared_by from read_level into the derived value
    // (see the module doc and Level::shared_by): L1d and L3 each budget their
    // whole level to a single panel, so shared_by is 1 regardless of the raw
    // sharing count. L2 holds one worker's private A macro-panel, so its
    // shared_by is the physical-core sharing degree: the raw L2 shared_cpu_list
    // count divided by the SMT degree, which L1d's own shared_cpu_list gives
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
// x86-only: cross-checks the sysfs parser against the CPUID backend, plus the
// parse_size/count_cpu_list unit tests
mod tests {
    /// A real x86 Linux host has both backends reading the same physical caches,
    /// so their **L1d/L2** sizes and line sizes must agree exactly; CPUID is the
    /// oracle here, so this checks the sysfs parser against it without needing
    /// aarch64 hardware. L3 is deliberately *not* compared: on a multi-CCD AMD
    /// part CPUID reports the whole-package L3, while sysfs reports only the
    /// per-CCD slice `cpu0` sits behind, so the 2 readings legitimately differ. Either
    /// backend can return `None` in a container that masks CPUID or hides
    /// `/sys`, so the test skips instead of failing when that happens
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
        // L1d and L3 shared_by is hard-set to 1 by this backend. L2 shared_by is
        // derived (the physical-core L2-sharing degree), which is 1 on a
        // private-L2 part (mainstream Core/Zen, Neoverse) but the cluster size
        // on a shared-L2 part (Intel Atom/E-core modules, Apple), so its value
        // depends on the host's micro-architecture
        assert_eq!(sf.l1d.shared_by, 1, "L1d shared_by must be 1");
        assert!(
            sf.l2.shared_by >= 1,
            "L2 shared_by must be a positive count"
        );
        if let Some(s3) = sf.l3 {
            assert_eq!(s3.shared_by, 1, "L3 shared_by must be 1");
        }
    }

    /// Drive every `parse_size` suffix arm (K, M, G, each cased both ways, and the
    /// bare-byte fall-through), plus the empty/non-numeric/overflow rejections. The
    /// dev box's own sysfs only ever emits `K` sizes, so the `M`/`G`/bare-byte arms
    /// need a synthetic input to run at all
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
        assert_eq!(super::parse_size(""), None); // empty string
        assert_eq!(super::parse_size("K"), None); // suffix with no digits
        assert_eq!(super::parse_size("notanumber"), None);
        // The multiply overflows usize and is rejected, not wrapped
        assert_eq!(super::parse_size(&format!("{}G", usize::MAX)), None);
    }

    /// Drive `count_cpu_list` over a single id, an `a-b` range, several
    /// comma-separated parts, and the empty/malformed inputs that must clamp to
    /// at least 1. The dev box's own `shared_cpu_list` values are always
    /// comma-separated (SMT sibling pairs, an L3 range), never a single id, so
    /// that arm and the malformed-input clamps still need synthetic input
    #[test]
    fn count_cpu_list_all_arms() {
        assert_eq!(super::count_cpu_list("0-7,16-23"), 16); // 2 ranges
        assert_eq!(super::count_cpu_list("0-7"), 8); // 1 range
        assert_eq!(super::count_cpu_list("0,16"), 2); // 2 single ids
        assert_eq!(super::count_cpu_list("3"), 1); // 1 single id
        assert_eq!(super::count_cpu_list(""), 1); // empty -> clamp to 1
        assert_eq!(super::count_cpu_list("  "), 1); // whitespace -> clamp to 1
        assert_eq!(super::count_cpu_list("5-2"), 1); // reversed range ignored -> clamp to 1
        assert_eq!(super::count_cpu_list("0-3,,8"), 5); // empty part skipped: 4 + 1
    }
}
