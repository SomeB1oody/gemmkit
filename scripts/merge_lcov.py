#!/usr/bin/env python3
"""Merge LCOV coverage traces at line granularity, across machines/arches

Why this exists:
the lcov/genhtml CLIs are not available in the gemmkit dev environment and nothing may be installed
the LCOV text format is trivially mergeable at line granularity with the stdlib,
which is all cross-arch triage needs

Usage:
    merge_lcov.py [--rev SHA] OUT.info IN.info:STRIP_PREFIX [IN.info:PREFIX ...]

Each input is `path:prefix`.
cargo-llvm-cov emits ABSOLUTE `SF:` paths that differ per machine; `
prefix` (the repo root on that machine) is stripped so merged `SF:` paths are repo-relative and align
An `SF:` line that does not start with its prefix is a hard error:
it flags a wrong prefix or mixed-in foreign data
"""

import re
import sys
from collections import defaultdict


def die(msg):
    sys.exit(f"merge_lcov: error: {msg}")


def sidecar_rev(lcov_path):
    """Return the git rev from the coverage.sh meta sidecar, or None"""
    import os

    d, base = os.path.split(lcov_path)
    m = re.fullmatch(r"lcov-(.+)\.info", base)
    if not m:
        return None
    meta = os.path.join(d, f"meta-{m.group(1)}.txt")
    if not os.path.isfile(meta):
        return None
    with open(meta, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("rev="):
                return line[4:].strip()
    return None


def parse(path, prefix, lines, fn_name, fn_count):
    """Accumulate one lcov file into the shared dicts (keyed by repo-rel file)

    lines[file][ln]      -> summed DA count
    fn_name[file][fnln]  -> a representative FN name for that line
    fn_count[file][fnln] -> summed FNDA count
    """
    cur = None
    name2line = {}  # per-file FN name -> line, to route FNDA (name) to a line
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if raw.startswith("SF:"):
                p = raw[3:]
                # Require a path-component boundary so a sibling checkout sharing the
                # string prefix (e.g. .../gemmkit-exp under prefix .../gemmkit) is
                # rejected, not silently merged under a mangled name
                pfx = prefix.rstrip("/")
                if p != pfx and not p.startswith(pfx + "/"):
                    die(f"{path}: SF path {p!r} lacks prefix {prefix!r} "
                        f"(wrong prefix or mixed-revision input)")
                cur = p[len(pfx):].lstrip("/")
                name2line = {}
            elif cur is None:
                continue
            elif raw.startswith("DA:"):
                parts = raw[3:].split(",")
                ln, cnt = int(parts[0]), int(parts[1])
                lines[cur][ln] = lines[cur].get(ln, 0) + cnt
            elif raw.startswith("FN:"):
                lns, name = raw[3:].split(",", 1)
                ln = int(lns)
                name2line[name] = ln
                fn_name[cur].setdefault(ln, name)
                fn_count[cur].setdefault(ln, 0)
            elif raw.startswith("FNDA:"):
                cnts, name = raw[5:].split(",", 1)
                ln = name2line.get(name)
                if ln is not None:
                    fn_count[cur][ln] = fn_count[cur].get(ln, 0) + int(cnts)
            elif raw == "end_of_record":
                cur = None


def main(argv):
    args = argv[1:]
    want_rev = None
    if args and args[0] == "--rev":
        if len(args) < 2:
            die("--rev needs a value")
        want_rev = args[1]
        args = args[2:]
    if len(args) < 2:
        die("usage: merge_lcov.py [--rev SHA] OUT.info IN.info:PREFIX [IN.info:PREFIX ...]")

    out_path, inputs = args[0], args[1:]

    lines = defaultdict(dict)
    fn_name = defaultdict(dict)
    fn_count = defaultdict(dict)

    revs = {}
    if want_rev:
        revs["--rev"] = want_rev
    for spec in inputs:
        if ":" not in spec:
            die(f"input {spec!r} must be PATH:PREFIX")
        path, prefix = spec.rsplit(":", 1)
        rev = sidecar_rev(path)
        if rev is None:
            print(f"merge_lcov: warning: no meta sidecar for {path} "
                  f"(cannot verify git revision)", file=sys.stderr)
        else:
            revs[path] = rev
        parse(path, prefix, lines, fn_name, fn_count)

    distinct = set(revs.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in revs.items())
        die(f"git revision mismatch across inputs ({detail}); "
            f"merging traces from different source trees is unsound")

    out = []
    tot_lf = tot_lh = 0
    for f in sorted(lines):
        out.append(f"SF:{f}")
        fns = fn_name.get(f, {})
        for ln in sorted(fns):
            out.append(f"FN:{ln},{fns[ln]}")
        fnf = len(fns)
        fnh = sum(1 for ln in fns if fn_count[f].get(ln, 0) > 0)
        out.append(f"FNF:{fnf}")
        out.append(f"FNH:{fnh}")
        for ln in sorted(fns):
            out.append(f"FNDA:{fn_count[f].get(ln, 0)},{fns[ln]}")
        lf = lh = 0
        for ln in sorted(lines[f]):
            cnt = lines[f][ln]
            out.append(f"DA:{ln},{cnt}")
            lf += 1
            lh += 1 if cnt > 0 else 0
        out.append(f"LF:{lf}")
        out.append(f"LH:{lh}")
        out.append("end_of_record")
        tot_lf += lf
        tot_lh += lh

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")

    pct = (100.0 * tot_lh / tot_lf) if tot_lf else 0.0
    print(f"merge_lcov: {len(lines)} files, LH/LF = {tot_lh}/{tot_lf} "
          f"({pct:.2f}%) -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
