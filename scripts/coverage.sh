#!/usr/bin/env bash
# gemmkit coverage runner (cargo-llvm-cov, stable -C instrument-coverage).
#
# Accumulates coverage across the runtime-dispatch matrix (GEMMKIT_REQUIRE_ISA
# pins are memoized per process via a OnceLock -> one cargo llvm-cov invocation
# per ISA) and across feature combos, then emits ONE merged per-arch report.
#
# Usage:
#   scripts/coverage.sh                 # auto-detect arch, probe available ISAs
#   COVERAGE_ISAS="scalar fma" scripts/coverage.sh
#                                       # force the ISA list (skips the probe);
#                                       # CI uses this so the run is deterministic
#                                       # across the heterogeneous runner pool.
#   COVERAGE_COMBOS=0 scripts/coverage.sh
#                                       # skip the feature-combo passes (ISA
#                                       # passes + report only) -- for smoke runs.
#   COVERAGE_EXTRA_ARGS="--test env" scripts/coverage.sh
#                                       # extra cargo args appended to every
#                                       # --no-report pass (narrow the test set
#                                       # for a fast pathway smoke).
#
# Output (ARCH = x86_64 | aarch64):
#   coverage/lcov-<ARCH>.info           merged lcov (product code only)
#   coverage/html-<ARCH>/html/index.html  browsable report
#                                         (cargo-llvm-cov nests it under html/)
#   coverage/meta-<ARCH>.txt            git rev, host, date, ISA passes run
#   terminal summary
#
# Test-failure policy: passes run with --no-fail-fast, but --no-fail-fast only
# suppresses intra-run early abort -- cargo still exits non-zero if any test
# fails. This script therefore does NOT use `set -e`; instead every pass and
# report step records into FAIL and the script exits with that status at the
# end, so a red pass still produces the merged report (for triage) yet the run
# is visibly non-green.
#
# Doctests are deliberately NOT included: cargo-llvm-cov --doctests is unstable
# (nightly-only) and this workspace is stable-only (MSRV 1.89). The only doc
# example lives in gemmkit/src/lib.rs. Benches (criterion) and the #[ignore]
# perf suite are compiled but never run (no --benches, no --include-ignored).
set -uo pipefail
cd "$(dirname "$0")/.."

command -v cargo-llvm-cov >/dev/null || { echo "error: cargo-llvm-cov not installed" >&2; exit 1; }

# Product-code report only: the tests/ and benches/ trees and the gemmkit-tune
# binary (zero #[test]s, a benchmark-sweep main) are excluded from the *report*.
IGNORE_RE='(^|/)tests/|(^|/)benches/|gemmkit-tune/'
COVDIR="${COVERAGE_DIR:-coverage}"

ARCH="$(uname -m)"   # x86_64 | arm64 (macOS) | aarch64 (linux)
case "$ARCH" in arm64) ARCH=aarch64 ;; esac

# Sanitize ambient GEMMKIT_* (a sourced gemmkit-tune profile would skew the
# tuning-knob routes; cf. gemmkit/tests/env.rs which defends against the same).
while IFS='=' read -r v _; do unset "$v"; done < <(env | grep -E '^GEMMKIT_[A-Z0-9_]+=' || true)

# --- ISA list -----------------------------------------------------------
# COVERAGE_ISAS overrides the probe (deterministic CI; the GitHub ubuntu-latest
# pool is heterogeneous -- some machines expose avx512, some do not -- so probing
# there would make the coverage % swing run-to-run). Local runs probe cpuinfo.
have_flag() { grep -qwE "$1" /proc/cpuinfo 2>/dev/null; }
if [ -n "${COVERAGE_ISAS:-}" ]; then
    read -ra ISAS <<< "$COVERAGE_ISAS"
else
    ISAS=(scalar)
    if [ "$ARCH" = x86_64 ]; then
        # f16c is required for the fma pin on f16 (dispatch.rs f16 Fma arm:
        # "avx2+fma+f16c"); --all-features turns `half` on, so probe all three.
        { have_flag 'avx2' && have_flag 'fma' && have_flag 'f16c'; } && ISAS+=(fma)
        have_flag 'avx512f'                    && ISAS+=(avx512)
        have_flag 'avx512_vnni|avx512vnni'     && ISAS+=(avx512vnni)
        have_flag 'avx512_bf16|avx512bf16'     && ISAS+=(avx512bf16)
    elif [ "$ARCH" = aarch64 ]; then
        ISAS+=(neon)   # baseline on aarch64; fma/avx512*/simd128 pins panic here
    fi
fi
echo ">> arch=$ARCH  ISA passes: ${ISAS[*]}"

read -ra EXTRA <<< "${COVERAGE_EXTRA_ARGS:-}"
FAIL=0
run() {  # run a pass; accumulate FAIL rather than aborting under set -e
    echo ">> $1"; shift
    "$@" || { echo "!! pass FAILED (rc=$?): $*" >&2; FAIL=1; }
}

# --- accumulate ---------------------------------------------------------
run "clean" cargo llvm-cov clean --workspace

for isa in "${ISAS[@]}"; do
    run "ISA pass: $isa (all features)" \
        env GEMMKIT_REQUIRE_ISA="$isa" \
        cargo llvm-cov --no-report -p gemmkit --all-features --no-fail-fast \
        "${EXTRA[@]+"${EXTRA[@]}"}"
done

if [ "${COVERAGE_COMBOS:-1}" != 0 ]; then
    # One combo pass pins GEMMKIT_REQUIRE_ISA=auto to exercise the explicit
    # "auto" parse arm (dispatch.rs) that the unset (Err(_)) passes never reach.
    run "combo pass: workspace, default features (ISA=auto)" \
        env GEMMKIT_REQUIRE_ISA=auto \
        cargo llvm-cov --no-report --workspace --exclude gemmkit-tune --no-fail-fast \
        "${EXTRA[@]+"${EXTRA[@]}"}"
    run "combo pass: workspace, all features" \
        cargo llvm-cov --no-report --workspace --exclude gemmkit-tune --all-features --no-fail-fast \
        "${EXTRA[@]+"${EXTRA[@]}"}"
    run "combo pass: gemmkit, parallel OFF (std,half,int8,complex)" \
        cargo llvm-cov --no-report -p gemmkit --no-default-features \
        --features std,half,int8,complex --no-fail-fast \
        "${EXTRA[@]+"${EXTRA[@]}"}"
fi

# --- merged report ------------------------------------------------------
mkdir -p "$COVDIR"
run "report --lcov" \
    cargo llvm-cov report --lcov --output-path "$COVDIR/lcov-$ARCH.info" \
    --ignore-filename-regex "$IGNORE_RE"
# NB: cargo-llvm-cov nests the HTML under <output-dir>/html/, so the browsable
# index is $COVDIR/html-$ARCH/html/index.html.
run "report --html" \
    cargo llvm-cov report --html --output-dir "$COVDIR/html-$ARCH" \
    --ignore-filename-regex "$IGNORE_RE"
run "report --summary" \
    cargo llvm-cov report --ignore-filename-regex "$IGNORE_RE"

{ echo "rev=$(git rev-parse HEAD)"; echo "host=$(uname -a)"; echo "date=$(date -u +%FT%TZ)";
  echo "isas=${ISAS[*]}"; } > "$COVDIR/meta-$ARCH.txt"

echo ">> done: $COVDIR/lcov-$ARCH.info, $COVDIR/html-$ARCH/html/index.html (FAIL=$FAIL)"
exit "$FAIL"
