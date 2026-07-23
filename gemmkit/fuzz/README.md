[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.md)

# gemmkit fuzzing harness

A [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer + AddressSanitizer)
harness for gemmkit. **Nightly-only** (needs `-Z build-std` + sanitizers) and **excluded
from the stable workspace** — it is its own workspace root (`[workspace]` table in
`Cargo.toml`) with its own `Cargo.lock` / `target/`, so `cargo test/clippy/fmt --workspace`
and the MSRV-1.89 build never touch it.

Nothing here is committed to git (`corpus/`, `artifacts/`, `target/`, `Cargo.lock` are
`.gitignore`d).

## Prerequisites

```sh
rustup toolchain install nightly           # provides rust-src for build-std
rustup component add rust-src --toolchain nightly
cargo install cargo-fuzz --locked          # 0.13.2 verified
```

Run every command below **from this directory** (`gemmkit/fuzz`) with an explicit
`+nightly` (the project rule forbids ambient nightly; there is deliberately no
`rust-toolchain.toml`).

### Ambient-env hygiene (reproducibility)

The `GEMMKIT_*` env vars are resolved once per knob and cached, and `GEMMKIT_REQUIRE_ISA`
is memoized per process. Before any run:

```sh
env | grep '^GEMMKIT_'    # must print nothing, except a deliberate ISA pin (below)
```

An exported tuned profile would silently skew `fuzz_gemm`/`fuzz_batched`/`fuzz_prepack`
and make artifacts non-reproducible elsewhere. (`fuzz_knobs` is immune — it sets every
knob unconditionally each input, which wins over env.)

## The five targets

| target | what it fuzzes | panic policy |
|---|---|---|
| `fuzz_gemm` | valid `gemm`/`gemm_i8`/`gemm_cplx` (f32/f64/f16/bf16/i8/c32/c64), all layouts incl. broadcast A/B, `beta==0` NaN-C contract, optional caller-`Workspace` reuse; differential vs an f64/i32/complex reference | any panic = bug |
| `fuzz_knobs` | sets all 22 process-global tuning knobs to adversarial value classes every input, then runs one small scenario (plain / gemv / small-mn / prepack-B / prepack-A / i8 / batched); the arithmetic-overflow finder | any panic = bug |
| `fuzz_api_validation` | adversarial dims (incl. `2^33`, `usize::MAX`) + `isize` strides (incl. `isize::MIN/MAX`) into the **checked** `gemm`/`gemm_i8`/`gemm_cplx`/`gemm_batched`/`prepack_*` entries | documented `gemmkit:`-prefixed panic accepted; anything else = bug |
| `fuzz_batched` | valid strided-batched `gemm_batched` (broadcast A/B, valid batch strides) + `gemm_batched_slice`; element-wise differential | any panic = bug |
| `fuzz_prepack` | `prepack_rhs`→`gemm_packed_b` and `prepack_lhs`→`gemm_packed_a` round-trips (f32/f64/bf16); tolerance (not bitwise) vs reference | any panic = bug |

Every valid-input target also seeds C's backing buffer with a **canary sentinel** in the
non-view (interleave / pad / inter-element) slots and asserts they are untouched after the
call — a stray out-of-view write surfaces even without an ASan-visible boundary.

`fuzz_api_validation` runs under `catch_unwind` with a silent panic hook; it treats a
`gemmkit:`-prefixed panic as an accepted rejection and `abort()`s (a real finding) on any
other panic (index OOB, arithmetic overflow, …). It skips only plans that *would* fully
pass validation and then do unbounded work (a `WORK_CAP` of 2^24 MACs / a huge single dim
/ a huge batch loop) — all rejection paths stay fully fuzzed.

## Smoke (per target, ~45–60 s — CI-sized)

```sh
for t in fuzz_gemm fuzz_knobs fuzz_api_validation fuzz_batched fuzz_prepack; do
  cargo +nightly fuzz run "$t" -- \
    -max_total_time=45 -max_len=512 -timeout=60 -malloc_limit_mb=1024 -print_final_stats=1
done
```

`-malloc_limit_mb=1024`: a >1 GB single allocation on these tiny dims *is* a knob-robustness
bug — treat such an artifact as a finding, raise the limit only if triage proves it benign.
`-timeout=60`: a validated-but-degenerate huge-dim/huge-batch input that spins shows up as a
timeout — triage as a finding.

## Soak (overnight, for the user)

```sh
# process-parallel, shared corpus (prefer -jobs/-workers over -fork under ASan+threads)
cargo +nightly fuzz run fuzz_knobs -- \
  -max_total_time=14400 -max_len=512 -timeout=60 -malloc_limit_mb=1024 -jobs=4 -workers=4

# per-ISA passes: the dispatch pin is once-per-process, so use SEPARATE processes
for isa in scalar fma avx512f avx512vnni avx512bf16; do
  GEMMKIT_REQUIRE_ISA=$isa \
    cargo +nightly fuzz run fuzz_gemm -- -max_total_time=3600 -max_len=512 -timeout=60
done

# corpus maintenance afterwards
cargo +nightly fuzz cmin fuzz_gemm
```

The corpus in `corpus/<target>/` is auto-created and grows across runs; no manual seeds are
needed (plans are `int_in_range`-driven, so even the empty input decodes to a minimal valid
plan).

### Coverage report over the accumulated corpus

`cargo fuzz coverage` builds with `--build-std=false` by default (composes cleanly), then
renders an llvm-cov report to find dispatch routes / scenarios the plans never reach:

```sh
cargo +nightly fuzz coverage fuzz_gemm
# then render (the exact profdata/binary paths are printed by the command):
$(rustc +nightly --print target-libdir)/../bin/llvm-cov show \
  target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fuzz_gemm \
  -instr-profile=coverage/fuzz_gemm/coverage.profdata \
  -Xdemangler=rustfilt -format=html > /tmp/fuzz_gemm-cov.html
```

## Crash → minimize → Miri-replay → stable regression test

When a target crashes, libFuzzer writes `artifacts/<target>/crash-<sha>` (or
`timeout-…`/`oom-…`) and prints the plan `Debug`.

1. **Reproduce**
   ```sh
   cargo +nightly fuzz run <target> artifacts/<target>/crash-<sha>
   ```
2. **Minimize the input**
   ```sh
   cargo +nightly fuzz tmin <target> artifacts/<target>/crash-<sha>
   ```
3. **Decode to test parameters.** Because every plan stores *resolved* values (manual
   `Arbitrary`), its `Debug` output is literally the dims / strides-layout / alpha-beta
   indices / knob array / parallelism:
   ```sh
   cargo +nightly fuzz fmt <target> artifacts/<target>/<minimized>
   ```
4. **(Optional) Miri replay.** ASan misses uninit reads / provenance bugs that Miri catches,
   and the gemmkit kernels maintain Miri compatibility (`cfg(miri)` paths in
   `tests/correctness.rs`). Translate the decoded plan into a tiny `#[test]` and run it under
   Miri **on the stable workspace** (never depend on this nightly-only fuzz crate):
   ```sh
   cargo +nightly miri test -p gemmkit --test <file> <testname>
   ```
5. **Hand-write a platform-independent stable regression** in `gemmkit/tests/` (assert
   behavior, never machine constants):
   - knob-class crash → `tests/tuning.rs`, holding `knob_guard()` and restoring every knob
     touched (the `KNOB_LOCK` pattern);
   - env-contract crash → a new one-test-per-binary file following `tests/env.rs`;
   - shape/validation crash → `tests/correctness.rs` (`#[should_panic]` for a validation gap
     promoted to a documented panic — precedent: `panic_extent_overflow_view`) or a new
     `tests/fuzz_regressions.rs`.
6. **Verify on stable:**
   ```sh
   cargo test -p gemmkit --all-features --test <file>
   ```
   and confirm the fuzz target no longer crashes on the artifact.

## Work-cap policy (`fuzz_api_validation`)

The prepack entries skip only plans whose pack would be *representable but huge*
(element count fits `usize` yet exceeds `WORK_CAP`) — running those would OOM on
correct behavior. Empty operands (prepack short-circuits) and pack sizes that
overflow `usize` (a documented `gemmkit: … too large` reject) stay fuzzed; the
regression tests for that overflow class live in `gemmkit/tests/props_packed.rs`
(`prepack_*`) and `gemmkit/tests/props_api.rs` (`mixed_huge_k_fails_closed`).
