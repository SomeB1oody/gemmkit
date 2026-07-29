[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.md)

# gemmkit fuzzing harness

A [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer + AddressSanitizer)
harness for gemmkit. It is **nightly-only**, since it needs `-Z build-std` and
sanitizers. It is also **excluded from the stable workspace**: it is its own
workspace root (a `[workspace]` table in `Cargo.toml`), with its own
`Cargo.lock` and `target/`. So `cargo test/clippy/fmt --workspace` and the
MSRV-1.89 build never touch it.

Git ignores everything here: `corpus/`, `artifacts/`, `target/`, and
`Cargo.lock`.

## Prerequisites

```sh
rustup toolchain install nightly           # provides rust-src for build-std
rustup component add rust-src --toolchain nightly
cargo install cargo-fuzz --locked          # 0.13.2 verified
```

Run every command below **from this directory** (`gemmkit/fuzz`). Use an
explicit `+nightly` flag every time. The project rule forbids an ambient
nightly toolchain, so this directory deliberately has no
`rust-toolchain.toml`.

### Ambient-env hygiene (reproducibility)

gemmkit resolves each `GEMMKIT_*` environment variable once per knob, then
caches it. It memoizes `GEMMKIT_REQUIRE_ISA` per process. Before any run:

```sh
env | grep '^GEMMKIT_'    # must print nothing, except a deliberate ISA pin (below)
```

An exported tuned profile would silently skew `fuzz_gemm`, `fuzz_batched`,
`fuzz_prepack`, and `fuzz_prepack_i8`. It would also make artifacts
non-reproducible elsewhere. `fuzz_knobs` is immune. It sets every knob
unconditionally on each input, which wins over the environment.

## The 6 targets

| target | what it fuzzes | panic policy |
|---|---|---|
| `fuzz_gemm` | valid `gemm`/`gemm_i8`/`gemm_cplx` calls (f32, f64, f16, bf16, i8, c32, c64), every layout including broadcast A/B, the `beta==0` NaN-C contract, and optional caller-`Workspace` reuse. Gated against an f64/i32/complex differential reference | any panic = bug |
| `fuzz_knobs` | sets all 31 process-global tuning knobs to adversarial value classes on every input, then runs one small scenario (plain, gemv, small-mn, prepack-B, prepack-A, i8, batched). The main finder of arithmetic-overflow bugs | any panic = bug |
| `fuzz_api_validation` | adversarial dims (including `2^33` and `usize::MAX`) and `isize` strides (including `isize::MIN` and `isize::MAX`) into the **checked** `gemm`/`gemm_i8`/`gemm_cplx`/`gemm_batched`/`prepack_*` entries | a documented `gemmkit:`-prefixed panic is accepted, anything else is a bug |
| `fuzz_batched` | valid strided-batched `gemm_batched` (broadcast A/B, valid batch strides) plus `gemm_batched_slice`, gated element-wise against a differential reference | any panic = bug |
| `fuzz_prepack` | `prepack_rhs`->`gemm_packed_b` and `prepack_lhs`->`gemm_packed_a` round trips (f32, f64, bf16), gated at tolerance, not bit-for-bit, against the reference | any panic = bug |
| `fuzz_prepack_i8` | `prepack_rhs_i8`->`gemm_i8_packed_b` round trip (i8), gated bit-for-bit against the wrapping-i32 reference and a plain `gemm_i8` call | any panic = bug |

Every valid-input target also seeds C's backing buffer with a **canary
sentinel** in the non-view slots: interleave, pad, and inter-element. It
asserts those slots stay untouched after the call. This surfaces a stray
out-of-view write, even when ASan sees no boundary violation.

`fuzz_api_validation` runs under `catch_unwind`, with a silent panic hook. It
treats a `gemmkit:`-prefixed panic as an accepted rejection. On any other
panic, such as an index out-of-bounds or an arithmetic overflow, it calls
`abort()`, which counts as a real finding.

It skips only a plan that *would* fully pass validation and then do
unbounded work. That means a `WORK_CAP` of 2^24 MACs, a huge single
dimension, or a huge batch loop. Every rejection path stays fully fuzzed.

## Smoke (per target, ~45-60 s, CI-sized)

```sh
for t in fuzz_gemm fuzz_knobs fuzz_api_validation fuzz_batched fuzz_prepack fuzz_prepack_i8; do
  cargo +nightly fuzz run "$t" -- \
    -max_total_time=45 -max_len=512 -timeout=60 -malloc_limit_mb=1024 -print_final_stats=1
done
```

`-malloc_limit_mb=1024`: a single allocation over 1 GB on these tiny dims is
itself a knob-robustness bug. Treat such an artifact as a finding. Raise the
limit only if triage proves it benign.

`-timeout=60`: a validated but degenerate huge-dim or huge-batch input that
spins shows up as a timeout. Triage it as a finding.

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

cargo-fuzz creates the corpus in `corpus/<target>/` automatically. It grows
across runs, so no manual seed is needed. Each plan is `int_in_range`-driven,
so even an empty input decodes to a minimal valid plan.

### Coverage report over the accumulated corpus

`cargo fuzz coverage` builds with `--build-std=false` by default, since that
composes cleanly. It then renders an llvm-cov report, to find a dispatch
route or a scenario the plans never reach:

```sh
cargo +nightly fuzz coverage fuzz_gemm
# then render (the exact profdata/binary paths are printed by the command):
$(rustc +nightly --print target-libdir)/../bin/llvm-cov show \
  target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fuzz_gemm \
  -instr-profile=coverage/fuzz_gemm/coverage.profdata \
  -Xdemangler=rustfilt -format=html > /tmp/fuzz_gemm-cov.html
```

## Crash -> minimize -> Miri-replay -> stable regression test

When a target crashes, libFuzzer writes `artifacts/<target>/crash-<sha>`. It
might instead write a `timeout-...` or an `oom-...` file. It also prints the
plan's `Debug` output.

1. **Reproduce**
   ```sh
   cargo +nightly fuzz run <target> artifacts/<target>/crash-<sha>
   ```
2. **Minimize the input**
   ```sh
   cargo +nightly fuzz tmin <target> artifacts/<target>/crash-<sha>
   ```
3. **Decode to test parameters.** Every plan stores *resolved* values, from a
   manual `Arbitrary` implementation. Its `Debug` output is literally the
   dims, the strides layout, the alpha-beta indices, the knob array, and the
   parallelism value:
   ```sh
   cargo +nightly fuzz fmt <target> artifacts/<target>/<minimized>
   ```
4. **(Optional) Miri replay.** ASan misses an uninitialized read or a
   provenance bug that Miri catches. The gemmkit kernels stay
   Miri-compatible, through `cfg(miri)` paths in `gemmkit/src/scalar.rs` and
   in the correctness suite's shared oracle (`tests/oracle_common/mod.rs`).
   Translate the decoded plan into a tiny `#[test]`, and run it under Miri
   **on the stable workspace**. Never depend on this nightly-only fuzz crate
   for that:
   ```sh
   cargo +nightly miri test -p gemmkit --test <file> <testname>
   ```
5. **Hand-write a platform-independent stable regression** in
   `gemmkit/tests/`. Assert behavior, never a machine constant.
   - A knob-class crash goes to `tests/tuning.rs`. Hold `knob_guard()` and
     restore every knob touched (the `KNOB_LOCK` pattern).
   - An env-contract crash goes to a new one-test-per-binary file, following
     `tests/env.rs`.
   - A shape or validation crash goes to `tests/correctness/api.rs`. Use
     `#[should_panic]` when a validation gap gets promoted to a documented
     panic, following the precedent of `panic_extent_overflow_view`. Or add
     a new `tests/fuzz_regressions.rs`.
6. **Verify on stable:**
   ```sh
   cargo test -p gemmkit --all-features --test <file>
   ```
   Then confirm the fuzz target no longer crashes on the artifact.

## Work-cap policy (`fuzz_api_validation`)

The prepack entries skip only a plan whose pack would be *representable but
huge*: the element count fits `usize`, yet it exceeds `WORK_CAP`. Running
such a plan would OOM on otherwise correct behavior.

An empty operand still gets fuzzed, since prepack short-circuits on it. A
pack size that overflows `usize` also stays fuzzed. That overflow is a
documented `gemmkit: ... too large` reject. The regression tests for that
overflow class live in `gemmkit/tests/props_packed.rs` (`prepack_*`) and in
`gemmkit/tests/props_api.rs` (`mixed_huge_k_fails_closed`).
