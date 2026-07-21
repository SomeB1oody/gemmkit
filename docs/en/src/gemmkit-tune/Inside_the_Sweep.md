# Inside the Sweep

[Tuning with gemmkit-tune](Tuning_with_gemmkit-tune.md) covers how to run the tool. This page is the mechanism: what the sweep measures, how it decides a winner, why it is biased toward the shipped default, and how to check that a profile actually helps you.

## One knob at a time

The sweep is a set of independent one-dimensional searches, not a joint optimization. For each knob, every other knob is held at its default, the knob's candidate values are measured back to back, a winner is chosen, and then the knob is restored to its default before the next one is swept. So each knob is evaluated against an otherwise-default engine. This is a deliberate simplification: a full joint search over two dozen knobs is combinatorially hopeless and would be dominated by noise, whereas the crossovers these knobs gate are, by design, individually meaningful. The cost is that cross-knob interactions are not explored, which is an acceptable trade because the defaults already sit at a good joint operating point and the tool's job is to move individual crossovers to where the host puts them.

Candidates are measured in a fixed, RNG-free order — the default first (it is the tie-break incumbent), then the distinct extra candidates. Buffers are rebuilt for every shape with identical seeds, so an A/B between two candidate values sees byte-identical inputs and any machine drift cancels.

## The sweep table stays in lockstep with the engine

gemmkit's knobs are enumerated in one place: `gemmkit::tuning::knob_env_names()`, a machine-readable registry that is the single source of truth for every `GEMMKIT_*` name. The tuner classifies every knob as either TUNED (it has a real sweep) or NEVER_TUNED (with a reason), and a test asserts that those two lists partition `knob_env_names()` exactly — no knob missing, no stale entry. The practical guarantee: a knob added to gemmkit cannot silently escape the autotuner. The build fails until someone either writes a sweep for it or records why it is deliberately left alone. When you read the tool's list of swept knobs, you are reading a list the compiler keeps honest against the engine.

## What is measured, and in what unit

Each candidate's score is a throughput. A GEMM or i8 or batched probe is scored in GFLOP/s (`2*m*k*n` per call, times the batch count for batched); a gemv probe is scored in GB/s, because a matrix-times-vector is bandwidth-bound and bytes moved is the honest figure of merit there.

A single-shape estimate is deliberately robust. The probe closure is warmed up a few times, then an iteration count is auto-sized so one timed batch runs for about 50 ms, then several batches are timed and the *median* rate is reported along with the observed min and max. The min/max are not cosmetic: they are the run-to-run spread, and the spread is what the winner logic uses to stay honest under noise.

## Scoring: geometric mean over a probe-shape set

A knob is never judged on one shape. Each knob carries a small set of probe shapes chosen so that the knob actually binds and so its crossover is bracketed on both sides, and a candidate's score is the **geometric mean** of its per-shape median throughputs. The geometric mean gives every shape equal weight regardless of absolute size, so one big shape cannot flatter a value that only helps that shape; a winner has to be a broad improvement across the set. The worst shape's spread is carried through the geomean, so the noise gate below stays conservative across the whole set rather than trusting the calmest shape.

The probes are picked per knob to make the knob bind. A few examples:

| knob | probe family | why these shapes |
| --- | --- | --- |
| `MC_REG_PANELS` | square f32, 512 to 3072, parallel | the 3072 tier stresses A-macro-panel residency in L2 |
| `LHS_PACK_THRESHOLD` | col-major A, candidates 32..MAX | brackets both the aarch64 low-reuse plateau and the x86 default of 1024 |
| `SMALL_K_THRESHOLD` | skinny large-`m,n` small-`k`, e.g. 4096x16x4096 | `k` straddles the in-place / packed-driver crossover |
| `GEMV_PARALLEL_BYTES` | huge-`m` gemv, GB/s | spans the LLC-resident / DRAM-bound byte floor |
| `SEQ_INTERNAL_BYTES_PER_WORKER` (aarch64) | batched shapes giving 96/192/384/432 KiB per batch-worker | straddles the ~128 KiB default on both sides — a two-sided validator |
| `I8_VNNI_MIN_PAR_MNK` (x86) | square i8, 384/512/640 | brackets the VNNI / widen-fallback parallel crossover |

## The tie-break is default-biased and noise-aware

Picking the highest geomean would be wrong, because a 1% edge on a noisy machine is usually luck. The winner logic instead starts at the default and upgrades to a candidate only when that candidate's geomean beats the current best by **more than the larger of the two candidates' measured spreads**. Run-to-run noise, by construction, cannot clear that bar, so it can never rewrite a knob; and an exact tie keeps the default. There is a further margin for the "auto" knobs whose default is `0` (derive the value from the machine — LLC size, core count, page size): a fixed candidate must beat auto by an extra 5% beyond noise. Those auto derivations adapt to shapes the probe set does not cover, so a fixed number that wins by a hair on the probes is not worth trading the adaptivity for.

This default-bias is the right call under noise for a plain reason: the default is a known-good, deliberately-chosen value, and the tool runs unattended on machines nobody is watching. The asymmetric bar means the worst case is that the tool reproduces the defaults — it never regresses you into a measurement artifact. Combined with the absence of any RNG, a run is safe to trust: at worst it does nothing, and when it moves a knob it is because a real, repeatable improvement cleared the noise.

## How the time budget caps and coarsens the sweep

`--time-budget` acts in two ways. It coarsens each estimate up front — 7 timing repetitions with no budget, 5 under 90 seconds, 3 under 30 — trading a little measurement stability for speed. And it enforces a hard deadline: before each knob the tool checks the clock, and once the deadline has passed it stops starting new sweeps and records every remaining knob as skipped for "time budget exhausted". So a tight budget both blurs the measurements it does take and drops knobs off the tail. With no budget the sweep runs to completion at full repetitions.

## Which knobs are skipped, and why

Some knobs are never swept, and the report and the profile footer say why for each:

- `PARALLEL_THRESHOLD` — the serial/parallel break-even is strongly shape-dependent; a single `m*n*k` scalar cannot fit every aspect ratio, so the calibrated cross-shape default is kept rather than auto-fit. (Contrast `GEMV_THRESHOLD`, which is a clean binary on/off decision and *is* swept.)
- `DEEP_KC_BYTES` — this gates the f16/bf16 deep-contraction twin, and the tuner runs no narrow-type probe; its auto default is derived from L2, a machine property. Override it directly if you need to retune the narrow deep-`k` engage point.
- `PREFETCH_MIN_BYTES` — this gates the driver's C-tile prefetch; its auto default is derived from the detected LLC, a machine property, and probing the crossover would need a beyond-LLC working set on every candidate. Override it directly (`usize::MAX` disables the prefetch, `1` forces it on) to retune the engage point.

Others are inert on the current target and skipped for that reason: `SEQ_INTERNAL_BYTES_PER_WORKER` is read only by the aarch64 batched-split planner (swept there, inert and skipped on x86); `I8_VNNI_MIN_PAR_MNK` gates the x86 VNNI small-parallel fallback that no other target's i8 kernel has; `NC_NO_L3_PANELS` is consulted only on a machine with no L3 (swept there, inert and skipped on an L3 host). The two heavy knobs are skipped unless `--large-matrices` is passed.

## What --large-matrices unlocks

Two knobs only matter in a regime that is expensive to reproduce, so they are opt-in behind a memory budget.

`K_STREAM_MAX` caps how far the axpy-gemv output stays register-blocked. It only *wins* once the output is clearly DRAM-bound, so its probe fixes the output at about twice the last-level cache — a 1x-LLC output sits on the cache boundary and measures nothing decisive — and sweeps `k` around the calibrated ceiling. That output is fixed, not budget-scaled, so reaching it takes multi-gigabyte matrices; if the budget you passed cannot hold the largest probe, the tool skips the knob and prints the GiB figure (rounded up) to re-run with, and on a 32-bit target it skips because the matrices do not fit the address space at all.

`SHARED_LHS_MNK` gates the shared-LHS pre-pass, which removes redundant per-worker A-packing but adds a fork-join barrier and so only pays above a large `m*n*k` (about 8e9 on x86). Its probes are tall high-FLOP shapes above that crossover. Both knobs are neutralized during the ordinary sweeps whether or not they are themselves swept, so a stale env value cannot skew a baseline that reads them.

## Reading the terminal report

The report opens with a one-row-per-knob summary table — knob, unit, shape count, default, winner, speedup, and a result column that reads `keeps default` or `→ <value>`, with moved knobs highlighted. Below it, a candidate-detail block prints the full sweep landscape for each knob: every candidate's geomean median, with `*` marking the default and `‹` marking the winner, so you can see how flat or sharp the optimum was. Then the skipped list with reasons, and a footer counting how many knobs were swept, how many moved off default, and how many were skipped. On the calibration machine the footer notes that all knobs kept their defaults — expected, and the profile reproduces them.

## Sanity-checking a profile

The sweep measures synthetic, roughly-square probes. That is the right choice for finding a machine's crossovers, but your workload has its own shapes, so confirm the win transfers before you trust a profile in production.

Two ways to check. The direct one: time your own application with and without `gemmkit-tune.env` sourced, on the deploy host, and compare. The reproducible one: run gemmkit's criterion benches, which cover five headline groups — `sgemm`, `dtypes`, `gemv`, `prepacked`, `batched` — under a saved baseline:

```sh
cargo bench -p gemmkit -- --save-baseline stock
source gemmkit-tune.env
cargo bench -p gemmkit -- --baseline stock
```

If a knob moved and something you care about regressed, the profile is a plain text file: delete or comment out that one `export` line and keep the rest. The header stamp and per-line `tuned`/`default` tags make it easy to see which line to touch.
