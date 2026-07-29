# Inside the Sweep

[Tuning with gemmkit-tune](Tuning_with_gemmkit-tune.md) covers how to run the tool. This
page is the mechanism behind the sweep. It covers:

- what the sweep measures
- how it decides a winner
- why it is biased toward the shipped default
- how to check that a profile actually helps you

## One knob at a time

The sweep is a set of independent one-dimensional searches, not a joint optimization.
For each knob, every other knob stays at its default. The tool measures the knob's
candidate values back to back and chooses a winner. Then it restores the knob to its
default before it sweeps the next one. Each knob is therefore evaluated against an
otherwise-default engine.

This is a deliberate simplification. A full joint search over about 30 knobs is
combinatorially hopeless, and it would be dominated by noise. The crossovers these knobs
gate are, by design, individually meaningful, so a one-dimensional search suits them.

The cost is that the sweep does not explore cross-knob interactions. This is an
acceptable trade, because the defaults already sit at a good joint operating point. The
tool's job is only to move individual crossovers to where the host puts them.

The tool measures candidates in a fixed order with no randomness. The default value goes
first, since it is the tie-break incumbent, then each distinct extra candidate follows.
It rebuilds the buffers for every shape with the same seeds. An A/B comparison between 2
candidate values therefore sees byte-identical inputs, and any machine drift cancels out.

## The sweep table stays in lockstep with the engine

gemmkit enumerates its knobs in one place: `gemmkit::tuning::knob_env_names()`. This
machine-readable registry is the single source of truth for every `GEMMKIT_*` name. The
tuner classifies each knob as either TUNED, meaning it has a real sweep, or NEVER_TUNED,
with a reason. A test asserts that these 2 lists partition `knob_env_names()` exactly.
No knob is missing, and no entry is stale.

The practical guarantee is direct: a knob added to gemmkit cannot silently escape the
autotuner. The build fails until someone writes a sweep for it, or records why it is
deliberately left alone. So when you read the tool's list of swept knobs, you are
reading a list the compiler keeps honest against the engine.

## What is measured, and in what unit

Each candidate's score is a throughput. A GEMM, i8, or batched probe is scored in
GFLOP/s, computed as `2*m*k*n` per call, times the batch count for a batched probe. A
gemv probe is scored in GB/s instead, because a matrix-times-vector is bandwidth-bound,
and the bytes moved is the honest figure of merit there.

A single-shape estimate is deliberately robust. First, the tool warms up the probe
closure a few times. Then it auto-sizes an iteration count so one timed batch runs for
about 50 ms. It times several such batches and reports the *median* rate, along with the
observed min and max. The min and max are not cosmetic. They record the run-to-run
spread, and the winner logic uses that spread to stay honest under noise.

## Scoring: geometric mean over a probe-shape set

A knob is never judged on one shape. Each knob carries a small set of probe shapes. The
tool chooses these shapes so the knob actually binds and so its crossover is bracketed
on both sides. A candidate's score is then the **geometric mean** of its per-shape
median throughputs.

The geometric mean gives every shape equal weight, regardless of its absolute size. One
big shape therefore cannot flatter a value that only helps that shape. A winner has to
be a broad improvement across the whole set. The worst shape's spread carries through
the geomean, so the noise gate stays conservative across the whole set instead of
trusting only the calmest shape.

The probes are picked per knob to make the knob bind. A few examples:

| knob | probe family | why these shapes |
| --- | --- | --- |
| `MC_REG_PANELS` | square f32, 512 to 3072, parallel | the 3072 tier stresses A-macro-panel residency in L2 |
| `LHS_PACK_THRESHOLD` | col-major A, candidates 32..MAX | brackets both the aarch64 low-reuse plateau and the x86 default of 1024 |
| `SMALL_K_THRESHOLD` | skinny large-`m,n` small-`k`, e.g. 4096x16x4096 | `k` straddles the in-place / packed-driver crossover |
| `GEMV_PARALLEL_BYTES` | huge-`m` gemv, GB/s | spans the cache-resident / DRAM-bound byte floor |
| `GEMV_TIER_STEP`, `GEMV_THREAD_CAP` | gemv from about 2.4 to 134 MiB touched, GB/s | straddles a rung of the gemv worker ladder, since a probe set sitting entirely in one rung would score every candidate the same |
| `SEQ_INTERNAL_BYTES_PER_WORKER` (aarch64) | batched shapes giving 96/192/384/432 KiB per batch-worker | straddles the ~128 KiB default on both sides, a two-sided validator |
| `I8_VNNI_MIN_PAR_MNK` (x86) | square i8, 384/512/640 | brackets the VNNI / widen-fallback parallel crossover |

## The tie-break is default-biased and noise-aware

Picking the highest geomean would be wrong. On a noisy machine, a 1% edge is usually
luck. The winner logic instead starts at the default. It upgrades to a candidate only
when that candidate's geomean beats the current best by **more than the larger of the
2 candidates' measured spreads**. Run-to-run noise cannot clear that bar by
construction, so it can never rewrite a knob. An exact tie keeps the default.

There is a further margin for the "auto" knobs, whose default is `0`. These knobs
derive their value from the machine, such as LLC size, core count, or page size. A
fixed candidate must beat auto by an extra 5% beyond noise. These auto derivations
adapt to shapes the probe set does not cover. A fixed number that wins by a hair on the
probes is not worth trading that adaptivity for.

This default bias is the right call under noise. The default is a known-good,
deliberately chosen value, and the tool often runs unattended on a machine nobody is
watching. The asymmetric bar means the worst case is that the tool just reproduces the
defaults. It never regresses you into a measurement artifact.

The sweep also has no RNG anywhere, so a run is safe to trust. At worst it does nothing.
When it does move a knob, a real and repeatable improvement cleared the noise.

## How the time budget caps and coarsens the sweep

`--time-budget` acts in 2 ways.

First, it coarsens each estimate up front: 7 timing repetitions with no budget, 5 under
90 seconds, and 3 under 30 seconds. This trades a little measurement stability for
speed.

Second, it enforces a hard deadline. Before each knob, the tool checks the clock. Once
the deadline has passed, it stops starting new sweeps and records every remaining knob
as skipped for "time budget exhausted".

A tight budget therefore both blurs the measurements it does take and drops knobs off
the tail. With no budget, the sweep runs to completion at full repetitions.

## Which knobs are skipped, and why

Some knobs are never swept. The report and the profile footer say why for each:

- `PARALLEL_THRESHOLD`: the serial/parallel break-even is strongly shape-dependent. A
  single `m*n*k` scalar cannot fit every aspect ratio, so the tool keeps the calibrated
  cross-shape default instead of auto-fitting it. Contrast `GEMV_THRESHOLD`, which is a
  clean binary on/off decision and *is* swept.
- `DEEP_KC_BYTES`: this gates the f16/bf16 deep-contraction twin, and the tuner runs no
  narrow-type probe. Its auto default derives from L2, a machine property. Override it
  directly if you need to retune the narrow deep-`k` engage point.
- `PREFETCH_MIN_BYTES`: this gates the driver's C-tile prefetch. Its auto default
  derives from the detected LLC, a machine property, and probing the crossover would
  need a beyond-LLC working set on every candidate. Override it directly to retune the
  engage point (`usize::MAX` disables the prefetch, and `1` forces it on).

Other knobs are inert on the current target and get skipped for that reason.
`SEQ_INTERNAL_BYTES_PER_WORKER` is read only by the aarch64 batched-split planner. It is
swept there, and it is inert and skipped on x86. `I8_VNNI_MIN_PAR_MNK` gates the x86
VNNI small-parallel fallback, which no other target's i8 kernel has. `NC_NO_L3_PANELS`
is consulted only on a machine with no L3. It is swept there, and it is inert and
skipped on an L3 host.

The 2 heavy knobs are skipped unless you pass `--large-matrices`.

## What --large-matrices unlocks

2 knobs only matter in a regime that is expensive to reproduce, so they are opt-in
behind a memory budget.

`K_STREAM_MAX` caps how far the axpy-gemv output stays register-blocked. It only *wins*
once the output is clearly DRAM-bound. Its probe therefore fixes the output at about
twice the last-level cache. A 1x-LLC output sits on the cache boundary and measures
nothing decisive, so the probe avoids that size. The probe then sweeps `k` around the
calibrated ceiling.

That output size is fixed, not budget-scaled, so reaching it takes multi-gigabyte
matrices. If the budget you passed cannot hold the largest probe, the tool skips the
knob. It prints the GiB figure you need to re-run with, rounded up. On a 32-bit target,
it skips the knob outright, because the matrices do not fit the address space at all.

`SHARED_LHS_MNK` gates the shared-LHS pre-pass. This pre-pass removes redundant
per-worker A-packing, but it adds a fork-join barrier. So it only pays off above a
large `m*n*k` value (about 8e9 on x86). Its probes use tall, high-FLOP shapes above
that crossover.

The tool neutralizes both knobs during the ordinary sweeps, whether or not it is
sweeping them itself. This way a stale env value cannot skew a baseline that reads them.

## Reading the terminal report

The report opens with a one-row-per-knob summary table. Its columns are knob, unit,
shape count, default, winner, speedup, and a result column that reads `keeps default` or
`-> <value>`, with moved knobs highlighted.

Below the table, a candidate-detail block prints the full sweep landscape for each
knob. It lists every candidate's geomean median. A leading mark flags the default
value, and a separate mark flags the winner, so you can see how flat or sharp the
optimum was.

After that comes the skipped list with reasons, then a footer. The footer counts how
many knobs were swept, how many moved off default, and how many were skipped. On the
reference machine, the footer notes that all knobs kept their defaults, which is
expected, and the profile reproduces them.

## Sanity-checking a profile

The sweep measures synthetic, roughly-square probes. That is the right choice for
finding a machine's crossovers. Your workload has its own shapes, though, so confirm
the win transfers before you trust a profile in production.

There are 2 ways to check. The direct one: time your own application with and without
`gemmkit-tune.env` sourced, on the deploy host, and compare. The reproducible one: run
gemmkit's criterion benches, which cover 5 headline groups (`sgemm`, `dtypes`, `gemv`,
`prepacked`, `batched`) under a saved baseline:

```sh
cargo bench -p gemmkit -- --save-baseline stock
source gemmkit-tune.env
cargo bench -p gemmkit -- --baseline stock
```

If a knob moved and something you care about regressed, the profile is a plain text
file. Delete or comment out that one `export` line and keep the rest. The header stamp
and per-line `tuned`/`default` tags make it easy to see which line to touch.
