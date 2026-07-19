# Tuning with gemmkit-tune

`gemmkit-tune` is a small command-line autotuner. You run it once on the machine that will run your gemmkit workload; it sweeps gemmkit's runtime `GEMMKIT_*` knobs, measures a representative set of matrix shapes for each, and writes a shell profile of `export GEMMKIT_*=...` lines. Source that file before you launch a gemmkit binary and the already-built binary is retuned for the host — no recompile, no code change, no dependency on how the binary was distributed.

## Why it exists

gemmkit's compiled-in defaults are not arbitrary. Every threshold in [`gemmkit::tuning`](https://docs.rs/gemmkit) was hand-calibrated against real measurements — but against measurements on one machine, a Ryzen 9950X (Zen5). The value that is optimal there encodes that machine's cache sizes, core count, and DRAM-to-compute ratio. A different CPU has a different L2, a different number of cores, a different memory bandwidth, and its crossovers land in different places: the `k` at which packing starts to pay, the problem size at which a shared pre-pass beats per-worker packing, the byte floor below which a bandwidth-bound gemv should stay single-threaded. `gemmkit-tune` re-discovers those crossovers on the silicon in front of it and pins them.

There is a useful corollary. Run the tool on a 9950X and it re-selects essentially the shipped defaults; the report says every knob kept its default and the overall speedup is about 1.0x. That is the correct outcome, not a disappointment — it validates the tool. The payoff is on a machine *unlike* the calibration box: a laptop, a cloud instance sharing a socket, a wide Graviton, an Apple part with a shared cluster-L2 and no L3. The further the deploy host is from a 9950X, the more there is to win.

The sweep contains no randomness. Given the same machine and the same flags it produces the same profile every time, so a profile is a reproducible artifact you can commit alongside a deployment.

## Why the deploy host, and never build.rs

The knobs are calibrated against the CPU that *executes* the work, so the tool has to run there. Two consequences follow.

First, do not run it in a `build.rs`. The build host is usually not the deploy host — you compile on a CI runner or a developer laptop and ship the binary to something else entirely — and a `build.rs` autotune would measure the builder, then bake the builder's crossovers into a binary that runs somewhere with a completely different cache hierarchy. Worse, a cross-compiled build cannot execute the target's code at all. The whole point is to measure the real silicon, which means running on it.

Second, because the knobs are plain runtime env vars read once per process at startup, you do not need to rebuild to apply a profile. The same shipped binary reads whatever `GEMMKIT_*` values are in its environment. Tune the host, source the profile, launch — the engine reconfigures itself.

## Install and a first run

Install the binary and run it on the target machine:

```sh
cargo install gemmkit-tune
gemmkit-tune
```

A full run takes a minute or two and prints a report as it goes. By default it writes `gemmkit-tune.env` in the current directory. Source that in the shell that launches your application:

```sh
source gemmkit-tune.env
./your-gemmkit-app
```

That is the whole workflow. Everything below is refinement: bounding the run, matching it to how you deploy, and knowing when to do it again.

## The flags

`gemmkit-tune` takes no positional arguments. All behavior is on five flags.

### `--threads <n>`

Tune for this worker count. Every parallel probe runs under `Parallelism::Rayon(n)`, so the scheduling knobs (the oversample factors, the auto worker-count ramp) are optimized for exactly that width. The default is the machine's available parallelism. It is capped at the machine width — you cannot tune for more workers than the box physically has, and the stamped worker count is always truthful.

The rule of thumb: run the sweep with the same worker count your application will actually use. If your app pins `Parallelism::Rayon(8)`, pass `--threads 8`; a profile tuned for 32 workers can pick a different scheduling grain than one tuned for 8, and the mismatch costs you.

### `--time-budget <dur>`

Cap the sweep and coarsen it to fit. Accepts `30s`, `2m`, `1h`, or a bare number of seconds. Under a budget the tool takes fewer timing repetitions per estimate (7 by default, 5 under 90s, 3 under 30s), and once the deadline passes it stops sweeping and lists the remaining knobs as skipped for "time budget exhausted". Use it when install time must be bounded; leave it off for the most reliable profile. If the budget is so small that not even one knob is measured, the report says so and tells you to raise it.

### `--large-matrices <GiB>`

Opt into the two memory-heavy probes, `GEMMKIT_K_STREAM_MAX` and `GEMMKIT_SHARED_LHS_MNK`, with the given GiB figure as the budget for the giant gemv matrices. These knobs only bite in an expensive regime — one needs a gemv output that spills the last-level cache (multi-gigabyte matrices), the other needs a very high-FLOP shape above the shared-pre-pass crossover — so they are off by default. If the budget you pass cannot hold the required probe, the tool skips that knob cleanly and prints the exact GiB figure to re-run with. The `GEMMKIT_K_STREAM_MAX` probe is 64-bit only; `GEMMKIT_SHARED_LHS_MNK` still sweeps on 32-bit. Start with `4` or `8` and follow the advice if it asks for more. See [Inside the Sweep](Inside_the_Sweep.md) for what these two probes actually do.

### `--out <path>`

Write the profile somewhere other than `./gemmkit-tune.env`.

### `--dry-run`

Run the full sweep and print the report, but write no profile. Good for previewing what a machine would choose before committing a file. `-h` / `--help` prints usage.

## Anatomy of the emitted profile

The file is a header comment followed by one `export` line per swept knob, followed by a footer listing what was not swept. It looks like this (values depend entirely on the host):

```sh
# gemmkit-tune profile — source before running a gemmkit app: `source <this file>`
# generated 2026-07-19 14:12:03 UTC by gemmkit-tune 0.1.0
# host: 16 logical cores; L1d 32 KiB, L2 1024 KiB, L3 32 MiB; page 4 KiB
# tuned for 16 worker(s)

export GEMMKIT_MC_REG_PANELS=8  # default (1.00x)
export GEMMKIT_LHS_PACK_THRESHOLD=256  # tuned (1.07x)
export GEMMKIT_THREAD_DIM_STRIDE=32  # tuned (1.03x)

# not swept on this host:
#   GEMMKIT_PARALLEL_THRESHOLD: serial/parallel break-even is strongly shape-dependent ...
#   GEMMKIT_DEEP_KC_BYTES: narrow-only (f16/bf16 deep-contraction twin); no narrow probe here ...
```

The header records the host it was tuned on: logical core count, the three cache levels, the page size, the worker count, and the tool version and UTC timestamp. That stamp is what tells you, months later, whether a profile still matches the box you are looking at. Each `export` line carries a trailing comment marking it `default` (the winner equalled the shipped default) or `tuned` (the winner moved), with the measured speedup over the default. A knob that kept its default is still written, so the profile is a complete, self-documenting record of the decision, not just the deltas.

The values are always raw integers — an "unbounded" winner is written as its numeric value, never a `MAX` alias — because gemmkit's env parser reads a plain decimal integer. A malformed `GEMMKIT_*` value is not fatal: gemmkit warns once on stderr and falls back to the compiled default, so a hand-edited typo degrades to the default rather than crashing.

## Deploying the profile

gemmkit reads each `GEMMKIT_*` variable once, on first access, then caches it for the life of the process. So the profile must be in the environment *before* the first GEMM call — which is exactly what sourcing before launch guarantees. There are three common shapes for that.

**Shell profile or launch script.** The direct case: `source gemmkit-tune.env` in the same shell that runs the binary, or in the service's launch script. Full shell semantics apply, so the file drops in unchanged.

**Container entrypoint.** Bake the profile into the image and source it in the entrypoint before `exec`-ing your app, so every container starts pre-tuned. Tune on a host matching the container's runtime hardware, not the image builder.

**systemd `EnvironmentFile`.** This works, with one caveat: systemd's `EnvironmentFile` parser wants bare `NAME=value` lines — it does not understand the `export` keyword or the trailing `# tuned (...)` comments. Convert the profile first, for example `grep '^export' gemmkit-tune.env | sed -e 's/^export //' -e 's/[[:space:]]*#.*$//' > gemmkit.env`, and point `EnvironmentFile=` at the result. The `#`-comment header is fine to leave in; only the assignment lines need the transform.

One precedence note: a `GEMMKIT_*` env var is overridden by a programmatic `tuning::set_*` call in the application. If your app tunes a knob in code, the profile will not change that knob (this is deliberate — self-tuning code wins over a deployment profile). Apps that want the profile to apply simply do not call the setters. See [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md) for the full precedence order.

## Run in a clean environment

Any `GEMMKIT_*` variable already set in the tuning shell skews the sweep, because gemmkit reads it while measuring the baseline. The tool neutralizes the knobs it sweeps and warns you about any `GEMMKIT_*` it finds set, but the reliable move is to tune from a shell with none of them present. Do not source a previous `gemmkit-tune.env` and then re-run the tool in the same shell — that is exactly the polluted baseline the warning is about.

## When to retune

Retune when the thing the profile was stamped for changes: a different deploy machine (different CPU, different cache sizes — the whole reason the tool exists), or a different worker count (a profile tuned for 8 workers is not the right one for 32). A gemmkit or gemmkit-tune version bump can also add knobs, so regenerate after upgrading. A profile that no longer matches its header stamp is a profile to throw away and regenerate.

To understand what the sweep is actually measuring, how it scores candidates, and how to sanity-check a profile against your own workload, read [Inside the Sweep](Inside_the_Sweep.md).
