[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-tune/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-tune/README.md)

# gemmkit-tune

[![crates.io](https://img.shields.io/crates/v/gemmkit-tune.svg)](https://crates.io/crates/gemmkit-tune)

Install-time autotuner for the [`gemmkit`](https://crates.io/crates/gemmkit) GEMM
engine. It runs on the machine that will run your gemmkit application, sweeps
gemmkit's runtime `GEMMKIT_*` tuning knobs by measuring a representative set of matrix
shapes for each, and writes a shell profile of `export GEMMKIT_*=...` lines. Source
that profile before launching your gemmkit binary to retune it for the host with no
recompile.

Run it on the deploy machine, not in a `build.rs`: the build host is usually not the
deploy host, and the knobs are calibrated against the CPU that executes the work.
gemmkit's compiled-in defaults were calibrated on one machine (a Ryzen 9950X), so on
that same machine a run re-discovers essentially the same values; the payoff is on a
different machine.

A fuller guide, including how the sweep works inside, lives in the
[gemmkit Guide](https://someb1oody.github.io/gemmkit/en/gemmkit-tune/Tuning_with_gemmkit-tune.html).

## Usage

Install the binary and run it on the target machine:

```sh
cargo install gemmkit-tune
gemmkit-tune
```

By default this writes a `gemmkit-tune.env` file in the current directory. Source it
in the shell that launches your application:

```sh
source gemmkit-tune.env
./your-gemmkit-app
```

The generated file is a list of `export GEMMKIT_*=<value>` lines with a header comment
recording the host and the worker count it was tuned for. gemmkit reads those variables
at runtime, so no rebuild is needed. The tool also prints a report to the terminal
showing, per knob, the swept default, the value it chose, and which knobs were skipped
and why.

Any `GEMMKIT_*` variables should be unset in the tuning shell: they influence the
measured baseline, and the tool warns when it finds any.

## Options

`gemmkit-tune` takes no positional arguments. The flags are:

- `--threads <n>`: tune for this worker count (default: available parallelism).
- `--time-budget <dur>`: cap the sweep and coarsen measurements to fit, given as `30s`,
  `2m`, `1h`, or a bare number of seconds.
- `--large-matrices <GiB>`: also probe the memory-heavy knobs (`GEMMKIT_K_STREAM_MAX`,
  `GEMMKIT_SHARED_LHS_MNK`), using up to the given GiB budget for the large gemv probe
  matrices. Off by default.
- `--out <path>`: output profile path (default: `gemmkit-tune.env`).
- `--dry-run`: print the report only, do not write a profile.
- `-h`, `--help`: print usage.

## Feature flags

None. `gemmkit-tune` is a binary crate with no Cargo features of its own; it builds
against `gemmkit` with the `complex`, `half`, and `int8` families enabled so every
element type can be probed.

## Related crates

- [`gemmkit`](https://crates.io/crates/gemmkit): the GEMM engine whose runtime knobs
  this tool tunes. The emitted profile also affects the zero-copy adapter crates
  ([`gemmkit-ndarray`](https://crates.io/crates/gemmkit-ndarray),
  [`gemmkit-nalgebra`](https://crates.io/crates/gemmkit-nalgebra),
  [`gemmkit-faer`](https://crates.io/crates/gemmkit-faer)), since they forward to the
  same engine.

## License

Licensed under either of
[MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) or
[Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE), at your
option.

Minimum supported Rust version: 1.89.
