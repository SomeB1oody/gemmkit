//! `gemmkit-tune`: install-time autotuner for [`gemmkit`]
//!
//! Run this on the deploy machine, never from `build.rs`: the machine that builds a binary is
//! not necessarily the machine that runs it. It sweeps gemmkit's runtime knobs (the `set_*`/
//! `GEMMKIT_*` pairs in `gemmkit::tuning`) over a representative shape set for each, then writes
//! a `gemmkit-tune.env` profile of `export GEMMKIT_*=...` lines. `source` that file before
//! running a gemmkit application to retune the shipped binary for the host with no recompile
//!
//! ## How it sweeps
//!
//! Each knob is swept independently: every other knob stays at its default while this knob's
//! candidate values are measured back-to-back through its public setter, against freshly filled
//! but identically seeded buffers so machine drift and input data both cancel out of the
//! comparison. A candidate's score is the geometric mean throughput over the knob's probe-shape
//! set rather than any single shape's number, so a winner reflects a broad improvement, not one
//! shape's quirk. The tie-break is deliberately default-biased and noise-aware: a candidate
//! replaces the incumbent (which starts as the default) only when its geomean beats the
//! incumbent's by more than the larger of the 2 measured spreads, so run-to-run noise can never
//! flip a knob. There is no RNG anywhere, so the output is a deterministic function of
//! (machine, config)
//!
//! gemmkit's defaults were hand-calibrated on a Ryzen 9950X (Zen5), so a run on that same
//! machine re-derives essentially the same values (overall speedup around 1.0x): that is the
//! expected outcome and is what validates the tool, not a bug. The payoff is on a different
//! machine
//!
//! Knobs this probe cannot safely tune are skipped, each with a reason, in both the report and
//! the profile's footer comment: some are inert on this machine (aarch64-only, x86-only, or
//! no-L3-only knobs), and `PARALLEL_THRESHOLD` is skipped because its serial/parallel break-even
//! depends heavily on shape, not just size, so no single `m*n*k` scalar generalizes across
//! aspect ratios; the hand-calibrated cross-shape default is kept instead. `GEMV_THRESHOLD`, by
//! contrast, is a clean on/off decision and is swept

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use console::style;
use gemmkit::{MatMut, MatRef, Parallelism, gemm, tuning};
use indicatif::{ProgressBar, ProgressStyle};

// Timing harness: a trimmed copy of the private one in gemmkit's test suite
// (gemmkit/tests/perf/harness.rs)

/// Deterministic pseudo-random `f32` fill (xorshift) in roughly `[-0.5, 0.5)`, so probe buffers
/// are non-trivial and reproducible run to run
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// Deterministic pseudo-random `i8` fill (same xorshift as [`fill`], reduced into range) for the
/// integer probe. Only the x86 VNNI small-parallel fallback knob is swept, so this and
/// [`sweep_i8`] are x86-only
#[cfg(target_arch = "x86_64")]
fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i8) % 64 - 8
        })
        .collect()
}

/// One throughput measurement: the median plus min/max, so run-to-run spread stays visible
/// without ever being what decides a winner
#[derive(Clone, Copy)]
struct Stat {
    median: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn spread_pct(&self) -> f64 {
        100.0 * (self.max - self.min) / self.median.max(1e-9)
    }
}

/// Measurement effort: rep count and the per-batch timing target; `--time-budget` coarsens the
/// rep count
#[derive(Clone, Copy)]
struct Timing {
    reps: usize,
    batch_secs: f64,
}

/// Robust throughput estimate for one workload: warms up, auto-sizes a batch to hit
/// `t.batch_secs`, then times `t.reps` such batches and reports the median (with min/max) of
/// `to_rate` applied to each batch's average per-call seconds. Steadier than a single
/// fixed-iteration timing, which one slow tick can throw off
fn measure<F: FnMut(), R: Fn(f64) -> f64>(t: &Timing, mut f: F, to_rate: R) -> Stat {
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    f();
    let one = t0.elapsed().as_secs_f64().max(1e-9);
    let iters = ((t.batch_secs / one).ceil() as usize).clamp(1, 200_000);
    let mut g: Vec<f64> = Vec::with_capacity(t.reps);
    for _ in 0..t.reps {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        g.push(to_rate(secs));
    }
    g.sort_by(f64::total_cmp);
    Stat {
        median: g[t.reps / 2],
        min: g[0],
        max: g[t.reps - 1],
    }
}

/// Reduce several per-shape `Stat`s to one score: the geometric mean of the per-shape medians
/// (equal weight, so no single large shape dominates), with min/max reconstructed from the worst
/// per-shape spread so the returned `Stat`'s own spread carries that same worst-case noise into
/// the sweep's noise gate
fn geomean(stats: &[Stat]) -> Stat {
    let n = stats.len().max(1) as f64;
    let ln_sum: f64 = stats.iter().map(|s| s.median.max(1e-9).ln()).sum();
    let median = (ln_sum / n).exp();
    let spread = stats.iter().map(Stat::spread_pct).fold(0.0, f64::max);
    let half = spread / 200.0; // keeps the reconstructed spread_pct() == `spread`
    Stat {
        median,
        min: median * (1.0 - half),
        max: median * (1.0 + half),
    }
}

fn gflops(m: usize, k: usize, n: usize) -> impl Fn(f64) -> f64 {
    move |secs| 2.0 * m as f64 * k as f64 * n as f64 / secs / 1e9
}

fn gbps(bytes: usize) -> impl Fn(f64) -> f64 {
    move |secs| bytes as f64 / secs / 1e9
}

// Sweep engine

/// One candidate's measured result: the value tried and the geomean `Stat` scoring it over the
/// probe-shape set
struct Row {
    value: usize,
    stat: Stat,
}

/// One knob's completed sweep: every candidate tried, the chosen winner, and enough context
/// (env name, unit, probe-shape count) to render a report row and a profile line
struct KnobResult {
    env: &'static str,
    default: usize,
    winner: usize,
    unit: &'static str,
    shapes: usize,
    rows: Vec<Row>,
    baseline: f64, // the default value's own score, i.e. the sweep's baseline
    winner_median: f64,
}

impl KnobResult {
    fn speedup(&self) -> f64 {
        self.winner_median / self.baseline.max(1e-9)
    }
}

/// Sweep one knob: score the default first as the baseline incumbent, then each extra candidate
/// value, then pick the noise-aware, default-biased winner. Restores the knob to its default
/// before returning, so sweeps stay independent of each other. `probe` measures the workload
/// under whichever value was just `set` and returns one geomean `Stat`
fn sweep(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    unit: &'static str,
    shapes: usize,
    mut probe: impl FnMut() -> Stat,
) -> KnobResult {
    // candidate order: default (the incumbent) first, then each distinct extra, always the same
    // fixed order
    let mut cands = vec![default];
    for &c in extras {
        if !cands.contains(&c) {
            cands.push(c);
        }
    }

    let mut rows = Vec::with_capacity(cands.len());
    for &v in &cands {
        set(v);
        rows.push(Row {
            value: v,
            stat: probe(),
        });
    }
    set(default); // restore the default so the next sweep starts clean

    // Winner starts at the default; a candidate only takes over once it clears the incumbent by
    // more than the larger of the 2 measured spreads, so noise never decides and ties keep default
    // "0" always means auto (machine-derived) rather than a real value, so give it extra headroom:
    // a fixed override must clear a higher bar than replacing one literal value with another would
    let auto_margin = if default == 0 { 0.05 } else { 0.0 };
    let baseline = rows[0].stat.median;
    let mut best = 0usize;
    for i in 1..rows.len() {
        let noise = rows[best].stat.spread_pct().max(rows[i].stat.spread_pct()) / 100.0;
        if rows[i].stat.median > rows[best].stat.median * (1.0 + noise + auto_margin) {
            best = i;
        }
    }

    KnobResult {
        env,
        default,
        winner: rows[best].value,
        unit,
        shapes,
        baseline,
        winner_median: rows[best].stat.median,
        rows,
    }
}

// Per-shape-family sweep helpers: each shape gets its own freshly filled, identically seeded
// buffers, so every candidate value sees the same input data

/// GEMM (f32) sweep helper: for each `(m, k, n)` probe shape, fills fresh seeded operands (A
/// optionally row-major, B/C always column-major), runs `gemm` at `alpha = 1, beta = 0`, and
/// folds the per-shape GFLOP/s into one geomean `Stat` via [`sweep`]
#[allow(clippy::too_many_arguments)]
fn sweep_sgemm(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    t: &Timing,
    shapes: &[(usize, usize, usize)],
    par: Parallelism,
    row_major_a: bool,
) -> KnobResult {
    sweep(env, set, default, extras, "GFLOP/s", shapes.len(), || {
        let stats: Vec<Stat> = shapes
            .iter()
            .map(|&(m, k, n)| {
                let a = fill(m * k, 1);
                let b = fill(k * n, 2);
                let mut c = vec![0.0f32; m * n];
                let am = if row_major_a {
                    MatRef::from_row_major(&a, m, k)
                } else {
                    MatRef::from_col_major(&a, m, k)
                };
                measure(
                    t,
                    || {
                        gemm(
                            1.0,
                            am,
                            MatRef::from_col_major(&b, k, n),
                            0.0,
                            MatMut::from_col_major(&mut c, m, n),
                            par,
                        );
                    },
                    gflops(m, k, n),
                )
            })
            .collect();
        geomean(&stats)
    })
}

/// gemv (matrix-vector) sweep helper: for each `(m, k)` probe shape, fills fresh seeded
/// column-major operands and a length-`k` vector (`n = 1` routes `gemm` to the dedicated gemv
/// path), runs it at `alpha = 1, beta = 0`, and folds the per-shape GB/s (bytes of A, the
/// vector, and the output, over time) into one geomean `Stat` via [`sweep`]
fn sweep_gemv(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    t: &Timing,
    shapes: &[(usize, usize)],
    par: Parallelism,
) -> KnobResult {
    sweep(env, set, default, extras, "GB/s", shapes.len(), || {
        let stats: Vec<Stat> = shapes
            .iter()
            .map(|&(m, k)| {
                let a = fill(m * k, 1);
                let x = fill(k, 2);
                let mut y = vec![0.0f32; m];
                let bytes = (m * k + k + m) * 4;
                measure(
                    t,
                    || {
                        gemm(
                            1.0,
                            MatRef::from_col_major(&a, m, k),
                            MatRef::from_col_major(&x, k, 1),
                            0.0,
                            MatMut::from_col_major(&mut y, m, 1),
                            par,
                        );
                    },
                    gbps(bytes),
                )
            })
            .collect();
        geomean(&stats)
    })
}

#[cfg(target_arch = "x86_64")]
fn sweep_i8(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    t: &Timing,
    sizes: &[usize],
    par: Parallelism,
) -> KnobResult {
    sweep(env, set, default, extras, "GFLOP/s", sizes.len(), || {
        let stats: Vec<Stat> = sizes
            .iter()
            .map(|&s| {
                let a = fill_i8(s * s, 1);
                let b = fill_i8(s * s, 2);
                let mut c = vec![0i32; s * s];
                measure(
                    t,
                    || {
                        gemmkit::gemm_i8(
                            1,
                            MatRef::from_col_major(&a, s, s),
                            MatRef::from_col_major(&b, s, s),
                            0,
                            MatMut::from_col_major(&mut c, s, s),
                            par,
                        );
                    },
                    gflops(s, s, s),
                )
            })
            .collect();
        geomean(&stats)
    })
}

/// Batched-GEMM sweep helper (mirrors `sweep_sgemm`): for each `(batch, m, k, n)` probe shape,
/// fills fresh seeded column-major operands and runs `gemm_batched`, folding whole-batch GFLOP/s
/// (`batch * 2*m*k*n` work) into one geomean `Stat` via [`sweep`]. aarch64-only because the only
/// knob it backs, `SEQ_INTERNAL_BYTES_PER_WORKER`, is read solely by the aarch64 `resolve_batch`
#[cfg(target_arch = "aarch64")]
fn sweep_batched(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    t: &Timing,
    shapes: &[(usize, usize, usize, usize)],
    par: Parallelism,
) -> KnobResult {
    sweep(env, set, default, extras, "GFLOP/s", shapes.len(), || {
        let stats: Vec<Stat> = shapes
            .iter()
            .map(|&(batch, m, k, n)| {
                // each batch element is one contiguous col-major m x k block; batch stride = element size
                let a = fill(batch * m * k, 1);
                let b = fill(batch * k * n, 2);
                let mut c = vec![0.0f32; batch * m * n];
                measure(
                    t,
                    || {
                        gemmkit::gemm_batched(
                            batch,
                            1.0,
                            MatRef::new(&a, m, k, 1, m as isize),
                            (m * k) as isize,
                            MatRef::new(&b, k, n, 1, k as isize),
                            (k * n) as isize,
                            0.0,
                            MatMut::new(&mut c, m, n, 1, m as isize),
                            (m * n) as isize,
                            par,
                        );
                    },
                    gflops(batch * m, k, n),
                )
            })
            .collect();
        geomean(&stats)
    })
}

// CLI

struct Cli {
    threads: Option<usize>,
    budget_secs: Option<f64>,
    out: String,
    dry_run: bool,
    /// `Some(gib)` after `--large-matrices <GiB>`: opts into the heavy, memory-/FLOP-intensive
    /// probes (`K_STREAM_MAX`, `SHARED_LHS_MNK`), with `gib` GiB as the budget for the giant
    /// probe matrices
    large_gib: Option<f64>,
}

/// Parse a `--time-budget` value like `30s`, `2m`, `1h`, or a bare number of seconds. Returns
/// `None` for anything that is not finite, non-negative, and within a sane range, so the caller
/// can reject a negative, NaN, infinite, or overflowing input cleanly instead of letting it panic
/// inside `Duration::from_secs_f64`
fn parse_duration(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num, mult) = match s.as_bytes().last()? {
        b's' => (&s[..s.len() - 1], 1.0),
        b'm' => (&s[..s.len() - 1], 60.0),
        b'h' => (&s[..s.len() - 1], 3600.0),
        _ => (s, 1.0),
    };
    let v = num.trim().parse::<f64>().ok()? * mult;
    (v.is_finite() && (0.0..1e12).contains(&v)).then_some(v)
}

/// Print a usage error to stderr, show usage, and exit(2). Every malformed or missing CLI value
/// routes through this, so a bad flag is reported clearly instead of silently ignored
fn die(msg: &str) -> ! {
    eprintln!("gemmkit-tune: {msg}\n");
    print_usage();
    std::process::exit(2);
}

/// The value token that must follow `flag`, or a clean usage error via [`die`]
fn take(next: Option<String>, flag: &str) -> String {
    next.unwrap_or_else(|| die(&format!("{flag} requires a value")))
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        threads: None,
        budget_secs: None,
        out: "gemmkit-tune.env".to_string(),
        dry_run: false,
        large_gib: None,
    };
    // args_os + lossy, not args: a non-UTF-8 argument must produce a clean usage error, not a
    // panic; a mangled arg just fails to match any known flag and falls through to `die`
    let mut args = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => {
                let v = take(args.next(), "--threads");
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => cli.threads = Some(n),
                    _ => die(&format!("--threads expects a positive integer, got {v:?}")),
                }
            }
            "--time-budget" => {
                let v = take(args.next(), "--time-budget");
                match parse_duration(&v) {
                    Some(secs) => cli.budget_secs = Some(secs),
                    None => die(&format!(
                        "--time-budget expects a non-negative duration like 30s/2m/1h, got {v:?}"
                    )),
                }
            }
            "--out" => cli.out = take(args.next(), "--out"),
            "--dry-run" => cli.dry_run = true,
            "--large-matrices" => {
                let v = take(args.next(), "--large-matrices");
                // must be positive, finite, and within a sane range: rejects 0, NaN, inf,
                // negative, and absurdly large input so a giant probe matrix is never sized from
                // garbage
                match v.parse::<f64>() {
                    Ok(g) if g.is_finite() && (0.0..=4096.0).contains(&g) && g > 0.0 => {
                        cli.large_gib = Some(g)
                    }
                    _ => die(&format!(
                        "--large-matrices expects a positive GiB budget like 4 or 8, got {v:?}"
                    )),
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => die(&format!("unknown argument {other:?}")),
        }
    }
    cli
}

fn print_usage() {
    println!(
        "gemmkit-tune — sweep gemmkit's runtime knobs on this machine and emit a GEMMKIT_* profile\n\n\
         USAGE:\n    gemmkit-tune [OPTIONS]\n\n\
         OPTIONS:\n    \
         --threads <n>        Tune for this worker count (default: available parallelism)\n    \
         --time-budget <dur>  Cap the sweep; coarsens to fit (e.g. 30s, 2m, 1h)\n    \
         --large-matrices <G> Also probe the heavy knobs (K_STREAM_MAX, SHARED_LHS_MNK) using up\n    \
         \x20                    to <G> GiB for the giant gemv matrices (off by default; needs GiB)\n    \
         --out <path>         Output profile path (default: gemmkit-tune.env)\n    \
         --dry-run            Print the report only; do not write the profile\n    \
         -h, --help           Show this help\n\n\
         Then, before running your gemmkit app:  source gemmkit-tune.env"
    );
}

// Host stamp

/// UTC calendar date (year, month, day) from a Unix timestamp, via Howard Hinnant's
/// `civil_from_days` algorithm
fn ymd_from_unix(secs: u64) -> (i64, u32, u32) {
    let days = (secs / 86400) as i64 + 719468;
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + i64::from(m <= 2), m, d)
}

fn host_stamp(threads: usize) -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let topo = gemmkit::topology();
    let kib = |b: usize| b / 1024;
    let l3 = match topo.l3 {
        Some(l) => format!("{} MiB", l.bytes / (1024 * 1024)),
        None => "none".to_string(),
    };
    let page = gemmkit::Machine::current().page_size;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, da) = ymd_from_unix(now);
    let (hh, mm, ss) = {
        let t = now % 86400;
        (t / 3600, (t % 3600) / 60, t % 60)
    };
    format!(
        "# gemmkit-tune profile — source before running a gemmkit app: `source <this file>`\n\
         # generated {y:04}-{mo:02}-{da:02} {hh:02}:{mm:02}:{ss:02} UTC by gemmkit-tune {ver}\n\
         # host: {cores} logical cores; L1d {l1} KiB, L2 {l2} KiB, L3 {l3}; page {pg} KiB\n\
         # tuned for {threads} worker(s)\n",
        ver = env!("CARGO_PKG_VERSION"),
        l1 = kib(topo.l1d.bytes),
        l2 = kib(topo.l2.bytes),
        pg = kib(page),
    )
}

// main

const MAX: usize = usize::MAX;

/// Build the probe plan for the opt-in `K_STREAM_MAX` sweep. The gemv register-block gate engages
/// once the output crosses a last-level-cache-derived threshold (see `gemv_regblock_engage_bytes`
/// in gemmkit), but only wins once the output is clearly DRAM-bound, so the probe fixes the
/// output size at about 2x the LLC: an output right at the LLC boundary would sit on the cache
/// edge and measure nothing decisive. Returns the shared row count and the probe `k` values, or a
/// reason string when the probe cannot fit this target's address space, or when `budget_bytes`
/// cannot hold the whole probe at the largest `k`. That reason rounds the advised budget up to
/// the printed precision, so re-running with it actually clears the gate
fn plan_k_stream(budget_bytes: u64, gib: f64) -> Result<(usize, Vec<usize>), String> {
    const MAX_PROBE_K: usize = 48; // top candidate; with the k=24 probe, brackets {16, 32, 48}
    let topo = gemmkit::topology();
    let llc = topo.l3.map(|l| l.bytes).unwrap_or(topo.l2.bytes).max(1);
    // about 2x the LLC so the plain form's per-column output re-reads clearly spill to DRAM
    // (where register-blocking pays); a fixed target, not budget-scaled, since a smaller output
    // would just run cache-resident and measure noise instead of the DRAM-bound regime
    let out = llc.saturating_mul(2);
    // peak live bytes of the largest (k = MAX_PROBE_K) col-major f32 probe, matching sweep_gemv's
    // own allocation: matrix a (rows*k*4 = out*k) + output y (rows*4 = out) + input x (k*4)
    // computed in u64 so a 32-bit usize can't wrap `need` (or the budget check) and silently pass
    let need = (out as u64)
        .saturating_mul(MAX_PROBE_K as u64)
        .saturating_add(out as u64)
        .saturating_add(MAX_PROBE_K as u64 * 4);
    // the probe has to fit this target's address space at all: on 32-bit, a multi-GB matrix can't
    // be allocated, so skip cleanly here instead of letting the Vec allocation abort the process
    if need > usize::MAX as u64 {
        return Err(format!(
            "a DRAM-bound ~2x-LLC ({} MiB) gemv probe needs multi-GB matrices that do not fit this \
             target's address space — --large-matrices is only usable on 64-bit",
            out / (1 << 20),
        ));
    }
    if budget_bytes < need {
        // round the advised budget UP to the printed 0.1 GiB, so following it actually clears
        // `budget_bytes < need` (a truncated "1.5 GiB" that is really 1.53 would loop forever)
        let need_gib = (need as f64 / (1u64 << 30) as f64 * 10.0).ceil() / 10.0;
        return Err(format!(
            "--large-matrices {gib} GiB cannot hold a DRAM-bound ~2x-LLC ({} MiB) gemv probe at \
             k={} (need >= {:.1} GiB); the K_STREAM_MAX cap only bites on a DRAM-bound output, so \
             a smaller probe would measure nothing",
            out / (1 << 20),
            MAX_PROBE_K,
            need_gib,
        ));
    }
    let rows = (out / 4).max(1); // out is bytes; f32 elements = out/4
    Ok((rows, vec![24, MAX_PROBE_K]))
}

fn main() {
    let cli = parse_cli();
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // `--threads` is already validated >= 1 in parse_cli, but clamp to `avail` too: gemmkit caps
    // Rayon(n) internally to the machine width anyway, so this just keeps the reported and
    // stamped worker count truthful instead of echoing an unusable request
    let threads = cli.threads.unwrap_or(avail).min(avail).max(1);
    let par = Parallelism::Rayon(threads);
    let ser = Parallelism::Serial;

    // fewer reps under a tight time budget; the hard deadline below (once set) skips whole knobs
    // near the end instead
    let reps = match cli.budget_secs {
        Some(b) if b < 30.0 => 3,
        Some(b) if b < 90.0 => 5,
        _ => 7,
    };
    let timing = Timing {
        reps,
        batch_secs: 0.05,
    };
    let start = Instant::now();
    let deadline = cli.budget_secs.map(|b| start + Duration::from_secs_f64(b));

    eprintln!("gemmkit-tune: tuning for {threads} worker(s); this takes a minute or two...");

    // a GEMMKIT_* var left set in this shell can skew a sweep's baseline: gemmkit reads env vars
    // for any knob it has not been explicitly `set()` for, and NEUTRALIZE below does not reach
    // every knob a sweep might consult, so warn rather than silently mis-tune. A clean
    // environment gives the most faithful profile
    // vars_os, not vars: an unrelated non-UTF-8 environment variable must not panic the tool
    let dirty: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| k.starts_with("GEMMKIT_"))
        .collect();
    if !dirty.is_empty() {
        eprintln!(
            "gemmkit-tune: warning: GEMMKIT_* variables are set in this environment ({}); they \
             influence the baseline and may not all be written to the profile — run in a clean \
             environment for a faithful profile.",
            dirty.join(", ")
        );
    }

    // reset every knob in NEUTRALIZE to its compiled default via `set()` (never the env-consulting
    // getter), so a GEMMKIT_* value the tuning shell happens to carry cannot leak into a baseline
    for &(set, def) in NEUTRALIZE {
        set(def);
    }

    let mut results: Vec<KnobResult> = Vec::new();
    let mut budget_skipped: Vec<&str> = Vec::new();
    let mut large_skipped: Vec<(&'static str, String)> = Vec::new();

    let sweep_total: u64 = {
        let mut n: u64 = 16;
        n += u64::from(gemmkit::topology().l3.is_none());
        n += u64::from(cfg!(target_arch = "x86_64"));
        n += u64::from(cfg!(target_arch = "aarch64"));
        if let Some(g) = cli.large_gib {
            n += 1; // SHARED_LHS_MNK sweeps unconditionally once --large-matrices is given
            n += u64::from(plan_k_stream((g * (1u64 << 30) as f64) as u64, g).is_ok());
        }
        n
    };
    let bar = ProgressBar::new(sweep_total);
    bar.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} [{elapsed_precise}] [{bar:28.cyan/blue}] {pos:>2}/{len:<2}  {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    bar.enable_steady_tick(Duration::from_millis(120));

    macro_rules! knob {
        ($env:literal, $body:expr) => {{
            bar.set_message($env.strip_prefix("GEMMKIT_").unwrap_or($env));
            if deadline.is_some_and(|d| Instant::now() >= d) {
                budget_skipped.push($env);
            } else {
                results.push($body);
            }
            bar.inc(1);
        }};
    }

    // Blocking knobs for the register-tiling driver (MC, KC, tiny-shortcut, pack-transpose tile)
    knob!(
        "GEMMKIT_MC_REG_PANELS",
        sweep_sgemm(
            "GEMMKIT_MC_REG_PANELS",
            tuning::set_mc_reg_panels,
            tuning::MC_REG_PANELS_DEFAULT,
            // candidates run up to 32: a large shared L2 can keep a taller A macro-panel resident,
            // so the optimum may sit above a lower ceiling; the 3072-shape probe stresses that case
            &[4, 6, 12, 16, 24, 32],
            &timing,
            &[
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (3072, 3072, 3072),
            ],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_KC_MIN",
        sweep_sgemm(
            "GEMMKIT_KC_MIN",
            tuning::set_kc_min,
            tuning::KC_MIN_DEFAULT,
            &[256, 384, 768, 1024],
            &timing,
            &[(768, 768, 768), (1024, 1024, 1024), (1536, 1536, 1536)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_TINY_BLOCK_DIM",
        sweep_sgemm(
            "GEMMKIT_TINY_BLOCK_DIM",
            tuning::set_tiny_block_dim,
            tuning::TINY_BLOCK_DIM_DEFAULT,
            &[32, 48, 96, 128],
            &timing,
            &[(48, 512, 48), (64, 512, 64), (96, 512, 96)],
            ser,
            false,
        )
    );
    knob!(
        "GEMMKIT_KC",
        sweep_sgemm(
            "GEMMKIT_KC",
            tuning::set_kc,
            tuning::KC_DEFAULT,
            &[128, 256, 1024],
            &timing,
            &[(48, 1024, 48), (40, 2048, 40)],
            ser,
            false,
        )
    );
    knob!(
        "GEMMKIT_PACK_TRANSPOSE_TILE",
        sweep_sgemm(
            "GEMMKIT_PACK_TRANSPOSE_TILE",
            tuning::set_pack_transpose_tile,
            tuning::PACK_TRANSPOSE_TILE_DEFAULT,
            &[8, 32, 64],
            &timing,
            &[(1024, 512, 256), (768, 512, 512)],
            par,
            true, // row-major A takes the cache-blocked transpose packer, exercising this knob
        )
    );

    // No-L3 column-block cap: only consulted when the machine reports no L3, so it is dead (and
    // stays in `skipped` below) on an L3 host. NC = min(this * NR, N), so N must clear
    // default * NR (2048 f32 columns) for a candidate value to actually bind
    if gemmkit::topology().l3.is_none() {
        knob!(
            "GEMMKIT_NC_NO_L3_PANELS",
            sweep_sgemm(
                "GEMMKIT_NC_NO_L3_PANELS",
                tuning::set_nc_no_l3_panels,
                tuning::NC_NO_L3_PANELS_DEFAULT,
                &[256, 1024, 2048, MAX],
                &timing,
                &[(2048, 1024, 16384), (1024, 2048, 16384)],
                par,
                false,
            )
        );
    }

    // Parallel scheduling knobs
    knob!(
        "GEMMKIT_PARALLEL_OVERSAMPLE",
        sweep_sgemm(
            "GEMMKIT_PARALLEL_OVERSAMPLE",
            tuning::set_parallel_oversample,
            tuning::PARALLEL_OVERSAMPLE_DEFAULT,
            &[2, 4, 16, 32],
            &timing,
            &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_PACKED_OVERSAMPLE",
        sweep_sgemm(
            "GEMMKIT_PACKED_OVERSAMPLE",
            tuning::set_packed_oversample,
            tuning::PACKED_OVERSAMPLE_DEFAULT,
            &[1, 4, 8],
            &timing,
            &[(2048, 256, 256), (4096, 128, 512)],
            par,
            true, // row-major A forces the packed-LHS path, exercising this knob
        )
    );
    knob!(
        "GEMMKIT_PAR_MNK_PER_WORKER",
        sweep_sgemm(
            "GEMMKIT_PAR_MNK_PER_WORKER",
            tuning::set_par_mnk_per_worker,
            tuning::PAR_MNK_PER_WORKER_DEFAULT,
            &[500_000, 1_000_000, 4_000_000, 8_000_000],
            &timing,
            &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
            par,
            false,
        )
    );
    // Exact-fit size-class pool tiers: 0 (disabled), 1 (half width only), or 3 (adds eighth
    // width). The probes span tier-8 territory (128^3), the 8/16 crossover (256^3), and the
    // 16-vs-full-width regime (384^3)
    knob!(
        "GEMMKIT_POOL_CLASSES",
        sweep_sgemm(
            "GEMMKIT_POOL_CLASSES",
            tuning::set_pool_classes,
            tuning::POOL_CLASSES_DEFAULT,
            &[0, 1, 3],
            &timing,
            &[(128, 128, 128), (256, 256, 256), (384, 384, 384)],
            par,
            false,
        )
    );
    // Full-machine-width work gate: the m*n*k above which auto leaves the largest pool tier for
    // the full machine width. MAX means never leave that tier. The probes bracket the measured
    // 448^3/512^3 crossover, plus a larger 640^3 that full width should clearly win
    knob!(
        "GEMMKIT_FULL_WIDTH_MNK",
        sweep_sgemm(
            "GEMMKIT_FULL_WIDTH_MNK",
            tuning::set_full_width_mnk,
            tuning::FULL_WIDTH_MNK_DEFAULT,
            &[64_000_000, 134_000_000, MAX],
            &timing,
            &[(448, 448, 448), (512, 512, 512), (640, 640, 640)],
            par,
            false,
        )
    );

    // Packing gates: pack/no-pack route crossovers, each probed near its crossover point
    knob!(
        "GEMMKIT_RHS_PACK_THRESHOLD",
        sweep_sgemm(
            "GEMMKIT_RHS_PACK_THRESHOLD",
            tuning::set_rhs_pack_threshold,
            tuning::RHS_PACK_THRESHOLD_DEFAULT,
            &[512, 1024, 4096, MAX],
            &timing,
            &[(2048, 256, 256), (3072, 192, 256)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_LHS_PACK_THRESHOLD",
        sweep_sgemm(
            "GEMMKIT_LHS_PACK_THRESHOLD",
            tuning::set_lhs_pack_threshold,
            tuning::LHS_PACK_THRESHOLD_DEFAULT,
            // candidates span 32..2048 to bracket both arch optima: aarch64's cheap-packing win is
            // a flat plateau from 32 up through 256 then a drop-off (needs the low candidates to
            // find the plateau top), while x86's default of 1024 needs 2048 above it to confirm
            // there is no further gain
            &[32, 64, 128, 256, 512, 1024, 2048, MAX],
            &timing,
            &[(1024, 512, 512), (2048, 256, 256)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_LHS_PACK_STRIDE",
        sweep_sgemm(
            "GEMMKIT_LHS_PACK_STRIDE",
            tuning::set_lhs_pack_stride,
            tuning::LHS_PACK_STRIDE_DEFAULT, // 0 = auto (page-derived)
            &[2048, 4096, 8192, MAX],
            &timing,
            // Square col-major shapes: the reuse floor now vetoes the force-pack on the
            // tall/skinny shapes stride was first probed on (too few column tiles), so every
            // stride candidate measured identical there. Probe the same square trio the span
            // sweep uses, where the stride gate still governs the pack decision
            &[(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_LHS_PACK_SPAN",
        sweep_sgemm(
            "GEMMKIT_LHS_PACK_SPAN",
            tuning::set_lhs_pack_span,
            tuning::LHS_PACK_SPAN_DEFAULT, // 0 = auto (4 MiB)
            &[1 << 20, 2 << 20, 8 << 20, MAX],
            &timing,
            // Square col-major shapes bracketing the in-place/pack crossover the
            // span gate controls (the 9950X measured it between the 2 MiB walk of
            // n = 1024 and the 4 MiB walk of n = 2048)
            &[(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_LHS_PACK_REUSE",
        sweep_sgemm(
            "GEMMKIT_LHS_PACK_REUSE",
            tuning::set_lhs_pack_reuse,
            tuning::LHS_PACK_REUSE_DEFAULT,
            &[0, 64, 256, MAX],
            &timing,
            // Tall/skinny shapes where the floor keeps A in place, plus a deep-k
            // square that still wants the pack: both sides of the n_nt crossover
            &[
                (4096, 512, 512),
                (4096, 512, 1024),
                (8192, 512, 2048),
                (2048, 2048, 2048),
            ],
            par,
            false,
        )
    );

    // Route thresholds: small-k and small-mn shortcut gates
    knob!(
        "GEMMKIT_SMALL_K_THRESHOLD",
        sweep_sgemm(
            "GEMMKIT_SMALL_K_THRESHOLD",
            tuning::set_small_k_threshold,
            tuning::SMALL_K_THRESHOLD_DEFAULT,
            &[4, 8, 24, 32],
            &timing,
            &[(4096, 16, 4096), (8192, 16, 2048)],
            par,
            false,
        )
    );
    knob!(
        "GEMMKIT_SMALL_MN_DIM",
        sweep_sgemm(
            "GEMMKIT_SMALL_MN_DIM",
            tuning::set_small_mn_dim,
            tuning::SMALL_MN_DIM_DEFAULT,
            &[8, 24, 32],
            &timing,
            &[(16, 4096, 16), (12, 2048, 12)],
            par,
            false,
        )
    );
    // Small-m,n horizontal PACK-tier k gate. Probed on shapes ineligible for the zero-copy tier
    // (col-major A, so A is strided along k), which is what routes them through this gate instead
    // of the driver. `MAX` disables the tier (falls back to the driver), so raising the gate only
    // helps if packing ever loses; it wins at every measured k here, so the default (fire right
    // from the small-k boundary) holds
    knob!(
        "GEMMKIT_SMALL_MN_PACK_MIN_K",
        sweep_sgemm(
            "GEMMKIT_SMALL_MN_PACK_MIN_K",
            tuning::set_small_mn_pack_min_k,
            tuning::SMALL_MN_PACK_MIN_K_DEFAULT,
            &[64, 256, MAX],
            &timing,
            &[(16, 4096, 16), (8, 8192, 8)],
            par,
            false,
        )
    );

    // Bandwidth-bound gemv knobs
    // GEMMKIT_K_STREAM_MAX is a heavy opt-in knob swept only under --large-matrices; see the
    // "Large-matrix probes" block below
    knob!(
        "GEMMKIT_GEMV_THREAD_CAP",
        sweep_gemv(
            "GEMMKIT_GEMV_THREAD_CAP",
            tuning::set_gemv_thread_cap,
            tuning::GEMV_THREAD_CAP_DEFAULT, // 0 = auto
            &[2, 4, 8, 16],
            &timing,
            &[(1 << 20, 32), (1 << 21, 16)],
            par,
        )
    );
    knob!(
        "GEMMKIT_GEMV_PARALLEL_BYTES",
        sweep_gemv(
            "GEMMKIT_GEMV_PARALLEL_BYTES",
            tuning::set_gemv_parallel_bytes,
            tuning::GEMV_PARALLEL_BYTES_DEFAULT, // 0 = auto (LLC-derived)
            &[1 << 20, 1 << 22, 1 << 24],
            &timing,
            &[(1 << 16, 8), (1 << 19, 4), (1 << 22, 8)],
            par,
        )
    );

    // Integer (i8) knobs
    // gates the VNNI -> widen small-parallel fallback, which exists only for the x86 VNNI i8
    // kernel (`small_par_fallback` is `None` for every other kernel, so elsewhere the knob is read
    // but never acts); inert on a non-x86 target, so sweeping it there would just measure noise
    #[cfg(target_arch = "x86_64")]
    knob!(
        "GEMMKIT_I8_VNNI_MIN_PAR_MNK",
        sweep_i8(
            "GEMMKIT_I8_VNNI_MIN_PAR_MNK",
            tuning::set_i8_vnni_min_par_mnk,
            tuning::I8_VNNI_MIN_PAR_MNK_DEFAULT,
            &[0, 256 * 256 * 256, 512 * 512 * 512, 1024 * 1024 * 1024, MAX],
            &timing,
            &[384, 512, 640],
            par,
        )
    );

    // aarch64 batched-GEMM split crossover: SequentialInternal (split each element across the
    // machine in turn) vs BatchParallel (run the `batch` elements 1-per-worker) when `batch <
    // workers`. Only the aarch64 `resolve_batch` reads this knob (x86 uses an L2-residency test
    // instead), so it is skipped on x86. The probe shapes give per-batch-worker shares
    // (`elem_bytes / batch`) of 96/192/384/432 KiB, straddling the 128 KiB default on both sides,
    // so the sweep is a 2-sided validator: a lower candidate (64 KiB) would wrongly split the
    // 96 KiB shape (256^3 batch 8, where 1-per-worker is faster), and a higher one (320 KiB)
    // would wrongly serialize the 192 KiB shape (256^3 batch 4, where splitting is faster)
    #[cfg(target_arch = "aarch64")]
    knob!(
        "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
        sweep_batched(
            "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
            tuning::set_seq_internal_bytes_per_worker,
            tuning::SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT,
            &[64 * 1024, 320 * 1024, MAX],
            &timing,
            &[
                (8, 512, 512, 512),
                (4, 256, 256, 256),
                (4, 384, 384, 384),
                (8, 256, 256, 256), // share 96 KiB: the sub-default point bracketing 128 KiB from below
            ],
            par,
        )
    );

    // gemv path vs general driver: a binary on/off decision. For a vector shape (min(m, n) == 1)
    // the cap has no intermediate value, so the dedicated gemv path is either on (MAX, default) or
    // off (0, general driver). Measured in parallel across a range of m/k, since the dedicated
    // path's edge is mostly its bandwidth-parallel behavior; default-biased, so it only flips off
    // if the driver robustly wins across every probed shape
    knob!(
        "GEMMKIT_GEMV_THRESHOLD",
        sweep_gemv(
            "GEMMKIT_GEMV_THRESHOLD",
            tuning::set_gemv_threshold,
            tuning::GEMV_THRESHOLD_DEFAULT,
            &[0],
            &timing,
            &[(4096, 64), (65536, 16), (1 << 20, 8), (1024, 256)],
            par,
        )
    );

    // Large-matrix probes: opt-in via --large-matrices <GiB>. Both knobs bite only in an expensive
    // regime (K_STREAM_MAX once the gemv output spills the LLC, needing multi-GB matrices;
    // SHARED_LHS_MNK above its high-FLOP pre-pass crossover), so both run only once the user opts
    // in with a memory budget
    match cli.large_gib {
        None => {
            large_skipped.push((
                "GEMMKIT_K_STREAM_MAX",
                "gemv register-block cap; only bites in the DRAM-bound huge-m regime (output \
                 spilling the LLC) — pass --large-matrices <GiB> (needs multi-GB matrices) to probe \
                 it. The maintainer bench perf_k_stream also covers this calibration"
                    .to_string(),
            ));
            large_skipped.push((
                "GEMMKIT_SHARED_LHS_MNK",
                "shared-A pre-pass crossover; the pre-pass engages only above a large m*n*k (~8e9 \
                 on x86), so probing needs high-FLOP shapes — pass --large-matrices <GiB> to enable"
                    .to_string(),
            ));
        }
        Some(gib) => {
            // u64, not usize: a validated `gib` (<= 4096) cannot overflow u64, so the budget stays
            // exact on every target; a 32-bit usize would saturate and defeat plan_k_stream's gate
            let budget = (gib * (1u64 << 30) as f64) as u64;
            match plan_k_stream(budget, gib) {
                Ok((rows, ks)) => {
                    let shapes: Vec<(usize, usize)> = ks.iter().map(|&k| (rows, k)).collect();
                    knob!(
                        "GEMMKIT_K_STREAM_MAX",
                        sweep_gemv(
                            "GEMMKIT_K_STREAM_MAX",
                            tuning::set_k_stream_max,
                            tuning::K_STREAM_MAX_DEFAULT,
                            &[16, 48],
                            &timing,
                            &shapes,
                            par,
                        )
                    );
                }
                Err(reason) => large_skipped.push(("GEMMKIT_K_STREAM_MAX", reason)),
            }
            knob!(
                "GEMMKIT_SHARED_LHS_MNK",
                sweep_sgemm(
                    "GEMMKIT_SHARED_LHS_MNK",
                    tuning::set_shared_lhs_mnk,
                    tuning::SHARED_LHS_MNK_DEFAULT,
                    &[2_000_000_000, 4_000_000_000, MAX],
                    &timing,
                    &[(2048, 1024, 1024), (6144, 1024, 1024), (12288, 1024, 1024)],
                    par,
                    false,
                )
            );
        }
    }

    bar.finish_and_clear();

    // knobs deliberately not swept on this machine, each with why; reasons are owned Strings so
    // the opt-in and budget branches below (whose reasons are built at runtime) can push here too
    let mut skipped: Vec<(&'static str, String)> = vec![(
        "GEMMKIT_PARALLEL_THRESHOLD",
        "serial/parallel break-even is strongly shape-dependent \
             and the default is a deliberate cross-shape compromise"
            .to_string(),
    )];
    // DEEP_KC_BYTES gates the f16/bf16 deep-contraction twin: this tool runs no narrow-type
    // probes, and the auto default is derived from L2 (a machine property); override
    // GEMMKIT_DEEP_KC_BYTES directly to retune the narrow deep-k engage point
    skipped.push((
        "GEMMKIT_DEEP_KC_BYTES",
        "narrow-only (f16/bf16 deep-contraction twin engage gate); no narrow probe here, and the \
         auto default is derived from L2"
            .to_string(),
    ));
    // PREFETCH_MIN_BYTES gates the driver's C-tile prefetch on working-set-vs-LLC: the auto
    // default is derived from the detected LLC (a machine property, like DEEP_KC_BYTES), and
    // probing the crossover needs beyond-LLC working sets on every candidate; override
    // GEMMKIT_PREFETCH_MIN_BYTES directly (usize::MAX disables) to retune the engage point
    skipped.push((
        "GEMMKIT_PREFETCH_MIN_BYTES",
        "C-tile prefetch engage gate; the auto default is derived from the detected LLC, and \
         probing the crossover needs beyond-LLC working sets"
            .to_string(),
    ));
    // SEQ_INTERNAL_BYTES_PER_WORKER is swept above on aarch64, the only arch whose resolve_batch
    // reads it; inert on x86, so skip it there
    #[cfg(not(target_arch = "aarch64"))]
    skipped.push((
        "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
        "aarch64-only effect (batched split plan); inert on x86".to_string(),
    ));
    // I8_VNNI_MIN_PAR_MNK is swept above on x86, the only arch with a VNNI i8 small-parallel
    // fallback; elsewhere no i8 kernel has a fallback, so the knob is inert - skip it there
    #[cfg(not(target_arch = "x86_64"))]
    skipped.push((
        "GEMMKIT_I8_VNNI_MIN_PAR_MNK",
        "x86 VNNI-fallback knob; the i8 kernels on other targets have no small-parallel fallback, \
         so it is inert"
            .to_string(),
    ));
    // NC_NO_L3_PANELS is swept above only on a no-L3 host; the cap is dead on an L3 host, so skip it
    if gemmkit::topology().l3.is_some() {
        skipped.push((
            "GEMMKIT_NC_NO_L3_PANELS",
            "inert here (this host has an L3); the no-L3 column-block cap is only consulted on a \
             no-L3 host"
                .to_string(),
        ));
    }
    // heavy knobs skipped for want of --large-matrices or time budget (see the match above)
    skipped.append(&mut large_skipped);
    for &env in &budget_skipped {
        skipped.push((env, "time budget exhausted".to_string()));
    }

    report(&results, &skipped, threads, start.elapsed());

    if cli.dry_run {
        eprintln!("gemmkit-tune: --dry-run, no profile written.");
    } else {
        let profile = build_profile(&results, &skipped, threads);
        match std::fs::write(&cli.out, &profile) {
            Ok(()) => eprintln!("gemmkit-tune: wrote {}", cli.out),
            Err(e) => {
                eprintln!("gemmkit-tune: failed to write {}: {e}", cli.out);
                std::process::exit(1);
            }
        }
    }
}

/// A knob's public setter, e.g. `tuning::set_mc_reg_panels`
type Setter = fn(usize);

/// Setter/default pairs neutralized before every sweep run, so a stale `GEMMKIT_*` env value
/// cannot leak into a baseline measurement: covers knobs a sweep's `gemm` calls may consult even
/// when that particular knob is not the one being swept (`PARALLEL_THRESHOLD` gates every probe's
/// serial/parallel choice), or that are swept only under some other config (`NC_NO_L3_PANELS`,
/// `SEQ_INTERNAL_BYTES_PER_WORKER`)
const NEUTRALIZE: &[(Setter, usize)] = &[
    (tuning::set_mc_reg_panels, tuning::MC_REG_PANELS_DEFAULT),
    (tuning::set_kc_min, tuning::KC_MIN_DEFAULT),
    (tuning::set_tiny_block_dim, tuning::TINY_BLOCK_DIM_DEFAULT),
    (tuning::set_kc, tuning::KC_DEFAULT),
    (
        tuning::set_pack_transpose_tile,
        tuning::PACK_TRANSPOSE_TILE_DEFAULT,
    ),
    (
        tuning::set_parallel_oversample,
        tuning::PARALLEL_OVERSAMPLE_DEFAULT,
    ),
    (
        tuning::set_packed_oversample,
        tuning::PACKED_OVERSAMPLE_DEFAULT,
    ),
    (
        tuning::set_par_mnk_per_worker,
        tuning::PAR_MNK_PER_WORKER_DEFAULT,
    ),
    (tuning::set_pool_classes, tuning::POOL_CLASSES_DEFAULT),
    (tuning::set_full_width_mnk, tuning::FULL_WIDTH_MNK_DEFAULT),
    (
        tuning::set_rhs_pack_threshold,
        tuning::RHS_PACK_THRESHOLD_DEFAULT,
    ),
    (
        tuning::set_lhs_pack_threshold,
        tuning::LHS_PACK_THRESHOLD_DEFAULT,
    ),
    (tuning::set_lhs_pack_stride, tuning::LHS_PACK_STRIDE_DEFAULT),
    (tuning::set_lhs_pack_span, tuning::LHS_PACK_SPAN_DEFAULT),
    (tuning::set_lhs_pack_reuse, tuning::LHS_PACK_REUSE_DEFAULT),
    (
        tuning::set_small_k_threshold,
        tuning::SMALL_K_THRESHOLD_DEFAULT,
    ),
    (tuning::set_small_mn_dim, tuning::SMALL_MN_DIM_DEFAULT),
    (tuning::set_gemv_thread_cap, tuning::GEMV_THREAD_CAP_DEFAULT),
    (
        tuning::set_gemv_parallel_bytes,
        tuning::GEMV_PARALLEL_BYTES_DEFAULT,
    ),
    (
        tuning::set_i8_vnni_min_par_mnk,
        tuning::I8_VNNI_MIN_PAR_MNK_DEFAULT,
    ),
    (tuning::set_gemv_threshold, tuning::GEMV_THRESHOLD_DEFAULT),
    // heavy knobs, only swept under --large-matrices, but neutralized unconditionally: both
    // getters are still consulted on every ordinary gemv/sgemm sweep call, so a stale env value
    // could skew their baselines even when this sweep itself never runs
    (tuning::set_k_stream_max, tuning::K_STREAM_MAX_DEFAULT),
    (tuning::set_shared_lhs_mnk, tuning::SHARED_LHS_MNK_DEFAULT),
    // consulted by gemm calls during every sweep, so a stale env value would silently skew a
    // baseline regardless of whether the knob is itself being swept. PARALLEL_THRESHOLD gates
    // serial/parallel dispatch in every sgemm/gemv sweep; NC_NO_L3_PANELS caps the block on a
    // no-L3 host (swept there, skipped on an L3 host); SEQ_INTERNAL_BYTES_PER_WORKER drives the
    // aarch64 batched split (swept there, skipped on x86)
    (
        tuning::set_parallel_threshold,
        tuning::PARALLEL_THRESHOLD_DEFAULT,
    ),
    (tuning::set_nc_no_l3_panels, tuning::NC_NO_L3_PANELS_DEFAULT),
    (
        tuning::set_seq_internal_bytes_per_worker,
        tuning::SEQ_INTERNAL_BYTES_PER_WORKER_DEFAULT,
    ),
];

/// Render a knob value for display: the `usize::MAX` / `usize::MAX - 1` unbounded sentinels print
/// as `MAX`, everything else as its decimal value
fn show(v: usize) -> String {
    if v == usize::MAX || v == usize::MAX - 1 {
        "MAX".to_string()
    } else {
        v.to_string()
    }
}

fn report(
    results: &[KnobResult],
    skipped: &[(&'static str, String)],
    threads: usize,
    elapsed: Duration,
) {
    let secs = elapsed.as_secs_f64();
    println!(
        "\n{}",
        style(format!(
            "── gemmkit-tune · {threads} worker(s) · {secs:.1}s ──"
        ))
        .bold()
    );

    let mut moved = 0usize;

    if !results.is_empty() {
        // summary table: one row per swept knob
        const RIGHT: [bool; 7] = [false, false, true, true, true, true, false];
        let headers = [
            "knob", "unit", "shapes", "default", "winner", "speedup", "result",
        ];
        let mut rows: Vec<(Vec<String>, bool)> = Vec::with_capacity(results.len());
        for r in results {
            let changed = r.winner != r.default;
            moved += usize::from(changed);
            let knob = r.env.strip_prefix("GEMMKIT_").unwrap_or(r.env).to_string();
            let result = if changed {
                format!("→ {}", show(r.winner))
            } else {
                "keeps default".to_string()
            };
            rows.push((
                vec![
                    knob,
                    r.unit.to_string(),
                    r.shapes.to_string(),
                    show(r.default),
                    show(r.winner),
                    format!("{:.2}×", r.speedup()),
                    result,
                ],
                changed,
            ));
        }
        // column widths measured from plain text; color wraps the whole padded line afterward,
        // so it never perturbs alignment
        let mut w: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
        for (cells, _) in &rows {
            for (c, cell) in cells.iter().enumerate() {
                w[c] = w[c].max(cell.chars().count());
            }
        }
        let fmt = |cells: &[String]| -> String {
            let mut s = String::from("  ");
            for (c, cell) in cells.iter().enumerate() {
                let pad = " ".repeat(w[c].saturating_sub(cell.chars().count()));
                if RIGHT[c] {
                    s.push_str(&pad);
                    s.push_str(cell);
                } else {
                    s.push_str(cell);
                    s.push_str(&pad);
                }
                s.push_str("   ");
            }
            s.trim_end().to_string()
        };
        println!();
        let hdr: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        println!("{}", style(fmt(&hdr)).bold());
        let sep: Vec<String> = w.iter().map(|&x| "─".repeat(x)).collect();
        println!("{}", style(fmt(&sep)).dim());
        for (cells, changed) in &rows {
            let line = fmt(cells);
            if *changed {
                println!("{}", style(line).yellow());
            } else {
                println!("{line}");
            }
        }

        // candidate detail: the sweep landscape behind each winner
        println!(
            "\n{}",
            style("  candidates · geomean over probe shapes · * default  ‹ winner").dim()
        );
        let kw = results
            .iter()
            .map(|r| {
                r.env
                    .strip_prefix("GEMMKIT_")
                    .unwrap_or(r.env)
                    .chars()
                    .count()
            })
            .max()
            .unwrap_or(0);
        for r in results {
            let knob = r.env.strip_prefix("GEMMKIT_").unwrap_or(r.env);
            let mut line = format!("  {knob:<kw$}  ");
            for row in &r.rows {
                let mark = if row.value == r.default { "*" } else { "" };
                let win = if row.value == r.winner { "‹" } else { "" };
                line.push_str(&format!(
                    "{mark}{}={:.1}{win}  ",
                    show(row.value),
                    row.stat.median
                ));
            }
            println!("{}", line.trim_end());
        }
    }

    // skipped
    if !skipped.is_empty() {
        println!("\n{}", style("  skipped").bold());
        let kw = skipped
            .iter()
            .map(|(e, _)| e.strip_prefix("GEMMKIT_").unwrap_or(e).chars().count())
            .max()
            .unwrap_or(0);
        for (env, why) in skipped {
            let knob = env.strip_prefix("GEMMKIT_").unwrap_or(env);
            println!(
                "  {}  {}",
                style(format!("{knob:<kw$}")).dim(),
                style(why).dim()
            );
        }
    }

    // footer
    println!(
        "\n  {}",
        style(format!(
            "{} swept · {moved} moved off default · {} skipped · {secs:.1}s",
            results.len(),
            skipped.len(),
        ))
        .bold()
    );
    if results.is_empty() {
        println!(
            "  {}",
            style("no knobs swept — the time budget was too small to measure even one; raise --time-budget").dim()
        );
    } else if moved == 0 {
        println!(
            "  {}",
            style("all knobs kept their defaults — expected on the machine the defaults were calibrated on; the profile reproduces them").dim()
        );
    }
}

fn build_profile(
    results: &[KnobResult],
    skipped: &[(&'static str, String)],
    threads: usize,
) -> String {
    let mut s = host_stamp(threads);
    s.push('\n');
    for r in results {
        let tag = if r.winner == r.default {
            "default"
        } else {
            "tuned"
        };
        // emit the raw integer, never the `show()` "MAX" alias: gemmkit's resolve_env only parses
        // a decimal integer, so an unbounded (usize::MAX) winner must be written numerically (it
        // gets clamped back to MAX - 1 on load, equivalent for a disable/unbounded gate)
        s.push_str(&format!(
            "export {}={}  # {} ({:.2}x)\n",
            r.env,
            r.winner,
            tag,
            r.speedup()
        ));
    }
    s.push_str("\n# not swept on this host:\n");
    for (env, why) in skipped {
        s.push_str(&format!("#   {env}: {why}\n"));
    }
    s
}

// Knob-coverage guard
//
// Every canonical gemmkit knob must be either swept by this tool (on at least one supported
// target/config) or listed in the small never-tuned allowlist below, with a reason. Both lists are
// asserted against `gemmkit::tuning::knob_env_names` in the test below, so a knob added to gemmkit
// cannot silently escape the autotuner: it fails the test until classified as TUNED (with a real
// `knob!` sweep) or NEVER_TUNED

/// Knobs this tool sweeps on at least one supported target/config. Some are arch- or flag-gated:
/// `SEQ_INTERNAL_BYTES_PER_WORKER` only on aarch64, `I8_VNNI_MIN_PAR_MNK` only on x86,
/// `NC_NO_L3_PANELS` only on a no-L3 host, `K_STREAM_MAX`/`SHARED_LHS_MNK` only under
/// `--large-matrices`, but each appears as a `knob!` sweep somewhere in `main`
#[cfg(test)]
const TUNED: &[&str] = &[
    "GEMMKIT_RHS_PACK_THRESHOLD",
    "GEMMKIT_LHS_PACK_THRESHOLD",
    "GEMMKIT_LHS_PACK_STRIDE",
    "GEMMKIT_LHS_PACK_SPAN",
    "GEMMKIT_LHS_PACK_REUSE",
    "GEMMKIT_GEMV_THRESHOLD",
    "GEMMKIT_SMALL_K_THRESHOLD",
    "GEMMKIT_SMALL_MN_DIM",
    "GEMMKIT_SMALL_MN_PACK_MIN_K",
    "GEMMKIT_GEMV_PARALLEL_BYTES",
    "GEMMKIT_GEMV_THREAD_CAP",
    "GEMMKIT_PARALLEL_OVERSAMPLE",
    "GEMMKIT_PAR_MNK_PER_WORKER",
    "GEMMKIT_POOL_CLASSES",
    "GEMMKIT_FULL_WIDTH_MNK",
    "GEMMKIT_SHARED_LHS_MNK",
    "GEMMKIT_K_STREAM_MAX",
    "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
    "GEMMKIT_PACKED_OVERSAMPLE",
    "GEMMKIT_MC_REG_PANELS",
    "GEMMKIT_NC_NO_L3_PANELS",
    "GEMMKIT_TINY_BLOCK_DIM",
    "GEMMKIT_KC",
    "GEMMKIT_KC_MIN",
    "GEMMKIT_PACK_TRANSPOSE_TILE",
    "GEMMKIT_I8_VNNI_MIN_PAR_MNK",
];

/// Knobs deliberately never swept, each with why; mirrors the owned `skipped` reasons `report`
/// builds at runtime
#[cfg(test)]
const NEVER_TUNED: &[(&str, &str)] = &[
    (
        "GEMMKIT_PARALLEL_THRESHOLD",
        "serial/parallel break-even is strongly shape-dependent; the cross-shape default is kept",
    ),
    (
        "GEMMKIT_DEEP_KC_BYTES",
        "narrow-only (f16/bf16 deep-contraction twin); no narrow probe here, auto default is L2-derived",
    ),
    (
        "GEMMKIT_PREFETCH_MIN_BYTES",
        "C-tile prefetch engage gate; auto default is LLC-derived, probing needs beyond-LLC working sets",
    ),
];

// test-only: checks that TUNED and NEVER_TUNED exactly partition gemmkit's knob_env_names()
#[cfg(test)]
mod knob_coverage {
    use super::{NEVER_TUNED, TUNED};
    use std::collections::BTreeSet;

    #[test]
    fn sweep_table_covers_every_knob() {
        // gemmkit-tune enables the int8 feature but not wasm_threads (see Cargo.toml), so
        // knob_env_names() is always the 28 base knobs plus I8_VNNI_MIN_PAR_MNK, 29 total; TUNED
        // and NEVER_TUNED must partition that set exactly
        let names: BTreeSet<&str> = gemmkit::tuning::knob_env_names().iter().copied().collect();
        let tuned: BTreeSet<&str> = TUNED.iter().copied().collect();
        let never: BTreeSet<&str> = NEVER_TUNED.iter().map(|&(n, _)| n).collect();

        assert_eq!(tuned.len(), TUNED.len(), "TUNED has a duplicate entry");
        assert_eq!(
            never.len(),
            NEVER_TUNED.len(),
            "NEVER_TUNED has a duplicate entry"
        );
        assert!(
            tuned.is_disjoint(&never),
            "a knob is both tuned and never-tuned"
        );

        let handled: BTreeSet<&str> = tuned.union(&never).copied().collect();

        // coverage: every canonical knob must be classified, catching one added to gemmkit that
        // this sweep table forgot
        let missing: Vec<&str> = names.difference(&handled).copied().collect();
        assert!(
            missing.is_empty(),
            "gemmkit knobs neither swept nor in the never-tuned allowlist: {missing:?}"
        );
        // no stale or misspelled entries: every TUNED/NEVER_TUNED name must be a real canonical knob
        let unknown: Vec<&str> = handled.difference(&names).copied().collect();
        assert!(
            unknown.is_empty(),
            "TUNED/NEVER_TUNED name is not a gemmkit knob (typo or removed?): {unknown:?}"
        );
    }
}
