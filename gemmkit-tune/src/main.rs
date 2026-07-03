//! `gemmkit-tune` — install-time autotuner for [`gemmkit`].
//!
//! Run this **on the deploy machine** (never in `build.rs`: the build host is not the deploy
//! host). It sweeps gemmkit's runtime knobs — the ones exposed as `set_*`/`GEMMKIT_*` in
//! `gemmkit::tuning` — measuring a representative set of shapes for each, then writes a
//! `gemmkit-tune.env` profile of `export GEMMKIT_*=…` lines. `source` that file before running a
//! gemmkit application to retune the shipped binary for the host with no recompile.
//!
//! ## How it sweeps
//!
//! Each knob is swept **independently**: all other knobs are held at their defaults, its candidate
//! values are measured back-to-back (A/B via the public setter, freshly-filled but identically
//! seeded buffers so machine drift and data cancel). Each candidate's score is the **geometric mean
//! throughput over the knob's probe-shape set** — one shape can flatter a value, a set cannot — so
//! a winner reflects a broad improvement, not a single-shape quirk. The tie-break is deliberately
//! **default-biased and noise-aware**: a candidate replaces the incumbent (which starts as the
//! default) only if its geomean beats it by more than the worst shape's measured spread, so
//! run-to-run noise can never rewrite a knob. There is no RNG anywhere, so the output is a
//! deterministic function of (machine, config).
//!
//! Because gemmkit's defaults were hand-calibrated on a Ryzen 9950X (Zen5), a run on that box
//! re-discovers essentially the same values (overall speedup ≈ 1.0×) — that is the *correct*
//! outcome and validates the tool; the payoff is on a *different* machine.
//!
//! Knobs whose value cannot be safely tuned by this probe are skipped and listed (with a reason) in
//! the report and the profile footer: ones inert on this machine (aarch64-only, no-L3-only,
//! wasm-only), and `PARALLEL_THRESHOLD` — its serial/parallel break-even is strongly shape-dependent
//! (a single `m*n*k` scalar can't fit all aspect ratios), so the calibrated cross-shape default is
//! kept rather than auto-tuned. `GEMV_THRESHOLD`, by contrast, is a clean binary on/off decision and
//! is swept.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use gemmkit::{MatMut, MatRef, Parallelism, gemm, tuning};

// ---------------------------------------------------------------------------
// Timing harness (a lean local copy of the private one in gemmkit/tests/perf.rs).
// ---------------------------------------------------------------------------

/// Deterministic `f32` fill (xorshift, so values are not all equal and reductions are non-trivial).
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

/// Deterministic `i8` fill for the integer probe.
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

/// A throughput sample: the median plus min/max so run-to-run spread is visible and never decides.
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

/// Measurement effort: reps per estimate and the per-batch time target. Coarsened by `--time-budget`.
#[derive(Clone, Copy)]
struct Timing {
    reps: usize,
    batch_secs: f64,
}

/// Robust single-shape throughput estimate over `reps` auto-sized batches, reported via `to_rate`
/// (`secs -> unit/s`). Warms up first, then reports the median (with min/max) — far steadier than a
/// single fixed-iteration timing.
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

/// Combine per-shape samples into one score: the **geometric mean** of the per-shape medians (equal
/// weight, so no single big shape dominates), with the *worst* shape's spread carried through so the
/// noise gate stays conservative across the whole set.
fn geomean(stats: &[Stat]) -> Stat {
    let n = stats.len().max(1) as f64;
    let ln_sum: f64 = stats.iter().map(|s| s.median.max(1e-9).ln()).sum();
    let median = (ln_sum / n).exp();
    let spread = stats.iter().map(Stat::spread_pct).fold(0.0, f64::max);
    let half = spread / 200.0; // so the reconstructed spread_pct() == `spread`
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

// ---------------------------------------------------------------------------
// Sweep engine
// ---------------------------------------------------------------------------

/// One measured candidate value (its score is the geomean over the probe-shape set).
struct Row {
    value: usize,
    stat: Stat,
}

/// The outcome of sweeping one knob.
struct KnobResult {
    env: &'static str,
    default: usize,
    winner: usize,
    unit: &'static str,
    shapes: usize,
    rows: Vec<Row>,
    baseline: f64, // the default value's score (the `--baseline`)
    winner_median: f64,
}

impl KnobResult {
    fn speedup(&self) -> f64 {
        self.winner_median / self.baseline.max(1e-9)
    }
}

/// Sweep one knob: score the default first (the baseline), then each extra candidate, then pick the
/// noise-aware, default-biased winner. Restores the knob to its default afterward so every sweep
/// sees an otherwise-default configuration. `probe` scores the workload under the *current*
/// (just-`set`) global knob value and returns one geomean `Stat`.
fn sweep(
    env: &'static str,
    set: fn(usize),
    default: usize,
    extras: &[usize],
    unit: &'static str,
    shapes: usize,
    mut probe: impl FnMut() -> Stat,
) -> KnobResult {
    // Candidates: default first (the tie-break incumbent), then the distinct extras in the given
    // order — a fixed, RNG-free order.
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
    set(default); // restore: keep sweeps independent

    // Winner: start at the default row, upgrade only when a candidate beats the incumbent by more
    // than the larger of the two spreads (so noise never decides, and ties keep the default).
    let baseline = rows[0].stat.median;
    let mut best = 0usize;
    for i in 1..rows.len() {
        let noise = rows[best].stat.spread_pct().max(rows[i].stat.spread_pct()) / 100.0;
        if rows[i].stat.median > rows[best].stat.median * (1.0 + noise) {
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

// ---- Per-shape-family sweep helpers (buffers rebuilt per shape; identical seeds → fair) ----

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

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Cli {
    threads: Option<usize>,
    budget_secs: Option<f64>,
    out: String,
    dry_run: bool,
    /// `Some(gib)` when `--large-matrices <GiB>` was passed: opt into the memory-/FLOP-heavy probes
    /// (`K_STREAM_MAX`, `SHARED_LHS_MNK`) with `gib` GiB as the peak-matrix budget for the giant ones.
    large_gib: Option<f64>,
}

/// Parse a `--time-budget` like `30s`, `2m`, `1h`, or a bare number of seconds. Returns `None` for
/// anything that is not a **finite, non-negative, non-absurd** value, so the caller rejects it
/// cleanly rather than letting a negative / NaN / infinite / overflowing value panic
/// `Duration::from_secs_f64`.
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

/// Print a usage error and exit — used for every malformed / missing CLI value so the tool never
/// silently ignores a flag the user passed.
fn die(msg: &str) -> ! {
    eprintln!("gemmkit-tune: {msg}\n");
    print_usage();
    std::process::exit(2);
}

/// The value token that must follow a flag, or a clean usage error.
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
    // `args_os` + lossy (not `args`): a non-UTF-8 argument must yield a clean usage error, not a
    // panic. A mangled arg simply fails to match a known flag and falls through to `die`.
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
                // A positive, finite GiB budget. Reject 0/NaN/inf/negative/absurd so a giant probe
                // matrix is never sized from garbage; the upper bound is a sanity cap, not a limit.
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

// ---------------------------------------------------------------------------
// Host stamp
// ---------------------------------------------------------------------------

/// UTC calendar date (Y, M, D) from a Unix timestamp (Howard Hinnant's `civil_from_days`).
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

const MAX: usize = usize::MAX;

/// Probe plan for the opt-in `K_STREAM_MAX` sweep. The gemv register-block gate engages past ~L3/2
/// and only *wins* once the output is clearly DRAM-bound, so the probe fixes the output at ~2x the
/// LLC — a 1x-LLC output sits on the cache boundary and measures nothing decisive. Returns the
/// shared row count and probe `k` set, or a reason string when the probe cannot fit this target's
/// address space, or when `budget_bytes` cannot hold the whole probe at the largest `k`; that reason
/// rounds the advised budget *up* to the printed precision, so re-running with it clears the gate.
fn plan_k_stream(budget_bytes: u64, gib: f64) -> Result<(usize, Vec<usize>), String> {
    const MAX_PROBE_K: usize = 48; // straddles the candidate caps {16, 32, 48}
    let topo = gemmkit::topology();
    let llc = topo.l3.map(|l| l.bytes).unwrap_or(topo.l2.bytes).max(1);
    // ~2x LLC so the plain form's per-column output re-reads clearly spill to DRAM (where register-
    // blocking pays). This is a fixed target, not budget-scaled: reaching it or skipping is the
    // whole point — a smaller output would run cache-resident and measure noise.
    let out = llc.saturating_mul(2);
    // Peak live bytes of the largest (k = MAX_PROBE_K) col-major f32 probe, matching `sweep_gemv`'s
    // allocation exactly: matrix a (rows*k*4 = out*k) + output y (rows*4 = out) + input x (k*4).
    // In u64 so a 32-bit `usize` can't saturate `need` (and the budget) and silently pass the gate.
    let need = (out as u64)
        .saturating_mul(MAX_PROBE_K as u64)
        .saturating_add(out as u64)
        .saturating_add(MAX_PROBE_K as u64 * 4);
    // The probe must fit this target's address space at all: on 32-bit a multi-GB matrix cannot be
    // allocated, so skip cleanly rather than let the Vec allocation abort the process.
    if need > usize::MAX as u64 {
        return Err(format!(
            "a DRAM-bound ~2x-LLC ({} MiB) gemv probe needs multi-GB matrices that do not fit this \
             target's address space — --large-matrices is only usable on 64-bit",
            out / (1 << 20),
        ));
    }
    if budget_bytes < need {
        // Round the advised budget UP to the printed 0.1 GiB, so following it actually clears the
        // `budget_bytes < need` gate (a truncated "1.5 GiB" that is really 1.53 would loop forever).
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
    let rows = (out / 4).max(1); // f32 output vector
    Ok((rows, vec![24, MAX_PROBE_K]))
}

fn main() {
    let cli = parse_cli();
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Honor `--threads` (>= 1, validated in parse_cli), but never claim more workers than gemmkit
    // will actually use (`Rayon(n)` is capped at the machine width internally), so the reported /
    // stamped worker count is truthful.
    let threads = cli.threads.unwrap_or(avail).min(avail).max(1);
    let par = Parallelism::Rayon(threads);
    let ser = Parallelism::Serial;

    // Coarsen reps to fit a tight budget; a hard deadline (below) skips whole knobs at the tail.
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

    // A GEMMKIT_* var in the tuning environment skews the baseline — gemmkit reads it during the
    // sweep, including for knobs this tool does not sweep and so never writes to the profile — so
    // warn rather than silently mis-tune. Run in a clean environment for a faithful profile.
    // `vars_os` (not `vars`): an unrelated non-UTF-8 environment variable must not panic the tool.
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

    // Neutralize any pre-existing GEMMKIT_* for the swept knobs so the baseline is the shipped
    // default — set via `set()` (never the env-consulting getter), so a value the tuning shell
    // carried does not leak into the baseline.
    for &(set, def) in NEUTRALIZE {
        set(def);
    }

    let mut results: Vec<KnobResult> = Vec::new();
    let mut budget_skipped: Vec<&str> = Vec::new();
    // Opt-in heavy knobs that were not run: either `--large-matrices` was absent, or its budget was
    // too small to reach the regime the knob controls. Owned reasons (the budget one is dynamic).
    let mut large_skipped: Vec<(&'static str, String)> = Vec::new();

    macro_rules! knob {
        ($env:literal, $body:expr) => {{
            if deadline.is_some_and(|d| Instant::now() >= d) {
                budget_skipped.push($env);
            } else {
                eprintln!("  {} ...", $env);
                results.push($body);
            }
        }};
    }

    // --- General register-tiling driver (blocking knobs) ---
    knob!(
        "GEMMKIT_MC_REG_PANELS",
        sweep_sgemm(
            "GEMMKIT_MC_REG_PANELS",
            tuning::set_mc_reg_panels,
            tuning::MC_REG_PANELS_DEFAULT,
            &[4, 6, 12, 16],
            &timing,
            &[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)],
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
            true, // row-major A -> the transpose packer
        )
    );

    // --- No-L3 column-block cap (only consulted where the machine reports no L3) ---
    // Gated on topology: dead on an L3 host, so it stays in `skipped` below. Large-`N` shapes make
    // the cap bind — `NC` is `min(this*NR, N)`, so `N` must clear `default*NR` (2048 f32 cols) for a
    // candidate to move it.
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

    // --- Parallel scheduling ---
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
            true, // row-major A -> the packed-LHS path
        )
    );
    knob!(
        "GEMMKIT_THREAD_DIM_STRIDE",
        sweep_sgemm(
            "GEMMKIT_THREAD_DIM_STRIDE",
            tuning::set_thread_dim_stride,
            tuning::THREAD_DIM_STRIDE_DEFAULT, // 0 = auto
            &[16, 24, 32, 48, 64],
            &timing,
            &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
            par,
            false,
        )
    );

    // --- Packing gates (route crossovers, probed near the crossover) ---
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
            &[256, 512, 2048, MAX],
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
            &[(1024, 512, 512), (1536, 384, 384)],
            par,
            false,
        )
    );

    // --- Route thresholds (small-k / small-mn) ---
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

    // --- Bandwidth-bound gemv ---
    // (GEMMKIT_K_STREAM_MAX is a heavy opt-in knob — swept only under --large-matrices; see the
    // "Large-matrix probes" block below.)
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
            &[(1 << 18, 8), (1 << 19, 4)],
            par,
        )
    );

    // --- Integer (i8) ---
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

    // --- gemv path vs general driver ---
    // Binary on/off: for a vector shape `min(m,n) == 1`, so the cap has no intermediate value — the
    // dedicated gemv path is either on (`MAX`, default) or off (`0`, general driver). Measured in
    // parallel over a range of `m`/`k` (the dedicated path's edge is largely its bandwidth-parallel
    // behavior); default-biased, so it flips off only if the driver robustly wins across the shapes.
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

    // --- Large-matrix probes (opt-in: --large-matrices <GiB>) ---
    // Both knobs only bite in an expensive regime — K_STREAM_MAX once the gemv output spills the LLC
    // (needs multi-GB matrices), SHARED_LHS_MNK above its high-FLOP pre-pass crossover — so they run
    // only when the user opts in with a memory budget.
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
            // u64 (not usize): a validated `gib` (<= 4096) can't overflow u64, so the budget is exact
            // on every target — a 32-bit usize would saturate and defeat plan_k_stream's gate.
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

    // Knobs deliberately not swept on this machine (with why). Owned reasons so the opt-in and
    // budget branches (whose reasons are dynamic) can append here uniformly.
    let mut skipped: Vec<(&'static str, String)> = vec![
        (
            "GEMMKIT_PARALLEL_THRESHOLD",
            "serial/parallel break-even is strongly shape-dependent \
             and the default is a deliberate cross-shape compromise"
                .to_string(),
        ),
        (
            "GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER",
            "aarch64-only effect (batched split plan); inert on x86 — tune on an M4".to_string(),
        ),
    ];
    // NC_NO_L3_PANELS is swept above on a no-L3 host; on an L3 host the cap is dead, so skip it.
    if gemmkit::topology().l3.is_some() {
        skipped.push((
            "GEMMKIT_NC_NO_L3_PANELS",
            "inert here (this host has an L3); the no-L3 column-block cap is only consulted on a \
             no-L3 host"
                .to_string(),
        ));
    }
    // Heavy knobs skipped for want of `--large-matrices` / budget (see the match above).
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

/// A knob's public setter, e.g. `tuning::set_mc_reg_panels`.
type Setter = fn(usize);

/// Every swept knob's (setter, default), used to neutralize pre-existing env before measuring.
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
        tuning::set_thread_dim_stride,
        tuning::THREAD_DIM_STRIDE_DEFAULT,
    ),
    (
        tuning::set_rhs_pack_threshold,
        tuning::RHS_PACK_THRESHOLD_DEFAULT,
    ),
    (
        tuning::set_lhs_pack_threshold,
        tuning::LHS_PACK_THRESHOLD_DEFAULT,
    ),
    (tuning::set_lhs_pack_stride, tuning::LHS_PACK_STRIDE_DEFAULT),
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
    // Heavy knobs (only swept under --large-matrices); neutralized unconditionally so a pre-set env
    // value cannot skew the baseline of the gemv / sgemm sweeps that read them.
    (tuning::set_k_stream_max, tuning::K_STREAM_MAX_DEFAULT),
    (tuning::set_shared_lhs_mnk, tuning::SHARED_LHS_MNK_DEFAULT),
    // Read by gemms during the other sweeps, so a stale env value would silently skew every
    // baseline — neutralize them whether or not they are themselves swept. PARALLEL_THRESHOLD gates
    // parallelism in every sgemm/gemv sweep; NC_NO_L3_PANELS caps the block on a no-L3 host (swept
    // there, skipped on an L3 host); SEQ_INTERNAL_BYTES_PER_WORKER is read only by batched GEMM.
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

/// A compact, human-readable value (turns the `usize::MAX` sentinel into `MAX`).
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
    println!("\n=== gemmkit-tune report (tuned for {threads} worker(s)) ===");
    println!(
        "(each candidate score is the geometric-mean throughput over the knob's probe shapes)\n"
    );
    let mut moved = 0;
    for r in results {
        println!(
            "{} (default {}) [{}, {} shapes]:",
            r.env,
            show(r.default),
            r.unit,
            r.shapes
        );
        let mut line = String::from("    ");
        for row in &r.rows {
            let mark = if row.value == r.default { "*" } else { " " };
            let win = if row.value == r.winner { "<" } else { " " };
            line.push_str(&format!(
                "{}{}={:>7.1}{}  ",
                mark,
                show(row.value),
                row.stat.median,
                win
            ));
        }
        println!("{line}");
        let sp = r.speedup();
        let note = if r.winner == r.default {
            "keeps default".to_string()
        } else {
            moved += 1;
            format!("CHANGED {} -> {}", show(r.default), show(r.winner))
        };
        println!(
            "    winner {} ({sp:.2}x vs default) — {note}\n",
            show(r.winner)
        );
    }

    if !skipped.is_empty() {
        println!("skipped:");
        for (env, why) in skipped {
            println!("    {env}: {why}");
        }
        println!();
    }
    println!(
        "{} knob(s) swept, {moved} moved off default, {} skipped; {:.1}s.",
        results.len(),
        skipped.len(),
        elapsed.as_secs_f64()
    );
    if results.is_empty() {
        println!(
            "No knobs were swept — the time budget was too small to measure even one. The profile \
             contains no tuned values; raise --time-budget."
        );
    } else if moved == 0 {
        println!(
            "All swept knobs kept their defaults — expected on the Zen5 box the defaults were \
             calibrated on. The profile reproduces them; run gemmkit-tune on a different machine \
             to specialize."
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
        // Emit the raw integer, never the `show()` "MAX" alias: gemmkit's resolve_env only parses
        // a decimal integer, so an unbounded (usize::MAX) winner must be written numerically (it is
        // clamped back to `MAX - 1` on load, which is equivalent for a "disable/unbounded" gate).
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
