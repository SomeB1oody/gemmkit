//! Deep-contraction narrow-twin route parity (feature `half`)
//!
//! A narrow-output family (`f16`/`bf16`, `OUT_IS_ACC = false`) runs the whole contraction as one
//! depth panel (`kc = k`); above the `GEMMKIT_DEEP_KC_BYTES` engage gate the dispatch instead runs
//! the f32-output twin (`OUT_IS_ACC = true`, multi-slice) into an f32 scratch and narrows once. The
//! twin seeds each slice's accumulators from the scratch, continuing the single panel's ascending-k
//! accumulation split at slice boundaries, so for the common `beta in {0, 1}` the deep-k route is
//! **byte-for-byte** the single-panel route; for a general `beta` it is accurate to tolerance (the
//! single panel itself fuses `beta*C + AB` on full tiles but not on edge tiles, so no single sweep
//! matches both). Serial and parallel are always bit-identical (the twin's blocking is thread-count
//! independent; the narrowing sweep is elementwise)
//!
//! Platform-independent: the route is toggled by the runtime `GEMMKIT_DEEP_KC_BYTES` knob (`1`
//! forces the twin at any `k`, `MAX` forces the single panel), so this exercises whichever ISA the
//! host selected. The x86 avx512 / avx512bf16 pins that isolate the widen (`MixedGemm`) vs dot
//! (`Bf16DotGemm`) bf16 families live in the `env_isa_avx512` (widen) and `env_isa_bf16` (dot)
//! binaries (memoized ISA needs its own process)
#![cfg(all(
    feature = "half",
    feature = "std",
    not(miri),
    not(target_family = "wasm")
))]

use gemmkit::{MatMut, MatRef, Parallelism, bf16, f16, gemm, tuning};

/// Serializes the knob-mutating tests in this binary (the harness runs them concurrently and
/// `GEMMKIT_DEEP_KC_BYTES` is process-global). Poison-tolerant so one panicking test does not
/// cascade
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Deterministic spread-out narrow fill (xorshift), values in roughly `[-0.5, 0.5)` so bf16/f16
/// products stay in range and the reductions are non-trivial
fn fill<N: gemmkit::NarrowFloat>(n: usize, seed: u64) -> Vec<N> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            N::narrow((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        })
        .collect()
}

/// Run one narrow GEMM with the deep-k route toggled by the engage gate: `engage = true` forces the
/// f32-output twin (gate `1`), `false` forces the single depth panel (gate `MAX`). Column-major A/B;
/// C uses `(rsc, csc)` so the strided (scalar) narrowing-sweep branch is reachable. Returns the raw
/// output bits
#[allow(clippy::too_many_arguments)]
fn run<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(
    engage: bool,
    m: usize,
    k: usize,
    n: usize,
    a: &[N],
    b: &[N],
    c0: &[N],
    rsc: isize,
    csc: isize,
    alpha: N,
    beta: N,
    par: Parallelism,
) -> Vec<u16> {
    tuning::set_deep_kc_bytes(if engage { 1 } else { usize::MAX });
    let mut c = c0.to_vec();
    gemm(
        alpha,
        MatRef::from_col_major(a, m, k),
        MatRef::from_col_major(b, k, n),
        beta,
        MatMut::new(&mut c, m, n, rsc, csc),
        par,
    );
    // Raw bits so a bf16 NaN payload compares exactly and never trips a float PartialEq
    c.iter().map(|v| bits16(*v)).collect()
}

/// Raw 16-bit pattern of a narrow float (f16/bf16), for exact bitwise comparison
fn bits16<N: gemmkit::NarrowFloat>(v: N) -> u16 {
    // widen->narrow is exact for a real narrow value, but to read the *stored* bits we transmute the
    // 2-byte element; both f16 and bf16 are `#[repr(transparent)]` over u16
    // SAFETY: N is f16 or bf16, both 2-byte transparent-over-u16
    unsafe { core::mem::transmute_copy::<N, u16>(&v) }
}

/// Max element magnitude of a widened narrow buffer (for a relative tolerance)
fn max_abs<N: gemmkit::NarrowFloat>(bits: &[u16], to: fn(u16) -> N) -> f32 {
    bits.iter()
        .map(|&b| to(b).widen().abs())
        .fold(0.0, f32::max)
}

fn f16_from_bits(b: u16) -> f16 {
    f16::from_bits(b)
}
fn bf16_from_bits(b: u16) -> bf16 {
    bf16::from_bits(b)
}

/// Bit-identity of deep-k vs single-slice for `beta in {0, 1}`, plus serial==parallel bit-identity
/// of the deep-k route, sweeping shapes that mix full and edge tiles and both the unit-stride
/// (vector) and strided (scalar) narrowing-sweep branches
fn bit_identity_case<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(label: &str) {
    let _lock = knob_guard();
    let restore = tuning::deep_kc_bytes();

    // (m, n, rsc, csc): full+edge tile mixes; the last one gives a strided (rsc != 1) C so the
    // scalar narrowing-sweep branch is exercised alongside the unit-stride vector one
    let shapes: &[(usize, usize, isize, isize)] = &[
        (48, 48, 1, 48),     // aligned-ish, unit-stride C
        (40, 50, 1, 40),     // edge row/col tiles, unit-stride C
        (130, 33, 1, 130),   // wide row blocks + edge, unit-stride C
        (37, 41, 2, 2 * 37), // strided C (rsc = 2): scalar sweep branch
    ];
    // k past the small-k gate; small enough for GEMMKIT_FAST_TEST. The twin multi-slices at the
    // cache-model kc, so a few slices run
    let k = 4096usize;

    for &(m, n, rsc, csc) in shapes {
        let a: Vec<N> = fill(m * k, 0x1234 ^ m as u64);
        let b: Vec<N> = fill(k * n, 0x9abc ^ n as u64);
        // C backing large enough for the strided view: max index (m-1)*rsc + (n-1)*csc
        let cbacking = (m as isize * rsc.abs() + n as isize * csc.abs()) as usize + 8;
        let c0: Vec<N> = fill(cbacking, 0x55 ^ (m * n) as u64);

        for &beta_f in &[0.0f32, 1.0] {
            let (alpha, beta) = (N::narrow(1.25), N::narrow(beta_f));
            let single = run(
                false,
                m,
                k,
                n,
                &a,
                &b,
                &c0,
                rsc,
                csc,
                alpha,
                beta,
                Parallelism::Serial,
            );
            let deep_ser = run(
                true,
                m,
                k,
                n,
                &a,
                &b,
                &c0,
                rsc,
                csc,
                alpha,
                beta,
                Parallelism::Serial,
            );
            let deep_par = run(
                true,
                m,
                k,
                n,
                &a,
                &b,
                &c0,
                rsc,
                csc,
                alpha,
                beta,
                Parallelism::Rayon(0),
            );
            assert_eq!(
                deep_ser, single,
                "{label}: deep-k must be bit-identical to the single panel (m={m} n={n} k={k} beta={beta_f} rsc={rsc})"
            );
            assert_eq!(
                deep_par, deep_ser,
                "{label}: deep-k serial vs parallel must be bit-identical (m={m} n={n} beta={beta_f} rsc={rsc})"
            );
        }
    }

    tuning::set_deep_kc_bytes(restore);
}

/// General `beta` (neither 0 nor 1): the deep-k route is *not* required to be bitwise equal to the
/// single panel (the single panel fuses `beta*C + AB` on full tiles but not on edge tiles), only
/// accurate to a tight relative tolerance. Confirms the fallback contract holds. Swept over both a
/// unit-stride C (`rsc == 1`, the twin's vectorized narrowing sweep) and a strided C (`rsc == 2`,
/// the twin's scalar per-element narrowing sweep): the general-`beta` `BetaStatus::Other` arm of
/// that scalar branch has no other coverage (the bit-identity case exercises the strided branch only
/// for `beta in {0, 1}`)
fn tolerance_case<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(
    label: &str,
    to: fn(u16) -> N,
) {
    let _lock = knob_guard();
    let restore = tuning::deep_kc_bytes();

    let (m, n, k) = (40usize, 50, 4096);
    let a: Vec<N> = fill(m * k, 0x2222);
    let b: Vec<N> = fill(k * n, 0x3333);
    let (alpha, beta) = (N::narrow(0.75), N::narrow(2.5));

    // (rsc, csc): unit-stride C (vector sweep) then strided C (rsc = 2, the scalar sweep branch)
    for &(rsc, csc) in &[(1isize, m as isize), (2isize, 2 * m as isize)] {
        // C backing large enough for the strided view: max index (m-1)*rsc + (n-1)*csc
        let cbacking = (m as isize * rsc.abs() + n as isize * csc.abs()) as usize + 8;
        let c0: Vec<N> = fill(cbacking, 0x4444 ^ rsc as u64);

        let single = run(
            false,
            m,
            k,
            n,
            &a,
            &b,
            &c0,
            rsc,
            csc,
            alpha,
            beta,
            Parallelism::Serial,
        );
        let deep = run(
            true,
            m,
            k,
            n,
            &a,
            &b,
            &c0,
            rsc,
            csc,
            alpha,
            beta,
            Parallelism::Serial,
        );

        // Two roundings of the same math: they differ only where the single panel used an edge tile
        // (unfused combine), by at most ~1 narrow ULP. Assert a tight relative Frobenius bound
        let scale = max_abs(&single, to).max(1e-6);
        let mut max_diff = 0.0f32;
        for (&s, &d) in single.iter().zip(&deep) {
            max_diff = max_diff.max((to(s).widen() - to(d).widen()).abs());
        }
        assert!(
            max_diff <= 0.05 * scale,
            "{label}: deep-k must match the single panel within tolerance for general beta (rsc={rsc}, max_diff={max_diff}, scale={scale})"
        );
    }

    tuning::set_deep_kc_bytes(restore);
}

#[test]
fn deep_k_bit_identical_f16() {
    bit_identity_case::<f16>("f16");
}

#[test]
fn deep_k_bit_identical_bf16() {
    bit_identity_case::<bf16>("bf16");
}

#[test]
fn deep_k_tolerance_f16() {
    tolerance_case::<f16>("f16", f16_from_bits);
}

#[test]
fn deep_k_tolerance_bf16() {
    tolerance_case::<bf16>("bf16", bf16_from_bits);
}
