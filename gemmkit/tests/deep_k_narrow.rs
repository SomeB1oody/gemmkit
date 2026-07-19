//! Deep-contraction narrow-twin route parity (feature `half`)
//!
//! A narrow-output family (f16/bf16, `OUT_IS_ACC = false`) runs a whole contraction as 1 depth
//! panel (`kc = k`), so a large-`k` micropanel can outgrow L2. Above the `GEMMKIT_DEEP_KC_BYTES`
//! engage gate, dispatch instead re-blocks through the family's f32-output twin (`OUT_IS_ACC =
//! true`), which multi-slices the contraction at the cache-model `kc` into an f32 scratch buffer
//! and narrows only once at the end. The twin continues the single panel's ascending-k
//! accumulation across slice boundaries, so for `beta in {0, 1}` deep-k is byte-for-byte the
//! single-panel route; for a general `beta` it only has to hold to tolerance, since the single
//! panel fuses `beta*C + AB` on full tiles but not on edge tiles and no single sweep can match
//! both. Serial and parallel stay bit-identical either way: the twin's blocking does not depend
//! on the thread count, and the final narrowing sweep is a plain elementwise pass
//!
//! Platform-independent: the route is toggled purely by the runtime `GEMMKIT_DEEP_KC_BYTES` knob
//! (`1` forces the twin at any `k`, `usize::MAX` forces the single panel), so this exercises
//! whichever ISA the host actually selects. The x86 `avx512` / `avx512bf16` pins that isolate the
//! widen (`MixedGemm`) bf16 twin from the dot (`Bf16DotGemm`) one live in the separate
//! `env_isa_avx512` and `env_isa_bf16` binaries, since a memoized ISA choice needs its own process
//! to force
#![cfg(all(
    feature = "half",
    feature = "std",
    not(miri),
    not(target_family = "wasm")
))]

use gemmkit::{MatMut, MatRef, Parallelism, bf16, f16, gemm, tuning};

/// Serializes this binary's tests around the process-global `GEMMKIT_DEEP_KC_BYTES` knob, since
/// libtest otherwise runs them concurrently. `knob_guard` below recovers from poisoning, so 1
/// panicking test does not fail every test that runs after it
static KNOB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn knob_guard() -> std::sync::MutexGuard<'static, ()> {
    KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Deterministic xorshift fill in roughly `[-0.5, 0.5)`, narrowed to f16/bf16: keeps products in
/// range while spreading values enough that the reduction is not trivially uniform
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

/// Run 1 narrow GEMM with the deep-k engage gate set directly by `engage`: `true` forces the
/// f32-output twin (gate `1`), `false` forces the single depth panel (gate `usize::MAX`).
/// Column-major A/B; C takes explicit `(rsc, csc)` so a caller can drive the strided (scalar)
/// narrowing-sweep branch as well as the unit-stride one. Returns the raw 16-bit output pattern
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
    // Compared as raw bits, not floats, so a bf16 NaN payload compares exactly and PartialEq's
    // NaN-never-equals-itself rule never masks a mismatch
    c.iter().map(|v| bits16(*v)).collect()
}

/// The stored 16-bit pattern of a narrow float (f16/bf16), for exact bitwise comparison
fn bits16<N: gemmkit::NarrowFloat>(v: N) -> u16 {
    // NarrowFloat exposes only widen/narrow, no generic to_bits, so reading the stored pattern
    // needs a transmute here
    // SAFETY: N is f16 or bf16, both #[repr(transparent)] 2-byte wrappers over u16
    unsafe { core::mem::transmute_copy::<N, u16>(&v) }
}

/// Largest magnitude in a widened narrow buffer, as the scale for a relative tolerance
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

/// For `beta in {0, 1}`, checks that deep-k is bit-identical to the single panel and that the
/// deep-k route is itself serial == parallel bit-identical, sweeping shapes that mix full and
/// edge tiles and both the unit-stride (vector) and strided (scalar) narrowing-sweep branches
fn bit_identity_case<N: gemmkit::NarrowFloat + gemmkit::GemmScalar + Copy>(label: &str) {
    let _lock = knob_guard();
    let restore = tuning::deep_kc_bytes();

    // (m, n, rsc, csc): a mix of full and edge tiles, unit-stride C except the last shape, which
    // sets rsc = 2 to reach the scalar narrowing-sweep branch
    let shapes: &[(usize, usize, isize, isize)] = &[
        (48, 48, 1, 48),
        (40, 50, 1, 40),
        (130, 33, 1, 130),
        (37, 41, 2, 2 * 37),
    ];
    // Past the small-k gate, and small enough to run under GEMMKIT_FAST_TEST; large enough that
    // the twin's cache-model kc splits it into a few slices
    let k = 4096usize;

    for &(m, n, rsc, csc) in shapes {
        let a: Vec<N> = fill(m * k, 0x1234 ^ m as u64);
        let b: Vec<N> = fill(k * n, 0x9abc ^ n as u64);
        // Large enough for the strided view: its highest index is (m-1)*rsc + (n-1)*csc
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

/// For a general `beta` (neither 0 nor 1), deep-k is only guaranteed to match the single panel to
/// a tight relative tolerance, not bit-for-bit: the single panel fuses `beta*C + AB` on full tiles
/// but not edge tiles, so no single rounding order matches both. Swept over a unit-stride C (`rsc ==
/// 1`, the twin's vectorized narrowing sweep) and a strided C (`rsc == 2`, its scalar per-element
/// sweep); [`bit_identity_case`] only exercises that scalar sweep's `beta in {0, 1}` arms, so this
/// is the only coverage of its general-`beta` `BetaStatus::Other` arm
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

    // (rsc, csc): unit-stride C (vector sweep), then strided C (rsc = 2, scalar sweep branch)
    for &(rsc, csc) in &[(1isize, m as isize), (2isize, 2 * m as isize)] {
        // Large enough for the strided view: its highest index is (m-1)*rsc + (n-1)*csc
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

        // Both results round the same math; they can differ only on an edge tile, where the
        // single panel's combine is unfused, by roughly 1 narrow ULP
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
