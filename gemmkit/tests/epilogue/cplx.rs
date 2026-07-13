//! Complex fused-bias suite (`gemm_cplx_fused`): the contract is `gemm_cplx_fused == gemm_cplx`
//! then the same element-wise bias add, **bitwise** on both the real and imaginary parts, for
//! every shape and every conj combination. There is **no activation** on the complex frame
//! (ordering-based activations are undefined on `ℂ`), so this suite exercises bias only. All fills
//! are deterministic and platform-independent; `c32` and `c64` are covered via a generic `Cx`
//! element trait.

use crate::common::Rng;
use gemmkit::{
    Bias, Complex, MatMut, MatRef, Parallelism, Workspace, gemm_cplx, gemm_cplx_fused_with,
};

/// The complex element under test (`c32` / `c64`): deterministic construction and per-part bit
/// compare. Complex bitwise equality means both the `re` and `im` bit patterns match.
trait Cx: gemmkit::ComplexScalar {
    fn of(re: f64, im: f64) -> Self;
    fn bits(self) -> (u64, u64);
    fn name() -> &'static str;
}
impl Cx for Complex<f32> {
    fn of(re: f64, im: f64) -> Self {
        Complex::new(re as f32, im as f32)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits() as u64, self.im.to_bits() as u64)
    }
    fn name() -> &'static str {
        "c32"
    }
}
impl Cx for Complex<f64> {
    fn of(re: f64, im: f64) -> Self {
        Complex::new(re, im)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn name() -> &'static str {
        "c64"
    }
}

/// Deterministic complex fill (re/im each in ~`[-2, 2)`).
fn fill<T: Cx>(rng: &mut Rng, n: usize) -> Vec<T> {
    (0..n)
        .map(|_| T::of(rng.unit() * 2.0, rng.unit() * 2.0))
        .collect()
}

/// Drive `gemm_cplx_fused` and its `gemm_cplx`-then-bias oracle over a chosen conj combination,
/// bias kind, `beta`, and C layout (column- or row-major, the latter forcing the orientation swap
/// that flips the bias axis). Assert **bitwise** equality (re/im `to_bits`) at every element. A/B
/// are column-major; the reference reads back plain `gemm_cplx`'s output and adds the same bias
/// term with the same complex `+`, so any divergence in the fused store or its post-pass is caught.
#[allow(clippy::too_many_arguments)]
fn check_fused_bitwise<T: Cx>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    ca: bool,
    cb: bool,
    bias_kind: u8, // 0 none, 1 per-row, 2 per-col
    beta: T,
    row_major_c: bool,
    tag: &str,
) {
    let a = fill::<T>(rng, m * k);
    let b = fill::<T>(rng, k * n);
    let c0 = fill::<T>(rng, m * n);
    let bias_row = fill::<T>(rng, m);
    let bias_col = fill::<T>(rng, n);
    let alpha = T::of(1.1, -0.4);
    let (rsc, csc) = if row_major_c {
        (n as isize, 1isize)
    } else {
        (1isize, m as isize)
    };

    // --- fused ---
    let mut c_fused = c0.clone();
    {
        let mut ws = Workspace::new();
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_fused, m, n, rsc, csc);
        let bias = match bias_kind {
            1 => Some(Bias::PerRow(&bias_row)),
            2 => Some(Bias::PerCol(&bias_col)),
            _ => None,
        };
        gemm_cplx_fused_with(
            &mut ws,
            alpha,
            ar,
            ca,
            br,
            cb,
            beta,
            cm,
            bias,
            Parallelism::Serial,
        );
    }

    // --- oracle: plain gemm_cplx then element-wise bias add (user frame) ---
    let mut c_ref = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_ref, m, n, rsc, csc);
        gemm_cplx(alpha, ar, ca, br, cb, beta, cm, Parallelism::Serial);
    }
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let bt = match bias_kind {
                1 => Some(bias_row[i]),
                2 => Some(bias_col[j]),
                _ => None,
            };
            if let Some(bt) = bt {
                c_ref[idx] = c_ref[idx] + bt;
            }
        }
    }

    // --- bitwise compare (re + im) ---
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            assert_eq!(
                c_fused[idx].bits(),
                c_ref[idx].bits(),
                "{} {tag}: fused != gemm_cplx-then-bias at ({i},{j}) \
                 [m={m} k={k} n={n} ca={ca} cb={cb} bias_kind={bias_kind} row_major_c={row_major_c}]",
                T::name(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// a. general driver: every conj combo × bias kind × beta × C layout, bitwise
// ---------------------------------------------------------------------------

fn cplx_fused_bitwise_for<T: Cx>() {
    let mut rng = Rng::new(0x00CF_15ED);
    let (m, k, n) = (48, 33, 29);
    for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
        for bias_kind in 0u8..=2 {
            for &beta in &[T::of(0.0, 0.0), T::of(1.0, 0.0), T::of(0.5, 0.7)] {
                for row_major_c in [false, true] {
                    check_fused_bitwise::<T>(
                        &mut rng,
                        m,
                        k,
                        n,
                        ca,
                        cb,
                        bias_kind,
                        beta,
                        row_major_c,
                        "bitwise",
                    );
                }
            }
        }
    }
}

#[test]
fn cplx_fused_bitwise() {
    cplx_fused_bitwise_for::<Complex<f32>>();
    cplx_fused_bitwise_for::<Complex<f64>>();
}

// ---------------------------------------------------------------------------
// b. multi depth-panel: proves the last_k gating (intermediate partials untouched)
// ---------------------------------------------------------------------------

/// `k = 2048` with the default `KC = 512` splits the contraction into ~4 depth panels (complex is
/// `OUT_IS_ACC = true`, so the driver re-reads C between panels). The bias must fire exactly once,
/// on the completed sum — intermediate panels keep their raw partials. Bitwise vs `gemm_cplx`-then-add.
#[test]
fn cplx_fused_multi_panel() {
    fn go<T: Cx>() {
        let mut rng = Rng::new(0x33_5A_11);
        let (m, k, n) = (12, 2048, 10);
        for &(ca, cb) in &[(false, false), (true, true)] {
            for bias_kind in 1u8..=2 {
                check_fused_bitwise::<T>(
                    &mut rng,
                    m,
                    k,
                    n,
                    ca,
                    cb,
                    bias_kind,
                    T::of(0.5, -0.3),
                    false,
                    "multi_panel",
                );
            }
        }
    }
    go::<Complex<f32>>();
    go::<Complex<f64>>();
}

// ---------------------------------------------------------------------------
// c. serial == parallel bit-identity through the fused route
// ---------------------------------------------------------------------------

fn check_par_eq<T: Cx>(rng: &mut Rng, m: usize, k: usize, n: usize, ca: bool, cb: bool) {
    let a = fill::<T>(rng, m * k);
    let b = fill::<T>(rng, k * n);
    let c0 = fill::<T>(rng, m * n);
    let bias = fill::<T>(rng, m);
    let alpha = T::of(1.1, -0.4);
    let beta = T::of(0.5, 0.7);

    let run = |par: Parallelism| -> Vec<T> {
        let mut c = c0.clone();
        let mut ws = Workspace::new();
        let ar = MatRef::from_col_major(&a, m, k);
        let br = MatRef::from_col_major(&b, k, n);
        let cm = MatMut::from_col_major(&mut c, m, n);
        gemm_cplx_fused_with(
            &mut ws,
            alpha,
            ar,
            ca,
            br,
            cb,
            beta,
            cm,
            Some(Bias::PerRow(&bias)),
            par,
        );
        c
    };

    let cs = run(Parallelism::Serial);
    let cp = run(Parallelism::Rayon(4));
    for idx in 0..m * n {
        assert_eq!(
            cs[idx].bits(),
            cp[idx].bits(),
            "{}: serial != parallel at {idx} (ca={ca} cb={cb})",
            T::name(),
        );
    }
}

#[test]
fn cplx_fused_parallel_bitwise() {
    fn go<T: Cx>() {
        let mut rng = Rng::new(0x9A_4A_C1);
        for &(ca, cb) in &[(false, false), (true, true)] {
            check_par_eq::<T>(&mut rng, 200, 130, 175, ca, cb);
        }
    }
    go::<Complex<f32>>();
    go::<Complex<f64>>();
}

// ---------------------------------------------------------------------------
// d. bias None delegates to gemm_cplx bitwise (zero-cost identity)
// ---------------------------------------------------------------------------

#[test]
fn cplx_fused_identity_delegates() {
    fn go<T: Cx>() {
        let mut rng = Rng::new(0x1D_C0_DE);
        let (m, k, n) = (40, 33, 28);
        for &(ca, cb) in &[(false, false), (true, false), (true, true)] {
            let a = fill::<T>(&mut rng, m * k);
            let b = fill::<T>(&mut rng, k * n);
            let c0 = fill::<T>(&mut rng, m * n);
            let (alpha, beta) = (T::of(1.1, -0.3), T::of(0.5, 0.7));

            let mut c_fused = c0.clone();
            {
                let mut ws = Workspace::new();
                let ar = MatRef::from_col_major(&a, m, k);
                let br = MatRef::from_col_major(&b, k, n);
                let cm = MatMut::from_col_major(&mut c_fused, m, n);
                gemm_cplx_fused_with(
                    &mut ws,
                    alpha,
                    ar,
                    ca,
                    br,
                    cb,
                    beta,
                    cm,
                    None,
                    Parallelism::Serial,
                );
            }
            let mut c_ref = c0.clone();
            {
                let ar = MatRef::from_col_major(&a, m, k);
                let br = MatRef::from_col_major(&b, k, n);
                let cm = MatMut::from_col_major(&mut c_ref, m, n);
                gemm_cplx(alpha, ar, ca, br, cb, beta, cm, Parallelism::Serial);
            }
            for idx in 0..m * n {
                assert_eq!(
                    c_fused[idx].bits(),
                    c_ref[idx].bits(),
                    "{}: bias-None fused != gemm_cplx at {idx} (ca={ca} cb={cb})",
                    T::name(),
                );
            }
        }
    }
    go::<Complex<f32>>();
    go::<Complex<f64>>();
}

// ---------------------------------------------------------------------------
// e. degenerate (k == 0 / alpha == 0) with bias: C <- beta·C + bias
// ---------------------------------------------------------------------------

/// The `A·B` term vanishes (`k == 0` or `alpha == 0`): `C <- beta·C + bias`, in the user frame.
/// conj is set `true` on both operands on purpose — with no product to conjugate it must be
/// ignored. Compared against a scalar model, bitwise. `beta` is `{0, 1, other}`, exactly the
/// special-cases of the engine's degenerate path.
#[allow(clippy::too_many_arguments)]
fn check_degenerate<T: Cx>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    bias_kind: u8,
    beta: T,
    tag: &str,
) {
    let a = fill::<T>(rng, m * k);
    let b = fill::<T>(rng, k * n);
    let c0 = fill::<T>(rng, m * n);
    let bias_row = fill::<T>(rng, m);
    let bias_col = fill::<T>(rng, n);

    let mut c = c0.clone();
    {
        let mut ws = Workspace::new();
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::from_col_major(&mut c, m, n);
        let bias = match bias_kind {
            1 => Some(Bias::PerRow(&bias_row)),
            2 => Some(Bias::PerCol(&bias_col)),
            _ => None,
        };
        gemm_cplx_fused_with(
            &mut ws,
            alpha,
            ar,
            true,
            br,
            true,
            beta,
            cm,
            bias,
            Parallelism::Serial,
        );
    }

    // Scalar model (col-major): mirrors the engine's `beta ∈ {0, 1, other}` special-casing exactly,
    // then the bias add.
    let zero = T::of(0.0, 0.0);
    let one = T::of(1.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            let idx = j * m + i;
            let base = if beta == zero {
                zero
            } else if beta == one {
                c0[idx]
            } else {
                beta * c0[idx]
            };
            let want = match bias_kind {
                1 => base + bias_row[i],
                2 => base + bias_col[j],
                _ => base,
            };
            assert_eq!(
                c[idx].bits(),
                want.bits(),
                "{} {tag}: degenerate mismatch at ({i},{j}) [beta_bits={:?} bias_kind={bias_kind}]",
                T::name(),
                beta.bits(),
            );
        }
    }
}

#[test]
fn cplx_fused_degenerate() {
    fn go<T: Cx>() {
        let mut rng = Rng::new(0xDE_9E_11);
        let (m, n) = (20, 14);
        for bias_kind in 1u8..=2 {
            for &beta in &[T::of(0.0, 0.0), T::of(1.0, 0.0), T::of(0.5, 0.7)] {
                // k == 0 (alpha nonzero): the A·B term vanishes structurally.
                check_degenerate::<T>(&mut rng, m, 0, n, T::of(1.1, -0.4), bias_kind, beta, "k0");
                // alpha == 0 (k != 0): the scale-only path.
                check_degenerate::<T>(
                    &mut rng,
                    m,
                    7,
                    n,
                    T::of(0.0, 0.0),
                    bias_kind,
                    beta,
                    "alpha0",
                );
            }
        }
    }
    go::<Complex<f32>>();
    go::<Complex<f64>>();
}

// ---------------------------------------------------------------------------
// f. validation panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "PerRow bias length")]
fn cplx_fused_bias_row_wrong_len() {
    let (m, k, n) = (8, 5, 6);
    let a = vec![Complex::new(0.0f32, 0.0); m * k];
    let b = vec![Complex::new(0.0f32, 0.0); k * n];
    let mut c = vec![Complex::new(0.0f32, 0.0); m * n];
    let bias = vec![Complex::new(0.0f32, 0.0); m + 1]; // should be m (PerRow == A.rows)
    let mut ws = Workspace::new();
    gemm_cplx_fused_with(
        &mut ws,
        Complex::new(1.0f32, 0.0),
        MatRef::from_col_major(&a, m, k),
        false,
        MatRef::from_col_major(&b, k, n),
        false,
        Complex::new(0.0f32, 0.0),
        MatMut::from_col_major(&mut c, m, n),
        Some(Bias::PerRow(&bias)),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "PerCol bias length")]
fn cplx_fused_bias_col_wrong_len() {
    let (m, k, n) = (8, 5, 6);
    let a = vec![Complex::new(0.0f32, 0.0); m * k];
    let b = vec![Complex::new(0.0f32, 0.0); k * n];
    let mut c = vec![Complex::new(0.0f32, 0.0); m * n];
    let bias = vec![Complex::new(0.0f32, 0.0); n + 1]; // should be n (PerCol == B.cols)
    let mut ws = Workspace::new();
    gemm_cplx_fused_with(
        &mut ws,
        Complex::new(1.0f32, 0.0),
        MatRef::from_col_major(&a, m, k),
        false,
        MatRef::from_col_major(&b, k, n),
        false,
        Complex::new(0.0f32, 0.0),
        MatMut::from_col_major(&mut c, m, n),
        Some(Bias::PerCol(&bias)),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "bias slice overlaps C")]
fn cplx_fused_bias_overlaps_c() {
    let (m, k, n) = (4, 4, 4);
    let a = vec![Complex::new(1.0f32, 0.0); m * k];
    let b = vec![Complex::new(1.0f32, 0.0); k * n];
    let mut buf = vec![Complex::new(0.0f32, 0.0); m * n];
    // A bias slice aliasing C's storage. It is raw-derived (its lifetime is not tied to `buf`), so
    // `&mut buf` still type-checks; the overlap check panics before any element is read or written.
    let bias: &[Complex<f32>] = unsafe { core::slice::from_raw_parts(buf.as_ptr(), m) };
    let mut ws = Workspace::new();
    gemm_cplx_fused_with(
        &mut ws,
        Complex::new(1.0f32, 0.0),
        MatRef::from_col_major(&a, m, k),
        false,
        MatRef::from_col_major(&b, k, n),
        false,
        Complex::new(0.0f32, 0.0),
        MatMut::from_col_major(&mut buf, m, n),
        Some(Bias::PerRow(bias)),
        Parallelism::Serial,
    );
}
