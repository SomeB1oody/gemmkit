//! Correctness suite: numerical accuracy, full shape / layout / alpha-beta
//! coverage, parallel==serial bit-identity, per-ISA kernels, gemv, and the safe
//! API's panic guarantees.

#![allow(clippy::too_many_arguments)]

use gemmkit::driver;
use gemmkit::kernel::FloatGemm;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
use gemmkit::simd::ScalarTok;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::{Avx512, Fma};
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_unchecked};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Trait letting the harness be generic over f32/f64.
trait Elem: gemmkit::GemmScalar {
    const EPS: f64;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}
impl Elem for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}
impl Elem for f64 {
    const EPS: f64 = f64::EPSILON;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64) -> Self {
        x
    }
}
// Narrow types accumulate in f32 and round outputs to 16 bits, so their `EPS` is the
// 16-bit machine epsilon (f16 ≈ 9.8e-4, bf16 ≈ 7.8e-3) — the dominant error is the
// final round, the f32 accumulation being far more accurate.
#[cfg(feature = "half")]
impl Elem for gemmkit::f16 {
    const EPS: f64 = 9.765625e-4; // 2^-10
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(x: f64) -> Self {
        gemmkit::f16::from_f64(x)
    }
}
#[cfg(feature = "half")]
impl Elem for gemmkit::bf16 {
    const EPS: f64 = 7.8125e-3; // 2^-7
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(x: f64) -> Self {
        gemmkit::bf16::from_f64(x)
    }
}

/// Deterministic pseudo-random fill in [-1, 1).
fn rand_vec<T: Elem>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            T::from_f64(2.0 * u - 1.0)
        })
        .collect()
}

/// Logical matrix in row-major order plus its dimensions, for building views.
struct Mat<T> {
    v: Vec<T>,
    rows: usize,
    cols: usize,
}
impl<T: Elem> Mat<T> {
    fn rand(rows: usize, cols: usize, seed: u64) -> Self {
        Mat {
            v: rand_vec(rows * cols, seed),
            rows,
            cols,
        }
    }
    fn at(&self, i: usize, j: usize) -> T {
        self.v[i * self.cols + j]
    }
}

#[derive(Copy, Clone, Debug)]
enum Layout {
    Row,
    Col,
    /// Padded leading dimension (general strides, both > 1).
    GeneralPad,
}

/// Build a backing buffer + (rs, cs) for `m`, presenting it in `layout`.
fn build_view<T: Elem>(m: &Mat<T>, layout: Layout) -> (Vec<T>, isize, isize) {
    let (r, c) = (m.rows, m.cols);
    match layout {
        Layout::Row => {
            let pad = 0;
            let rs = (c + pad) as isize;
            let mut buf = vec![T::from_f64(0.0); r * (c + pad)];
            for i in 0..r {
                for j in 0..c {
                    buf[i * (c + pad) + j] = m.at(i, j);
                }
            }
            (buf, rs, 1)
        }
        Layout::Col => {
            let cs = r as isize;
            let mut buf = vec![T::from_f64(0.0); r * c];
            for i in 0..r {
                for j in 0..c {
                    buf[j * r + i] = m.at(i, j);
                }
            }
            (buf, 1, cs)
        }
        Layout::GeneralPad => {
            // row-major with padded rows: rs = c+3, cs = 1 -> general but cs==1;
            // make cs=2 too by interleaving a dummy column.
            let cs = 2isize;
            let rs = (2 * c + 5) as isize;
            let total = r * (2 * c + 5);
            let mut buf = vec![T::from_f64(0.0); total];
            for i in 0..r {
                for j in 0..c {
                    buf[i * (2 * c + 5) + j * 2] = m.at(i, j);
                }
            }
            (buf, rs, cs)
        }
    }
}

/// f64 reference: `C <- beta*C0 + alpha*A*B`.
fn reference<T: Elem>(a: &Mat<T>, b: &Mat<T>, c0: &Mat<T>, alpha: f64, beta: f64) -> Vec<f64> {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a.at(i, p).to_f64() * b.at(p, j).to_f64();
            }
            let base = if beta == 0.0 {
                0.0
            } else {
                beta * c0.at(i, j).to_f64()
            };
            out[i * n + j] = base + alpha * acc;
        }
    }
    out
}

/// Relative Frobenius error gate: ||C - Cref|| / (||A||*||B|| + tiny) <= 8*k*eps.
fn assert_accurate<T: Elem>(
    got: &[T],
    got_rs: isize,
    got_cs: isize,
    m: usize,
    n: usize,
    cref: &[f64],
    a: &Mat<T>,
    b: &Mat<T>,
    k: usize,
    ctx: &str,
) {
    let norm = |it: &mut dyn Iterator<Item = f64>| -> f64 { it.map(|x| x * x).sum::<f64>().sqrt() };
    let na = norm(&mut a.v.iter().map(|x| x.to_f64()));
    let nb = norm(&mut b.v.iter().map(|x| x.to_f64()));
    let mut diff2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let g = got[(i as isize * got_rs + j as isize * got_cs) as usize].to_f64();
            let r = cref[i * n + j];
            assert!(g.is_finite(), "{ctx}: non-finite output at ({i},{j})");
            let d = g - r;
            diff2 += d * d;
        }
    }
    let rel = diff2.sqrt() / (na * nb + 1e-30);
    let tol = 8.0 * (k.max(1) as f64) * T::EPS;
    assert!(
        rel <= tol,
        "{ctx}: relative error {rel:.3e} > tol {tol:.3e} (m={m},k={k},n={n})"
    );
}

// ---------------------------------------------------------------------------
// main correctness: shapes x layouts x alpha/beta via the public dispatched API
// ---------------------------------------------------------------------------

fn run_case<T: Elem>(
    m: usize,
    k: usize,
    n: usize,
    la: Layout,
    lb: Layout,
    lc: Layout,
    alpha: T,
    beta: T,
    par: Parallelism,
) {
    let a = Mat::<T>::rand(m, k, 0x1111 + (m * 7 + k * 13 + n) as u64);
    let b = Mat::<T>::rand(k, n, 0x2222 + (m + k * 5 + n * 11) as u64);
    let c0 = Mat::<T>::rand(m, n, 0x3333 + (m * 3 + k + n * 2) as u64);

    let (abuf, rsa, csa) = build_view(&a, la);
    let (bbuf, rsb, csb) = build_view(&b, lb);
    let (mut cbuf, rsc, csc) = build_view(&c0, lc);

    let cref = reference(&a, &b, &c0, alpha.to_f64(), beta.to_f64());

    gemm(
        alpha,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        beta,
        MatMut::new(&mut cbuf, m, n, rsc, csc),
        par,
    );

    let ctx = format!(
        "T={} {m}x{k}x{n} la={la:?} lb={lb:?} lc={lc:?} a={} b={} par={par:?}",
        core::any::type_name::<T>(),
        alpha.to_f64(),
        beta.to_f64()
    );
    assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, &ctx);
}

fn dims() -> Vec<(usize, usize, usize)> {
    // Edge values around the AVX-512 f32 tile (MR=32, NR=12) and blocking.
    let vals = [
        0usize, 1, 2, 5, 11, 12, 13, 16, 31, 32, 33, 48, 64, 100, 257,
    ];
    let mut out = Vec::new();
    // A representative cross-section (full cross product is huge).
    for &m in &vals {
        for &k in &[1usize, 2, 7, 32, 65] {
            for &n in &[1usize, 11, 12, 13, 64] {
                out.push((m, k, n));
            }
        }
    }
    // A few big squares.
    for &s in &[128usize, 200, 384] {
        out.push((s, s, s));
    }
    out
}

#[test]
fn correctness_f32_layouts() {
    for (m, k, n) in dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            run_case::<f32>(
                m,
                k,
                n,
                Layout::Row,
                Layout::Col,
                lc,
                1.0,
                0.0,
                Parallelism::Serial,
            );
        }
    }
}

#[test]
fn correctness_f64_layouts() {
    for (m, k, n) in dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            run_case::<f64>(
                m,
                k,
                n,
                Layout::Col,
                Layout::Row,
                lc,
                1.0,
                0.0,
                Parallelism::Serial,
            );
        }
    }
}

/// Shapes for the mixed-precision (`f16`/`bf16`) accuracy and bit-identity tests.
#[cfg(feature = "half")]
fn mixed_dims() -> [(usize, usize, usize); 9] {
    [
        (1, 1, 1),
        (3, 4, 5),
        (16, 8, 7),
        (32, 32, 32),
        (33, 17, 19),
        (40, 33, 28),
        (64, 80, 48),
        (65, 64, 64),
        (128, 96, 112),
    ]
}

#[cfg(feature = "half")]
#[test]
fn correctness_f16_layouts() {
    for (m, k, n) in mixed_dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            for &(al, be) in &[(1.0f64, 0.0), (1.0, 1.0), (0.75, -0.5)] {
                run_case::<gemmkit::f16>(
                    m,
                    k,
                    n,
                    Layout::Row,
                    Layout::Col,
                    lc,
                    gemmkit::f16::from_f64(al),
                    gemmkit::f16::from_f64(be),
                    Parallelism::Serial,
                );
            }
        }
    }
}

#[cfg(feature = "half")]
#[test]
fn correctness_bf16_layouts() {
    for (m, k, n) in mixed_dims() {
        for &lc in &[Layout::Row, Layout::Col] {
            for &(al, be) in &[(1.0f64, 0.0), (1.0, 1.0), (0.75, -0.5)] {
                run_case::<gemmkit::bf16>(
                    m,
                    k,
                    n,
                    Layout::Col,
                    Layout::Row,
                    lc,
                    gemmkit::bf16::from_f64(al),
                    gemmkit::bf16::from_f64(be),
                    Parallelism::Serial,
                );
            }
        }
    }
}

/// Mixed-precision serial == parallel **bit-identity** across thread counts (the
/// hard determinism invariant must hold for the new types too — narrowing is a pure
/// per-position function of the f32 result, and the blocking is thread-independent).
#[cfg(feature = "half")]
#[test]
fn parallel_equals_serial_mixed() {
    fn check<T: Elem>(la: Layout) {
        for (m, k, n) in [(200, 130, 175), (256, 64, 200), (384, 96, 320)] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                let a = Mat::<T>::rand(m, k, 0xF16 + m as u64);
                let b = Mat::<T>::rand(k, n, 0xBF + n as u64);
                let c0 = Mat::<T>::rand(m, n, 0xCD + k as u64);
                let (abuf, rsa, csa) = build_view(&a, la);
                let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
                let (al, be) = (T::from_f64(al), T::from_f64(be));

                let mut c_ser = cbase.clone();
                gemm(
                    al,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be,
                    MatMut::new(&mut c_ser, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                for t in [2usize, 4, 8, 16] {
                    let mut c_par = cbase.clone();
                    gemm(
                        al,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        MatRef::new(&bbuf, k, n, rsb, csb),
                        be,
                        MatMut::new(&mut c_par, m, n, rsc, csc),
                        Parallelism::Rayon(t),
                    );
                    assert!(
                        c_ser
                            .iter()
                            .zip(&c_par)
                            .all(|(a, b)| a.to_f64().to_bits() == b.to_f64().to_bits()),
                        "mixed serial != parallel({t}) for {m}x{k}x{n}"
                    );
                }
            }
        }
    }
    check::<gemmkit::f16>(Layout::Row);
    check::<gemmkit::bf16>(Layout::Col);
}

/// Cross-check `f16` against the `gemm` crate (the ecosystem oracle, which also
/// accumulates `f16` in `f32`): the two must agree to a tight `f16` tolerance.
/// `gemm`'s `f16` *is* `half::f16` *is* `gemmkit::f16`, so the comparison is direct.
/// Gated out of Miri (the `gemm` dev-dep is `cfg(not(miri))`).
#[test]
#[cfg(all(not(miri), feature = "half"))]
fn mixed_f16_matches_gemm_crate() {
    // Includes a large-k case (k > the f32 kc blocking ≈ 512) to exercise the
    // cross-depth-panel accumulation, where the running sum round-trips through the
    // narrow C between kc panels.
    for (m, k, n) in [(64, 48, 40), (96, 65, 72), (33, 17, 19), (64, 2048, 64)] {
        let a = Mat::<gemmkit::f16>::rand(m, k, 0x16A + m as u64);
        let b = Mat::<gemmkit::f16>::rand(k, n, 0x16B + n as u64);
        // Column-major buffers (gemm's preferred orientation), zero beta.
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let mut c_kit = vec![gemmkit::f16::from_f64(0.0); m * n];
        let mut c_gemm = vec![gemmkit::f16::from_f64(0.0); m * n];

        gemm(
            gemmkit::f16::from_f64(1.0),
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            gemmkit::f16::from_f64(0.0),
            MatMut::from_col_major(&mut c_kit, m, n),
            Parallelism::Serial,
        );
        // gemm crate: column-major operands → (cs = leading dim, rs = 1), matching
        // the bench harness; read_dst=false (beta=0).
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c_gemm.as_mut_ptr(),
                m as isize,
                1,
                false,
                abuf.as_ptr(),
                m as isize,
                1,
                bbuf.as_ptr(),
                k as isize,
                1,
                gemmkit::f16::from_f64(0.0),
                gemmkit::f16::from_f64(1.0),
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        }
        // Both accumulate in f32 then round to f16; allow a few f16 ULPs of slack
        // (the accumulation order differs). `assert_accurate` wants a *row-major*
        // reference, so transpose the column-major `c_gemm` into one.
        let mut cref = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                cref[i * n + j] = c_gemm[i + j * m].to_f64();
            }
        }
        assert_accurate(
            &c_kit,
            1,
            m as isize,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "f16 vs gemm crate",
        );
    }
}

#[test]
fn correctness_general_strides() {
    for (m, k, n) in [(7, 9, 5), (32, 32, 32), (33, 17, 19), (64, 64, 64)] {
        run_case::<f32>(
            m,
            k,
            n,
            Layout::GeneralPad,
            Layout::GeneralPad,
            Layout::GeneralPad,
            1.0,
            0.0,
            Parallelism::Serial,
        );
        run_case::<f64>(
            m,
            k,
            n,
            Layout::GeneralPad,
            Layout::Row,
            Layout::Col,
            1.0,
            0.0,
            Parallelism::Serial,
        );
    }
}

#[test]
fn correctness_alpha_beta() {
    let combos = [
        (0.0f64, 0.0),
        (0.0, 1.0),
        (0.0, 2.5),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, -1.5),
        (2.0, 0.0),
        (-0.5, 3.0),
    ];
    for (m, k, n) in [(5, 6, 7), (32, 40, 24), (64, 31, 48)] {
        for &(al, be) in &combos {
            run_case::<f32>(
                m,
                k,
                n,
                Layout::Row,
                Layout::Row,
                Layout::Row,
                al as f32,
                be as f32,
                Parallelism::Serial,
            );
            run_case::<f64>(
                m,
                k,
                n,
                Layout::Col,
                Layout::Col,
                Layout::Col,
                al,
                be,
                Parallelism::Serial,
            );
        }
    }
}

/// beta==0 must not read C — prove it by seeding C with NaN.
#[test]
fn beta_zero_does_not_read_c() {
    let (m, k, n) = (40, 33, 28);
    let a = Mat::<f32>::rand(m, k, 7);
    let b = Mat::<f32>::rand(k, n, 9);
    let cref = reference(
        &a,
        &b,
        &Mat {
            v: vec![0.0; m * n],
            rows: m,
            cols: n,
        },
        1.0,
        0.0,
    );
    let (abuf, rsa, csa) = build_view(&a, Layout::Col);
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let mut cbuf = vec![f32::NAN; m * n];
    gemm(
        1.0,
        MatRef::new(&abuf, m, k, rsa, csa),
        MatRef::new(&bbuf, k, n, rsb, csb),
        0.0,
        MatMut::from_col_major(&mut cbuf, m, n),
        Parallelism::Serial,
    );
    assert_accurate(&cbuf, 1, m as isize, m, n, &cref, &a, &b, k, "beta=0 NaN C");
}

/// Serial and parallel runs must be bit-identical.
#[test]
fn parallel_equals_serial_bit_identical() {
    for (m, k, n) in [
        (64, 64, 64),
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
    ] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0xABC + m as u64);
            let b = Mat::<f32>::rand(k, n, 0xDEF + n as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x123 + k as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Col);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

            let mut c_serial = cbase.clone();
            let mut c_par = cbase.clone();
            gemm(
                al as f32,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                be as f32,
                MatMut::new(&mut c_serial, m, n, rsc, csc),
                Parallelism::Serial,
            );
            for threads in [2usize, 4, 8, 16] {
                c_par.copy_from_slice(&cbase);
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_par, m, n, rsc, csc),
                    Parallelism::Rayon(threads),
                );
                assert_eq!(
                    c_serial, c_par,
                    "serial != parallel({threads}) for {m}x{k}x{n} a={al} b={be}"
                );
            }
        }
    }
}

/// Bit-identity with a **row-major A** (`rsa != 1`), which forces per-row-block LHS
/// packing and so exercises the dynamic scheduler's whole-row-block ("packed")
/// grain path under multiple threads — distinct from the column-major case above.
/// Sizes are chosen so the row-block count straddles the thread count (so both the
/// `grain = n_nt` branch and its fine-grain fallback run).
#[test]
fn parallel_equals_serial_row_major_a() {
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
            let a = Mat::<f32>::rand(m, k, 0xA11 + m as u64);
            let b = Mat::<f32>::rand(k, n, 0xB22 + n as u64);
            let c0 = Mat::<f32>::rand(m, n, 0xC33 + k as u64);
            let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k != 1 → packs A
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

            let mut c_serial = cbase.clone();
            let mut c_par = cbase.clone();
            gemm(
                al as f32,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                be as f32,
                MatMut::new(&mut c_serial, m, n, rsc, csc),
                Parallelism::Serial,
            );
            for threads in [2usize, 4, 8, 16] {
                c_par.copy_from_slice(&cbase);
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_par, m, n, rsc, csc),
                    Parallelism::Rayon(threads),
                );
                assert_eq!(
                    c_serial, c_par,
                    "row-major A: serial != parallel({threads}) for {m}x{k}x{n} a={al} b={be}"
                );
            }
        }
    }
}

/// Prepacked-RHS must be **bit-identical** to a plain `gemm()` on the same
/// inputs, for any thread count and any B layout (C column-major = the supported
/// no-swap orientation). This is the determinism gate for the reuse path: packing
/// only rearranges B's values, so the microkernel does the identical fused FMAs in
/// the identical order.
#[test]
fn prepack_equals_gemm() {
    // All shapes are non-both-tiny (not m<=64 && n<=64), the supported regime.
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // just above the tiny shortcut on m
        (64, 64, 65), // just above on n
        (300, 1, 256),
        (40, 200, 300),
    ] {
        for &lb in &[Layout::Col, Layout::Row] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3), (-0.5, 2.0)] {
                let a = Mat::<f32>::rand(m, k, 0x5A + (m * 7 + n) as u64);
                let b = Mat::<f32>::rand(k, n, 0x6B + (n * 3 + k) as u64);
                let c0 = Mat::<f32>::rand(m, n, 0x7C + (k + m) as u64);
                let (abuf, rsa, csa) = build_view(&a, Layout::Col);
                let (bbuf, rsb, csb) = build_view(&b, lb);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );

                let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
                assert_eq!(packed.rows(), k);
                assert_eq!(packed.cols(), n);
                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(4),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_b(
                        al as f32,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        &packed,
                        be as f32,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack != gemm for {m}x{k}x{n} lb={lb:?} a={al} b={be} par={par:?}"
                    );
                }
            }
        }
    }
}

/// Mixed-precision prepacked-RHS must be **bit-identical** to plain `gemm()` for
/// the narrow types too: the prepack blocks with the accumulator size and the same
/// `kc = k` the driver uses, so packed and unpacked never diverge. Includes a
/// `k > 512` cross-panel case.
#[cfg(feature = "half")]
#[test]
fn prepack_equals_gemm_mixed() {
    fn check<T: Elem>() {
        for (m, k, n) in [(200, 130, 175), (96, 65, 72), (128, 1024, 96)] {
            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3)] {
                let a = Mat::<T>::rand(m, k, 0x9A + (m * 3 + n) as u64);
                let b = Mat::<T>::rand(k, n, 0x9B + (n + k) as u64);
                let c0 = Mat::<T>::rand(m, n, 0x9C + (k + m) as u64);
                let (abuf, rsa, csa) = build_view(&a, Layout::Col);
                let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
                let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
                let (al, be) = (T::from_f64(al), T::from_f64(be));

                let mut c_ref = cbase.clone();
                gemm(
                    al,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );
                let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
                for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_b(
                        al,
                        MatRef::new(&abuf, m, k, rsa, csa),
                        &packed,
                        be,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert!(
                        c_ref
                            .iter()
                            .zip(&c_pk)
                            .all(|(a, b)| a.to_f64().to_bits() == b.to_f64().to_bits()),
                        "mixed prepack != gemm for {m}x{k}x{n} par={par:?}"
                    );
                }
            }
        }
    }
    check::<gemmkit::f16>();
    check::<gemmkit::bf16>();
}

/// f64 prepacked path is bit-identical too (exercises the f64 tile + the packed
/// geometry for a second element type).
#[test]
fn prepack_equals_gemm_f64() {
    for (m, k, n) in [(160, 96, 208), (96, 65, 65)] {
        let a = Mat::<f64>::rand(m, k, 0x1234);
        let b = Mat::<f64>::rand(k, n, 0x5678);
        let c0 = Mat::<f64>::rand(m, n, 0x9abc);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

        let mut c_ref = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
        let mut c_pk = cbase.clone();
        gemmkit::gemm_packed_b(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            &packed,
            -0.3,
            MatMut::new(&mut c_pk, m, n, rsc, csc),
            Parallelism::Rayon(8),
        );
        assert_eq!(c_ref, c_pk, "f64 prepack != gemm for {m}x{k}x{n}");
    }
}

/// Both-tiny products (`m <= 64 && n <= 64`) still work via the prepacked path: it
/// uses the buffer's own blocking (which may round differently from plain gemm's
/// small-matrix shortcut), so we check *accuracy* against the f64 reference rather
/// than bit-identity to plain gemm — and the output must stay bit-identical across
/// thread counts. `(60, 600, 60)` exercises the `k > 512` case where the
/// general/tiny blocking diverges.
#[test]
fn prepack_both_tiny_accurate_and_deterministic() {
    for (m, k, n) in [(48, 40, 48), (60, 600, 60), (10, 9, 12)] {
        let a = Mat::<f32>::rand(m, k, 0x11 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0x22 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0x33 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);
        let cref = reference(&a, &b, &c0, 1.0, 0.5);

        let packed = gemmkit::prepack_rhs(MatRef::new(&bbuf, k, n, rsb, csb));
        let mut c_ser = cbase.clone();
        gemmkit::gemm_packed_b(
            1.0,
            MatRef::new(&abuf, m, k, rsa, csa),
            &packed,
            0.5,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        assert_accurate(
            &c_ser,
            rsc,
            csc,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "both-tiny prepack",
        );
        for threads in [2usize, 8] {
            let mut c_par = cbase.clone();
            gemmkit::gemm_packed_b(
                1.0,
                MatRef::new(&abuf, m, k, rsa, csa),
                &packed,
                0.5,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(threads),
            );
            assert_eq!(
                c_ser, c_par,
                "both-tiny prepack serial != parallel({threads}) for {m}x{k}x{n}"
            );
        }
    }
}

/// Prepacked-LHS (the mirror of `prepack_equals_gemm`): reusing a prepacked `A`
/// must be **bit-identical** to a plain `gemm()` for any thread count and any A
/// layout, with a **row-major-ish C** (the supported no-extra-swap orientation —
/// the engine drives the prepacked-A product transposed). Packing only rearranges
/// A's values, so the microkernel does the identical fused FMAs in the identical
/// order.
#[test]
fn prepack_lhs_equals_gemm() {
    for (m, k, n) in [
        (200, 130, 175),
        (384, 96, 320),
        (256, 257, 129),
        (65, 64, 64), // just above the tiny shortcut on m
        (64, 64, 65), // just above on n
        (300, 1, 256),
        (40, 200, 300),
    ] {
        for &la in &[Layout::Col, Layout::Row] {
            // A and its packed buffer depend only on the (shape, layout) — hoist the
            // pack above the alpha/beta loop so it happens once, not per combo.
            let a = Mat::<f32>::rand(m, k, 0x5A + (m * 7 + n) as u64);
            let b = Mat::<f32>::rand(k, n, 0x6B + (n * 3 + k) as u64);
            let c0 = Mat::<f32>::rand(m, n, 0x7C + (k + m) as u64);
            let (abuf, rsa, csa) = build_view(&a, la);
            let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
            // Row-major C is the supported orientation for the packed-LHS path.
            let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
            let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
            assert_eq!(packed.rows(), m);
            assert_eq!(packed.cols(), k);

            for &(al, be) in &[(1.0f64, 0.0), (0.7, 1.3), (-0.5, 2.0)] {
                let mut c_ref = cbase.clone();
                gemm(
                    al as f32,
                    MatRef::new(&abuf, m, k, rsa, csa),
                    MatRef::new(&bbuf, k, n, rsb, csb),
                    be as f32,
                    MatMut::new(&mut c_ref, m, n, rsc, csc),
                    Parallelism::Serial,
                );

                for par in [
                    Parallelism::Serial,
                    Parallelism::Rayon(2),
                    Parallelism::Rayon(4),
                    Parallelism::Rayon(8),
                ] {
                    let mut c_pk = cbase.clone();
                    gemmkit::gemm_packed_a(
                        al as f32,
                        &packed,
                        MatRef::new(&bbuf, k, n, rsb, csb),
                        be as f32,
                        MatMut::new(&mut c_pk, m, n, rsc, csc),
                        par,
                    );
                    assert_eq!(
                        c_ref, c_pk,
                        "prepack_lhs != gemm for {m}x{k}x{n} la={la:?} a={al} b={be} par={par:?}"
                    );
                }
            }
        }
    }
}

/// f64 prepacked-LHS path is bit-identical too (exercises the f64 tile + the packed
/// geometry for a second element type).
#[test]
fn prepack_lhs_equals_gemm_f64() {
    for (m, k, n) in [(160, 96, 208), (96, 65, 65)] {
        let a = Mat::<f64>::rand(m, k, 0x1234);
        let b = Mat::<f64>::rand(k, n, 0x5678);
        let c0 = Mat::<f64>::rand(m, n, 0x9abc);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Row);

        let mut c_ref = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_ref, m, n, rsc, csc),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
        let mut c_pk = cbase.clone();
        gemmkit::gemm_packed_a(
            0.9,
            &packed,
            MatRef::new(&bbuf, k, n, rsb, csb),
            -0.3,
            MatMut::new(&mut c_pk, m, n, rsc, csc),
            Parallelism::Rayon(8),
        );
        assert_eq!(c_ref, c_pk, "f64 prepack_lhs != gemm for {m}x{k}x{n}");
    }
}

/// Both-tiny products (`m <= 64 && n <= 64`) via the prepacked-LHS path: like the
/// RHS case it uses the buffer's own blocking, so check *accuracy* against the f64
/// reference rather than bit-identity to plain gemm — and the output must stay
/// bit-identical across thread counts.
#[test]
fn prepack_lhs_both_tiny_accurate_and_deterministic() {
    for (m, k, n) in [(48, 40, 48), (60, 600, 60), (10, 9, 12)] {
        let a = Mat::<f32>::rand(m, k, 0x11 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0x22 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0x33 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Row);
        let cref = reference(&a, &b, &c0, 1.0, 0.5);

        let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
        let mut c_ser = cbase.clone();
        gemmkit::gemm_packed_a(
            1.0,
            &packed,
            MatRef::new(&bbuf, k, n, rsb, csb),
            0.5,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        assert_accurate(
            &c_ser,
            rsc,
            csc,
            m,
            n,
            &cref,
            &a,
            &b,
            k,
            "both-tiny prepack_lhs",
        );
        for threads in [2usize, 8] {
            let mut c_par = cbase.clone();
            gemmkit::gemm_packed_a(
                1.0,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                0.5,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(threads),
            );
            assert_eq!(
                c_ser, c_par,
                "both-tiny prepack_lhs serial != parallel({threads}) for {m}x{k}x{n}"
            );
        }
    }
}

/// gemv shapes (`n == 1` and `m == 1`) through the prepacked-LHS path. Plain `gemm`
/// routes these to the dedicated gemv kernel, but `gemm_packed_a` runs them through
/// the general driver (the transpose maps a unit dimension onto a unit *driver*
/// dimension), so this checks **accuracy** against the f64 reference — and that the
/// row-major-ish C contract admits the natural vector layouts (a unit column/row is
/// addressed with `|csc| <= |rsc|`).
#[test]
fn prepack_lhs_gemv_accurate() {
    // (m, k, n, rsc, csc): n==1 column-vector C (rsc=1,csc=1) and m==1 row-vector C
    // (rsc=n,csc=1), both row-major-ish so the packed-LHS guard accepts them.
    for &(m, k, n, rsc, csc) in &[
        (64usize, 40, 1usize, 1isize, 1isize),
        (255, 129, 1, 1, 1),
        (1, 40, 64, 64, 1),
        (1, 100, 255, 255, 1),
    ] {
        let a = Mat::<f32>::rand(m, k, 0xAA + (m * 5 + k) as u64);
        let b = Mat::<f32>::rand(k, n, 0xBB + (n * 7 + k) as u64);
        let c0 = Mat::<f32>::rand(m, n, 0xCC + (m + n) as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Col);
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let mut cbuf = c0.v.clone(); // row-major m*n vector (rs=csc-major chosen above)
        let cref = reference(&a, &b, &c0, 1.3, -0.4);

        for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
            cbuf.copy_from_slice(&c0.v);
            let packed = gemmkit::prepack_lhs(MatRef::new(&abuf, m, k, rsa, csa));
            gemmkit::gemm_packed_a(
                1.3,
                &packed,
                MatRef::new(&bbuf, k, n, rsb, csb),
                -0.4,
                MatMut::new(&mut cbuf, m, n, rsc, csc),
                par,
            );
            assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, "prepack_lhs gemv");
        }
    }
}

/// A column-major-ish C is unsupported by the prepacked-LHS path (it would keep A
/// in the genuine LHS role); `gemm_packed_a` must reject it.
#[test]
#[should_panic(expected = "row-major-ish C")]
fn prepack_lhs_col_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = gemmkit::prepack_lhs(MatRef::from_col_major(&a, m, k));
    gemmkit::gemm_packed_a(
        1.0,
        &packed,
        MatRef::from_col_major(&b, k, n),
        0.0,
        MatMut::from_col_major(&mut c, m, n), // column-major C -> reject
        Parallelism::Serial,
    );
}

/// A row-major-ish C is unsupported by the prepacked path (it would swap A/B);
/// `gemm_packed_b` must reject it instead of silently computing the wrong thing.
#[test]
#[should_panic(expected = "column-major-ish C")]
fn prepack_row_major_c_panics() {
    let (m, k, n) = (100, 80, 120);
    let a = vec![0.0f32; m * k];
    let b = vec![0.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let packed = gemmkit::prepack_rhs(MatRef::from_col_major(&b, k, n));
    gemmkit::gemm_packed_b(
        1.0,
        MatRef::from_col_major(&a, m, k),
        &packed,
        0.0,
        MatMut::from_row_major(&mut c, m, n), // row-major C -> swap -> reject
        Parallelism::Serial,
    );
}

/// Shared-LHS A-pack: with the workload gate forced fully open, the shared
/// pre-pack path (one pack per row-block + indexed read) must stay bit-identical
/// to the serial per-worker path, for every thread count. These sizes sit below
/// the default gate, so this is the only coverage of the shared pre-pass; bit-
/// identity holds whether the gate is on or off, so forcing it cannot disturb
/// concurrently-running tests. Row-major A (`rsa != 1`) forces the packed path.
#[test]
fn shared_lhs_a_bit_identical() {
    let prev = gemmkit::tuning::shared_lhs_mnk();
    gemmkit::tuning::set_shared_lhs_mnk(1); // force shared-A on for any parallel run
    for (m, k, n) in [(200, 130, 175), (384, 96, 320), (256, 64, 200)] {
        let a = Mat::<f32>::rand(m, k, 0xA1 + m as u64);
        let b = Mat::<f32>::rand(k, n, 0xB2 + n as u64);
        let c0 = Mat::<f32>::rand(m, n, 0xC3 + k as u64);
        let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k != 1 -> packs A
        let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
        let (cbase, rsc, csc) = build_view(&c0, Layout::Col);

        let mut c_ser = cbase.clone();
        gemm(
            0.9,
            MatRef::new(&abuf, m, k, rsa, csa),
            MatRef::new(&bbuf, k, n, rsb, csb),
            0.4,
            MatMut::new(&mut c_ser, m, n, rsc, csc),
            Parallelism::Serial,
        );
        for t in [2usize, 4, 8, 16] {
            let mut c_par = cbase.clone();
            gemm(
                0.9,
                MatRef::new(&abuf, m, k, rsa, csa),
                MatRef::new(&bbuf, k, n, rsb, csb),
                0.4,
                MatMut::new(&mut c_par, m, n, rsc, csc),
                Parallelism::Rayon(t),
            );
            assert_eq!(
                c_ser, c_par,
                "shared-A: serial != parallel({t}) for {m}x{k}x{n}"
            );
        }
    }
    gemmkit::tuning::set_shared_lhs_mnk(prev);
}

/// Negative strides via the unchecked API (reversed-row view of A).
#[test]
fn negative_strides_unchecked() {
    let (m, k, n) = (12, 9, 7);
    let a = Mat::<f64>::rand(m, k, 5);
    let b = Mat::<f64>::rand(k, n, 6);
    let cref = reference(
        &a,
        &b,
        &Mat {
            v: vec![0.0; m * n],
            rows: m,
            cols: n,
        },
        1.0,
        0.0,
    );

    // A laid out row-major; present it with a *negative* row stride by pointing
    // at the last row and walking backwards. C is also stored that way.
    let (abuf, rsa, csa) = build_view(&a, Layout::Row); // rsa = k, csa = 1
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let mut cbuf = vec![0.0f64; m * n];

    unsafe {
        let a_last = abuf.as_ptr().add(((m - 1) as isize * rsa) as usize);
        let c_ptr = cbuf.as_mut_ptr();
        // Reversed A rows: base = last row, row stride = -rsa, but C in natural
        // order means C rows also reverse — instead reverse A's rows and B stays,
        // producing C with reversed rows; compare against reversed reference.
        gemm_unchecked(
            m,
            k,
            n,
            1.0,
            a_last,
            -rsa,
            csa,
            bbuf.as_ptr(),
            rsb,
            csb,
            0.0,
            c_ptr,
            n as isize,
            1,
            Parallelism::Serial,
        );
    }
    // gemm computed C[i,j] = sum_k A[m-1-i, k] * B[k,j]; compare to reversed ref.
    for i in 0..m {
        for j in 0..n {
            let got = cbuf[i * n + j];
            let exp = cref[(m - 1 - i) * n + j];
            assert!(
                (got - exp).abs() <= 1e-10 * (1.0 + exp.abs()),
                "neg stride mismatch"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// per-ISA kernels via the generic driver (column-major, no orientation needed)
// ---------------------------------------------------------------------------

/// Run `FloatGemm` through the driver with an explicit ISA token + tile, all
/// column-major (rsc==1), and check accuracy. Exercises each kernel directly.
fn driver_case<T, S, const MR_REG: usize, const NR: usize>(simd: S, m: usize, k: usize, n: usize)
where
    T: Elem + gemmkit::Float<Acc = T>,
    S: gemmkit::simd::SimdOps<T>,
{
    let a = Mat::<T>::rand(m, k, 0x55 + (m + k + n) as u64);
    let b = Mat::<T>::rand(k, n, 0x66 + (m * 2 + n) as u64);
    let c0 = Mat::<T>::rand(m, n, 0x77 + (k * 3) as u64);
    let (abuf, rsa, csa) = build_view(&a, Layout::Col);
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let (mut cbuf, rsc, csc) = build_view(&c0, Layout::Col);
    let alpha = T::from_f64(1.3);
    let beta = T::from_f64(-0.7);
    let cref = reference(&a, &b, &c0, alpha.to_f64(), beta.to_f64());

    let mut ws = Workspace::new();
    unsafe {
        driver::run::<FloatGemm<T>, S, MR_REG, NR>(
            simd,
            m,
            k,
            n,
            alpha,
            abuf.as_ptr(),
            rsa,
            csa,
            bbuf.as_ptr(),
            rsb,
            csb,
            beta,
            cbuf.as_mut_ptr(),
            rsc,
            csc,
            Parallelism::Serial,
            &mut ws,
        );
    }
    assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, "driver per-ISA");
}

fn isa_shapes() -> [(usize, usize, usize); 6] {
    [
        (1, 1, 1),
        (3, 4, 5),
        (32, 32, 32),
        (33, 17, 19),
        (64, 80, 48),
        (128, 96, 112),
    ]
}

#[test]
fn isa_scalar() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, ScalarTok, 4, 4>(ScalarTok, m, k, n);
        driver_case::<f64, ScalarTok, 4, 4>(ScalarTok, m, k, n);
    }
}

/// Miri (no SIMD intrinsics) over the scalar `SimdOps` + packing + scratch +
/// epilogue + gemv + driver. Small shapes keep Miri fast. Under Miri the runtime
/// feature detection reports nothing, so the dispatched `gemm` also takes the
/// scalar path.
#[test]
fn miri_scalar_path() {
    for (m, k, n) in [
        (5, 4, 6),
        (8, 8, 8),
        (3, 7, 2),
        (1, 5, 4),
        (6, 2, 1),
        (4, 4, 4),
    ] {
        driver_case::<f32, ScalarTok, 4, 4>(ScalarTok, m, k, n);
        driver_case::<f64, ScalarTok, 4, 4>(ScalarTok, m, k, n);
    }
    // Safe API end-to-end (partial tiles, beta != 0, general strides).
    run_case::<f32>(
        7,
        9,
        5,
        Layout::Row,
        Layout::Col,
        Layout::Row,
        1.3,
        0.5,
        Parallelism::Serial,
    );
    run_case::<f64>(
        6,
        6,
        6,
        Layout::GeneralPad,
        Layout::Row,
        Layout::Col,
        0.0,
        2.0,
        Parallelism::Serial,
    );
    // Mixed-precision (f16 / bf16) scalar path: widen-load, f32 accumulate, narrow
    // store, plus the strided copy-back epilogue and beta != 0 read of narrow C.
    #[cfg(feature = "half")]
    {
        run_case::<gemmkit::f16>(
            7,
            9,
            5,
            Layout::Row,
            Layout::Col,
            Layout::Row,
            gemmkit::f16::from_f64(1.0),
            gemmkit::f16::from_f64(0.5),
            Parallelism::Serial,
        );
        run_case::<gemmkit::bf16>(
            6,
            6,
            6,
            Layout::Col,
            Layout::Row,
            Layout::Col,
            gemmkit::bf16::from_f64(0.75),
            gemmkit::bf16::from_f64(-0.5),
            Parallelism::Serial,
        );
    }
    // Complex (c32) scalar path with conj-A: the conjugate-on-pack variant + the
    // scalar complex multiply and epilogue.
    #[cfg(feature = "complex")]
    {
        let (m, k, n) = (5usize, 4, 6);
        let a = rand_cplx::<gemmkit::c32>(m * k, 11);
        let b = rand_cplx::<gemmkit::c32>(k * n, 12);
        let c0 = rand_cplx::<gemmkit::c32>(m * n, 13);
        let (alpha, beta) = (
            gemmkit::Complex::new(1.0f32, 0.0),
            gemmkit::Complex::new(0.5f32, -0.25),
        );
        let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, true, false);
        let mut c = c0.clone();
        gemmkit::gemm_cplx(
            alpha,
            MatRef::from_col_major(&a, m, k),
            true,
            MatRef::from_col_major(&b, k, n),
            false,
            beta,
            MatMut::from_col_major(&mut c, m, n),
            Parallelism::Serial,
        );
        assert_cplx_accurate(&c, m, n, &cref, k, "miri complex conjA");
    }
    // Integer (i8 -> i32) scalar path: widen-load, i32 accumulate, partial-tile
    // copy-back, and the beta != 0 i32 read of C.
    #[cfg(feature = "int8")]
    {
        let (m, k, n) = (7usize, 9, 5);
        let a = rand_i8(m * k, 1);
        let b = rand_i8(k * n, 2);
        let c0: Vec<i32> = (0..m * n).map(|x| x as i32 % 4 - 2).collect();
        let cref = ref_i8(&a, &b, &c0, m, k, n, 3, -2);
        let mut c = c0.clone();
        gemmkit::gemm_i8(
            3,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_row_major(&b, k, n),
            -2,
            MatMut::from_row_major(&mut c, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c, cref, "miri: i8 mismatch");
    }
    // gemv shapes.
    run_case::<f32>(
        8,
        5,
        1,
        Layout::Col,
        Layout::Col,
        Layout::Col,
        1.0,
        0.0,
        Parallelism::Serial,
    );
    run_case::<f64>(
        1,
        5,
        8,
        Layout::Row,
        Layout::Row,
        Layout::Row,
        1.0,
        0.0,
        Parallelism::Serial,
    );
    // Prepacked-RHS path on the scalar engine: bit-identical to plain gemm.
    // Shape is not both-tiny (m > 64), so the prepacked geometry matches.
    {
        let (m, k, n) = (66usize, 4, 6);
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        let c0 = rand_vec::<f32>(m * n, 3);
        let mut c_ref = c0.clone();
        gemm(
            1.3,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.5,
            MatMut::from_col_major(&mut c_ref, m, n),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_rhs(MatRef::from_col_major(&b, k, n));
        let mut c_pk = c0.clone();
        gemmkit::gemm_packed_b(
            1.3,
            MatRef::from_col_major(&a, m, k),
            &packed,
            0.5,
            MatMut::from_col_major(&mut c_pk, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c_ref, c_pk, "miri: prepack != gemm");
    }
    // Prepacked-LHS path on the scalar engine: bit-identical to plain gemm.
    // Shape is not both-tiny (m > 64), and C is row-major (the supported
    // orientation), so the prepacked geometry matches plain gemm exactly.
    {
        let (m, k, n) = (66usize, 4, 6);
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        let c0 = rand_vec::<f32>(m * n, 3);
        let mut c_ref = c0.clone();
        gemm(
            1.3,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_row_major(&b, k, n),
            0.5,
            MatMut::from_row_major(&mut c_ref, m, n),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_lhs(MatRef::from_row_major(&a, m, k));
        let mut c_pk = c0.clone();
        gemmkit::gemm_packed_a(
            1.3,
            &packed,
            MatRef::from_row_major(&b, k, n),
            0.5,
            MatMut::from_row_major(&mut c_pk, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c_ref, c_pk, "miri: prepack_lhs != gemm");
    }
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn isa_fma() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("skipping FMA test: not supported");
        return;
    }
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Fma, 2, 6>(Fma, m, k, n);
        driver_case::<f64, Fma, 2, 6>(Fma, m, k, n);
    }
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn isa_avx512() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping AVX-512 test: not supported");
        return;
    }
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Avx512, 2, 12>(Avx512, m, k, n);
        driver_case::<f64, Avx512, 2, 12>(Avx512, m, k, n);
    }
}

/// NEON is baseline on aarch64, so no feature-detection guard is needed: the
/// kernel always runs here. Tile matches the production dispatch choice (4×4).
#[test]
#[cfg(target_arch = "aarch64")]
fn isa_neon() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Neon, 4, 4>(Neon, m, k, n);
        driver_case::<f64, Neon, 4, 4>(Neon, m, k, n);
    }
}

/// The SIMD narrow-store (`KernelSimd::store_out`) must be **bit-identical** to the
/// scalar `NarrowFloat::narrow` (= `half::from_f32`) across edge values — normals,
/// subnormals, ±0, ±Inf, and NaN — so the full-tile vector path and the partial-tile
/// scalar path never disagree. AVX-512, 16-wide.
#[test]
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn simd_narrow_store_matches_half_avx512() {
    use gemmkit::simd::{Avx512, KernelSimd, Simd, SimdOps};
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: no avx512f");
        return;
    }
    // A spread of representative f32 bit patterns.
    let vals: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.0 / 3.0,
        65504.0,   // f16 max
        70000.0,   // overflows f16 -> Inf
        1.0e-5,    // f16 subnormal-ish
        1.0e-8,    // underflows to 0 in f16
        1.2340001, // rounding
        2.5001,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F800001), // sNaN-ish (exp all 1, low mantissa)
        f32::from_bits(0x7FFFFFFF), // NaN, all mantissa bits set
        f32::from_bits(0xFFC00000), // negative qNaN
        123.456,
    ];
    // Pad to a multiple of 16 lanes.
    let mut padded = vals.clone();
    while !padded.len().is_multiple_of(16) {
        padded.push(0.0);
    }

    // f16
    unsafe {
        Avx512.vectorize(|| {
            for chunk in padded.chunks(16) {
                let reg = <Avx512 as SimdOps<f32>>::loadu(Avx512, chunk.as_ptr());
                let mut out = [gemmkit::f16::from_f32(0.0); 16];
                <Avx512 as KernelSimd<gemmkit::f16, gemmkit::f16, f32, gemmkit::f16>>::store_out(
                    Avx512,
                    out.as_mut_ptr(),
                    reg,
                );
                for (i, &v) in chunk.iter().enumerate() {
                    let scalar = gemmkit::f16::from_f32(v);
                    assert_eq!(
                        out[i].to_bits(),
                        scalar.to_bits(),
                        "f16 narrow mismatch for {v:?} (bits {:#010x})",
                        v.to_bits()
                    );
                }
            }
        });
    }
    // bf16
    unsafe {
        Avx512.vectorize(|| {
            for chunk in padded.chunks(16) {
                let reg = <Avx512 as SimdOps<f32>>::loadu(Avx512, chunk.as_ptr());
                let mut out = [gemmkit::bf16::from_f32(0.0); 16];
                <Avx512 as KernelSimd<gemmkit::bf16, gemmkit::bf16, f32, gemmkit::bf16>>::store_out(
                    Avx512,
                    out.as_mut_ptr(),
                    reg,
                );
                for (i, &v) in chunk.iter().enumerate() {
                    let scalar = gemmkit::bf16::from_f32(v);
                    assert_eq!(
                        out[i].to_bits(),
                        scalar.to_bits(),
                        "bf16 narrow mismatch for {v:?} (bits {:#010x})",
                        v.to_bits()
                    );
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// integer GEMM (i8 -> i32)
// ---------------------------------------------------------------------------

/// Deterministic i8 fill in [-100, 100] (kept small so the i32 reference never
/// overflows for the tested k, making the comparison exact).
#[cfg(feature = "int8")]
fn rand_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 24) as i64 % 201 - 100) as i8
        })
        .collect()
}

/// Exact i32 GEMM reference (row-major), accumulated in i64 then range-checked, so
/// the integer kernel must match it **bit-for-bit**.
#[cfg(feature = "int8")]
fn ref_i8(
    a: &[i8],
    b: &[i8],
    c0: &[i32],
    m: usize,
    k: usize,
    n: usize,
    alpha: i32,
    beta: i32,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for p in 0..k {
                acc += a[i * k + p] as i64 * b[p * n + j] as i64;
            }
            let v = beta as i64 * c0[i * n + j] as i64 + alpha as i64 * acc;
            assert!(
                (i32::MIN as i64..=i32::MAX as i64).contains(&v),
                "reference overflow — tighten test sizes"
            );
            out[i * n + j] = v as i32;
        }
    }
    out
}

#[cfg(feature = "int8")]
#[test]
fn correctness_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [
        (1, 1, 1),
        (3, 4, 5),
        (16, 8, 7),
        (32, 32, 32),
        (33, 17, 19),
        (40, 33, 28),
        (64, 80, 48),
        (65, 64, 64),
        (128, 96, 112),
    ] {
        for &(alpha, beta) in &[(1i32, 0i32), (1, 1), (3, -2)] {
            let a = rand_i8(m * k, 0x100 + (m * 7 + k) as u64);
            let b = rand_i8(k * n, 0x200 + (n * 3 + k) as u64);
            let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 7) - 3).collect();
            let cref = ref_i8(&a, &b, &c0, m, k, n, alpha, beta);

            // Row-major A, column-major B, row-major C32.
            let bcol: Vec<i8> = {
                let mut v = vec![0i8; k * n];
                for i in 0..k {
                    for j in 0..n {
                        v[j * k + i] = b[i * n + j];
                    }
                }
                v
            };
            let mut c = c0.clone();
            gemmkit::gemm_i8(
                alpha,
                MatRef::from_row_major(&a, m, k),
                MatRef::new(&bcol, k, n, 1, k as isize),
                beta,
                MatMut::from_row_major(&mut c, m, n),
                Parallelism::Serial,
            );
            assert_eq!(c, cref, "i8 mismatch {m}x{k}x{n} alpha={alpha} beta={beta}");
        }
    }
}

/// Integer serial == parallel **bit-identity** (integer accumulation is exact, so
/// any thread count must produce identical i32 output).
#[cfg(feature = "int8")]
#[test]
fn parallel_equals_serial_i8() {
    use gemmkit::{MatMut, MatRef};
    for (m, k, n) in [(200, 130, 175), (256, 64, 200), (384, 96, 320)] {
        let a = rand_i8(m * k, 0x300 + m as u64);
        let b = rand_i8(k * n, 0x400 + n as u64);
        let c0: Vec<i32> = (0..m * n).map(|x| (x as i32 % 5) - 2).collect();
        for &(alpha, beta) in &[(1i32, 0i32), (2, 3)] {
            let mut c_ser = c0.clone();
            gemmkit::gemm_i8(
                alpha,
                MatRef::from_col_major(&a, m, k),
                MatRef::from_col_major(&b, k, n),
                beta,
                MatMut::from_col_major(&mut c_ser, m, n),
                Parallelism::Serial,
            );
            for t in [2usize, 4, 8, 16] {
                let mut c_par = c0.clone();
                gemmkit::gemm_i8(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    MatRef::from_col_major(&b, k, n),
                    beta,
                    MatMut::from_col_major(&mut c_par, m, n),
                    Parallelism::Rayon(t),
                );
                assert_eq!(c_ser, c_par, "i8 serial != parallel({t}) for {m}x{k}x{n}");
            }
        }
    }
}

/// Negative strides for the integer path via [`gemmkit::gemm_i8_unchecked`] (the
/// heterogeneous escape hatch — the homogeneous `gemm_unchecked` can't serve
/// `i8 -> i32`). Reversed-row A, compared to the row-reversed exact reference.
#[cfg(feature = "int8")]
#[test]
fn i8_negative_strides_unchecked() {
    let (m, k, n) = (12usize, 9, 7);
    let a = rand_i8(m * k, 5); // row-major m×k
    let b = rand_i8(k * n, 6); // row-major k×n
    let c0 = vec![0i32; m * n];
    let cref = ref_i8(&a, &b, &c0, m, k, n, 1, 0);

    let mut c = vec![0i32; m * n];
    unsafe {
        let a_last = a.as_ptr().add((m - 1) * k); // base = last row
        gemmkit::gemm_i8_unchecked(
            m,
            k,
            n,
            1,
            a_last,
            -(k as isize), // reversed rows of A
            1,
            b.as_ptr(),
            n as isize, // row-major B
            1,
            0,
            c.as_mut_ptr(),
            n as isize, // row-major C
            1,
            Parallelism::Serial,
        );
    }
    // Computed C[i,j] = sum_k A[m-1-i,k]·B[k,j]; compare to the reversed reference.
    for i in 0..m {
        for j in 0..n {
            assert_eq!(
                c[i * n + j],
                cref[(m - 1 - i) * n + j],
                "i8 neg stride ({i},{j})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// complex GEMM (c32 / c64, conj variants)
// ---------------------------------------------------------------------------

/// A complex element type the complex test harness is generic over.
#[cfg(feature = "complex")]
trait CElem: gemmkit::ComplexScalar {
    const EPS: f64;
    fn of(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
#[cfg(feature = "complex")]
impl CElem for gemmkit::c32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn of(re: f64, im: f64) -> Self {
        gemmkit::Complex::new(re as f32, im as f32)
    }
    fn parts(self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
}
#[cfg(feature = "complex")]
impl CElem for gemmkit::c64 {
    const EPS: f64 = f64::EPSILON;
    fn of(re: f64, im: f64) -> Self {
        gemmkit::Complex::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

#[cfg(feature = "complex")]
fn rand_cplx<T: CElem>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    (0..n).map(|_| T::of(next(), next())).collect()
}

/// f64 complex reference (column-major), with conj of A / B as selected.
#[cfg(feature = "complex")]
fn ref_cplx<T: CElem>(
    a: &[T],
    b: &[T],
    c0: &[T],
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> Vec<(f64, f64)> {
    let cmul = |x: (f64, f64), y: (f64, f64)| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
    let (al, be) = (alpha.parts(), beta.parts());
    let mut out = vec![(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = (0.0f64, 0.0f64);
            for p in 0..k {
                let mut av = a[p * m + i].parts(); // column-major A
                let mut bv = b[j * k + p].parts(); // column-major B
                if conj_a {
                    av.1 = -av.1;
                }
                if conj_b {
                    bv.1 = -bv.1;
                }
                let pr = cmul(av, bv);
                acc = (acc.0 + pr.0, acc.1 + pr.1);
            }
            let term = cmul(al, acc);
            let bc = cmul(be, c0[j * m + i].parts());
            out[i * n + j] = (bc.0 + term.0, bc.1 + term.1);
        }
    }
    out
}

#[cfg(feature = "complex")]
fn assert_cplx_accurate<T: CElem>(
    got: &[T],
    m: usize,
    n: usize,
    cref: &[(f64, f64)],
    k: usize,
    ctx: &str,
) {
    // Relative error over the whole matrix (column-major `got`).
    let mut diff2 = 0.0;
    let mut ref2 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let (gr, gi) = got[j * m + i].parts();
            let (rr, ri) = cref[i * n + j];
            assert!(
                gr.is_finite() && gi.is_finite(),
                "{ctx}: non-finite ({i},{j})"
            );
            diff2 += (gr - rr).powi(2) + (gi - ri).powi(2);
            ref2 += rr * rr + ri * ri;
        }
    }
    let rel = diff2.sqrt() / (ref2.sqrt() + 1e-30);
    let tol = 16.0 * (k.max(1) as f64) * T::EPS;
    assert!(rel <= tol, "{ctx}: rel err {rel:.3e} > tol {tol:.3e}");
}

#[cfg(feature = "complex")]
#[test]
fn correctness_complex() {
    fn check<T: CElem>() {
        for (m, k, n) in [
            (1, 1, 1),
            (3, 4, 5),
            (32, 32, 32),
            (33, 17, 19),
            (64, 80, 48),
        ] {
            for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
                let a = rand_cplx::<T>(m * k, 0xC0 + (m * 7 + k) as u64);
                let b = rand_cplx::<T>(k * n, 0xC1 + (n * 3 + k) as u64);
                let c0 = rand_cplx::<T>(m * n, 0xC2 + (m + n) as u64);
                let (alpha, beta) = (T::of(1.1, -0.3), T::of(0.5, 0.7));
                let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, ca, cb);
                let mut c = c0.clone();
                // All column-major.
                gemmkit::gemm_cplx(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    ca,
                    MatRef::from_col_major(&b, k, n),
                    cb,
                    beta,
                    MatMut::from_col_major(&mut c, m, n),
                    Parallelism::Serial,
                );
                assert_cplx_accurate(&c, m, n, &cref, k, &format!("{m}x{k}x{n} ca={ca} cb={cb}"));
            }
        }
    }
    check::<gemmkit::c32>();
    check::<gemmkit::c64>();
}

/// Complex serial == parallel bit-identity across thread counts.
#[cfg(feature = "complex")]
#[test]
fn parallel_equals_serial_complex() {
    for (m, k, n) in [(200, 130, 175), (256, 64, 200)] {
        for &(ca, cb) in &[(false, false), (true, true)] {
            let a = rand_cplx::<gemmkit::c32>(m * k, 0xD0 + m as u64);
            let b = rand_cplx::<gemmkit::c32>(k * n, 0xD1 + n as u64);
            let c0 = rand_cplx::<gemmkit::c32>(m * n, 0xD2 + k as u64);
            let (alpha, beta) = (
                gemmkit::Complex::new(0.7f32, 0.2),
                gemmkit::Complex::new(1.3f32, -0.4),
            );
            let mut c_ser = c0.clone();
            gemmkit::gemm_cplx(
                alpha,
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                beta,
                MatMut::from_col_major(&mut c_ser, m, n),
                Parallelism::Serial,
            );
            for t in [2usize, 4, 8, 16] {
                let mut c_par = c0.clone();
                gemmkit::gemm_cplx(
                    alpha,
                    MatRef::from_col_major(&a, m, k),
                    ca,
                    MatRef::from_col_major(&b, k, n),
                    cb,
                    beta,
                    MatMut::from_col_major(&mut c_par, m, n),
                    Parallelism::Rayon(t),
                );
                let bits = |v: &[gemmkit::c32]| {
                    v.iter()
                        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&c_ser), bits(&c_par), "complex serial != par({t})");
            }
        }
    }
}

/// The raw `gemm_cplx_unchecked` entry (added for the ndarray adapter): exercise it
/// with a **negative-row-stride** A view + conj, against the row-reversed reference.
/// The safe `MatRef` surface can't express reversed strides, so this raw path is what
/// arbitrary-stride callers (the ndarray adapter) rely on.
#[cfg(feature = "complex")]
#[test]
fn cplx_unchecked_negative_strides() {
    let (m, k, n) = (12, 9, 7);
    for &(ca, cb) in &[(false, false), (true, false), (true, true)] {
        let a = rand_cplx::<gemmkit::c32>(m * k, 0xF0 + (m + k) as u64);
        let b = rand_cplx::<gemmkit::c32>(k * n, 0xF1 + (n + k) as u64);
        let c0 = rand_cplx::<gemmkit::c32>(m * n, 0xF2 + (m + n) as u64);
        let (alpha, beta) = (
            gemmkit::Complex::new(1.1f32, -0.3),
            gemmkit::Complex::new(0.5f32, 0.7),
        );
        // Row-reversed copy of the (column-major) A, for the reference.
        let mut a_rev = a.clone();
        for p in 0..k {
            for i in 0..m {
                a_rev[p * m + i] = a[p * m + (m - 1 - i)];
            }
        }
        let cref = ref_cplx(&a_rev, &b, &c0, m, k, n, alpha, beta, ca, cb);
        let mut c = c0.clone();
        // A: base at physical row m-1, row stride -1 (col-major col stride m); B/C col-major.
        unsafe {
            gemmkit::gemm_cplx_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr().add(m - 1),
                -1,
                m as isize,
                ca,
                b.as_ptr(),
                1,
                k as isize,
                cb,
                beta,
                c.as_mut_ptr(),
                1,
                m as isize,
                Parallelism::Serial,
            );
        }
        assert_cplx_accurate(
            &c,
            m,
            n,
            &cref,
            k,
            &format!("unchecked neg-stride ca={ca} cb={cb}"),
        );
    }
}

/// Cross-check complex (c32) against the `gemm` crate (which has native c32 and
/// `conj_lhs`/`conj_rhs` flags); `gemm::c32 == num_complex::Complex32 == gemmkit::c32`,
/// so the comparison is direct. Gated out of Miri.
#[test]
#[cfg(all(not(miri), feature = "complex"))]
fn complex_matches_gemm_crate() {
    for (m, k, n) in [(64, 48, 40), (96, 65, 72), (33, 17, 19)] {
        for &(ca, cb) in &[(false, false), (true, false), (false, true), (true, true)] {
            let a = rand_cplx::<gemmkit::c32>(m * k, 0xE0 + (m + k) as u64);
            let b = rand_cplx::<gemmkit::c32>(k * n, 0xE1 + (n + k) as u64);
            let mut c_kit = vec![gemmkit::Complex::new(0.0f32, 0.0); m * n];
            let mut c_gemm = c_kit.clone();

            gemmkit::gemm_cplx(
                gemmkit::Complex::new(1.0f32, 0.0),
                MatRef::from_col_major(&a, m, k),
                ca,
                MatRef::from_col_major(&b, k, n),
                cb,
                gemmkit::Complex::new(0.0f32, 0.0),
                MatMut::from_col_major(&mut c_kit, m, n),
                Parallelism::Serial,
            );
            // gemm crate: dst = alpha*dst + beta*op(lhs)*op(rhs); alpha=0, beta=1.
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    c_gemm.as_mut_ptr(),
                    m as isize,
                    1,
                    false,
                    a.as_ptr(),
                    m as isize,
                    1,
                    b.as_ptr(),
                    k as isize,
                    1,
                    gemmkit::Complex::new(0.0f32, 0.0),
                    gemmkit::Complex::new(1.0f32, 0.0),
                    false, // conj_dst
                    ca,    // conj_lhs
                    cb,    // conj_rhs
                    gemm::Parallelism::None,
                );
            }
            // Both column-major; build a row-major (f64,f64) reference from c_gemm.
            let mut cref = vec![(0.0f64, 0.0f64); m * n];
            for i in 0..m {
                for j in 0..n {
                    let z = c_gemm[j * m + i];
                    cref[i * n + j] = (z.re as f64, z.im as f64);
                }
            }
            assert_cplx_accurate(
                &c_kit,
                m,
                n,
                &cref,
                k,
                &format!("c32 vs gemm {m}x{k}x{n} ca={ca} cb={cb}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// gemv shapes
// ---------------------------------------------------------------------------

#[test]
fn gemv_shapes() {
    // n == 1 and m == 1, across layouts.
    for &(m, k, n) in &[
        (64, 40, 1),
        (1, 40, 64),
        (100, 1, 1),
        (1, 1, 100),
        (255, 129, 1),
    ] {
        for &la in &[Layout::Row, Layout::Col] {
            for &lb in &[Layout::Row, Layout::Col] {
                run_case::<f32>(m, k, n, la, lb, Layout::Col, 1.3, -0.4, Parallelism::Serial);
                run_case::<f64>(m, k, n, la, lb, Layout::Row, 0.5, 2.0, Parallelism::Serial);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cache detection (§7.5)
// ---------------------------------------------------------------------------

#[test]
fn cache_topology_is_plausible() {
    let t = gemmkit::topology();
    // Sane bounds (true on any real CPU).
    assert!(
        t.l1d.bytes >= 8 * 1024 && t.l1d.bytes <= 256 * 1024,
        "L1d={}",
        t.l1d.bytes
    );
    assert!(t.l1d.line >= 32 && t.l1d.line <= 256);
    assert!(t.l2.bytes >= 128 * 1024, "L2={}", t.l2.bytes);
    // Blocking parameters are sane for a big problem.
    let blk = t.blocking(32, 12, 4, 4096, 4096, 4096);
    assert!(blk.mc >= 32 && blk.kc >= 1 && blk.nc >= 12);
    assert!(blk.mc.is_multiple_of(32), "mc should be a multiple of MR");
    eprintln!(
        "topology: L1d={}K L2={}K L3={:?}K  blocking(4096³): mc={} kc={} nc={}",
        t.l1d.bytes / 1024,
        t.l2.bytes / 1024,
        t.l3.map(|l| l.bytes / 1024),
        blk.mc,
        blk.kc,
        blk.nc
    );
}

// ---------------------------------------------------------------------------
// safe-API panic guarantees
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "A.cols")]
fn panic_shape_mismatch() {
    let a = vec![0.0f32; 6];
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 2, 3), // B.rows=2 != A.cols=3
        0.0,
        MatMut::from_row_major(&mut c, 2, 3),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "needs")]
fn panic_out_of_bounds_view() {
    let a = vec![0.0f32; 3]; // too small for 2x3
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 4];
    gemm(
        1.0,
        MatRef::new(&a, 2, 3, 3, 1),
        MatRef::from_row_major(&b, 3, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "aliases itself")]
fn panic_self_aliasing_c() {
    // rsc == 0 collapses all rows of C onto the same memory — accepted by the
    // bounds check (extent fits) but a data race in parallel. Must panic.
    let a = vec![0.0f32; 8]; // 4x2
    let b = vec![0.0f32; 6]; // 2x3
    let mut c = vec![0.0f32; 3];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 4, 2),
        MatRef::from_row_major(&b, 2, 3),
        0.0,
        MatMut::new(&mut c, 4, 3, 0, 1),
        Parallelism::Rayon(0),
    );
}

#[test]
#[should_panic(expected = "aliases")]
fn panic_c_aliases_a() {
    // Force an alias via raw slices over the same buffer through MatRef/MatMut.
    let mut buf = vec![1.0f32; 16];
    let (a_part, c_part) = buf.split_at_mut(0); // a_part empty? need overlap
    let _ = (a_part, c_part);
    // Build overlapping views by unsafe transmute of lifetimes is messy; instead
    // use the same slice for A and C via raw pointers is not possible in safe
    // API. We simulate by pointing both at `buf` through separate borrows is
    // disallowed; so construct via std::slice::from_raw_parts.
    let ptr = buf.as_mut_ptr();
    let len = buf.len();
    unsafe {
        let a_slice = std::slice::from_raw_parts(ptr, len);
        let c_slice = std::slice::from_raw_parts_mut(ptr, len);
        gemm(
            1.0,
            MatRef::from_row_major(a_slice, 2, 2),
            MatRef::from_row_major(a_slice, 2, 2),
            0.0,
            MatMut::from_row_major(c_slice, 2, 2),
            Parallelism::Serial,
        );
    }
}
