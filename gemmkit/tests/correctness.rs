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
/// thread counts. `(60, 600, 60)` is the case whose general/tiny blocking actually
/// diverges (`k > 512`), which the old recompute-and-assert design would have
/// rejected; here it must compute correctly instead.
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
    // Sane bounds (true on any real CPU; on the 9950X these read close to the
    // Zen5 values L1d≈48K, L2≈1M, L3≈32M).
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
