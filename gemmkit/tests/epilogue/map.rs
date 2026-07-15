//! User-defined per-element map-epilogue tests (`gemm_map`): the headline `map == gemm-then-map`
//! oracle, every route (general driver, gemv both orientations, small-`m,n`, small-`k`), strided /
//! row-major (swapped) C, degenerate cases, environment-capturing closures, serial == parallel, and
//! the `_with` / `_unchecked` twin equivalences
//!
//! Every comparison is **bitwise**. The oracle is "plain `gemm`, then the exact scalar map": the
//! map epilogue sets `VECTOR = true` so every element takes the **same** fast/scratch path plain
//! `gemm` takes (`apply_reg` drains the fast-path fused register per lane), making the value handed
//! to the closure bit-for-bit the value plain `gemm` stores. A scratch-only (`VECTOR = false`)
//! epilogue would instead diverge by 1 ULP for `beta` outside `{0, 1}` (unfused `mul_add` vs the
//! fast path's hardware FMA), and the `beta = 0.7` cases here would catch that. So the fused result
//! must equal the oracle for **every** shape and route. The closure is asymmetric and position-dependent
//! (`f(v, r, c) != f(v, c, r)`), so a wrong orientation-coordinate flip (row-major C) diverges loudly

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_map, gemm_map_unchecked};

/// The asymmetric, value- and position-dependent map, computed **entirely in `T`** so it is
/// bitwise-reproducible, and used verbatim by both the closure under test and the reference. It is
/// deliberately **not symmetric** in `(r, c)`: `lut[r % L]` depends only on the row, and the offset
/// `r - 0.25*c` weights row and column differently, so swapping `r` and `c` changes the result -
/// which is what makes the row-major-C (orientation-swapped) route a falsifier for the coordinate
/// flip. `mul_add` mirrors the kernel's fused form
fn map_expr<T: Flt>(v: T, r: usize, c: usize, lut: &[T]) -> T {
    let scale = lut[r % lut.len()];
    let offset = T::of(r as f64) - T::of(0.25) * T::of(c as f64);
    v.mul_add(scale, offset)
}

/// A small captured lookup table (length 5, coprime with the tile widths so `r % 5` cycles across
/// tile boundaries). Deterministic from a seed so the oracle uses the identical values
fn make_lut<T: Flt>(rng: &mut Rng) -> Vec<T> {
    (0..5).map(|_| T::of(rng.unit() * 1.5 + 0.5)).collect()
}

/// Run 1 `gemm_map` case and its `gemm`-then-`map_expr` oracle over the identical inputs; assert
/// bitwise-equal C. The closure captures `lut` **by reference** (the environment-capture contract).
/// Handles the degenerate shapes too (`k == 0` builds empty A/B; `alpha == 0` is passed straight
/// through), since the oracle path (plain `gemm` then the map) is identical there
#[allow(clippy::too_many_arguments)]
fn check_map<T: Flt + gemmkit::MapScalar>(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    beta: T,
    layout: Layout,
    par: Parallelism,
    tag: &str,
) {
    let a = make::<T>(rng, m, k); // col-major mxk (empty when k == 0)
    let b = make::<T>(rng, k, n); // col-major kxn
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen.max(1), 1);
    let lut = make_lut::<T>(rng);

    // fused: gemm_map with an environment-capturing closure (borrows `lut`)
    let mut c_map = c0.clone();
    {
        let f = |v: T, r: usize, c: usize| map_expr::<T>(v, r, c, &lut);
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_map, m, n, rsc, csc);
        let mut ws = Workspace::new();
        gemmkit::gemm_map_with(&mut ws, alpha, ar, br, beta, cm, &f, par);
    }

    // oracle: plain gemm, then the identical scalar map in the user frame
    let mut c_ref = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_ref, m, n, rsc, csc);
        gemm(alpha, ar, br, beta, cm, par);
    }
    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            c_ref[idx] = map_expr::<T>(c_ref[idx], i, j, &lut);
        }
    }

    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            assert_eq!(
                c_map[idx].bits(),
                c_ref[idx].bits(),
                "{} {tag}: map != gemm-then-map at ({i},{j}) [m={m} k={k} n={n} layout]",
                T::name(),
            );
        }
    }
}

// the headline oracle: gemm_map == gemm-then-map, bitwise, across every driver shape

fn driver_matrix<T: Flt + gemmkit::MapScalar>(par: Parallelism) {
    let mut rng = Rng::new(0x11A9_02DE);
    // driver shapes: above the small_mn/small_k thresholds, non-tile-multiple m/n/k
    let shapes = [
        (17usize, 20usize, 19usize),
        (33, 40, 24),
        (48, 96, 129),
        (40, 4096, 40), // multi-panel K: fire-once (a per-panel map would diverge)
    ];
    for &(m, k, n) in &shapes {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
                    check_map::<T>(&mut rng, m, k, n, alpha, beta, layout, par, "driver");
                }
            }
        }
    }
}

#[test]
fn map_eq_gemm_then_map_serial() {
    driver_matrix::<f32>(Parallelism::Serial);
    driver_matrix::<f64>(Parallelism::Serial);
}

#[test]
fn map_eq_gemm_then_map_parallel() {
    driver_matrix::<f32>(Parallelism::Rayon(8));
    driver_matrix::<f64>(Parallelism::Rayon(8));
}

// the special routes: gemv (both orientations), small-m,n, small-k

/// gemv (`m == 1` and `n == 1`): the map is applied as a final in-place sweep in the **user** frame
/// (gemv routes before orientation), so both the `(i, 0)` and `(0, i)` coordinate mappings are
/// exercised
#[test]
fn gemv_both_orientations() {
    let mut rng = Rng::new(0x9E0F_1201);
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        for &beta in &[0.0f64, 1.0, 0.7] {
            // n == 1: C is m x 1 (coordinate (i, 0))
            check_map::<f64>(&mut rng, 50, 30, 1, 1.0, beta, Layout::Col, par, "gemv-n1");
            check_map::<f32>(
                &mut rng,
                64,
                48,
                1,
                0.9,
                beta as f32,
                Layout::Col,
                par,
                "gemv-n1",
            );
            // m == 1: C is 1 x n (coordinate (0, j))
            check_map::<f64>(&mut rng, 1, 30, 50, 1.0, beta, Layout::Col, par, "gemv-m1");
            check_map::<f32>(
                &mut rng,
                1,
                48,
                64,
                0.9,
                beta as f32,
                Layout::Col,
                par,
                "gemv-m1",
            );
        }
    }
}

/// small-`m,n` horizontal route: `m, n <= 16` with a long contraction (`k > small_k_threshold`),
/// A rows / B cols unit-stride so the gate engages. Col-major C keeps `csa == 1 && rsb == 1`
#[test]
fn small_mn_route() {
    let mut rng = Rng::new(0x5A11_3D0E);
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        for &(m, n) in &[(4usize, 4usize), (7, 13), (16, 16), (3, 9)] {
            for &beta in &[0.0f32, 1.0, 0.7] {
                check_map::<f32>(&mut rng, m, 96, n, 1.0, beta, Layout::Col, par, "small_mn");
                check_map::<f64>(
                    &mut rng,
                    m,
                    96,
                    n,
                    0.9,
                    beta as f64,
                    Layout::Col,
                    par,
                    "small_mn",
                );
            }
        }
    }
}

/// small-`k` route: `k <= small_k_threshold` (skinny / low-depth), col-major A so the in-place
/// microkernel path is taken (`rsa == 1`). Non-tile-multiple m/n exercise the edge tiles + pad panel
#[test]
fn small_k_route() {
    let mut rng = Rng::new(0x0DEE_5C0F);
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        for &k in &[1usize, 2, 4, 8] {
            for &(m, n) in &[(20usize, 24usize), (33, 17), (40, 40)] {
                check_map::<f32>(&mut rng, m, k, n, 1.0, 0.7, Layout::Col, par, "small_k");
                check_map::<f64>(&mut rng, m, k, n, 0.9, 0.0, Layout::Col, par, "small_k");
            }
        }
    }
}

// the swapped-coordinate falsifier: row-major C must still see user-frame (r, c)

/// A row-major C makes the driver compute `C^T = B^T*A^T` (swapping `m<->n`), so the epilogue runs in
/// the transposed frame. `MapEpi::apply` must flip the coordinates back so the closure sees the
/// **user** `(r, c)`. With the asymmetric `map_expr` and a non-square shape, a mishandled flip would
/// feed the closure `(c, r)` and diverge; the `check_map` oracle (which maps in the user frame) is
/// therefore the falsifier. This test first asserts the map really is asymmetric (so the check has
/// teeth), then drives row-major C on a rectangular driver shape across parallelism
#[test]
fn swapped_coord_falsifier() {
    // Sanity: the map is genuinely asymmetric, so feeding transposed coordinates gives a *different*
    // answer - i.e. a wrong `swapped` flag cannot accidentally pass
    let lut = make_lut::<f64>(&mut Rng::new(1));
    let v = 1.25f64;
    assert_ne!(
        map_expr(v, 2, 5, &lut).to_bits(),
        map_expr(v, 5, 2, &lut).to_bits(),
        "map_expr must be asymmetric for the falsifier to have teeth"
    );

    let mut rng = Rng::new(0x5A9A_FEED);
    // rectangular (m != n) so transposition is observable, row-major C forces the orientation swap
    for &(m, k, n) in &[(33usize, 40usize, 24usize), (48, 33, 65), (17, 96, 29)] {
        for par in [Parallelism::Serial, Parallelism::Rayon(8)] {
            for &beta in &[0.0f32, 1.0, 0.7] {
                check_map::<f32>(&mut rng, m, k, n, 0.9, beta, Layout::Row, par, "swap");
                check_map::<f64>(
                    &mut rng,
                    m,
                    k,
                    n,
                    1.0,
                    beta as f64,
                    Layout::Row,
                    par,
                    "swap",
                );
            }
        }
    }
}

// degenerate cases: alpha == 0 and k == 0 => C[r,c] = f(beta*C[r,c], r, c)

#[test]
fn degenerate_alpha0_and_k0() {
    let mut rng = Rng::new(0xDE6E_0A11);
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
            for &beta in &[0.0f32, 1.0, 0.7] {
                // alpha == 0: A*B term dropped
                check_map::<f32>(&mut rng, 20, 24, 19, 0.0, beta, layout, par, "alpha0");
                check_map::<f64>(
                    &mut rng,
                    20,
                    24,
                    19,
                    0.0,
                    beta as f64,
                    layout,
                    par,
                    "alpha0",
                );
                // k == 0: empty contraction
                check_map::<f32>(&mut rng, 20, 0, 19, 1.0, beta, layout, par, "k0");
                check_map::<f64>(&mut rng, 20, 0, 19, 1.0, beta as f64, layout, par, "k0");
            }
        }
    }
}

// environment capture: the closure reads a captured &[T] table

/// The closure captures a lookup-table **slice** (`&[T]`) by reference and indexes it per output
/// element. Exercised end-to-end through `gemm_map` and compared bitwise to the same table applied
/// scalar-side, proving borrowed environment capture works across the parallel workers
#[test]
fn closure_captures_slice() {
    let mut rng = Rng::new(0xCAB7_5111);
    let m = 40usize;
    let k = 33usize;
    let n = 24usize;
    let a = make::<f64>(&mut rng, m, k);
    let b = make::<f64>(&mut rng, k, n);
    let c0 = make::<f64>(&mut rng, m * n, 1);
    // a per-column gain table, captured by reference
    let table: Vec<f64> = (0..n).map(|_| rng.unit() * 2.0 + 0.1).collect();
    let tref: &[f64] = &table;

    let mut c_map = c0.clone();
    {
        let f = |v: f64, _r: usize, c: usize| v * tref[c] + tref[c];
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_map, m, n, 1, m as isize);
        gemm_map(1.0, ar, br, 0.5, cm, &f, Parallelism::Rayon(8));
    }

    let mut c_ref = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_ref, m, n, 1, m as isize);
        gemm(1.0, ar, br, 0.5, cm, Parallelism::Rayon(8));
    }
    for j in 0..n {
        for i in 0..m {
            let idx = i + j * m;
            let want = c_ref[idx] * table[j] + table[j];
            assert_eq!(
                c_map[idx].to_bits(),
                want.to_bits(),
                "captured-slice map mismatch at ({i},{j})"
            );
        }
    }
}

// serial == parallel, bitwise

/// The output-tile / output-row partition never splits an element's reduction, so `gemm_map` is
/// bit-identical across thread counts (same guarantee plain `gemm` has). Compared directly here
/// (not only via the oracle) on a driver shape and a small-`k` shape
#[test]
fn serial_eq_parallel_bitwise() {
    let mut rng = Rng::new(0x5E21_9A11);
    let lut = make_lut::<f64>(&mut rng);
    for &(m, k, n) in &[(48usize, 96usize, 65usize), (40, 6, 40)] {
        let a = make::<f64>(&mut rng, m, k);
        let b = make::<f64>(&mut rng, k, n);
        let c0 = make::<f64>(&mut rng, m * n, 1);

        let run = |par| {
            let mut c = c0.clone();
            let f = |v: f64, r: usize, c2: usize| map_expr::<f64>(v, r, c2, &lut);
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut c, m, n, 1, m as isize);
            gemm_map(0.9, ar, br, 0.7, cm, &f, par);
            c
        };
        let c_serial = run(Parallelism::Serial);
        let c_par = run(Parallelism::Rayon(8));
        for idx in 0..m * n {
            assert_eq!(
                c_serial[idx].to_bits(),
                c_par[idx].to_bits(),
                "serial != parallel at {idx} [m={m} k={k} n={n}]"
            );
        }
    }
}

// twin equivalences: _with == allocating, unchecked == checked

/// `gemm_map` (thread-local pool) and `gemm_map_with` (caller workspace) are parallel entry points;
/// exercise them against each other bit-for-bit
#[test]
fn map_with_matches_allocating() {
    let mut rng = Rng::new(0x0F05_ED20);
    let (m, k, n) = (33usize, 24usize, 40usize);
    let a = make::<f32>(&mut rng, m, k);
    let b = make::<f32>(&mut rng, k, n);
    let c0 = make::<f32>(&mut rng, m * n, 1);
    let lut = make_lut::<f32>(&mut rng);
    let f = |v: f32, r: usize, c: usize| map_expr::<f32>(v, r, c, &lut);

    let mut c_alloc = c0.clone();
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_alloc, m, n, 1, m as isize);
        gemm_map(0.9, ar, br, 0.7, cm, &f, Parallelism::Serial);
    }
    let mut c_with = c0.clone();
    {
        let mut ws = Workspace::new();
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_with, m, n, 1, m as isize);
        gemmkit::gemm_map_with(&mut ws, 0.9, ar, br, 0.7, cm, &f, Parallelism::Serial);
    }
    for idx in 0..m * n {
        assert_eq!(
            c_alloc[idx].to_bits(),
            c_with[idx].to_bits(),
            "gemm_map_with != gemm_map at {idx}"
        );
    }
}

/// `gemm_map_unchecked` and `gemm_map` are parallel entry points (the checked twin does not delegate
/// to the unchecked one); exercise the raw fn against the checked twin bit-for-bit on a driver shape,
/// a small-`k` shape, and a gemv shape
#[test]
fn map_unchecked_matches_checked() {
    let mut rng = Rng::new(0x0F05_ED21);
    for &(m, k, n) in &[(33usize, 24usize, 40usize), (30, 5, 28), (48, 30, 1)] {
        let a = make::<f64>(&mut rng, m, k);
        let b = make::<f64>(&mut rng, k, n);
        let c0 = make::<f64>(&mut rng, m * n, 1);
        let lut = make_lut::<f64>(&mut rng);

        let mut c_checked = c0.clone();
        {
            let f = |v: f64, r: usize, c: usize| map_expr::<f64>(v, r, c, &lut);
            let ar = MatRef::new(&a, m, k, 1, m as isize);
            let br = MatRef::new(&b, k, n, 1, k as isize);
            let cm = MatMut::new(&mut c_checked, m, n, 1, m as isize);
            gemm_map(0.9, ar, br, 0.7, cm, &f, Parallelism::Serial);
        }
        let mut c_unchecked = c0.clone();
        {
            let f = |v: f64, r: usize, c: usize| map_expr::<f64>(v, r, c, &lut);
            // SAFETY: valid in-bounds col-major layouts; C aliases neither A nor B
            unsafe {
                gemm_map_unchecked(
                    m,
                    k,
                    n,
                    0.9,
                    a.as_ptr(),
                    1,
                    m as isize,
                    b.as_ptr(),
                    1,
                    k as isize,
                    0.7,
                    c_unchecked.as_mut_ptr(),
                    1,
                    m as isize,
                    &f,
                    Parallelism::Serial,
                );
            }
        }
        for idx in 0..m * n {
            assert_eq!(
                c_checked[idx].to_bits(),
                c_unchecked[idx].to_bits(),
                "map unchecked != checked at {idx} [m={m} k={k} n={n}]"
            );
        }
    }
}

// validation panics: the standard gemm view checks (byte-identical wording)

mod validation {
    use super::*;

    #[test]
    #[should_panic(expected = "A.cols")]
    fn dim_mismatch() {
        let a = vec![1.0f32; 4 * 3];
        let b = vec![1.0f32; 4 * 4]; // B.rows = 4 != A.cols = 3
        let mut c = vec![0.0f32; 4 * 4];
        let f = |v: f32, _r: usize, _c: usize| v;
        gemm_map(
            1.0,
            MatRef::from_col_major(&a, 4, 3),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::from_col_major(&mut c, 4, 4),
            &f,
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "C aliases A or B")]
    fn c_aliases_a() {
        let mut buf = vec![1.0f32; 4 * 4];
        let b = vec![1.0f32; 4 * 4];
        // A and C share `buf`: an aliasing output, rejected before any compute
        let a: &[f32] = unsafe { core::slice::from_raw_parts(buf.as_ptr(), 16) };
        let f = |v: f32, _r: usize, _c: usize| v;
        gemm_map(
            1.0,
            MatRef::from_col_major(a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::from_col_major(&mut buf, 4, 4),
            &f,
            Parallelism::Serial,
        );
    }

    #[test]
    #[should_panic(expected = "aliases itself")]
    fn c_self_aliases() {
        let a = vec![1.0f32; 4 * 4];
        let b = vec![1.0f32; 4 * 4];
        let mut c = vec![0.0f32; 4];
        let f = |v: f32, _r: usize, _c: usize| v;
        // rsc = 0 maps every row of a column to the same cell: a self-aliasing output
        gemm_map(
            1.0,
            MatRef::from_col_major(&a, 4, 4),
            MatRef::from_col_major(&b, 4, 4),
            0.0,
            MatMut::new(&mut c, 4, 4, 0, 1),
            &f,
            Parallelism::Serial,
        );
    }
}
