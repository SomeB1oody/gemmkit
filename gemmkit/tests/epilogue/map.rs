//! User-defined per-element map-epilogue tests (`gemm_map`): the headline `map == gemm-then-map`
//! oracle across every route (general driver, both gemv orientations, small-`m,n`, small-`k`),
//! strided / row-major (orientation-swapped) C, degenerate cases, environment-capturing closures,
//! serial == parallel, and the `_with` / `_unchecked` twin equivalences
//!
//! Every comparison is **bitwise**. The oracle is "plain `gemm`, then the exact scalar map,
//! applied by hand": the map epilogue sets `VECTOR = true`, so a full column-major tile takes the
//! same `apply_reg` fast path plain `gemm`'s own store would, handing the closure the identical
//! value plain `gemm` writes. A `VECTOR = false` epilogue would instead force every element
//! through the scratch path regardless of tile shape, risking a different rounding of the
//! `beta*C + alpha*A*B` combine once `beta` sits outside `{0, 1}`; the `beta = 0.7` cases here are
//! what would catch that. The closure is asymmetric and position-dependent (`f(v, r, c) != f(v, c,
//! r)`), so a wrong orientation-coordinate flip (row-major C) diverges loudly

use crate::common::*;
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_map, gemm_map_unchecked};

/// The asymmetric, position-dependent map both the closure under test and the oracle use
/// verbatim, computed entirely in `T` so it is bitwise-reproducible. `lut[r % L]` depends only on
/// the row, and the offset `r - 0.25*c` weights row and column differently, so `f(v, r, c) !=
/// f(v, c, r)` in general: this is what makes the row-major-C (orientation-swapped) route a
/// falsifier for a wrong coordinate flip. `mul_add` mirrors the kernel's own fused combine
fn map_expr<T: Flt>(v: T, r: usize, c: usize, lut: &[T]) -> T {
    let scale = lut[r % lut.len()];
    let offset = T::of(r as f64) - T::of(0.25) * T::of(c as f64);
    v.mul_add(scale, offset)
}

/// A small captured lookup table (length 5, prime, so `r % 5` does not stay aligned with a
/// power-of-two tile width). Deterministic from a seed, so the oracle can rebuild the same values
fn make_lut<T: Flt>(rng: &mut Rng) -> Vec<T> {
    (0..5).map(|_| T::of(rng.unit() * 1.5 + 0.5)).collect()
}

/// Runs 1 `gemm_map_with` case and its plain-`gemm`-then-[`map_expr`] oracle over identical
/// inputs, then asserts every C element's bits match. The closure captures `lut` by reference,
/// exercising the environment-capture contract. Also covers the degenerate shapes (`k == 0`
/// builds empty A/B; `alpha == 0` is passed straight through), since the oracle path is the same
/// either way
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
    let a = make::<T>(rng, m, k); // col-major m x k (empty when k == 0)
    let b = make::<T>(rng, k, n); // col-major k x n
    let (rsc, csc, clen) = c_strides(layout, m, n);
    let c0 = make::<T>(rng, clen.max(1), 1);
    let lut = make_lut::<T>(rng);

    // the call under test: a closure borrowing lut, run through gemm_map_with
    let mut c_map = c0.clone();
    {
        let f = |v: T, r: usize, c: usize| map_expr::<T>(v, r, c, &lut);
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_map, m, n, rsc, csc);
        let mut ws = Workspace::new();
        gemmkit::gemm_map_with(&mut ws, alpha, ar, br, beta, cm, &f, par);
    }

    // oracle: plain gemm, then map_expr in the user frame
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
        (40, 4096, 40), // multi-panel K: fires only on the last panel, a per-panel map would diverge
    ];
    // Under GEMMKIT_FAST_TEST, run the full lattice only on shape index 0 (the smallest) and
    // reduce the other 3 shapes to 1 non-trivial combo each. The k = 4096 fire-once shape still
    // runs at least once, with the divergence-catching beta = 0.7. Off, fast is false and nothing
    // is skipped
    let fast = fast_test();
    let full_lattice = 0usize;
    for (si, &(m, k, n)) in shapes.iter().enumerate() {
        for &beta in &[T::ZERO, T::ONE, T::of(0.7)] {
            for &alpha in &[T::ONE, T::of(0.9)] {
                for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
                    if fast
                        && si != full_lattice
                        && !(beta == T::of(0.7)
                            && alpha == T::of(0.9)
                            && matches!(layout, Layout::ColPadded))
                    {
                        continue;
                    }
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

/// gemv (`n == 1` or `m == 1`) routes before any orientation swap, so the map runs directly in
/// the user frame; running both an `n == 1` (coordinate `(i, 0)`) and an `m == 1` (coordinate
/// `(0, j)`) shape exercises both output layouts
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

/// small-`m,n` horizontal route: `m, n <= 16` (the `small_mn_dim` default) with a contraction
/// past `small_mn_pack_min_k`. `check_map`'s A and B are always column-major, giving `rsb == 1`
/// but `csa != 1`, so this engages the pack tier (`small_mn_pack_eligible`), which copies A into
/// k-contiguous scratch before running the same horizontal kernel
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

/// small-`k` route: `k <= small_k_threshold` (skinny, low-depth), col-major A so its unit row
/// stride (`rsa == 1`) takes the in-place microkernel path. Non-tile-multiple m/n exercise the
/// edge tiles and the pad panel
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

// the swapped-coordinate falsifier: row-major C must still see the user-frame (r, c)

/// A row-major C makes the driver compute `C^T = B^T*A^T` (swapping `m` and `n` internally), so
/// the epilogue runs in the transposed frame; `MapEpi::apply` flips `(row, col)` back to `(col,
/// row)` before calling the closure, so it always sees the **user** `(r, c)`. With the asymmetric
/// `map_expr` and a non-square shape, a mishandled flip would feed the closure `(c, r)` instead
/// and diverge from `check_map`'s user-frame oracle. This first asserts the map really is
/// asymmetric (so the check below has teeth), then drives row-major C on rectangular driver
/// shapes across parallelism
#[test]
fn swapped_coord_falsifier() {
    // sanity: the map is genuinely asymmetric, so a swapped (c, r) call gives a different answer,
    // meaning a wrong coordinate flip cannot accidentally pass the check below
    let lut = make_lut::<f64>(&mut Rng::new(1));
    let v = 1.25f64;
    assert_ne!(
        map_expr(v, 2, 5, &lut).to_bits(),
        map_expr(v, 5, 2, &lut).to_bits(),
        "map_expr must be asymmetric for the falsifier to have teeth"
    );

    let mut rng = Rng::new(0x5A9A_FEED);
    // rectangular (m != n) so a wrong swap is observable; Layout::Row forces the orientation swap
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

// degenerate cases: alpha == 0 or k == 0 collapses the A*B term, so C[r,c] = f(beta*C[r,c], r, c)

#[test]
fn degenerate_alpha0_and_k0() {
    let mut rng = Rng::new(0xDE6E_0A11);
    for par in [Parallelism::Serial, Parallelism::Rayon(4)] {
        for layout in [Layout::Col, Layout::Row, Layout::ColPadded] {
            for &beta in &[0.0f32, 1.0, 0.7] {
                // alpha == 0: the A*B term is dropped, but A/B are still real-sized
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
                // k == 0: A and B are both empty
                check_map::<f32>(&mut rng, 20, 0, 19, 1.0, beta, layout, par, "k0");
                check_map::<f64>(&mut rng, 20, 0, 19, 1.0, beta as f64, layout, par, "k0");
            }
        }
    }
}

// environment capture: the closure reads a captured &[T] table

/// The closure captures a per-column gain table (`&[f64]`) by reference and indexes it per output
/// element, run through `gemm_map` at `Parallelism::Rayon(8)` and compared bitwise to the same
/// table applied by hand on the plain-`gemm` result, so the borrow crosses into the workers intact
#[test]
fn closure_captures_slice() {
    let mut rng = Rng::new(0xCAB7_5111);
    let m = 40usize;
    let k = 33usize;
    let n = 24usize;
    let a = make::<f64>(&mut rng, m, k);
    let b = make::<f64>(&mut rng, k, n);
    let c0 = make::<f64>(&mut rng, m * n, 1);
    // the table `f` below borrows through `tref`, not owns
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

/// Worker partitioning never splits a single output element's reduction, so `gemm_map` stays
/// bit-identical across thread counts, the same guarantee plain `gemm` has. Compared directly
/// here (not only through the `check_map` oracle) on a driver shape and a small-`k` shape
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

/// `gemm_map` (thread-local pool) and `gemm_map_with` (caller-owned `Workspace`) are parallel
/// entry points; drives them against each other bit-for-bit on the same closure and inputs
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

/// `gemm_map_unchecked` and `gemm_map` are parallel entry points (the checked one does not call
/// the unchecked one internally); drives the raw fn against the checked twin bit-for-bit on a
/// driver shape, a small-`k` shape, and a gemv shape
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
            // SAFETY: valid in-bounds col-major views; C aliases neither A nor B
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

// validation panics: gemm_map runs the same view checks plain gemm does, same wording

mod validation {
    use super::*;

    #[test]
    #[should_panic(expected = "A.cols")]
    fn dim_mismatch() {
        let a = vec![1.0f32; 4 * 3];
        let b = vec![1.0f32; 4 * 4]; // B.rows == 4, A.cols == 3: mismatched
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
        // `a` is a raw view into the same storage `buf` (and C) uses, so the call must reject
        // the aliasing before any compute
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
        // rsc == 0 maps every row of a column onto the same cell: C aliases itself
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
