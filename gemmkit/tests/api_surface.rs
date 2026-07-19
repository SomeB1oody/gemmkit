//! Coverage for public API surface that a correctness sweep would not otherwise reach: the
//! [`MatRef`]/[`MatMut`] shape accessors, the [`Workspace`] constructors, and the raw
//! `gemm_unchecked_with` entry point over a caller-owned workspace
#![cfg(all(not(miri), not(target_family = "wasm")))]

use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm, gemm_unchecked_with};

/// `rows()`/`cols()` report the logical shape passed to a view's constructor, regardless of
/// which constructor built it or what strides it carries
#[test]
fn matref_matmut_accessors() {
    let data = [0.0f32; 12];
    let r = MatRef::from_row_major(&data, 3, 4);
    assert_eq!(r.rows(), 3);
    assert_eq!(r.cols(), 4);

    // MatRef::new takes explicit strides; the shape it reports is independent of them
    let r2 = MatRef::new(&data, 4, 3, 1, 4);
    assert_eq!((r2.rows(), r2.cols()), (4, 3));

    let mut md = [0.0f32; 12];
    let m = MatMut::from_col_major(&mut md, 4, 3);
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 3);

    let mut md2 = [0.0f32; 6];
    let m2 = MatMut::new(&mut md2, 2, 3, 3, 1);
    assert_eq!((m2.rows(), m2.cols()), (2, 3));
}

/// An empty `Workspace` (`default()`, `new()`, or `with_capacity(0)`) and a pre-sized one both
/// drive `gemm_unchecked_with` to the same result as the safe, allocating `gemm` entry
#[test]
fn workspace_constructors_and_unchecked_with() {
    let (m, k, n) = (48usize, 40, 32);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
    let c0: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 * 0.05).collect();
    let (alpha, beta) = (1.25f32, -0.5);

    // Reference result via the safe entry (pool-allocated workspace)
    let mut c_ref = c0.clone();
    gemm(
        alpha,
        MatRef::from_row_major(&a, m, k),
        MatRef::from_row_major(&b, k, n),
        beta,
        MatMut::from_row_major(&mut c_ref, m, n),
        Parallelism::Serial,
    );

    // default() and with_capacity(0) both construct empty; with_capacity(0) must not itself allocate
    let mut ws_default = Workspace::default();
    let _ws_zero = Workspace::with_capacity(0);
    // Primed above what this shape needs, so the run below never grows it
    let mut ws_cap = Workspace::with_capacity(1 << 20);

    for ws in [&mut ws_default, &mut ws_cap] {
        let mut c = c0.clone();
        // SAFETY: row-major strides are in bounds for m/k/n, and C (its own Vec) aliases
        // neither A nor B
        unsafe {
            gemm_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                beta,
                c.as_mut_ptr(),
                n as isize,
                1,
                Parallelism::Serial,
            );
        }
        assert_eq!(
            c, c_ref,
            "gemm_unchecked_with must match the safe gemm entry"
        );
    }
}
