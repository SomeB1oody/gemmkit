//! Guard on the adapter's re-export surface. Nothing here may name a `gemmkit::` path: every
//! gemmkit item is reached through `gemmkit_faer`, exactly as a downstream crate that depends on
//! this adapter alone would reach it. A gemmkit type that lands in a public signature without
//! being re-exported therefore fails to compile in this binary. The other test binaries in this
//! crate all import from `gemmkit::` directly and so cannot catch that, which is how the scalar
//! bounds stayed unexported while every entry taking them was already covered by tests

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use gemmkit_faer::{
    GemmScalar, PackedRhs, Parallelism, Workspace, gemm, gemm_packed_b, gemm_with, prepack_rhs,
    tuning,
};

// The wrapper a downstream crate writes over the plain real entries: it is generic over the
// element type, so it has to name the bound
fn wrap_gemm<T: GemmScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, c: MatMut<'_, T>) {
    gemm(T::ONE, a, b, T::ZERO, c, Parallelism::Serial);
}

// The same over the workspace-reusing twin, the prepack constructor, and its consumer
fn wrap_packed<T: GemmScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, mut c: MatMut<'_, T>) {
    let mut ws = Workspace::new();
    gemm_with(
        &mut ws,
        T::ONE,
        a,
        b,
        T::ZERO,
        c.rb_mut(),
        Parallelism::Serial,
    );
    let packed: PackedRhs<T> = prepack_rhs(b);
    gemm_packed_b(T::ONE, a, &packed, T::ZERO, c, Parallelism::Serial);
}

#[test]
fn generic_over_the_real_bound() {
    let a = Mat::from_fn(2, 2, |i, j| [[1.0_f32, 2.0], [3.0, 4.0]][i][j]);
    let b = Mat::from_fn(2, 2, |i, j| [[5.0_f32, 6.0], [7.0, 8.0]][i][j]);
    let exp = [[19.0_f32, 22.0], [43.0, 50.0]];

    let mut c = Mat::<f32>::from_fn(2, 2, |_, _| 0.0);
    wrap_gemm(a.as_dyn_stride(), b.as_dyn_stride(), c.as_dyn_stride_mut());
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(c[(i, j)], exp[i][j]);
        }
    }

    // a 2nd instantiation, so the bound is exercised generically rather than at 1 fixed type
    let a64 = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
    let b64 = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
    let mut c64 = Mat::<f64>::from_fn(2, 2, |_, _| 0.0);
    wrap_gemm(
        a64.as_dyn_stride(),
        b64.as_dyn_stride(),
        c64.as_dyn_stride_mut(),
    );
    assert_eq!(c64[(1, 1)], 50.0);

    // a faer Mat is column-major, the layout the prepacked consumer requires
    let mut c = Mat::<f32>::from_fn(2, 2, |_, _| 0.0);
    wrap_packed(a.as_dyn_stride(), b.as_dyn_stride(), c.as_dyn_stride_mut());
    assert_eq!(c[(1, 1)], 50.0);
}

#[cfg(feature = "epilogue")]
mod epilogue {
    use super::*;
    use gemmkit_faer::{Activation, Bias, FusedScalar, MapScalar, gemm_fused, gemm_map};

    fn wrap_fused<T: FusedScalar>(
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
        bias: &[T],
    ) {
        gemm_fused(
            T::ONE,
            a,
            b,
            T::ZERO,
            c,
            Some(Bias::PerRow(bias)),
            Some(Activation::Relu),
            Parallelism::Serial,
        );
    }

    fn wrap_map<T: MapScalar>(
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        c: MatMut<'_, T>,
        f: &(dyn Fn(T, usize, usize) -> T + Sync),
    ) {
        gemm_map(T::ONE, a, b, T::ZERO, c, f, Parallelism::Serial);
    }

    #[test]
    fn generic_over_the_fused_bounds() {
        let a = Mat::from_fn(2, 2, |i, j| [[1.0_f32, 2.0], [3.0, 4.0]][i][j]);
        let b = Mat::from_fn(2, 2, |i, j| [[5.0_f32, 6.0], [7.0, 8.0]][i][j]);

        // row 0 gets -100 added and clamps to 0 under relu, row 1 gets +1
        let mut c = Mat::<f32>::from_fn(2, 2, |_, _| 0.0);
        wrap_fused(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            c.as_dyn_stride_mut(),
            &[-100.0, 1.0],
        );
        assert_eq!(c[(0, 0)], 0.0);
        assert_eq!(c[(1, 1)], 51.0);

        let mut c = Mat::<f32>::from_fn(2, 2, |_, _| 0.0);
        wrap_map(
            a.as_dyn_stride(),
            b.as_dyn_stride(),
            c.as_dyn_stride_mut(),
            &|v, _, _| v * 2.0,
        );
        assert_eq!(c[(1, 1)], 100.0);
    }
}

#[cfg(feature = "complex")]
mod complex {
    use super::*;
    use gemmkit_faer::{Complex, ComplexScalar, c32, c64, gemm_cplx};

    fn wrap_cplx<T: ComplexScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, c: MatMut<'_, T>) {
        gemm_cplx(T::ONE, a, false, b, false, T::ZERO, c, Parallelism::Serial);
    }

    #[test]
    fn generic_over_the_complex_bound() {
        // (i) * (i) = -1 summed over 2 terms, at both precisions, so the aliases are named too
        let a = Mat::from_fn(2, 2, |_, _| c32::new(0.0, 1.0));
        let mut c = Mat::from_fn(2, 2, |_, _| c32::new(0.0, 0.0));
        wrap_cplx(a.as_dyn_stride(), a.as_dyn_stride(), c.as_dyn_stride_mut());
        assert_eq!(c[(0, 0)], Complex::new(-2.0, 0.0));

        let a = Mat::from_fn(2, 2, |_, _| c64::new(0.0, 1.0));
        let mut c = Mat::from_fn(2, 2, |_, _| c64::new(0.0, 0.0));
        wrap_cplx(a.as_dyn_stride(), a.as_dyn_stride(), c.as_dyn_stride_mut());
        assert_eq!(c[(0, 0)], Complex::new(-2.0, 0.0));
    }
}

#[cfg(feature = "half")]
#[test]
fn generic_over_the_narrow_element_types() {
    use gemmkit_faer::{bf16, f16};

    let a = Mat::from_fn(4, 4, |_, _| f16::from_f32(1.0));
    let mut c = Mat::from_fn(4, 4, |_, _| f16::from_f32(0.0));
    wrap_gemm(a.as_dyn_stride(), a.as_dyn_stride(), c.as_dyn_stride_mut());
    assert_eq!(c[(0, 0)], f16::from_f32(4.0));

    let b = Mat::from_fn(4, 4, |_, _| bf16::from_f32(1.0));
    let mut c = Mat::from_fn(4, 4, |_, _| bf16::from_f32(0.0));
    wrap_gemm(b.as_dyn_stride(), b.as_dyn_stride(), c.as_dyn_stride_mut());
    assert_eq!(c[(0, 0)], bf16::from_f32(4.0));
}

#[test]
fn the_knob_surface_is_reachable() {
    // Read-only: the knobs are process-global atomics, so the setters stay in the binaries that
    // own them (see the tuning tests in the core crate)
    assert!(tuning::parallel_threshold() > 0);
    assert_eq!(tuning::PARALLEL_THRESHOLD_DEFAULT, 48 * 48 * 256);
    assert!(!tuning::knob_env_names().is_empty());
}
