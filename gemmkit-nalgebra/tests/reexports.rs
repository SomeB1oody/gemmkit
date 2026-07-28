//! Guard on the adapter's re-export surface. Nothing here may name a `gemmkit::` path: every
//! gemmkit item is reached through `gemmkit_nalgebra`, exactly as a downstream crate that depends
//! on this adapter alone would reach it. A gemmkit type that lands in a public signature without
//! being re-exported therefore fails to compile in this binary. The other test binaries in this
//! crate all import from `gemmkit::` directly and so cannot catch that, which is how the scalar
//! bounds stayed unexported while every entry taking them was already covered by tests

use gemmkit_nalgebra::{
    GemmScalar, PackedRhs, Parallelism, Workspace, gemm, gemm_packed_b, gemm_with, prepack_rhs,
    tuning,
};
use nalgebra::DMatrix;

// The wrapper a downstream crate writes over the plain real entries: it is generic over the
// element type, so it has to name the bound
fn wrap_gemm<T: GemmScalar>(a: &DMatrix<T>, b: &DMatrix<T>, c: &mut DMatrix<T>) {
    gemm(T::ONE, a, b, T::ZERO, c, Parallelism::Serial);
}

// The same over the workspace-reusing twin, the prepack constructor, and its consumer
fn wrap_packed<T: GemmScalar>(a: &DMatrix<T>, b: &DMatrix<T>, c: &mut DMatrix<T>) {
    let mut ws = Workspace::new();
    gemm_with(&mut ws, T::ONE, a, b, T::ZERO, c, Parallelism::Serial);
    let packed: PackedRhs<T> = prepack_rhs(b);
    gemm_packed_b(T::ONE, a, &packed, T::ZERO, c, Parallelism::Serial);
}

#[test]
fn generic_over_the_real_bound() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);
    let exp = DMatrix::from_row_slice(2, 2, &[19.0_f32, 22.0, 43.0, 50.0]);
    let mut c = DMatrix::<f32>::zeros(2, 2);
    wrap_gemm(&a, &b, &mut c);
    assert_eq!(c, exp);

    // a 2nd instantiation, so the bound is exercised generically rather than at 1 fixed type
    let a64 = DMatrix::from_row_slice(2, 2, &[1.0_f64, 2.0, 3.0, 4.0]);
    let b64 = DMatrix::from_row_slice(2, 2, &[5.0_f64, 6.0, 7.0, 8.0]);
    let mut c64 = DMatrix::<f64>::zeros(2, 2);
    wrap_gemm(&a64, &b64, &mut c64);
    assert_eq!(
        c64,
        DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0])
    );

    // DMatrix is column-major, the layout the prepacked consumer requires
    let mut c = DMatrix::<f32>::zeros(2, 2);
    wrap_packed(&a, &b, &mut c);
    assert_eq!(c, exp);
}

#[cfg(feature = "epilogue")]
mod epilogue {
    use super::*;
    use gemmkit_nalgebra::{Activation, Bias, FusedScalar, MapScalar, gemm_fused, gemm_map};

    fn wrap_fused<T: FusedScalar>(a: &DMatrix<T>, b: &DMatrix<T>, c: &mut DMatrix<T>, bias: &[T]) {
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
        a: &DMatrix<T>,
        b: &DMatrix<T>,
        c: &mut DMatrix<T>,
        f: &(dyn Fn(T, usize, usize) -> T + Sync),
    ) {
        gemm_map(T::ONE, a, b, T::ZERO, c, f, Parallelism::Serial);
    }

    #[test]
    fn generic_over_the_fused_bounds() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
        let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);

        // row 0 gets -100 added and clamps to 0 under relu, row 1 gets +1
        let mut c = DMatrix::<f32>::zeros(2, 2);
        wrap_fused(&a, &b, &mut c, &[-100.0, 1.0]);
        assert_eq!(c, DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 44.0, 51.0]));

        let mut c = DMatrix::<f32>::zeros(2, 2);
        wrap_map(&a, &b, &mut c, &|v, _, _| v * 2.0);
        assert_eq!(c, DMatrix::from_row_slice(2, 2, &[38.0, 44.0, 86.0, 100.0]));
    }
}

#[cfg(feature = "complex")]
mod complex {
    use super::*;
    use gemmkit_nalgebra::{Complex, ComplexScalar, c32, c64, gemm_cplx};

    fn wrap_cplx<T: ComplexScalar>(a: &DMatrix<T>, b: &DMatrix<T>, c: &mut DMatrix<T>) {
        gemm_cplx(T::ONE, a, false, b, false, T::ZERO, c, Parallelism::Serial);
    }

    #[test]
    fn generic_over_the_complex_bound() {
        // (i) * (i) = -1 summed over 2 terms, at both precisions, so the aliases are named too
        let a = DMatrix::from_element(2, 2, c32::new(0.0, 1.0));
        let mut c = DMatrix::from_element(2, 2, c32::new(0.0, 0.0));
        wrap_cplx(&a, &a, &mut c);
        assert_eq!(c[(0, 0)], Complex::new(-2.0, 0.0));

        let a = DMatrix::from_element(2, 2, c64::new(0.0, 1.0));
        let mut c = DMatrix::from_element(2, 2, c64::new(0.0, 0.0));
        wrap_cplx(&a, &a, &mut c);
        assert_eq!(c[(0, 0)], Complex::new(-2.0, 0.0));
    }
}

#[cfg(feature = "half")]
#[test]
fn generic_over_the_narrow_element_types() {
    use gemmkit_nalgebra::{bf16, f16};

    let a = DMatrix::from_element(4, 4, f16::from_f32(1.0));
    let mut c = DMatrix::from_element(4, 4, f16::from_f32(0.0));
    wrap_gemm(&a, &a, &mut c);
    assert_eq!(c[(0, 0)], f16::from_f32(4.0));

    let b = DMatrix::from_element(4, 4, bf16::from_f32(1.0));
    let mut c = DMatrix::from_element(4, 4, bf16::from_f32(0.0));
    wrap_gemm(&b, &b, &mut c);
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
