//! Guard on the adapter's re-export surface. Nothing here may name a `gemmkit::` path: every
//! gemmkit item is reached through `gemmkit_ndarray`, exactly as a downstream crate that depends
//! on this adapter alone would reach it. A gemmkit type that lands in a public signature without
//! being re-exported therefore fails to compile in this binary. The other test binaries in this
//! crate all import from `gemmkit::` directly and so cannot catch that, which is how the scalar
//! bounds stayed unexported while every entry taking them was already covered by tests

use gemmkit_ndarray::{
    GemmScalar, PackedRhs, Parallelism, Workspace, gemm, gemm_packed_b, gemm_with, prepack_rhs,
    tuning,
};
use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2, ShapeBuilder, array};

// The wrapper a downstream crate writes over the plain real entries: it is generic over the
// element type, so it has to name the bound
fn wrap_gemm<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::Serial);
    c
}

// The same over the workspace-reusing twin, the prepack constructor, and its consumer
fn wrap_packed<T, S1, S2, SC>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    c: &mut ArrayBase<SC, Ix2>,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let mut ws = Workspace::new();
    gemm_with(&mut ws, T::ONE, a, b, T::ZERO, c, Parallelism::Serial);
    let packed: PackedRhs<T> = prepack_rhs(b);
    gemm_packed_b(T::ONE, a, &packed, T::ZERO, c, Parallelism::Serial);
}

#[test]
fn generic_over_the_real_bound() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
    assert_eq!(wrap_gemm(&a, &b), array![[19.0, 22.0], [43.0, 50.0]]);

    // a 2nd instantiation, so the bound is exercised generically rather than at 1 fixed type
    let a64 = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let b64 = array![[5.0_f64, 6.0], [7.0, 8.0]];
    assert_eq!(wrap_gemm(&a64, &b64), array![[19.0, 22.0], [43.0, 50.0]]);

    // C is column-major here, the layout the prepacked consumer requires
    let mut c = Array2::<f32>::zeros((2, 2).f());
    wrap_packed(&a, &b, &mut c);
    assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
}

#[cfg(feature = "epilogue")]
mod epilogue {
    use super::*;
    use gemmkit_ndarray::{Activation, Bias, FusedScalar, MapScalar, gemm_fused, gemm_map};

    fn wrap_fused<T: FusedScalar>(a: &Array2<T>, b: &Array2<T>, bias: &[T]) -> Array2<T> {
        let (m, _) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::from_elem((m, n), T::ZERO);
        gemm_fused(
            T::ONE,
            a,
            b,
            T::ZERO,
            &mut c,
            Some(Bias::PerRow(bias)),
            Some(Activation::Relu),
            Parallelism::Serial,
        );
        c
    }

    fn wrap_map<T: MapScalar>(
        a: &Array2<T>,
        b: &Array2<T>,
        f: &(dyn Fn(T, usize, usize) -> T + Sync),
    ) -> Array2<T> {
        let (m, _) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::from_elem((m, n), T::ZERO);
        gemm_map(T::ONE, a, b, T::ZERO, &mut c, f, Parallelism::Serial);
        c
    }

    #[test]
    fn generic_over_the_fused_bounds() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
        // row 0 gets -100 added and clamps to 0 under relu, row 1 gets +1
        assert_eq!(
            wrap_fused(&a, &b, &[-100.0, 1.0]),
            array![[0.0, 0.0], [44.0, 51.0]]
        );
        assert_eq!(
            wrap_map(&a, &b, &|v, _, _| v * 2.0),
            array![[38.0, 44.0], [86.0, 100.0]]
        );
    }
}

#[cfg(feature = "complex")]
mod complex {
    use super::*;
    use gemmkit_ndarray::{Complex, ComplexScalar, c32, c64, gemm_cplx};

    fn wrap_cplx<T: ComplexScalar>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
        let (m, _) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::from_elem((m, n), T::ZERO);
        gemm_cplx(
            T::ONE,
            a,
            false,
            b,
            false,
            T::ZERO,
            &mut c,
            Parallelism::Serial,
        );
        c
    }

    #[test]
    fn generic_over_the_complex_bound() {
        // (i) * (i) = -1, at both precisions, so the alias names are exercised too
        let a32 = Array2::<c32>::from_elem((2, 2), Complex::new(0.0, 1.0));
        assert_eq!(wrap_cplx(&a32, &a32)[(0, 0)], Complex::new(-2.0, 0.0));
        let a64 = Array2::<c64>::from_elem((2, 2), Complex::new(0.0, 1.0));
        assert_eq!(wrap_cplx(&a64, &a64)[(0, 0)], Complex::new(-2.0, 0.0));
    }
}

#[cfg(feature = "half")]
#[test]
fn generic_over_the_narrow_element_types() {
    use gemmkit_ndarray::{bf16, f16};

    let a = Array2::from_elem((4, 4), f16::from_f32(1.0));
    assert_eq!(wrap_gemm(&a, &a)[(0, 0)], f16::from_f32(4.0));
    let b = Array2::from_elem((4, 4), bf16::from_f32(1.0));
    assert_eq!(wrap_gemm(&b, &b)[(0, 0)], bf16::from_f32(4.0));
}

#[test]
fn the_knob_surface_is_reachable() {
    // Read-only: the knobs are process-global atomics, so the setters stay in the binaries that
    // own them (see the tuning tests in the core crate)
    assert!(tuning::parallel_threshold() > 0);
    assert_eq!(tuning::PARALLEL_THRESHOLD_DEFAULT, 48 * 48 * 256);
    assert!(!tuning::knob_env_names().is_empty());
}
