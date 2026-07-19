//! The plain real-scalar `gemm`/`gemm_with`/`dot` entries: `f32`/`f64` always, plus `f16`/`bf16`
//! under the `half` feature (any `T: GemmScalar`)
use super::*;
use crate::common::{dims_strides, filled_dmatrix};

/// `C <- alpha*A*B + beta*C`
///
/// # Panics
/// If the inner dimensions disagree
pub fn gemm<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// As [`gemm`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool [`gemm`]
/// draws from
///
/// # Panics
/// If the inner dimensions disagree
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.shape();
    assert_eq!(k, kb, "gemmkit-nalgebra: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-nalgebra: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-nalgebra: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();

    // SAFETY: dims checked above; nalgebra guarantees the storage's pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) can't alias `a`/`b`
    unsafe {
        match ws {
            Some(ws) => gemm_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// `A*B` into a fresh column-major [`DMatrix`]: a `.dot()`-style convenience over [`gemm`]
pub fn dot<T, R1, C1, S1, R2, C2, S2>(
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
) -> DMatrix<T>
where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta = 0, so gemm overwrites every cell; the fill value is never read
    let mut c = filled_dmatrix(m, n, T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}
