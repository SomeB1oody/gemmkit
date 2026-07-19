//! Real f32/f64 (plus f16/bf16 under the half feature) ndarray GEMM entries
use super::*;
use crate::common::dims_strides;

/// `C <- alpha*A*B + beta*C`. Reads the pointer/strides directly out of `a`/`b`/`c`, so
/// C-order, F-order, general-stride, and reversed (negative-stride) views all work without
/// copying
///
/// # Panics
/// If the inner dimensions disagree
pub fn gemm<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();

    // SAFETY: dims validated above; ndarray guarantees the pointer/strides are in-bounds, and
    // `c` (a `&mut` borrow) cannot alias `a`/`b`
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

/// `A*B` into a fresh row-major [`Array2`]: the `.dot()`-style convenience over [`gemm`]
pub fn dot<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta is 0 here, so the fill value below is never read
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}
