//! GEMM entries for gemmkit's homogeneous scalar types: f32/f64 unconditionally, plus f16/bf16
//! under the half feature
use super::*;
use crate::common::{filled_mat, ref_parts};

/// `C <- alpha*A*B + beta*C`
///
/// # Panics
/// If the inner dimensions disagree
pub fn gemm<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// [`gemm`], threading a caller-owned [`Workspace`] through instead of the thread-local pool
///
/// # Panics
/// If the inner dimensions disagree
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(k, kb, "gemmkit-faer: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();

    // SAFETY: dims validated above; faer's `MatRef`/`MatMut` guarantee the pointer + element-unit
    // `isize` strides describe a valid in-bounds layout (possibly negative for a reversed view,
    // which gemmkit's unchecked path handles), and `c` (a `MatMut` exclusive borrow) can't alias A/B
    unsafe {
        match ws {
            Some(ws) => gemm_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
            None => gemm_unchecked(
                m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
        }
    }
}

/// `A*B` into a fresh column-major [`Mat`] (a `.dot()`-style convenience over [`gemm`])
pub fn dot<T: GemmScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    // beta == 0, so the fill value below is never read
    let mut c = filled_mat(m, n, T::ZERO);
    gemm(
        T::ONE,
        a,
        b,
        T::ZERO,
        c.as_dyn_stride_mut(),
        Parallelism::default(),
    );
    c
}
