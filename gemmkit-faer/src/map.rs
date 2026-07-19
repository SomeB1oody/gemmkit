//! User-defined per-element map-epilogue GEMM entries
use super::*;
use crate::common::ref_parts;

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in 1 fused pass, the faer adapter over gemmkit's
/// [`gemmkit::gemm_map`]. The closure `f(value, row, col)` runs on each output element at its final
/// value, with `(row, col)` in the **user** frame of `C`, exactly once per element. `T` is
/// `f32`/`f64` only: fusing a narrow-float closure after the `f32` accumulate would double-round,
/// so gemmkit's map bound leaves `f16`/`bf16` out. Like [`gemm`], it reads the pointer/strides
/// directly and forwards to gemmkit's raw engine, so transposed, sub-matrix, and reversed
/// (negative-stride) views all work without copying
///
/// For a bias/activation prefer [`gemm_fused`] (it vectorizes); `gemm_map` is the general
/// per-element extension point (GELU, sigmoid, clamps, position-dependent transforms), at the cost
/// of 1 indirect call per output element. The result is bit-identical to [`gemm`] followed by
/// mapping each `C[r, c]` through `f(C[r, c], r, c)`, for every shape
///
/// # Panics
/// If the inner dimensions disagree (same conditions as [`gemm`])
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map<T: MapScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    gemm_map_common(None, alpha, a, b, beta, c, f, par);
}

/// [`gemm_map`], threading a caller-owned [`Workspace`] through instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_map`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map_with<T: MapScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) {
    gemm_map_common(Some(ws), alpha, a, b, beta, c, f, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_map_common<T: MapScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
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

    // SAFETY: dims validated above; faer guarantees the pointer + element-unit `isize` strides
    // describe a valid in-bounds layout (negative for a reversed view, which the raw engine
    // handles), `c` (a `MatMut` exclusive borrow) can't alias `a`/`b`, and `f` is total (applied to
    // every output element, never skipped)
    unsafe {
        match ws {
            Some(ws) => gemm_map_unchecked_with(
                ws, m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, f, par,
            ),
            None => gemm_map_unchecked(
                m, k, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc, f, par,
            ),
        }
    }
}
