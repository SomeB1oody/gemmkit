//! Per-element closure fused into the GEMM store, over gemmkit's `gemm_map`
use super::*;
use crate::common::dims_strides;

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in **1 fused pass**: the nalgebra adapter over
/// gemmkit's [`gemmkit::gemm_map`]. The closure `f(value, row, col)` runs on each output element's
/// final value exactly once, with `(row, col)` in the **user** frame of `C`. `T` is `f32`/`f64`
/// only. As with [`gemm`], the pointer/strides are read directly and forwarded to gemmkit's raw
/// engine, so column-major, transposed (`.transpose()`), and general-stride (`.rows_with_step()`)
/// views all work without copying
///
/// For a bias / activation, prefer [`gemm_fused`] instead (it vectorizes); `gemm_map` is the
/// general per-element extension point (GELU, sigmoid, clamps, position-dependent transforms), at
/// the cost of 1 indirect call per output element. For `f32`/`f64`, the result is bit-identical to
/// running [`gemm`] and then mapping each `C[r, c]` through `f(C[r, c], r, c)`, for every shape
///
/// # Panics
/// If the inner dimensions disagree (same conditions as [`gemm`])
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
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
    gemm_map_common(None, alpha, a, b, beta, c, f, par);
}

/// As [`gemm_map`], but reuses a caller-owned [`Workspace`] instead of the thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_map`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
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
    gemm_map_common(Some(ws), alpha, a, b, beta, c, f, par);
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "epilogue")]
fn gemm_map_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
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

    // SAFETY: dims checked above; nalgebra guarantees the pointer/strides describe a valid
    // in-bounds layout and `c` (a `&mut` borrow) can't alias `a`/`b`; `f` is total, so it accepts
    // every element the engine stores
    unsafe {
        match ws {
            Some(ws) => gemm_map_unchecked_with(
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
                f,
                par,
            ),
            None => gemm_map_unchecked(
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
                f,
                par,
            ),
        }
    }
}
