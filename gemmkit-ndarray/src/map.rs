//! User-defined per-element map-epilogue ndarray GEMM entries
use super::*;
use crate::common::dims_strides;

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in 1 fused pass: the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_map`]. The closure `f(value, row, col)` is applied to each output element at
/// its final value, with `(row, col)` in the **user** frame of `C`, fired exactly once per
/// element. `T` is `f32`/`f64` only. Like [`gemm`], it reads the pointer/strides directly and
/// forwards to gemmkit's raw engine, so C-order, F-order, general-stride, transposed, and
/// reversed (negative-stride) views all work without copying
///
/// For a bias / activation prefer [`gemm_fused`] (it vectorizes); `gemm_map` is the general
/// per-element extension point (GELU, sigmoid, clamps, position-dependent transforms), at the
/// cost of 1 indirect call per output element. For `f32`/`f64` the result is bit-identical to
/// [`gemm`] followed by mapping each `C[r, c]` through `f(C[r, c], r, c)`, for every shape
///
/// # Panics
/// Same conditions as [`gemm`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_map_common(None, alpha, a, b, beta, c, f, par);
}

/// Like [`gemm_map`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_map`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_map_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_map_common(Some(ws), alpha, a, b, beta, c, f, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_map_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    f: &(dyn Fn(T, usize, usize) -> T + Sync),
    par: Parallelism,
) where
    T: MapScalar,
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

    // SAFETY: dims validated above; ndarray guarantees the pointer/strides are in-bounds and `c`
    // (a `&mut` borrow) can't alias `a`/`b`; `f` is total (applied to every output element), which
    // is what the `_unchecked` closure contract requires
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
