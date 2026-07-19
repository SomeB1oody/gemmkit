//! Batched ndarray GEMM entries: the batch dimension is axis 0 of a 3-D array, with every
//! element's dims/strides read straight out of the array (any layout, not just C-order)
use super::*;
#[cfg(feature = "epilogue")]
use crate::common::lower_bias;

#[inline]
fn dims_strides3<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix3>,
) -> (usize, usize, usize, isize, isize, isize) {
    let (b, r, c) = a.dim();
    let s = a.strides();
    (b, r, c, s[0], s[1], s[2])
}

/// Batched `C_e <- alpha*A_e*B_e + beta*C_e` for every element `e`, batch on **axis 0**: `a` is
/// `(batch, m, k)`, `b` is `(batch, k, n)`, `c` is `(batch, m, n)`. The axis-0 stride is each
/// operand's batch stride and axes 1/2 the element strides, read directly, so C-order, F-order,
/// and general-stride 3-D views all work without copying. The batch is parallelized (each
/// element runs wholly on 1 worker), so the result reproduces a loop of [`gemm`] calls
///
/// # Panics
/// If the batch sizes or inner dimensions disagree
pub fn gemm_batched<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_batched`] but reuses a caller-owned [`Workspace`]: no heap allocation after the
/// 1st sufficiently large call, for a stream of batched products
///
/// # Panics
/// Same conditions as [`gemm_batched`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_batched_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (batch, m, k, as0, as1, as2) = dims_strides3(a);
    let (bb, kb, n, bs0, bs1, bs2) = dims_strides3(b);
    let (cb, cm, cn, cs0, cs1, cs2) = dims_strides3(c);
    assert_eq!(
        batch, bb,
        "gemmkit-ndarray: A batch ({batch}) != B batch ({bb})"
    );
    assert_eq!(
        batch, cb,
        "gemmkit-ndarray: A batch ({batch}) != C batch ({cb})"
    );
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cp = c.as_mut_ptr();

    // SAFETY: ndarray guarantees each element's pointer/strides are in-bounds; `c` (a `&mut`
    // borrow) can't alias `a`/`b`, and its batch elements are pairwise-disjoint axis-0 slices of
    // 1 real array
    unsafe {
        match ws {
            Some(ws) => gemm_batched_unchecked_with(
                ws,
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                par,
            ),
            None => gemm_batched_unchecked(
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                par,
            ),
        }
    }
}

/// `A_e * B_e` for every batch element, into a fresh row-major `(batch, m, n)` [`Array3`]: the
/// batched analogue of [`dot`]
pub fn dot_batched<T, S1, S2>(a: &ArrayBase<S1, Ix3>, b: &ArrayBase<S2, Ix3>) -> Array3<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (batch, m, _) = a.dim();
    let (_, _, n) = b.dim();
    // beta is 0 here, so the fill value below is never read
    let mut c = Array3::from_elem((batch, m, n), T::ZERO);
    gemm_batched(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

/// Batched `C_e <- act(alpha*A_e*B_e + beta*C_e + bias)`, batch on **axis 0**, with **1 shared**
/// [`Bias`]/[`Activation`] applied to every element (the batched-linear-layer case): the ndarray
/// adapter over gemmkit's [`gemmkit::gemm_batched_fused`]. Shapes match [`gemm_batched`] (`a` is
/// `(batch, m, k)`, `b` is `(batch, k, n)`, `c` is `(batch, m, n)`); the bias is sized for a
/// **single** element (`PerRow` length `m`, `PerCol` length `n`), not the whole batch. Each
/// element reproduces a [`gemm_fused`] call, so `bias == None && act == None` is exactly
/// [`gemm_batched`]. Reads the pointers/strides directly and forwards to gemmkit's raw engine, so
/// general-stride and reversed (negative-stride) 3-D views all work without copying
///
/// # Panics
/// If the batch sizes or inner dimensions disagree, or on a bias/activation the adapter rejects (a
/// `PerRow`/`PerCol` bias of the wrong length, a bias slice overlapping `C`, or a non-finite
/// `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_fused<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// Like [`gemm_batched_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_batched_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_fused_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_batched_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_batched_fused_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix3>,
    b: &ArrayBase<S2, Ix3>,
    beta: T,
    c: &mut ArrayBase<SC, Ix3>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (batch, m, k, as0, as1, as2) = dims_strides3(a);
    let (bb, kb, n, bs0, bs1, bs2) = dims_strides3(b);
    let (cb, cm, cn, cs0, cs1, cs2) = dims_strides3(c);
    assert_eq!(
        batch, bb,
        "gemmkit-ndarray: A batch ({batch}) != B batch ({bb})"
    );
    assert_eq!(
        batch, cb,
        "gemmkit-ndarray: A batch ({batch}) != C batch ({cb})"
    );
    assert_eq!(k, kb, "gemmkit-ndarray: A.cols ({k}) != B.rows ({kb})");
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cp = c.as_mut_ptr();

    // Bias/activation validation matching gemmkit's checked-entry wording: the 1 shared bias is
    // sized for a single element (length m or n, not batch*axis) and must not overlap C's
    // whole-stack footprint (raw pointer math only, over all 3 axes); a LeakyRelu slope is finite
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, n, cp, &[(cb, cs0), (cm, cs1), (cn, cs2)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated above; ndarray guarantees each element's layout is in-bounds and `c`
    // (a `&mut` borrow) can't alias `a`/`b`, its batch elements being pairwise-disjoint axis-0
    // slices; the shared bias was validated disjoint from C above
    unsafe {
        match ws {
            Some(ws) => gemm_batched_fused_unchecked_with(
                ws,
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
            None => gemm_batched_fused_unchecked(
                batch,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                as1,
                as2,
                as0,
                b.as_ptr(),
                bs1,
                bs2,
                bs0,
                beta,
                cp,
                cs1,
                cs2,
                cs0,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
        }
    }
}
