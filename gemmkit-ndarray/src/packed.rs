//! Prepacked-operand (PackedLhs/PackedRhs) ndarray GEMM entries
use super::*;
use crate::common::dims_strides;
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// Pre-pack a 2-D RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path):
/// pack once here, then skip the per-call repack across many [`gemm_packed_b`] calls that share
/// this `B`. Reads B's pointer/strides directly, so any layout works without copying
pub fn prepack_rhs<T, S>(b: &ArrayBase<S, Ix2>) -> PackedRhs<T>
where
    T: GemmScalar,
    S: Data<Elem = T>,
{
    let (k, n, rsb, csb) = dims_strides(b);
    // SAFETY: ndarray guarantees B's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_rhs_unchecked(b.as_ptr(), rsb, csb, k, n) }
}

/// `C <- alpha*A*B + beta*C` reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|col stride| >= |row stride|`): a row-major `C` would swap A/B and
/// invalidate the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout
///
/// # Panics
/// If the dimensions disagree, or if `C` is not column-major-ish
pub fn gemm_packed_b<T, S1, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_common(None, alpha, a, packed, beta, c, par);
}

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`]: no per-call allocation, for a
/// fixed-weight inference loop
///
/// # Panics
/// Same conditions as [`gemm_packed_b`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_with<T, S1, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_common(Some(ws), alpha, a, packed, beta, c, par);
}

fn gemm_packed_b_common<T, S1, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (cm, cn) = c.dim();
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-ndarray: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-ndarray: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();
    // SAFETY: dims validated above; ndarray guarantees A/C layouts are in-bounds; `c` (a `&mut`
    // borrow) can't alias A, and the prepacked B is a separate owned buffer
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_unchecked_with(
                ws,
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_packed_b_unchecked(
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Pre-pack a 2-D LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of
/// differently-shaped `B` operands): pack once, then skip the per-call repack across many
/// [`gemm_packed_a`] calls. Reads A's pointer/strides directly, so any layout works without
/// copying
pub fn prepack_lhs<T, S>(a: &ArrayBase<S, Ix2>) -> PackedLhs<T>
where
    T: GemmScalar,
    S: Data<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    // SAFETY: ndarray guarantees A's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_lhs_unchecked(a.as_ptr(), rsa, csa, m, k) }
}

/// `C <- alpha*A*B + beta*C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be
/// row-major-ish (`|col stride| <= |row stride|`): a column-major `C` would keep A in the LHS
/// role and invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout
///
/// # Panics
/// If the dimensions disagree, or if `C` is not row-major-ish
pub fn gemm_packed_a<T, S2, SC>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_common(None, alpha, packed, b, beta, c, par);
}

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`]: no per-call allocation
///
/// # Panics
/// Same conditions as [`gemm_packed_a`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_with<T, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_common(Some(ws), alpha, packed, b, beta, c, par);
}

fn gemm_packed_a_common<T, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-ndarray: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-ndarray: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();
    // SAFETY: dims validated above; ndarray guarantees B/C layouts are in-bounds; `c` (a `&mut`
    // borrow) can't alias B, and the prepacked A is a separate owned buffer
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_unchecked_with(
                ws,
                alpha,
                packed,
                n,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_packed_a_unchecked(
                alpha,
                packed,
                n,
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

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in 1 fused pass, reusing a prepacked `B`
/// ([`prepack_rhs`]): the ndarray adapter over gemmkit's [`gemmkit::gemm_packed_b_fused`]. The
/// **same** [`PackedRhs`] serves both [`gemm_packed_b`] and this fused entry (the epilogue is
/// store-side only). `C` must be column-major-ish (`|col stride| >= |row stride|`); a row-major
/// `C` would swap A/B and invalidate the prepacked RHS, which gemmkit rejects. The optional
/// [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`) and the
/// optional [`Activation`] is applied last; `bias == None && act == None` matches
/// [`gemm_packed_b`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not column-major-ish, or on a bias/activation the
/// adapter rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu`
/// slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused<T, S1, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_fused_common(None, alpha, a, packed, beta, c, bias, act, par);
}

/// Like [`gemm_packed_b_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_b_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused_with<T, S1, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_b_fused_common(Some(ws), alpha, a, packed, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_fused_common<T, S1, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (cm, cn) = c.dim();
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-ndarray: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-ndarray: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-ndarray: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();

    // Bias/activation validation matching gemmkit's checked-entry wording: the bias length
    // matches its axis (PerRow == A.rows, PerCol == packed B.cols == C.cols) and does not overlap
    // C (raw pointer math only), and a LeakyRelu slope must be finite. The packed-B path never
    // swaps A/B, so the user-frame bias passes through with no axis flip
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, packed.cols(), cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated above; ndarray guarantees A/C layouts are in-bounds and `c` (a
    // `&mut` borrow) can't alias A; the prepacked B is a separate owned buffer; the bias was
    // validated disjoint from C above. The `_unchecked` tier below raises the
    // column-major-ish-C panic if C is not column-major-ish
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_fused_unchecked_with(
                ws,
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
            None => gemm_packed_b_fused_unchecked(
                alpha,
                m,
                a.as_ptr(),
                rsa,
                csa,
                packed,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
        }
    }
}

/// `C <- act(alpha*(prepacked A)*B + beta*C + bias)` in 1 fused pass, reusing a prepacked `A`
/// ([`prepack_lhs`]): the ndarray adapter over gemmkit's [`gemmkit::gemm_packed_a_fused`]. The
/// **same** [`PackedLhs`] serves both [`gemm_packed_a`] and this fused entry. `C` must be
/// row-major-ish (`|col stride| <= |row stride|`); a column-major `C` would keep A in the LHS
/// role and invalidate the prepacked LHS, which gemmkit rejects. The optional [`Bias`] is
/// [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), specified in the
/// user frame; the optional [`Activation`] is applied last; `bias == None && act == None`
/// matches [`gemm_packed_a`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not row-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused<T, S2, SC>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_fused_common(None, alpha, packed, b, beta, c, bias, act, par);
}

/// Like [`gemm_packed_a_fused`] but reuses a caller-owned [`Workspace`]: no per-call allocation
///
/// # Panics
/// Same conditions as [`gemm_packed_a_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused_with<T, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_packed_a_fused_common(Some(ws), alpha, packed, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_fused_common<T, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.dim();
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-ndarray: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-ndarray: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-ndarray: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs[0], cs[1]);
    let cp = c.as_mut_ptr();

    // Bias/activation validation matching gemmkit's checked-entry wording: the bias length
    // matches its USER axis (PerRow == packed A.rows == C.rows, PerCol == B.cols) and does not
    // overlap C, and a LeakyRelu slope must be finite. The bias stays in the user frame here;
    // gemmkit's packed-A path flips it internally to match the transposed product it drives
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, packed.rows(), n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated above; ndarray guarantees B/C layouts are in-bounds and `c` (a
    // `&mut` borrow) can't alias B; the prepacked A is a separate owned buffer; the bias was
    // validated disjoint from C above. The `_unchecked` tier below raises the row-major-ish-C
    // panic if C is not row-major-ish
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_fused_unchecked_with(
                ws,
                alpha,
                packed,
                n,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
            None => gemm_packed_a_fused_unchecked(
                alpha,
                packed,
                n,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
        }
    }
}
