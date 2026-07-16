//! Prepacked-operand (PackedLhs/PackedRhs) entries
use super::*;
#[cfg(feature = "epilogue")]
use crate::common::lower_bias;
use crate::common::ref_parts;

/// Pre-pack a RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack once
/// here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this `B`.
/// Reads B's pointer/strides directly, so any layout works without copying
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    let (k, n, rsb, csb, bp) = ref_parts(b);
    // SAFETY: faer guarantees B's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_rhs_unchecked(bp, rsb, csb, k, n) }
}

/// `C <- alpha*A*B + beta*C` reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|col stride| >= |row stride|`): a row-major `C` would swap A/B and invalidate
/// the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout
///
/// # Panics
/// If the dimensions disagree, or if `C` is not column-major-ish
pub fn gemm_packed_b<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_b_common(None, alpha, a, packed, beta, c, par);
}

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`] (the fixed-cost inference loop)
///
/// # Panics
/// Same conditions as [`gemm_packed_b`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_b_common(Some(ws), alpha, a, packed, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-faer: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-faer: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();
    // SAFETY: faer guarantees A/C layouts are valid in-bounds; `c` (a `MatMut` exclusive borrow)
    // can't alias A, and the prepacked B is a separate owned buffer
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_unchecked_with(
                ws, alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, par,
            ),
            None => {
                gemm_packed_b_unchecked(alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, par)
            }
        }
    }
}

/// Pre-pack an LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of right
/// operands): pack once, then skip the per-call repack across many [`gemm_packed_a`] calls. Reads
/// A's pointer/strides directly, so any layout works without copying
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    // SAFETY: faer guarantees A's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_lhs_unchecked(ap, rsa, csa, m, k) }
}

/// `C <- alpha*A*B + beta*C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be row-major-ish
/// (`|col stride| <= |row stride|`): a column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout
///
/// # Panics
/// If the dimensions disagree, or if `C` is not row-major-ish
pub fn gemm_packed_a<T: GemmScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_a_common(None, alpha, packed, b, beta, c, par);
}

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_a`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    gemm_packed_a_common(Some(ws), alpha, packed, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_common<T: GemmScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-faer: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-faer: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();
    // SAFETY: faer guarantees B/C layouts are valid in-bounds; `c` (a `MatMut` exclusive borrow)
    // can't alias B, and the prepacked A is a separate owned buffer
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_unchecked_with(
                ws, alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, par,
            ),
            None => {
                gemm_packed_a_unchecked(alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, par)
            }
        }
    }
}

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in 1 fused pass, reusing a prepacked `B`
/// ([`prepack_rhs`]): the faer adapter over gemmkit's [`gemmkit::gemm_packed_b_fused`]. The **same**
/// [`PackedRhs`] serves both [`gemm_packed_b`] and this fused entry (the epilogue is store-side
/// only). `C` must be column-major-ish (`|col stride| >= |row stride|`); a row-major `C` would swap
/// A/B and invalidate the prepacked RHS, which gemmkit rejects. The optional [`Bias`] is
/// [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`) and the optional
/// [`Activation`] is applied last; `bias == None && act == None` matches [`gemm_packed_b`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not column-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused<T: FusedScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_packed_b_fused_common(None, alpha, a, packed, beta, c, bias, act, par);
}

/// Like [`gemm_packed_b_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_b_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_packed_b_fused_common(Some(ws), alpha, a, packed, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_fused_common<T: FusedScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: MatRef<'_, T>,
    packed: &PackedRhs<T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-faer: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-faer: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-faer: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its axis (PerRow == A.rows, PerCol == packed B.cols == C.cols) and does
    // not overlap C, and a LeakyRelu slope is finite. The packed path never swaps, so the
    // user-frame bias forwards unflipped
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, packed.cols(), cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; faer guarantees A/C layouts are valid in-bounds and `c` (a `MatMut`
    // exclusive borrow) can't alias A; the prepacked B is a separate owned buffer; the bias was
    // validated disjoint from C. The core `_unchecked` tier raises the column-major-ish-C panic
    unsafe {
        match ws {
            Some(ws) => gemm_packed_b_fused_unchecked_with(
                ws, alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, bias_ptr, bias_dim,
                has_bias, act, par,
            ),
            None => gemm_packed_b_fused_unchecked(
                alpha, m, ap, rsa, csa, packed, beta, cp, rsc, csc, bias_ptr, bias_dim, has_bias,
                act, par,
            ),
        }
    }
}

/// `C <- act(alpha*(prepacked A)*B + beta*C + bias)` in 1 fused pass, reusing a prepacked `A`
/// ([`prepack_lhs`]): the faer adapter over gemmkit's [`gemmkit::gemm_packed_a_fused`]. The **same**
/// [`PackedLhs`] serves both [`gemm_packed_a`] and this fused entry. `C` must be row-major-ish
/// (`|col stride| <= |row stride|`); a column-major `C` would keep A in the LHS role and invalidate
/// the prepacked LHS, which gemmkit rejects. The optional [`Bias`] is [`Bias::PerRow`] (length
/// `A.rows`) or [`Bias::PerCol`] (length `B.cols`), in the user frame; the optional [`Activation`]
/// is applied last; `bias == None && act == None` matches [`gemm_packed_a`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not row-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused<T: FusedScalar>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_packed_a_fused_common(None, alpha, packed, b, beta, c, bias, act, par);
}

/// Like [`gemm_packed_a_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_packed_a_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    gemm_packed_a_fused_common(Some(ws), alpha, packed, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_fused_common<T: FusedScalar>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let (kb, n, rsb, csb, bp) = ref_parts(b);
    let (cm, cn) = (c.nrows(), c.ncols());
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-faer: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-faer: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-faer: B.cols ({n}) != C.cols ({cn})");
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    let cp = c.as_ptr_mut();

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its USER axis (PerRow == packed A.rows == C.rows, PerCol == B.cols) and
    // does not overlap C, and a LeakyRelu slope is finite. The bias stays in the user frame; the
    // core `gemm_packed_a_fused` flips the axis to match the transposed consume it drives
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, packed.rows(), n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; faer guarantees B/C layouts are valid in-bounds and `c` (a `MatMut`
    // exclusive borrow) can't alias B; the prepacked A is a separate owned buffer; the bias was
    // validated disjoint from C. The core `_unchecked` tier raises the row-major-ish-C panic
    unsafe {
        match ws {
            Some(ws) => gemm_packed_a_fused_unchecked_with(
                ws, alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, bias_ptr, bias_dim,
                has_bias, act, par,
            ),
            None => gemm_packed_a_fused_unchecked(
                alpha, packed, n, bp, rsb, csb, beta, cp, rsc, csc, bias_ptr, bias_dim, has_bias,
                act, par,
            ),
        }
    }
}
