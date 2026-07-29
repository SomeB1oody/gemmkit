//! Prepacked-operand (PackedLhs/PackedRhs) GEMM entries: pack an operand once, reuse it across
//! many calls
use super::*;
use crate::common::ref_parts;
#[cfg(feature = "epilogue")]
use gemmkit::adapter::lower_bias;

/// Pre-pack a RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path). It
/// packs `B` once, so every later [`gemm_packed_b`] call that shares this `B` skips its own
/// repack
///
/// This function reads `B`'s pointer and strides directly. It packs any layout without copying
/// it into a canonical form first
pub fn prepack_rhs<T: GemmScalar>(b: MatRef<'_, T>) -> PackedRhs<T> {
    let (k, n, rsb, csb, bp) = ref_parts(b);
    // SAFETY: faer guarantees B's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_rhs_unchecked(bp, rsb, csb, k, n) }
}

/// `C <- alpha*A*B + beta*C`, reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|csc| >= |rsc|`). A row-major `C` would make the engine swap the A/B
/// roles, which the prepacked RHS cannot follow. Use [`gemm`] for that layout instead
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

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool. The caller-owned workspace avoids a fixed allocation cost in a repeated inference loop
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
    // SAFETY: faer guarantees that A and C are valid in-bounds layouts. `c` is a `MatMut`
    // exclusive borrow, so it cannot alias A, and `packed` owns its buffer independently of
    // A/C. The core `_unchecked` tier raises the column-major-ish-C panic
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

/// Pre-pack an LHS `A` into a reusable [`PackedLhs`], for a fixed `A` against a stream of
/// differently shaped `B` operands. It packs `A` once, so every later [`gemm_packed_a`] call
/// that shares this `A` skips its own repack
///
/// This function reads `A`'s pointer and strides directly. It packs any layout without copying
/// it into a canonical form first
pub fn prepack_lhs<T: GemmScalar>(a: MatRef<'_, T>) -> PackedLhs<T> {
    let (m, k, rsa, csa, ap) = ref_parts(a);
    // SAFETY: faer guarantees A's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_lhs_unchecked(ap, rsa, csa, m, k) }
}

/// `C <- alpha*A*B + beta*C`, reusing a prepacked `A` ([`prepack_lhs`]). `C` must be
/// row-major-ish (`|csc| <= |rsc|`). A column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS. Use [`gemm`] for that layout instead
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

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool
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
    // SAFETY: faer guarantees that B and C are valid in-bounds layouts. `c` is a `MatMut`
    // exclusive borrow, so it cannot alias B, and `packed` owns its buffer independently of
    // B/C. The core `_unchecked` tier raises the row-major-ish-C panic
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
/// ([`prepack_rhs`]). This is the faer adapter over gemmkit's [`gemmkit::gemm_packed_b_fused`].
/// The same [`PackedRhs`] serves both [`gemm_packed_b`] and this fused entry, because the
/// epilogue only changes how the result is stored
///
/// `C` must be column-major-ish (`|csc| >= |rsc|`). A row-major `C` would make the engine swap
/// the A/B roles, which the prepacked RHS cannot follow. The optional [`Bias`] is
/// [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), added before the
/// optional [`Activation`], which runs last. `bias == None && act == None` behaves exactly like
/// [`gemm_packed_b`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not column-major-ish, or if the adapter rejects the
/// bias or activation. Rejection conditions are a wrong-length bias, a bias slice that
/// overlaps `C`, or a non-finite `LeakyRelu` slope
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

/// Like [`gemm_packed_b_fused`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
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

    // Validates the bias and activation the same way gemmkit's checked entry does, with the
    // same panic wording. The bias length matches its axis (PerRow == A.rows, PerCol == packed
    // B.cols == C.cols), and it does not overlap C. A LeakyRelu slope must be finite. The
    // packed-B path never swaps A/B, so the user-frame bias forwards unflipped
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, packed.cols(), cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: the dims are validated above. faer guarantees that A and C are valid in-bounds
    // layouts. `c` is a `MatMut` exclusive borrow, so it cannot alias A. `packed` owns its
    // buffer independently of A/C, and the bias was validated disjoint from C above. The core
    // `_unchecked` tier raises the column-major-ish-C panic
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
/// ([`prepack_lhs`]). This is the faer adapter over gemmkit's [`gemmkit::gemm_packed_a_fused`].
/// The same [`PackedLhs`] serves both [`gemm_packed_a`] and this fused entry
///
/// `C` must be row-major-ish (`|csc| <= |rsc|`). A column-major `C` would keep A in the LHS
/// role and invalidate the prepacked LHS. The optional [`Bias`] is [`Bias::PerRow`] (length
/// `A.rows`) or [`Bias::PerCol`] (length `B.cols`), given in the user frame. The optional
/// [`Activation`] runs last, and `bias == None && act == None` behaves exactly like
/// [`gemm_packed_a`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not row-major-ish, or if the adapter rejects the bias
/// or activation. Rejection conditions are a wrong-length bias, a bias slice that overlaps `C`,
/// or a non-finite `LeakyRelu` slope
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

/// Like [`gemm_packed_a_fused`] but reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
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

    // Validates the bias and activation the same way gemmkit's checked entry does, with the same
    // panic wording. The bias length matches its user axis (PerRow == packed A.rows == C.rows,
    // PerCol == B.cols), and it does not overlap C. A LeakyRelu slope must be finite. This bias
    // stays in the user frame. The core `gemm_packed_a_fused` call below flips the axis internally,
    // to match the transposed product the packed-A path drives
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, packed.rows(), n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: the dims are validated above. faer guarantees that B and C are valid in-bounds
    // layouts. `c` is a `MatMut` exclusive borrow, so it cannot alias B. `packed` owns its
    // buffer independently of B/C, and the bias was validated disjoint from C above. The core
    // `_unchecked` tier raises the row-major-ish-C panic
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
