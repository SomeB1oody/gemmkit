//! Entries that reuse a pre-packed [`PackedLhs`]/[`PackedRhs`] operand instead of repacking it on
//! every call
use super::*;
use crate::common::dims_strides;
#[cfg(feature = "epilogue")]
use crate::common::lower_bias;

/// Pre-packs a RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack once
/// here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this `B`.
/// Reads `B`'s pointer/strides directly, so any layout works without copying
pub fn prepack_rhs<T, R, C, S>(b: &Matrix<T, R, C, S>) -> PackedRhs<T>
where
    T: GemmScalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let (k, n, rsb, csb) = dims_strides(b);
    // SAFETY: nalgebra guarantees B's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_rhs_unchecked(b.as_ptr(), rsb, csb, k, n) }
}

/// `C <- alpha*A*B + beta*C`, reusing a prepacked `B` from [`prepack_rhs`]. `C` must be
/// column-major-ish (`|col stride| >= |row stride|`): a row-major `C` would force the engine to
/// swap the `A`/`B` roles, which would invalidate the prepacked RHS, so gemmkit rejects it; use
/// [`gemm`] for a row-major `C`
///
/// # Panics
/// If the dimensions disagree, or if `C` is not column-major-ish
pub fn gemm_packed_b<T, R1, C1, S1, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_b_common(None, alpha, a, packed, beta, c, par);
}

/// As [`gemm_packed_b`], but also reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool: the fixed-cost inference loop (fixed weights packed once, a stream of activation batches)
///
/// # Panics
/// Same conditions as [`gemm_packed_b`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_with<T, R1, C1, S1, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_b_common(Some(ws), alpha, a, packed, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_common<T, R1, C1, S1, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (cm, cn) = c.shape();
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-nalgebra: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-nalgebra: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-nalgebra: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();
    // SAFETY: nalgebra guarantees A/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't
    // alias A, and the prepacked B lives in its own owned buffer, not in A's or C's storage
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

/// Pre-packs an LHS `A` into a reusable [`PackedLhs`] (a fixed `A` multiplied against a stream of
/// right-hand operands): pack once, then skip the per-call repack across many [`gemm_packed_a`]
/// calls. Reads `A`'s pointer/strides directly, so any layout works without copying
pub fn prepack_lhs<T, R, C, S>(a: &Matrix<T, R, C, S>) -> PackedLhs<T>
where
    T: GemmScalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    // SAFETY: nalgebra guarantees A's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_lhs_unchecked(a.as_ptr(), rsa, csa, m, k) }
}

/// `C <- alpha*A*B + beta*C`, reusing a prepacked `A` from [`prepack_lhs`]. `C` must be
/// row-major-ish (`|col stride| <= |row stride|`): a column-major `C` would force the engine to
/// keep `A` in the LHS role rather than swap it, which would invalidate the prepacked LHS, so
/// gemmkit rejects it; use [`gemm`] for a column-major `C`
///
/// # Panics
/// If the dimensions disagree, or if `C` is not row-major-ish
pub fn gemm_packed_a<T, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_a_common(None, alpha, packed, b, beta, c, par);
}

/// As [`gemm_packed_a`], but also reuses a caller-owned [`Workspace`] instead of the thread-local
/// pool
///
/// # Panics
/// Same conditions as [`gemm_packed_a`]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_with<T, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_a_common(Some(ws), alpha, packed, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_common<T, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.shape();
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-nalgebra: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-nalgebra: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-nalgebra: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();
    // SAFETY: nalgebra guarantees B/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't
    // alias B, and the prepacked A lives in its own owned buffer, not in B's or C's storage
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

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in 1 fused pass, reusing a prepacked `B` from
/// [`prepack_rhs`]: the nalgebra adapter over gemmkit's [`gemmkit::gemm_packed_b_fused`]. The
/// **same** [`PackedRhs`] serves both [`gemm_packed_b`] and this fused entry, since the epilogue
/// only changes how the result is stored, not how `B` was packed. `C` must be column-major-ish
/// (`|col stride| >= |row stride|`), for the same reason as [`gemm_packed_b`]. The optional
/// [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), and the
/// optional [`Activation`] is applied last; `bias == None && act == None` behaves exactly like
/// [`gemm_packed_b`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not column-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused<T, R1, C1, S1, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_b_fused_common(None, alpha, a, packed, beta, c, bias, act, par);
}

/// As [`gemm_packed_b_fused`], but also reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_packed_b_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_b_fused_with<T, R1, C1, S1, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_b_fused_common(Some(ws), alpha, a, packed, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_b_fused_common<T, R1, C1, S1, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    packed: &PackedRhs<T>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    let (cm, cn) = c.shape();
    assert_eq!(
        k,
        packed.rows(),
        "gemmkit-nalgebra: A.cols ({k}) != packed B.rows ({})",
        packed.rows()
    );
    assert_eq!(m, cm, "gemmkit-nalgebra: A.rows ({m}) != C.rows ({cm})");
    assert_eq!(
        packed.cols(),
        cn,
        "gemmkit-nalgebra: packed B.cols ({}) != C.cols ({cn})",
        packed.cols()
    );
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();

    // Checks the bias length against its axis (PerRow == A.rows, PerCol == packed B.cols ==
    // C.cols) and rejects an overlap with C, then a finite LeakyRelu slope, matching the core
    // checked entry's wording. The packed-B path never swaps A/B, so the bias stays in the user
    // frame and needs no axis flip before being passed down
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, packed.cols(), cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims checked above; nalgebra guarantees A/C layouts are valid in-bounds and `c` (a
    // `&mut` borrow) can't alias A; the prepacked B lives in its own owned buffer; the bias was
    // checked disjoint from C above. The core `_unchecked` tier itself raises the
    // column-major-ish-C panic
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

/// `C <- act(alpha*(prepacked A)*B + beta*C + bias)` in 1 fused pass, reusing a prepacked `A` from
/// [`prepack_lhs`]: the nalgebra adapter over gemmkit's [`gemmkit::gemm_packed_a_fused`]. The
/// **same** [`PackedLhs`] serves both [`gemm_packed_a`] and this fused entry. `C` must be
/// row-major-ish (`|col stride| <= |row stride|`), for the same reason as [`gemm_packed_a`]. The
/// optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`),
/// given in the user frame; the optional [`Activation`] is applied last. `bias == None && act ==
/// None` behaves exactly like [`gemm_packed_a`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not row-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused<T, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_a_fused_common(None, alpha, packed, b, beta, c, bias, act, par);
}

/// As [`gemm_packed_a_fused`], but also reuses a caller-owned [`Workspace`] instead of the
/// thread-local pool
///
/// # Panics
/// Same conditions as [`gemm_packed_a_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_packed_a_fused_with<T, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_packed_a_fused_common(Some(ws), alpha, packed, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_a_fused_common<T, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    packed: &PackedLhs<T>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    let (kb, n, rsb, csb) = dims_strides(b);
    let (cm, cn) = c.shape();
    assert_eq!(
        packed.cols(),
        kb,
        "gemmkit-nalgebra: packed A.cols ({}) != B.rows ({kb})",
        packed.cols()
    );
    assert_eq!(
        packed.rows(),
        cm,
        "gemmkit-nalgebra: packed A.rows ({}) != C.rows ({cm})",
        packed.rows()
    );
    assert_eq!(n, cn, "gemmkit-nalgebra: B.cols ({n}) != C.cols ({cn})");
    let cs = c.strides();
    let (rsc, csc) = (cs.0 as isize, cs.1 as isize);
    let cp = c.as_mut_ptr();

    // Checks the bias length against its USER axis (PerRow == packed A.rows == C.rows, PerCol ==
    // B.cols) and rejects an overlap with C, then a finite LeakyRelu slope, matching the core
    // checked entry's wording. This adapter passes the bias down in the user frame; the core
    // `gemm_packed_a_fused` is the one that flips its axis to match the transposed consume it drives
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, packed.rows(), n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims checked above; nalgebra guarantees B/C layouts are valid in-bounds and `c` (a
    // `&mut` borrow) can't alias B; the prepacked A lives in its own owned buffer; the bias was
    // checked disjoint from C above. The core `_unchecked` tier itself raises the row-major-ish-C
    // panic
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
