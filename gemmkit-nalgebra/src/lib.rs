//! # gemmkit-nalgebra
//!
//! A thin [`nalgebra`] adapter over the [`gemmkit`] GEMM engine. It accepts
//! `&Matrix<T, R, C, S>` for any storage `S: RawStorage` (so `DMatrix`, static
//! `SMatrix`, and every view type work), pulls the pointer and strides
//! straight out of the matrix, and forwards to gemmkit's raw engine.
//! Column-major (nalgebra's natural layout), row-major, and general-stride
//! views therefore all work without copying
//!
//! ```
//! use nalgebra::{DMatrix, Matrix2};
//! let a = Matrix2::new(1.0_f32, 2.0, 3.0, 4.0);
//! let b = Matrix2::new(5.0_f32, 6.0, 7.0, 8.0);
//! let c = gemmkit_nalgebra::dot(&a, &b);
//! assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! always, plus `f16`/`bf16` under the `half` feature. [`prepack_rhs`]/[`prepack_lhs`]
//! (with their [`gemm_packed_b`]/[`gemm_packed_a`] consumers) pre-pack one reused operand
//! for the fixed-weight loop. Complex (`Complex<f32>`/`Complex<f64>`, with optional
//! conjugation) needs the separate [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the
//! `complex` feature, since the conj flags don't fit the homogeneous surface. The integer
//! (`i8 -> i32`) path likewise gets its own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the
//! `int8` feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused-epilogue entries mirror gemmkit's own:
//! [`gemm_fused`]/[`gemm_fused_with`] (`C <- act(alpha*A*B + beta*C + bias)` in 1 pass, an
//! optional [`Bias`] plus an optional [`Activation`]). `f16`/`bf16` ride the same generic when
//! `half` is on. Requantized output needs `int8` + `epilogue`:
//! [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse the requantize
//! into a quantized `i8` (resp. `u8`) output. Complex-fused needs `complex` + `epilogue`: the
//! bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] (no activation: undefined on complex numbers)
//!
//! nalgebra has no 3-D array type, so the batched (`gemm_batched`, `gemm_batched_fused`) entries of
//! the ndarray adapter have no analogue here

/// The requantization parameters for the `int8` fused entries, re-exported so callers of
/// [`gemm_i8_requant`] need not depend on `gemmkit` directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::Requantize;
/// The fused-epilogue selectors, re-exported so callers of [`gemm_fused`] need not depend on
/// `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
#[cfg(feature = "epilogue")]
use gemmkit::{BiasDim, FusedScalar, gemm_fused_unchecked, gemm_fused_unchecked_with};
#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmScalar, Parallelism, Workspace, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_b_unchecked, gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with,
    prepack_lhs_unchecked, prepack_rhs_unchecked,
};
/// The prepacked-operand handles, re-exported so callers of [`prepack_rhs`] / [`prepack_lhs`] need
/// not depend on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::{
    gemm_i8_requant_u8_unchecked, gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with,
};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};
use nalgebra::{DMatrix, Dim, Dyn, Matrix, RawStorage, RawStorageMut, VecStorage};

/// Pull `(rows, cols, row-stride, col-stride)` out of a matrix of any storage. nalgebra reports
/// non-negative `usize` strides in element units; widen to the `isize` gemmkit's raw engine takes
#[inline]
fn dims_strides<T, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    a: &Matrix<T, R, C, S>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.shape();
    let (rs, cs) = a.strides();
    (r, c, rs as isize, cs as isize)
}

/// The half-open byte range `[lo, hi)` a strided C view based at `cp` (element `(0, 0)`) actually
/// touches, from the raw pointer plus `(dim, element-stride)` pairs. Strides are non-negative on
/// nalgebra, but the negative branch is kept for parity with gemmkit's own `extent`; an empty
/// (`dim == 0`) axis yields an empty range. **Raw pointer arithmetic only**: no reference is ever
/// formed over the span, which is why the fused entries forward raw parts to gemmkit's `_unchecked`
/// engine instead of fabricating a slice here
#[cfg(feature = "epilogue")]
#[inline]
fn c_byte_range<T>(cp: *const T, dims: &[(usize, isize)]) -> (usize, usize) {
    let sz = core::mem::size_of::<T>() as isize;
    if dims.iter().any(|&(d, _)| d == 0) {
        let b = cp as usize;
        return (b, b);
    }
    let (mut lo, mut hi): (isize, isize) = (0, 0);
    for &(d, s) in dims {
        if d <= 1 {
            continue; // a length-1 axis spans nothing, so its stride (any sign) is irrelevant
        }
        let e = (d as isize - 1) * s;
        if e < 0 {
            lo += e;
        } else {
            hi += e;
        }
    }
    let base = cp as isize;
    ((base + lo * sz) as usize, (base + (hi + 1) * sz) as usize)
}

/// `true` if the `bias` slice (`len` elements of `TB`) overlaps the byte range the strided C view
/// touches: the raw-pointer replication of gemmkit's own byte-range overlap test (`a0 < b1 && b0 <
/// a1`), so the adapter reproduces the core checked entry's bias-vs-`C` rejection without ever
/// fabricating a `C` slice
#[cfg(feature = "epilogue")]
#[inline]
fn bias_overlaps_c<TC, TB>(
    cp: *const TC,
    c_dims: &[(usize, isize)],
    bias: *const TB,
    len: usize,
) -> bool {
    let (c_lo, c_hi) = c_byte_range(cp, c_dims);
    if c_lo == c_hi || len == 0 {
        return false;
    }
    let b_lo = bias as usize;
    let b_hi = b_lo + len * core::mem::size_of::<TB>();
    c_lo < b_hi && b_lo < c_hi
}

/// Validate a fused `Option<Bias>` against the output shape and `C`'s footprint (replicating the
/// core checked entry's `validate_bias`; panic wording is byte-identical), and lower it to the raw
/// `(ptr, BiasDim, has_bias)` triple the `_unchecked` core entries take. `cp`/`c_dims` describe `C`
/// for the overlap test via [`bias_overlaps_c`] (raw pointer math; `C` is never referenced)
#[cfg(feature = "epilogue")]
fn lower_bias<T>(
    bias: Option<Bias<'_, T>>,
    m: usize,
    n: usize,
    cp: *const T,
    c_dims: &[(usize, isize)],
) -> (*const T, BiasDim, bool) {
    match bias {
        None => (core::ptr::null(), BiasDim::PerRow, false),
        Some(Bias::PerRow(s)) => {
            assert_eq!(
                s.len(),
                m,
                "gemmkit: PerRow bias length ({}) != A.rows ({})",
                s.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, s.as_ptr(), s.len()) {
                panic!("gemmkit: bias slice overlaps C");
            }
            (s.as_ptr(), BiasDim::PerRow, true)
        }
        Some(Bias::PerCol(s)) => {
            assert_eq!(
                s.len(),
                n,
                "gemmkit: PerCol bias length ({}) != B.cols ({})",
                s.len(),
                n
            );
            if bias_overlaps_c(cp, c_dims, s.as_ptr(), s.len()) {
                panic!("gemmkit: bias slice overlaps C");
            }
            (s.as_ptr(), BiasDim::PerCol, true)
        }
    }
}

/// Validate a requantize per-row bias against `A.rows` and `C`'s footprint (replicating the core
/// `requant_bias`; panic wording is byte-identical), and lower it to the raw `(ptr, has_bias)` the
/// `_unchecked` requant entries take. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap
/// test; raw pointer math only
#[cfg(all(feature = "int8", feature = "epilogue"))]
fn requant_bias<TC>(
    m: usize,
    cp: *const TC,
    c_dims: &[(usize, isize)],
    bias: Option<&[i32]>,
) -> (*const i32, bool) {
    match bias {
        Some(bias) => {
            assert_eq!(
                bias.len(),
                m,
                "gemmkit: requantize bias length ({}) != A.rows ({})",
                bias.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, bias.as_ptr(), bias.len()) {
                panic!("gemmkit: requantize bias overlaps C");
            }
            (bias.as_ptr(), true)
        }
        None => (core::ptr::null(), false),
    }
}

/// Allocate an `m x n` column-major [`DMatrix`] whose cells are all `zero`. Used only by the
/// `dot`-family convenience wrappers: they call gemm with `beta == 0`, so gemmkit overwrites every
/// element and the fill is never read: it exists solely to hand the engine an initialized buffer.
/// Passing the zero value in (rather than going through `DMatrix::zeros`) keeps the bound at
/// `T: Copy`, so the engine's element types (`f16`/`bf16`, `i32`) need not satisfy nalgebra's own
/// `Scalar`/`Zero`
#[inline]
fn filled_dmatrix<T: Copy>(m: usize, n: usize, zero: T) -> DMatrix<T> {
    DMatrix::from_data(VecStorage::new(Dyn(m), Dyn(n), vec![zero; m * n]))
}

/// `C <- alpha*A*B + beta*C`
///
/// # Panics
/// If the inner dimensions disagree
pub fn gemm<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
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
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// If the inner dimensions disagree
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
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
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
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

    // SAFETY: dims validated; nalgebra guarantees the storage's pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`
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

/// `A*B` into a fresh column-major [`DMatrix`]: the `.dot()`-style convenience
pub fn dot<T, R1, C1, S1, R2, C2, S2>(
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
) -> DMatrix<T>
where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta == 0, so the initial fill is never read
    let mut c = filled_dmatrix(m, n, T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 fused pass: the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`) and the optional [`Activation`] is applied last;
/// `bias == None && act == None` is exactly [`gemm`]. `T` is `f32`/`f64` (plus `f16`/`bf16` under
/// `half`, whose epilogue applies in `f32` before the single narrowing). Like [`gemm`], it reads the
/// pointer/strides directly and forwards to gemmkit's raw engine, so column-major, row-major, and
/// general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias/activation the adapter rejects (a `PerRow`/`PerCol`
/// bias of the wrong length, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
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
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
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
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_fused_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
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

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its axis and does not overlap C (raw pointer math, C is never
    // referenced), and a LeakyRelu slope is finite
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; nalgebra guarantees the pointer/strides describe a valid in-bounds
    // layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C
    // above
    unsafe {
        match ws {
            Some(ws) => gemm_fused_unchecked_with(
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
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
            None => gemm_fused_unchecked(
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
                bias_ptr,
                bias_dim,
                has_bias,
                act,
                par,
            ),
        }
    }
}

/// Pre-pack a RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack once
/// here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this `B`.
/// Reads B's pointer/strides directly, so any layout works without copying
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

/// `C <- alpha*A*B + beta*C` reusing a prepacked `B` ([`prepack_rhs`]). `C` must be
/// column-major-ish (`|col stride| >= |row stride|`): a row-major `C` would swap A/B and invalidate
/// the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout
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

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`]: the fixed-cost inference loop
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
    // SAFETY: nalgebra guarantees A/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
    // A, and the prepacked B is a separate owned buffer
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

/// Pre-pack an LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of right
/// operands): pack once, then skip the per-call repack across many [`gemm_packed_a`] calls. Reads
/// A's pointer/strides directly, so any layout works without copying
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

/// `C <- alpha*A*B + beta*C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be row-major-ish
/// (`|col stride| <= |row stride|`): a column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout
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

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`]
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
    // SAFETY: nalgebra guarantees B/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
    // B, and the prepacked A is a separate owned buffer
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

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`, the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ. Reads pointers/strides directly,
/// so transposed / general-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "int8")]
pub fn gemm_i8<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
{
    gemm_i8_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm_i8`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized-inference
/// loop
///
/// # Panics
/// Same conditions as [`gemm_i8`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
{
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    beta: i32,
    c: &mut Matrix<i32, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i32, RC, CC>,
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
    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut i32` borrow)
    // can't alias `a`/`b` (`&i8`): different element types over distinct storage
    unsafe {
        match ws {
            Some(ws) => gemm_i8_unchecked_with(
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
            None => gemm_i8_unchecked(
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

/// `A(i8)*B(i8)` into a fresh column-major `DMatrix<i32>`: the i8 analogue of [`dot`]
#[cfg(feature = "int8")]
pub fn dot_i8<R1, C1, S1, R2, C2, S2>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
) -> DMatrix<i32>
where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta == 0, so the initial fill is never read
    let mut c = filled_dmatrix(m, n, 0i32);
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then requantized to
/// an `i8` output in 1 pass: the nalgebra adapter over gemmkit's [`gemmkit::gemm_i8_requant`]. The
/// [`Requantize`] carries the per-tensor `scale`, `zero_point`, and an optional per-row `i32` bias;
/// there is no `alpha` (folds into `scale`) or `beta`. Reads the pointers/strides directly and
/// forwards to gemmkit's raw engine, so transposed and general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale`, a `zero_point` outside `[-128, 127]`, or a bias whose length is not
/// `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
{
    gemm_i8_requant_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized
/// inference loop
///
/// # Panics
/// Same conditions as [`gemm_i8_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
{
    gemm_i8_requant_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<i8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<i8, RC, CC>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): finite,
    // positive scale; zero_point in the i8 band; a per-row bias of length A.rows disjoint from C
    // (raw pointer math, C is never referenced)
    assert!(
        req.scale.is_finite() && req.scale > 0.0,
        "gemmkit: requantize scale ({}) must be finite and > 0",
        req.scale
    );
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut i8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_unchecked_with(
                ws,
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                req.scale,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_i8_requant_unchecked(
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                req.scale,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Requantizing integer GEMM with an **unsigned `u8` output** (ONNX-QLinearMatMul-style
/// activation): the nalgebra adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`]. The `u8`-output
/// twin of [`gemm_i8_requant`], differing only in the output domain `[0, 255]` and the `zero_point`
/// range
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale`, a `zero_point` outside `[0, 255]`, or a bias whose length is not
/// `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
{
    gemm_i8_requant_u8_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8_with<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
{
    gemm_i8_requant_u8_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_u8_common<R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    a: &Matrix<i8, R1, C1, S1>,
    b: &Matrix<i8, R2, C2, S2>,
    req: Requantize<'_>,
    c: &mut Matrix<u8, RC, CC, SC>,
    par: Parallelism,
) where
    R1: Dim,
    C1: Dim,
    S1: RawStorage<i8, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<i8, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<u8, RC, CC>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): finite,
    // positive scale; zero_point in the u8 band; a per-row bias of length A.rows disjoint from C
    // (raw pointer math, C is never referenced)
    assert!(
        req.scale.is_finite() && req.scale > 0.0,
        "gemmkit: requantize scale ({}) must be finite and > 0",
        req.scale
    );
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; nalgebra guarantees valid in-bounds layouts; `c` (a `&mut u8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above
    unsafe {
        match ws {
            Some(ws) => gemm_i8_requant_u8_unchecked_with(
                ws,
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                req.scale,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_i8_requant_u8_unchecked(
                m,
                k,
                n,
                a.as_ptr(),
                rsa,
                csa,
                b.as_ptr(),
                rsb,
                csb,
                req.scale,
                req.zero_point,
                bias_ptr,
                has_bias,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Complex `C <- alpha*op(A)*op(B) + beta*C`, with `op(A) = conj(A)` when `conj_a` (resp.
/// `conj(B)` when `conj_b`). `T` is `Complex<f32>`/`Complex<f64>`; needs the `complex`
/// feature. Like [`gemm`], it reads pointer/strides directly, so transposed, row-major,
/// and general-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    gemm_cplx_common(None, alpha, a, conj_a, b, conj_b, beta, c, par);
}

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    gemm_cplx_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, par);
}

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: ComplexScalar,
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

    // SAFETY: dims validated; nalgebra guarantees the storage's pointer/strides describe a
    // valid in-bounds layout, and `c` (a `&mut` borrow) cannot alias `a`/`b`
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
            None => gemm_cplx_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                par,
            ),
        }
    }
}

/// Non-conjugated complex `A*B` into a fresh column-major [`DMatrix`]: the complex
/// analogue of [`dot`]. For conjugated products use [`gemm_cplx`]. Needs the
/// `complex` feature
#[cfg(feature = "complex")]
pub fn dot_cplx<T, R1, C1, S1, R2, C2, S2>(
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
) -> DMatrix<T>
where
    T: ComplexScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
{
    let (m, _) = a.shape();
    let (_, n) = b.shape();
    // beta == 0, so the initial fill is never read
    let mut c = filled_dmatrix(m, n, T::ZERO);
    gemm_cplx(
        T::ONE,
        a,
        false,
        b,
        false,
        T::ZERO,
        &mut c,
        Parallelism::default(),
    );
    c
}

/// Complex `C <- alpha*op(A)*op(B) + beta*C + bias` in 1 fused pass, with `op(A) = conj(A)` when
/// `conj_a` (resp. `conj(B)` when `conj_b`): the nalgebra adapter over gemmkit's
/// [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added verbatim (never conjugated); `bias == None` is exactly
/// [`gemm_cplx`]. There is **no** activation parameter: an ordering activation is undefined on
/// complex numbers. Like [`gemm_cplx`], it reads the pointer/strides directly and forwards to
/// gemmkit's raw engine, so transposed and general-stride views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects (a `PerRow`/`PerCol` bias of
/// the wrong length, or a bias slice overlapping `C`)
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    gemm_cplx_fused_common(None, alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

/// Like [`gemm_cplx_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    gemm_cplx_fused_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_fused_common<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    conj_a: bool,
    b: &Matrix<T, R2, C2, S2>,
    conj_b: bool,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    // Fused-bias validation, replicating gemmkit's checked entry (byte-identical wording): the bias
    // length matches its axis and does not overlap C (raw pointer math, C is never referenced)
    // Complex has no activation (undefined on complex numbers), so there is no slope check
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: dims validated; nalgebra guarantees the pointer/strides describe a valid in-bounds
    // layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C
    // above
    unsafe {
        match ws {
            Some(ws) => gemm_cplx_fused_unchecked_with(
                ws,
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                par,
            ),
            None => gemm_cplx_fused_unchecked(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                rsa,
                csa,
                conj_a,
                b.as_ptr(),
                rsb,
                csb,
                conj_b,
                beta,
                cp,
                rsc,
                csc,
                bias_ptr,
                bias_dim,
                has_bias,
                par,
            ),
        }
    }
}
