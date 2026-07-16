//! # gemmkit-ndarray
//!
//! A thin [`ndarray`] adapter over the [`gemmkit`] GEMM engine. It accepts
//! `&ArrayBase<S, Ix2>` for any storage `S: Data` (so both `ArrayView2` and
//! `&Array2` work), pulls the pointer and strides straight out of the array, and
//! forwards to gemmkit's raw engine, so C-order, F-order, general-stride, and
//! reversed (negative-stride) views all work without copying
//!
//! ```
//! use ndarray::array;
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//! let c = gemmkit_ndarray::dot(&a, &b);
//! assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
//! ```
//!
//! [`gemm`]/[`gemm_with`]/[`dot`] are generic over [`gemmkit::GemmScalar`]: `f32`/`f64`
//! always, plus `f16`/`bf16` under the `half` feature. [`gemm_batched`]/[`gemm_batched_with`]/
//! [`dot_batched`] extend the same idea to a stack of matrices (a 3-D array with the batch on
//! axis 0) and [`prepack_rhs`]/[`prepack_lhs`] (with their [`gemm_packed_b`]/[`gemm_packed_a`]
//! consumers) pre-pack one reused operand for the fixed-weight loop. Complex
//! (`Complex<f32>`/`Complex<f64>`, with optional conjugation) needs the separate
//! [`gemm_cplx`]/[`gemm_cplx_with`]/[`dot_cplx`] under the `complex` feature, since the
//! conj flags don't fit the homogeneous surface. The integer (`i8 -> i32`) path likewise gets its
//! own [`gemm_i8`]/[`gemm_i8_with`]/[`dot_i8`] under the `int8` feature (`i8` inputs, `i32` output)
//!
//! Under the `epilogue` feature the fused-epilogue entries mirror gemmkit's own:
//! [`gemm_fused`]/[`gemm_fused_with`] (`C <- act(alpha*A*B + beta*C + bias)` in 1 pass, an
//! optional [`Bias`] plus an optional [`Activation`]), their batched twins
//! [`gemm_batched_fused`]/[`gemm_batched_fused_with`] (one shared bias/activation over every element
//! of the stack), and the prepacked-operand twins
//! [`gemm_packed_b_fused`]/[`gemm_packed_b_fused_with`] and
//! [`gemm_packed_a_fused`]/[`gemm_packed_a_fused_with`] (the same reused [`PackedRhs`]/[`PackedLhs`]
//! handle plus a fused bias/activation). `f16`/`bf16` ride the same generic when `half` is on.
//! Requantized output needs
//! `int8` + `epilogue`: [`gemm_i8_requant`]/[`gemm_i8_requant_with`] (and the `u8`-output
//! [`gemm_i8_requant_u8`]/[`gemm_i8_requant_u8_with`]) take a [`Requantize`] and fuse the requantize
//! into a quantized `i8` (resp. `u8`) output. Complex-fused needs `complex` + `epilogue`: the
//! bias-only [`gemm_cplx_fused`]/[`gemm_cplx_fused_with`] (no activation: undefined on complex numbers)

#![cfg_attr(docsrs, feature(doc_cfg))]

/// The fused-epilogue selectors, re-exported so callers of [`gemm_fused`] need not depend on
/// `gemmkit` directly
#[cfg(feature = "epilogue")]
pub use gemmkit::{Activation, Bias};
#[cfg(feature = "epilogue")]
use gemmkit::{
    BiasDim, FusedScalar, MapScalar, gemm_batched_fused_unchecked,
    gemm_batched_fused_unchecked_with, gemm_fused_unchecked, gemm_fused_unchecked_with,
    gemm_map_unchecked, gemm_map_unchecked_with, gemm_packed_a_fused_unchecked,
    gemm_packed_a_fused_unchecked_with, gemm_packed_b_fused_unchecked,
    gemm_packed_b_fused_unchecked_with,
};
#[cfg(feature = "complex")]
use gemmkit::{ComplexScalar, gemm_cplx_unchecked, gemm_cplx_unchecked_with};
use gemmkit::{
    GemmScalar, Parallelism, Workspace, gemm_batched_unchecked, gemm_batched_unchecked_with,
    gemm_packed_a_unchecked, gemm_packed_a_unchecked_with, gemm_packed_b_unchecked,
    gemm_packed_b_unchecked_with, gemm_unchecked, gemm_unchecked_with, prepack_lhs_unchecked,
    prepack_rhs_unchecked,
};
/// The prepacked-operand handles, re-exported so callers of [`prepack_rhs`] / [`prepack_lhs`] need
/// not depend on `gemmkit` directly
pub use gemmkit::{PackedLhs, PackedRhs};
/// The requantization parameters ([`Requantize`]) and its per-tensor / per-row output scale
/// ([`RequantScale`]) for the `int8` fused entries, re-exported so callers of [`gemm_i8_requant`]
/// need not depend on `gemmkit` directly
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use gemmkit::{RequantScale, Requantize};
#[cfg(all(feature = "complex", feature = "epilogue"))]
use gemmkit::{gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with};
#[cfg(all(feature = "int8", feature = "epilogue"))]
use gemmkit::{
    gemm_i8_requant_u8_unchecked, gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with,
};
#[cfg(feature = "int8")]
use gemmkit::{gemm_i8_unchecked, gemm_i8_unchecked_with};
use ndarray::{Array2, Array3, ArrayBase, Data, DataMut, Ix2, Ix3};

#[inline]
fn dims_strides<T, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}

/// The half-open byte range `[lo, hi)` a strided C view based at `cp` (element `(0, ..., 0)`) actually
/// touches, from the raw pointer plus `(dim, element-stride)` pairs. Strides may be negative
/// (reversed views), so a negative axis extends `lo` below the base and a positive one extends `hi`
/// above it; an empty (`dim == 0`) axis yields an empty range. **Raw pointer arithmetic only**: no
/// reference is ever formed over the (possibly gappy / partly-uninitialized) span, which is exactly
/// why the fused entries forward raw parts to gemmkit's `_unchecked` engine instead of fabricating a
/// slice here
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

/// Validate a fused `Option<Bias>` against the output shape and `C`'s footprint, replicating the
/// core checked entry's `validate_bias` (byte-identical panic wording), and lower it to the raw
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

/// Validate a requantize per-row bias against `A.rows` and `C`'s footprint, replicating the core
/// `requant_bias` (byte-identical panic wording), and lower it to the raw `(ptr, has_bias)` the
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

/// Validate a requantize [`RequantScale`] against `A.rows` and `C`'s footprint, replicating the
/// core `requant_scale` (byte-identical panic wording), and lower it to the raw `(scale,
/// row_scales, has_row_scales)` the `_unchecked` requant entries take. A `PerTensor(s)` must be
/// finite and `> 0`; a `PerRow` slice must have length `m`, every element finite and `> 0`, and
/// must not overlap `C`. `cp`/`c_dims` describe the (`i8`/`u8`) `C` for the overlap test; raw
/// pointer math only
#[cfg(all(feature = "int8", feature = "epilogue"))]
fn requant_scale<TC>(
    m: usize,
    cp: *const TC,
    c_dims: &[(usize, isize)],
    scale: RequantScale<'_>,
) -> (f32, *const f32, bool) {
    match scale {
        RequantScale::PerTensor(s) => {
            assert!(
                s.is_finite() && s > 0.0,
                "gemmkit: requantize scale ({s}) must be finite and > 0"
            );
            (s, core::ptr::null(), false)
        }
        RequantScale::PerRow(scales) => {
            assert_eq!(
                scales.len(),
                m,
                "gemmkit: requantize scales length ({}) != A.rows ({})",
                scales.len(),
                m
            );
            if bias_overlaps_c(cp, c_dims, scales.as_ptr(), scales.len()) {
                panic!("gemmkit: requantize scales overlap C");
            }
            for &s in scales {
                assert!(
                    s.is_finite() && s > 0.0,
                    "gemmkit: requantize scale ({s}) must be finite and > 0"
                );
            }
            (0.0, scales.as_ptr(), true)
        }
    }
}

/// `C <- alpha*A*B + beta*C`
///
/// # Panics
/// If the inner dimensions disagree
pub fn gemm<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// If the inner dimensions disagree
#[allow(clippy::too_many_arguments)]
pub fn gemm_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_common(Some(ws), alpha, a, b, beta, c, par);
}

#[allow(clippy::too_many_arguments)]
fn gemm_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: GemmScalar,
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

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
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

/// `A*B` into a fresh row-major [`Array2`]: the `.dot()`-style convenience
pub fn dot<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read
    let mut c = Array2::from_elem((m, n), T::ZERO);
    gemm(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

/// `C <- act(alpha*A*B + beta*C + bias)` in 1 fused pass: the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`) and the optional [`Activation`] is applied last;
/// `bias == None && act == None` is exactly [`gemm`]. `T` is `f32`/`f64` (plus `f16`/`bf16` under
/// `half`, whose epilogue applies in `f32` before the single narrowing). Like [`gemm`], it reads the
/// pointer/strides directly and forwards to gemmkit's raw engine, so C-order, F-order,
/// general-stride, transposed, and reversed (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias/activation the adapter rejects (a `PerRow`/`PerCol`
/// bias of the wrong length, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_fused_common(None, alpha, a, b, beta, c, bias, act, par);
}

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_fused`]
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_fused_common(Some(ws), alpha, a, b, beta, c, bias, act, par);
}

#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments)]
fn gemm_fused_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) where
    T: FusedScalar,
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

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its axis and does not overlap C (raw pointer math, C is never
    // referenced), and a LeakyRelu slope is finite
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a valid in-bounds
    // layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C
    // above. Negative (reversed) strides forward straight through, exactly as the plain entry
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

/// `C[r, c] <- f(alpha*A*B + beta*C, r, c)` in 1 fused pass: the ndarray adapter over
/// gemmkit's [`gemmkit::gemm_map`]. The closure `f(value, row, col)` is applied to each output
/// element at its final value, with `(row, col)` in the **user** frame of `C`, fired exactly once
/// per element. `T` is `f32`/`f64` only. Like [`gemm`], it reads the pointer/strides directly and
/// forwards to gemmkit's raw engine, so C-order, F-order, general-stride, transposed, and reversed
/// (negative-stride) views all work without copying
///
/// For a bias / activation prefer [`gemm_fused`] (it vectorizes); `gemm_map` is the general
/// per-element extension point (GELU, sigmoid, clamps, position-dependent transforms), at the cost
/// of 1 indirect call per output element. For `f32`/`f64` the result is bit-identical to [`gemm`]
/// followed by mapping each `C[r, c]` through `f(C[r, c], r, c)`, for every shape
///
/// # Panics
/// If the inner dimensions disagree (same conditions as [`gemm`])
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

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a valid in-bounds
    // layout and `c` (a `&mut` borrow) can't alias `a`/`b`. Negative (reversed) strides forward
    // straight through, exactly as the plain entry. The closure is total (applied to every element)
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

#[inline]
fn dims_strides3<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix3>,
) -> (usize, usize, usize, isize, isize, isize) {
    let (b, r, c) = a.dim();
    let s = a.strides();
    (b, r, c, s[0], s[1], s[2])
}

/// Strided-batched `C_e <- alpha*A_e*B_e + beta*C_e`, batch on **axis 0**: `a` is `(batch, m, k)`,
/// `b` is `(batch, k, n)`, `c` is `(batch, m, n)`. The axis-0 stride is each operand's batch stride
/// and axes 1/2 the element strides, read directly, so C-order, F-order, and general-stride 3-D
/// views all work without copying. Parallelizes across the batch (each element serial on one
/// worker), so the result reproduces a loop of [`gemm`] calls
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

/// Like [`gemm_batched`] but reuses a caller-owned [`Workspace`]: zero heap allocation after the
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

    // SAFETY: ndarray guarantees each element's pointer/strides describe a valid in-bounds layout;
    // `c` (a `&mut` borrow) can't alias `a`/`b`, and its batch elements (distinct axis-0 slices of
    // a real array) are pairwise disjoint
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

/// `A_e * B_e` for each batch element into a fresh row-major `(batch, m, n)` [`Array3`]: the
/// batched analogue of [`dot`]
pub fn dot_batched<T, S1, S2>(a: &ArrayBase<S1, Ix3>, b: &ArrayBase<S2, Ix3>) -> Array3<T>
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (batch, m, _) = a.dim();
    let (_, _, n) = b.dim();
    // beta == 0, so the initial fill is never read
    let mut c = Array3::from_elem((batch, m, n), T::ZERO);
    gemm_batched(T::ONE, a, b, T::ZERO, &mut c, Parallelism::default());
    c
}

/// Strided-batched `C_e <- act(alpha*A_e*B_e + beta*C_e + bias)`, batch on **axis 0** with **one
/// shared** [`Bias`]/[`Activation`] applied to every element (the batched-linear-layer case): the
/// ndarray adapter over gemmkit's [`gemmkit::gemm_batched_fused`]. Shapes match [`gemm_batched`]
/// (`a` is `(batch, m, k)`, `b` `(batch, k, n)`, `c` `(batch, m, n)`); the bias is sized for a
/// **single** element (`PerRow` length `m`, `PerCol` length `n`), not the whole batch. Each element
/// reproduces a [`gemm_fused`] call, so `bias == None && act == None` is exactly [`gemm_batched`].
/// Reads the pointers/strides directly and forwards to gemmkit's raw engine, so general-stride and
/// reversed (negative-stride) 3-D views all work without copying
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

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // one shared bias is sized for a single element (length m/n, not batch*axis) and must not
    // overlap C's whole-stack footprint (raw pointer math, C is never referenced); a LeakyRelu
    // slope is finite
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, n, cp, &[(cb, cs0), (cm, cs1), (cn, cs2)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; ndarray guarantees each element's layout is valid in-bounds and `c`
    // (a `&mut` borrow) can't alias `a`/`b`, its batch elements being pairwise-disjoint axis-0
    // slices; the shared bias was validated disjoint from C above. Reversed strides forward through
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

/// Pre-pack a 2-D RHS `B` into a reusable [`PackedRhs`] (gemmkit's fixed-weight reuse path): pack
/// once here, then skip the per-call repack across many [`gemm_packed_b`] calls that share this
/// `B`. Reads B's pointer/strides directly, so any layout works without copying
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
/// column-major-ish (`|col stride| >= |row stride|`): a row-major `C` would swap A/B and invalidate
/// the prepacked RHS, which gemmkit rejects; use [`gemm`] for that layout
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

/// Like [`gemm_packed_b`] but reuses a caller-owned [`Workspace`]: the fixed-cost inference loop
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
    // SAFETY: ndarray guarantees A/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
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

/// Pre-pack a 2-D LHS `A` into a reusable [`PackedLhs`] (a fixed `A` against a stream of right
/// operands): pack once, then skip the per-call repack across many [`gemm_packed_a`] calls. Reads
/// A's pointer/strides directly, so any layout works without copying
pub fn prepack_lhs<T, S>(a: &ArrayBase<S, Ix2>) -> PackedLhs<T>
where
    T: GemmScalar,
    S: Data<Elem = T>,
{
    let (m, k, rsa, csa) = dims_strides(a);
    // SAFETY: ndarray guarantees A's pointer/strides describe a valid in-bounds layout
    unsafe { prepack_lhs_unchecked(a.as_ptr(), rsa, csa, m, k) }
}

/// `C <- alpha*A*B + beta*C` reusing a prepacked `A` ([`prepack_lhs`]). `C` must be row-major-ish
/// (`|col stride| <= |row stride|`): a column-major `C` would keep A in the LHS role and
/// invalidate the prepacked LHS, which gemmkit rejects; use [`gemm`] for that layout
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

/// Like [`gemm_packed_a`] but reuses a caller-owned [`Workspace`]
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
    // SAFETY: ndarray guarantees B/C layouts are valid in-bounds; `c` (a `&mut` borrow) can't alias
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

/// `C <- act(alpha*A*(prepacked B) + beta*C + bias)` in 1 fused pass, reusing a prepacked `B`
/// ([`prepack_rhs`]): the ndarray adapter over gemmkit's [`gemmkit::gemm_packed_b_fused`]. The
/// **same** [`PackedRhs`] serves both [`gemm_packed_b`] and this fused entry (the epilogue is
/// store-side only). `C` must be column-major-ish (`|col stride| >= |row stride|`); a row-major `C`
/// would swap A/B and invalidate the prepacked RHS, which gemmkit rejects. The optional [`Bias`] is
/// [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`) and the optional
/// [`Activation`] is applied last; `bias == None && act == None` matches [`gemm_packed_b`]
///
/// # Panics
/// If the dimensions disagree, if `C` is not column-major-ish, or on a bias/activation the adapter
/// rejects (wrong-length bias, a bias slice overlapping `C`, or a non-finite `LeakyRelu` slope)
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

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its axis (PerRow == A.rows, PerCol == packed B.cols == C.cols) and does
    // not overlap C (raw pointer math), and a LeakyRelu slope is finite. The packed path never
    // swaps, so the user-frame bias forwards unflipped
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, m, packed.cols(), cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; ndarray guarantees A/C layouts are valid in-bounds and `c` (a `&mut`
    // borrow) can't alias A; the prepacked B is a separate owned buffer; the bias was validated
    // disjoint from C. The core `_unchecked` tier raises the column-major-ish-C panic
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
/// row-major-ish (`|col stride| <= |row stride|`); a column-major `C` would keep A in the LHS role
/// and invalidate the prepacked LHS, which gemmkit rejects. The optional [`Bias`] is
/// [`Bias::PerRow`] (length `A.rows`) or [`Bias::PerCol`] (length `B.cols`), specified in the user
/// frame; the optional [`Activation`] is applied last; `bias == None && act == None` matches
/// [`gemm_packed_a`]
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

/// Like [`gemm_packed_a_fused`] but reuses a caller-owned [`Workspace`]
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

    // Fused-epilogue validation, replicating gemmkit's checked entry (byte-identical wording): the
    // bias length matches its USER axis (PerRow == packed A.rows == C.rows, PerCol == B.cols) and
    // does not overlap C, and a LeakyRelu slope is finite. The bias stays in the user frame; the
    // core `gemm_packed_a_fused` flips the axis to match the transposed consume it drives
    let (bias_ptr, bias_dim, has_bias) =
        lower_bias(bias, packed.rows(), n, cp, &[(cm, rsc), (cn, csc)]);
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(T::finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // SAFETY: dims validated; ndarray guarantees B/C layouts are valid in-bounds and `c` (a `&mut`
    // borrow) can't alias B; the prepacked A is a separate owned buffer; the bias was validated
    // disjoint from C. The core `_unchecked` tier raises the row-major-ish-C panic
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

/// Integer `C(i32) <- alpha*A(i8)*B(i8) + beta*C`, the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_i8`]. `i8` inputs accumulate into an `i32` output (`alpha`/`beta`/`C` are `i32`);
/// arithmetic wraps on overflow, the conventional integer-GEMM semantics. A separate entry from
/// [`gemm`] because input (`i8`) and output (`i32`) types differ. Reads pointers/strides directly,
/// so transposed / F-order / general-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "int8")]
pub fn gemm_i8<S1, S2, SC>(
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
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
pub fn gemm_i8_with<S1, S2, SC>(
    ws: &mut Workspace,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
{
    gemm_i8_common(Some(ws), alpha, a, b, beta, c, par);
}

#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
fn gemm_i8_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: i32,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: i32,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i32>,
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
    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut i32` borrow)
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

/// `A(i8)*B(i8)` into a fresh row-major `Array2<i32>`: the i8 analogue of [`dot`]
#[cfg(feature = "int8")]
pub fn dot_i8<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<i32>
where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read
    let mut c = Array2::<i32>::zeros((m, n));
    gemm_i8(1, a, b, 0, &mut c, Parallelism::default());
    c
}

/// Requantizing integer GEMM: `i8` inputs multiplied into an `i32` accumulator, then requantized to
/// an `i8` output in 1 pass: the ndarray adapter over gemmkit's [`gemmkit::gemm_i8_requant`]. The
/// [`Requantize`] carries the per-tensor or per-row `scale`, `zero_point`, and an optional per-row
/// `i32` bias; there is no `alpha` (folds into `scale`) or `beta`. Reads the pointers/strides
/// directly and forwards to gemmkit's raw engine, so transposed, general-stride, and reversed
/// (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[-128, 127]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant<S1, S2, SC>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
{
    gemm_i8_requant_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant`] but reuses a caller-owned [`Workspace`]: the fixed-cost quantized
/// inference loop
///
/// # Panics
/// Same conditions as [`gemm_i8_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_with<S1, S2, SC>(
    ws: &mut Workspace,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
{
    gemm_i8_requant_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = i8>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): a
    // finite, positive per-tensor or per-row scale (per-row length A.rows disjoint from C);
    // zero_point in the i8 band; a per-row bias of length A.rows disjoint from C (raw pointer math,
    // C is never referenced)
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (-128..=127).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of i8 range [-128, 127]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut i8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above. Reversed strides
    // forward straight through, exactly as the plain entry
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
                scale,
                row_scales,
                has_row_scales,
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
                scale,
                row_scales,
                has_row_scales,
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
/// activation): the ndarray adapter over gemmkit's [`gemmkit::gemm_i8_requant_u8`]. The `i8`-output
/// twin of [`gemm_i8_requant`], differing only in the output domain `[0, 255]` and the `zero_point`
/// range
///
/// # Panics
/// If the inner dimensions disagree, or on the requant parameters the adapter rejects (a non-finite
/// or non-positive `scale` (per-tensor or any per-row element), a per-row scale slice whose length
/// is not `A.rows` or which overlaps `C`, a `zero_point` outside `[0, 255]`, or a bias whose
/// length is not `A.rows` or which overlaps `C`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8<S1, S2, SC>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
{
    gemm_i8_requant_u8_common(None, a, b, req, c, par);
}

/// Like [`gemm_i8_requant_u8`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_i8_requant_u8`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub fn gemm_i8_requant_u8_with<S1, S2, SC>(
    ws: &mut Workspace,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
{
    gemm_i8_requant_u8_common(Some(ws), a, b, req, c, par);
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
fn gemm_i8_requant_u8_common<S1, S2, SC>(
    ws: Option<&mut Workspace>,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    req: Requantize<'_>,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    S1: Data<Elem = i8>,
    S2: Data<Elem = i8>,
    SC: DataMut<Elem = u8>,
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
    // Requantize validation, replicating gemmkit's checked entry (byte-identical wording): a
    // finite, positive per-tensor or per-row scale (per-row length A.rows disjoint from C);
    // zero_point in the u8 band; a per-row bias of length A.rows disjoint from C (raw pointer math,
    // C is never referenced)
    let (scale, row_scales, has_row_scales) =
        requant_scale(m, cp, &[(cm, rsc), (cn, csc)], req.scale);
    assert!(
        (0..=255).contains(&req.zero_point),
        "gemmkit: requantize zero_point ({}) out of u8 range [0, 255]",
        req.zero_point
    );
    let (bias_ptr, has_bias) = requant_bias(m, cp, &[(cm, rsc), (cn, csc)], req.bias);

    // SAFETY: dims validated; ndarray guarantees valid in-bounds layouts; `c` (a `&mut u8` borrow)
    // can't alias `a`/`b`, and the bias was validated disjoint from C above. Reversed strides
    // forward straight through, exactly as the plain entry
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
                scale,
                row_scales,
                has_row_scales,
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
                scale,
                row_scales,
                has_row_scales,
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
/// feature. Like [`gemm`], it reads pointer/strides directly, so transposed, F-order,
/// and negative-stride views work without copying
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_common(None, alpha, a, conj_a, b, conj_b, beta, c, par);
}

/// Like [`gemm_cplx`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// If the inner dimensions disagree
#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, par);
}

#[cfg(feature = "complex")]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
) where
    T: ComplexScalar,
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

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a
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

/// Non-conjugated complex `A*B` into a fresh row-major [`Array2`]: the complex
/// analogue of [`dot`]. For conjugated products use [`gemm_cplx`]. Needs the
/// `complex` feature
#[cfg(feature = "complex")]
pub fn dot_cplx<T, S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<T>
where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let (m, _) = a.dim();
    let (_, n) = b.dim();
    // beta == 0, so the initial fill is never read
    let mut c = Array2::from_elem((m, n), T::ZERO);
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
/// `conj_a` (resp. `conj(B)` when `conj_b`): the ndarray adapter over gemmkit's
/// [`gemmkit::gemm_cplx_fused`]. The optional [`Bias`] is [`Bias::PerRow`] (length `A.rows`) or
/// [`Bias::PerCol`] (length `B.cols`), added verbatim (never conjugated); `bias == None` is exactly
/// [`gemm_cplx`]. There is **no** activation parameter: an ordering activation is undefined on
/// complex numbers. Like [`gemm_cplx`], it reads the pointer/strides directly and forwards to
/// gemmkit's raw engine, so transposed and reversed (negative-stride) views all work without copying
///
/// # Panics
/// If the inner dimensions disagree, or on a bias the adapter rejects (a `PerRow`/`PerCol` bias of
/// the wrong length, or a bias slice overlapping `C`)
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_fused_common(None, alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

/// Like [`gemm_cplx_fused`] but reuses a caller-owned [`Workspace`]
///
/// # Panics
/// Same conditions as [`gemm_cplx_fused`]
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_cplx_fused_with<T, S1, S2, SC>(
    ws: &mut Workspace,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>,
{
    gemm_cplx_fused_common(Some(ws), alpha, a, conj_a, b, conj_b, beta, c, bias, par);
}

#[cfg(all(feature = "complex", feature = "epilogue"))]
#[allow(clippy::too_many_arguments)]
fn gemm_cplx_fused_common<T, S1, S2, SC>(
    ws: Option<&mut Workspace>,
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    conj_a: bool,
    b: &ArrayBase<S2, Ix2>,
    conj_b: bool,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    bias: Option<Bias<'_, T>>,
    par: Parallelism,
) where
    T: ComplexScalar,
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
    // Fused-bias validation, replicating gemmkit's checked entry (byte-identical wording): the bias
    // length matches its axis and does not overlap C (raw pointer math, C is never referenced)
    // Complex has no activation (undefined on complex numbers), so there is no slope check
    let (bias_ptr, bias_dim, has_bias) = lower_bias(bias, m, n, cp, &[(cm, rsc), (cn, csc)]);

    // SAFETY: dims validated; ndarray guarantees the pointer/strides describe a valid in-bounds
    // layout and `c` (a `&mut` borrow) can't alias `a`/`b`; the bias was validated disjoint from C
    // above. Negative (reversed) strides forward straight through, exactly as the plain entry
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
