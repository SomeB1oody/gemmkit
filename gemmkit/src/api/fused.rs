//! Fused-epilogue (bias / activation) GEMM entries.
use super::*;
use crate::dispatch::FusedScalar;
use crate::kernel::epilogue::{Act, BiasDim, BiasSpec, FusedEpi};
use crate::parallel::Ptr;

/// A bias vector fused into a [`gemm_fused`] call: one value per output **row** (length `m`)
/// or per output **column** (length `n`), added to every element of that row / column after
/// the product and before the activation.
pub enum Bias<'a, T> {
    /// One value per output row (length `m`).
    PerRow(&'a [T]),
    /// One value per output column (length `n`).
    PerCol(&'a [T]),
}

/// An activation fused into a [`gemm_fused`] call, applied last (after the bias add).
pub enum Activation<T> {
    /// `max(v, 0)` (NaN maps to 0).
    Relu,
    /// `max(v, 0) + slope·min(v, 0)` (NaN maps to 0, −0 to +0).
    LeakyRelu(T),
}

/// `true` iff `s` is finite, expressed generically (no `f32`/`f64` inherent method on the
/// generic `T`): a finite value equals itself and `s - s == 0`, while `±inf - ±inf = NaN != 0`
/// and `NaN != NaN`. The `eq_op` lint is expected — the self-comparison is the point.
#[inline]
#[allow(clippy::eq_op)]
fn is_finite<T: FusedScalar>(s: T) -> bool {
    s == s && (s - s) == T::ZERO
}

/// `C <- act(alpha·A·B + beta·C + bias)` in one pass: a **fused** GEMM epilogue over safe
/// slice views, using the thread-local workspace pool. The bias is added by one IEEE add
/// after the final `beta`-fold, then the activation is applied. `bias == None && act == None`
/// delegates to plain [`gemm`].
///
/// The fused engine routes every shape through the **same** kernel `gemm` would use — the
/// general register-blocked driver, gemv (`m == 1` / `n == 1`), the small-`m,n` horizontal
/// path, or the small-`k` path — fusing the epilogue into that kernel's store without
/// perturbing its accumulation order. So the result is **bit-identical** to `gemm()` followed
/// by the same scalar map, for **every** shape, and deterministic across thread counts.
///
/// # Panics
/// Same conditions as [`gemm`], plus: a `PerRow` bias whose length is not `A.rows` (or a
/// `PerCol` bias not `B.cols`); a bias slice that overlaps `C`; or a non-finite `LeakyRelu`
/// slope.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused<T: FusedScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_fused_with(ws, alpha, a, b, beta, c, bias, act, par));
}

/// Like [`gemm_fused`] but reuses a caller-owned [`Workspace`].
///
/// # Panics
/// Same conditions as [`gemm_fused`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_fused_with<T: FusedScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    bias: Option<Bias<'_, T>>,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    validate_gemm_views(&a, &b, &c);

    // Fused-epilogue validation: bias length matches its axis and does not overlap C.
    if let Some(bd) = &bias {
        let (bp, bl) = match bd {
            Bias::PerRow(s) => {
                assert_eq!(
                    s.len(),
                    a.rows,
                    "gemmkit: PerRow bias length ({}) != A.rows ({})",
                    s.len(),
                    a.rows
                );
                (s.as_ptr(), s.len())
            }
            Bias::PerCol(s) => {
                assert_eq!(
                    s.len(),
                    b.cols,
                    "gemmkit: PerCol bias length ({}) != B.cols ({})",
                    s.len(),
                    b.cols
                );
                (s.as_ptr(), s.len())
            }
        };
        if overlaps(c.data.as_ptr(), c.data.len(), bp, bl) {
            panic!("gemmkit: bias slice overlaps C");
        }
    }
    if let Some(Activation::LeakyRelu(s)) = &act {
        assert!(is_finite(*s), "gemmkit: LeakyRelu slope must be finite");
    }

    // The identity-fused case cannot even reach a fused monomorphization: delegate to plain
    // gemm so the zero-cost path is guaranteed.
    if bias.is_none() && act.is_none() {
        gemm_with(ws, alpha, a, b, beta, c, par);
        return;
    }

    let bias_spec = match bias {
        None => BiasSpec::None,
        Some(Bias::PerRow(s)) => BiasSpec::Row(Ptr(s.as_ptr() as *mut T)),
        Some(Bias::PerCol(s)) => BiasSpec::Col(Ptr(s.as_ptr() as *mut T)),
    };
    let act_e = match act {
        None => Act::None,
        Some(Activation::Relu) => Act::Relu,
        Some(Activation::LeakyRelu(s)) => Act::LeakyRelu(s),
    };
    let epi = FusedEpi {
        bias: bias_spec,
        act: act_e,
    };

    // SAFETY: validated above — shapes agree, every stride is in bounds, C addresses each
    // (i,j) uniquely and does not alias A/B, and the bias slice (borrowed for this call) does
    // not overlap C. The bias pointer stays valid for the whole `execute_fused` frame.
    unsafe {
        dispatch::execute_fused(
            Task {
                m: a.rows,
                k: a.cols,
                n: b.cols,
                alpha,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                b: b.data.as_ptr(),
                rsb: b.rs,
                csb: b.cs,
                beta,
                c: c.data.as_mut_ptr(),
                rsc: c.rs,
                csc: c.cs,
            },
            epi,
            par,
            ws,
        );
    }
}

/// The raw fused engine: `C <- act(alpha·A·B + beta·C + bias)` over pointers and `isize`
/// strides, with **no** bounds/alias/shape checks. `bias` is a `(ptr, dim)` pair enabled by
/// `has_bias` (`bias` is ignored when `has_bias == false`). Uses the thread-local workspace
/// pool.
///
/// # Safety
/// As [`gemm_unchecked`], plus: when `has_bias`, `bias` is valid for reads of `m` (`PerRow`)
/// or `n` (`PerCol`) elements and does not alias `c`; and a non-finite `LeakyRelu` slope is
/// the caller's responsibility (the checked API rejects it).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_fused_unchecked<T: FusedScalar>(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
    par: Parallelism,
) {
    let bias_spec = if has_bias {
        match bias_dim {
            BiasDim::PerRow => BiasSpec::Row(Ptr(bias as *mut T)),
            BiasDim::PerCol => BiasSpec::Col(Ptr(bias as *mut T)),
        }
    } else {
        BiasSpec::None
    };
    let act_e = match act {
        None => Act::None,
        Some(Activation::Relu) => Act::Relu,
        Some(Activation::LeakyRelu(s)) => Act::LeakyRelu(s),
    };
    let epi = FusedEpi {
        bias: bias_spec,
        act: act_e,
    };
    // SAFETY: preconditions forwarded to the caller (see # Safety).
    unsafe {
        workspace::with_thread_pool(|ws| {
            dispatch::execute_fused(
                Task {
                    m,
                    k,
                    n,
                    alpha,
                    a,
                    rsa,
                    csa,
                    b,
                    rsb,
                    csb,
                    beta,
                    c,
                    rsc,
                    csc,
                },
                epi,
                par,
                ws,
            );
        });
    }
}
