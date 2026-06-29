//! The complex GEMM family: `Complex<f32>` / `Complex<f64>`, with optional
//! conjugation of `A` and/or `B`.
//!
//! Complex is *homogeneous* (`Lhs = Rhs = Acc = Out`) and has native arithmetic, so
//! it rides the entire float path: the per-ISA `SimdOps<Complex<_>>` make `mul` /
//! `mul_add` the **vectorized complex multiply** (interleaved re/im, shuffle +
//! `fmaddsub`), and `ComplexGemm`'s microkernel simply **delegates to
//! [`super::float::FloatGemm`]'s** — one kernel serves both.
//!
//! **Conjugation is on the pack seam, not the hot loop.** `conjA` / `conjB` are
//! `const` parameters: when set, the packed `A` (resp. `B`) panel is conjugated
//! *during packing*, so `A̅·B` / `A·B̅` fall out of the same plain complex FMA. There
//! is no per-element conj branch in the microkernel, exactly as the op-family seam
//! intends. `conjC` (a final output conjugation) is a deferred increment.

use core::marker::PhantomData;

use super::float::FloatGemm;
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::scalar::{Conjugate, Float};
use crate::simd::KernelSimd;

/// The complex GEMM family. `T` is `Complex<f32>` or `Complex<f64>`; `CONJ_A` /
/// `CONJ_B` select which input is conjugated (both `false` = the plain product).
pub struct ComplexGemm<T, const CONJ_A: bool, const CONJ_B: bool>(PhantomData<T>);

impl<T, const CONJ_A: bool, const CONJ_B: bool> Clone for ComplexGemm<T, CONJ_A, CONJ_B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const CONJ_A: bool, const CONJ_B: bool> Copy for ComplexGemm<T, CONJ_A, CONJ_B> {}

/// Conjugate every element of a just-packed panel in place.
///
/// # Safety
/// `dst` valid for `count` reads+writes.
#[inline]
unsafe fn conjugate_panel<T: Conjugate>(dst: *mut T, count: usize) {
    unsafe {
        for i in 0..count {
            *dst.add(i) = (*dst.add(i)).conjugate();
        }
    }
}

impl<T, const CONJ_A: bool, const CONJ_B: bool> KernelFamily for ComplexGemm<T, CONJ_A, CONJ_B>
where
    T: Float<Acc = T> + Conjugate,
{
    type Lhs = T;
    type Rhs = T;
    type Acc = T;
    type Out = T;

    // Conjugation happens at pack time, so a conjugated operand must always be
    // packed — otherwise the driver could read it in place, unconjugated.
    const FORCE_PACK_LHS: bool = CONJ_A;
    const FORCE_PACK_RHS: bool = CONJ_B;

    #[inline]
    unsafe fn pack_lhs(
        dst: *mut T,
        src: *const T,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        unsafe {
            pack_panels(
                dst, src, /*lead*/ rs, /*depth*/ cs, /*n_lead*/ mc, kc, mr,
            );
            if CONJ_A {
                // Conjugate the whole packed panel (live rows + zero pad; conj of 0
                // is 0, so the pad is unaffected). Count = ceil(mc/mr)*mr*kc.
                conjugate_panel(dst, mc.div_ceil(mr) * mr * kc);
            }
        }
    }

    #[inline]
    unsafe fn pack_rhs(
        dst: *mut T,
        src: *const T,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe {
            pack_panels(
                dst, src, /*lead*/ cs, /*depth*/ rs, /*n_lead*/ nc, kc, nr,
            );
            if CONJ_B {
                conjugate_panel(dst, nc.div_ceil(nr) * nr * kc);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel<S, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: T,
        beta: T,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const T,
        a_cs: isize,
        b: *const T,
        b_rs: isize,
        b_cs: isize,
        c: *mut T,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut T,
    ) where
        S: KernelSimd<T, T, T, T>,
    {
        // The conj is already baked into the packed A/B, so the arithmetic is the
        // plain complex GEMM — reuse FloatGemm's microkernel verbatim (its `mul_add`
        // is the vectorized complex FMA via `SimdOps<Complex<_>>`).
        unsafe {
            <FloatGemm<T> as KernelFamily>::microkernel::<S, MR_REG, NR>(
                simd,
                kc,
                alpha,
                beta,
                alpha_status,
                beta_status,
                a,
                a_cs,
                b,
                b_rs,
                b_cs,
                c,
                rsc,
                csc,
                mr_eff,
                nr_eff,
                scratch,
            )
        }
    }
}
