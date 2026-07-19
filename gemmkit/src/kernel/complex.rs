//! The complex GEMM family ([`ComplexGemm`]): `Complex<f32>` / `Complex<f64>`, with
//! `CONJ_A` / `CONJ_B` selecting conjugation of either input
//!
//! Unlike the other families this one does not build on [`super::float::FloatGemm`]:
//! it packs both operands into a split (structure-of-arrays) layout, real plane then
//! imaginary plane per depth step, so the microkernel's hot loop is pure real FMAs
//! (no in-loop shuffle or complex-multiply instruction). 1 complex multiply-accumulate
//! becomes 4 real FMAs into 2 accumulator banks: `acc_re += ar*br`, `acc_re -= ai*bi`,
//! `acc_im += ar*bi`, `acc_im += ai*br` (see `crate::simd::complex::soa_microkernel`,
//! which actually runs the loop)
//!
//! `pack_planar` (called from [`ComplexGemm::pack_lhs`] / [`ComplexGemm::pack_rhs`]) does
//! the de-interleaving once per element, O(MK + KN) total rather than redone every `kc`
//! step, so the kernel only ever issues contiguous loads. Because the kernel cannot
//! consume an interleaved operand, packing is mandatory: `FORCE_PACK_LHS = FORCE_PACK_RHS
//! = true`
//!
//! Conjugation is folded into the pack: a `const` conj flag negates the imaginary plane
//! as it is written, so `conj(A)*B` / `A*conj(B)` run through the identical real-FMA loop
//! with no per-element branch. Output conjugation is not implemented
//!
//! The family type stays homogeneous (`Lhs = Rhs = Acc = Out = T`), which is what lets
//! complex `alpha`/`beta` pass through the driver like any other family's, but the
//! driver's `KernelSimd<T, T, T, T>` bound only exposes `SimdOps<T>` (a thin, mostly
//! `unreachable!` shim, see `simd/complex.rs`), not the real-typed ops the kernel needs.
//! `ComplexGemm::microkernel` bridges that gap by calling
//! [`crate::simd::SimdOps::cplx_microkernel`], whose per-ISA implementation holds the
//! real `SimdOps<T::Real>` token and runs the one shared `soa_microkernel`. The pack and
//! the C store stay scalar: a vectorized de-interleave was tried on NEON and lost to the
//! inner FMA loop, so it was not kept (see `simd/neon.rs`)

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, Epilogue, KernelFamily};
use crate::scalar::{ComplexFloat, Scalar};
use crate::simd::KernelSimd;

/// The complex GEMM family: `T` is `Complex<f32>` or `Complex<f64>`, and `CONJ_A` /
/// `CONJ_B` select which input the pack conjugates (both `false` computes the plain
/// product `A*B`)
pub struct ComplexGemm<T, const CONJ_A: bool, const CONJ_B: bool>(PhantomData<T>);

impl<T, const CONJ_A: bool, const CONJ_B: bool> Clone for ComplexGemm<T, CONJ_A, CONJ_B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const CONJ_A: bool, const CONJ_B: bool> Copy for ComplexGemm<T, CONJ_A, CONJ_B> {}

/// Pack an `n_lead x depth_len` source block into `ceil(n_lead/width)` planar
/// micropanels: within a panel, every depth step writes `width` real parts
/// immediately followed by `width` imaginary parts (`conj` negates the imaginary
/// values as they are written). A tail panel is zero-padded past `n_lead`. This is
/// the de-interleaving counterpart of [`crate::pack`]'s micropanel copy: same 2-path
/// split (a contiguous-leading walk vs. a cache-blocked transpose for a strided
/// leading dimension, so a row-major source still packs without a cache miss per
/// element), same `lead`/`depth` convention (LHS: `lead = rows`, `depth = cols`; RHS
/// swaps them), but each element also splits into the re/im planes on the way in.
/// The 2 branches write identical bytes, only the traversal order differs
///
/// # Safety
/// `src` must cover the `n_lead x depth_len` region addressed by `lead`/`depth`;
/// `dst` must hold `ceil(n_lead/width) * width * depth_len` complex elements
/// (i.e. twice that many `T::Real`)
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn pack_planar<T: ComplexFloat>(
    dst: *mut T,
    src: *const T,
    lead: isize,
    depth: isize,
    n_lead: usize,
    depth_len: usize,
    width: usize,
    conj: bool,
) {
    unsafe {
        let tile = crate::tuning::pack_transpose_tile();
        let zero = <T::Real as Scalar>::ZERO;
        // `conj` negates on the way in, matching `num_complex::Complex::conj` (also
        // turns +0.0 into -0.0)
        let pack_im = |im: T::Real| if conj { -im } else { im };
        // dst is complex-typed but each panel interleaves 2 real planes, so write
        // through a real-typed cursor: `depth_len * 2 * width` reals per panel
        let mut panel = dst as *mut T::Real;
        let mut base = 0usize;
        while base < n_lead {
            let live = core::cmp::min(width, n_lead - base);
            if lead == 1 {
                // lead == 1: the `live` leading complex at each depth step are
                // contiguous in src, so read them straight and split re/im on the way in
                for p in 0..depth_len {
                    let re_off = p * 2 * width;
                    let s = src.offset(base as isize + p as isize * depth);
                    for i in 0..width {
                        if i < live {
                            let z = *s.add(i);
                            *panel.add(re_off + i) = z.re();
                            *panel.add(re_off + width + i) = pack_im(z.im());
                        } else {
                            *panel.add(re_off + i) = zero;
                            *panel.add(re_off + width + i) = zero;
                        }
                    }
                }
            } else {
                // Strided leading dimension: walk short tile-long strips along the
                // contiguous depth axis per leading row and scatter into the panel,
                // instead of gathering `width` strided elements per depth step (a cache
                // miss per element once lead is large)
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
                    for i in 0..width {
                        if i < live {
                            let row = src.offset((base + i) as isize * lead);
                            for p in p0..pe {
                                let z = *row.offset(p as isize * depth);
                                *panel.add(p * 2 * width + i) = z.re();
                                *panel.add(p * 2 * width + width + i) = pack_im(z.im());
                            }
                        } else {
                            for p in p0..pe {
                                *panel.add(p * 2 * width + i) = zero;
                                *panel.add(p * 2 * width + width + i) = zero;
                            }
                        }
                    }
                    p0 = pe;
                }
            }
            panel = panel.add(depth_len * 2 * width);
            base += width;
        }
    }
}

impl<T, const CONJ_A: bool, const CONJ_B: bool> KernelFamily for ComplexGemm<T, CONJ_A, CONJ_B>
where
    T: ComplexFloat,
{
    type Lhs = T;
    type Rhs = T;
    type Acc = T;
    type Out = T;

    // The SoA kernel only reads the planar layout, so both operands must always be
    // packed, never read in place
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;

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
        // LHS: rows are leading (stride rs), columns are depth (stride cs)
        unsafe {
            pack_planar(
                dst, src, /*lead*/ rs, /*depth*/ cs, mc, kc, mr, CONJ_A,
            );
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
        // RHS: columns are leading (stride cs), rows are depth (stride rs)
        unsafe {
            pack_planar(
                dst, src, /*lead*/ cs, /*depth*/ rs, nc, kc, nr, CONJ_B,
            );
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
        _b_cs: isize,
        c: *mut T,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut T,
    ) where
        S: KernelSimd<T, T, T, T>,
    {
        // Forward to the L0 SoA seam, translating AlphaStatus/BetaStatus to plain bools
        // (L0 must not depend on the L1 status enums). b_cs is dropped: a packed RHS
        // panel is always contiguous (b_cs == 1)
        unsafe {
            simd.cplx_microkernel::<MR_REG, NR>(
                kc,
                alpha,
                beta,
                alpha_status == AlphaStatus::One,
                beta_status == BetaStatus::Zero,
                beta_status == BetaStatus::One,
                a,
                a_cs,
                b,
                b_rs,
                c,
                rsc,
                csc,
                mr_eff,
                nr_eff,
                scratch,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
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
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut T,
    ) where
        S: KernelSimd<T, T, T, T>,
        E: Epilogue<Self>,
    {
        // cplx_microkernel (L0) cannot take E directly: it must not depend on the L1
        // Epilogue trait, same layering rule that keeps AlphaStatus/BetaStatus out of L0
        // So run the plain microkernel first (it stores the finished alpha*AB + beta*C
        // tile, the same bits plain gemm_cplx writes), then, only on the final depth
        // panel, sweep epi over the now-complete tile in place: since the store already
        // matches gemm_cplx exactly, this makes a fused call bitwise gemm_cplx followed
        // by the same per-element epi
        //
        // last_k gates the sweep because ComplexGemm defaults to OUT_IS_ACC = true: the
        // driver re-reads/re-writes C once per kc panel, so an earlier panel holds a raw
        // partial sum, not the finished value the epilogue must see exactly once
        //
        // !E::IS_IDENTITY is a const check, so with E = Identity (every non-fused call,
        // and every non-final panel of a fused one) the sweep const-folds away and this
        // override compiles to the same code as a bare call to microkernel
        unsafe {
            Self::microkernel::<S, MR_REG, NR>(
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
            );
            if !E::IS_IDENTITY && last_k {
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        let cp = c.offset(i as isize * rsc + j as isize * csc);
                        *cp = epi.apply(*cp, row0 + i, col0 + j);
                    }
                }
            }
        }
    }
}
