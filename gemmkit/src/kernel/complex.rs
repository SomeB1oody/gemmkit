//! The complex GEMM family ([`ComplexGemm`]) covers `Complex<f32>` and `Complex<f64>`.
//! `CONJ_A` and `CONJ_B` select conjugation of either input
//!
//! Unlike the other families, this one does not build on [`super::float::FloatGemm`]. It
//! packs both operands into a split, structure-of-arrays layout, with the real plane then
//! the imaginary plane stored per depth step. This keeps the microkernel's hot loop to
//! pure real FMAs, with no in-loop shuffle or complex-multiply instruction. 1 complex
//! multiply-accumulate becomes 4 real FMAs into 2 accumulator banks: `acc_re += ar*br`,
//! `acc_re -= ai*bi`, `acc_im += ar*bi`, and `acc_im += ai*br`. See
//! `crate::simd::complex::soa_microkernel`, which runs the loop
//!
//! [`ComplexGemm::pack_lhs`] and [`ComplexGemm::pack_rhs`] call `pack_planar`, which
//! de-interleaves each element once. The total cost is `O(MK + KN)` rather than redoing
//! the work on every `kc` step. This lets the kernel issue only contiguous loads. The
//! kernel cannot consume an interleaved operand, so packing is mandatory: both
//! `FORCE_PACK_LHS` and `FORCE_PACK_RHS` are `true`
//!
//! Conjugation is folded into the pack. A conj flag negates the imaginary plane as it is
//! written, so `conj(A)*B` and `A*conj(B)` run through the identical real-FMA loop with no
//! per-element branch. Output conjugation is not implemented
//!
//! The family type stays homogeneous, with `Lhs = Rhs = Acc = Out = T`. This lets complex
//! `alpha` and `beta` pass through the driver like any other family's. The driver's
//! `KernelSimd<T, T, T, T>` bound only exposes `SimdOps<T>`, not the real-typed ops the
//! kernel needs. `SimdOps<T>` is a thin shim that is mostly `unreachable!` (see
//! `simd/complex.rs`). `ComplexGemm::microkernel` bridges that gap by calling
//! [`crate::simd::SimdOps::cplx_microkernel`]. Its per-ISA implementation holds the real
//! `SimdOps<T::Real>` token and runs the one shared `soa_microkernel`. The pack and the C
//! store stay scalar. De-interleaving and re-interleaving are a small part of the total
//! cost next to the vectorized inner FMA loop

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, Epilogue, KernelFamily};
use crate::scalar::{ComplexFloat, Scalar};
use crate::simd::KernelSimd;

/// The complex GEMM family. `T` is `Complex<f32>` or `Complex<f64>`, and `CONJ_A` and
/// `CONJ_B` select which input the pack conjugates. Both `false` computes the plain
/// product `A*B`
pub struct ComplexGemm<T, const CONJ_A: bool, const CONJ_B: bool>(PhantomData<T>);

impl<T, const CONJ_A: bool, const CONJ_B: bool> Clone for ComplexGemm<T, CONJ_A, CONJ_B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const CONJ_A: bool, const CONJ_B: bool> Copy for ComplexGemm<T, CONJ_A, CONJ_B> {}

/// Pack an `n_lead x depth_len` source block into `ceil(n_lead/width)` planar
/// micropanels. Within a panel, every depth step writes `width` real parts immediately
/// followed by `width` imaginary parts. `conj` negates the imaginary values as they are
/// written. A tail panel is zero-padded past `n_lead`
///
/// This is the de-interleaving counterpart of [`crate::pack`]'s micropanel copy. It uses
/// the same 2-path split: a contiguous-leading walk, or a cache-blocked transpose for a
/// strided leading dimension. The transpose path lets a row-major source pack without a
/// cache miss per element. It uses the same `lead`/`depth` convention: for the LHS,
/// `lead = rows` and `depth = cols`, and the RHS swaps them. Each element also splits
/// into the real and imaginary planes on the way in. The 2 branches write identical
/// bytes. Only the traversal order differs
///
/// # Safety
///
/// `src` must cover the `n_lead x depth_len` region addressed by `lead` and `depth`.
/// `dst` must hold `ceil(n_lead/width) * width * depth_len` complex elements, twice that
/// many `T::Real`
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
        // `conj` negates on the way in, matching `num_complex::Complex::conj`. This also
        // turns +0.0 into -0.0
        let pack_im = |im: T::Real| if conj { -im } else { im };
        // dst is complex-typed but each panel interleaves 2 real planes, so write through
        // a real-typed cursor: `depth_len * 2 * width` reals per panel
        let mut panel = dst as *mut T::Real;
        let mut base = 0usize;
        while base < n_lead {
            let live = core::cmp::min(width, n_lead - base);
            if lead == 1 {
                // lead == 1: the `live` leading complex at each depth step are
                // contiguous in src. Read them straight and split re/im on the way in
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
                // Strided leading dimension: walk short strips along the contiguous depth
                // axis per leading row, instead of gathering strided elements one cache miss
                // at a time
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
        // Forward to the L0 seam as plain bools (L0 must not depend on the L4 status enums)
        // A packed RHS panel is always contiguous, so b_cs is dropped
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
        // cplx_microkernel (L0) must not depend on the L4 Epilogue trait. This override
        // runs the plain microkernel first and then sweeps epi over the finished tile
        // That keeps a fused call bitwise equal to gemm_cplx followed by the same
        // per-element epi
        //
        // last_k gates the sweep, because ComplexGemm defaults to OUT_IS_ACC = true, so an
        // earlier panel holds a raw partial sum rather than the finished value
        // `!E::IS_IDENTITY` is a const check, so an unfused call or a non-final panel
        // const-folds this override to a bare call to `microkernel`
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
