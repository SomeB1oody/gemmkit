//! The mixed-precision GEMM family: narrow (`f16`/`bf16`) inputs and output,
//! `f32` accumulator. The first family where `Acc != Lhs`.
//!
//! It mirrors [`super::float::FloatGemm`] structurally but reaches every input
//! through the [`KernelSimd`] widen-load / narrow-store seam: A and B are widened
//! to `f32` registers on load, the dot products accumulate in `f32`, and the
//! epilogue rounds back to the narrow output (reading a narrow `C` widened for the
//! `beta != 0` term). No instruction variation leaks into the driver — the widening
//! lives entirely behind `KernelSimd`, so the same five-loop nest drives it.

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::scalar::NarrowFloat;
use crate::simd::{KernelSimd, SimdOps};

/// The mixed-precision GEMM family: `Lhs = Rhs = Out = N` (a [`NarrowFloat`], i.e.
/// `f16` or `bf16`), `Acc = f32`.
pub struct MixedGemm<N>(PhantomData<N>);

impl<N> Clone for MixedGemm<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N> Copy for MixedGemm<N> {}

impl<N> KernelFamily for MixedGemm<N>
where
    N: NarrowFloat,
{
    type Lhs = N;
    type Rhs = N;
    type Acc = f32;
    type Out = N;

    // `Out` (f16/bf16) is narrower than `Acc` (f32), so the running sum must NOT
    // round-trip through C between depth panels — the driver uses `kc = k` and the
    // whole contraction accumulates in f32 registers, rounding to N once.
    const OUT_IS_ACC: bool = false;

    #[inline]
    unsafe fn pack_lhs(
        dst: *mut N,
        src: *const N,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        // The packed layout is the plain micropanel copy (narrow elements stored as
        // is); widening happens later, on load in the microkernel.
        unsafe {
            pack_panels(
                dst, src, /*lead*/ rs, /*depth*/ cs, /*n_lead*/ mc, kc, mr,
            )
        }
    }

    #[inline]
    unsafe fn pack_rhs(
        dst: *mut N,
        src: *const N,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe {
            pack_panels(
                dst, src, /*lead*/ cs, /*depth*/ rs, /*n_lead*/ nc, kc, nr,
            )
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn microkernel<S, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: f32,
        beta: f32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const N,
        a_cs: isize,
        b: *const N,
        b_rs: isize,
        b_cs: isize,
        c: *mut N,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut f32,
    ) where
        S: KernelSimd<N, N, f32, N>,
    {
        unsafe {
            let lanes = <S as SimdOps<f32>>::LANES;
            let mr = MR_REG * lanes;

            // --- accumulate in f32: widen-load A, widen-broadcast B, fused FMA ---
            let mut acc: [[<S as SimdOps<f32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            if nr_eff == NR {
                // Full tile: the const-`NR` column loop fully unrolls, so every
                // accumulator stays in a register (the same discipline FloatGemm's
                // hot loop relies on). A runtime `nr_eff` bound here would collapse it.
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<f32>>::Reg; MR_REG] =
                        core::array::from_fn(|i| simd.load_lhs(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..NR {
                        let bj = simd.splat_rhs(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            } else {
                // Edge column tile (`nr_eff < NR`): read exactly `nr_eff` columns so
                // an unpacked B is never read past its last real column. `acc[nr_eff..]`
                // stay zero and are ignored by the scratch epilogue.
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<f32>>::Reg; MR_REG] =
                        core::array::from_fn(|i| simd.load_lhs(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..nr_eff {
                        let bj = simd.splat_rhs(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            }

            // --- fold alpha into the f32 accumulators (skip when alpha == 1) ---
            if alpha_status == AlphaStatus::Other {
                let av = simd.splat(alpha);
                for j in 0..NR {
                    for i in 0..MR_REG {
                        acc[j][i] = simd.mul(acc[j][i], av);
                    }
                }
            }

            // --- epilogue: read narrow C (widened), combine in f32, round to N ---
            if mr_eff == mr && nr_eff == NR && rsc == 1 {
                // Fast path: full tile, column-major C → vector widen-load / store.
                match beta_status {
                    BetaStatus::Zero => {
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                simd.store_out(col.add(i * lanes), acc[j][i]);
                            }
                        }
                    }
                    BetaStatus::One => {
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                let cv = simd.load_out(col.add(i * lanes));
                                simd.store_out(col.add(i * lanes), simd.add(cv, acc[j][i]));
                            }
                        }
                    }
                    BetaStatus::Other => {
                        let bv = simd.splat(beta);
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                let cv = simd.load_out(col.add(i * lanes));
                                // beta*C + alpha*AB, all in f32, then rounded on store.
                                simd.store_out(col.add(i * lanes), simd.mul_add(cv, bv, acc[j][i]));
                            }
                        }
                    }
                }
            } else {
                // General / partial path: drain f32 acc to f32 scratch, then strided
                // copy-back with a per-element widen (read C) / narrow (write C).
                for j in 0..NR {
                    for i in 0..MR_REG {
                        simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                    }
                }
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        let v = *scratch.add(j * mr + i); // f32 == alpha*AB
                        let cp = c.offset(i as isize * rsc + j as isize * csc);
                        let out = match beta_status {
                            BetaStatus::Zero => v,
                            BetaStatus::One => (*cp).widen() + v,
                            BetaStatus::Other => beta * (*cp).widen() + v,
                        };
                        *cp = N::narrow(out);
                    }
                }
            }
        }
    }
}
