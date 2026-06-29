//! The integer GEMM family: `i8` inputs, `i32` accumulator and output.
//!
//! Like [`super::mixed::MixedGemm`], inputs reach the kernel through the
//! [`KernelSimd`] widen seam (`i8 -> i32` sign-extend on load) and accumulate in
//! the wider type. Since `Out == Acc == i32` the result is exact, needs no
//! narrowing, and the driver blocks K normally (`OUT_IS_ACC` stays `true`).
//! Arithmetic wraps on overflow, the conventional integer-GEMM semantics.
//!
//! This is the widen-and-multiply kernel. A denser VNNI `vpdpbusd` dot kernel
//! would need its own K-interleaved pack layout and `u8 × i8` signedness handling
//! (the `pack.rs` doc anticipates that interleave).

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::simd::{KernelSimd, SimdOps};

/// The integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`.
pub struct IntGemm(PhantomData<()>);

impl Clone for IntGemm {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for IntGemm {}

impl KernelFamily for IntGemm {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = i32;

    #[inline]
    unsafe fn pack_lhs(
        dst: *mut i8,
        src: *const i8,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        // Plain micropanel copy of the `i8` inputs; sign-extension to `i32`
        // happens on load in the kernel.
        unsafe {
            pack_panels(
                dst, src, /*lead*/ rs, /*depth*/ cs, /*n_lead*/ mc, kc, mr,
            )
        }
    }

    #[inline]
    unsafe fn pack_rhs(
        dst: *mut i8,
        src: *const i8,
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
        alpha: i32,
        beta: i32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const i8,
        a_cs: isize,
        b: *const i8,
        b_rs: isize,
        b_cs: isize,
        c: *mut i32,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut i32,
    ) where
        S: KernelSimd<i8, i8, i32, i32>,
    {
        unsafe {
            let lanes = <S as SimdOps<i32>>::LANES;
            let mr = MR_REG * lanes;

            // --- accumulate in i32: sign-extend A/B to i32, multiply-add ---
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            if nr_eff == NR {
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<i32>>::Reg; MR_REG] =
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
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<i32>>::Reg; MR_REG] =
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

            // --- fold alpha (skip when alpha == 1) ---
            if alpha_status == AlphaStatus::Other {
                let av = simd.splat(alpha);
                for j in 0..NR {
                    for i in 0..MR_REG {
                        acc[j][i] = simd.mul(acc[j][i], av);
                    }
                }
            }

            // --- epilogue (Out == Acc == i32, exact) ---
            if mr_eff == mr && nr_eff == NR && rsc == 1 {
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
                                // beta*C + alpha*AB (wrapping i32).
                                simd.store_out(col.add(i * lanes), simd.mul_add(cv, bv, acc[j][i]));
                            }
                        }
                    }
                }
            } else {
                // General / partial path: drain acc to i32 scratch, strided copy-back.
                for j in 0..NR {
                    for i in 0..MR_REG {
                        simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                    }
                }
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        let v = *scratch.add(j * mr + i); // i32 == alpha*AB
                        let cp = c.offset(i as isize * rsc + j as isize * csc);
                        let out = match beta_status {
                            BetaStatus::Zero => v,
                            BetaStatus::One => (*cp).wrapping_add(v),
                            BetaStatus::Other => beta.wrapping_mul(*cp).wrapping_add(v),
                        };
                        *cp = out;
                    }
                }
            }
        }
    }
}
