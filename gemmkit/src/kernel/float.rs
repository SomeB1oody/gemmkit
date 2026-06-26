//! The floating-point GEMM family: the *single* generic microkernel of the
//! library, plus the float pack layout and epilogue.
//!
//! This one function — [`FloatGemm::microkernel`] — covers every ISA (scalar,
//! FMA, AVX-512) and every tile (`MR_REG`, `NR`), because all the instruction
//! variation is hidden behind [`SimdOps`] and all the geometry variation behind
//! const generics. There is no macro, no per-ISA copy.

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::scalar::Float;
use crate::simd::SimdOps;

/// The real floating-point GEMM family (`Lhs = Rhs = Acc = Out = T`).
pub struct FloatGemm<T>(PhantomData<T>);

impl<T> Clone for FloatGemm<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for FloatGemm<T> {}

impl<T> KernelFamily for FloatGemm<T>
where
    T: Float<Acc = T>,
{
    type Lhs = T;
    type Rhs = T;
    type Acc = T;
    type Out = T;

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
            )
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
        // RHS panels are `nr` columns wide, stored row-by-row: the "leading"
        // direction is columns (stride `cs`) and the "depth" is rows (stride
        // `rs`) — the transpose of the LHS case, handled by swapping strides.
        unsafe {
            pack_panels(
                dst, src, /*lead*/ cs, /*depth*/ rs, /*n_lead*/ nc, kc, nr,
            )
        }
    }

    // The const-bounded index loops over `acc[j][i]` are deliberate: with
    // `MR_REG`/`NR` monomorphized they fully unroll and the optimizer promotes
    // each accumulator to a register. Iterator forms obscure that, so the range
    // loops stay.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
        S: SimdOps<T>,
    {
        unsafe {
            let lanes = <S as SimdOps<T>>::LANES;
            let mr = MR_REG * lanes;

            // --- accumulate: acc[j][i] = column j, rows i*lanes..(i+1)*lanes ---
            let mut acc: [[<S as SimdOps<T>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];

            if nr_eff == NR {
                // Hot path: full NR-wide tile. The const-bounded `j` loop
                // monomorphizes and fully unrolls (no `seq!` macro), so the
                // optimizer keeps every `acc[j][i]` in a register. Reads exactly
                // NR real RHS columns — correct whether B is packed or read in
                // place (adaptive skip).
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<T>>::Reg; MR_REG] =
                        core::array::from_fn(|i| simd.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..NR {
                        let bj = simd.splat(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            } else {
                // Edge column tile (`nr_eff < NR`): read exactly `nr_eff` columns
                // so an *unpacked* B is never read past its last real column. The
                // runtime bound costs unrolling, but this is only the trailing
                // column tile of a panel. (`acc[nr_eff..NR]` stay zero and are
                // ignored by the scratch epilogue below.)
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [<S as SimdOps<T>>::Reg; MR_REG] =
                        core::array::from_fn(|i| simd.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..nr_eff {
                        let bj = simd.splat(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            }

            // --- fold alpha into the accumulators (skip when alpha == 1) ---
            if alpha_status == AlphaStatus::Other {
                let av = simd.splat(alpha);
                for j in 0..NR {
                    for i in 0..MR_REG {
                        acc[j][i] = simd.mul(acc[j][i], av);
                    }
                }
            }

            // --- epilogue ---
            if mr_eff == mr && nr_eff == NR && rsc == 1 {
                // Fast path: full tile, column-major C → vector load/store.
                match beta_status {
                    BetaStatus::Zero => {
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                simd.storeu(col.add(i * lanes), acc[j][i]);
                            }
                        }
                    }
                    BetaStatus::One => {
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                let cv = simd.loadu(col.add(i * lanes));
                                simd.storeu(col.add(i * lanes), simd.add(cv, acc[j][i]));
                            }
                        }
                    }
                    BetaStatus::Other => {
                        let bv = simd.splat(beta);
                        for j in 0..NR {
                            let col = c.offset(j as isize * csc);
                            for i in 0..MR_REG {
                                let cv = simd.loadu(col.add(i * lanes));
                                // beta*C + alpha*AB
                                simd.storeu(col.add(i * lanes), simd.mul_add(cv, bv, acc[j][i]));
                            }
                        }
                    }
                }
            } else {
                // General / partial path: drain to contiguous column-major
                // scratch (`scratch[j*mr + row]`), then strided copy-back.
                for j in 0..NR {
                    for i in 0..MR_REG {
                        simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                    }
                }
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        let v = *scratch.add(j * mr + i); // == alpha*AB
                        let cp = c.offset(i as isize * rsc + j as isize * csc);
                        let out = match beta_status {
                            BetaStatus::Zero => v,
                            BetaStatus::One => *cp + v,
                            BetaStatus::Other => beta.mul_add(*cp, v), // beta*C + v
                        };
                        *cp = out;
                    }
                }
            }
        }
    }
}
