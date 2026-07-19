//! The floating-point GEMM family: `Lhs = Rhs = Acc = Out`, the plain `f32`/`f64` case
//!
//! 1 generic function (`microkernel_impl`) covers every ISA and every tile size:
//! the instruction set varies only through [`SimdOps`]/[`KernelSimd`], and the tile
//! shape only through the `MR_REG`/`NR` const generics, so there is no macro and no
//! per-ISA copy of the kernel body. [`FloatGemm`] wires it in via
//! [`KernelFamily::microkernel_epi`], the fused entry the driver calls; the plain
//! [`KernelFamily::microkernel`] is left at the trait's `unreachable!` default since
//! this family never takes that path

use core::marker::PhantomData;

use super::epilogue::Epilogue;
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::scalar::Float;
use crate::simd::{KernelSimd, SimdOps};

/// The real floating-point GEMM family: `Lhs = Rhs = Acc = Out = T`
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
        // RHS panels lead on columns (stride `cs`) and step depth on rows (stride
        // `rs`), the transpose of pack_lhs's roles: swap the strides passed to pack_panels
        unsafe {
            pack_panels(
                dst, src, /*lead*/ cs, /*depth*/ rs, /*n_lead*/ nc, kc, nr,
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
        unsafe {
            microkernel_impl::<T, S, E, MR_REG, NR>(
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
                row0,
                col0,
                last_k,
                epi,
                scratch,
            )
        }
    }
}

/// Compute one `MR x NR` output tile and store it through the fused epilogue `E`: the shared
/// body behind [`FloatGemm::microkernel_epi`], generic over the ISA token `S` and the tile
/// shape `MR_REG`/`NR`
///
/// Every epilogue application is gated on `!E::IS_IDENTITY`, and every `const` involved
/// (`E::IS_IDENTITY`, `E::VECTOR`) is known at monomorphization time, so with `E = Identity`
/// the guards fold to `false` before codegen, `row0`/`col0`/`last_k` become dead arguments, and
/// the emitted code matches a kernel with no epilogue seam at all. With a `VECTOR` epilogue the
/// fast path applies `E` directly to the register the store would otherwise have written
/// unchanged, so a fused GEMM equals plain `gemm()` followed by a scalar map, bit-for-bit
///
/// The index loops over `acc[j][i]` are bounded by the const generics `MR_REG`/`NR` rather than
/// written as iterators, so the compiler fully unrolls them and keeps every accumulator in a
/// register instead of spilling to the stack
///
/// # Safety
/// As [`KernelFamily::microkernel_epi`]; run inside `S`'s [`crate::simd::Simd::vectorize`]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn microkernel_impl<T, S, E, const MR_REG: usize, const NR: usize>(
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
    T: Float<Acc = T>,
    S: KernelSimd<T, T, T, T>,
    E: Epilogue<FloatGemm<T>>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let mr = MR_REG * lanes;

        // acc[j][i]: column j, rows [i*lanes, (i+1)*lanes)
        let mut acc: [[<S as SimdOps<T>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];

        if nr_eff == NR {
            // Full tile: the hot kc-loop, via the overridable accumulate_tile seam so an
            // ISA can substitute its own schedule without touching this function
            simd.accumulate_tile::<MR_REG, NR>(kc, a, a_cs, b, b_rs, b_cs, &mut acc);
        } else {
            // Edge tile: bound the column loop to nr_eff so an unpacked B is never read
            // past its last real column; acc[nr_eff..] stays zero, dropped below
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

        // fold alpha into the accumulators; skip the multiply entirely when alpha == 1
        if alpha_status == AlphaStatus::Other {
            let av = simd.splat(alpha);
            for j in 0..NR {
                for i in 0..MR_REG {
                    acc[j][i] = simd.mul(acc[j][i], av);
                }
            }
        }

        // A scalar-only epilogue (VECTOR == false) has no apply_reg, so it must take the
        // scratch route below for every tile; Identity and any VECTOR epilogue can take
        // the vector route whenever the tile itself is full and column-major
        if (E::IS_IDENTITY || E::VECTOR) && mr_eff == mr && nr_eff == NR && rsc == 1 {
            // Vector load/store of the full tile; apply_reg runs on the LANES rows just
            // loaded/computed, only on the last depth panel and only for a real epilogue
            match beta_status {
                BetaStatus::Zero => {
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let r = acc[j][i];
                            let r = if !E::IS_IDENTITY && last_k {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.storeu(col.add(i * lanes), r);
                        }
                    }
                }
                BetaStatus::One => {
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let cv = simd.loadu(col.add(i * lanes));
                            let r = simd.add(cv, acc[j][i]);
                            let r = if !E::IS_IDENTITY && last_k {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.storeu(col.add(i * lanes), r);
                        }
                    }
                }
                BetaStatus::Other => {
                    let bv = simd.splat(beta);
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let cv = simd.loadu(col.add(i * lanes));
                            // beta*C + alpha*AB, one fused multiply-add
                            let r = simd.mul_add(cv, bv, acc[j][i]);
                            let r = if !E::IS_IDENTITY && last_k {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.storeu(col.add(i * lanes), r);
                        }
                    }
                }
            }
        } else {
            // Edge or non-unit-stride tile: drain to contiguous column-major scratch,
            // then copy back element by element under the tile's real strides
            for j in 0..NR {
                for i in 0..MR_REG {
                    simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                }
            }
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    let v = *scratch.add(j * mr + i); // alpha*AB
                    let cp = c.offset(i as isize * rsc + j as isize * csc);
                    let out = match beta_status {
                        BetaStatus::Zero => v,
                        BetaStatus::One => *cp + v,
                        BetaStatus::Other => beta.mul_add(*cp, v), // beta*C + alpha*AB
                    };
                    // apply on the last depth panel only, matching the vector path above
                    *cp = if !E::IS_IDENTITY && last_k {
                        epi.apply(out, row0 + i, col0 + j)
                    } else {
                        out
                    };
                }
            }
        }
    }
}
