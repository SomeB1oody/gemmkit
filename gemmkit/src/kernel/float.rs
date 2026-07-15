//! The floating-point GEMM family: the single generic microkernel of the
//! library, plus the float pack layout and epilogue
//!
//! One generic function (`microkernel_impl`) covers every ISA (scalar, FMA,
//! AVX-512) and every tile (`MR_REG`, `NR`), because all the instruction variation
//! is hidden behind [`SimdOps`] and all the geometry variation behind const generics.
//! There is no macro, no per-ISA copy. The family exposes it through
//! [`FloatGemm::microkernel_epi`], the fused entry the driver calls (the plain
//! [`KernelFamily::microkernel`] is left as the trait's `unreachable!` default)

use core::marker::PhantomData;

use super::epilogue::Epilogue;
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::pack_panels;
use crate::scalar::Float;
use crate::simd::{KernelSimd, SimdOps};

/// The real floating-point GEMM family (`Lhs = Rhs = Acc = Out = T`)
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
        // `rs`), the transpose of the LHS case, handled by swapping strides
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

/// The one generic float microkernel, now parameterized over the fused [`Epilogue`] `E`
///
/// **Zero-cost identity.** Every epilogue use is gated on `!E::IS_IDENTITY` (and the fast
/// route on `E::IS_IDENTITY || E::VECTOR`), all associated `const`s. With `E = Identity`
/// the guards const-fold to `false`/`true` before LLVM sees them, `row0`/`col0`/`last_k`
/// become dead arguments, and the emitted code is the pre-epilogue kernel byte-for-byte.
/// With a `VECTOR` epilogue (`FusedEpi`) the fast path applies `E` to the very register the
/// store would have written, so a fused GEMM equals `gemm()` then a scalar map, bitwise
///
/// The const-bounded index loops over `acc[j][i]` are deliberate: with `MR_REG`/`NR`
/// monomorphized they fully unroll and the optimizer promotes each accumulator to a
/// register. Iterator forms obscure that, so the range loops stay
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

        // accumulate: acc[j][i] = column j, rows i*lanes..(i+1)*lanes
        let mut acc: [[<S as SimdOps<T>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];

        if nr_eff == NR {
            // Full tile: the hot kc-loop, routed through the overridable
            // [`SimdOps::accumulate_tile`] seam. The default impl is a
            // lane/splat loop the optimizer turns into the canonical
            // register-blocked kernel on wide OoO cores; a load-bound ISA
            // (e.g. AArch64 NEON, where the FMA pipes stall on operand
            // delivery at kc-loop boundaries) can override just this loop
            // with a software-pipelined schedule, leaving the driver,
            // packing, and epilogue untouched; the override reorders loads,
            // not arithmetic, so it rounds consistently with the edge path
            simd.accumulate_tile::<MR_REG, NR>(kc, a, a_cs, b, b_rs, b_cs, &mut acc);
        } else {
            // Edge column tile (`nr_eff < NR`): read exactly `nr_eff` columns
            // so an unpacked B is never read past its last real column. The
            // runtime bound costs unrolling, but this is only the trailing
            // column tile of a panel. (`acc[nr_eff..NR]` stay zero and are
            // ignored by the scratch epilogue below)
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

        // fold alpha into the accumulators (skip when alpha == 1)
        if alpha_status == AlphaStatus::Other {
            let av = simd.splat(alpha);
            for j in 0..NR {
                for i in 0..MR_REG {
                    acc[j][i] = simd.mul(acc[j][i], av);
                }
            }
        }

        // epilogue
        // The fast (vector) route is available to the identity kernel and to a `VECTOR`
        // epilogue; a scalar-only epilogue (`VECTOR = false`) forces the scratch route for
        // every tile. With `E = Identity` this is `(true || _) && ...`: the pre-epilogue
        // condition exactly
        if (E::IS_IDENTITY || E::VECTOR) && mr_eff == mr && nr_eff == NR && rsc == 1 {
            // Fast path: full tile, column-major C -> vector load/store, then (on the final
            // depth panel, for a non-identity epilogue) the fused vector transform applied
            // to the LANES consecutive rows the store would have written
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
                            // beta*C + alpha*AB
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
            // General / partial path: drain to contiguous column-major
            // scratch (`scratch[j*mr + row]`), then strided copy-back
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
                    // Apply the fused scalar transform on the final depth panel; on
                    // intermediate panels store the raw `Acc` partial. Bit-identical to the
                    // vector form above under the same token (the edge-consistency contract)
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
