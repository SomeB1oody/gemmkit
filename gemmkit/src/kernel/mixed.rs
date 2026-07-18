//! The mixed-precision GEMM family: narrow (`f16`/`bf16`) inputs and output,
//! `f32` accumulator. The first family where `Acc != Lhs`
//!
//! Structurally mirrors [`super::float::FloatGemm`] but reaches every input
//! through the [`KernelSimd`] widen-load / narrow-store seam: A and B widen to
//! `f32` on load, dot products accumulate in `f32`, and the epilogue rounds back
//! to the narrow output (widening a narrow `C` for the `beta != 0` term). The
//! widening lives entirely behind `KernelSimd`, so the same 5-loop nest drives
//! it with no instruction variation in the driver

use core::marker::PhantomData;

use super::epilogue::Epilogue;
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::{pack_kgroup_panels, pack_panels};
use crate::scalar::NarrowFloat;
use crate::simd::{KernelSimd, SimdOps};

/// The widen-FMA accumulation loops of [`MixedGemm::microkernel_epi`], factored into their own
/// helper so the fused kernel keeps a clean inner nest. Widen-loads A, widen-broadcasts B, and
/// fuses into the `f32` `acc` (**not** pre-zeroed - the caller may seed it with the running
/// partial `C` to continue the ascending-k chain across depth slices). `nr_eff == NR` fully
/// unrolls the const-`NR` column loop (every accumulator in a register); the edge branch reads
/// exactly `nr_eff` columns so an unpacked B is never read past its last real column
///
/// Generic over the output type `O` of the `KernelSimd` seam so both the narrow family
/// (`O = N`) and its f32-output twin (`O = f32`) drive it: only `load_lhs`/`splat_rhs` are
/// used (widen `N -> f32`, identical bits for either `O` since the f32-out seam forwards them),
/// never `load_out`/`store_out`, so the accumulator each produces is byte-for-byte the same
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn mixed_accumulate<N, S, O, const MR_REG: usize, const NR: usize>(
    simd: S,
    kc: usize,
    a: *const N,
    a_cs: isize,
    b: *const N,
    b_rs: isize,
    b_cs: isize,
    nr_eff: usize,
    acc: &mut [[<S as SimdOps<f32>>::Reg; MR_REG]; NR],
) where
    N: NarrowFloat,
    O: crate::scalar::Scalar,
    S: KernelSimd<N, N, f32, O>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        if nr_eff == NR {
            // Full tile: the const-`NR` column loop fully unrolls, keeping every
            // accumulator in a register. A runtime `nr_eff` bound would collapse it
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
            // Edge column tile (`nr_eff < NR`): read exactly `nr_eff` columns so an
            // unpacked B is never read past its last real column. `acc[nr_eff..]`
            // stay zero and are ignored by the scratch epilogue
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
    }
}

/// Fold `alpha` into the `f32` accumulators and apply the mixed-precision epilogue, threading the
/// fused [`Epilogue`] `E`: read the narrow `C` widened, combine in `f32`, apply `E` (bias /
/// activation) in `f32`, and round to `N` once on store. Shared verbatim by [`MixedGemm`]
/// (widen-and-FMA) and [`Bf16DotGemm`] (`vdpbf16ps` dot): they differ only in how `acc` is
/// produced, so the `f32`-acc / narrow-`Out` epilogue lives here once. The whole contraction has
/// accumulated in `f32` (`OUT_IS_ACC = false` => `kc = k`, a single depth panel), so there is no
/// `last_k` gate: the epilogue fires unconditionally here, exactly once per element. With
/// `E = Identity` every epilogue hook const-folds away (`E::IS_IDENTITY`), so the non-fused call
/// is byte-for-byte the pre-epilogue kernel
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`]. `E`'s
/// interior pointers must be valid for the problem's `m`/`n`
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn mixed_epilogue<Fam, N, S, E, const MR_REG: usize, const NR: usize>(
    simd: S,
    alpha: f32,
    beta: f32,
    alpha_status: AlphaStatus,
    beta_status: BetaStatus,
    acc: &mut [[<S as SimdOps<f32>>::Reg; MR_REG]; NR],
    c: *mut N,
    rsc: isize,
    csc: isize,
    mr_eff: usize,
    nr_eff: usize,
    row0: usize,
    col0: usize,
    epi: &E,
    scratch: *mut f32,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
    E: Epilogue<Fam>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mr = MR_REG * lanes;

        // fold alpha into the f32 accumulators (skip when alpha == 1)
        if alpha_status == AlphaStatus::Other {
            let av = simd.splat(alpha);
            for j in 0..NR {
                for i in 0..MR_REG {
                    acc[j][i] = simd.mul(acc[j][i], av);
                }
            }
        }

        // epilogue: read narrow C (widened), combine in f32, apply E, round to N
        // These families are `OUT_IS_ACC = false` (`kc = k`, a single panel), so the epilogue
        // fires here unconditionally: no `last_k` gate. `E::IS_IDENTITY` const-folds the hook
        // away for the plain kernel; a `VECTOR` epilogue applies `apply_reg` to the `f32` register
        // and `store_out` performs the single narrowing
        if (E::IS_IDENTITY || E::VECTOR) && mr_eff == mr && nr_eff == NR && rsc == 1 {
            // Fast path: full tile, column-major C -> vector widen-load / store
            match beta_status {
                BetaStatus::Zero => {
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let r = acc[j][i];
                            let r = if !E::IS_IDENTITY {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.store_out(col.add(i * lanes), r);
                        }
                    }
                }
                BetaStatus::One => {
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let cv = simd.load_out(col.add(i * lanes));
                            let r = simd.add(cv, acc[j][i]);
                            let r = if !E::IS_IDENTITY {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.store_out(col.add(i * lanes), r);
                        }
                    }
                }
                BetaStatus::Other => {
                    let bv = simd.splat(beta);
                    for j in 0..NR {
                        let col = c.offset(j as isize * csc);
                        for i in 0..MR_REG {
                            let cv = simd.load_out(col.add(i * lanes));
                            // beta*C + alpha*AB, all in f32
                            let r = simd.mul_add(cv, bv, acc[j][i]);
                            let r = if !E::IS_IDENTITY {
                                epi.apply_reg(simd, r, row0 + i * lanes, col0 + j)
                            } else {
                                r
                            };
                            simd.store_out(col.add(i * lanes), r);
                        }
                    }
                }
            }
        } else {
            // General / partial path: drain f32 acc to scratch, then strided copy-back
            // with a per-element widen (read C) / narrow (write C)
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
                    // The narrow epilogue does the single narrowing itself (`apply` returns `N`);
                    // the identity path narrows the raw combine. Bit-identical to the fast vector
                    // path above under the same token (the edge-consistency contract)
                    *cp = if !E::IS_IDENTITY {
                        epi.apply(out, row0 + i, col0 + j)
                    } else {
                        N::narrow(out)
                    };
                }
            }
        }
    }
}

/// The mixed-precision GEMM family: `Lhs = Rhs = Out = N` (a [`NarrowFloat`], i.e.
/// `f16` or `bf16`), `Acc = f32`
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

    // `Out` is narrower than `Acc`, so the running sum must NOT round-trip through C
    // between depth panels: the driver uses `kc = k` and the whole contraction
    // accumulates in f32 registers, rounding to N once
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
        // Plain micropanel copy of the narrow elements; widening happens later, on
        // load in the microkernel
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

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
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
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut f32,
    ) where
        S: KernelSimd<N, N, f32, N>,
        E: Epilogue<Self>,
    {
        // `OUT_IS_ACC = false` => `kc = k` (a single depth panel), so `last_k` is structurally
        // true: the epilogue fires exactly once per element
        debug_assert!(
            last_k,
            "mixed families are single-panel (kc = k); last_k must be true"
        );
        let _ = last_k;
        unsafe {
            let mut acc: [[<S as SimdOps<f32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            mixed_accumulate::<N, S, N, MR_REG, NR>(
                simd, kc, a, a_cs, b, b_rs, b_cs, nr_eff, &mut acc,
            );
            mixed_epilogue::<Self, N, S, E, MR_REG, NR>(
                simd,
                alpha,
                beta,
                alpha_status,
                beta_status,
                &mut acc,
                c,
                rsc,
                csc,
                mr_eff,
                nr_eff,
                row0,
                col0,
                epi,
                scratch,
            );
        }
    }
}

/// The bf16 dot GEMM family: `Lhs = Rhs = Out = bf16`, `Acc = f32`, driven by AVX-512
/// `vdpbf16ps` (2 bf16 depth steps per instruction) instead of [`MixedGemm`]'s widen-FMA.
/// A sibling of `MixedGemm<bf16>`, not a branch in it: `pack_lhs`/`pack_rhs` take no ISA
/// parameter, so the differing k-pair interleave must key off the family
///
/// What changes versus `MixedGemm<bf16>`, both isolated here:
///
/// * **Pack layout** (`DEPTH_MULTIPLE = 2`): A and B are k-pair-interleaved: 2
///   consecutive depth steps contiguous per lane/column (a 32-bit `__m512bh` pair) to feed
///   one `vdpbf16ps`. Depth is padded to a multiple of 2 with bf16 `0` (`0*0 = 0`); both
///   operands are always packed (`FORCE_PACK_*`)
/// * **Inner loop**: [`crate::simd::KernelSimd::dot_accumulate`] replaces the widen-FMA
///   loop. `OUT_IS_ACC = false` keeps `kc = k`, so the whole contraction accumulates in
///   `f32` and rounds to bf16 once; alpha fold and narrow epilogue (`mixed_epilogue`)
///   are shared with `MixedGemm`
///
/// `vdpbf16ps`'s fused 2-term dot rounds differently from the widen path, so the result is
/// only tolerance-equal, not bitwise. It is still fully deterministic, and
/// serial/parallel/prepacked all share this kernel and layout, so they reproduce each
/// other bit-for-bit
#[derive(Clone, Copy)]
pub struct Bf16DotGemm(PhantomData<()>);

impl Bf16DotGemm {
    /// Depth steps folded per `vdpbf16ps`
    const Q: usize = 2;
}

impl KernelFamily for Bf16DotGemm {
    type Lhs = half::bf16;
    type Rhs = half::bf16;
    type Acc = f32;
    type Out = half::bf16;

    const OUT_IS_ACC: bool = false;
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;
    const DEPTH_MULTIPLE: usize = Self::Q;

    /// Pack the `mc x kc` LHS k-pair-interleaved (2 contiguous depth bf16 per row, a
    /// `__m512bh` pair), via the shared `pack_kgroup_panels`. Identity transform; rows
    /// past `mc` and depth past `kc` pad with bf16 `0` (`xform(0)`)
    #[inline]
    unsafe fn pack_lhs(
        dst: *mut half::bf16,
        src: *const half::bf16,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        // lead = rows (`rs`), depth = cols (`cs`)
        unsafe {
            pack_kgroup_panels::<half::bf16, { Self::Q }, _>(dst, src, rs, cs, mc, kc, mr, |v| v)
        }
    }

    /// Pack one `kc x nr` RHS panel k-pair-interleaved (2 contiguous depth bf16 per column,
    /// ready for an i32 broadcast), via the shared `pack_kgroup_panels`. Identity
    /// transform; columns past `nc` and depth past `kc` pad with bf16 `0`
    #[inline]
    unsafe fn pack_rhs(
        dst: *mut half::bf16,
        src: *const half::bf16,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        // lead = cols (`cs`), depth = rows (`rs`)
        unsafe {
            pack_kgroup_panels::<half::bf16, { Self::Q }, _>(dst, src, cs, rs, nc, kc, nr, |v| v)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: f32,
        beta: f32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const half::bf16,
        _a_cs: isize,
        b: *const half::bf16,
        _b_rs: isize,
        _b_cs: isize,
        c: *mut half::bf16,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut f32,
    ) where
        S: KernelSimd<half::bf16, half::bf16, f32, half::bf16>,
        E: Epilogue<Self>,
    {
        // `OUT_IS_ACC = false` => `kc = k` (a single depth panel), so `last_k` is structurally
        // true: the epilogue fires exactly once per element
        debug_assert!(
            last_k,
            "mixed families are single-panel (kc = k); last_k must be true"
        );
        let _ = last_k;
        unsafe {
            let mut acc: [[<S as SimdOps<f32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            simd.dot_accumulate::<MR_REG, NR>(kc, a, b, &mut acc);
            mixed_epilogue::<Self, half::bf16, S, E, MR_REG, NR>(
                simd,
                alpha,
                beta,
                alpha_status,
                beta_status,
                &mut acc,
                c,
                rsc,
                csc,
                mr_eff,
                nr_eff,
                row0,
                col0,
                epi,
                scratch,
            );
        }
    }
}

/// Seed the `f32` accumulator registers for one deep-k twin tile. On an accumulate slice
/// (`BetaStatus::One`) it loads the running partial from the `f32` scratch `C` so the following
/// [`mixed_accumulate`] / `dot_accumulate` **continues the ascending-k FMA chain** rather than
/// summing the slice from zero and adding `C` afterward: store/reload of an `f32` is exact, so the
/// multi-slice result is byte-for-byte the single-panel one. On the first slice
/// (`BetaStatus::Zero`) it returns zeroed accumulators (C is not read; it may be uninitialized).
/// The full-tile fast path (`mr_eff == mr`, `nr_eff == NR`, column-major C) vector-loads C; the
/// edge path builds a zero-padded seed in `scratch` and loads that (a partial vector load of C
/// could read past the live rows). `BetaStatus::Other` never reaches here (the twin is driven with
/// `beta == 0`)
///
/// # Safety
/// `c` valid for the live tile at `rsc`/`csc`; `scratch` holds at least `NR * mr` `f32`; run inside
/// `S`'s [`crate::simd::Simd::vectorize`]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn twin_seed<S, const MR_REG: usize, const NR: usize>(
    simd: S,
    beta_status: BetaStatus,
    c: *const f32,
    rsc: isize,
    csc: isize,
    mr_eff: usize,
    nr_eff: usize,
    scratch: *mut f32,
) -> [[<S as SimdOps<f32>>::Reg; MR_REG]; NR]
where
    S: SimdOps<f32>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mr = MR_REG * lanes;
        let mut acc: [[<S as SimdOps<f32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
        if beta_status == BetaStatus::One {
            if mr_eff == mr && nr_eff == NR && rsc == 1 {
                for j in 0..NR {
                    let col = c.offset(j as isize * csc);
                    for i in 0..MR_REG {
                        acc[j][i] = simd.loadu(col.add(i * lanes));
                    }
                }
            } else {
                // Zero-pad a full `NR x mr` seed tile, fill the live sub-tile from C, then load it
                // into the registers: dead lanes seed to 0 and, since each accumulator is
                // independent, never perturb a live output
                for x in 0..NR * mr {
                    *scratch.add(x) = 0.0;
                }
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        *scratch.add(j * mr + i) = *c.offset(i as isize * rsc + j as isize * csc);
                    }
                }
                for j in 0..NR {
                    for i in 0..MR_REG {
                        acc[j][i] = simd.loadu(scratch.add(j * mr + i * lanes));
                    }
                }
            }
        }
        acc
    }
}

/// Store one deep-k twin tile's `f32` accumulators back to the `f32` scratch `C` (the running
/// partial for the next slice, or the final sum consumed by the narrowing sweep). Mirror of
/// [`twin_seed`]: the full-tile fast path stores whole vectors column-major; the edge path drains
/// to `scratch` then copies only the live `mr_eff x nr_eff` sub-tile out with arbitrary strides
///
/// # Safety
/// As [`twin_seed`], with `c` writable
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn twin_store<S, const MR_REG: usize, const NR: usize>(
    simd: S,
    acc: &[[<S as SimdOps<f32>>::Reg; MR_REG]; NR],
    c: *mut f32,
    rsc: isize,
    csc: isize,
    mr_eff: usize,
    nr_eff: usize,
    scratch: *mut f32,
) where
    S: SimdOps<f32>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mr = MR_REG * lanes;
        if mr_eff == mr && nr_eff == NR && rsc == 1 {
            for j in 0..NR {
                let col = c.offset(j as isize * csc);
                for i in 0..MR_REG {
                    simd.storeu(col.add(i * lanes), acc[j][i]);
                }
            }
        } else {
            for j in 0..NR {
                for i in 0..MR_REG {
                    simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                }
            }
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    *c.offset(i as isize * rsc + j as isize * csc) = *scratch.add(j * mr + i);
                }
            }
        }
    }
}

/// The f32-output twin of [`MixedGemm`]: `Lhs = Rhs = N` (a [`NarrowFloat`]), `Acc = Out = f32`.
/// It exists solely for the deep-contraction route (`dispatch::mixed`): when a narrow GEMM's `k`
/// is large enough that the single-panel `MixedGemm` streams an L2-overflowing RHS micropanel, the
/// contraction is re-blocked here into an `f32` scratch. Because `Out == Acc` the driver's
/// **multi-slice** cache blocking applies unchanged (`OUT_IS_ACC = true`), keeping each slice's
/// panels L2-resident; the dispatch then narrows the scratch once. The pack layout and widen-FMA
/// accumulate are `MixedGemm`'s verbatim (same narrow panels), so the only difference is the
/// seed-from-C / `f32`-store epilogue that continues the accumulation across slices
pub struct MixedGemmF32<N>(PhantomData<N>);

impl<N> Clone for MixedGemmF32<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N> Copy for MixedGemmF32<N> {}

impl<N> KernelFamily for MixedGemmF32<N>
where
    N: NarrowFloat,
{
    type Lhs = N;
    type Rhs = N;
    type Acc = f32;
    type Out = f32;

    // Out == Acc == f32, so the whole multi-slice K accumulation round-trips through the f32
    // scratch exactly: `OUT_IS_ACC` stays at its `true` default and the driver blocks K at the
    // cache-model `kc` (the entire point of the twin)

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
        // Identical narrow micropanel copy to `MixedGemm`; widening happens on load
        unsafe { pack_panels(dst, src, rs, cs, mc, kc, mr) }
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
        unsafe { pack_panels(dst, src, cs, rs, nc, kc, nr) }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
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
        c: *mut f32,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut f32,
    ) where
        S: KernelSimd<N, N, f32, f32>,
        E: Epilogue<Self>,
    {
        // The twin is a deep-k internal: driven with alpha=1 / beta=0 and the plain (Identity)
        // epilogue, so it just accumulates raw f32 partials; the dispatch applies alpha/beta and
        // narrows afterward
        assert!(E::IS_IDENTITY, "deep-k twin does not fuse epilogues");
        debug_assert!(
            alpha_status == AlphaStatus::One,
            "deep-k twin runs alpha = 1"
        );
        debug_assert!(
            beta_status != BetaStatus::Other,
            "deep-k twin runs beta in {{0, 1}}"
        );
        let _ = (alpha, beta, alpha_status, row0, col0, last_k, epi);
        unsafe {
            let mut acc =
                twin_seed::<S, MR_REG, NR>(simd, beta_status, c, rsc, csc, mr_eff, nr_eff, scratch);
            mixed_accumulate::<N, S, f32, MR_REG, NR>(
                simd, kc, a, a_cs, b, b_rs, b_cs, nr_eff, &mut acc,
            );
            twin_store::<S, MR_REG, NR>(simd, &acc, c, rsc, csc, mr_eff, nr_eff, scratch);
        }
    }
}

/// The f32-output twin of [`Bf16DotGemm`]: `Lhs = Rhs = bf16`, `Acc = Out = f32`, `vdpbf16ps` dot.
/// The deep-contraction sibling of `Bf16DotGemm` exactly as [`MixedGemmF32`] is of [`MixedGemm`]:
/// same k-pair-interleaved pack (`DEPTH_MULTIPLE = 2`, both operands force-packed) and same
/// `dot_accumulate`, but `Out = f32` so the driver multi-slices. The driver rounds each interior
/// slice's `kc` up to `DEPTH_MULTIPLE` (see `driver::run_inner`), so a k-pair never straddles a
/// slice boundary and the multi-slice dot is bit-identical to the single panel; only the final
/// short tail is depth-padded, as in the single-panel case
#[derive(Clone, Copy)]
pub struct Bf16DotGemmF32(PhantomData<()>);

impl KernelFamily for Bf16DotGemmF32 {
    type Lhs = half::bf16;
    type Rhs = half::bf16;
    type Acc = f32;
    type Out = f32;

    // Out == Acc, so OUT_IS_ACC stays true and K multi-slices; the k-pair pack still needs both
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;
    const DEPTH_MULTIPLE: usize = 2;

    #[inline]
    unsafe fn pack_lhs(
        dst: *mut half::bf16,
        src: *const half::bf16,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        // Identical k-pair-interleaved pack to `Bf16DotGemm`
        unsafe { pack_kgroup_panels::<half::bf16, 2, _>(dst, src, rs, cs, mc, kc, mr, |v| v) }
    }

    #[inline]
    unsafe fn pack_rhs(
        dst: *mut half::bf16,
        src: *const half::bf16,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { pack_kgroup_panels::<half::bf16, 2, _>(dst, src, cs, rs, nc, kc, nr, |v| v) }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: f32,
        beta: f32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const half::bf16,
        _a_cs: isize,
        b: *const half::bf16,
        _b_rs: isize,
        _b_cs: isize,
        c: *mut f32,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut f32,
    ) where
        S: KernelSimd<half::bf16, half::bf16, f32, f32>,
        E: Epilogue<Self>,
    {
        assert!(E::IS_IDENTITY, "deep-k twin does not fuse epilogues");
        debug_assert!(
            alpha_status == AlphaStatus::One,
            "deep-k twin runs alpha = 1"
        );
        debug_assert!(
            beta_status != BetaStatus::Other,
            "deep-k twin runs beta in {{0, 1}}"
        );
        let _ = (alpha, beta, alpha_status, row0, col0, last_k, epi);
        unsafe {
            let mut acc =
                twin_seed::<S, MR_REG, NR>(simd, beta_status, c, rsc, csc, mr_eff, nr_eff, scratch);
            simd.dot_accumulate::<MR_REG, NR>(kc, a, b, &mut acc);
            twin_store::<S, MR_REG, NR>(simd, &acc, c, rsc, csc, mr_eff, nr_eff, scratch);
        }
    }
}
