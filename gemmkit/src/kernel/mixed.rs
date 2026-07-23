//! The mixed-precision GEMM families: narrow (`f16`/`bf16`) inputs, `f32` accumulator,
//! narrow or `f32` output
//!
//! [`MixedGemm`]/[`Bf16DotGemm`] are structurally close to [`super::float::FloatGemm`], but
//! every input reaches the kernel through the [`KernelSimd`] widen-load seam (A and B widen
//! to `f32` on load, products accumulate in `f32`) and the store narrows once, on the way
//! out. [`MixedGemmF32`]/[`Bf16DotGemmF32`] are the `Out = f32` twins that back the
//! deep-contraction dispatch route: same pack layout and accumulate, but `Out == Acc` lets
//! the driver's normal multi-slice K blocking apply

use core::marker::PhantomData;

use super::epilogue::Epilogue;
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::{pack_kgroup_panels, pack_panels};
use crate::scalar::NarrowFloat;
use crate::simd::{KernelSimd, SimdOps};

/// Widen-load A/B and fold `MR_REG x NR` products into `acc` over `kc` depth steps
///
/// Shared by [`MixedGemm::microkernel_epi`] and [`MixedGemmF32::microkernel_epi`], the
/// hot loop factored out of both. `acc` is not zeroed here: the narrow-output family always
/// hands it a fresh zero, but the `f32`-output twin may hand it a seed loaded from the
/// running partial `C`, so this function just continues whatever chain the caller started.
/// Generic over the `KernelSimd` output type `O` so both callers drive it: only
/// `load_lhs`/`splat_rhs` are used, which widen `N -> f32` identically regardless of `O`
/// (the `f32`-output impls forward to the narrow ones), so the 2 callers accumulate bit-for-bit
/// the same value
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
            // Full tile: NR is a const bound, so the column loop fully unrolls
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
            // Edge tile: bound the column loop to nr_eff so an unpacked B is never read
            // past its last real column; acc[nr_eff..] stays whatever the caller seeded
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

/// Fold `alpha` into `acc`, apply the fused epilogue `E`, and store to the narrow `Out = N`
///
/// Shared verbatim by [`MixedGemm`] (widen-and-FMA) and [`Bf16DotGemm`] (`vdpbf16ps` dot):
/// the 2 differ only in how `acc` is produced, so the identical `f32`-acc / narrow-`Out`
/// store logic lives here once. Both families are `OUT_IS_ACC = false`, so the driver runs a
/// single depth panel (`kc = k`) and `E` applies unconditionally here, once per element, with
/// no `last_k` check needed. With `E = Identity` every epilogue hook const-folds away, so the
/// call is byte-for-byte the pre-epilogue kernel
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

        // fold alpha into the f32 accumulators; skip the multiply entirely when alpha == 1
        if alpha_status == AlphaStatus::Other {
            let av = simd.splat(alpha);
            for j in 0..NR {
                for i in 0..MR_REG {
                    acc[j][i] = simd.mul(acc[j][i], av);
                }
            }
        }

        // A scalar-only epilogue has no apply_reg, so it must take the scratch route below
        // for every tile; Identity and any VECTOR epilogue can take the vector route
        // whenever the tile itself is full and column-major
        if (E::IS_IDENTITY || E::VECTOR) && mr_eff == mr && nr_eff == NR && rsc == 1 {
            // Vector widen-load / store of the full tile; apply_reg transforms the f32
            // register and store_out performs the single narrowing to N
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
                            // beta*C + alpha*AB, in f32
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
            // Edge or non-unit-stride tile: drain f32 acc to scratch, then copy back
            // element by element, widening the read of C and narrowing the write
            for j in 0..NR {
                for i in 0..MR_REG {
                    simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                }
            }
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    let v = *scratch.add(j * mr + i); // alpha*AB, f32
                    let cp = c.offset(i as isize * rsc + j as isize * csc);
                    let out = match beta_status {
                        BetaStatus::Zero => v,
                        BetaStatus::One => (*cp).widen() + v,
                        BetaStatus::Other => beta * (*cp).widen() + v,
                    };
                    // apply narrows to N itself; the identity branch narrows out directly
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

/// The widen-and-FMA mixed-precision GEMM family: `Lhs = Rhs = Out = N` (a [`NarrowFloat`],
/// i.e. `f16` or `bf16`), `Acc = f32`
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

    // Out (N) is narrower than Acc (f32): rounding a running partial through C at every
    // panel boundary would lose precision, so this forces the driver to kc = k and the
    // whole contraction stays in f32 registers, narrowing to N exactly once
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
        // Plain micropanel copy of the narrow elements; widening happens on load
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
        // OUT_IS_ACC = false means the driver always runs a single depth panel (kc = k),
        // so last_k is structurally always true here
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

/// The bf16 dot GEMM family: `Lhs = Rhs = Out = bf16`, `Acc = f32`, accumulated via
/// AVX-512 BF16 `vdpbf16ps` (2 bf16 depth steps folded per instruction) instead of
/// [`MixedGemm`]'s widen-and-FMA
///
/// A sibling of `MixedGemm<bf16>`, not a branch inside it: `pack_lhs`/`pack_rhs` take no
/// ISA parameter, so the differing k-pair interleave has to be a distinct family. 2 things
/// change versus `MixedGemm<bf16>`:
///
/// * Pack layout (`DEPTH_MULTIPLE = 2`): A and B are k-pair-interleaved, 2 consecutive
///   depth steps stored contiguous per row/column (a `__m512bh` pair) so one `vdpbf16ps`
///   reads a whole pair. Depth pads to a multiple of 2 with bf16 `0`; both operands are
///   always packed (`FORCE_PACK_*`) since the interleave needs the packed layout
/// * Inner loop: `dot_accumulate` replaces the widen-FMA loop. `OUT_IS_ACC = false` still
///   keeps `kc = k`, so the whole contraction accumulates in `f32` and narrows to bf16
///   once; alpha fold and the narrow epilogue reuse `mixed_epilogue` unchanged
///
/// `vdpbf16ps`'s fused 2-term dot rounds differently from the widen-FMA path, so the result
/// is only tolerance-equal to `MixedGemm<bf16>`, not bitwise. It is still fully
/// deterministic: serial, parallel, and prepacked runs all drive the same kernel and pack
/// layout, so they reproduce each other exactly
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

    /// Pack the `mc x kc` LHS k-pair-interleaved: 2 contiguous depth bf16 values per row (a
    /// `__m512bh` pair), values left unchanged. A padded row (past `mc` or `kc`) packs as bf16 `0`
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
        // lead = rows (rs), depth = cols (cs)
        unsafe {
            pack_kgroup_panels::<half::bf16, { Self::Q }, _>(dst, src, rs, cs, mc, kc, mr, |v| v)
        }
    }

    /// Pack one `kc x nr` RHS panel k-pair-interleaved: 2 contiguous depth bf16 values per
    /// column, so each pair reads back as one `i32` for a single broadcast. Values are left
    /// unchanged; a padded column or depth step packs as bf16 `0`
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
        // lead = cols (cs), depth = rows (rs)
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
        // OUT_IS_ACC = false means the driver always runs a single depth panel (kc = k),
        // so last_k is structurally always true here
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

/// Seed the `f32` accumulator registers for one deep-k twin tile
///
/// On an accumulate slice (`BetaStatus::One`) this loads the running partial from the `f32`
/// scratch `C`, so the caller's [`mixed_accumulate`]/`dot_accumulate` continues the
/// ascending-k chain instead of summing the slice from zero and adding `C` afterward:
/// storing and reloading an `f32` is exact, so the multi-slice result matches the
/// single-panel one bit-for-bit. On the 1st slice (`BetaStatus::Zero`) it returns zeroed
/// accumulators without reading `C` (which may still be uninitialized there).
/// `BetaStatus::Other` never reaches this function: the twin always runs with `beta` in `{0, 1}`
///
/// The full-tile fast path vector-loads `C` directly; the edge path builds a zero-padded
/// seed in `scratch` and loads that instead, since a partial vector load of `C` could read
/// past the live rows
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
                // Zero-pad a full NR x mr seed tile, fill in the live sub-tile from C, then load
                // it: the dead lanes seed to 0 and, since each accumulator is independent, never
                // perturb a live output
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

/// Store one deep-k twin tile's `f32` accumulators back to the `f32` scratch `C`: the running
/// partial for the next slice, or the final sum the narrowing sweep consumes
///
/// Mirrors [`twin_seed`]: the full-tile fast path stores whole vectors column-major; the
/// edge path drains to `scratch` and copies out only the live `mr_eff x nr_eff` sub-tile
/// under the tile's real strides
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

/// The `f32`-output twin of [`MixedGemm`]: `Lhs = Rhs = N` (a [`NarrowFloat`]), `Acc = Out = f32`
///
/// Exists solely for the deep-contraction dispatch route: `MixedGemm` runs a single depth
/// panel (`kc = k`), which at large `k` streams an L2-overflowing RHS micropanel from
/// L3/DRAM. This twin re-blocks the same contraction into an `f32` scratch instead; since
/// `Out == Acc` here, the driver's ordinary multi-slice K blocking applies (`OUT_IS_ACC`
/// stays at its `true` default), keeping every slice's panels L2-resident, and the dispatch
/// narrows the scratch to `N` once at the end. Pack layout and widen-FMA accumulate are
/// `MixedGemm`'s verbatim; the only difference is the epilogue, which seeds from `C` and
/// stores raw `f32` so the accumulation continues correctly across slices
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

    // Out == Acc == f32 here, so OUT_IS_ACC is left at its true default: the driver blocks K
    // at the cache-model kc and round-trips the partial through the f32 scratch exactly,
    // which is the entire point of this twin

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
        // Same narrow micropanel copy as MixedGemm; widening happens on load
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
        // A deep-k internal call, never a user-facing fused GEMM: alpha=1, beta in {0, 1},
        // and always the Identity epilogue, so this just accumulates raw f32 partials; the
        // dispatch layer applies the real alpha/beta and narrows once outside this function
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

/// The `f32`-output twin of [`Bf16DotGemm`]: `Lhs = Rhs = bf16`, `Acc = Out = f32`, `vdpbf16ps` dot
///
/// The deep-contraction sibling of `Bf16DotGemm`, exactly as [`MixedGemmF32`] is of
/// [`MixedGemm`]: same k-pair-interleaved pack (`DEPTH_MULTIPLE = 2`, both operands
/// force-packed) and the same `dot_accumulate`, but `Out = f32` so the driver multi-slices.
/// The driver rounds every interior slice's `kc` up to `DEPTH_MULTIPLE`, so a k-pair never
/// straddles a slice boundary and the multi-slice dot matches the single-panel one
/// bit-for-bit; only the final short tail is depth-padded, same as the single-panel case
#[derive(Clone, Copy)]
pub struct Bf16DotGemmF32(PhantomData<()>);

impl KernelFamily for Bf16DotGemmF32 {
    type Lhs = half::bf16;
    type Rhs = half::bf16;
    type Acc = f32;
    type Out = f32;

    // Out == Acc == f32, so OUT_IS_ACC stays at its true default and K multi-slices; the
    // k-pair pack still needs both operands force-packed regardless
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
        // Same k-pair-interleaved pack as Bf16DotGemm
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
