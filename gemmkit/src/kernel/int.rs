//! The integer GEMM families: `i8` inputs, `i32` accumulator, exact or requantized output
//!
//! [`IntGemm`] sign-extends `i8` A/B to `i32` on load (the [`KernelSimd`] widen seam) and
//! accumulates the products in `i32`. Since `Out == Acc == i32` the sum is exact and needs
//! no narrowing, so the driver blocks K the normal way (`OUT_IS_ACC` stays at its `true`
//! default) and lets `beta` round-trip a partial through `C` between panels. Overflow wraps,
//! the conventional integer-GEMM contract
//!
//! [`IntGemmVnni`] computes the identical `i32` sum via the denser `vpdpbusd` instruction,
//! which needs its own k-quad-interleaved pack layout and a `+128` bias to turn the signed
//! LHS into the unsigned operand `vpdpbusd` requires. `IntGemmQ`/`IntGemmVnniQ` are the
//! `i8 -> {i8, u8}` requantizing counterparts, fusing a quantize `Epilogue` into the store

use core::marker::PhantomData;

#[cfg(feature = "epilogue")]
use super::epilogue::{Epilogue, QuantOut};
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::{pack_kgroup_panels, pack_panels};
use crate::scalar::Scalar;
use crate::simd::{KernelSimd, SimdOps, VNNI_A_BIAS};

/// Fold `alpha` into `acc`, then store `C <- combine(alpha*A*B, beta*C)`, exactly, in `i32`
///
/// Shared verbatim by [`IntGemm`] and [`IntGemmVnni`]: the 2 families differ only in how
/// `acc` (already the finished sum `sum_k(A*B)`) is produced, so the identical `Out == Acc
/// == i32` store logic lives here once instead of twice
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn int32_epilogue<S, const MR_REG: usize, const NR: usize>(
    simd: S,
    alpha: i32,
    beta: i32,
    alpha_status: AlphaStatus,
    beta_status: BetaStatus,
    acc: &mut [[<S as SimdOps<i32>>::Reg; MR_REG]; NR],
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

        // fold alpha into acc; skip the multiply entirely when alpha == 1
        if alpha_status == AlphaStatus::Other {
            let av = simd.splat(alpha);
            for j in 0..NR {
                for i in 0..MR_REG {
                    acc[j][i] = simd.mul(acc[j][i], av);
                }
            }
        }

        // Out == Acc == i32 here, so store_out/load_out are plain vector load/store
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
                            // beta*C + alpha*AB, wrapping i32
                            simd.store_out(col.add(i * lanes), simd.mul_add(cv, bv, acc[j][i]));
                        }
                    }
                }
            }
        } else {
            // Edge or non-unit-stride tile: drain to i32 scratch, then copy back under
            // the tile's real strides
            for j in 0..NR {
                for i in 0..MR_REG {
                    simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
                }
            }
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    let v = *scratch.add(j * mr + i); // alpha*AB, i32
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

/// Widen-and-multiply accumulation of one `MR_REG x NR` `i32` microtile: sign-extend `i8`
/// A/B to `i32` on load and fold in ascending `k`
///
/// Factored out of [`IntGemm::microkernel`] so the requantizing family `IntGemmQ` shares the
/// same accumulation. The 2 callers differ only in the output type `O` (`i32` for `IntGemm`,
/// `i8`/`u8` for the requantizer), and the requant blankets forward `load_lhs`/`splat_rhs`
/// straight to the `<i8,i8,i32,i32>` impl, so the accumulation is identical whichever `O` is
/// selected. `load_lhs`/`splat_rhs` are called fully qualified because a requant token
/// implements `KernelSimd` for more than one `O`, which would make an unqualified method
/// call ambiguous
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn i32_accumulate<S, O, const MR_REG: usize, const NR: usize>(
    simd: S,
    kc: usize,
    a: *const i8,
    a_cs: isize,
    b: *const i8,
    b_rs: isize,
    b_cs: isize,
    nr_eff: usize,
    acc: &mut [[<S as SimdOps<i32>>::Reg; MR_REG]; NR],
) where
    O: Scalar,
    S: KernelSimd<i8, i8, i32, O>,
{
    unsafe {
        let lanes = <S as SimdOps<i32>>::LANES;
        if nr_eff == NR {
            for p in 0..kc {
                let pa = a.offset(p as isize * a_cs);
                let a_regs: [<S as SimdOps<i32>>::Reg; MR_REG] = core::array::from_fn(|i| {
                    <S as KernelSimd<i8, i8, i32, O>>::load_lhs(simd, pa.add(i * lanes))
                });
                let pb = b.offset(p as isize * b_rs);
                for j in 0..NR {
                    let bj = <S as KernelSimd<i8, i8, i32, O>>::splat_rhs(
                        simd,
                        *pb.offset(j as isize * b_cs),
                    );
                    for i in 0..MR_REG {
                        acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                    }
                }
            }
        } else {
            for p in 0..kc {
                let pa = a.offset(p as isize * a_cs);
                let a_regs: [<S as SimdOps<i32>>::Reg; MR_REG] = core::array::from_fn(|i| {
                    <S as KernelSimd<i8, i8, i32, O>>::load_lhs(simd, pa.add(i * lanes))
                });
                let pb = b.offset(p as isize * b_rs);
                for j in 0..nr_eff {
                    let bj = <S as KernelSimd<i8, i8, i32, O>>::splat_rhs(
                        simd,
                        *pb.offset(j as isize * b_cs),
                    );
                    for i in 0..MR_REG {
                        acc[j][i] = simd.mul_add(a_regs[i], bj, acc[j][i]);
                    }
                }
            }
        }
    }
}

/// Requantize a finished `i32` microtile to the output byte `O` (`i8` or `u8`): drain `acc`
/// to `i32` scratch, then map each live element through the epilogue `E` into `C`
///
/// Shared by both requantizing families ([`IntGemmQ`]/[`IntGemmVnniQ`]) and both output
/// bytes. Both requantizing families are `OUT_IS_ACC = false`, so the driver runs a single
/// depth panel (`kc = k`): the map below always applies, with no `last_k` gate to check
///
/// # Safety
/// As [`KernelFamily::microkernel_epi`]; `scratch` holds at least [`super::SCRATCH_LEN`]
/// `i32` elements
#[cfg(feature = "epilogue")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
unsafe fn requant_scratch_epilogue<F, S, E, O, const MR_REG: usize, const NR: usize>(
    simd: S,
    acc: &[[<S as SimdOps<i32>>::Reg; MR_REG]; NR],
    c: *mut O,
    rsc: isize,
    csc: isize,
    mr_eff: usize,
    nr_eff: usize,
    row0: usize,
    col0: usize,
    epi: &E,
    scratch: *mut i32,
) where
    O: QuantOut,
    F: KernelFamily<Acc = i32, Out = O>,
    // Bounded by the exact KernelSimd this needs (not just SimdOps<i32>) so
    // epi.apply_store::<S> below type-checks against it
    S: KernelSimd<F::Lhs, F::Rhs, i32, O>,
    E: Epilogue<F>,
{
    unsafe {
        let lanes = <S as SimdOps<i32>>::LANES;
        let mr = MR_REG * lanes;
        // Vectorized drain of the register-resident acc to contiguous i32 scratch
        for j in 0..NR {
            for i in 0..MR_REG {
                simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
            }
        }

        // A full lane-run takes the vector apply_store when both the epilogue and the token
        // support it and C is unit row stride; the row tail always falls to scalar apply, and
        // an unqualifying tile (strided C, no vector support) takes the scalar loop wholesale
        if E::VECTOR_STORE && <S as KernelSimd<F::Lhs, F::Rhs, i32, O>>::REQUANT_VECTOR && rsc == 1
        {
            for j in 0..nr_eff {
                let src_col = scratch.add(j * mr);
                let dst_col = c.offset(j as isize * csc); // rsc == 1: rows are unit-stride here
                let mut i = 0;
                while i + lanes <= mr_eff {
                    epi.apply_store(simd, src_col.add(i), dst_col.add(i), row0 + i, col0 + j);
                    i += lanes;
                }
                for i in i..mr_eff {
                    *dst_col.add(i) = epi.apply(*src_col.add(i), row0 + i, col0 + j);
                }
            }
        } else {
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    let v = *scratch.add(j * mr + i);
                    let cp = c.offset(i as isize * rsc + j as isize * csc);
                    *cp = epi.apply(v, row0 + i, col0 + j);
                }
            }
        }
    }
}

/// The widen-and-multiply integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`
#[derive(Clone, Copy)]
pub struct IntGemm(PhantomData<()>);

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
        // Plain micropanel copy of the i8 bytes; sign-extension to i32 happens on load
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
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            i32_accumulate::<S, i32, MR_REG, NR>(
                simd, kc, a, a_cs, b, b_rs, b_cs, nr_eff, &mut acc,
            );

            int32_epilogue::<S, MR_REG, NR>(
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
                scratch,
            );
        }
    }
}

/// The LHS pack transform for the VNNI families ([`IntGemmVnni`] and, by delegation,
/// `IntGemmVnniQ`): offset each byte by [`VNNI_A_BIAS`] into the unsigned domain `vpdpbusd`
/// reads. `vnni_a_xform(0) == 128`, so a padded row contributes a constant the per-column
/// bias correction cancels out exactly
#[inline(always)]
pub(crate) fn vnni_a_xform(v: i8) -> i8 {
    ((v as i32 + VNNI_A_BIAS) as u8) as i8
}

/// The VNNI integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`, accumulated via
/// `vpdpbusd` (4 depth steps folded per instruction) instead of [`IntGemm`]'s widen-and-multiply
///
/// 2 things differ from `IntGemm`, both isolated to this family:
///
/// * Pack layout (`DEPTH_MULTIPLE = 4`): A and B are k-quad-interleaved, 4 consecutive depth
///   steps stored contiguous per row/column so one `vpdpbusd` reads a whole group. Depth pads
///   to a multiple of 4 (A pads with `128`, B with `0`); both operands are always packed
///   (`FORCE_PACK_*`) since the interleave cannot be read from the unpacked layout in place
/// * Signedness: `vpdpbusd` needs an unsigned x signed operand pair, so the pack offsets A by
///   `+128` and `dot_accumulate` subtracts the resulting per-column bias to recover the true
///   `sum_k(A*B)`
///
/// `i32` addition is associative under wrapping, so grouping the sum by 4 and correcting the
/// bias afterward reproduces the ascending-`k` widen sum bit-for-bit: `IntGemmVnni` and
/// `IntGemm` always agree exactly. Alpha fold and the `i32` epilogue are the shared
/// `int32_epilogue` helper above
#[derive(Clone, Copy)]
pub struct IntGemmVnni(PhantomData<()>);

impl IntGemmVnni {
    /// Depth steps folded per `vpdpbusd`
    const Q: usize = 4;
}

impl KernelFamily for IntGemmVnni {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = i32;

    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;
    const DEPTH_MULTIPLE: usize = Self::Q;

    /// Pack the `mc x kc` LHS k-quad-interleaved: 4 contiguous depth bytes per row, ready
    /// for `vpdpbusd`, each offset `+128` by `vnni_a_xform`. A padded row (past `mc` or
    /// `kc`) packs as `xform(0) == 128`, which the bias correction cancels
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
        // lead = rows (rs), depth = cols (cs)
        unsafe {
            pack_kgroup_panels::<i8, { Self::Q }, _>(dst, src, rs, cs, mc, kc, mr, vnni_a_xform)
        }
    }

    /// Pack one `kc x nr` RHS panel k-quad-interleaved: 4 contiguous depth bytes per
    /// column, so each quad reads back as one `i32` for a single broadcast. Values stay
    /// signed (identity transform); a padded column or depth step packs as `0`, so it
    /// never enters the column sum
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
        // lead = cols (cs), depth = rows (rs)
        unsafe { pack_kgroup_panels::<i8, { Self::Q }, _>(dst, src, cs, rs, nc, kc, nr, |v| v) }
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
        _a_cs: isize,
        b: *const i8,
        _b_rs: isize,
        _b_cs: isize,
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
            // acc is filled for the full NR (FORCE_PACK_RHS zero-pads tail columns, so
            // they contribute 0 to the dot); int32_epilogue below copies only the live
            // mr_eff x nr_eff sub-tile out
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            // Fully qualified: a VNNI token implements dot_accumulate for both this
            // concrete <i8,i8,i32,i32> and the requant blanket, so a plain call is ambiguous
            <S as KernelSimd<i8, i8, i32, i32>>::dot_accumulate::<MR_REG, NR>(
                simd, kc, a, b, &mut acc,
            );

            int32_epilogue::<S, MR_REG, NR>(
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
                scratch,
            );
        }
    }
}

/// The `i8 -> O` requantizing integer family (widen-and-multiply), where the output byte
/// `O` is `i8` (`[-128, 127]`, the default) or `u8` (`[0, 255]`, ONNX-QLinearMatMul):
/// accumulate in `i32` exactly like [`IntGemm`], then apply an `i32 -> O` requantize
/// [`Epilogue`] (per-tensor scale, zero-point, optional integer bias) as the tile is
/// stored. The default type parameter means a bare `IntGemmQ` is `IntGemmQ<i8>`
///
/// `OUT_IS_ACC = false` forces the driver to a single depth panel (`kc = k`), so the
/// requantize fires exactly once per output element with no `last_k` gate to check, and an
/// `i32` partial never has to round-trip through `O`. `alpha` is folded into `scale` and
/// `beta` is disallowed (there is no well-defined way to accumulate into a quantized `C`);
/// the call site below enforces both (`AlphaStatus::One`, `BetaStatus::Zero`)
#[cfg(feature = "epilogue")]
#[derive(Clone, Copy)]
pub struct IntGemmQ<O = i8>(PhantomData<O>);

#[cfg(feature = "epilogue")]
impl<O: QuantOut> KernelFamily for IntGemmQ<O> {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = O;

    const OUT_IS_ACC: bool = false; // forces the driver to kc = k: fires the requantize once

    /// Same plain micropanel copy as [`IntGemm::pack_lhs`]; delegate to it
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
        unsafe { <IntGemm as KernelFamily>::pack_lhs(dst, src, rs, cs, mc, kc, mr) }
    }

    /// Same as [`IntGemm::pack_rhs`]; delegate to it
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
        unsafe { <IntGemm as KernelFamily>::pack_rhs(dst, src, rs, cs, kc, nc, nr) }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        _alpha: i32,
        _beta: i32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const i8,
        a_cs: isize,
        b: *const i8,
        b_rs: isize,
        b_cs: isize,
        c: *mut O,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        _last_k: bool,
        epi: &E,
        scratch: *mut i32,
    ) where
        S: KernelSimd<i8, i8, i32, O>,
        E: Epilogue<Self>,
    {
        debug_assert!(matches!(beta_status, BetaStatus::Zero));
        debug_assert!(matches!(alpha_status, AlphaStatus::One));
        unsafe {
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            i32_accumulate::<S, O, MR_REG, NR>(simd, kc, a, a_cs, b, b_rs, b_cs, nr_eff, &mut acc);
            requant_scratch_epilogue::<Self, S, E, O, MR_REG, NR>(
                simd, &acc, c, rsc, csc, mr_eff, nr_eff, row0, col0, epi, scratch,
            );
        }
    }
}

/// The VNNI requantizing family: the same `i8 -> O` requantize as [`IntGemmQ`], but the
/// underlying `i32` sum comes from `vpdpbusd` instead of widen-and-multiply, mirroring how
/// [`IntGemmVnni`] relates to [`IntGemm`] (`FORCE_PACK_* = true`, `DEPTH_MULTIPLE = 4`, the
/// k-quad-interleaved `+128`/identity pack). Since the grouped VNNI sum is bit-equal to the
/// widen sum, `IntGemmVnniQ<O>` and `IntGemmQ<O>` requantize to identical output. `O`
/// defaults to `i8`, so a bare `IntGemmVnniQ` is `IntGemmVnniQ<i8>`
#[cfg(feature = "epilogue")]
#[derive(Clone, Copy)]
pub struct IntGemmVnniQ<O = i8>(PhantomData<O>);

#[cfg(feature = "epilogue")]
impl<O> IntGemmVnniQ<O> {
    /// Depth steps folded per `vpdpbusd`
    const Q: usize = 4;
}

#[cfg(feature = "epilogue")]
impl<O: QuantOut> KernelFamily for IntGemmVnniQ<O> {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = O;

    const OUT_IS_ACC: bool = false;
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;
    const DEPTH_MULTIPLE: usize = Self::Q;

    /// Same k-quad-interleaved, `+128`-biased pack as [`IntGemmVnni::pack_lhs`]; delegate to it
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
        unsafe { <IntGemmVnni as KernelFamily>::pack_lhs(dst, src, rs, cs, mc, kc, mr) }
    }

    /// Same k-quad-interleaved, signed-value pack as [`IntGemmVnni::pack_rhs`]; delegate to it
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
        unsafe { <IntGemmVnni as KernelFamily>::pack_rhs(dst, src, rs, cs, kc, nc, nr) }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        _alpha: i32,
        _beta: i32,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const i8,
        _a_cs: isize,
        b: *const i8,
        _b_rs: isize,
        _b_cs: isize,
        c: *mut O,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        _last_k: bool,
        epi: &E,
        scratch: *mut i32,
    ) where
        S: KernelSimd<i8, i8, i32, O>,
        E: Epilogue<Self>,
    {
        debug_assert!(matches!(beta_status, BetaStatus::Zero));
        debug_assert!(matches!(alpha_status, AlphaStatus::One));
        unsafe {
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            // Fully qualified: this O-typed dot_accumulate is the requant blanket, which
            // forwards to the concrete <i8,i8,i32,i32> impl a plain call cannot disambiguate
            <S as KernelSimd<i8, i8, i32, O>>::dot_accumulate::<MR_REG, NR>(
                simd, kc, a, b, &mut acc,
            );
            requant_scratch_epilogue::<Self, S, E, O, MR_REG, NR>(
                simd, &acc, c, rsc, csc, mr_eff, nr_eff, row0, col0, epi, scratch,
            );
        }
    }
}
