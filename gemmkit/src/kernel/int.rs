//! The integer GEMM family: `i8` inputs, `i32` accumulator and output.
//!
//! Like [`super::mixed::MixedGemm`], inputs reach the kernel through the
//! [`KernelSimd`] widen seam (`i8 -> i32` sign-extend on load) and accumulate in
//! the wider type. Since `Out == Acc == i32` the result is exact, needs no
//! narrowing, and the driver blocks K normally (`OUT_IS_ACC` stays `true`).
//! Arithmetic wraps on overflow, the conventional integer-GEMM semantics.
//!
//! This is the widen-and-multiply kernel. The denser VNNI `vpdpbusd` dot kernel is
//! [`IntGemmVnni`] below: it carries its own K-quad-interleaved pack layout and the
//! `u8 × i8` signedness (`+128`) handling that `vpdpbusd` needs.

use core::marker::PhantomData;

use super::epilogue::{Epilogue, QuantOut};
use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::{pack_kgroup_panels, pack_panels};
use crate::scalar::Scalar;
use crate::simd::{KernelSimd, SimdOps};

/// Fold `alpha` into the register-resident `acc` and apply the `i32` epilogue
/// `C <- combine(alpha·A·B, beta·C)`. Shared verbatim by both integer families
/// ([`IntGemm`] and [`IntGemmVnni`]) — they differ only in how `acc` is *produced*
/// (widen-and-multiply vs. `vpdpbusd` dot), so the exact, `Out == Acc == i32`
/// epilogue lives here once. `acc` is already the (possibly bias-corrected) running
/// sum `Σ_k A·B`.
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`].
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

/// Widen-and-multiply accumulation of one `MR_REG × NR` `i32` microtile: sign-extend the
/// `i8` A/B inputs to `i32` on load and fold in ascending `k`. Factored out of
/// [`IntGemm::microkernel`] so the requantizing family [`IntGemmQ`] shares the exact same
/// accumulation. Callers differ only in the output type `O`, which selects the (bit-
/// identical) widen load: the `<i8,i8,i32,i8>` / `<i8,i8,i32,u8>` blankets delegate their
/// `load_lhs`/`splat_rhs` straight to the `<i8,i8,i32,i32>` impl, so `O = i8`, `O = u8`, and
/// `O = i32` accumulate identically. The `load_lhs`/`splat_rhs` calls are fully qualified
/// because a requant token carries both `KernelSimd` impls, which would otherwise make the
/// plain method call ambiguous.
///
/// # Safety
/// As [`KernelFamily::microkernel`]; run inside `S`'s [`crate::simd::Simd::vectorize`].
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

/// Requantize a computed `i32` microtile to the output byte `O` (`i8` or `u8`): drain `acc`
/// to `i32` scratch (vectorized), then map each live element through the scalar epilogue `E`,
/// which writes the `Out = O` directly (strided). Shared by both requantizing families and both
/// output domains. The epilogue always applies — these families are `OUT_IS_ACC = false`, so
/// `kc = k` and this is the single final panel, so there is no `last_k` gate.
///
/// # Safety
/// As [`KernelFamily::microkernel_epi`]; `scratch` holds at least [`super::SCRATCH_LEN`]
/// `i32` elements.
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
    // `KernelSimd<F::Lhs, F::Rhs, i32, O>` (not just `SimdOps<i32>`) so `epi.apply_store::<S>`
    // type-checks — its bound is exactly this. Both call sites carry `KernelSimd<i8, i8, i32, O>`.
    S: KernelSimd<F::Lhs, F::Rhs, i32, O>,
    E: Epilogue<F>,
{
    unsafe {
        let lanes = <S as SimdOps<i32>>::LANES;
        let mr = MR_REG * lanes;
        // Vectorized drain of the register-resident `acc` to contiguous `i32` scratch (unchanged).
        for j in 0..NR {
            for i in 0..MR_REG {
                simd.storeu(scratch.add(j * mr + i * lanes), acc[j][i]);
            }
        }

        // Map `i32` scratch -> `O` C. When the epilogue and token are both requant-vector-capable
        // and `C` has unit row stride, a full lane-run takes the vector store `apply_store`; the
        // sub-lane row tail falls to the scalar `apply`. The two agree bit-for-bit (the
        // `requant_store` equivalence contract), so one output mixes them — as it also mixes with
        // the strided-`C` / `k == 0` degenerate / VNNI small-parallel paths, all of which run the
        // scalar `apply` below. Non-vector tokens (scalar / NEON / wasm) and any strided `C` take
        // the plain scalar loop verbatim.
        if E::VECTOR_STORE && <S as KernelSimd<F::Lhs, F::Rhs, i32, O>>::REQUANT_VECTOR && rsc == 1
        {
            for j in 0..nr_eff {
                let src_col = scratch.add(j * mr);
                let dst_col = c.offset(j as isize * csc); // rsc == 1: rows are unit-stride at `dst_col`
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

/// The integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`.
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
            // --- accumulate in i32: sign-extend A/B to i32, multiply-add ---
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            i32_accumulate::<S, i32, MR_REG, NR>(
                simd, kc, a, a_cs, b, b_rs, b_cs, nr_eff, &mut acc,
            );

            // alpha fold + exact i32 epilogue (shared with `IntGemmVnni`).
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

/// The `+128` unsigned bias the VNNI families offset the LHS by: `vpdpbusd` computes
/// *unsigned* A × *signed* B, so A is packed as `A + 128` and the
/// [`crate::simd::KernelSimd::dot_accumulate`] override subtracts `VNNI_A_BIAS · Σ_k B`
/// to recover the true signed product. The pack transform ([`vnni_a_xform`]) and that
/// correction MUST use the same constant, so it lives here once.
pub(crate) const VNNI_A_BIAS: i32 = 128;

/// The LHS pack transform shared by the VNNI families ([`IntGemmVnni`], and — via
/// delegation — [`IntGemmVnniQ`]): offset each `i8` by [`VNNI_A_BIAS`] into the unsigned
/// domain `vpdpbusd` reads. `vnni_a_xform(0) = 128`, so the LHS pad contributes a constant
/// the colsum correction cancels.
#[inline(always)]
pub(crate) fn vnni_a_xform(v: i8) -> i8 {
    ((v as i32 + VNNI_A_BIAS) as u8) as i8
}

/// The VNNI integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`, driven by
/// `vpdpbusd` (4 depth steps × 16 lanes per instruction) instead of widen-and-multiply.
///
/// Two things change versus [`IntGemm`], both isolated to this family:
///
/// * **Pack layout** (`DEPTH_MULTIPLE = 4`): A and B are *k-quad-interleaved* — four
///   consecutive depth steps contiguous per lane/column to feed one `vpdpbusd`. Depth is
///   padded to a multiple of 4 (A pad = `128`, B pad = `0`); both operands are always
///   packed (`FORCE_PACK_*`), the interleave can't be read in place.
/// * **Signedness** (`u8 × i8`): `vpdpbusd` does *unsigned* A × *signed* B, but GEMM is
///   signed × signed. The pack offsets A by `+128`; the
///   [`crate::simd::KernelSimd::dot_accumulate`] override subtracts the per-column bias
///   `128·Σ_k B[k][j]` so the accumulator holds the true `Σ_k A·B`.
///
/// Accumulation stays exact: i32 add is associative under wrapping, so the 4-way grouping
/// plus the bias correction equals the ascending-`k` widen sum **bit-for-bit** — VNNI and
/// [`IntGemm`] produce identical output. Alpha fold and `i32` epilogue are shared.
#[derive(Clone, Copy)]
pub struct IntGemmVnni(PhantomData<()>);

impl IntGemmVnni {
    /// Depth steps folded per `vpdpbusd`.
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

    /// Pack the `mc × kc` LHS k-quad-interleaved (4 contiguous depth bytes per row, ready
    /// for `vpdpbusd`) via the shared `pack_kgroup_panels`. The transform offsets every
    /// byte `+128`; the pad (rows past `mc` / depth past `kc`) is `xform(0) = 128`,
    /// contributing nothing after the bias correction.
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
        // lead = rows (`rs`), depth = cols (`cs`).
        unsafe {
            pack_kgroup_panels::<i8, { Self::Q }, _>(dst, src, rs, cs, mc, kc, mr, vnni_a_xform)
        }
    }

    /// Pack one `kc × nr` RHS panel k-quad-interleaved (4 contiguous depth bytes per column,
    /// ready for an i32 broadcast), via the shared `pack_kgroup_panels`. Values stay
    /// *signed* (identity transform); columns past `nc` and depth past `kc` pad with `0`
    /// (excluded from the column sum).
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
        // lead = cols (`cs`), depth = rows (`rs`).
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
            // Dot accumulation (true signed `Σ_k A·B`, bias-corrected internally). The
            // full `NR` is always processed: `FORCE_PACK_RHS` zero-pads tail columns
            // (contributing 0, column sum 0); the epilogue copies only the live sub-tile.
            let mut acc: [[<S as SimdOps<i32>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
            // Fully qualified: a VNNI token carries both `<i8,i8,i32,i32>` (concrete) and
            // `<i8,i8,i32,i8>` (blanket) `dot_accumulate`, which the plain call cannot pick.
            <S as KernelSimd<i8, i8, i32, i32>>::dot_accumulate::<MR_REG, NR>(
                simd, kc, a, b, &mut acc,
            );

            // alpha fold + exact i32 epilogue (shared with `IntGemm`).
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

/// The `i8 -> O` **requantizing** integer family (widen-and-multiply), where the output byte
/// `O` is `i8` (`[-128, 127]`, the default) or `u8` (`[0, 255]`, ONNX-QLinearMatMul): accumulate
/// in `i32` exactly like [`IntGemm`], then apply an `i32 -> O` requantize [`Epilogue`] (per-tensor
/// scale + zero-point + optional integer bias) as the tile is stored. The default type parameter
/// keeps every bare `IntGemmQ` mention meaning `IntGemmQ<i8>`.
///
/// `OUT_IS_ACC = false`, so the driver forces `kc = k` — a single depth panel. That makes
/// the requantize fire **exactly once** per output element *structurally* (no `last_k` gate
/// needed), and an `i32` partial never has to round-trip through the `O` output. `alpha` is
/// folded into `scale` and `beta` is disallowed (accumulating into a quantized C is
/// ill-defined), which the epilogue call site enforces (`AlphaStatus::One`,
/// `BetaStatus::Zero`).
#[derive(Clone, Copy)]
pub struct IntGemmQ<O = i8>(PhantomData<O>);

impl<O: QuantOut> KernelFamily for IntGemmQ<O> {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = O;

    const OUT_IS_ACC: bool = false; // driver forces kc = k => fire-once, structurally

    /// Identical to [`IntGemm::pack_lhs`] (plain micropanel copy) — delegate to it.
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

    /// Identical to [`IntGemm::pack_rhs`] — delegate to it.
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
/// exact `i32` accumulation is produced by `vpdpbusd` (mirror of [`IntGemmVnni`]) —
/// `FORCE_PACK_* = true`, `DEPTH_MULTIPLE = 4`, the k-quad-interleaved `+128`/identity pack.
/// VNNI's grouped sum is bit-equal to the widen sum (int.rs modular associativity + bias
/// correction), so `IntGemmVnniQ<O>` and `IntGemmQ<O>` requantize to identical output. `O`
/// defaults to `i8` (so bare `IntGemmVnniQ` means `IntGemmVnniQ<i8>`).
#[derive(Clone, Copy)]
pub struct IntGemmVnniQ<O = i8>(PhantomData<O>);

impl<O> IntGemmVnniQ<O> {
    /// Depth steps folded per `vpdpbusd`.
    const Q: usize = 4;
}

impl<O: QuantOut> KernelFamily for IntGemmVnniQ<O> {
    type Lhs = i8;
    type Rhs = i8;
    type Acc = i32;
    type Out = O;

    const OUT_IS_ACC: bool = false;
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;
    const DEPTH_MULTIPLE: usize = Self::Q;

    /// Identical to [`IntGemmVnni::pack_lhs`] (k-quad-interleaved with the `+128` bias) —
    /// delegate to it.
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

    /// Identical to [`IntGemmVnni::pack_rhs`] (k-quad-interleaved, values stay signed) —
    /// delegate to it.
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
            // Fully qualified: a VNNI token carries both `<i8,i8,i32,i32>` and the blanket
            // `<i8,i8,i32,O>` `dot_accumulate`; the blanket delegates to the former.
            <S as KernelSimd<i8, i8, i32, O>>::dot_accumulate::<MR_REG, NR>(
                simd, kc, a, b, &mut acc,
            );
            requant_scratch_epilogue::<Self, S, E, O, MR_REG, NR>(
                simd, &acc, c, rsc, csc, mr_eff, nr_eff, row0, col0, epi, scratch,
            );
        }
    }
}
