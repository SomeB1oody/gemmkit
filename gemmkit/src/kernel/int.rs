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

use super::{AlphaStatus, BetaStatus, KernelFamily};
use crate::pack::{pack_kgroup_panels, pack_panels};
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

/// The VNNI integer GEMM family: `Lhs = Rhs = i8`, `Acc = Out = i32`, driven by
/// `vpdpbusd` (4 depth steps × 16 lanes per instruction) instead of widen-and-multiply.
///
/// Two things change versus [`IntGemm`], both isolated to this family:
///
/// * **Pack layout** (`DEPTH_MULTIPLE = 4`): A and B are packed *k-quad-interleaved* —
///   four consecutive depth steps contiguous per lane/column so the kernel can issue one
///   `vpdpbusd`. Depth is padded to a multiple of 4 (A pad = `128`, B pad = `0`). Both
///   operands are always packed (`FORCE_PACK_*`), since the interleave cannot be read
///   in place.
/// * **Signedness** (`u8 × i8`): `vpdpbusd` multiplies *unsigned* A by *signed* B, but
///   GEMM is signed × signed. The pack offsets A by `+128` into `[0, 255]`; the kernel's
///   [`crate::simd::KernelSimd::dot_accumulate`] override subtracts the per-column bias
///   `128·Σ_k B[k][j]` so the accumulator holds the true `Σ_k A·B` before the epilogue.
///
/// The accumulation stays exact: i32 add is associative under wrapping, so the 4-way
/// `vpdpbusd` grouping plus the integer bias correction equals the ascending-`k` widen
/// sum **bit-for-bit** — VNNI and [`IntGemm`] produce identical output. The alpha fold
/// and the `i32` epilogue are identical to [`IntGemm`].
pub struct IntGemmVnni(PhantomData<()>);

impl Clone for IntGemmVnni {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for IntGemmVnni {}

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

    /// Pack the `mc × kc` LHS into k-quad-interleaved micropanels (4 contiguous depth bytes
    /// per row, ready for `vpdpbusd`), via the shared [`pack_kgroup_panels`]. Every byte is
    /// offset `+128` into `[0, 255]` by the transform; the pad (rows past `mc` / depth past
    /// `kc`) is `xform(0) = 128`, the offset of `0`, contributing nothing after the bias
    /// correction.
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
            pack_kgroup_panels::<i8, { Self::Q }, _>(dst, src, rs, cs, mc, kc, mr, |v| {
                ((v as i32 + 128) as u8) as i8
            })
        }
    }

    /// Pack one `kc × nr` RHS panel k-quad-interleaved (4 contiguous depth bytes per column,
    /// ready for an i32 broadcast), via the shared [`pack_kgroup_panels`]. Values stay
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
            simd.dot_accumulate::<MR_REG, NR>(kc, a, b, &mut acc);

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
