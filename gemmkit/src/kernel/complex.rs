//! The complex GEMM family: `Complex<f32>` / `Complex<f64>`, with optional
//! conjugation of `A` and/or `B`
//!
//! Unlike the other families, complex does not ride [`super::float::FloatGemm`].
//! It is a dedicated kernel built on the split (structure-of-arrays) layout: the
//! real and imaginary planes live in separate accumulator registers, so the hot
//! loop is pure real FMAs (`vfmadd`/`vfnmadd`, no in-loop shuffles or
//! `fmaddsub`). 1 complex multiply-accumulate is 4 fused real steps into 2
//! running accumulators: `acc_re += ar*br`, `acc_re -= ai*bi`,
//! `acc_im += ar*bi`, `acc_im += ai*br`
//!
//! The de-interleave moves out of the `kc` inner loop into the pack (amortized
//! `O(MK+KN)` instead of `O(MNK)`): [`ComplexGemm::pack_lhs`]/[`ComplexGemm::pack_rhs`]
//! lay each micropanel down planar: for every depth step, the `mr` (resp. `nr`)
//! reals are followed by the `mr` (resp. `nr`) imags, so the kernel loads a
//! register of reals and a register of imags with plain contiguous loads.
//! Because the kernel can only consume that planar layout, both operands are
//! always packed (`FORCE_PACK_LHS = FORCE_PACK_RHS = true`)
//!
//! Conjugation is a sign flip on the packed imaginary plane: `conjA` / `conjB`
//! are `const` parameters, and a set flag negates the imag plane during packing,
//! so `conj(A)*B` / `A*conj(B)` fall out of the same real-FMA loop, with no
//! per-element conj branch. `conjC` (output conjugation) is not yet implemented
//!
//! The family stays homogeneous (`Acc = T`, so complex `alpha`/`beta` thread
//! through the driver unchanged), but the hot loop runs on the real component.
//! The driver bound `KernelSimd<T, T, T, T>` only yields `SimdOps<T>` (complex),
//! not the real ops, so the microkernel below forwards to
//! [`crate::simd::SimdOps::cplx_microkernel`]: the L0 seam (the complex analogue
//! of `accumulate_tile`) whose per-ISA override has the real `SimdOps<T::Real>`
//! and runs the one shared SoA kernel. (The de-interleave pack and C epilogue
//! stay scalar: a vectorized `vld2`/`vst2` per-ISA seam was measured on NEON and
//! did not pay, the inner loop dominates, so the generic scalar path is the
//! floor; see `simd/neon.rs`)

use core::marker::PhantomData;

use super::{AlphaStatus, BetaStatus, Epilogue, KernelFamily};
use crate::scalar::{ComplexFloat, Scalar};
use crate::simd::KernelSimd;

/// The complex GEMM family. `T` is `Complex<f32>` or `Complex<f64>`; `CONJ_A` /
/// `CONJ_B` select which input is conjugated (both `false` = the plain product)
pub struct ComplexGemm<T, const CONJ_A: bool, const CONJ_B: bool>(PhantomData<T>);

impl<T, const CONJ_A: bool, const CONJ_B: bool> Clone for ComplexGemm<T, CONJ_A, CONJ_B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const CONJ_A: bool, const CONJ_B: bool> Copy for ComplexGemm<T, CONJ_A, CONJ_B> {}

/// Pack one `n_lead x depth_len` block into planar micropanels: `ceil(n_lead/width)`
/// panels, each storing, for every depth step, `width` real parts immediately
/// followed by `width` imaginary parts (`conj` negates the imags). Tail leading
/// positions past `n_lead` are zero-filled. This is the de-interleaved analogue
/// of [`crate::pack`]'s interleaved micropanel copy; LHS sets `lead = rows` /
/// `depth = cols`, RHS swaps them. It mirrors that helper's 2 paths (a
/// contiguous-leading walk and a cache-blocked transpose for a strided source),
/// so a row-major operand packs without a cache miss per element. The 2 paths
/// write byte-identical panels (only the write order differs)
///
/// # Safety
/// `src` must cover the `n_lead x depth_len` region addressed by `lead`/`depth`;
/// `dst` must hold `ceil(n_lead/width) * width * depth_len` complex elements
/// (i.e. twice that many `T::Real`)
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn pack_planar<T: ComplexFloat>(
    dst: *mut T,
    src: *const T,
    lead: isize,
    depth: isize,
    n_lead: usize,
    depth_len: usize,
    width: usize,
    conj: bool,
) {
    unsafe {
        let tile = crate::tuning::pack_transpose_tile();
        let zero = <T::Real as Scalar>::ZERO;
        // conj = negate the imaginary plane (true `-im`, so `+0.0` maps to `-0.0`,
        // matching `num_complex`'s `.conj()`); else copy it through
        let pack_im = |im: T::Real| if conj { -im } else { im };
        // Each panel occupies `depth_len * 2 * width` reals (re plane + im plane per
        // depth step). Write through a real-typed cursor
        let mut panel = dst as *mut T::Real;
        let mut base = 0usize;
        while base < n_lead {
            let live = core::cmp::min(width, n_lead - base);
            if lead == 1 {
                // Contiguous leading dimension: each depth step's `width` complex are
                // adjacent, so walk them in order and de-interleave
                for p in 0..depth_len {
                    let re_off = p * 2 * width;
                    let s = src.offset(base as isize + p as isize * depth);
                    for i in 0..width {
                        if i < live {
                            let z = *s.add(i);
                            *panel.add(re_off + i) = z.re();
                            *panel.add(re_off + width + i) = pack_im(z.im());
                        } else {
                            *panel.add(re_off + i) = zero;
                            *panel.add(re_off + width + i) = zero;
                        }
                    }
                }
            } else {
                // Cache-blocked transpose: walk the source along its contiguous `depth`
                // dimension (stride 1 for a row-major LHS / column-major RHS) in short
                // strips per leading row, scattering into the planar panel, rather than
                // gathering `width` strided elements per depth step (a cache miss each)
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
                    for i in 0..width {
                        if i < live {
                            let row = src.offset((base + i) as isize * lead);
                            for p in p0..pe {
                                let z = *row.offset(p as isize * depth);
                                *panel.add(p * 2 * width + i) = z.re();
                                *panel.add(p * 2 * width + width + i) = pack_im(z.im());
                            }
                        } else {
                            for p in p0..pe {
                                *panel.add(p * 2 * width + i) = zero;
                                *panel.add(p * 2 * width + width + i) = zero;
                            }
                        }
                    }
                    p0 = pe;
                }
            }
            panel = panel.add(depth_len * 2 * width);
            base += width;
        }
    }
}

impl<T, const CONJ_A: bool, const CONJ_B: bool> KernelFamily for ComplexGemm<T, CONJ_A, CONJ_B>
where
    T: ComplexFloat,
{
    type Lhs = T;
    type Rhs = T;
    type Acc = T;
    type Out = T;

    // The SoA kernel can only consume the planar (de-interleaved) layout, so both
    // operands must always be packed, never read in place
    const FORCE_PACK_LHS: bool = true;
    const FORCE_PACK_RHS: bool = true;

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
        // LHS: leading dimension is rows (stride `rs`), depth is columns (stride `cs`)
        unsafe {
            pack_planar(
                dst, src, /*lead*/ rs, /*depth*/ cs, mc, kc, mr, CONJ_A,
            );
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
        // RHS: leading dimension is columns (stride `cs`), depth is rows (stride `rs`)
        unsafe {
            pack_planar(
                dst, src, /*lead*/ cs, /*depth*/ rs, nc, kc, nr, CONJ_B,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        _b_cs: isize,
        c: *mut T,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut T,
    ) where
        S: KernelSimd<T, T, T, T>,
    {
        // Forward to the L0 SoA seam, translating the alpha/beta state to plain bools
        // (the L0 method must not depend on the L1 status enums). `b_cs` is unused: a
        // packed RHS is contiguous (`b_cs == 1`)
        unsafe {
            simd.cplx_microkernel::<MR_REG, NR>(
                kc,
                alpha,
                beta,
                alpha_status == AlphaStatus::One,
                beta_status == BetaStatus::Zero,
                beta_status == BetaStatus::One,
                a,
                a_cs,
                b,
                b_rs,
                c,
                rsc,
                csc,
                mr_eff,
                nr_eff,
                scratch,
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
        // The complex kernel's alpha/beta epilogue lives inside the L0 `cplx_microkernel`
        // seam, which must not depend on the L1 [`Epilogue`] trait (the same layering rule
        // that keeps `AlphaStatus` out of L0). So `E` cannot be threaded into the store the
        // way `FloatGemm` does; instead, the unchanged SoA kernel runs first (it stores the
        // finished alpha*AB + beta*C tile to C, exactly the bits plain `gemm_cplx` would),
        // and then, on the final depth panel only, sweeps the live tile applying `epi` in
        // place
        //
        // The `!E::IS_IDENTITY` guard is a `const`, so the `Identity` instantiation (every
        // non-fused complex call, and every intermediate depth panel of a fused one)
        // const-folds the whole post-pass away, and this override is byte-for-byte the bare
        // `microkernel`
        //
        // `last_k` gates the sweep because complex is `OUT_IS_ACC = true`: the driver splits
        // K and re-reads C between depth panels, so intermediate panels must leave their raw
        // partials untouched (the bias may fire exactly once, on the completed sum). The
        // post-pass costs one cache-hot `O(mr*nr)` sweep, fires once per element, and is
        // bitwise-identical to `gemm_cplx`-then-map by construction (same reasoning as the P1
        // gemv sweep)
        unsafe {
            Self::microkernel::<S, MR_REG, NR>(
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
                scratch,
            );
            if !E::IS_IDENTITY && last_k {
                for j in 0..nr_eff {
                    for i in 0..mr_eff {
                        let cp = c.offset(i as isize * rsc + j as isize * csc);
                        *cp = epi.apply(*cp, row0 + i, col0 + j);
                    }
                }
            }
        }
    }
}
