//! The shared split (structure-of-arrays) complex microkernel, plus the per-ISA glue
//! macro that wires each token into it
//!
//! [`crate::kernel::complex::ComplexGemm`] is homogeneous at the family level
//! (`Lhs = Rhs = Acc = Out = Complex<_>`), so the driver binds it through
//! `KernelSimd<Complex<_>, Complex<_>, Complex<_>, Complex<_>>`, which resolves to plain
//! `SimdOps<Complex<_>>`, not the *real*-typed ops the split kernel needs to run its FMAs.
//! [`SimdOps::cplx_microkernel`] bridges that gap: each token's override (emitted by
//! [`impl_complex_simd!`] below) forwards to the one ISA-generic [`soa_microkernel`],
//! generic over `S: SimdOps<R>` for the real component type `R`, which the token supplies
//! concretely
//!
//! The `SimdOps<Complex<_>>` impl the macro generates is glue, not a kernel: it exists so
//! the driver can read `LANES` and `Reg` and so the homogeneous `KernelSimd` blanket
//! applies to `ComplexGemm`. Its element ops (`zero`/`splat`/`mul`/...) are never called
//! and stub out to `unreachable!`; `LANES` is the **real** lane count, since one real lane
//! spans one complex row of the tile

use super::SimdOps;
use crate::scalar::ComplexFloat;

/// The split (SoA) complex microkernel: accumulate one `MR x NR` complex tile into
/// separate real/imaginary register banks over `kc` depth steps, then apply the complex
/// `alpha`/`beta` epilogue and write the interleaved output
///
/// Each complex multiply-accumulate is 4 real FMAs into the 2 banks, in a fixed order:
/// `acc_re += ar*br`, `acc_re -= ai*bi`, `acc_im += ar*bi`, `acc_im += ai*br`. Ascending
/// `p` and this exact step order are what make full and edge tiles of the same matrix
/// round identically. `a`/`b` are the **planar** packed panels (the real plane then the
/// imaginary plane per depth step); both operands are always packed, so `a_cs`/`b_rs`
/// equal the panel widths `mr`/`NR`. `c` is the interleaved output tile: the epilogue
/// de-interleaves through `scratch`, folds complex `alpha`/`beta` once per output element
/// (`O(MN)`, not `O(MNK)`), and re-interleaves on store
///
/// # Safety
/// `a`/`b` valid for the packed planar panels over `kc` depth; `c` valid for the
/// `mr_eff x nr_eff` output sub-tile at `rsc`/`csc`; `scratch` valid for `2*mr*NR`
/// reals; run inside this token's [`super::Simd::vectorize`] context
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
pub(crate) unsafe fn soa_microkernel<C, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    kc: usize,
    alpha: C,
    beta: C,
    alpha_is_one: bool,
    beta_is_zero: bool,
    beta_is_one: bool,
    a: *const C,
    a_cs: isize,
    b: *const C,
    b_rs: isize,
    c: *mut C,
    rsc: isize,
    csc: isize,
    mr_eff: usize,
    nr_eff: usize,
    scratch: *mut C,
) where
    C: ComplexFloat,
    S: SimdOps<C::Real>,
{
    unsafe {
        let lanes = <S as SimdOps<C::Real>>::LANES;
        let mr = MR_REG * lanes; // complex rows spanned by the tile
        // Both operands are always packed for this kernel, so the depth stride is
        // exactly the panel width
        debug_assert_eq!(
            a_cs as usize, mr,
            "complex LHS panel must be packed (a_cs == mr)"
        );
        debug_assert_eq!(
            b_rs as usize, NR,
            "complex RHS panel must be packed (b_rs == NR)"
        );

        let a_re = a as *const C::Real;
        let b_re = b as *const C::Real;
        let scratch = scratch as *mut C::Real;

        // Real and imaginary accumulator banks, zeroed before the depth loop
        let mut acc_re: [[<S as SimdOps<C::Real>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];
        let mut acc_im: [[<S as SimdOps<C::Real>>::Reg; MR_REG]; NR] = [[simd.zero(); MR_REG]; NR];

        // Hot loop: one mul_add and one fnma build acc_re, 2 mul_adds build acc_im, in
        // this fixed order every step
        for p in 0..kc {
            let are_p = a_re.add(p * 2 * mr); // real plane of this depth step
            let aim_p = are_p.add(mr); // imaginary plane follows, offset by mr
            let ar: [<S as SimdOps<C::Real>>::Reg; MR_REG] =
                core::array::from_fn(|i| simd.loadu(are_p.add(i * lanes)));
            let ai: [<S as SimdOps<C::Real>>::Reg; MR_REG] =
                core::array::from_fn(|i| simd.loadu(aim_p.add(i * lanes)));
            let bre_p = b_re.add(p * 2 * NR);
            let bim_p = bre_p.add(NR);
            for j in 0..NR {
                let br = simd.splat(*bre_p.add(j));
                let bi = simd.splat(*bim_p.add(j));
                for i in 0..MR_REG {
                    acc_re[j][i] = simd.mul_add(ar[i], br, acc_re[j][i]); // acc_re += ar*br
                    acc_re[j][i] = simd.fnma(ai[i], bi, acc_re[j][i]); //    acc_re -= ai*bi
                    acc_im[j][i] = simd.mul_add(ar[i], bi, acc_im[j][i]); // acc_im += ar*bi
                    acc_im[j][i] = simd.mul_add(ai[i], br, acc_im[j][i]); // acc_im += ai*br
                }
            }
        }

        // Drain both banks to planar scratch (real block, then imaginary block), each laid
        // out `scratch[j*mr + row]`, column-major within the tile, so the scalar epilogue
        // below can index any live (row, col) uniformly regardless of tile shape or stride
        let im_base = mr * NR;
        for j in 0..NR {
            for i in 0..MR_REG {
                simd.storeu(scratch.add(j * mr + i * lanes), acc_re[j][i]);
                simd.storeu(scratch.add(im_base + j * mr + i * lanes), acc_im[j][i]);
            }
        }

        // Epilogue: read the drained scratch tile, apply complex alpha/beta, and
        // re-interleave real/imaginary into the output on store
        let (al_re, al_im) = (alpha.re(), alpha.im());
        let (be_re, be_im) = (beta.re(), beta.im());
        for j in 0..nr_eff {
            for i in 0..mr_eff {
                let ab_re = *scratch.add(j * mr + i);
                let ab_im = *scratch.add(im_base + j * mr + i);
                // t = alpha * AB, skipping the complex multiply when alpha == 1
                let (t_re, t_im) = if alpha_is_one {
                    (ab_re, ab_im)
                } else {
                    (al_re * ab_re - al_im * ab_im, al_re * ab_im + al_im * ab_re)
                };
                let cp = c.offset(i as isize * rsc + j as isize * csc);
                let out = if beta_is_zero {
                    C::new(t_re, t_im)
                } else if beta_is_one {
                    let cz = *cp;
                    C::new(cz.re() + t_re, cz.im() + t_im)
                } else {
                    let cz = *cp;
                    let (c_re, c_im) = (cz.re(), cz.im());
                    // out = beta*C + t, complex multiply expanded
                    C::new(
                        be_re * c_re - be_im * c_im + t_re,
                        be_re * c_im + be_im * c_re + t_im,
                    )
                };
                *cp = out;
            }
        }
    }
}

/// Generate the `SimdOps<Complex<$real>>` glue impl for one `($tok, $real)` pair: the
/// `LANES` (the **real** lane count) and `Reg` the driver reads, every element op
/// stubbed to `unreachable!` (complex GEMM never calls them, it routes through
/// `cplx_microkernel`), and the `cplx_microkernel` override forwarding to
/// [`soa_microkernel`]. This is boilerplate wiring, not a kernel body: the actual kernel
/// is the single `soa_microkernel` above, shared by every invocation of this macro
macro_rules! impl_complex_simd {
    ($tok:ty, $real:ty, $reg:ty, $lanes:expr) => {
        impl $crate::simd::SimdOps<num_complex::Complex<$real>> for $tok {
            type Reg = $reg;
            // LANES is the real lane count: one real lane is one complex row, so the
            // driver's mr = MR_REG * LANES is the tile's complex-row count
            const LANES: usize = $lanes;

            // Never called: complex GEMM runs its arithmetic through SimdOps<$real> inside
            // soa_microkernel, not through these. Present only to satisfy the trait
            #[inline(always)]
            unsafe fn zero(self) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn splat(self, _v: num_complex::Complex<$real>) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn loadu(self, _p: *const num_complex::Complex<$real>) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn storeu(self, _p: *mut num_complex::Complex<$real>, _v: $reg) {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn mul(self, _a: $reg, _b: $reg) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn add(self, _a: $reg, _b: $reg) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn mul_add(self, _a: $reg, _b: $reg, _c: $reg) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn fnma(self, _a: $reg, _b: $reg, _c: $reg) -> $reg {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }
            #[inline(always)]
            unsafe fn reduce_sum(self, _v: $reg) -> num_complex::Complex<$real> {
                unreachable!("complex GEMM routes through cplx_microkernel")
            }

            #[allow(clippy::too_many_arguments)]
            #[inline(always)]
            unsafe fn cplx_microkernel<const MR_REG: usize, const NR: usize>(
                self,
                kc: usize,
                alpha: num_complex::Complex<$real>,
                beta: num_complex::Complex<$real>,
                alpha_is_one: bool,
                beta_is_zero: bool,
                beta_is_one: bool,
                a: *const num_complex::Complex<$real>,
                a_cs: isize,
                b: *const num_complex::Complex<$real>,
                b_rs: isize,
                c: *mut num_complex::Complex<$real>,
                rsc: isize,
                csc: isize,
                mr_eff: usize,
                nr_eff: usize,
                scratch: *mut num_complex::Complex<$real>,
            ) {
                unsafe {
                    $crate::simd::complex::soa_microkernel::<
                        num_complex::Complex<$real>,
                        $tok,
                        MR_REG,
                        NR,
                    >(
                        self,
                        kc,
                        alpha,
                        beta,
                        alpha_is_one,
                        beta_is_zero,
                        beta_is_one,
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
        }
    };
}
