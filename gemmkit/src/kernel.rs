//! Kernel families (layer L4): the seam between the driver and a concrete kind of GEMM
//!
//! The driver (layer L5) is generic over [`KernelFamily`], not over "do an FMA on
//! `T`". A family bundles everything that distinguishes one kind of GEMM from
//! another: the input/accumulator/output element types, the pack layout, the
//! microkernel, and the epilogue. [`float::FloatGemm`] is the baseline family;
//! complex, integer, and mixed-precision GEMM are separate families that reuse the
//! driver, the packing framework, the cache model, and the parallelism layer
//! unchanged. That open/closed property is part of the architecture contract, and
//! a test proves it by declaring a 2nd, trivial family
//!
//! The tile geometry (`MR_REG`, `NR`) is not part of this trait: it is a pair of
//! const generics chosen per `(family, ISA)` at the dispatch site, so a new tile
//! is a new instantiation, never a new type or macro

use crate::scalar::Scalar;
use crate::simd::KernelSimd;

// Complex GEMM family (Complex<f32> / Complex<f64>): dedicated split (SoA) kernel
// with optional conjugation of A and/or B; gated on the `complex` feature
#[cfg(feature = "complex")]
pub mod complex;
// Fused epilogue trait, plus the built-in bias/activation and requantize epilogues
pub mod epilogue;
// Floating-point GEMM family: the one generic microkernel shared by every ISA
pub mod float;
// Integer GEMM family (i8 in, i32 accumulator): widen-and-multiply and VNNI dot
// variants; gated on the `int8` feature
#[cfg(feature = "int8")]
pub mod int;
// Mixed-precision GEMM family (f16/bf16 in, f32 accumulator); gated on the
// `half` feature
#[cfg(feature = "half")]
pub mod mixed;

#[cfg(feature = "complex")]
pub use complex::ComplexGemm;
pub use epilogue::{Epilogue, Identity};
pub use float::FloatGemm;
#[cfg(feature = "int8")]
pub use int::{IntGemm, IntGemmVnni};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use int::{IntGemmQ, IntGemmVnniQ};
#[cfg(feature = "half")]
pub use mixed::{Bf16DotGemm, Bf16DotGemmF32, MixedGemm, MixedGemmF32};

/// Precomputed state of the `alpha` scale, so the microkernel branches on this
/// enum instead of comparing floats. `alpha == 0` is intercepted upstream (the
/// dispatch layer routes it to a beta-only scale of `C`), so only `One` and
/// `Other` ever reach a family's microkernel
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AlphaStatus {
    /// `alpha == 1`: the product needs no extra scale
    One,
    /// Any other `alpha`
    Other,
}

/// Precomputed state of the effective `beta` for the current depth slice; see
/// [`AlphaStatus`] for why this is an enum rather than a float comparison
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BetaStatus {
    /// `beta == 0`: `C` is not read on this slice (its contents may be
    /// uninitialized or NaN)
    Zero,
    /// `beta == 1`: this slice accumulates straight into the existing `C`
    One,
    /// Any other `beta`: this slice scales `C` before accumulating
    Other,
}

/// Upper bound on any family's microkernel row count (`MR_REG * LANES`); sizes
/// the stack scratch buffer used for edge / general-stride tiles
pub const MAX_MR: usize = 64;
/// Upper bound on any family's microkernel column count (`NR`)
pub const MAX_NR: usize = 32;
/// Size, in accumulator elements, of the per-call scratch tile bounded by
/// [`MAX_MR`] x [`MAX_NR`]
pub const SCRATCH_LEN: usize = MAX_MR * MAX_NR;

/// The operation-family seam: everything the generic driver needs to run one
/// particular kind of GEMM
///
/// Every method is `unsafe` (raw pointers, and a target-feature codegen context
/// for the microkernel methods); see each method for its exact contract
pub trait KernelFamily: Copy + Send + Sync + 'static {
    /// Left-hand input element type
    type Lhs: Scalar;
    /// Right-hand input element type
    type Rhs: Scalar;
    /// Accumulator element type: `Lhs::Acc` in every shipped family (`f32`/`f64`
    /// for float, `i32` for `i8`, `f32` for `f16`/`bf16`, same as `Lhs` for complex)
    type Acc: Scalar;
    /// Output element type
    type Out: Scalar;

    /// Whether a running `Acc` partial sum can round-trip through `C` (`Out`)
    /// between `kc` depth panels without losing precision. `true` exactly when
    /// `Out == Acc`; that is also the default, so a family need not set it
    /// unless it narrows on store
    ///
    /// The driver's default K-blocking re-reads and re-writes `C` once per `kc`
    /// panel (`beta` becomes `1` after the 1st), which is exact only when
    /// `Out == Acc`. A narrowing family (`Out` = `f16`/`bf16`/`i8`/`u8`, `Acc` =
    /// `f32`/`i32`) would round at every panel boundary if it did that, so it
    /// sets this `false`; the driver responds by using `kc = k` (a single depth
    /// panel), so the whole contraction accumulates in `Acc` and narrows to
    /// `Out` exactly once
    const OUT_IS_ACC: bool = true;

    /// Force the driver to always pack the LHS, bypassing its cost-based
    /// pack/no-pack decision. Set this when packing is not a plain copy (a
    /// k-group interleave, a signedness bias, a conjugation): the driver's
    /// in-place read path would then be wrong, not just slower, so the
    /// transform must always run. Default `false`
    const FORCE_PACK_LHS: bool = false;
    /// See [`KernelFamily::FORCE_PACK_LHS`]
    const FORCE_PACK_RHS: bool = false;

    /// Depth-panel padding multiple: the driver rounds every packed panel's
    /// depth up to this before sizing and addressing the A/B pack buffers. `1`
    /// (the default) is a no-op: every homogeneous or widen-and-multiply family.
    /// A dot-product family that folds `Q` consecutive depth steps into one
    /// hardware instruction (VNNI `vpdpbusd`: `Q = 4`; `vdpbf16ps`: `Q = 2`)
    /// sets this to `Q`, so each packed micropanel holds a whole number of
    /// instruction-groups and the kernel reads `ceil(kc / Q)` of them
    ///
    /// Contract: a family with `DEPTH_MULTIPLE = Q` must make [`pack_lhs`] /
    /// [`pack_rhs`] write `width * kc.next_multiple_of(Q)` elements per panel,
    /// zero-padding the depth tail (VNNI pads `A` with its `+128` zero bias, `B`
    /// with zero). The driver strides packed panels by that same padded depth,
    /// so the 2 stay in lockstep. `Q = 1` reproduces the unpadded behavior every
    /// other family relies on
    ///
    /// [`pack_lhs`]: KernelFamily::pack_lhs
    /// [`pack_rhs`]: KernelFamily::pack_rhs
    const DEPTH_MULTIPLE: usize = 1;

    /// Pack an `mc x kc` LHS block into micropanel-major layout: `ceil(mc/mr)`
    /// panels of `mr` rows each, every panel stored one depth step at a time
    /// with `mr` contiguous rows per step (`mr == MR_REG*LANES`; a short tail
    /// row-block is zero-padded up to `mr`)
    ///
    /// # Safety
    /// `src` must be valid for the `mc x kc` region described by `rs`/`cs`;
    /// `dst` must be valid for `ceil(mc/mr)*mr*kc` writes
    unsafe fn pack_lhs(
        dst: *mut Self::Lhs,
        src: *const Self::Lhs,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    );

    /// Pack a `kc x nc` RHS block into micropanel-major layout: `ceil(nc/nr)`
    /// panels of `nr` columns each, every panel stored one depth step at a time
    /// with `nr` contiguous columns per step (a short tail column-block is
    /// zero-padded up to `nr`)
    ///
    /// # Safety
    /// `src` must be valid for the `kc x nc` region described by `rs`/`cs`;
    /// `dst` must be valid for `ceil(nc/nr)*nr*kc` writes
    unsafe fn pack_rhs(
        dst: *mut Self::Rhs,
        src: *const Self::Rhs,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    );

    /// Compute one `MR x NR` output tile (`MR == MR_REG*LANES`, `LANES` the
    /// `Acc` lane count for token `S`) and store `C <- combine(alpha*A*B, beta*C)`
    ///
    /// # Parameters
    /// - `a`/`a_cs` - LHS panel base and column (depth) stride; a packed panel
    ///   has `a_cs == mr`, an unpacked column-major LHS has `a_cs == csa`; rows
    ///   are always unit-stride
    /// - `b`/`b_rs`/`b_cs` - RHS panel base and strides; a packed panel is
    ///   `(nr, 1)`
    /// - `c`/`rsc`/`csc` - output tile base and strides; the fast vector store
    ///   needs `rsc == 1` and a full tile, else the call drains through
    ///   `scratch` and copies back under arbitrary strides
    /// - `mr_eff`/`nr_eff` - live sub-tile size (`<= MR`/`NR` at an edge tile)
    /// - `scratch` - at least [`SCRATCH_LEN`] accumulator elements of stack
    ///   space
    ///
    /// A family overrides exactly one of the 2 microkernel methods. A
    /// non-fusing family (`IntGemm`, `IntGemmVnni`, the open/closed test
    /// family) overrides this plain method and inherits the default
    /// [`microkernel_epi`], which forwards straight here after a fail-closed
    /// `assert!(E::IS_IDENTITY)`. A fusing family (`FloatGemm`, the mixed
    /// families, the requantizing integer families) instead overrides
    /// [`microkernel_epi`] to thread `E` through its own store, leaving this
    /// default body `unreachable!`. `ComplexGemm` overrides both: its
    /// [`microkernel_epi`] calls this method first and then sweeps the
    /// epilogue over the finished tile
    ///
    /// # Safety
    /// All pointers must be valid for the accesses implied by the strides and
    /// dimensions, and the call must run inside the matching
    /// [`crate::simd::Simd::vectorize`] context for `S`
    ///
    /// [`microkernel_epi`]: KernelFamily::microkernel_epi
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel<S, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: Self::Acc,
        beta: Self::Acc,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const Self::Lhs,
        a_cs: isize,
        b: *const Self::Rhs,
        b_rs: isize,
        b_cs: isize,
        c: *mut Self::Out,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        scratch: *mut Self::Acc,
    ) where
        S: KernelSimd<Self::Lhs, Self::Rhs, Self::Acc, Self::Out>,
    {
        let _ = (
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
        unreachable!("this family fuses via microkernel_epi and has no plain microkernel")
    }

    /// Compute one `MR x NR` tile and store the fused [`Epilogue`] `E` applied
    /// to each element. `row0`/`col0` are the tile's origin in the oriented
    /// problem frame (so a per-row/per-col bias resolves its absolute base);
    /// `last_k` marks the final depth panel, the only one on which an
    /// `OUT_IS_ACC` family may apply the epilogue (earlier panels store raw
    /// `Acc` partials, per the contract on [`KernelFamily::OUT_IS_ACC`])
    ///
    /// The default forwards to [`KernelFamily::microkernel`] and
    /// unconditionally asserts `E::IS_IDENTITY`, so it is correct only when the
    /// driver never fuses a real epilogue into this family. `IntGemm`,
    /// `IntGemmVnni`, and the open/closed test family rely on this default
    /// unedited. A family that supports fusion (`FloatGemm`, the mixed/narrow
    /// families, the requantizing integer families, and `ComplexGemm`, which
    /// also happens to call [`KernelFamily::microkernel`] from its override)
    /// overrides this method to thread `E` through its own store
    ///
    /// # Safety
    /// As [`KernelFamily::microkernel`]; `epi`'s interior pointers must be
    /// valid for the problem's `m`/`n`
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn microkernel_epi<S, E, const MR_REG: usize, const NR: usize>(
        simd: S,
        kc: usize,
        alpha: Self::Acc,
        beta: Self::Acc,
        alpha_status: AlphaStatus,
        beta_status: BetaStatus,
        a: *const Self::Lhs,
        a_cs: isize,
        b: *const Self::Rhs,
        b_rs: isize,
        b_cs: isize,
        c: *mut Self::Out,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        row0: usize,
        col0: usize,
        last_k: bool,
        epi: &E,
        scratch: *mut Self::Acc,
    ) where
        S: KernelSimd<Self::Lhs, Self::Rhs, Self::Acc, Self::Out>,
        E: Epilogue<Self>,
    {
        // A family that does not override this must never silently drop a real epilogue
        // `E::IS_IDENTITY` is a const, so the check folds away entirely when `E = Identity`
        // and becomes an unconditional panic for any other `E` reaching a non-overriding
        // family
        assert!(
            E::IS_IDENTITY,
            "this family does not implement fused epilogues"
        );
        let _ = (row0, col0, last_k, epi);
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
            )
        }
    }
}
