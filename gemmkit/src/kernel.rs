//! Kernel families (layer L4): the seam between the driver and a concrete kind of GEMM
//!
//! The driver (layer L5) is generic over [`KernelFamily`], not over how to run an FMA
//! on `T`. A family bundles everything that distinguishes one kind of GEMM from
//! another. This includes the input, accumulator, and output element types, the pack
//! layout, the microkernel, and the epilogue. [`float::FloatGemm`] is the baseline
//! family. Complex, integer, and mixed-precision GEMM are separate families that reuse
//! the driver, the packing framework, the cache model, and the parallelism layer
//! without change. A test proves this open and closed property by declaring a 2nd,
//! trivial family
//!
//! The tile geometry (`MR_REG`, `NR`) is not part of this trait. It is a pair of const
//! generics chosen per family and ISA at the dispatch site. A new tile is therefore a
//! new instantiation, never a new type or macro

use crate::scalar::Scalar;
use crate::simd::KernelSimd;

// The complex GEMM family: a dedicated split kernel for Complex<f32> and Complex<f64>,
// with optional conjugation of A and/or B. Enabled by the `complex` feature
#[cfg(feature = "complex")]
pub mod complex;
// The fused epilogue trait, plus the built-in bias, activation, and requantize epilogues
pub mod epilogue;
// The floating-point GEMM family: 1 generic microkernel shared by every ISA
pub mod float;
// The integer GEMM family (i8 input, i32 accumulator): widen-and-multiply and VNNI dot
// variants. Enabled by the `int8` feature
#[cfg(feature = "int8")]
pub mod int;
// The mixed-precision GEMM family (f16 or bf16 input, f32 accumulator). Enabled by the
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

/// Precomputed state of the `alpha` scale, so the microkernel branches on this enum
/// instead of comparing floats. The dispatch layer intercepts `alpha == 0` upstream and
/// routes it to a beta-only scale of `C`. Only `One` and `Other` ever reach a family's
/// microkernel
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AlphaStatus {
    /// `alpha == 1`, so the product needs no extra scale
    One,
    /// Any other `alpha` value
    Other,
}

/// Precomputed state of the effective `beta` for the current depth slice. See
/// [`AlphaStatus`] for why this is an enum rather than a float comparison
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BetaStatus {
    /// `beta == 0`, so `C` is not read on this slice and its contents may be
    /// uninitialized or NaN
    Zero,
    /// `beta == 1`, so this slice accumulates straight into the existing `C`
    One,
    /// Any other `beta` value, so this slice scales `C` before accumulating
    Other,
}

/// Upper bound on any family's microkernel row count (`MR_REG * LANES`). Sizes the
/// stack scratch buffer used for edge or general-stride tiles
pub const MAX_MR: usize = 64;
/// Upper bound on any family's microkernel column count (`NR`)
pub const MAX_NR: usize = 32;
/// Size, in accumulator elements, of the per-call scratch tile bounded by [`MAX_MR`]
/// x [`MAX_NR`]
pub const SCRATCH_LEN: usize = MAX_MR * MAX_NR;

/// The operation-family seam: everything the generic driver needs to run one
/// particular kind of GEMM
///
/// Every method is `unsafe`, because it takes raw pointers and, for the microkernel
/// methods, needs a target-feature codegen context. See each method for its exact
/// contract
pub trait KernelFamily: Copy + Send + Sync + 'static {
    /// Left-hand input element type
    type Lhs: Scalar;
    /// Right-hand input element type
    type Rhs: Scalar;
    /// Accumulator element type. This is `Lhs::Acc` in every shipped family. That is
    /// `f32` or `f64` for float, `i32` for `i8`, `f32` for `f16` or `bf16`, and the
    /// same as `Lhs` for complex
    type Acc: Scalar;
    /// Output element type
    type Out: Scalar;

    /// Whether a running `Acc` partial sum can round-trip through `C` (`Out`) between
    /// `kc` depth panels without losing precision. This is `true` exactly when `Out ==
    /// Acc`, which is also the default, so a family need not set it unless it narrows
    /// on store
    ///
    /// The driver's default K-blocking re-reads and re-writes `C` once per `kc` panel,
    /// with `beta` becoming `1` after the 1st panel. That is exact only when `Out ==
    /// Acc`. A narrowing family, where `Out` is `f16`, `bf16`, `i8`, or `u8` and `Acc`
    /// is `f32` or `i32`, sets this to `false`. Round-tripping through `C` at every
    /// panel boundary would lose precision for that family. The driver then uses `kc =
    /// k`, a single depth panel, so the whole contraction accumulates in `Acc` and
    /// narrows to `Out` exactly once
    const OUT_IS_ACC: bool = true;

    /// Force the driver to always pack the LHS, bypassing its cost-based pack or
    /// no-pack decision. Set this when packing is not a plain copy, such as a k-group
    /// interleave, a signedness bias, or a conjugation. In that case the driver's
    /// in-place read path would be wrong, not just slower, so the transform must
    /// always run. The default is `false`
    const FORCE_PACK_LHS: bool = false;
    /// See [`KernelFamily::FORCE_PACK_LHS`]
    const FORCE_PACK_RHS: bool = false;

    /// Depth-panel padding multiple. The driver rounds every packed panel's depth up to
    /// this before it sizes and addresses the A/B pack buffers. The default, `1`, is a
    /// no-op for every homogeneous or widen-and-multiply family
    ///
    /// A dot-product family folds `Q` consecutive depth steps into 1 hardware
    /// instruction. VNNI's `vpdpbusd` uses `Q = 4` and `vdpbf16ps` uses `Q = 2`. Such a
    /// family sets this constant to `Q`. Each packed micropanel then holds a whole
    /// number of instruction groups, and the kernel reads `ceil(kc / Q)` of them
    ///
    /// A family with `DEPTH_MULTIPLE = Q` must make [`pack_lhs`] and [`pack_rhs`] write
    /// `width * kc.next_multiple_of(Q)` elements per panel, zero-padding the depth
    /// tail. VNNI pads `A` with its `+128` zero bias and pads `B` with zero. The driver
    /// strides packed panels by that same padded depth, so the 2 stay in lockstep. `Q =
    /// 1` reproduces the unpadded behavior every other family relies on
    ///
    /// [`pack_lhs`]: KernelFamily::pack_lhs
    /// [`pack_rhs`]: KernelFamily::pack_rhs
    const DEPTH_MULTIPLE: usize = 1;

    /// Pack an `mc x kc` LHS block into micropanel-major layout. The result is
    /// `ceil(mc/mr)` panels of `mr` rows each. Every panel stores one depth step at a
    /// time, with `mr` contiguous rows per step (`mr == MR_REG*LANES`). A short tail
    /// row-block is zero-padded up to `mr`
    ///
    /// # Safety
    ///
    /// `src` must be valid for the `mc x kc` region described by `rs` and `cs`. `dst`
    /// must be valid for `ceil(mc/mr)*mr*kc` writes
    unsafe fn pack_lhs(
        dst: *mut Self::Lhs,
        src: *const Self::Lhs,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    );

    /// Pack a `kc x nc` RHS block into micropanel-major layout. The result is
    /// `ceil(nc/nr)` panels of `nr` columns each, with every panel stored one depth
    /// step at a time and `nr` contiguous columns per step. A short tail column-block
    /// is zero-padded up to `nr`
    ///
    /// # Safety
    ///
    /// `src` must be valid for the `kc x nc` region described by `rs` and `cs`. `dst`
    /// must be valid for `ceil(nc/nr)*nr*kc` writes
    unsafe fn pack_rhs(
        dst: *mut Self::Rhs,
        src: *const Self::Rhs,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    );

    /// Compute one `MR x NR` output tile and store `C <- combine(alpha*A*B, beta*C)`.
    /// `MR == MR_REG*LANES`, where `LANES` is the `Acc` lane count for token `S`
    ///
    /// # Parameters
    ///
    /// - `a`/`a_cs` - LHS panel base and column (depth) stride. A packed panel has
    ///   `a_cs == mr`. An unpacked column-major LHS has `a_cs == csa`. Rows are always
    ///   unit-stride
    /// - `b`/`b_rs`/`b_cs` - RHS panel base and strides. A packed panel uses
    ///   `(nr, 1)`
    /// - `c`/`rsc`/`csc` - output tile base and strides. The fast vector store needs
    ///   `rsc == 1` and a full tile. Otherwise the call drains through `scratch` and
    ///   copies back under arbitrary strides
    /// - `mr_eff`/`nr_eff` - live sub-tile size, at most `MR`/`NR` at an edge tile
    /// - `scratch` - at least [`SCRATCH_LEN`] accumulator elements of stack space
    ///
    /// # Notes
    ///
    /// A family overrides exactly 1 of the 2 microkernel methods. A non-fusing
    /// family overrides this plain method and inherits the default
    /// [`microkernel_epi`]. Examples are `IntGemm`, `IntGemmVnni`, and the
    /// open/closed test family. The default forwards straight here after a
    /// fail-closed `assert!(E::IS_IDENTITY)`. A fusing family instead overrides
    /// [`microkernel_epi`] to thread `E` through its own store, leaving this default
    /// body `unreachable!`. Examples are `FloatGemm`, the mixed families, and the
    /// requantizing integer families. `ComplexGemm` overrides both. Its
    /// [`microkernel_epi`] calls this method first and then sweeps the epilogue over
    /// the finished tile
    ///
    /// # Safety
    ///
    /// All pointers must be valid for the accesses implied by the strides and
    /// dimensions. The call must run inside the matching
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

    /// Compute one `MR x NR` tile and store the fused [`Epilogue`] `E` applied to each
    /// element. `row0` and `col0` are the tile's origin in the oriented problem frame,
    /// so a per-row or per-column bias can resolve its absolute base. `last_k` marks
    /// the final depth panel, the only one on which an `OUT_IS_ACC` family may apply
    /// the epilogue. Earlier panels store raw `Acc` partials, per the contract on
    /// [`KernelFamily::OUT_IS_ACC`]
    ///
    /// # Notes
    ///
    /// The default forwards to [`KernelFamily::microkernel`] and unconditionally
    /// asserts `E::IS_IDENTITY`, so it is correct only when the driver never fuses a
    /// real epilogue into this family. `IntGemm`, `IntGemmVnni`, and the open/closed
    /// test family rely on this default unedited. A family that supports fusion
    /// overrides this method to thread `E` through its own store. Examples are
    /// `FloatGemm`, the mixed and narrow families, the requantizing integer families,
    /// and `ComplexGemm`, which also calls [`KernelFamily::microkernel`] from its
    /// override
    ///
    /// # Safety
    ///
    /// As [`KernelFamily::microkernel`]. `epi`'s interior pointers must be valid for
    /// the problem's `m` and `n`
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
        // Fail closed: a family that does not override this method must never silently
        // drop a real epilogue. The check folds away at compile time when `E = Identity`,
        // since `E::IS_IDENTITY` is a const
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
