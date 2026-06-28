//! Kernel families (layer L1): the *operation-family* seam.
//!
//! The driver (layer L4) is generic over [`KernelFamily`], not over "do an FMA
//! on `T`". A family bundles everything that distinguishes one *kind* of GEMM
//! from another: the input/accumulator/output types, the pack layout, the
//! microkernel, and the epilogue. v1 ships exactly one family,
//! [`float::FloatGemm`]; complex and integer GEMM would arrive as *new families*
//! with the driver, packing framework, cache model and parallelism untouched —
//! that open/closed property is part of the architecture contract and is proven
//! by a test that declares a second trivial family.
//!
//! Note the tile geometry (`MR_REG`, `NR`) is **not** on this trait: it is a
//! pair of const generics chosen per `(family, ISA)` at the dispatch site, so a
//! new tile is a new instantiation, never a new type or macro.

use crate::scalar::Scalar;
use crate::simd::KernelSimd;

pub mod float;
pub mod mixed;

pub use float::FloatGemm;
pub use mixed::MixedGemm;

/// State of the `alpha` scale, precomputed once by the driver so the microkernel
/// never compares floats. `Zero` never reaches the microkernel (the driver
/// routes `alpha == 0` to a scale-only path), but is included for completeness.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AlphaStatus {
    /// `alpha == 0`.
    Zero,
    /// `alpha == 1` — no scaling needed.
    One,
    /// Any other `alpha`.
    Other,
}

/// State of the (effective) `beta` scale for the current depth slice.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BetaStatus {
    /// `beta == 0` — C is *not read* (it may be uninitialized / NaN).
    Zero,
    /// `beta == 1` — accumulate into C.
    One,
    /// Any other `beta`.
    Other,
}

/// Largest microkernel row count any family may request (`MR_REG * LANES`).
/// Bounds the stack scratch buffer used for partial / general-stride tiles.
pub const MAX_MR: usize = 64;
/// Largest microkernel column count any family may request.
pub const MAX_NR: usize = 32;
/// Capacity (in accumulator elements) of the per-call scratch tile.
pub const SCRATCH_LEN: usize = MAX_MR * MAX_NR;

/// An operation family: the seam that lets one driver serve many kinds of GEMM.
///
/// All methods are `unsafe` (raw pointers, target-feature codegen context). See
/// each method for its contract.
pub trait KernelFamily: Copy + Send + Sync + 'static {
    /// Left-hand input element type.
    type Lhs: Scalar;
    /// Right-hand input element type.
    type Rhs: Scalar;
    /// Accumulator element type (`Lhs::Acc` for the float family).
    type Acc: Scalar;
    /// Output element type.
    type Out: Scalar;

    /// Whether `Out` holds the accumulator at full `Acc` precision — i.e. whether a
    /// running partial sum can round-trip through `C` between depth (`kc`) panels
    /// **without losing precision**. `true` exactly when `Out == Acc` (every
    /// homogeneous float family; the default).
    ///
    /// The driver accumulates across K by reading and re-writing `C` once per `kc`
    /// panel (`beta == 1` after the first). For a homogeneous family that is exact.
    /// For a **mixed-precision** family (`Out = f16/bf16`, `Acc = f32`) it would
    /// round the running sum to 16 bits at every `kc` boundary, so such a family
    /// sets this `false` and the driver then uses `kc = k` (one panel — the whole
    /// contraction accumulates in the `f32` registers and rounds to `Out` once).
    const OUT_IS_ACC: bool = true;

    /// Pack an `mc × kc` LHS block into micropanel-major layout: a sequence of
    /// panels each `mr` rows tall, every panel stored column-by-column with `mr`
    /// contiguous rows per column (tail rows zero-filled). `mr == MR_REG*LANES`.
    ///
    /// # Safety
    /// `src` must be valid for the `mc × kc` region described by `rs`/`cs`, and
    /// `dst` valid for `ceil(mc/mr)*mr*kc` writes.
    unsafe fn pack_lhs(
        dst: *mut Self::Lhs,
        src: *const Self::Lhs,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    );

    /// Pack a `kc × nc` RHS block into micropanel-major layout: panels each `nr`
    /// columns wide, stored row-by-row with `nr` contiguous columns per row
    /// (tail columns zero-filled).
    ///
    /// # Safety
    /// `src` valid for the `kc × nc` region; `dst` valid for
    /// `ceil(nc/nr)*nr*kc` writes.
    unsafe fn pack_rhs(
        dst: *mut Self::Rhs,
        src: *const Self::Rhs,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    );

    /// Compute one `MR × NR` tile (`MR == MR_REG*LANES`, with `LANES` the **`Acc`**
    /// lane count) and apply the epilogue `C <- combine(alpha·A·B, beta·C)`.
    ///
    /// * `a`/`a_cs`: LHS panel base and column (depth) stride. For packed input
    ///   `a_cs == mr`; for adaptive (unpacked) column-major input `a_cs == csa`.
    ///   LHS rows are always unit-stride.
    /// * `b`/`b_rs`/`b_cs`: RHS panel base and strides. Packed: `(nr, 1)`.
    /// * `c`/`rsc`/`csc`: output tile strides. The fast vector epilogue requires
    ///   `rsc == 1` and a full tile; otherwise the call drains to `scratch` and
    ///   copies back with arbitrary strides.
    /// * `mr_eff`/`nr_eff`: live sub-tile dimensions (≤ MR / NR at edges).
    /// * `scratch`: at least [`SCRATCH_LEN`] accumulator elements.
    ///
    /// # Safety
    /// All pointers must be valid for the accesses implied by the strides and
    /// dimensions, and the call must run inside the matching [`crate::simd::Simd::vectorize`]
    /// context for `S`.
    #[allow(clippy::too_many_arguments)]
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
        S: KernelSimd<Self::Lhs, Self::Rhs, Self::Acc, Self::Out>;
}
