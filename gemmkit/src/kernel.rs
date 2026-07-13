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

#[cfg(feature = "complex")]
pub mod complex;
pub mod epilogue;
pub mod float;
#[cfg(feature = "int8")]
pub mod int;
#[cfg(feature = "half")]
pub mod mixed;

#[cfg(feature = "complex")]
pub use complex::ComplexGemm;
pub use epilogue::{Epilogue, Identity};
pub use float::FloatGemm;
#[cfg(feature = "int8")]
pub use int::{IntGemm, IntGemmQ, IntGemmVnni, IntGemmVnniQ};
#[cfg(feature = "half")]
pub use mixed::{Bf16DotGemm, MixedGemm};

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

    /// Whether a running partial sum can round-trip through `C` between `kc` panels
    /// without losing precision. `true` exactly when `Out == Acc` (every homogeneous
    /// float family; the default).
    ///
    /// The driver accumulates across K by re-reading and re-writing `C` once per `kc`
    /// panel (`beta == 1` after the first) — exact when `Out == Acc`. A mixed-precision
    /// family (`Out = f16/bf16`, `Acc = f32`) would round to 16 bits at every panel
    /// boundary, so it sets this `false`; the driver then uses `kc = k` (one panel, so
    /// the whole contraction accumulates in `f32` and rounds to `Out` once).
    const OUT_IS_ACC: bool = true;

    /// Force the driver to always pack the LHS / RHS, overriding the cost-based pack
    /// decision. Required when packing does more than a plain copy — the complex conj
    /// variants conjugate the operand during packing — so the transform always runs
    /// instead of the driver reading the operand in place. Default `false`.
    const FORCE_PACK_LHS: bool = false;
    /// See [`KernelFamily::FORCE_PACK_LHS`].
    const FORCE_PACK_RHS: bool = false;

    /// Packed-panel **depth** is rounded up to this multiple before the driver sizes and
    /// addresses the packed A/B buffers. `1` (the default) means no rounding — every
    /// homogeneous / widen family. A *dot-product* family that folds `Q` consecutive depth
    /// steps into one instruction (VNNI `vpdpbusd`: `Q = 4`; `vdpbf16ps`: `Q = 2`) sets it
    /// to `Q`, so each micropanel holds whole instruction-groups and the kernel reads
    /// `ceil(kc/Q)` of them.
    ///
    /// Contract: a family with `DEPTH_MULTIPLE = Q` MUST make [`pack_lhs`] / [`pack_rhs`]
    /// write a panel of `width · kc.next_multiple_of(Q)` elements, depth-padding the tail
    /// (pad value is family-specific — VNNI pads A with its `+128` bias of zero, B with
    /// zero). The driver strides packed panels by the same padded depth, keeping them in
    /// lockstep. `Q = 1` is the current no-padding behavior, so existing families are
    /// unaffected.
    ///
    /// [`pack_lhs`]: KernelFamily::pack_lhs
    /// [`pack_rhs`]: KernelFamily::pack_rhs
    const DEPTH_MULTIPLE: usize = 1;

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

    /// Compute one `MR × NR` tile (`MR == MR_REG*LANES`, `LANES` the `Acc` lane count)
    /// and apply the epilogue `C <- combine(alpha·A·B, beta·C)`.
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

    /// Compute one `MR × NR` tile and apply the fused [`Epilogue`] `E` to each stored
    /// element. `row0`/`col0` are the tile's origin in the **oriented** problem frame (so
    /// a per-row/per-col bias resolves its absolute base), and `last_k` is whether this is
    /// the final depth panel — the epilogue applies only then for `OUT_IS_ACC` families
    /// (intermediate panels store raw `Acc` partials; see [`KernelFamily`]).
    ///
    /// The **default** forwards to [`KernelFamily::microkernel`], ignoring the epilogue
    /// arguments: it is correct only for `E = Identity` (a `debug_assert` enforces this),
    /// which is exactly what the driver passes on every non-fused path — so `mixed`,
    /// `complex`, and the `i32`-out integer families need no edit and stay byte-for-byte
    /// unchanged. A family that supports fusion (the float family and the requantizing
    /// integer families) overrides this to thread `E` through its store.
    ///
    /// # Safety
    /// As [`KernelFamily::microkernel`]; `epi`'s interior pointers must be valid for the
    /// problem's `m`/`n`.
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
        // Fail closed in every build: a family that does not override this must never
        // silently drop a real epilogue. `E::IS_IDENTITY` is a const, so this folds away
        // entirely for `Identity` (the only in-crate caller) and becomes an unconditional
        // panic for any non-identity `E` reaching a non-overriding family.
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
