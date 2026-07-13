//! Fused epilogues (layer L1): the transform a family applies to each output element
//! as the microkernel stores it, instead of materializing the raw product and mapping
//! it in a second pass.
//!
//! The seam is the [`Epilogue`] trait, threaded through
//! [`crate::kernel::KernelFamily::microkernel_epi`]. Its central invariant is
//! **zero-cost identity**: with `E = Identity` every hook const-folds away
//! ([`Epilogue::IS_IDENTITY`]) and the monomorphized kernel is bit-identical to the
//! non-fused kernel, so plain `gemm`/`gemm_i8` are unchanged. The determinism contract
//! is stronger still: a fused GEMM is `gemm()` followed by a scalar map, **bit-for-bit**
//! for floats, because the fused engine routes every shape through the same kernel `gemm`
//! would — the general driver *and* the gemv / small-`m,n` / small-`k` special paths, each
//! fused too — blocking is epilogue-independent, and the epilogue is applied to the very
//! register the store would have written (see [`FusedEpi`]).
//!
//! Two built-ins ship in v1: [`FusedEpi`] (per-row/per-col bias + ReLU/LeakyReLU, via the
//! fast vector path — for `f32`/`f64` and, under `half`, `f16`/`bf16` where the bias/slope
//! widen exactly to `f32`, the transform applies in `f32`, and the single narrowing to the
//! output happens on store) and [`KRequantize`] (`i8 -> i8` quantized output, scalar map with
//! the exact round-half-to-even [`round_ne_f64`]).

use super::KernelFamily;
use super::float::FloatGemm;
use crate::parallel::Ptr;
use crate::scalar::Float;
use crate::simd::{KernelSimd, SimdOps};

/// A transform fused into the microkernel's store:
/// `C[r, c] <- apply(alpha·(A·B)[r, c] + beta·C[r, c], r, c)`, applied **exactly once**
/// per output element: on the final depth panel for `OUT_IS_ACC` families (the driver
/// passes `last_k`; intermediate panels store raw `Acc` partials per the contract at
/// [`KernelFamily`]), and unconditionally for `OUT_IS_ACC = false` families (`kc = k`, so
/// there is only one panel).
///
/// The two application paths — the fast vector [`Epilogue::apply_reg`] and the scalar
/// [`Epilogue::apply`] — MUST agree bit-for-bit under the same token (the
/// `accumulate_tile` edge-consistency discipline): a full column-major tile takes the
/// vector path, an edge/strided tile drains to scratch and takes the scalar path, and the
/// two must produce identical output.
pub trait Epilogue<Fam: KernelFamily>: Copy + Send + Sync {
    /// Compile-time identity marker: `true` => every kernel hook const-folds away and the
    /// monomorphization is bit-identical to the non-fused kernel.
    const IS_IDENTITY: bool = false;
    /// `true` iff [`Epilogue::apply_reg`] is implemented: this enables the fast vector store
    /// path. The contract is that [`Epilogue::apply_reg`] transforms the `Fam::Acc`-typed
    /// register and the family's store applies any narrowing **after** the transform —
    /// `FloatGemm` stores the register as-is (`Out == Acc`), while the mixed (`f16`/`bf16`)
    /// families narrow via `store_out` (`Out` narrower than `Acc = f32`). `false` routes every
    /// tile through the scratch/scalar path (correct for any tile shape), which is the right
    /// call for an epilogue whose vector form is not yet worth it (requantize).
    const VECTOR: bool = false;

    /// Scalar transform at absolute `(row, col)` in the **oriented** problem frame.
    ///
    /// # Safety
    /// Interior pointers (bias) must be valid for the problem's `m`/`n`; run inside the
    /// matching [`crate::simd::Simd::vectorize`] context.
    unsafe fn apply(&self, v: Fam::Acc, row: usize, col: usize) -> Fam::Out;

    /// Vector transform of `LANES` consecutive rows `[row, row + LANES)` of column `col`
    /// (the fast path is a full tile with `rsc == 1`, so the rows are unit-stride). It
    /// MUST agree with [`Epilogue::apply`] bit-for-bit under the same token. The default
    /// is unreachable — only a `VECTOR = true` epilogue overrides it (the `dot_accumulate`
    /// pattern).
    ///
    /// # Safety
    /// As [`Epilogue::apply`]; `simd` is the token whose [`crate::simd::Simd::vectorize`]
    /// context is active.
    #[inline(always)]
    unsafe fn apply_reg<S>(
        &self,
        _simd: S,
        _v: <S as SimdOps<Fam::Acc>>::Reg,
        _row: usize,
        _col: usize,
    ) -> <S as SimdOps<Fam::Acc>>::Reg
    where
        S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    {
        unreachable!("apply_reg requires VECTOR = true")
    }

    /// `true` iff [`Epilogue::apply_store`] is implemented: the **vector store-transform** for
    /// an `Out != Acc` epilogue (the requantize pattern), enabling the fast store path when the
    /// token is also requant-vector-capable ([`crate::simd::KernelSimd::REQUANT_VECTOR`]). This
    /// is orthogonal to [`Epilogue::VECTOR`] (which governs the float-style *in-register*
    /// `apply_reg` path, where `store_out` narrows *after* the transform): requantize keeps
    /// `VECTOR = false` (no in-register clamp seam) but sets `VECTOR_STORE = true`.
    const VECTOR_STORE: bool = false;

    /// Vector store-transform: read `LANES` consecutive-row `Acc` values from contiguous
    /// scratch at `src`, apply the full epilogue, and write `LANES` `Out` values to `dst` at
    /// **unit row stride** (the caller guarantees `rsc == 1` on this path). It MUST agree with
    /// [`Epilogue::apply`] bit-for-bit on the same values — the row tail, the strided-`C`
    /// drain, and the `k == 0` degenerate fill all take `apply`, and a single output mixes the
    /// two freely. The default is unreachable — only a `VECTOR_STORE = true` epilogue overrides
    /// it (the `apply_reg`/`dot_accumulate` seam pattern).
    ///
    /// # Safety
    /// `src` valid for `LANES` `Acc` reads, `dst` for `LANES` `Out` writes; interior pointers
    /// (bias) valid for the problem; `simd` is the token whose [`crate::simd::Simd::vectorize`]
    /// context is active.
    #[inline(always)]
    unsafe fn apply_store<S>(
        &self,
        _simd: S,
        _src: *const Fam::Acc,
        _dst: *mut Fam::Out,
        _row: usize,
        _col: usize,
    ) where
        S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    {
        unreachable!("apply_store requires VECTOR_STORE = true")
    }
}

/// The identity epilogue: the seam's zero-cost default. Every kernel hook is gated on
/// `!E::IS_IDENTITY`, so with `E = Identity` it const-folds to nothing and the kernel is
/// bit-identical to the non-fused one.
#[derive(Copy, Clone, Default)]
pub struct Identity;

impl<Fam: KernelFamily> Epilogue<Fam> for Identity {
    const IS_IDENTITY: bool = true;
    #[inline(always)]
    unsafe fn apply(&self, _: Fam::Acc, _: usize, _: usize) -> Fam::Out {
        // Kernels gate on `IS_IDENTITY`, so this is never reached.
        unreachable!("identity epilogue is never applied")
    }
}

/// Which axis a bias vector is indexed on: per output **row** (`m`) or per output
/// **column** (`n`). The dispatch layer flips this on an orientation swap.
#[derive(Copy, Clone)]
pub enum BiasDim {
    /// One bias value per output row (length `m`).
    PerRow,
    /// One bias value per output column (length `n`).
    PerCol,
}

/// A bias vector in the driver (oriented) frame. `Ptr` is the `Send + Sync` raw-pointer
/// shim (the bias slice outlives the call; the borrow is erased to `Ptr` in the call frame
/// where it is live).
#[derive(Copy, Clone)]
pub(crate) enum BiasSpec<T> {
    /// No bias.
    None,
    /// One value per output row (added to every column of that row).
    Row(Ptr<T>),
    /// One value per output column (added to every row of that column).
    Col(Ptr<T>),
}

/// The activation applied after the bias add.
#[derive(Copy, Clone)]
pub(crate) enum Act<T> {
    /// No activation.
    None,
    /// `max(v, 0)` (NaN maps to 0).
    Relu,
    /// `max(v, 0) + slope·min(v, 0)` (NaN maps to 0, −0 to +0).
    LeakyRelu(T),
}

/// The single runtime-composed float epilogue: bias (row / col / none) then activation
/// (none / ReLU / LeakyReLU). One monomorphization covers every combination — the enum
/// branches are ~2 predictable tests per tile, amortized over the `mr·nr·kc` FMA loop —
/// so the fused kernel is **not** multiplied by the number of epilogue kinds.
///
/// It is a `pub` type with crate-private fields (constructed only by the API layer), so it
/// can appear in the dispatch slot's function-pointer type without leaking its internals.
#[derive(Copy, Clone)]
pub struct FusedEpi<T> {
    pub(crate) bias: BiasSpec<T>,
    pub(crate) act: Act<T>,
}

// The bound is `Float<Acc = T> + PartialOrd` rather than the public `FusedScalar`: it keeps
// this kernel-layer file free of any dispatch-layer dependency, and it selects exactly the
// real floats — `Complex` implements `Float` but not `PartialOrd`, so it is excluded, and
// `f16`/`bf16` are not `Float`. The public API seals the surface with `FusedScalar`.
impl<T: Float<Acc = T> + PartialOrd> Epilogue<FloatGemm<T>> for FusedEpi<T> {
    const VECTOR: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => v + unsafe { *p.0.add(c) },
        };
        match self.act {
            Act::None => v,
            // NaN -> ZERO (`NaN > 0` is false), matching the vector `max(v, 0)`.
            Act::Relu => {
                if v > T::ZERO {
                    v
                } else {
                    T::ZERO
                }
            }
            // Exact scalar mirror of the vector form `max(v, 0) + s·min(v, 0)`: NaN -> 0,
            // −0.0 -> +0.0. Written as the identical composition so the fast and scratch
            // paths agree bit-for-bit.
            Act::LeakyRelu(s) => {
                let hi = if v > T::ZERO { v } else { T::ZERO };
                let lo = if v < T::ZERO { v } else { T::ZERO };
                hi + s * lo
            }
        }
    }

    #[inline(always)]
    unsafe fn apply_reg<S>(&self, s: S, v: S::Reg, r: usize, c: usize) -> S::Reg
    where
        S: KernelSimd<T, T, T, T>,
    {
        unsafe {
            let v = match self.bias {
                // Fast path is a full tile with `rsc == 1`, so the `LANES` rows at `r` are
                // consecutive in the bias vector.
                BiasSpec::None => v,
                BiasSpec::Row(p) => s.add(v, s.loadu(p.0.add(r))),
                BiasSpec::Col(p) => s.add(v, s.splat(*p.0.add(c))),
            };
            match self.act {
                Act::None => v,
                Act::Relu => s.max(v, s.zero()),
                Act::LeakyRelu(sl) => {
                    s.add(s.max(v, s.zero()), s.mul(s.splat(sl), s.min(v, s.zero())))
                }
            }
        }
    }
}

// Narrow-family (`f16`/`bf16`) blanket: bias/slope are the narrow type `N`, widened **exactly**
// to `f32` (both are a subset of `f32`); the epilogue applies in `f32` to the accumulator, then
// the single round-to-nearest-even narrowing to `N` happens on store. It covers `MixedGemm<f16>`,
// `MixedGemm<bf16>`, and `Bf16DotGemm` at once (all `Lhs = Rhs = Out = N`, `Acc = f32`). It cannot
// overlap the `FloatGemm` impl above: `f32`/`f64` are not `NarrowFloat`.
//
// This is **more** precise than `gemm()` then a separate map (which would round to `N`, widen
// back, and round again) — and therefore NOT bitwise-equal to `gemm`-then-map (unlike `f32`/`f64`,
// whose every-shape bitwise contract is unchanged). Within the fused run, the vector fast path and
// the scalar/scratch path agree bit-for-bit: both compute `act(bias(v))` in `f32` and round
// exactly once (`apply` via [`crate::scalar::NarrowFloat::narrow`], `apply_reg` via `store_out`),
// and widening is exact — the `accumulate_tile` edge-consistency contract.
#[cfg(feature = "half")]
impl<N, Fam> Epilogue<Fam> for FusedEpi<N>
where
    N: crate::scalar::NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
{
    const VECTOR: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: f32, r: usize, c: usize) -> N {
        // Bias add in `f32` (the narrow bias value widened exactly).
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { (*p.0.add(r)).widen() },
            BiasSpec::Col(p) => v + unsafe { (*p.0.add(c)).widen() },
        };
        // Activation in `f32`, the EXACT same scalar forms as the `FloatGemm` impl above.
        let v = match self.act {
            Act::None => v,
            // NaN -> 0 (`NaN > 0` is false), matching the vector `max(v, 0)`.
            Act::Relu => {
                if v > 0.0 {
                    v
                } else {
                    0.0
                }
            }
            // Exact scalar mirror of the vector `max(v, 0) + s·min(v, 0)`: NaN -> 0, −0.0 -> +0.0.
            Act::LeakyRelu(s) => {
                let hi = if v > 0.0 { v } else { 0.0 };
                let lo = if v < 0.0 { v } else { 0.0 };
                hi + s.widen() * lo
            }
        };
        // The single round-to-nearest-even narrowing is the epilogue's job here — the kernel's
        // scratch path stores exactly what this returns.
        N::narrow(v)
    }

    #[inline(always)]
    unsafe fn apply_reg<S>(&self, s: S, v: S::Reg, r: usize, c: usize) -> S::Reg
    where
        S: KernelSimd<N, N, f32, N>,
    {
        unsafe {
            let v = match self.bias {
                // Fast path is a full tile with `rsc == 1`, so the `LANES` rows at `r` are
                // consecutive narrow bias values — widen-load them into one `f32` register.
                BiasSpec::None => v,
                BiasSpec::Row(p) => s.add(v, s.load_lhs(p.0.add(r))),
                BiasSpec::Col(p) => s.add(v, s.splat((*p.0.add(c)).widen())),
            };
            // Same register forms as the `FloatGemm` impl; returns the `f32` register — the
            // family's `store_out` performs the single narrowing store.
            match self.act {
                Act::None => v,
                Act::Relu => s.max(v, s.zero()),
                Act::LeakyRelu(sl) => s.add(
                    s.max(v, s.zero()),
                    s.mul(s.splat(sl.widen()), s.min(v, s.zero())),
                ),
            }
        }
    }
}

// Complex-family fused epilogue: **bias only**. `ComplexGemm<T, CA, CB>` is a distinct family
// type from `FloatGemm` / the narrow families / `KRequantize`, so this impl cannot overlap any of
// them (no coherence conflict). It deliberately has **no activation**: an ordering-based activation
// (ReLU / LeakyReLU) is mathematically undefined on complex numbers, so the complex public entry
// (`gemm_cplx_fused`) constructs only `Act::None` — the other `Act` arms are therefore
// `unreachable!`.
//
// `VECTOR` stays `false` (the default): every complex tile is stored by the SoA kernel's own
// scalar alpha/beta epilogue (inside the L0 `cplx_microkernel` seam, which must not depend on this
// L1 trait), and this `apply` rides the tile-local in-place post-pass that
// `ComplexGemm::microkernel_epi` runs on the final depth panel. So there is no `apply_reg` /
// `apply_store` here — only the scalar `apply`. Because the kernel first stores exactly the bits
// plain `gemm_cplx` would, the post-pass makes `gemm_cplx_fused` bitwise-identical to `gemm_cplx`
// then the same element-wise bias add.
#[cfg(feature = "complex")]
impl<T, const CA: bool, const CB: bool> Epilogue<crate::kernel::ComplexGemm<T, CA, CB>>
    for FusedEpi<T>
where
    T: crate::scalar::ComplexFloat,
{
    #[inline(always)]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        // Bias add only (the fast path is a full tile, so the `r`/`c` coordinate resolves the
        // per-row / per-col base directly). Complex addition is `num_complex`'s `Add`, the same
        // operation the `gemm_cplx`-then-map oracle applies — hence bitwise-identical.
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => v + unsafe { *p.0.add(c) },
        };
        match self.act {
            Act::None => v,
            // The complex entry never constructs an activation (undefined on `C`).
            Act::Relu | Act::LeakyRelu(_) => {
                unreachable!("complex fused epilogue has no activation")
            }
        }
    }
}

/// The output domain of the requantizing [`KRequantize`] epilogue: the inclusive clamp
/// bounds and the final narrowing of an already-clamped `i64` to the output byte. Implemented
/// for the two quantized output types — `i8` (`[-128, 127]`) and `u8` (`[0, 255]`) — so one
/// `KRequantize` impl and one requant family serve both. The narrowing `from_clamped` is only
/// ever handed a value already in `[LO, HI]`, so the `as` cast is a plain reinterpret of the
/// low byte (no saturation), matching the vector store's low-byte write.
#[cfg(feature = "int8")]
pub(crate) trait QuantOut: crate::scalar::Scalar {
    /// Inclusive clamp bounds of the output domain.
    const LO: i32;
    /// Inclusive clamp bounds of the output domain.
    const HI: i32;
    /// Truncate an already-clamped `i64` (in `[LO, HI]`) to the output byte.
    fn from_clamped(q: i64) -> Self;
}

#[cfg(feature = "int8")]
impl QuantOut for i8 {
    const LO: i32 = -128;
    const HI: i32 = 127;
    #[inline(always)]
    fn from_clamped(q: i64) -> Self {
        q as i8
    }
}

#[cfg(feature = "int8")]
impl QuantOut for u8 {
    const LO: i32 = 0;
    const HI: i32 = 255;
    #[inline(always)]
    fn from_clamped(q: i64) -> Self {
        q as u8
    }
}

/// The requantizing epilogue: `C[r, c] = clamp(zp + round_ne(scale·(acc + bias)), LO, HI)`,
/// with an optional per-row / per-col `i32` bias joined **in integer** before the single f64
/// rounding. The clamp band `[LO, HI]` is the output domain, chosen per output type by
/// [`QuantOut`]: `i8` → `[-128, 127]` (signed) and `u8` → `[0, 255]` (the ONNX-QLinearMatMul
/// activation output). The struct itself carries no `Out` — one value drives both, the domain
/// coming from the `Fam::Out` the [`Epilogue`] impl is monomorphized for.
///
/// Every tile drains its `i32` accumulators to scratch (vectorized), then maps each element to
/// the output byte. The map itself is split: on requant-vector-capable tokens (x86 —
/// [`KRequantize::apply_store`] via [`crate::simd::KernelSimd::requant_store`]) a full lane-run is
/// vectorized in `f64`; the row tail, a strided `C`, the `k == 0` degenerate fill, and non-vector
/// ISAs (scalar / NEON / wasm) take the scalar [`KRequantize::apply`]. The two are **bit-identical**
/// (the `requant_store` equivalence contract), so a single matrix mixes them freely. Either way
/// this beats the unfused flow, which materializes a full `m·n` `i32` C.
#[cfg(feature = "int8")]
#[derive(Copy, Clone)]
pub(crate) struct KRequantize {
    pub scale: f32,
    pub zp: i32,
    pub bias: Ptr<i32>,
    pub has_bias: bool,
    /// Bias axis in the driver (oriented) frame; the dispatch layer flips it on a swap.
    pub bias_dim: BiasDim,
}

#[cfg(feature = "int8")]
impl<O: QuantOut, Fam: KernelFamily<Acc = i32, Out = O>> Epilogue<Fam> for KRequantize {
    // `VECTOR = false`: requantize has no float-style in-register `apply_reg` path (`Out != Acc`,
    // and the clamp/round is not a `store_out` narrowing). Its vector form is the store-transform
    // `apply_store` below, gated by `VECTOR_STORE` instead.
    const VECTOR: bool = false;
    const VECTOR_STORE: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: i32, r: usize, c: usize) -> O {
        let b = if self.has_bias {
            let idx = match self.bias_dim {
                BiasDim::PerRow => r,
                BiasDim::PerCol => c,
            };
            unsafe { *self.bias.0.add(idx) }
        } else {
            0
        };
        // `i32` and `f32` are exactly representable in `f64`, so this is ONE rounding step
        // total. `zp` joins in integer (round-half-to-even is not shift-invariant, so it
        // must not be folded into the pre-round expression). The `f64 -> i64` cast
        // saturates, and `saturating_add`/`clamp` keep the whole map panic-free and
        // bit-exact across every ISA. `O::LO`/`O::HI` select the output band (`i8` or `u8`);
        // `from_clamped` then reinterprets the already-clamped low byte.
        let scaled = round_ne_f64(f64::from(v.wrapping_add(b)) * f64::from(self.scale));
        let q = (scaled as i64).saturating_add(i64::from(self.zp));
        O::from_clamped(q.clamp(i64::from(O::LO), i64::from(O::HI)))
    }

    #[inline(always)]
    unsafe fn apply_store<S>(&self, simd: S, src: *const i32, dst: *mut O, row: usize, col: usize)
    where
        S: KernelSimd<Fam::Lhs, Fam::Rhs, i32, O>,
    {
        unsafe {
            // `LANES` consecutive-row `i32` accumulators from contiguous scratch.
            let v = simd.loadu(src);
            // Bias add **in integer** — SIMD `i32` add is wrapping (`paddd`), matching `apply`'s
            // `wrapping_add`. The fast path is a full tile with `rsc == 1`, so the `LANES` rows at
            // `row` are consecutive `PerRow` bias values (one aligned load); a `PerCol` bias is a
            // single value broadcast across the column. No bias => `v` unchanged.
            let v = if self.has_bias {
                match self.bias_dim {
                    BiasDim::PerRow => simd.add(v, simd.loadu(self.bias.0.add(row))),
                    BiasDim::PerCol => simd.add(v, simd.splat(*self.bias.0.add(col))),
                }
            } else {
                v
            };
            // The vector `f64` map, bit-identical to `apply` (the `requant_store` contract):
            // scale widens `f32 -> f64` exactly; `O::LO`/`O::HI` are the output clamp band.
            //
            // The `*mut O -> *mut i8` cast is sound and value-correct: `requant_store` writes the
            // LOW BYTE of each pre-clamped lane, and for any `x` in `[-128, 255]` the bytes of
            // `(x as i8)` and `(x as u8)` are identical — so the byte written equals the scalar
            // `O::from_clamped` result whether `O = i8` or `O = u8`.
            simd.requant_store(
                dst as *mut i8,
                v,
                f64::from(self.scale),
                self.zp,
                O::LO,
                O::HI,
            );
        }
    }
}

/// Round-half-to-even of a finite `f64`, `no_std`-safe (`f64::round_ties_even` lives in
/// `std`; this crate is `no_std` without `std`). The classic `2^52` trick: for
/// `|x| < 2^52`, adding then removing the `2^52` magnitude constant snaps `x` to the
/// nearest integer under the default round-to-nearest-even mode; `|x| >= 2^52` is already
/// integral. Uses only comparisons and `f64` add/sub, so it needs no `std` float methods.
#[cfg(feature = "int8")]
#[inline(always)]
pub(crate) fn round_ne_f64(x: f64) -> f64 {
    const C: f64 = 4503599627370496.0; // 2^52
    // NaN or already integral (`|x| >= 2^52`) pass through unchanged. `f64::is_nan` is `core`
    // (only `round_ties_even` is `std`); `|x| >= 2^52` is spelled as two comparisons to avoid
    // the `std`-only `f64::abs`.
    if x.is_nan() || x >= C || x <= -C {
        x
    } else if x >= 0.0 {
        (x + C) - C
    } else {
        (x - C) + C
    }
}
