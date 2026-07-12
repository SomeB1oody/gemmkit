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
//! for floats, because blocking is epilogue-independent and the epilogue is applied to
//! the very register the store would have written (see [`FusedEpi`]).
//!
//! Two built-ins ship in v1: [`FusedEpi`] (per-row/per-col bias + ReLU/LeakyReLU for
//! `f32`/`f64`, via the fast vector path) and [`KRequantize`] (`i8 -> i8` quantized
//! output, scalar map with the exact round-half-to-even [`round_ne_f64`]).

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
    /// `true` iff [`Epilogue::apply_reg`] is implemented **and** `Fam::Out == Fam::Acc`
    /// (a documented contract): this enables the fast vector store path. `false` routes
    /// every tile through the scratch/scalar path (correct for any tile shape), which is
    /// the right call for an epilogue whose vector form is not yet worth it (requantize).
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

/// The `i8 -> i8` requantizing epilogue: `C[r, c] = clamp(zp + round_ne(scale·(acc + bias)),
/// -128, 127)`, with an optional per-row / per-col `i32` bias joined **in integer** before
/// the single f64 rounding. `VECTOR = false` in v1, so every tile drains to `i32` scratch
/// (vectorized) and only this map is scalar — still a strict win over the unfused flow,
/// which materializes a full `m·n` `i32` C.
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
impl<Fam: KernelFamily<Acc = i32, Out = i8>> Epilogue<Fam> for KRequantize {
    const VECTOR: bool = false; // scratch/scalar path for every tile in v1

    #[inline(always)]
    unsafe fn apply(&self, v: i32, r: usize, c: usize) -> i8 {
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
        // bit-exact across every ISA.
        let scaled = round_ne_f64(f64::from(v.wrapping_add(b)) * f64::from(self.scale));
        let q = (scaled as i64).saturating_add(i64::from(self.zp));
        q.clamp(-128, 127) as i8
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
