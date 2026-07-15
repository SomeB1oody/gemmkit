//! Fused epilogues (layer L1): the transform a family applies to each output element
//! as the microkernel stores it, instead of materializing the raw product and mapping
//! it in a 2nd pass
//!
//! The seam is the [`Epilogue`] trait, threaded through
//! [`crate::kernel::KernelFamily::microkernel_epi`]. Its central invariant is
//! **zero-cost identity**: with `E = Identity` every hook const-folds away
//! ([`Epilogue::IS_IDENTITY`]) and the monomorphized kernel is bit-identical to the
//! non-fused kernel, so plain `gemm`/`gemm_i8` are unchanged. The determinism contract
//! is stronger still: a fused GEMM is `gemm()` followed by a scalar map, **bit-for-bit**
//! for floats, because the fused engine routes every shape through the same kernel `gemm`
//! would (the general driver and the gemv / small-`m,n` / small-`k` special paths, each
//! fused too): blocking is epilogue-independent, and the epilogue is applied to the very
//! register the store would have written (see [`FusedEpi`])
//!
//! 2 built-in epilogues ship: [`FusedEpi`] (per-row/per-col bias + ReLU/LeakyReLU, via
//! the fast vector path: for `f32`/`f64` and, under `half`, `f16`/`bf16` where the
//! bias/slope widen exactly to `f32`, the transform applies in `f32`, and the single
//! narrowing to the output happens on store) and `KRequantize` (i32 accumulator ->
//! quantized i8/u8 output, scalar map with the exact round-half-to-even `round_ne_f64`)

use super::KernelFamily;
#[cfg(feature = "epilogue")]
use super::float::FloatGemm;
#[cfg(feature = "epilogue")]
use crate::parallel::Ptr;
#[cfg(feature = "epilogue")]
use crate::scalar::Float;
use crate::simd::{KernelSimd, SimdOps};

/// A transform fused into the microkernel's store:
/// `C[r, c] <- apply(alpha*(A*B)[r, c] + beta*C[r, c], r, c)`, applied **exactly once**
/// per output element: on the final depth panel for `OUT_IS_ACC` families (the driver
/// passes `last_k`; intermediate panels store raw `Acc` partials per the contract at
/// [`KernelFamily`]), and unconditionally for `OUT_IS_ACC = false` families (`kc = k`, so
/// there is only one panel)
///
/// The 2 application paths (the fast vector [`Epilogue::apply_reg`] and the scalar
/// [`Epilogue::apply`]) MUST agree bit-for-bit under the same token (the
/// `accumulate_tile` edge-consistency discipline): a full column-major tile takes the
/// vector path, an edge/strided tile drains to scratch and takes the scalar path, and the
/// 2 must produce identical output
pub trait Epilogue<Fam: KernelFamily>: Copy + Send + Sync {
    /// Compile-time identity marker: `true` => every kernel hook const-folds away and the
    /// monomorphization is bit-identical to the non-fused kernel
    const IS_IDENTITY: bool = false;
    /// `true` iff [`Epilogue::apply_reg`] is implemented: this enables the fast vector store
    /// path. The contract is that [`Epilogue::apply_reg`] transforms the `Fam::Acc`-typed
    /// register and the family's store applies any narrowing after the transform:
    /// `FloatGemm` stores the register as-is (`Out == Acc`), while the mixed (`f16`/`bf16`)
    /// families narrow via `store_out` (`Out` narrower than `Acc = f32`). `false` routes every
    /// tile through the scratch/scalar path (correct for any tile shape), which is the right
    /// call for an epilogue whose vector form is not yet worth it (requantize)
    const VECTOR: bool = false;

    /// Scalar transform at absolute `(row, col)` in the oriented problem frame
    ///
    /// # Safety
    /// Interior pointers (bias) must be valid for the problem's `m`/`n`; run inside the
    /// matching [`crate::simd::Simd::vectorize`] context
    unsafe fn apply(&self, v: Fam::Acc, row: usize, col: usize) -> Fam::Out;

    /// Vector transform of `LANES` consecutive rows `[row, row + LANES)` of column `col`
    /// (the fast path is a full tile with `rsc == 1`, so the rows are unit-stride). It
    /// MUST agree with [`Epilogue::apply`] bit-for-bit under the same token. The default
    /// is unreachable: only a `VECTOR = true` epilogue overrides it (the `dot_accumulate`
    /// pattern)
    ///
    /// # Safety
    /// As [`Epilogue::apply`]; `simd` is the token whose [`crate::simd::Simd::vectorize`]
    /// context is active
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

    /// `true` iff [`Epilogue::apply_store`] is implemented: the vector store-transform for
    /// an `Out != Acc` epilogue (the requantize pattern), enabling the fast store path when the
    /// token is also requant-vector-capable ([`crate::simd::KernelSimd::REQUANT_VECTOR`]). This
    /// is orthogonal to [`Epilogue::VECTOR`] (which governs the float-style in-register
    /// `apply_reg` path, where `store_out` narrows after the transform): requantize keeps
    /// `VECTOR = false` (no in-register clamp seam) but sets `VECTOR_STORE = true`
    const VECTOR_STORE: bool = false;

    /// Vector store-transform: read `LANES` consecutive-row `Acc` values from contiguous
    /// scratch at `src`, apply the full epilogue, and write `LANES` `Out` values to `dst` at
    /// unit row stride (the caller guarantees `rsc == 1` on this path). It MUST agree with
    /// [`Epilogue::apply`] bit-for-bit on the same values: the row tail, the strided-`C`
    /// drain, and the `k == 0` degenerate fill all take `apply`, and a single output mixes the
    /// 2 freely. The default is unreachable: only a `VECTOR_STORE = true` epilogue overrides
    /// it (the `apply_reg`/`dot_accumulate` seam pattern)
    ///
    /// # Safety
    /// `src` valid for `LANES` `Acc` reads, `dst` for `LANES` `Out` writes; interior pointers
    /// (bias) valid for the problem; `simd` is the token whose [`crate::simd::Simd::vectorize`]
    /// context is active
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
/// bit-identical to the non-fused one
#[derive(Copy, Clone, Default)]
pub struct Identity;

impl<Fam: KernelFamily> Epilogue<Fam> for Identity {
    const IS_IDENTITY: bool = true;
    #[inline(always)]
    unsafe fn apply(&self, _: Fam::Acc, _: usize, _: usize) -> Fam::Out {
        // Kernels gate on `IS_IDENTITY`, so this is never reached
        unreachable!("identity epilogue is never applied")
    }
}

/// Which axis a bias vector is indexed on: per output row (`m`) or per output
/// column (`n`). The dispatch layer flips this on an orientation swap
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub enum BiasDim {
    /// 1 bias value per output row (length `m`)
    PerRow,
    /// 1 bias value per output column (length `n`)
    PerCol,
}

/// A bias vector in the driver (oriented) frame. `Ptr` is the `Send + Sync` raw-pointer
/// shim (the bias slice outlives the call; the borrow is erased to `Ptr` in the call frame
/// where it is live)
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub(crate) enum BiasSpec<T> {
    /// No bias
    None,
    /// 1 value per output row (added to every column of that row)
    Row(Ptr<T>),
    /// 1 value per output column (added to every row of that column)
    Col(Ptr<T>),
}

/// An output scale in the driver (oriented) frame: a single per-tensor value, or a per-row /
/// per-col `f32` vector (the per-channel quantized-inference convention). Mirrors [`BiasSpec`];
/// the dispatch layer flips the per-row / per-col axis on an orientation swap, in lockstep with
/// the bias axis (they are the same user axis). `Ptr` is the `Send + Sync` raw-pointer shim (the
/// scale slice outlives the call; the borrow is erased to `Ptr` in the live call frame)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) enum ScaleSpec {
    /// 1 per-tensor scale (applied to every element)
    Tensor(f32),
    /// 1 scale per output row (applied to every column of that row)
    Row(Ptr<f32>),
    /// 1 scale per output column (applied to every row of that column)
    Col(Ptr<f32>),
}

/// The activation applied after the bias add
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub(crate) enum Act<T> {
    /// No activation
    None,
    /// `max(v, 0)` (NaN maps to 0)
    Relu,
    /// `max(v, 0) + slope*min(v, 0)` (NaN maps to 0, -0 to +0)
    LeakyRelu(T),
}

/// The single runtime-composed float epilogue: bias (row / col / none) then activation
/// (none / ReLU / LeakyReLU). One monomorphization covers every combination (the enum
/// branches are ~2 predictable tests per tile, amortized over the `mr*nr*kc` FMA loop), so
/// the fused kernel is not multiplied by the number of epilogue kinds
///
/// It is a `pub` type with crate-private fields (constructed only by the API layer), so it
/// can appear in the dispatch slot's function-pointer type without leaking its internals
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub struct FusedEpi<T> {
    pub(crate) bias: BiasSpec<T>,
    pub(crate) act: Act<T>,
}

// The bound is `Float<Acc = T> + PartialOrd` rather than the public `FusedScalar`: it keeps
// this kernel-layer file free of any dispatch-layer dependency, and it selects exactly the
// real floats: `Complex` implements `Float` but not `PartialOrd`, so it is excluded, and
// `f16`/`bf16` are not `Float`. The public API seals the surface with `FusedScalar`
#[cfg(feature = "epilogue")]
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
            // NaN -> ZERO (`NaN > 0` is false), matching the vector `max(v, 0)`
            Act::Relu => {
                if v > T::ZERO {
                    v
                } else {
                    T::ZERO
                }
            }
            // Exact scalar mirror of the vector form `max(v, 0) + s*min(v, 0)`: NaN -> 0,
            // -0.0 -> +0.0. Written as the identical composition so the fast and scratch
            // paths agree bit-for-bit
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
                // consecutive in the bias vector
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

/// A user-defined per-element epilogue: the closure `f` applied to each stored output element at
/// its final value, `C[r, c] <- f(alpha*(A*B)[r, c] + beta*C[r, c], r, c)`, with `(r, c)` in the
/// **user** frame of `C`. The public [`crate::gemm_map`] entry (feature `epilogue`, `f32`/`f64`)
/// lowers to this
///
/// It applies the closure **scalar, per output element** (one indirect call per element, amortized
/// by the `O(k)` FLOPs, one monomorphization per `(T, ISA)` not per closure - the reason the seam is
/// a borrowed trait object rather than a generic `F`), but it sets `VECTOR = true` so the kernel
/// takes the **same path selection plain `gemm` does**: the vector fast path for a full column-major
/// tile, the scratch/scalar path for an edge/strided tile (the `E::IS_IDENTITY || E::VECTOR` guard is
/// identical for `Identity` and this epilogue, so `gemm` and `gemm_map` route every element the same
/// way). The value handed to `f` is therefore **bitwise** the value plain `gemm` writes on *every*
/// path - crucially the fast path's *fused* `beta*C + alpha*AB` store, which the scalar path's
/// *unfused* [`crate::scalar::Float::mul_add`] does **not** reproduce for `beta != 0, 1` (so a
/// scratch-only `VECTOR = false` epilogue would diverge from `gemm` by 1 ULP there). On the fast path
/// [`Epilogue::apply_reg`] drains the register to a stack buffer and calls the same scalar `apply` on
/// each lane, so it agrees with the scratch path bit-for-bit - hence `gemm_map` is `gemm()` then the
/// per-element `f`, bit-for-bit, for every shape and route
///
/// The closure is a shared reference `&'u (dyn Fn + Sync)`, **not** erased to a `Ptr` shim: the
/// reference is `Copy` and its referent is `Sync`, so `MapEpi` is `Copy + Send + Sync` (the
/// [`Epilogue`] supertrait bounds) with **no** `'static` bound, and it is captured by value into the
/// scoped parallel workers (a blocking rayon `for_each`, which joins before the borrow ends), so a
/// non-`'static` borrow is sound - a closure may capture its environment by reference
///
/// `swapped` restores the user frame for `f`: the driver / small-`m,n` / small-`k` routes run in the
/// **oriented** frame (a row-major-ish C makes the engine compute `C^T = B^T*A^T`, swapping
/// `m<->n`), so when the orientation swap fired their `(row, col)` are transposed and `apply` flips
/// them back to `(col, row)` before calling `f`. gemv routes before orientation, so it leaves
/// `swapped = false` and its own `swap_rc` already yields user-frame coordinates. (This is the
/// closure analogue of [`FusedEpi`]'s per-axis bias flip: relabelling an axis suffices for a bias,
/// but a coordinate-dependent closure needs the full `(r, c)` transpose)
///
/// Like [`FusedEpi`] it is a `pub` type with crate-private fields (constructed only by the API
/// layer, from a borrowed closure), so it can appear in the dispatch slot's function-pointer type
/// and the `MapScalar` dispatch methods without leaking its internals or being externally
/// constructible
#[cfg(feature = "epilogue")]
pub struct MapEpi<'u, T> {
    /// The user closure `f(value, row, col) -> value`, borrowed for the call. The `+ Sync` makes
    /// the shared reference `Send + Sync`, so `MapEpi` threads through the scoped workers
    pub(crate) f: &'u (dyn Fn(T, usize, usize) -> T + Sync),
    /// Whether the orientation swap fired: then `apply` transposes `(row, col)` to restore the user
    /// frame for `f`
    pub(crate) swapped: bool,
}

// `#[derive(Copy, Clone)]` would demand `T: Copy`/`T: Clone`; `MapEpi`'s only fields (a shared
// reference and a `bool`) are `Copy` for **any** `T`, so implement both by hand without that bound
#[cfg(feature = "epilogue")]
impl<T> Copy for MapEpi<'_, T> {}
#[cfg(feature = "epilogue")]
impl<T> Clone for MapEpi<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Real-float (`f32`/`f64`) only, matching the public `MapScalar` seal: the narrow types would
// double-round a `T`-domain closure after the `f32` accumulate, and complex/int have no `apply` seam
// wired. `Float<Acc = T>` selects exactly `f32`/`f64` (the same reasoning as the `FusedEpi` impl)
#[cfg(feature = "epilogue")]
impl<T: Float<Acc = T>> Epilogue<FloatGemm<T>> for MapEpi<'_, T> {
    // `VECTOR = true` so the kernel keeps the fast vector path for a full column-major tile (the exact
    // path plain `gemm` takes), then applies the closure per-lane in `apply_reg`. This matters for the
    // *store* rounding: the fast path fuses `beta*C + alpha*AB` (hardware FMA), which the scalar
    // path's unfused `Float::mul_add` does not reproduce for `beta != 0, 1`; forcing scratch would
    // therefore diverge from plain `gemm` by 1 ULP. The closure itself is still applied scalar,
    // per element (no vectorized closure)
    const VECTOR: bool = true;

    #[inline]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        // Restore the user frame for the closure: the oriented routes transposed `(r, c)` on a swap
        if self.swapped {
            (self.f)(v, c, r)
        } else {
            (self.f)(v, r, c)
        }
    }

    #[inline]
    unsafe fn apply_reg<S>(&self, s: S, v: S::Reg, r: usize, c: usize) -> S::Reg
    where
        S: KernelSimd<T, T, T, T>,
    {
        unsafe {
            let lanes = <S as SimdOps<T>>::LANES;
            debug_assert!(
                lanes <= MAP_REG_LANES,
                "map apply_reg buffer holds MAP_REG_LANES lanes"
            );
            // Drain the `LANES` consecutive rows `[r, r + lanes)` at column `c` to a stack buffer,
            // apply the scalar `apply` (identical coordinate transform) to each, and reload. `v` is
            // the fast-path *fused* `beta*C + alpha*AB` store - the exact bits plain `gemm` writes -
            // so `f` sees the plain-`gemm` store value, and this per-lane form agrees with the
            // scratch-path `apply` bit-for-bit (same closure, same coordinates, exact drain/reload)
            let mut buf = [T::ZERO; MAP_REG_LANES];
            s.storeu(buf.as_mut_ptr(), v);
            for (l, slot) in buf.iter_mut().enumerate().take(lanes) {
                *slot = self.apply(*slot, r + l, c);
            }
            s.loadu(buf.as_ptr())
        }
    }
}

/// Stack-buffer width for [`MapEpi::apply_reg`]'s per-lane drain: an upper bound on any float
/// [`crate::simd::SimdOps`] lane count (`f32` on AVX-512 is the widest at 16)
#[cfg(feature = "epilogue")]
const MAP_REG_LANES: usize = 16;

// Narrow-family (`f16`/`bf16`) blanket: bias/slope are the narrow type `N`, widened exactly
// to `f32` (both are a subset of `f32`); the epilogue applies in `f32` to the accumulator, then
// the single round-to-nearest-even narrowing to `N` happens on store. It covers `MixedGemm<f16>`,
// `MixedGemm<bf16>`, and `Bf16DotGemm` at once (all `Lhs = Rhs = Out = N`, `Acc = f32`). It cannot
// overlap the `FloatGemm` impl above: `f32`/`f64` are not `NarrowFloat`
//
// This is more precise than `gemm()` then a separate map (which would round to `N`, widen
// back, and round again), and therefore NOT bitwise-equal to `gemm`-then-map (unlike `f32`/`f64`,
// whose every-shape bitwise contract is unchanged). Within the fused run, the vector fast path and
// the scalar/scratch path agree bit-for-bit: both compute `act(bias(v))` in `f32` and round
// exactly once (`apply` via `crate::scalar::NarrowFloat::narrow`, `apply_reg` via `store_out`),
// and widening is exact: the `accumulate_tile` edge-consistency contract
#[cfg(all(feature = "half", feature = "epilogue"))]
impl<N, Fam> Epilogue<Fam> for FusedEpi<N>
where
    N: crate::scalar::NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
{
    const VECTOR: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: f32, r: usize, c: usize) -> N {
        // Bias add in `f32` (the narrow bias value widened exactly)
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { (*p.0.add(r)).widen() },
            BiasSpec::Col(p) => v + unsafe { (*p.0.add(c)).widen() },
        };
        // Activation in `f32`, the EXACT same scalar forms as the `FloatGemm` impl above
        let v = match self.act {
            Act::None => v,
            // NaN -> 0 (`NaN > 0` is false), matching the vector `max(v, 0)`
            Act::Relu => {
                if v > 0.0 {
                    v
                } else {
                    0.0
                }
            }
            // Exact scalar mirror of the vector `max(v, 0) + s*min(v, 0)`: NaN -> 0, -0.0 -> +0.0
            Act::LeakyRelu(s) => {
                let hi = if v > 0.0 { v } else { 0.0 };
                let lo = if v < 0.0 { v } else { 0.0 };
                hi + s.widen() * lo
            }
        };
        // The single round-to-nearest-even narrowing is the epilogue's job here: the kernel's
        // scratch path stores exactly what this returns
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
                // consecutive narrow bias values: widen-load them into one `f32` register
                BiasSpec::None => v,
                BiasSpec::Row(p) => s.add(v, s.load_lhs(p.0.add(r))),
                BiasSpec::Col(p) => s.add(v, s.splat((*p.0.add(c)).widen())),
            };
            // Same register forms as the `FloatGemm` impl; returns the `f32` register: the
            // family's `store_out` performs the single narrowing store
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

// Complex-family fused epilogue: bias only. `ComplexGemm<T, CA, CB>` is a distinct family
// type from `FloatGemm` / the narrow families / `KRequantize`, so this impl cannot overlap any of
// them (no coherence conflict). It deliberately has no activation: an ordering-based activation
// (ReLU / LeakyReLU) is mathematically undefined on complex numbers, so the complex public entry
// (`gemm_cplx_fused`) constructs only `Act::None`; the other `Act` arms are therefore
// `unreachable!`
//
// `VECTOR` stays `false` (the default): every complex tile is stored by the SoA kernel's own
// scalar alpha/beta epilogue (inside the L0 `cplx_microkernel` seam, which must not depend on this
// L1 trait), and this `apply` rides the tile-local in-place post-pass that
// `ComplexGemm::microkernel_epi` runs on the final depth panel. So there is no `apply_reg` /
// `apply_store` here, only the scalar `apply`. Because the kernel first stores exactly the bits
// plain `gemm_cplx` would, the post-pass makes `gemm_cplx_fused` bitwise-identical to `gemm_cplx`
// then the same element-wise bias add
#[cfg(all(feature = "complex", feature = "epilogue"))]
impl<T, const CA: bool, const CB: bool> Epilogue<crate::kernel::ComplexGemm<T, CA, CB>>
    for FusedEpi<T>
where
    T: crate::scalar::ComplexFloat,
{
    #[inline(always)]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        // Bias add only (the fast path is a full tile, so the `r`/`c` coordinate resolves the
        // per-row / per-col base directly). Complex addition is `num_complex`'s `Add`, the same
        // operation the `gemm_cplx`-then-map oracle applies, hence bitwise-identical
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => v + unsafe { *p.0.add(c) },
        };
        match self.act {
            Act::None => v,
            // The complex entry never constructs an activation (undefined on `C`)
            Act::Relu | Act::LeakyRelu(_) => {
                unreachable!("complex fused epilogue has no activation")
            }
        }
    }
}

/// The output domain of the requantizing [`KRequantize`] epilogue: the inclusive clamp
/// bounds and the final narrowing of an already-clamped `i64` to the output byte. Implemented
/// for the 2 quantized output types, `i8` (`[-128, 127]`) and `u8` (`[0, 255]`), so one
/// `KRequantize` impl and one requant family serve both. The narrowing `from_clamped` is only
/// ever handed a value already in `[LO, HI]`, so the `as` cast is a plain reinterpret of the
/// low byte (no saturation), matching the vector store's low-byte write
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) trait QuantOut: crate::scalar::Scalar {
    /// Inclusive clamp bounds of the output domain
    const LO: i32;
    /// Inclusive clamp bounds of the output domain
    const HI: i32;
    /// Truncate an already-clamped `i64` (in `[LO, HI]`) to the output byte
    fn from_clamped(q: i64) -> Self;
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl QuantOut for i8 {
    const LO: i32 = -128;
    const HI: i32 = 127;
    #[inline(always)]
    fn from_clamped(q: i64) -> Self {
        q as i8
    }
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl QuantOut for u8 {
    const LO: i32 = 0;
    const HI: i32 = 255;
    #[inline(always)]
    fn from_clamped(q: i64) -> Self {
        q as u8
    }
}

/// The requantizing epilogue: `C[r, c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`,
/// where `scale` is per-tensor or per-row / per-col ([`ScaleSpec`], the per-channel
/// quantized-inference convention) and `bias` is an optional per-row / per-col `i32` joined in
/// integer before the single f64 rounding. The clamp band `[LO, HI]` is the output domain, chosen
/// per output type by [`QuantOut`]: `i8` -> `[-128, 127]` (signed) and `u8` -> `[0, 255]` (the
/// ONNX-QLinearMatMul activation output). The struct itself carries no `Out`: one value drives
/// both, the domain coming from the `Fam::Out` the [`Epilogue`] impl is monomorphized for
///
/// Every tile drains its `i32` accumulators to scratch (vectorized), then maps each element to
/// the output byte. The map itself is split: on requant-vector-capable tokens (x86:
/// [`KRequantize::apply_store`] via [`crate::simd::KernelSimd::requant_store`]) a full lane-run
/// whose scale is constant across the run (per-tensor, or per-col in the driver frame) is
/// vectorized in `f64`; a per-row scale (which varies per lane) plus the row tail, a strided `C`,
/// the `k == 0` degenerate fill, and non-vector ISAs (scalar / NEON / wasm) take the scalar
/// [`KRequantize::apply`]. All paths are **bit-identical** (the `requant_store` equivalence
/// contract), so a single matrix mixes them freely. Either way this beats the unfused flow, which
/// materializes a full `m*n` `i32` C
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) struct KRequantize {
    /// Per-tensor / per-row / per-col output scale in the driver (oriented) frame; the dispatch
    /// layer flips the per-row / per-col axis (`Row` <-> `Col`) on a swap, in lockstep with the
    /// bias axis (the same user axis)
    pub scale: ScaleSpec,
    pub zp: i32,
    /// Optional per-row / per-col `i32` bias in the driver (oriented) frame; the dispatch
    /// layer flips the axis (`Row` <-> `Col`) on a swap
    pub bias: BiasSpec<i32>,
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl<O: QuantOut, Fam: KernelFamily<Acc = i32, Out = O>> Epilogue<Fam> for KRequantize {
    // `VECTOR = false`: requantize has no float-style in-register `apply_reg` path (`Out != Acc`,
    // and the clamp/round is not a `store_out` narrowing). Its vector form is the store-transform
    // `apply_store` below, gated by `VECTOR_STORE` instead
    const VECTOR: bool = false;
    const VECTOR_STORE: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: i32, r: usize, c: usize) -> O {
        let b = match self.bias {
            BiasSpec::None => 0,
            BiasSpec::Row(p) => unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => unsafe { *p.0.add(c) },
        };
        // Resolve the scale at `(r, c)`: a per-tensor constant, or a per-row / per-col lookup.
        // `ScaleSpec::Tensor(s)` and `ScaleSpec::Row(p)` therefore feed the identical `f64`
        // arithmetic below (only the lookup differs), so they are bitwise-equal
        let scale = match self.scale {
            ScaleSpec::Tensor(s) => s,
            ScaleSpec::Row(p) => unsafe { *p.0.add(r) },
            ScaleSpec::Col(p) => unsafe { *p.0.add(c) },
        };
        // `i32` and `f32` are exactly representable in `f64`, so this is 1 rounding step
        // total. `zp` joins in integer (round-half-to-even is not shift-invariant, so it
        // must not be folded into the pre-round expression). The `f64 -> i64` cast
        // saturates, and `saturating_add`/`clamp` keep the whole map panic-free and
        // bit-exact across every ISA. `O::LO`/`O::HI` select the output band (`i8` or `u8`);
        // `from_clamped` then reinterprets the already-clamped low byte
        let scaled = round_ne_f64(f64::from(v.wrapping_add(b)) * f64::from(scale));
        let q = (scaled as i64).saturating_add(i64::from(self.zp));
        O::from_clamped(q.clamp(i64::from(O::LO), i64::from(O::HI)))
    }

    #[inline(always)]
    unsafe fn apply_store<S>(&self, simd: S, src: *const i32, dst: *mut O, row: usize, col: usize)
    where
        S: KernelSimd<Fam::Lhs, Fam::Rhs, i32, O>,
    {
        unsafe {
            // A per-row scale varies per lane, so the single-`f64`-scale vector store cannot serve
            // it: fall to a per-lane scalar map straight over the raw accumulators in `src`. Since
            // it defers to `apply` (which resolves bias and the `Row` scale at `row + l`), it is
            // trivially bit-identical to the scalar path the caller freely mixes with (the row
            // tail, strided `C`, `k == 0` fill)
            if let ScaleSpec::Row(_) = self.scale {
                let lanes = <S as SimdOps<i32>>::LANES;
                for l in 0..lanes {
                    *dst.add(l) = <Self as Epilogue<Fam>>::apply(self, *src.add(l), row + l, col);
                }
                return;
            }
            // `LANES` consecutive-row `i32` accumulators from contiguous scratch
            let v = simd.loadu(src);
            // Bias add in integer: SIMD `i32` add is wrapping (`paddd`), matching `apply`'s
            // `wrapping_add`. The fast path is a full tile with `rsc == 1`, so the `LANES` rows at
            // `row` are consecutive `Row` bias values (one aligned load); a `Col` bias is a single
            // value broadcast across the column. `None` => `v` unchanged
            let v = match self.bias {
                BiasSpec::None => v,
                BiasSpec::Row(p) => simd.add(v, simd.loadu(p.0.add(row))),
                BiasSpec::Col(p) => simd.add(v, simd.splat(*p.0.add(col))),
            };
            // The scale is constant across this `LANES`-row run: a per-tensor value, or a per-col
            // value fixed at `col`. Widen it `f32 -> f64` exactly and take the single-scale vector
            // store, keeping the fast path for the common swapped (row-major C) `Col` orientation
            let scale = match self.scale {
                ScaleSpec::Tensor(s) => f64::from(s),
                ScaleSpec::Col(p) => f64::from(*p.0.add(col)),
                // `Row` handled by the per-lane branch above
                ScaleSpec::Row(_) => unreachable!("per-row scale takes the per-lane path"),
            };
            // The vector `f64` map, bit-identical to `apply` (the `requant_store` contract):
            // `O::LO`/`O::HI` are the output clamp band
            //
            // The `*mut O -> *mut i8` cast is sound and value-correct: `requant_store` writes the
            // low byte of each pre-clamped lane, and for any `x` in `[-128, 255]` the bytes of
            // `(x as i8)` and `(x as u8)` are identical, so the byte written equals the scalar
            // `O::from_clamped` result whether `O = i8` or `O = u8`
            simd.requant_store(dst as *mut i8, v, scale, self.zp, O::LO, O::HI);
        }
    }
}

/// Round-half-to-even of a finite `f64`, `no_std`-safe (`f64::round_ties_even` lives in
/// `std`; this crate is `no_std` without `std`). The classic `2^52` trick: for
/// `|x| < 2^52`, adding then removing the `2^52` magnitude constant snaps `x` to the
/// nearest integer under the default round-to-nearest-even mode; `|x| >= 2^52` is already
/// integral. Uses only comparisons and `f64` add/sub, so it needs no `std` float methods
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[inline(always)]
pub(crate) fn round_ne_f64(x: f64) -> f64 {
    const C: f64 = 4503599627370496.0; // 2^52
    // NaN or already integral (`|x| >= 2^52`) pass through unchanged. `f64::is_nan` is `core`
    // (only `round_ties_even` is `std`); `|x| >= 2^52` is spelled as 2 comparisons to avoid
    // the `std`-only `f64::abs`
    if x.is_nan() || x >= C || x <= -C {
        x
    } else if x >= 0.0 {
        (x + C) - C
    } else {
        (x - C) + C
    }
}
