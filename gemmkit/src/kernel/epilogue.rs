//! Fused epilogues (layer L4) are a transform a family applies to each output element as
//! the microkernel stores it. This replaces writing the raw `alpha*A*B + beta*C` and
//! mapping over `C` in a 2nd pass
//!
//! The seam is the [`Epilogue`] trait, threaded through
//! [`crate::kernel::KernelFamily::microkernel_epi`]. Its central invariant is zero-cost
//! identity. With `E = Identity`, [`Epilogue::IS_IDENTITY`] lets every hook const-fold away,
//! so the monomorphized kernel matches the non-fused kernel exactly. Plain
//! `gemm`/`gemm_i8` pay nothing for the seam. For a real epilogue, `gemm()` followed by a
//! scalar map matches the fused call bit-for-bit for floats. Every route fuses the same way:
//! blocking never depends on the epilogue, and [`Epilogue::apply_tile`] transforms the exact
//! registers the plain store would have written. See [`FusedEpi`] for the full argument
//!
//! 2 built-in epilogues ship. [`FusedEpi`] applies a per-row or per-col bias then
//! ReLU/LeakyReLU. It uses the vector path for `f32`/`f64` directly and, under `half`, for
//! `f16`/`bf16` by computing in `f32` and narrowing once on store. `KRequantize` maps an
//! `i32` accumulator to a clamped `i8`/`u8` through the scalar `round_ne_f64`
//! round-half-to-even map

use super::KernelFamily;
#[cfg(feature = "epilogue")]
use super::float::FloatGemm;
#[cfg(feature = "epilogue")]
use crate::parallel::Ptr;
#[cfg(feature = "epilogue")]
use crate::scalar::Float;
use crate::simd::{KernelSimd, SimdOps};

/// A transform fused into the microkernel's store
///
/// `C[r, c] <- apply(alpha*(A*B)[r, c] + beta*C[r, c], r, c)`. The kernel applies it exactly
/// once per output element. For an `OUT_IS_ACC` family, [`KernelFamily::microkernel_epi`]
/// fires it only on the final depth panel (`last_k`). Earlier panels hold a raw `Acc`
/// partial rather than the finished sum. For an `OUT_IS_ACC = false` family it fires
/// unconditionally, since that family never splits `k` into more than 1 panel
///
/// A tile takes 1 of 2 application paths, and both must return the same bits for the same
/// input. The fast vector [`Epilogue::apply_tile`] handles a full column-major tile
/// (`rsc == 1`). The scalar [`Epilogue::apply`] handles an edge or arbitrarily strided tile,
/// after draining it through scratch. [`Epilogue::apply_reg`] is the per-register form
/// `apply_tile` defaults to, and is what a `VECTOR` epilogue normally implements. The `gemv`
/// and small-`m,n` special paths are scalar throughout and call only [`Epilogue::apply`]
pub trait Epilogue<Fam: KernelFamily>: Copy + Send + Sync {
    /// `true` marks the identity transform. Every hook the kernel gates on this constant
    /// const-folds away, so the monomorphized kernel matches a non-fused kernel exactly
    const IS_IDENTITY: bool = false;
    /// `true` enables the fast vector-register path. The kernel calls
    /// [`Epilogue::apply_reg`] on the raw `Fam::Acc` register and lets the family's own
    /// store narrow it afterward. `FloatGemm` stores it as-is since `Out == Acc`, while
    /// the mixed `f16`/`bf16` families narrow through `store_out`. `false` (the default)
    /// routes every tile through the scratch/scalar path instead. That path is correct for
    /// any tile shape but slower, the right tradeoff when a vector form is not worth
    /// writing. A requantize epilogue instead sets `VECTOR_STORE` and uses
    /// [`Epilogue::apply_store`]
    const VECTOR: bool = false;

    /// Scalar transform at absolute `(row, col)` in the oriented problem frame
    ///
    /// # Safety
    ///
    /// Interior pointers (bias) must be valid for the problem's `m`/`n`. Call this only
    /// inside the matching [`crate::simd::Simd::vectorize`] context
    unsafe fn apply(&self, v: Fam::Acc, row: usize, col: usize) -> Fam::Out;

    /// Vector transform of `LANES` consecutive rows `[row, row + LANES)` of column `col`
    ///
    /// The fast path guarantees a full tile with `rsc == 1`, so the rows are unit-stride.
    /// This must return the same bits [`Epilogue::apply`] returns for the same input. The
    /// default is unreachable, since only a `VECTOR = true` epilogue overrides it, the same
    /// optional-override convention [`crate::simd::KernelSimd::dot_accumulate`] uses
    ///
    /// # Safety
    ///
    /// Same obligations as [`Epilogue::apply`]. `simd` is the token whose
    /// [`crate::simd::Simd::vectorize`] context is active
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

    /// Vector transform of a whole `MR_REG x NR` register tile
    ///
    /// `acc[j][i]` holds the finished values for rows
    /// `[row0 + i*LANES, row0 + (i+1)*LANES)` of column `col0 + j`. This must return the
    /// same bits as [`Epilogue::apply_reg`] applied to each register in turn
    ///
    /// The default does exactly that, and is correct for any `VECTOR` epilogue. An epilogue
    /// that dispatches on a runtime discriminant should still override it and match once,
    /// outside the loops. The kernel's store pass unrolls by the `MR_REG`/`NR` const
    /// generics, so a per-register match repeats once per accumulator. Past a few
    /// accumulators, the compiler stops holding the tile in registers. It spills every
    /// accumulator to the stack inside the kc loop, where the epilogue does not even run.
    /// Hoisting the match out of the loops avoids the spill
    ///
    /// # Safety
    ///
    /// Same obligations as [`Epilogue::apply_reg`]
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn apply_tile<S, const MR_REG: usize, const NR: usize>(
        &self,
        simd: S,
        mut acc: [[<S as SimdOps<Fam::Acc>>::Reg; MR_REG]; NR],
        row0: usize,
        col0: usize,
    ) -> [[<S as SimdOps<Fam::Acc>>::Reg; MR_REG]; NR]
    where
        S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    {
        let lanes = <S as SimdOps<Fam::Acc>>::LANES;
        for j in 0..NR {
            for i in 0..MR_REG {
                acc[j][i] = unsafe { self.apply_reg(simd, acc[j][i], row0 + i * lanes, col0 + j) };
            }
        }
        acc
    }

    /// `true` enables the vector store path through [`Epilogue::apply_store`]
    ///
    /// This applies to an `Out != Acc` epilogue, such as requantize, on a token that is
    /// also requant-vector-capable ([`crate::simd::KernelSimd::REQUANT_VECTOR`]). This is a
    /// separate axis from [`Epilogue::VECTOR`]. `VECTOR` governs the in-register
    /// `apply_reg` path, where the family's own store narrows the result afterward. A
    /// requantize epilogue has no such in-register form, so it leaves `VECTOR = false` and
    /// sets this instead
    const VECTOR_STORE: bool = false;

    /// Vector store-transform for a full tile
    ///
    /// Reads `LANES` consecutive-row `Acc` values from contiguous scratch at `src`, applies
    /// the full epilogue, and writes `LANES` `Out` values to `dst` at unit row stride. The
    /// caller guarantees `rsc == 1` on this path. This must produce the same bytes
    /// [`Epilogue::apply`] would for the same values, since a single output tile mixes both
    /// paths freely. The row tail, a strided `C`, and the `k == 0` degenerate fill all take
    /// `apply`. The default is unreachable, since only a `VECTOR_STORE = true` epilogue
    /// overrides it
    ///
    /// # Safety
    ///
    /// `src` must be valid for `LANES` `Acc` reads and `dst` for `LANES` `Out` writes.
    /// Interior pointers such as bias must be valid for the problem. `simd` is the token
    /// whose [`crate::simd::Simd::vectorize`] context is active
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

    /// React to an orientation-normalization swap that transposes the engine frame
    ///
    /// The float and mixed dispatch ladders call this exactly once, right after
    /// `orient_transpose` reports a swap. A row-major-ish `C` makes the engine compute
    /// `C^T = B^T*A^T`, swapping `m` and `n`. An epilogue whose state is tied to the
    /// problem frame re-orients itself here. [`FusedEpi`] flips its bias axis between row and
    /// column. [`MapEpi`] flags the coordinate transpose instead. A stateless epilogue keeps
    /// this no-op default, as [`Identity`] does. `KRequantize` also keeps the default. Its
    /// i8 requant ladder re-orients its bias and scale before constructing the epilogue, so
    /// this hook never runs for it
    #[inline(always)]
    fn on_orient_swap(&mut self) {}
}

/// The no-op epilogue
///
/// Every family's fused kernel hook checks `!E::IS_IDENTITY` before doing any work, so
/// `E = Identity` makes those checks const-fold away. The monomorphized kernel matches
/// the non-fused one exactly
#[derive(Copy, Clone, Default)]
pub struct Identity;

impl<Fam: KernelFamily> Epilogue<Fam> for Identity {
    const IS_IDENTITY: bool = true;
    #[inline(always)]
    unsafe fn apply(&self, _: Fam::Acc, _: usize, _: usize) -> Fam::Out {
        // Every caller gates on IS_IDENTITY before calling apply, so this never runs
        unreachable!("identity epilogue is never applied")
    }
}

/// Which axis a caller-supplied bias vector indexes: 1 entry per output row, or 1 per
/// output column
///
/// Callers use this to state their bias's shape. The dispatch layer swaps `PerRow` and
/// `PerCol` when it swaps the problem's orientation
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub enum BiasDim {
    /// 1 bias value per output row (length `m`), added to every column of that row
    PerRow,
    /// 1 bias value per output column (length `n`), added to every row of that column
    PerCol,
}

/// A resolved bias source in the driver's already-oriented frame
///
/// `Ptr` wraps the raw pointer as `Send + Sync` so this can cross into the parallel workers.
/// The pointee outlives the call because the borrow that produced it stays live in the
/// caller's stack frame
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub(crate) enum BiasSpec<T> {
    /// No bias
    None,
    /// 1 value per output row, added to every column of that row
    Row(Ptr<T>),
    /// 1 value per output column, added to every row of that column
    Col(Ptr<T>),
}

/// A resolved requantize scale in the driver's already-oriented frame
///
/// Holds 1 value shared by the whole tensor, or 1 `f32` per row or per column, the
/// per-channel quantized-inference convention. This has the same shape as [`BiasSpec`] and
/// the dispatch layer flips it in lockstep with the bias, since both share the same
/// user-facing axis. `Ptr` carries the same `Send + Sync` and lifetime reasoning as
/// `BiasSpec`
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) enum ScaleSpec {
    /// 1 scale applied to every element
    Tensor(f32),
    /// 1 scale per output row, applied to every column of that row
    Row(Ptr<f32>),
    /// 1 scale per output column, applied to every row of that column
    Col(Ptr<f32>),
}

/// The activation stage of [`FusedEpi`], applied after the bias add
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub(crate) enum Act<T> {
    /// No activation
    None,
    /// `max(v, 0)`. NaN maps to 0 since `NaN > 0` is false
    Relu,
    /// `max(v, 0) + slope*min(v, 0)`. NaN maps to 0, and `-0.0` maps to `+0.0`
    LeakyRelu(T),
}

/// The one runtime-composed float epilogue: bias (per-row, per-col, or none) then an
/// activation (none, ReLU, or LeakyReLU)
///
/// Every combination shares 1 monomorphization. Each tile pays 2 branches on top of the
/// `mr*nr*kc` FMA loop it sits inside, so the kernel is not duplicated per bias/activation
/// combination
///
/// This type is `pub` with crate-private fields. Only the API layer builds one, but the
/// type still appears in the dispatch table's function-pointer signature
#[cfg(feature = "epilogue")]
#[derive(Copy, Clone)]
pub struct FusedEpi<T> {
    pub(crate) bias: BiasSpec<T>,
    pub(crate) act: Act<T>,
}

#[cfg(feature = "epilogue")]
impl<T> FusedEpi<T> {
    /// Flip the bias axis between row and column
    ///
    /// This is the orientation-swap reaction shared by every family's [`Epilogue`] impl,
    /// through [`Epilogue::on_orient_swap`], and by the complex ladder. A row-major-ish `C`
    /// computes `C^T = B^T*A^T`, swapping `m` and `n`, so a per-row bias becomes per-col in
    /// the oriented frame. `None` stays `None`
    #[inline(always)]
    pub(crate) fn flip_bias(&mut self) {
        // Match by reference and rebuild from the raw pointer. *mut T is Copy for any T, so
        // this avoids a T: Copy bound that BiasSpec's derived Copy would otherwise demand
        self.bias = match &self.bias {
            BiasSpec::None => BiasSpec::None,
            BiasSpec::Row(p) => BiasSpec::Col(Ptr(p.0)),
            BiasSpec::Col(p) => BiasSpec::Row(Ptr(p.0)),
        };
    }
}

// Bound is `Float<Acc = T> + PartialOrd`, not the public `FusedScalar` trait. This keeps the
// kernel layer free of a dispatch-layer dependency and still selects exactly the real floats
// `Complex` implements `Float` but not `PartialOrd`, and `f16`/`bf16` do not implement
// `Float` at all
// `FusedScalar` is the seal the public API applies on top
#[cfg(feature = "epilogue")]
impl<T: Float<Acc = T> + PartialOrd> Epilogue<FloatGemm<T>> for FusedEpi<T> {
    const VECTOR: bool = true;

    #[inline(always)]
    fn on_orient_swap(&mut self) {
        self.flip_bias();
    }

    #[inline(always)]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => v + unsafe { *p.0.add(c) },
        };
        match self.act {
            Act::None => v,
            // v > T::ZERO is false for NaN, so this also matches the vector max(v, 0)
            Act::Relu => {
                if v > T::ZERO {
                    v
                } else {
                    T::ZERO
                }
            }
            // Written as the same hi/lo composition as apply_reg's vector form, so the 2
            // agree bit-for-bit, including on NaN (-> 0) and -0.0 (-> +0.0)
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
                // Full tile, rsc == 1: the LANES rows at r are consecutive in the bias slice
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

    // Both discriminants are decoded once for the whole tile, so the unrolled store pass
    // stays straight-line. See the trait method for why that matters on a wide tile
    // The per-element operations, and their order, match apply_reg's
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn apply_tile<S, const MR_REG: usize, const NR: usize>(
        &self,
        s: S,
        mut acc: [[S::Reg; MR_REG]; NR],
        row0: usize,
        col0: usize,
    ) -> [[S::Reg; MR_REG]; NR]
    where
        S: KernelSimd<T, T, T, T>,
    {
        unsafe {
            let lanes = <S as SimdOps<T>>::LANES;
            match self.bias {
                BiasSpec::None => {}
                // Full tile, rsc == 1: the LANES rows at row0 + i*lanes are consecutive
                // in the bias slice
                BiasSpec::Row(p) => {
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            let bv = s.loadu(p.0.add(row0 + i * lanes));
                            acc[j][i] = s.add(acc[j][i], bv);
                        }
                    }
                }
                // 1 bias element per column, so the splat is hoisted out of the row loop
                BiasSpec::Col(p) => {
                    for j in 0..NR {
                        let bv = s.splat(*p.0.add(col0 + j));
                        for i in 0..MR_REG {
                            acc[j][i] = s.add(acc[j][i], bv);
                        }
                    }
                }
            }
            match self.act {
                Act::None => {}
                Act::Relu => {
                    let z = s.zero();
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            acc[j][i] = s.max(acc[j][i], z);
                        }
                    }
                }
                Act::LeakyRelu(sl) => {
                    let z = s.zero();
                    let sv = s.splat(sl);
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            let v = acc[j][i];
                            acc[j][i] = s.add(s.max(v, z), s.mul(sv, s.min(v, z)));
                        }
                    }
                }
            }
            acc
        }
    }
}

/// A user-defined per-element epilogue
///
/// The closure `f` applies to each stored output element at its final value. That value is
/// `C[r, c] <- f(alpha*(A*B)[r, c] + beta*C[r, c], r, c)`, with `(r, c)` in the user frame of
/// `C`. The public [`crate::gemm_map`] entry (feature `epilogue`, `f32`/`f64`) lowers to this
///
/// The closure runs scalar, once per output element, an indirect call amortized over that
/// element's `O(k)` FLOPs. The seam is a borrowed trait object rather than a generic `F`, so
/// there is 1 monomorphization per `(T, ISA)`, not 1 per closure. It still sets
/// `VECTOR = true`, which keeps the kernel on plain `gemm`'s own path selection. That
/// selection uses the vector fast path for a full column-major tile and the scratch path
/// for an edge or strided one. This makes `gemm_map` visit every element the same route
/// `gemm` would and hand `f` the identical value `gemm` would have written there
///
/// That identity matters most on the fast path, which stores a hardware-fused
/// `beta*C + alpha*AB`. [`crate::scalar::Float::mul_add`] on the scratch path does not
/// reproduce that rounding for `beta != 0, 1`. An epilogue that forced every tile through
/// scratch (`VECTOR = false`) would hand `f` a value up to 1 ULP off from what `gemm` wrote.
/// [`Epilogue::apply_reg`] avoids that. It drains the fast-path register to a stack buffer
/// and calls the same `apply` per lane, so both paths hand `f` the same bits. `gemm_map` is
/// `gemm()` then `f`, bit-for-bit, everywhere
///
/// The closure is stored as a shared reference `&'u (dyn Fn + Sync)`, not an erased `Ptr`.
/// The reference itself is `Copy` and the referent is `Sync`. That already satisfies the
/// `Copy + Send + Sync` bound [`Epilogue`] requires, without a `'static` bound. The parallel
/// workers capture it by value inside a blocking `rayon` `for_each` that joins before the
/// borrow's scope ends, so the non-`'static` borrow stays sound
///
/// `swapped` records whether the caller ran in the oriented frame. The general driver and the
/// small-`m,n` and small-`k` routes may compute `C^T = B^T*A^T` for a row-major-ish `C`,
/// swapping `m` and `n`. In that case `apply` transposes `(row, col)` back to `(col, row)`
/// before calling `f`. gemv never takes that swap: it resolves its own row/column ambiguity
/// through `swap_rc` before coordinates reach here, so it always passes `swapped = false`.
/// [`FusedEpi`] only needs to relabel an axis for its per-row/per-col bias. A user closure
/// depends on both coordinates together, so it needs the full transpose rather than an axis
/// swap
///
/// Like [`FusedEpi`], this type is `pub` with crate-private fields. Only the API layer builds
/// one, from a borrowed closure. The type still appears in the dispatch slot's
/// function-pointer signature and the `MapScalar` dispatch methods
#[cfg(feature = "epilogue")]
pub struct MapEpi<'u, T> {
    /// The user closure `f(value, row, col) -> value`. `+ Sync` on the trait object makes
    /// the shared reference itself `Send + Sync`, which is what lets `MapEpi` cross into the
    /// workers
    pub(crate) f: &'u (dyn Fn(T, usize, usize) -> T + Sync),
    /// `true` when the caller ran in the transposed (oriented) frame, so `apply` must swap
    /// `(row, col)` back to `(col, row)` before handing them to `f`
    pub(crate) swapped: bool,
}

// A derived `Copy, Clone` would add a `T: Copy`/`T: Clone` bound these fields do not need
// A shared reference and a `bool` are `Copy` regardless of `T`, so both impls are written by
// hand to drop that bound
#[cfg(feature = "epilogue")]
impl<T> Copy for MapEpi<'_, T> {}
#[cfg(feature = "epilogue")]
impl<T> Clone for MapEpi<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

// `Float<Acc = T>` selects exactly f32/f64, matching the public MapScalar seal. A narrow type
// would double-round: once into f32 for the accumulate, and again narrowing the closure's
// T-domain result. Complex and integer types have no apply seam wired up at all
#[cfg(feature = "epilogue")]
impl<T: Float<Acc = T>> Epilogue<FloatGemm<T>> for MapEpi<'_, T> {
    // Keeps the kernel on the fast vector path for a full tile. See the type doc for why that
    // path's rounding must match plain gemm's. apply_reg then applies the closure per lane
    const VECTOR: bool = true;

    #[inline(always)]
    fn on_orient_swap(&mut self) {
        // The oriented routes compute C^T (m<->n swap), so flag apply to transpose (row, col)
        // back to the user frame before handing them to the closure
        self.swapped = true;
    }

    #[inline]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        // The oriented routes transpose (r, c) before it reaches here. Undo that for the closure
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
            // v is the fused beta*C + alpha*AB the fast path stores, the same bits plain gemm
            // writes. Drain the LANES rows [r, r + lanes) at column c to a buffer, run each
            // through the scalar apply, and reload
            let mut buf = [T::ZERO; MAP_REG_LANES];
            s.storeu(buf.as_mut_ptr(), v);
            for (l, slot) in buf.iter_mut().enumerate().take(lanes) {
                *slot = self.apply(*slot, r + l, c);
            }
            s.loadu(buf.as_ptr())
        }
    }
}

/// Stack-buffer width for [`MapEpi::apply_reg`]'s per-lane drain
///
/// Covers every float [`crate::simd::SimdOps`] lane count that can appear here. `f32` on
/// AVX-512F is the widest, at 16 lanes
#[cfg(feature = "epilogue")]
const MAP_REG_LANES: usize = 16;

// This is 1 blanket impl for every narrow family with Lhs = Rhs = Out = N and Acc = f32:
// MixedGemm<f16>, MixedGemm<bf16>, and Bf16DotGemm. It cannot overlap the FloatGemm impl
// above, since f32/f64 do not implement NarrowFloat. Bias and slope arrive as N and widen
// exactly into f32, because N is a strict subset of f32's range and precision. The whole
// bias-and-activation transform runs in f32 against the f32 accumulator, and N::narrow
// performs the single round-to-nearest-even step at the end
//
// That single narrowing is why a fused call here is not bitwise-equal to gemm() followed
// by a separate map. The unfused route would round Acc to N once for the plain store and
// again after the closure. That is 2 roundings, where the fused route has 1. The 2 fused
// paths still agree with each other. apply_reg computes the identical f32 bias-and-activation
// and leaves the narrowing to the family's own store_out. Both routes round exactly once and
// produce the same N
#[cfg(all(feature = "half", feature = "epilogue"))]
impl<N, Fam> Epilogue<Fam> for FusedEpi<N>
where
    N: crate::scalar::NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
{
    const VECTOR: bool = true;

    #[inline(always)]
    fn on_orient_swap(&mut self) {
        self.flip_bias();
    }

    #[inline(always)]
    unsafe fn apply(&self, v: f32, r: usize, c: usize) -> N {
        // Widen the narrow bias exactly and add in f32
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { (*p.0.add(r)).widen() },
            BiasSpec::Col(p) => v + unsafe { (*p.0.add(c)).widen() },
        };
        // Same scalar forms as the FloatGemm impl, just on f32 regardless of N
        let v = match self.act {
            Act::None => v,
            // v > 0.0 is false for NaN, so this also matches the vector max(v, 0)
            Act::Relu => {
                if v > 0.0 {
                    v
                } else {
                    0.0
                }
            }
            // hi/lo split matches apply_reg's vector form bit-for-bit, including NaN -> 0
            // and -0.0 -> +0.0
            Act::LeakyRelu(s) => {
                let hi = if v > 0.0 { v } else { 0.0 };
                let lo = if v < 0.0 { v } else { 0.0 };
                hi + s.widen() * lo
            }
        };
        // The only rounding step: narrows f32 to N once, on the way out
        N::narrow(v)
    }

    #[inline(always)]
    unsafe fn apply_reg<S>(&self, s: S, v: S::Reg, r: usize, c: usize) -> S::Reg
    where
        S: KernelSimd<N, N, f32, N>,
    {
        unsafe {
            let v = match self.bias {
                // Full tile, rsc == 1: the LANES narrow bias values at r are consecutive
                // This lets load_lhs widen the whole run into 1 f32 register in one shot
                BiasSpec::None => v,
                BiasSpec::Row(p) => s.add(v, s.load_lhs(p.0.add(r))),
                BiasSpec::Col(p) => s.add(v, s.splat((*p.0.add(c)).widen())),
            };
            // Leaves the result in f32. store_out does the single narrowing write
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

    // Decodes both discriminants once per tile, as the FloatGemm impl does and for the
    // same reason. The per-element operations, and their order, match apply_reg's
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn apply_tile<S, const MR_REG: usize, const NR: usize>(
        &self,
        s: S,
        mut acc: [[S::Reg; MR_REG]; NR],
        row0: usize,
        col0: usize,
    ) -> [[S::Reg; MR_REG]; NR]
    where
        S: KernelSimd<N, N, f32, N>,
    {
        unsafe {
            let lanes = <S as SimdOps<f32>>::LANES;
            match self.bias {
                BiasSpec::None => {}
                // load_lhs widens the LANES consecutive narrow bias values in one shot
                BiasSpec::Row(p) => {
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            let bv = s.load_lhs(p.0.add(row0 + i * lanes));
                            acc[j][i] = s.add(acc[j][i], bv);
                        }
                    }
                }
                BiasSpec::Col(p) => {
                    for j in 0..NR {
                        let bv = s.splat((*p.0.add(col0 + j)).widen());
                        for i in 0..MR_REG {
                            acc[j][i] = s.add(acc[j][i], bv);
                        }
                    }
                }
            }
            match self.act {
                Act::None => {}
                Act::Relu => {
                    let z = s.zero();
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            acc[j][i] = s.max(acc[j][i], z);
                        }
                    }
                }
                Act::LeakyRelu(sl) => {
                    let z = s.zero();
                    let sv = s.splat(sl.widen());
                    for j in 0..NR {
                        for i in 0..MR_REG {
                            let v = acc[j][i];
                            acc[j][i] = s.add(s.max(v, z), s.mul(sv, s.min(v, z)));
                        }
                    }
                }
            }
            acc
        }
    }
}

// FusedEpi<T> also implements Epilogue for ComplexGemm<T, CA, CB>, a distinct family type
// from FloatGemm, the narrow families, and KRequantize. This cannot conflict with any impl
// above
//
// There is no activation here on purpose. ReLU and LeakyRelu depend on an ordering that
// complex numbers do not have, so gemm_cplx_fused only ever constructs Act::None, leaving the
// other Act arms unreachable
//
// VECTOR stays false, the default. A complex tile is stored by the SoA kernel's own scalar
// alpha/beta epilogue. That epilogue lives inside the L0 cplx_microkernel seam, which must
// not depend on this L4 trait. This apply instead rides the in-place post-pass
// ComplexGemm::microkernel_epi runs over the finished tile on the final depth panel. So only
// apply is implemented here, not apply_reg or apply_store. That post-pass runs after the
// kernel already stored exactly the bits plain gemm_cplx would write. So gemm_cplx_fused ends
// up bitwise the same as gemm_cplx followed by this same bias add
#[cfg(all(feature = "complex", feature = "epilogue"))]
impl<T, const CA: bool, const CB: bool> Epilogue<crate::kernel::ComplexGemm<T, CA, CB>>
    for FusedEpi<T>
where
    T: crate::scalar::ComplexFloat,
{
    #[inline(always)]
    fn on_orient_swap(&mut self) {
        self.flip_bias();
    }

    #[inline(always)]
    unsafe fn apply(&self, v: T, r: usize, c: usize) -> T {
        // num_complex's Add, the same operation a gemm_cplx-then-map oracle would use
        let v = match self.bias {
            BiasSpec::None => v,
            BiasSpec::Row(p) => v + unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => v + unsafe { *p.0.add(c) },
        };
        match self.act {
            Act::None => v,
            // gemm_cplx_fused never constructs one of these (undefined on complex numbers)
            Act::Relu | Act::LeakyRelu(_) => {
                unreachable!("complex fused epilogue has no activation")
            }
        }
    }
}

/// The output domain of the requantizing `KRequantize` epilogue
///
/// Holds the clamp bounds and the final byte narrowing. It implements the bounds once each
/// for `i8` (`[-128, 127]`) and `u8` (`[0, 255]`), so a single generic `KRequantize` impl
/// serves both output types. `from_clamped` is only ever called with a value already inside
/// `[LO, HI]`. The `as` cast is then a plain low-byte reinterpret, not a saturating cast,
/// matching what the vector store writes
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) trait QuantOut: crate::scalar::Scalar {
    /// Inclusive lower clamp bound of the output domain
    const LO: i32;
    /// Inclusive upper clamp bound of the output domain
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

/// The requantizing epilogue: `C[r, c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`
///
/// `scale` is per-tensor or per-row or per-col ([`ScaleSpec`], the per-channel
/// quantized-inference convention). `bias` is an optional per-row or per-col `i32` added in
/// integer before the single `f64` rounding step. `[LO, HI]` is the output type's clamp band,
/// supplied by [`QuantOut`]: `i8` maps to `[-128, 127]`, `u8` maps to `[0, 255]`, the
/// ONNX-QLinearMatMul activation convention. The struct itself is generic over neither `O` nor
/// `Fam`, so the same value drives whichever output type the [`Epilogue`] impl below gets
/// monomorphized for
///
/// Every tile drains its `i32` accumulators to scratch, then maps each element to the output
/// byte. On a requant-vector-capable token, that map runs through [`KRequantize::apply_store`],
/// backed by [`crate::simd::KernelSimd::requant_store`]. It takes that path for any lane-run
/// whose scale does not vary within it: per-tensor, or per-col in the driver frame. A per-row
/// scale, the row tail, a strided `C`, the `k == 0` degenerate fill, and every non-vector ISA
/// instead take the scalar [`KRequantize::apply`]. Both produce identical bytes for the same
/// input, the `requant_store` contract, so 1 matrix can mix them freely between tiles. Either
/// path avoids materializing a full `m*n` `i32` `C` before requantizing it as a separate step
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) struct KRequantize {
    /// Output scale in the driver's oriented frame: per-tensor, per-row, or per-col
    ///
    /// The dispatch layer swaps `Row`/`Col` alongside the bias axis on an orientation swap,
    /// since both index the same user-facing axis
    pub scale: ScaleSpec,
    /// Output zero-point, added after rounding and before the clamp
    pub zp: i32,
    /// Optional integer bias in the driver's oriented frame, added before the scale
    ///
    /// The dispatch layer swaps `Row`/`Col` on an orientation swap, the same as `scale`
    pub bias: BiasSpec<i32>,
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl<O: QuantOut, Fam: KernelFamily<Acc = i32, Out = O>> Epilogue<Fam> for KRequantize {
    // No in-register apply_reg path: Out != Acc here, and the round-and-clamp is not the kind
    // of narrowing store_out performs. The vector form lives in apply_store below, gated by
    // VECTOR_STORE
    const VECTOR: bool = false;
    const VECTOR_STORE: bool = true;

    #[inline(always)]
    unsafe fn apply(&self, v: i32, r: usize, c: usize) -> O {
        let b = match self.bias {
            BiasSpec::None => 0,
            BiasSpec::Row(p) => unsafe { *p.0.add(r) },
            BiasSpec::Col(p) => unsafe { *p.0.add(c) },
        };
        // Resolve the scale at (r, c). Every variant feeds the same f64 arithmetic below, only
        // the lookup differs, so Tensor/Row/Col are bitwise-equal at a shared scale value
        let scale = match self.scale {
            ScaleSpec::Tensor(s) => s,
            ScaleSpec::Row(p) => unsafe { *p.0.add(r) },
            ScaleSpec::Col(p) => unsafe { *p.0.add(c) },
        };
        // i32 and f32 are both exact in f64, so this is 1 rounding step total. zp joins after
        // rounding, in integer, since round-half-to-even is not shift-invariant. saturating_add
        // and clamp keep the whole map panic-free and bit-exact on every ISA regardless of
        // overflow
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
            // A per-row scale varies within the LANES-row run, so the single-scale vector store
            // below cannot serve it. Fall back to calling apply per lane, which is trivially
            // bit-identical to the scalar path, since it is the scalar path
            if let ScaleSpec::Row(_) = self.scale {
                let lanes = <S as SimdOps<i32>>::LANES;
                for l in 0..lanes {
                    *dst.add(l) = <Self as Epilogue<Fam>>::apply(self, *src.add(l), row + l, col);
                }
                return;
            }
            // LANES consecutive-row i32 accumulators from contiguous scratch
            let v = simd.loadu(src);
            // SIMD i32 add wraps (paddd), matching apply's wrapping_add. Full tile, rsc == 1: a
            // Row bias's LANES values at row are consecutive, taking 1 load. A Col bias is 1
            // value, broadcast across the column
            let v = match self.bias {
                BiasSpec::None => v,
                BiasSpec::Row(p) => simd.add(v, simd.loadu(p.0.add(row))),
                BiasSpec::Col(p) => simd.add(v, simd.splat(*p.0.add(col))),
            };
            // Constant across this run now that Row is handled above: per-tensor, or per-col
            // fixed at col. Widen f32 -> f64 exactly for the single-scale vector store
            let scale = match self.scale {
                ScaleSpec::Tensor(s) => f64::from(s),
                ScaleSpec::Col(p) => f64::from(*p.0.add(col)),
                ScaleSpec::Row(_) => unreachable!("per-row scale takes the per-lane path"),
            };
            // requant_store must match apply bit-for-bit, per its own contract. O::LO/O::HI
            // pick the clamp band
            //
            // The *mut O -> *mut i8 cast is value-correct regardless of O: requant_store writes
            // only the low byte of each clamped lane. For any x in [-128, 255] that byte is
            // identical whether reached through (x as i8) or (x as u8)
            simd.requant_store(dst as *mut i8, v, scale, self.zp, O::LO, O::HI);
        }
    }
}

/// Round-half-to-even of a finite `f64`
///
/// Avoids `f64::round_ties_even`, a `std`-only method unavailable when the crate builds
/// without its `std` feature. Uses the `2^52` trick instead. For `|x| < 2^52`, adding then
/// subtracting that magnitude constant forces the hardware to round `x` to the nearest
/// integer, under the default round-to-nearest-even mode. `|x| >= 2^52` is already an
/// integer, so it needs no rounding. Every step is a comparison or an `f64` add or subtract,
/// both available in `core`
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[inline(always)]
pub(crate) fn round_ne_f64(x: f64) -> f64 {
    const C: f64 = 4503599627370496.0; // 2^52
    // NaN and already-integral values (|x| >= 2^52) pass through unchanged. is_nan is core-only,
    // like everything else here. The >= / <= pair avoids the std-only f64::abs
    if x.is_nan() || x >= C || x <= -C {
        x
    } else if x >= 0.0 {
        (x + C) - C
    } else {
        (x - C) + C
    }
}
