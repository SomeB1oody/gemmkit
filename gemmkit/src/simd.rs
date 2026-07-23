//! SIMD abstraction layer (L0): the per-ISA register vocabulary every microkernel builds on
//!
//! Self-contained: depends only on [`crate::scalar`] and `core`, never on the kernel,
//! driver, or cache layers above it, so it could move into its own crate unchanged
//!
//! # The 2 traits
//!
//! * [`Simd`]: a zero-sized ISA token (e.g. [`Fma`]), not parameterized by element type.
//!   Its only job is [`Simd::vectorize`], the `#[target_feature]` boundary (see below)
//! * [`SimdOps<T>`]: the per-element-type vocabulary of a token: register type, lane
//!   count, and every primitive the microkernel needs (load/store/broadcast/mul/add/
//!   fma/reduce). Token and element type are decoupled, so `LANES` depends on the
//!   `(ISA, T)` pair: `f32` is 8 lanes on FMA, 16 on AVX-512F; `f64` is half of `f32`
//!   on the same token
//!
//! Every primitive a microkernel needs lives on these 2 traits, so the microkernel body
//! is one generic function shared by every ISA. Adding an ISA means adding a token, its
//! `SimdOps` impls, and one dispatch line, never touching the microkernel itself
//!
//! # `#[target_feature]` correctness
//!
//! AVX/AVX-512 intrinsics only codegen correctly where the target feature is enabled,
//! but feature support is a runtime fact resolved by the dispatch layer, so a fixed
//! `#[target_feature]` attribute cannot sit on the generic microkernel. Instead each
//! token's [`Simd::vectorize`] runs a closure inside a small `#[target_feature]`-annotated
//! function; the closure, and the `#[inline]` primitives it calls, inline into that
//! function, so every intrinsic ends up codegen'd in a feature-enabled context. The same
//! mechanism works whether the closure runs on the calling thread or inside a rayon
//! worker closure

use crate::scalar::Scalar;

// Split (SoA) complex microkernel and the impl_complex_simd! macro every token below uses
#[cfg(feature = "complex")]
#[macro_use]
mod complex;
// AVX-512F ISA token, plus its VNNI and bf16 dot-product variants
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512;
// AVX2 + FMA ISA token
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod fma;
// AArch64 NEON ISA token
#[cfg(target_arch = "aarch64")]
mod neon;
// 1-lane scalar ISA token: the portability floor, always compiled
mod scalar;
// WebAssembly simd128 ISA token
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
pub use self::avx512::Avx512Bf16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx512::Avx512F;
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
pub use self::avx512::Avx512Vnni;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::fma::Fma;
#[cfg(target_arch = "aarch64")]
pub use self::neon::Neon;
pub use self::scalar::ScalarTok;
#[cfg(target_arch = "wasm32")]
pub use self::wasm::Simd128;

/// The capability an ISA token needs to drive a [`crate::kernel::KernelFamily`] whose
/// input types `L`/`R`, accumulator `A`, and output `O` are not all the same: accumulate
/// in `A` (the [`SimdOps<A>`] supertrait), and move family inputs/outputs into and out of
/// `A`-typed registers, widening on load and narrowing on store wherever the element type
/// is narrower than `A`
///
/// This is the seam that lets **mixed precision** (`A != L`) work without a per-type
/// branch in the driver. The homogeneous case (`L = R = A = O`) is covered once by the
/// blanket impl below, which forwards to plain [`SimdOps`] load/splat/store; a narrow
/// family (`f16`/`bf16` inputs, `f32` accumulator) instead gets an ISA impl whose
/// `load_*` widens and `store_out` narrows. The all-equal blanket and any mixed impl
/// (`L != A`) can never overlap, since a mixed impl's types are concrete and unequal
pub trait KernelSimd<L: Scalar, R: Scalar, A: Scalar, O: Scalar>: SimdOps<A> {
    /// Load `LANES` LHS values, widened to one `A` register (a plain load when `L == A`)
    ///
    /// # Safety
    /// `p` valid for `LANES` reads; run inside this token's [`Simd::vectorize`]
    unsafe fn load_lhs(self, p: *const L) -> <Self as SimdOps<A>>::Reg;
    /// Widen one RHS scalar and broadcast it to all `A` lanes (a plain splat when `R == A`)
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn splat_rhs(self, v: R) -> <Self as SimdOps<A>>::Reg;
    /// Load `LANES` output values, widened to one `A` register, for the `beta != 0`
    /// read of `C` (a plain load when `O == A`)
    ///
    /// # Safety
    /// `p` valid for `LANES` reads; run inside [`Simd::vectorize`]
    unsafe fn load_out(self, p: *const O) -> <Self as SimdOps<A>>::Reg;
    /// Narrow one `A` register to `LANES` output values and store them (a plain store
    /// when `O == A`; rounds to nearest-even when it actually narrows)
    ///
    /// # Safety
    /// `p` valid for `LANES` writes; run inside [`Simd::vectorize`]
    unsafe fn store_out(self, p: *mut O, v: <Self as SimdOps<A>>::Reg);

    /// Accumulate one full `MR_REG x NR` microtile from dot-product-packed panels into the
    /// register-resident `acc` (pre-zeroed by the caller). This is the seam a dot-kernel
    /// family ([`crate::kernel::KernelFamily::DEPTH_MULTIPLE`] `> 1`) drives on instead of
    /// [`SimdOps::accumulate_tile`]: it folds `DEPTH_MULTIPLE` consecutive depth steps into
    /// one hardware instruction (`vpdpbusd`, `vdpbf16ps`), which reshapes the accumulation
    /// rounding, something `accumulate_tile`'s contract forbids. `a`/`b` are the family's
    /// interleaved panels, laid out by contract between the family's packers and the
    /// overriding token. `kc` is the real (unpadded) depth; the token reads
    /// `ceil(kc / DEPTH_MULTIPLE)` instruction-groups from the depth-padded panel. Any
    /// signedness or bias correction (VNNI's `+128`) is applied internally, so `acc` holds
    /// the true `sum_k(A*B)` on return
    ///
    /// The default is unreachable: only a dot-capable token (e.g. `Avx512Vnni`,
    /// `Avx512Bf16`) overrides it, and only a dot family ever calls it
    ///
    /// # Safety
    /// `a`/`b` valid for the family's packed panel at this `(MR_REG, NR, kc)`; `acc`
    /// pre-initialized. Run inside this token's [`Simd::vectorize`] context
    #[inline(always)]
    unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
        self,
        _kc: usize,
        _a: *const L,
        _b: *const R,
        _acc: &mut [[<Self as SimdOps<A>>::Reg; MR_REG]; NR],
    ) {
        unreachable!("dot_accumulate is provided only by dot-capable ISA tokens")
    }

    /// `true` iff [`Self::requant_store`] is a genuine vector implementation, rather than
    /// the default `unreachable!` stub. The requantizing epilogue's vector store path is
    /// gated on this: a `false` token routes every element through the scalar map
    /// (`KRequantize::apply`) instead
    const REQUANT_VECTOR: bool = false;

    /// Vectorized `i32 -> i8` requantize store: clamp each `A`-accumulator lane through the
    /// exact requant map and write its low byte to `LANES` consecutive slots at `dst`. `dst`
    /// is a raw byte pointer regardless of `O`: a `u8` output casts its pointer to call this,
    /// which is bit-identical because the low byte of a value clamped into `[lo, hi]` reads
    /// the same whether the type is `i8` or `u8`. Only requant-vector-capable tokens
    /// ([`Self::REQUANT_VECTOR`] `= true`) override this; the default is `unreachable!`, the
    /// same seam pattern as [`Self::dot_accumulate`]
    ///
    /// # Contract: bit-for-bit agreement with the scalar map
    /// Each lane of an implementation is expected to: widen `i32 -> f64` (exact); multiply by
    /// `scale` widened `f32 -> f64` (exact widening, one IEEE multiply); round to nearest-even
    /// in hardware, which agrees with the scalar `round_ne_f64` because that function's `2^52`
    /// trick *is* roundTiesToEven below `2^52`, and above `2^52` every `f64` is already
    /// integral so hardware rounding is the identity there too; add `zp as f64`; clamp with
    /// `max(lo as f64)` then `min(hi as f64)`; convert `f64 -> i32`, exact since the value is
    /// now integral and inside `[lo, hi]`; store the low byte by truncation, never a saturating
    /// pack. That sequence equals the scalar `clamp(zp + round_ne(scale*v), lo, hi)` case by
    /// case:
    /// * `|t| < 2^52`: `t` is integral and exact, so the scalar `t as i64` is exact, its `zp`
    ///   add cannot saturate, and the `f64` `t + zp` is exact too since both stay far below
    ///   `2^53`: the 2 paths feed identical values into an identical clamp
    /// * `t >= 2^52`: both clamp to `hi` (scalar via a saturating `i64 + zp` then clamp; vector
    ///   via `f64 + zp` then `min(hi)`). Symmetrically `t <= -2^52` clamps both to `lo`
    /// * NaN cannot occur: the API validates `scale` finite and positive, and `v` is a finite
    ///   `i32`
    ///
    /// The caller supplies `v` already bias-added: SIMD `i32` add (`paddd`) wraps, matching the
    /// scalar `wrapping_add` it must agree with. `lo`/`hi` are parameters, `-128`/`127` for the
    /// `i8` output; the `u8` output phase reuses the same machinery with `(0, 255)`
    ///
    /// # Safety
    /// `dst` valid for `LANES` byte writes; run inside this token's [`Simd::vectorize`]
    #[inline(always)]
    unsafe fn requant_store(
        self,
        _dst: *mut i8,
        _v: <Self as SimdOps<A>>::Reg,
        _scale: f64,
        _zp: i32,
        _lo: i32,
        _hi: i32,
    ) {
        unreachable!("requant_store is provided only by requant-vector-capable ISA tokens")
    }
}

/// The `+128` bias the VNNI families add to the LHS: `vpdpbusd` multiplies an unsigned byte
/// by a signed byte, so packing A as `A + 128` turns the signed x signed GEMM product into
/// the unsigned x signed form the instruction computes. [`KernelSimd::dot_accumulate`] then
/// subtracts `VNNI_A_BIAS * sum_k(B)` per column to recover the true product. The pack
/// transform (`vnni_a_xform` in [`crate::kernel::int`]) and that correction must use the
/// same constant, hence one definition shared by both. It lives here at L0, beside the
/// `dot_accumulate` contract it corrects, so the kernel-side pack transform imports it
/// downward and this module keeps its no-upward-references invariant
#[cfg(feature = "int8")]
pub(crate) const VNNI_A_BIAS: i32 = 128;

/// Homogeneous blanket: when every family type equals the accumulator type there is
/// nothing to widen or narrow, so `load_lhs`/`splat_rhs`/`load_out`/`store_out` are
/// plain [`SimdOps`] load/splat/store and any homogeneous family (e.g.
/// `FloatGemm<f32>`/`FloatGemm<f64>`) needs zero per-ISA code to satisfy [`KernelSimd`]
impl<A: Scalar, S: SimdOps<A>> KernelSimd<A, A, A, A> for S {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.splat(v) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut A, v: <S as SimdOps<A>>::Reg) {
        unsafe { self.storeu(p, v) }
    }
}

/// Requantizing-integer blanket for a byte-typed output: `i8` inputs, `i32` accumulator, and an
/// `i8` (`[-128, 127]`) or `u8` (ONNX QLinearMatMul `[0, 255]` activation) output. This is the
/// seam the requantizing integer families ([`crate::kernel::IntGemmQ`] /
/// [`crate::kernel::IntGemmVnniQ`]) drive on. [`impl_requant_blanket!`] generates both output
/// variants as one delegating impl over every token that already provides the widen kernel
/// (`KernelSimd<i8, i8, i32, i32>`): the hot accumulate-side ops forward verbatim to that impl,
/// so e.g. `Avx512Vnni`'s `dot_accumulate` override flows through unchanged, and `requant_store`
/// forwards too, since it already takes a raw `*mut i8` byte pointer and writes each pre-clamped
/// lane's low byte, the same byte whether read back as `i8` or `u8` (the cast
/// `KRequantize::apply_store` relies on)
///
/// `load_out`/`store_out` are structurally unreachable here and stubbed with `unreachable!`,
/// the same satisfy-the-trait-only convention as the `dot_accumulate`/`requant_store` defaults
/// above: the family's [`crate::kernel::KernelFamily::microkernel`] drains every tile through the
/// requant epilogue's scratch/scalar path (`Epilogue::VECTOR = false`), and `beta` is always
/// `Zero` for these families, so C is never read through the `Out`-typed seam; the methods exist
/// only to satisfy the driver's `KernelSimd<Lhs, Rhs, Acc, Out>` bound. `Out` being a byte type
/// keeps this coherent with its neighbors: it cannot unify with the homogeneous blanket (which
/// needs all 4 types equal) or with the sibling byte blanket (whose `Out` is the other byte type)
#[cfg(feature = "int8")]
macro_rules! impl_requant_blanket {
    ($out:ty) => {
        impl<S: KernelSimd<i8, i8, i32, i32>> KernelSimd<i8, i8, i32, $out> for S {
            #[inline(always)]
            unsafe fn load_lhs(self, p: *const i8) -> <Self as SimdOps<i32>>::Reg {
                unsafe { <Self as KernelSimd<i8, i8, i32, i32>>::load_lhs(self, p) }
            }
            #[inline(always)]
            unsafe fn splat_rhs(self, v: i8) -> <Self as SimdOps<i32>>::Reg {
                unsafe { <Self as KernelSimd<i8, i8, i32, i32>>::splat_rhs(self, v) }
            }
            #[inline(always)]
            unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
                self,
                kc: usize,
                a: *const i8,
                b: *const i8,
                acc: &mut [[<Self as SimdOps<i32>>::Reg; MR_REG]; NR],
            ) {
                unsafe {
                    <Self as KernelSimd<i8, i8, i32, i32>>::dot_accumulate::<MR_REG, NR>(
                        self, kc, a, b, acc,
                    )
                }
            }
            // Forwards like `dot_accumulate` above: `dst` is a raw byte pointer either way, so the
            // vector map is identical whether `Out` is `i32` or a byte
            const REQUANT_VECTOR: bool = <S as KernelSimd<i8, i8, i32, i32>>::REQUANT_VECTOR;
            #[inline(always)]
            unsafe fn requant_store(
                self,
                dst: *mut i8,
                v: <Self as SimdOps<i32>>::Reg,
                scale: f64,
                zp: i32,
                lo: i32,
                hi: i32,
            ) {
                unsafe {
                    <S as KernelSimd<i8, i8, i32, i32>>::requant_store(
                        self, dst, v, scale, zp, lo, hi,
                    )
                }
            }
            // Output-side ops are unreachable on the requant path (see the type doc above)
            #[inline(always)]
            unsafe fn load_out(self, _p: *const $out) -> <Self as SimdOps<i32>>::Reg {
                unreachable!("requant families never touch Out-typed C")
            }
            #[inline(always)]
            unsafe fn store_out(self, _p: *mut $out, _v: <Self as SimdOps<i32>>::Reg) {
                unreachable!("requant families never touch Out-typed C")
            }
        }
    };
}

#[cfg(feature = "int8")]
impl_requant_blanket!(i8);
#[cfg(feature = "int8")]
impl_requant_blanket!(u8);

/// `f32`-output twin of the narrow mixed seam: `f16`/`bf16` inputs, `f32` accumulator, and an
/// `f32` output (`Out == Acc`). This is the seam the deep-contraction narrow twins
/// ([`crate::kernel::MixedGemmF32`] / [`crate::kernel::Bf16DotGemmF32`]) drive on when a large-`k`
/// narrow GEMM is re-blocked through an `f32` scratch buffer: the accumulate-side ops
/// (`load_lhs`/`splat_rhs` widen `f16 -> f32`, `dot_accumulate` folds pairs) forward verbatim to
/// the narrow `KernelSimd<f16, f16, f32, f16>` impl, so a token's override still applies and the
/// twin's accumulation is bit-identical to the narrow family's, while `load_out`/`store_out` are
/// a plain `f32` load/store since the C scratch is already `f32` and needs no widen or narrow.
/// This is written as 2 explicit impls, one per narrow type, rather than one blanket generic over
/// `N`: a `KernelSimd<N, N, f32, f32>` blanket generic in `N` would collide with the homogeneous
/// `<A, A, A, A>` blanket under the coherence check, since the compiler cannot rule out `N = f32`;
/// the concrete `f16`/`bf16` heads cannot unify with `<A, A, A, A>` (`f16 != f32`), so they are
/// coherent, the same trick as the concrete-type `impl_requant_blanket!` heads above
#[cfg(feature = "half")]
impl<S: KernelSimd<half::f16, half::f16, f32, half::f16>> KernelSimd<half::f16, half::f16, f32, f32>
    for S
{
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const half::f16) -> <Self as SimdOps<f32>>::Reg {
        unsafe { <Self as KernelSimd<half::f16, half::f16, f32, half::f16>>::load_lhs(self, p) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: half::f16) -> <Self as SimdOps<f32>>::Reg {
        unsafe { <Self as KernelSimd<half::f16, half::f16, f32, half::f16>>::splat_rhs(self, v) }
    }
    // Out == Acc == f32 here, so this is a plain load/store: nothing to widen or narrow
    #[inline(always)]
    unsafe fn load_out(self, p: *const f32) -> <Self as SimdOps<f32>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f32, v: <Self as SimdOps<f32>>::Reg) {
        unsafe { self.storeu(p, v) }
    }
    #[inline(always)]
    unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const half::f16,
        b: *const half::f16,
        acc: &mut [[<Self as SimdOps<f32>>::Reg; MR_REG]; NR],
    ) {
        unsafe {
            <Self as KernelSimd<half::f16, half::f16, f32, half::f16>>::dot_accumulate::<MR_REG, NR>(
                self, kc, a, b, acc,
            )
        }
    }
}

/// The `bf16` head of the `f32`-output twin above: its `dot_accumulate` forward carries a
/// token's `vdpbf16ps` override through to [`crate::kernel::Bf16DotGemmF32`]
#[cfg(feature = "half")]
impl<S: KernelSimd<half::bf16, half::bf16, f32, half::bf16>>
    KernelSimd<half::bf16, half::bf16, f32, f32> for S
{
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const half::bf16) -> <Self as SimdOps<f32>>::Reg {
        unsafe { <Self as KernelSimd<half::bf16, half::bf16, f32, half::bf16>>::load_lhs(self, p) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: half::bf16) -> <Self as SimdOps<f32>>::Reg {
        unsafe { <Self as KernelSimd<half::bf16, half::bf16, f32, half::bf16>>::splat_rhs(self, v) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f32) -> <Self as SimdOps<f32>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f32, v: <Self as SimdOps<f32>>::Reg) {
        unsafe { self.storeu(p, v) }
    }
    #[inline(always)]
    unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const half::bf16,
        b: *const half::bf16,
        acc: &mut [[<Self as SimdOps<f32>>::Reg; MR_REG]; NR],
    ) {
        unsafe {
            <Self as KernelSimd<half::bf16, half::bf16, f32, half::bf16>>::dot_accumulate::<
                MR_REG,
                NR,
            >(self, kc, a, b, acc)
        }
    }
}

/// An ISA token: a zero-sized marker naming a set of target features
///
/// Its only behavior is [`Simd::vectorize`], which establishes the
/// `#[target_feature]` codegen context for everything run inside it
pub trait Simd: Copy + Send + Sync + 'static {
    /// Run `f` with this ISA's target features enabled
    ///
    /// # Safety
    /// The caller must guarantee the current CPU actually supports this token's
    /// features (the runtime dispatcher checks this once, before dispatching)
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R;
}

/// The SIMD vocabulary for element type `T` under ISA token `Self`
///
/// Every method is `unsafe`: it assumes the target feature is already enabled in the
/// current codegen context (guaranteed by running inside [`Simd::vectorize`]) and that
/// any pointer arguments are valid for the access. Impls mark every method
/// `#[inline(always)]` so the intrinsics fold straight into the caller
pub trait SimdOps<T: Scalar>: Simd {
    /// The SIMD register type holding [`Self::LANES`] values of `T`
    type Reg: Copy;
    /// Number of `T` lanes per register
    const LANES: usize;
    /// Whether this ISA has a hardware lane-indexed FMA: broadcasting a multiplier
    /// straight out of a vector lane in one fused instruction (NEON `vfmaq_laneq`).
    /// When `true` the microkernel takes the lane path via [`Self::fma_bvec`] for a
    /// packed RHS, loading a block of `LANES` B columns as one vector instead of
    /// issuing a separate `splat` load per column. The default is `false`: per-column
    /// `splat` plus FMA, which on x86 the assembler already folds into a
    /// broadcast-from-memory operand, so the lane path buys nothing there
    const LANE_FMA: bool = false;

    /// A register of all zeros
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn zero(self) -> Self::Reg;
    /// Broadcast a scalar into every lane
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn splat(self, v: T) -> Self::Reg;
    /// Unaligned load of [`Self::LANES`] contiguous values
    ///
    /// # Safety
    /// `p` must be valid for `LANES` reads
    unsafe fn loadu(self, p: *const T) -> Self::Reg;
    /// Unaligned store of [`Self::LANES`] contiguous values
    ///
    /// # Safety
    /// `p` must be valid for `LANES` writes
    unsafe fn storeu(self, p: *mut T, v: Self::Reg);
    /// Lane-wise multiply
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise add
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise fused multiply-add `a * b + c` (a true hardware FMA where available)
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg;
    /// Lane-wise fused negative-multiply-add `c - a * b` (a true hardware FMA where
    /// available: x86 `fnmadd`, NEON `vfms`). The subtractive partner of [`Self::mul_add`]
    /// that the split (SoA) complex kernel needs for its `acc_re -= a_im * b_im` term: it
    /// rounds `c - a*b` in a single step, so it stays consistent with `mul_add`'s single
    /// rounding when the 2 accumulation chains interleave
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg;
    /// Horizontal sum of every lane, in this token's fixed lane order (used by the
    /// gemv and dot-product epilogues)
    ///
    /// # Safety
    /// See the trait-level note
    unsafe fn reduce_sum(self, v: Self::Reg) -> T;

    /// Lane-wise maximum. Contract: in any lane where `a` is `NaN`, the result is `b`'s
    /// lane. Fused-epilogue call sites always pass a finite splat or zero as `b`
    /// (`max(v, zero)`), so a `NaN` accumulator lane maps to that finite operand,
    /// giving `ReLU(NaN) = 0` and matching the scalar edge path's
    /// `if a > b { a } else { b }` bit-for-bit (a `NaN > b` comparison is always `false`)
    ///
    /// The default is unreachable: only the real-float (`f32`/`f64`) tokens override it,
    /// and only the fused float epilogue ever calls it, the same seam pattern as
    /// [`KernelSimd::dot_accumulate`]
    ///
    /// # Safety
    /// See the trait-level note
    #[inline(always)]
    unsafe fn max(self, _a: Self::Reg, _b: Self::Reg) -> Self::Reg {
        unreachable!("max is provided only by the real-float SimdOps tokens")
    }
    /// Lane-wise minimum, with the same NaN-in-`a` contract as [`Self::max`]: the call
    /// site passes `min(v, zero)`, so a `NaN` lane returns the finite `b`
    ///
    /// # Safety
    /// See the trait-level note
    #[inline(always)]
    unsafe fn min(self, _a: Self::Reg, _b: Self::Reg) -> Self::Reg {
        unreachable!("min is provided only by the real-float SimdOps tokens")
    }

    /// Accumulate one contiguous block of `B` columns, loaded as the single register
    /// `bvec`, against the `MR_REG` already-loaded `A` registers, broadcasting each `B`
    /// lane: for `l in 0..LANES` and `i in 0..MR_REG`,
    /// `acc[l][i] = a_regs[i] * bvec[l] + acc[l][i]`. `acc.len()` must be exactly
    /// `LANES`: the caller only ever hands it whole `B` registers (the kernel path that
    /// calls this is gated on `NR` being a multiple of `LANES`), so an override is free
    /// to hard-code its lane count
    ///
    /// This is the fused inner step of the lane-indexed kernel path, taken only when
    /// [`Self::LANE_FMA`] is set. The default spills `bvec` to the stack and broadcasts
    /// each lane through [`Self::splat`], which is correct on any ISA but no faster than
    /// the plain `splat` path; lane-capable ISAs override it with a single hardware
    /// lane-indexed FMA instead. Either way it performs the same fused `a*b + c` as the
    /// `splat` path, so the 2 paths round consistently within a run
    ///
    /// # Safety
    /// See the trait-level note; `acc.len()` must be exactly `LANES` and
    /// `a_regs` valid for `MR_REG` reads
    #[inline(always)]
    unsafe fn fma_bvec<const MR_REG: usize>(
        self,
        a_regs: &[Self::Reg; MR_REG],
        bvec: Self::Reg,
        acc: &mut [[Self::Reg; MR_REG]],
    ) {
        debug_assert_eq!(acc.len(), Self::LANES);
        unsafe {
            // 16 is the widest LANES across every ISA (AVX-512F f32), so this buffer
            // always fits the register being spilled regardless of the caller's token
            let mut buf = [T::ZERO; 16];
            self.storeu(buf.as_mut_ptr(), bvec);
            for l in 0..acc.len() {
                let bl = self.splat(buf[l]);
                for i in 0..MR_REG {
                    acc[l][i] = self.mul_add(a_regs[i], bl, acc[l][i]);
                }
            }
        }
    }

    /// Accumulate one full `MR_REG x NR` microtile over `kc` depth steps into the
    /// register-resident `acc` (pre-zeroed by the caller):
    /// `acc[j][i] += A[p][i] * B[p][j]` for every `p in 0..kc`, in ascending `p` with a
    /// fused multiply-add. This is the GEMM inner loop, and the single hottest piece of
    /// the library
    ///
    /// `a` points at the LHS micropanel (`a_cs` is the depth stride; rows are unit
    /// stride, `MR_REG` vectors of `LANES`); `b` points at the RHS panel (`b_rs` is the
    /// depth stride, `b_cs` the column stride: `(nr, 1)` when packed, `(rsb, csb)` when
    /// not)
    ///
    /// The default is the portable per-step schedule: one broadcast (`splat`) per RHS
    /// column, or the lane-indexed fast path ([`Self::fma_bvec`]) when [`Self::LANE_FMA`]
    /// is set, the RHS block is contiguous (`b_cs == 1`), and `NR` is a multiple of
    /// `LANES` so every `LANES`-wide column block is whole; otherwise the broadcast path
    /// runs
    ///
    /// Keep the default on any out-of-order core: on a wide OoO core LLVM already lowers
    /// it into the canonical register-blocked kernel that saturates the FMA pipes,
    /// scheduling the next step's loads in among the FMAs and unrolling the `kc` loop on
    /// its own
    ///
    /// Override only for a target whose generated schedule genuinely stalls in a way
    /// LLVM will not fix on its own, e.g. an in-order or narrow-OoO core, where explicitly
    /// hoisting the next step's loads (the textbook software pipeline) pays off because
    /// the hardware cannot reorder around it, or a scalable-vector ISA (SVE/SME, RVV)
    /// whose length is not a compile-time `LANES`, so the fixed-width loop needs
    /// rewriting outright. Both cases still do a per-element fused `a*b + c` in ascending
    /// `p`, so they round consistently with the edge path. Instructions that reshape the
    /// accumulation rounding itself (matrix or dot instructions: `bfmmla`, `sdot`, VNNI,
    /// `vdpbf16ps`) are out of scope for this seam: they arrive as a new
    /// [`crate::kernel::KernelFamily`] with a dedicated dot seam, which may round
    /// differently from the widen path within tolerance, rather than as an
    /// `accumulate_tile` override. Before keeping any override, prove it pays: check the
    /// disassembly for spills, confirm it stays deterministic and accurate to the same
    /// tolerance, and benchmark it, since a hand schedule is not guaranteed to help
    ///
    /// An override must stay deterministic and accurate to the same tolerance under a
    /// fixed config, and round consistently with the microkernel's edge path within a run
    /// (full and edge tiles of the same matrix must agree), but it need not be
    /// bitwise-identical to the default. The portable schedule keeps the ascending-`p`
    /// fused `a*b + c` order, and software pipelining reorders loads, never the
    /// arithmetic, so it trivially meets that bar. Called only for full tiles
    /// (`nr_eff == NR`); partial column tiles stay on the microkernel's edge path
    ///
    /// # Safety
    ///
    /// `a` valid for `MR_REG*LANES` rows x `kc` depth at stride `a_cs`; `b` valid for
    /// `NR` cols x `kc` depth at strides `b_rs`/`b_cs`; `acc` pre-initialized. Must run
    /// inside this token's [`Simd::vectorize`] context
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn accumulate_tile<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const T,
        a_cs: isize,
        b: *const T,
        b_rs: isize,
        b_cs: isize,
        acc: &mut [[Self::Reg; MR_REG]; NR],
    ) {
        let lanes = Self::LANES;
        unsafe {
            if Self::LANE_FMA && b_cs == 1 && NR.is_multiple_of(lanes) {
                // Lane-indexed fast path: load each contiguous lanes-wide RHS block as
                // one vector and fan it out via fused lane-indexed FMA, replacing NR
                // per-column splat loads with NR/lanes vector loads
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [Self::Reg; MR_REG] =
                        core::array::from_fn(|i| self.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for jb in (0..NR).step_by(lanes) {
                        let bvec = self.loadu(pb.add(jb));
                        self.fma_bvec(&a_regs, bvec, &mut acc[jb..jb + lanes]);
                    }
                }
            } else {
                // Splat path: one broadcast per RHS column, correct for any b_cs
                // (packed or unpacked) and the only full-tile path on ISAs without a
                // lane FMA. The const-bounded j loop fully unrolls
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [Self::Reg; MR_REG] =
                        core::array::from_fn(|i| self.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..NR {
                        let bj = self.splat(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = self.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            }
        }
    }

    /// Compute one `MR x NR` complex tile in the split (structure-of-arrays) layout and
    /// apply the complex `alpha`/`beta` epilogue. This is the complex analogue of
    /// [`Self::accumulate_tile`]: a per-ISA hot loop that lives on this L0 seam because it
    /// needs the real-valued intrinsics (`SimdOps<T::Real>`) the generic
    /// [`crate::kernel::ComplexGemm`] microkernel cannot name through its
    /// `KernelSimd<T, T, T, T>` bound. The default is unreachable: only the complex
    /// `SimdOps<Complex<_>>` impls override it, and each forwards to the shared,
    /// ISA-generic `complex::soa_microkernel`, which has the real ops available
    /// concretely. Alpha/beta state arrives as plain bools rather than the L4
    /// `AlphaStatus`/`BetaStatus` enums, since depending on those would be an upward
    /// dependency from this L0 seam
    ///
    /// * `a`/`b`: planar packed panels (the real plane then the imaginary plane per
    ///   depth step); `a_cs`/`b_rs` are their depth strides in complex elements (`mr`/`NR`)
    /// * `c`/`rsc`/`csc`: the interleaved output tile; `scratch` needs at least `2*mr*NR`
    ///   reals
    ///
    /// # Safety
    /// As [`crate::kernel::KernelFamily::microkernel`]; run inside [`Simd::vectorize`]
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn cplx_microkernel<const MR_REG: usize, const NR: usize>(
        self,
        _kc: usize,
        _alpha: T,
        _beta: T,
        _alpha_is_one: bool,
        _beta_is_zero: bool,
        _beta_is_one: bool,
        _a: *const T,
        _a_cs: isize,
        _b: *const T,
        _b_rs: isize,
        _c: *mut T,
        _rsc: isize,
        _csc: isize,
        _mr_eff: usize,
        _nr_eff: usize,
        _scratch: *mut T,
    ) {
        unreachable!("cplx_microkernel is provided only by the complex `SimdOps` impls")
    }
}

/// Direct unit test of the vectorized `requant_store` seam. For every runtime-available x86
/// vector-capable token, plus the aarch64 NEON baseline token, sweeps adversarial `i32`
/// accumulators x scale x zero-point x clamp bounds and asserts each stored byte equals an
/// independent scalar model of the map (std `round_ties_even`, not the kernel's `2^52` trick). The
/// `(0, 255)` bounds cover the `u8`-output phase. The oracle is the scalar model, never a captured
/// machine number, so the test is platform-independent. Gated to the arches that override
/// `requant_store` (x86 + NEON); on any other arch every token takes the scalar epilogue instead,
/// so the sweep would be vacuous and its helpers dead code
#[cfg(all(
    test,
    feature = "int8",
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
mod requant_store_tests {
    #![allow(clippy::needless_range_loop)]
    use super::{KernelSimd, SimdOps};

    /// Independent scalar model of one lane of `requant_store`, bias already folded into `v`:
    /// `low_byte(clamp(zp + round_ne(scale*v), lo, hi))`. Returns the stored low byte as `u8`,
    /// since for a value clamped into `[lo, hi]` the low byte reads the same as `i8` or `u8`
    fn scalar_low_byte(v: i32, scale: f64, zp: i32, lo: i32, hi: i32) -> u8 {
        let scaled = (v as f64 * scale).round_ties_even();
        let q = (scaled as i64).saturating_add(zp as i64);
        q.clamp(lo as i64, hi as i64) as u8
    }

    /// Sweep one token. The `KernelSimd<i8, i8, i32, i32>` bound brings in `REQUANT_VECTOR` and
    /// `requant_store` directly, and `loadu`/`LANES` through its `SimdOps<i32>` supertrait
    ///
    /// # Safety
    /// The caller guarantees the CPU supports `S`'s target features (checked in the `#[test]`)
    unsafe fn check_token<S: KernelSimd<i8, i8, i32, i32>>(simd: S, label: &str) {
        unsafe {
            simd.vectorize(|| {
                let lanes = <S as SimdOps<i32>>::LANES;
                assert!(
                    <S as KernelSimd<i8, i8, i32, i32>>::REQUANT_VECTOR,
                    "{label}: token is not requant-vector-capable",
                );

                // Adversarial accumulators (already bias-folded), plus an LCG-generated random tail
                let mut vals: Vec<i32> = vec![
                    i32::MIN,
                    i32::MIN + 1,
                    -1,
                    0,
                    1,
                    1 << 30,
                    i32::MAX - 1,
                    i32::MAX,
                ];
                let mut lcg = 0x1234_5678_9abc_def0u64;
                for _ in 0..96 {
                    lcg = lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    vals.push((lcg >> 32) as i32);
                }

                for &scale in &[1.0f64, 0.1, 1e30, 1e-30, 0.0078125] {
                    for &zp in &[0i32, -128, 127] {
                        for &(lo, hi) in &[(-128i32, 127i32), (0i32, 255i32)] {
                            let mut idx = 0;
                            while idx < vals.len() {
                                // A full lanes-wide register, tail padded with 0
                                let mut inbuf = [0i32; 16];
                                for l in 0..lanes {
                                    inbuf[l] = vals.get(idx + l).copied().unwrap_or(0);
                                }
                                let reg = simd.loadu(inbuf.as_ptr());
                                let mut out = [0i8; 16];
                                <S as KernelSimd<i8, i8, i32, i32>>::requant_store(
                                    simd,
                                    out.as_mut_ptr(),
                                    reg,
                                    scale,
                                    zp,
                                    lo,
                                    hi,
                                );
                                for l in 0..lanes {
                                    let want = scalar_low_byte(inbuf[l], scale, zp, lo, hi);
                                    assert_eq!(
                                        out[l] as u8, want,
                                        "{label}: v={} scale={scale} zp={zp} bounds=({lo},{hi})",
                                        inbuf[l],
                                    );
                                }
                                idx += lanes;
                            }
                        }
                    }
                }
            });
        }
    }

    #[test]
    fn requant_store_matches_scalar_map() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: each token runs only after `is_x86_feature_detected!` confirms its features
        unsafe {
            use super::{Avx512F, Avx512Vnni, Fma};
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                check_token(Fma, "fma");
            }
            if is_x86_feature_detected!("avx512f") {
                check_token(Avx512F, "avx512f");
            }
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
            {
                check_token(Avx512Vnni, "avx512vnni");
            }
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline and mandatory on aarch64, so its features are always present
        unsafe {
            use super::Neon;
            check_token(Neon, "neon");
        }
    }
}
