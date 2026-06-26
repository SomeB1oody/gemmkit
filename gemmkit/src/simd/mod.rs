//! SIMD abstraction (layer L0): the load-bearing wall of the library.
//!
//! This module is **self-contained**: it depends only on [`crate::scalar`] and
//! `core`, never on the kernel/driver/cache layers above it. That zero
//! reverse-dependency property is deliberate so the whole module could later be
//! split into its own crate unchanged.
//!
//! # The two traits
//!
//! * [`Simd`] — an ISA *token* (a zero-sized type like [`Fma`]). It is not
//!   parameterized by element type. Its sole job is [`Simd::vectorize`], the
//!   `#[target_feature]` boundary (see below).
//! * [`SimdOps<T>`] — the *thick* per-element-type vocabulary of a token: the
//!   register type, lane count, and every primitive the microkernel needs
//!   (load/store/broadcast/mul/add/fma/reduce). Because the token and the
//!   element type are decoupled, `LANES` varies with the `(ISA, T)` pair
//!   (`f32`@FMA = 8, `f32`@AVX-512 = 16, `f64` halved).
//!
//! This is the answer to matrixmultiply's thin-trait trap: *every* primitive the
//! kernel needs is here, so the kernel is **one** generic function over all ISAs.
//! Adding an ISA = a new token + its `SimdOps` impls + one dispatch line.
//!
//! # `#[target_feature]` correctness
//!
//! AVX/AVX-512 intrinsics must be code-generated in a context where the feature
//! is enabled. CPU support is decided at *runtime* (by the dispatch layer), so
//! we cannot put a fixed `#[target_feature]` on the generic kernel. Instead each
//! token's [`Simd::vectorize`] runs a closure inside a tiny
//! `#[target_feature]`-annotated function; the closure (and the `#[inline]`
//! primitives it calls) inline into that function, so all intrinsics land in a
//! feature-enabled codegen context. This is the proven pulp/faer pattern, and it
//! works for both the serial path and rayon worker closures.

use crate::scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod fma;
mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx512::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::fma::Fma;
pub use self::scalar::ScalarTok;

/// An ISA token: a zero-sized marker carrying a set of target features.
///
/// The only behaviour is [`Simd::vectorize`], which establishes the
/// `#[target_feature]` codegen context for everything it runs.
pub trait Simd: Copy + Send + Sync + 'static {
    /// Run `f` with this ISA's target features enabled.
    ///
    /// # Safety
    /// The caller must guarantee the current CPU actually supports this token's
    /// features (the runtime dispatcher does this once).
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R;
}

/// The thick SIMD vocabulary for element type `T` under ISA token `Self`.
///
/// All methods are `unsafe`: they assume (a) the target feature is enabled in
/// the current codegen context (guaranteed by running inside
/// [`Simd::vectorize`]) and (b) any pointers are valid for the access. They are
/// all `#[inline(always)]` in the impls so the intrinsics fold into the kernel.
pub trait SimdOps<T: Scalar>: Simd {
    /// The SIMD register type holding [`Self::LANES`] values of `T`.
    type Reg: Copy;
    /// Number of `T` lanes per register.
    const LANES: usize;
    /// Natural buffer alignment for this ISA in bytes (e.g. 32 for AVX2, 64 for
    /// AVX-512). Packed buffers are aligned to this.
    const ALIGN: usize;

    /// A register of all zeros.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn zero(self) -> Self::Reg;
    /// Broadcast a scalar into every lane.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn splat(self, v: T) -> Self::Reg;
    /// Aligned load of [`Self::LANES`] contiguous values.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` reads and aligned to [`Self::ALIGN`].
    unsafe fn load(self, p: *const T) -> Self::Reg;
    /// Unaligned load of [`Self::LANES`] contiguous values.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` reads.
    unsafe fn loadu(self, p: *const T) -> Self::Reg;
    /// Aligned store.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` writes and aligned to [`Self::ALIGN`].
    unsafe fn store(self, p: *mut T, v: Self::Reg);
    /// Unaligned store.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` writes.
    unsafe fn storeu(self, p: *mut T, v: Self::Reg);
    /// Lane-wise multiply.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise add.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise fused multiply-add `a * b + c` (true FMA where available).
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg;
    /// Horizontal sum of all lanes (used by gemv / dot epilogues).
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn reduce_sum(self, v: Self::Reg) -> T;
}
