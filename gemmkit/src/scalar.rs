//! Numeric scalar abstraction (layer L0).
//!
//! [`Scalar`] is the *data-type* seam of the library. It is deliberately tiny:
//! it carries only the identity constants and the mixed-precision accumulator
//! type. All real arithmetic lives in [`crate::simd::SimdOps`] (vectorized) and
//! in the per-family kernels (scalar epilogues), never on this trait — so adding
//! a new element type (`f16`, `bf16`, complex, integer) does not force a new set
//! of arithmetic methods here.

/// A numeric element type that gemmkit can multiply.
///
/// The associated [`Scalar::Acc`] type is the *mixed-precision accumulator*
/// seam: for `f32`/`f64` it is simply `Self` (the branch collapses at compile
/// time), but a future `f16` would set `Acc = f32` so products accumulate in
/// higher precision. `Acc::Acc == Acc` keeps the recursion well-founded.
pub trait Scalar: Copy + Send + Sync + PartialEq + 'static {
    /// The type in which products are accumulated. `Self` for `f32`/`f64`.
    type Acc: Scalar<Acc = Self::Acc>;
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
}

impl Scalar for f32 {
    type Acc = f32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Scalar for f64 {
    type Acc = f64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

/// Real floating-point elements: a `Scalar` that additionally supports the
/// scalar arithmetic the kernel epilogues need (`alpha`/`beta` scaling and the
/// strided copy-back path). Implemented for `f32` and `f64`.
///
/// This is a separate trait so that [`Scalar`] itself stays free of arithmetic;
/// a future complex or integer [`crate::kernel::KernelFamily`] would use its own
/// arithmetic trait without touching `Scalar`.
pub trait Float:
    Scalar
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Sub<Output = Self>
{
    /// Fused (or emulated) `self * b + c`, used in scalar epilogues.
    fn mul_add(self, b: Self, c: Self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Plain `a*b + c` (not the hardware FMA) so the scalar reference path is
        // reproducible and matches the non-FMA fallback kernel bit-for-bit.
        self * b + c
    }
}

impl Float for f64 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
