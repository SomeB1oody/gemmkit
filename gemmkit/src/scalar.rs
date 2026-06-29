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

// f16 / bf16 are the *mixed-precision* element types: they are stored as 16-bit
// inputs/outputs but **accumulate in `f32`** (`Acc = f32`), so they are `Scalar`
// but deliberately **not** [`Float`] — they carry no native arithmetic. The kernel
// widens them to `f32` on load and rounds back on store; the arithmetic happens in
// `f32` via [`SimdOps`] and the [`NarrowFloat`] scalar conversions below. This is
// the first place `Acc != Self`, the seam the mixed-precision family relies on.
#[cfg(feature = "half")]
impl Scalar for half::f16 {
    type Acc = f32;
    const ZERO: Self = half::f16::from_bits(0x0000);
    const ONE: Self = half::f16::from_bits(0x3C00);
}

#[cfg(feature = "half")]
impl Scalar for half::bf16 {
    type Acc = f32;
    const ZERO: Self = half::bf16::from_bits(0x0000);
    const ONE: Self = half::bf16::from_bits(0x3F80);
}

// Integer GEMM element types: `i8` inputs accumulate in `i32` (`Acc = i32`), and
// `i32` is its own accumulator/output. Like the narrow floats, `i8` is `Scalar` but
// not `Float` — the integer family widens `i8` to `i32` on load and does exact `i32`
// arithmetic (wrapping on overflow, the conventional integer-GEMM semantics).
#[cfg(feature = "int8")]
impl Scalar for i8 {
    type Acc = i32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

#[cfg(feature = "int8")]
impl Scalar for i32 {
    type Acc = i32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

// Complex GEMM elements: `Complex<f32>` / `Complex<f64>` are *homogeneous*
// (`Acc = Self`) and, unlike the narrow / integer types, have native arithmetic
// (`num-complex` provides Add/Mul/Sub), so they impl [`Float`] and ride the entire
// float path. The vectorized complex multiply lives in the per-ISA `SimdOps`; conj
// is a packing-time family variant, not a kernel branch.
#[cfg(feature = "complex")]
impl Scalar for num_complex::Complex<f32> {
    type Acc = Self;
    const ZERO: Self = num_complex::Complex::new(0.0, 0.0);
    const ONE: Self = num_complex::Complex::new(1.0, 0.0);
}

#[cfg(feature = "complex")]
impl Scalar for num_complex::Complex<f64> {
    type Acc = Self;
    const ZERO: Self = num_complex::Complex::new(0.0, 0.0);
    const ONE: Self = num_complex::Complex::new(1.0, 0.0);
}

#[cfg(feature = "complex")]
impl Float for num_complex::Complex<f32> {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Plain complex `a*b + c` (num-complex's `*` and `+`); the scalar epilogue
        // stays reproducible and matches the non-FMA fallback.
        self * b + c
    }
}

#[cfg(feature = "complex")]
impl Float for num_complex::Complex<f64> {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}

/// Conjugation, the operation that distinguishes the complex GEMM op-family
/// variants (`conjA` / `conjB`). A complex family applies it **at pack time** —
/// conjugating the packed `A` or `B` panel — so the hot loop stays a single plain
/// complex FMA with no per-element conj branch. Implemented only for the complex
/// types (a real type would just be identity, but no real family needs it).
#[cfg(feature = "complex")]
pub trait Conjugate: Scalar {
    /// The complex conjugate (`re - im·i`).
    fn conjugate(self) -> Self;
}

#[cfg(feature = "complex")]
impl Conjugate for num_complex::Complex<f32> {
    #[inline(always)]
    fn conjugate(self) -> Self {
        self.conj()
    }
}

#[cfg(feature = "complex")]
impl Conjugate for num_complex::Complex<f64> {
    #[inline(always)]
    fn conjugate(self) -> Self {
        self.conj()
    }
}

/// A narrow floating-point input type that accumulates in `f32` (`Acc = f32`):
/// `f16` and `bf16`. It supplies only the **scalar** widen / narrow conversions the
/// kernel epilogue's strided copy-back path needs; the vectorized hot loop uses the
/// SIMD widen-load / narrow-store on [`crate::simd::SimdOps`] instead. Kept separate
/// from [`Float`] precisely because these types have no native arithmetic — the
/// point of mixed precision.
#[cfg(feature = "half")]
pub trait NarrowFloat: Scalar<Acc = f32> {
    /// Widen one value to `f32` (exact — `f16`/`bf16` are a subset of `f32`).
    fn widen(self) -> f32;
    /// Round one `f32` to this narrow type (round-to-nearest-even).
    fn narrow(x: f32) -> Self;
}

#[cfg(feature = "half")]
impl NarrowFloat for half::f16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn narrow(x: f32) -> Self {
        half::f16::from_f32(x)
    }
}

#[cfg(feature = "half")]
impl NarrowFloat for half::bf16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn narrow(x: f32) -> Self {
        half::bf16::from_f32(x)
    }
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
