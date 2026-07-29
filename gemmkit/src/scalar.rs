//! The element-type seam (layer L0): what a value must supply to be a GEMM operand
//!
//! [`Scalar`] is deliberately thin: identity constants plus the accumulator type, and
//! nothing else. Arithmetic lives on [`Float`] (real add, multiply, subtract, negate,
//! and `mul_add`), on [`crate::simd::KernelSimd`] (the vectorized widen/narrow path),
//! and in the per-family kernel epilogues. It never lives on `Scalar` itself. This lets
//! a new element type implement `Scalar` without pulling in a full arithmetic surface

/// An element type gemmkit can use as a GEMM operand, supplying identity constants and
/// its accumulator type
///
/// [`Scalar::Acc`] is the mixed-precision seam. Wide types (`f32`, `f64`, `i32`,
/// `Complex<f32>`, `Complex<f64>`) set `Acc = Self`. Narrow types (`f16`, `bf16`, `i8`,
/// `u8`) set `Acc` to a wider type that they widen into before accumulating. The
/// `Acc: Scalar<Acc = Self::Acc>` bound forces `Acc` to be a fixed point of the mapping.
/// The accumulator type never needs a 2nd, wider accumulator of its own
pub trait Scalar: Copy + Send + Sync + PartialEq + 'static {
    /// The type products of `Self` accumulate in (`Self` for wide types)
    type Acc: Scalar<Acc = Self::Acc>;
    /// The additive identity
    const ZERO: Self;
    /// The multiplicative identity
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

// f16 and bf16 (16-bit storage) accumulate in f32, so `Acc = f32` here, the 1st pair
// in this file where `Acc != Self`. Neither implements `Float`, since neither has native
// add, multiply, or subtract. Each widens to f32 on load and narrows back on store,
// through `NarrowFloat` below and the vectorized path on `crate::simd::KernelSimd`
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

// i8 accumulates in i32, so `Acc = i32`, and i32 is its own accumulator. Like f16 and
// bf16, i8 has no `Float` impl. It widens to i32 on load, and the kernel does exact i32
// arithmetic that wraps on overflow, the standard integer-GEMM semantics
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

// u8 is the requantized (ONNX QLinearMatMul-style) output type, never a GEMM input
// It only appears as the `Out` type of the requantizing epilogue. It also
// accumulates in i32 (`Acc = i32`), for the same `Scalar` bound reason as i8
#[cfg(feature = "int8")]
impl Scalar for u8 {
    type Acc = i32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

// Complex<f32> and Complex<f64> have native arithmetic through num-complex's operator
// impls, so unlike the narrow types above, they implement `Float` directly
// (`Acc = Self`)
//
// Their GEMM runs through the dedicated split (SoA) complex kernel: `ComplexFloat`
// below, plus `crate::kernel::ComplexGemm`. That kernel accumulates through the real
// component type instead of a vectorized complex multiply. `Float::mul_add` here
// backs only the scalar alpha/beta epilogue path
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
        // Unfused a*b + c, so this scalar path stays reproducible against the
        // non-FMA fallback, instead of rounding once as a true FMA would
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

/// A complex element, `Complex<f32>` or `Complex<f64>`, exposed as its real and
/// imaginary components, for the split-accumulator (SoA) complex kernel
///
/// This trait supplies the real component type, the real and imaginary part
/// accessors, and the constructor that the de-interleaving pack and the kernel
/// epilogue need. Conjugation has no accessor here, because it is just a negation of
/// the imaginary part
#[cfg(feature = "complex")]
pub trait ComplexFloat: Float<Acc = Self> {
    /// The real component type (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`)
    type Real: Float<Acc = Self::Real>;
    /// The real part
    fn re(self) -> Self::Real;
    /// The imaginary part
    fn im(self) -> Self::Real;
    /// Assemble a complex value from its real and imaginary parts
    fn new(re: Self::Real, im: Self::Real) -> Self;
}

#[cfg(feature = "complex")]
impl ComplexFloat for num_complex::Complex<f32> {
    type Real = f32;
    #[inline(always)]
    fn re(self) -> f32 {
        self.re
    }
    #[inline(always)]
    fn im(self) -> f32 {
        self.im
    }
    #[inline(always)]
    fn new(re: f32, im: f32) -> Self {
        num_complex::Complex::new(re, im)
    }
}

#[cfg(feature = "complex")]
impl ComplexFloat for num_complex::Complex<f64> {
    type Real = f64;
    #[inline(always)]
    fn re(self) -> f64 {
        self.re
    }
    #[inline(always)]
    fn im(self) -> f64 {
        self.im
    }
    #[inline(always)]
    fn new(re: f64, im: f64) -> Self {
        num_complex::Complex::new(re, im)
    }
}

/// A narrow float that accumulates in `f32` (`f16`, `bf16`), exposing the scalar widen
/// and narrow conversions that the kernel epilogue's strided copy-back path needs
///
/// The hot loop widens and narrows through SIMD on [`crate::simd::KernelSimd`]
/// instead, so this trait covers only the scalar tail. It stays separate from
/// [`Float`] because `f16` and `bf16` have no native arithmetic of their own to
/// satisfy that trait
#[cfg(feature = "half")]
pub trait NarrowFloat: Scalar<Acc = f32> {
    /// Widen 1 value to `f32`, exact because `f16` and `bf16` are a strict subset
    /// of `f32`
    fn widen(self) -> f32;
    /// Round 1 `f32` value to this narrow type, using round-to-nearest-even
    fn narrow(x: f32) -> Self;
}

// `half`'s to_f32/from_f32 dispatch at runtime to a hardware conversion. On aarch64
// with the fp16 feature, that hardware path is inline `asm!`, and Miri cannot
// interpret inline assembly
//
// Under `cfg(miri)`, these route to half's own `*_const` conversions instead. Those
// conversions use the same round-to-nearest-even rounding as the hardware path, per
// half's own documentation. This keeps gemmkit's mixed-precision path exercisable
// under Miri without changing non-Miri builds
#[cfg(feature = "half")]
impl NarrowFloat for half::f16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        #[cfg(not(miri))]
        {
            self.to_f32()
        }
        #[cfg(miri)]
        {
            self.to_f32_const()
        }
    }
    #[inline(always)]
    fn narrow(x: f32) -> Self {
        #[cfg(not(miri))]
        {
            half::f16::from_f32(x)
        }
        #[cfg(miri)]
        {
            half::f16::from_f32_const(x)
        }
    }
}

#[cfg(feature = "half")]
impl NarrowFloat for half::bf16 {
    #[inline(always)]
    fn widen(self) -> f32 {
        #[cfg(not(miri))]
        {
            self.to_f32()
        }
        #[cfg(miri)]
        {
            self.to_f32_const()
        }
    }
    #[inline(always)]
    fn narrow(x: f32) -> Self {
        #[cfg(not(miri))]
        {
            half::bf16::from_f32(x)
        }
        #[cfg(miri)]
        {
            half::bf16::from_f32_const(x)
        }
    }
}

/// A [`Scalar`] with the real arithmetic that the kernel epilogues need:
/// `alpha`/`beta` scaling and the strided copy-back path
///
/// `f32`, `f64`, `Complex<f32>`, and `Complex<f64>` implement this trait. The complex
/// types use `num-complex`'s own operators to do so
///
/// `Float` stays separate from [`Scalar`], so `Scalar` stays free of arithmetic. The
/// integer family needs no arithmetic trait at all, and complex GEMM implements
/// `Float` through `num-complex`'s own operators instead of a hand-derived
/// `Add`/`Mul`/`Sub`/`Neg` set
pub trait Float:
    Scalar
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Fused (or emulated) `self * b + c`, used in scalar epilogues
    fn mul_add(self, b: Self, c: Self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Unfused a*b + c, not the hardware FMA, so this scalar reference path stays
        // reproducible and matches the non-FMA fallback kernel
        self * b + c
    }
}

impl Float for f64 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
