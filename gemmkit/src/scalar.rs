//! The element-type seam (layer L0): what a value must supply to be a GEMM operand
//!
//! [`Scalar`] is deliberately thin: identity constants plus the accumulator type,
//! nothing else. Actual arithmetic lives on [`Float`] (real add/mul/sub/neg and
//! `mul_add`), on [`crate::simd::SimdOps`] (the vectorized path), and in the
//! per-family kernel epilogues, never on `Scalar` itself, so a new element type
//! can implement `Scalar` without dragging in a full arithmetic surface

/// An element type gemmkit can multiply: identity constants plus the accumulator
/// type products land in
///
/// [`Scalar::Acc`] is the mixed-precision seam. Wide types (`f32`, `f64`, `i32`,
/// `Complex<f32>`, `Complex<f64>`) set `Acc = Self`; narrow types (`f16`, `bf16`,
/// `i8`, `u8`) set `Acc` to a wider type they widen into before accumulating. The
/// `Acc: Scalar<Acc = Self::Acc>` bound forces `Acc` to be a fixed point of the
/// mapping, so the accumulator type never needs a 2nd, wider accumulator of its own
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

// f16/bf16 (16-bit storage) accumulate in f32: Acc = f32, the 1st pair in this
// file where Acc != Self. Neither implements Float below (no native add/mul/sub);
// values are widened to f32 on load and rounded back on store, via the scalar
// NarrowFloat conversions here and the vectorized widen/narrow on SimdOps
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

// i8 accumulates in i32 (Acc = i32); i32 is its own accumulator. Like f16/bf16,
// i8 has no Float impl: it widens to i32 on load and the kernel does exact i32
// arithmetic, wrapping on overflow (standard integer-GEMM semantics)
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

// u8 is the requantized (ONNX QLinearMatMul-style) output type, never a GEMM
// input: it only ever appears as the Out of the requantizing epilogue. Also
// accumulates in i32 (Acc = i32) for the same Scalar-bound reason as i8
#[cfg(feature = "int8")]
impl Scalar for u8 {
    type Acc = i32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

// Complex<f32>/Complex<f64> have native arithmetic via num-complex's operator
// impls, so unlike the other narrow/exotic types above they implement Float
// directly (Acc = Self). Their GEMM runs through the dedicated split (SoA)
// complex kernel (ComplexFloat below plus crate::kernel::ComplexGemm), which
// accumulates via the real component type rather than a vectorized complex
// multiply; Float::mul_add here backs only the scalar alpha/beta epilogue path
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
        // non-FMA fallback rather than rounding once like a true FMA would
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

/// A complex element (`Complex<f32>` / `Complex<f64>`) exposed as its real and
/// imaginary components, for the split-accumulator (SoA) complex kernel
///
/// Supplies the real component type and the re/im accessors and constructor the
/// de-interleaving pack and the kernel epilogue need. Conjugation has no accessor
/// here because it is just a negate of the imaginary part
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

/// A narrow float that accumulates in `f32` (`f16`, `bf16`): the scalar widen/narrow
/// conversions the kernel epilogue's strided copy-back path needs
///
/// The hot loop widens/narrows via SIMD on [`crate::simd::SimdOps`] instead; this
/// trait covers only the scalar tail. Kept separate from [`Float`] because `f16`
/// and `bf16` have no native arithmetic of their own to satisfy that trait
#[cfg(feature = "half")]
pub trait NarrowFloat: Scalar<Acc = f32> {
    /// Widen one value to `f32` (exact: `f16`/`bf16` are a strict subset of `f32`)
    fn widen(self) -> f32;
    /// Round one `f32` to this narrow type (round-to-nearest-even)
    fn narrow(x: f32) -> Self;
}

// `half`'s to_f32/from_f32 runtime-dispatch to a hardware conversion that, on
// aarch64 with the fp16 feature, is inline asm!. Miri cannot interpret inline
// asm, so under cfg(miri) these route to half's own *_const conversions
// instead: bit-equivalent per half's own docs (same round-to-nearest-even), so
// gemmkit's mixed-precision pack/accumulate/epilogue path stays exercisable
// under Miri. Non-Miri builds are unaffected: the hardware path is unchanged
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

/// A `Scalar` with the real arithmetic the kernel epilogues need: `alpha`/`beta`
/// scaling and the strided copy-back path. Implemented for `f32`, `f64`, and
/// (via `num-complex`'s own operators) `Complex<f32>`/`Complex<f64>`
///
/// Kept separate from [`Scalar`] so that trait stays free of arithmetic: the
/// integer family needs no arithmetic trait at all, and complex GEMM implements
/// `Float` via `num-complex`'s own operators rather than a hand-derived
/// Add/Mul/Sub/Neg
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
        // Unfused a*b + c (not the hardware FMA) so the scalar reference path
        // is reproducible and matches the non-FMA fallback kernel
        self * b + c
    }
}

impl Float for f64 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
