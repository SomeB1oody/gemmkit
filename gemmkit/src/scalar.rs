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

// f16 / bf16: 16-bit storage that accumulates in `f32` (`Acc = f32`), so they are
// `Scalar` but not [`Float`] — they carry no native arithmetic. Widened to `f32` on
// load and rounded back on store; arithmetic happens in `f32` via `SimdOps` and the
// [`NarrowFloat`] scalar conversions below. First place `Acc != Self`.
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

// Integer GEMM: `i8` accumulates in `i32` (`Acc = i32`); `i32` is its own
// accumulator. Like the narrow floats, `i8` is `Scalar` but not `Float` — widened to
// `i32` on load, exact `i32` arithmetic (wrapping on overflow, conventional for
// integer GEMM).
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

// Quantized *unsigned* output type: `u8` is the ONNX-QLinearMatMul-style activation
// output of the requantizing GEMM. Like `i8` it accumulates in `i32` (`Acc = i32`) and
// is widened *zero-extending* wherever a widen is ever needed; it is `Scalar` but not
// `Float` — it only ever appears as the requant `Out`, never as a GEMM input.
#[cfg(feature = "int8")]
impl Scalar for u8 {
    type Acc = i32;
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

// Complex GEMM: `Complex<f32>` / `Complex<f64>` have native arithmetic (`num-complex`
// provides Add/Mul/Sub), so they impl [`Float`] — used by the degenerate `C <- beta·C`
// scale and the complex `alpha`/`beta` epilogue. Their GEMM rides the dedicated split
// (SoA) complex kernel ([`ComplexFloat`] + [`crate::kernel::ComplexGemm`]), which
// accumulates in the real component type, not via a vectorized complex multiply.
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
        // Plain complex `a*b + c`; keeps the scalar epilogue reproducible and
        // matching the non-FMA fallback.
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

/// A complex element type (`Complex<f32>` / `Complex<f64>`) viewed as a pair of real
/// components, for the split-accumulator (SoA) complex kernel. It exposes the real
/// component type [`ComplexFloat::Real`] and the re/im accessors and constructor the
/// de-interleaving pack and the kernel epilogue need; conjugation is a plain negate of
/// the imaginary part, so it needs no separate trait.
#[cfg(feature = "complex")]
pub trait ComplexFloat: Float<Acc = Self> {
    /// The real component type (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
    type Real: Float<Acc = Self::Real>;
    /// The real part.
    fn re(self) -> Self::Real;
    /// The imaginary part.
    fn im(self) -> Self::Real;
    /// Assemble a complex value from its real and imaginary parts.
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

/// A narrow floating-point input type that accumulates in `f32` (`Acc = f32`):
/// `f16` and `bf16`. Supplies only the scalar widen / narrow conversions the kernel
/// epilogue's strided copy-back path needs; the hot loop uses the SIMD widen-load /
/// narrow-store on [`crate::simd::SimdOps`] instead. Separate from [`Float`] because
/// these types have no native arithmetic.
#[cfg(feature = "half")]
pub trait NarrowFloat: Scalar<Acc = f32> {
    /// Widen one value to `f32` (exact — `f16`/`bf16` are a subset of `f32`).
    fn widen(self) -> f32;
    /// Round one `f32` to this narrow type (round-to-nearest-even).
    fn narrow(x: f32) -> Self;
}

// `half`'s default `to_f32`/`from_f32` dispatch to a hardware conversion that, on
// aarch64, is inline `asm!` (see `half`'s `binary16/arch/aarch64.rs`). Miri cannot
// interpret inline asm, so under `cfg(miri)` these route to `half`'s own pure-software
// `*_const` conversions instead — bit-equivalent (the same IEEE round-to-nearest-even),
// keeping gemmkit's mixed-precision scalar pack/accumulate/epilogue exercisable under
// Miri. Real builds are unchanged: `not(miri)` keeps the fast hardware path.
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
    + core::ops::Neg<Output = Self>
{
    /// Fused (or emulated) `self * b + c`, used in scalar epilogues.
    fn mul_add(self, b: Self, c: Self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // Plain `a*b + c` (not the hardware FMA) so the scalar reference path is
        // reproducible and agrees with the non-FMA fallback kernel.
        self * b + c
    }
}

impl Float for f64 {
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
