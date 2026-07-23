# Element Types

gemmkit multiplies more than `f32`. The same engine, driver, and blocking model serve four element-type families; each is a Cargo feature and each has a SIMD implementation on every backend over the portable scalar fallback. What changes between them is the input type, the accumulator type, and the output type - and, with that, the accuracy you should expect. This page is the map of what is available and how precise it is.

## The built-in real floats

`f32` and `f64` need no feature flag. They go through the generic `gemm` (and `gemm_with`, and the unchecked entries), accumulate in their own type, and are the baseline every other family is measured against:

```rust
use gemmkit::{gemm, MatMut, MatRef, Parallelism};

let a = [1.0_f64, 2.0, 3.0, 4.0];
let b = [5.0_f64, 6.0, 7.0, 8.0];
let mut c = [0.0_f64; 4];
gemm(2.0, MatRef::from_row_major(&a, 2, 2), MatRef::from_row_major(&b, 2, 2),
     0.0, MatMut::from_row_major(&mut c, 2, 2), Parallelism::Serial);
```

Accuracy is the textbook GEMM story: relative error grows roughly with the contraction depth `k` and the machine epsilon of the type. The correctness suite holds results to a relative Frobenius gate of `8*k*eps` against an independent `f64` reference, so `f64` is near-exact for any realistic `k` and `f32` carries its usual `~1e-7` per-element relative precision.

## Narrow floats: the `half` feature

With `half` on, `f16` and `bf16` become element types (re-exported as `gemmkit::f16` / `gemmkit::bf16`, so you need not depend on `half` directly). They share the generic `gemm` surface - `MatRef<'_, f16>` in, `MatMut<'_, f16>` out - because they implement the same scalar trait as the real floats. The defining property is mixed precision: **inputs are widened to `f32` on load, the entire contraction accumulates in `f32`, and the result is rounded back to the narrow type exactly once, at the store.** There is no repeated narrow rounding inside the `k`-loop, which is what keeps the accuracy usable.

```rust
use gemmkit::{f16, gemm, MatMut, MatRef, Parallelism};

let a: Vec<f16> = (0..6).map(|i| f16::from_f32(i as f32)).collect();
let b: Vec<f16> = (0..6).map(|i| f16::from_f32(i as f32)).collect();
let mut c = vec![f16::ZERO; 4];
gemm(f16::ONE, MatRef::from_row_major(&a, 2, 3), MatRef::from_row_major(&b, 3, 2),
     f16::ZERO, MatMut::from_row_major(&mut c, 2, 2), Parallelism::Serial);
```

Because the accumulation is in `f32`, the dominant error is that single final round, not the sum: `f16` carries about `9.8e-4` (2^-10) relative precision and `bf16` about `7.8e-3` (2^-7), essentially independent of `k`. A narrow-precision GEMM is therefore close to "do it in `f32`, then round once" - far more accurate than accumulating in 16 bits would be.

One consequence of "round once" is that at large `k` a single depth panel would stream an L2-overflowing intermediate. The engine handles this itself: past an auto-derived byte gate it switches to an `f32`-output internal twin that re-blocks the contraction to stay cache-resident and narrows at the end, byte-for-byte identical to the single panel for the common `beta in {0, 1}` and within tolerance otherwise. This is automatic and needs no configuration; the mechanism is detailed in [Dot Kernels and the Deep-K Twin](../architecture/Dot_Kernels_and_the_Deep-K_Twin.md). On AVX-512 BF16 hardware, `bf16` additionally uses the `vdpbf16ps` dot kernel; see [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md).

## Integer: the `int8` feature

`int8` adds `gemm_i8`, a separate entry because the input and output types differ - `i8` in, `i32` out - which the homogeneous `gemm<T>` cannot express. `alpha`, `beta`, and `C` are all `i32`:

```rust
use gemmkit::{gemm_i8, MatMut, MatRef, Parallelism};

let a = [1_i8, 2, 3, 4, 5, 6];
let b = [7_i8, 8, 9, 10, 11, 12];
let mut c = [0_i32; 4];
gemm_i8(1, MatRef::from_row_major(&a, 2, 3), MatRef::from_row_major(&b, 3, 2),
        0, MatMut::from_row_major(&mut c, 2, 2), Parallelism::Serial);
```

Integer GEMM is **exact**: it is `i32` ring arithmetic that **wraps on overflow**, the conventional integer-GEMM semantics. There is no tolerance to speak of, because there is no rounding - the result is bit-for-bit identical across every ISA (scalar, FMA, AVX-512F, and the AVX-512 VNNI `vpdpbusd` dot kernel), and identical serial versus parallel, since integer addition over a ring is order-independent. If you feed values whose products can exceed `i32`, the wraparound is defined and reproducible, not undefined behavior. The `int8` feature pulls in no extra dependency. Adding `epilogue` on top unlocks the requantizing entries (`i8`/`u8` output in one pass); see [Fused Epilogues](Fused_Epilogues.md).

## Complex: the `complex` feature

`complex` adds `gemm_cplx` over `num-complex` values, re-exported as `gemmkit::c32` (`Complex<f32>`) and `gemmkit::c64` (`Complex<f64>`). Its signature carries a conjugation flag for each operand:

```rust
use gemmkit::{c32, gemm_cplx, Complex, MatMut, MatRef, Parallelism};

let a = [Complex::new(1.0_f32, 1.0), Complex::new(2.0, 0.0)];
let b = [Complex::new(0.0_f32, 1.0), Complex::new(1.0, 0.0)];
let mut c = [c32::default(); 1];
gemm_cplx(
    Complex::new(1.0, 0.0),
    MatRef::from_row_major(&a, 1, 2), false, // conj_a
    MatRef::from_row_major(&b, 2, 1), false, // conj_b
    Complex::new(0.0, 0.0),
    MatMut::from_row_major(&mut c, 1, 1),
    Parallelism::Serial,
);
```

The computation is `C <- alpha*op(A)*op(B) + beta*C`, where `op(A)` is `conj(A)` when `conj_a` is set (and likewise `conj_b`); passing `false, false` is the plain product `A*B`. The flags conjugate the operands only. Complex accumulates in its own type and is held to a relative Frobenius gate of `16*k*eps` (with `eps` the real component's epsilon), so a `c32` GEMM is about as accurate as an `f32` one and a `c64` GEMM about as accurate as `f64`. `complex` pulls in `num-complex`. Internally, complex does not ride the float kernel - it uses a dedicated split (structure-of-arrays) kernel - which is why it is a separate entry; that design is covered in [The Complex Split Kernel](../architecture/The_Complex_Split_Kernel.md).

## Choosing a type

| Family | Feature | In / Acc / Out | Accuracy | Determinism |
| --- | --- | --- | --- | --- |
| `f32`, `f64` | (built in) | same / same / same | textbook, `~8*k*eps` | reproducible; bit-exact serial=parallel on driver paths |
| `f16`, `bf16` | `half` | narrow / `f32` / narrow | one final round; `~1e-3` (f16), `~8e-3` (bf16) | reproducible; deep-k twin bit-exact for `beta in {0,1}` |
| `i8` | `int8` | `i8` / `i32` / `i32` | exact, wrapping `i32` | bit-identical across every ISA and worker count |
| `c32`, `c64` | `complex` | same / same / same | `~16*k*eps` | reproducible; bit-exact serial=parallel |

If you need speed and can tolerate `~1e-3` precision, `bf16`/`f16` roughly halve memory traffic while accumulating in `f32`. If you need exactness, `int8` gives it. If you need range and precision, stay on `f32`/`f64`. The reproducibility contract is the same for all of them - see [Parallelism in Practice](Parallelism_in_Practice.md) for what "reproducible" does and does not promise.

## Where to next

- [Fused Epilogues](Fused_Epilogues.md) - bias, activation, requantization, and per-element maps fused into the store.
- [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md) - the VNNI and BF16 dot kernels these types can reach.
- [Matrix Views and Layouts](Matrix_Views_and_Layouts.md) - constructing the views every family shares.
