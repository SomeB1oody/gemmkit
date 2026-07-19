# nalgebra Adapter Advanced Usage

Beyond the real-scalar `gemm`/`gemm_with`/`dot` covered in [Using gemmkit with nalgebra](Using_gemmkit_with_nalgebra.md), the adapter exposes the engine's full surface behind Cargo features: integer GEMM, complex GEMM, fused epilogues, requantized output, per-element maps, prepacked operands, and batching. Every entry keeps the adapter's core promise, reading nalgebra's pointer and strides straight through with no copy, and mirrors a gemmkit core function of the same name. Each family is gated so you pay for only what you turn on.

| Feature | Adds |
|---|---|
| `half` | `f16`/`bf16` through the same `gemm`/`gemm_fused` generics |
| `int8` | `gemm_i8`, `gemm_i8_with`, `dot_i8` (`i8 -> i32`) |
| `complex` | `gemm_cplx`, `gemm_cplx_with`, `dot_cplx` |
| `epilogue` | `gemm_fused`, `gemm_map` (and prepacked-fused twins) |
| `int8` + `epilogue` | `gemm_i8_requant`, `gemm_i8_requant_u8` |
| `complex` + `epilogue` | `gemm_cplx_fused` |

The re-exported helper types (`Bias`, `Activation`, `RequantScale`, `Requantize`, `PackedLhs`, `PackedRhs`) come from the adapter crate itself, so you do not need to name `gemmkit` to use them. `Parallelism`, `Workspace`, and `Complex` still come from `gemmkit`.

## Integer GEMM

Under `int8`, `gemm_i8` multiplies two `i8` matrices into an `i32` output. The inputs are `i8`; `alpha`, `beta`, and `C` are `i32`, since an `i8*i8` product needs the wider accumulator. Arithmetic wraps on overflow, the conventional integer-GEMM semantics. It is a separate entry from `gemm` precisely because the input and output element types differ.

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{dot_i8, gemm_i8};
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 3, &[1_i8, 2, 3, 4, 5, 6]);
let b = DMatrix::from_row_slice(3, 2, &[1_i8, 0, 0, 1, 1, 1]);

// dot_i8: A*B into a fresh DMatrix<i32>
let c = dot_i8(&a, &b);

// gemm_i8: scale and accumulate into an i32 output
let mut acc = DMatrix::<i32>::zeros(2, 2);
gemm_i8(1, &a, &b, 0, &mut acc, Parallelism::Serial);
assert_eq!(acc, c);
```

`dot_i8(a, b) -> DMatrix<i32>` is the allocating convenience form, and `gemm_i8_with` reuses a caller-owned `Workspace` for the fixed-cost quantized-inference loop.

## Requantized output

Requantization folds the dequantize-scale-round-clamp step of quantized inference into the GEMM, so the `m*n` `i32` accumulator never has to be materialized in full. `gemm_i8_requant` takes `i8` inputs and writes an `i8` output; `gemm_i8_requant_u8` writes an unsigned `u8` output (the ONNX QLinearMatMul convention). Both need `int8` + `epilogue`. There is no `alpha` (it folds into the scale) and no `beta` (accumulating into a quantized `C` is ill-defined). The parameters ride in a re-exported `Requantize`:

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{RequantScale, Requantize, gemm_i8_requant};
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 3, &[10_i8, -4, 7, 3, 8, -2]);
let b = DMatrix::from_row_slice(3, 2, &[2_i8, 1, -1, 5, 4, 0]);
let bias = [100_i32, -50]; // per-row, length A.rows

let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,        // in [-128, 127] for the i8 output
    bias: Some(&bias),
};
let mut c = DMatrix::from_element(2, 2, 0_i8);
gemm_i8_requant(&a, &b, req, &mut c, Parallelism::Serial);
```

`RequantScale::PerTensor(s)` applies one scale to every element; `RequantScale::PerRow(&[f32])` gives one scale per output row (per output channel, the standard per-channel convention), length `A.rows`. Every scale must be finite and `> 0`. The `zero_point` is joined in integer after rounding and must lie in the output domain: `[-128, 127]` for `gemm_i8_requant`, `[0, 255]` for `gemm_i8_requant_u8`. The optional per-row `i32` bias, length `A.rows`, is added to the accumulator before scaling. The adapter validates all of this and panics with the core engine's wording on any violation (a non-finite or non-positive scale, a per-row scale or bias of the wrong length, a `zero_point` out of range, or a scale/bias slice that overlaps `C`).

## Complex GEMM

Under `complex`, `gemm_cplx` computes `C <- alpha*op(A)*op(B) + beta*C` for `T = Complex<f32>` or `Complex<f64>`, where `op(A) = conj(A)` when the `conj_a` flag is set and likewise for `conj_b`. The conjugation flags are why complex needs its own entry: they do not fit the homogeneous real-scalar signature. `dot_cplx(a, b)` is the non-conjugated `A*B` convenience form; for a conjugated product use `gemm_cplx` directly.

```rust
use gemmkit::{Complex, Parallelism};
use gemmkit_nalgebra::{dot_cplx, gemm_cplx};
use nalgebra::DMatrix;

type C = Complex<f64>;
let a = DMatrix::from_element(2, 2, C::new(1.0, 1.0));
let b = DMatrix::from_element(2, 2, C::new(0.0, -1.0));

// plain product
let p = dot_cplx(&a, &b);

// conjugate A, plain B, accumulate into an existing C
let mut acc = DMatrix::from_element(2, 2, C::new(0.0, 0.0));
gemm_cplx(C::new(1.0, 0.0), &a, true, &b, false,
          C::new(0.0, 0.0), &mut acc, Parallelism::Serial);
```

## Complex with a fused bias

`gemm_cplx_fused` (needs `complex` + `epilogue`) adds a bias in the same pass as the complex product: `C <- alpha*op(A)*op(B) + beta*C + bias`. The bias is a re-exported `Bias`, either `Bias::PerRow` (length `A.rows`) or `Bias::PerCol` (length `B.cols`), and it is added verbatim, never conjugated. There is deliberately no activation parameter here: an ordering activation like ReLU is undefined on complex numbers. With `bias == None` the call is exactly `gemm_cplx`.

## Fused epilogues

The `epilogue` feature adds `gemm_fused`, which computes `C <- act(alpha*A*B + beta*C + bias)` in a single pass over `f32`/`f64` (and `f16`/`bf16` when `half` is on, with the epilogue evaluated in `f32` before the one narrowing store). The optional `Bias` is `PerRow` (length `A.rows`) or `PerCol` (length `B.cols`); the optional `Activation` is `Relu` or `LeakyRelu(slope)`, applied last. Passing `None` for both is bit-for-bit identical to plain `gemm`.

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{Activation, Bias, gemm_fused};
use nalgebra::DMatrix;

let a = DMatrix::<f32>::from_element(12, 9, 0.5);
let b = DMatrix::<f32>::from_element(9, 7, -0.25);
let bias: Vec<f32> = (0..12).map(|i| 0.5 * i as f32 - 2.0).collect();
let mut c = DMatrix::<f32>::zeros(12, 7);

gemm_fused(1.3, &a, &b, -0.7, &mut c,
           Some(Bias::PerRow(&bias)), Some(Activation::Relu), Parallelism::Serial);
```

The fused pass is not just a convenience: it avoids a second sweep over `C` and the round trip through memory that a separate bias-add and activation would cost. For `f32`/`f64` the result is bit-identical to running `gemm` and then applying the same bias and activation element by element, so you can adopt it without changing numerical results. The design behind it is covered in [Fused Epilogues](../gemmkit-guide/Fused_Epilogues.md).

## Per-element map

For epilogues that do not fit a bias-plus-activation shape, `gemm_map` applies an arbitrary closure to each finished output element: `C[r, c] <- f(alpha*A*B + beta*C, r, c)`, fired exactly once per element, with `(r, c)` in the user frame of `C`. `T` is `f32`/`f64` only. The closure is `&(dyn Fn(T, usize, usize) -> T + Sync)`, so it must be `Sync` to run in parallel and can close over data by reference.

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::gemm_map;
use nalgebra::DMatrix;

let a = DMatrix::<f64>::from_element(8, 6, 0.3);
let b = DMatrix::<f64>::from_element(6, 5, 0.4);
let mut c = DMatrix::<f64>::zeros(8, 5);

// sigmoid, ignoring position
let sigmoid = |v: f64, _r: usize, _c: usize| 1.0 / (1.0 + (-v).exp());
gemm_map(1.0, &a, &b, 0.0, &mut c, &sigmoid, Parallelism::Serial);
```

Prefer `gemm_fused` for a plain bias or ReLU, since it vectorizes; `gemm_map` is the general extension point (GELU, sigmoid, clamps, position-dependent transforms) and pays one indirect call per output element. Like the fused entry, its `f32`/`f64` result is bit-identical to `gemm` followed by the same map, and `gemm_map_with` reuses a `Workspace`.

## Prepacked operands

When one operand is fixed across many multiplies, for example a weight matrix served against a stream of activations, prepacking it once removes the per-call repack. `prepack_rhs(b) -> PackedRhs<T>` packs a right operand; `gemm_packed_b` then consumes the handle in place of `B`. `prepack_lhs`/`gemm_packed_a` do the mirror for a fixed left operand.

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{gemm_packed_b, prepack_rhs};
use nalgebra::DMatrix;

let weights = DMatrix::<f32>::from_fn(64, 32, |i, j| 0.01 * (i as f32 - j as f32));
let packed = prepack_rhs(&weights); // pack the fixed B once

for step in 0..100 {
    let x = DMatrix::<f32>::from_element(16, 64, step as f32); // an activation batch
    let mut y = DMatrix::<f32>::zeros(16, 32);                 // column-major output
    gemm_packed_b(1.0, &x, &packed, 0.0, &mut y, Parallelism::default());
}
```

There is an orientation constraint. `gemm_packed_b` needs a column-major-ish `C` (`|col stride| >= |row stride|`); a row-major `C` would force the engine to swap `A` and `B` internally, which invalidates a prepacked RHS, so gemmkit rejects it. `gemm_packed_a` is the opposite: it needs a row-major-ish `C`, and rejects a column-major one. For a `C` in the wrong orientation, fall back to plain `gemm`. Each packed entry has a `_with` twin for workspace reuse, and under `epilogue` the fused twins `gemm_packed_b_fused` and `gemm_packed_a_fused` add a bias and activation off the same handle. `PackedRhs` and `PackedLhs` expose `.rows()` and `.cols()` if you need to re-check dimensions. See [Prepacked Operands](../gemmkit-guide/Prepacked_Operands.md) for the underlying reuse model.

## Batched GEMM

nalgebra has no rank-3 array type, so batched GEMM does not take a 3-D tensor the way the ndarray adapter does. Instead `gemm_batched` takes the batch as a slice of per-element `(&A, &B)` input pairs alongside a slice of `&mut C` outputs, matched positionally, and runs `C_e <- alpha*A_e*B_e + beta*C_e` for every element in one call over gemmkit's pointer-array engine. The `alpha`, `beta`, and `par` arguments are shared by the whole batch.

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::gemm_batched;
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);
let mut c = vec![DMatrix::<f32>::zeros(2, 2), DMatrix::<f32>::zeros(2, 2)];

let ab = [(&a, &b), (&a, &b)];
gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
assert_eq!(c[0], DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
```

Element shapes may differ (a heterogeneous batch) as long as each element's own dimensions agree, `A_e.cols == B_e.rows` and so on; the shared storage type carries the varying runtime dimensions, so `DMatrix` or dynamic-stride views cover heterogeneous shapes and mixed layouts under one type. The input count and output count must match (`ab.len() == c.len()`); a mismatch, or any element's dimension mismatch, panics. Parallelism is spread across the batch, whole GEMMs assigned to workers and each run serially and cache-hot, which is why the result reproduces a plain loop of `gemm` calls, stays deterministic across thread counts, and is bit-identical between serial and parallel runs. Because the C slice is a single storage type, there is no pointer-array analogue of a single shared fused epilogue, so unlike the ndarray adapter there is no `gemm_batched_fused` here. The batching model is discussed further in [Batched GEMM](../gemmkit-guide/Batched_GEMM.md).

## Where this sits next to nalgebra's own multiply

nalgebra already multiplies matrices: `&a * &b`, `a.mul_to(&b, &mut c)`, and the rest of its operator surface return properly typed matrices and integrate with its const-generic dimensions. For an ordinary `f32`/`f64` product, especially of small static matrices, those are the idiomatic choice and there is no reason to reach for this adapter. The adapter earns its place when you want something nalgebra's operators do not offer: the engine's runtime SIMD dispatch, which selects the best available instruction set on the machine at run time rather than at compile time; the `i8 -> i32` and requantizing integer paths; fused bias/activation and per-element epilogues in a single pass; batched multiplication of many small problems; and prepacked operands for a fixed weight reused across calls. When your matrices already live in nalgebra and you need any of those, the adapter gives you the engine's throughput and features without leaving nalgebra's types and without a copy.
