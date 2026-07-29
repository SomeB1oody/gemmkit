# ndarray Adapter Advanced Usage

Beyond the plain real product, the adapter mirrors the whole of gemmkit's surface:

- integer GEMM
- requantized quantized-inference output
- complex products with optional conjugation
- fused bias and activation
- a user-supplied per-element map
- batched multiplication over rank-3 arrays
- prepacked operands

The adapter gates each family behind the Cargo feature named for it. Each family also
keeps the shape of the plain entries from the
[getting-started page](Using_gemmkit_with_ndarray.md). Each reads strides straight from
the arrays, forwards to gemmkit, and panics only on a dimension mismatch. The fused
entries add 1 more panic case, for a bias slice that overlaps `C`. Every entry also has
a `_with` twin that threads a caller-owned `Workspace`.

## Integer GEMM (`int8`)

`gemm_i8` multiplies `i8` inputs into an `i32` accumulator:
`C(i32) <- alpha*A(i8)*B(i8) + beta*C`, with `alpha`, `beta`, and `C` all `i32`.
It is a separate entry from `gemm`
because the input and output element types differ. Arithmetic wraps on overflow. This is
the conventional integer-GEMM contract. `dot_i8` is the convenience twin. It returns a
fresh `Array2<i32>`.

```rust
use gemmkit_ndarray::{Parallelism, dot_i8, gemm_i8};
use ndarray::Array2;

let a = Array2::<i8>::zeros((16, 12));
let b = Array2::<i8>::zeros((12, 10));

// i8 inputs, i32 accumulator
let c: Array2<i32> = dot_i8(&a, &b);

// general form with i32 alpha/beta into an existing accumulator
let mut acc = Array2::<i32>::zeros((16, 10));
gemm_i8(2, &a, &b, 1, &mut acc, Parallelism::Serial);
```

## Requantized output (`int8` + `epilogue`)

Quantized inference rarely wants the raw `i32` accumulator. It wants an 8-bit tensor
back. `gemm_i8_requant` fuses the multiply and the requantize into 1 pass. It folds the
`i32` accumulator to an `i8` output without ever materializing the full `m*n`
intermediate. There is no `alpha` parameter, because it folds into the scale, and no
`beta` parameter, because accumulating into a quantized output is ill-defined. The
parameters live in a `Requantize`:

```rust
use gemmkit_ndarray::{Parallelism, RequantScale, Requantize, gemm_i8_requant, gemm_i8_requant_u8};
use ndarray::Array2;

let a = Array2::<i8>::zeros((16, 12));
let b = Array2::<i8>::zeros((12, 10));

// i8 output in [-128, 127], one per-tensor scale, per-row bias (length A.rows)
let bias: Vec<i32> = vec![0; 16];
let mut c = Array2::<i8>::zeros((16, 10));
let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,
    bias: Some(&bias),
};
gemm_i8_requant(&a, &b, req, &mut c, Parallelism::default());

// u8 output in [0, 255], per-channel scales, no bias
let scales: Vec<f32> = vec![0.02; 16]; // one per output row / channel
let mut cu = Array2::<u8>::zeros((16, 10));
gemm_i8_requant_u8(
    &a,
    &b,
    Requantize { scale: RequantScale::PerRow(&scales), zero_point: 128, bias: None },
    &mut cu,
    Parallelism::default(),
);
```

The output is `clamp(zero_point + round_ne(scale * (accumulator + bias[i])), LO, HI)`
with round-half-to-even, where `scale` is the per-tensor value or the per-row `scale_i`.
The `u8` variant is the ONNX-QLinearMatMul-style activation: identical to
`gemm_i8_requant` apart from the output domain `[0, 255]` and the `zero_point` band.
Both entries reject:

- a non-finite or non-positive scale, per-tensor or per-row
- a per-row scale or bias whose length is not `A.rows`
- a slice that overlaps `C`
- a `zero_point` outside the entry's domain

## Complex GEMM (`complex`)

Complex products get their own entries, because the 2 conjugation flags do not fit the
homogeneous real signature. `gemm_cplx` computes `C <- alpha*op(A)*op(B) + beta*C`, over
`Complex<f32>` or `Complex<f64>`. `op(A)` is `conj(A)` when `conj_a` is set. `op(B)` is
`conj(B)` when `conj_b` is set. `dot_cplx` is the non-conjugated convenience.

```rust
use gemmkit_ndarray::{Complex, Parallelism, dot_cplx, gemm_cplx};
use ndarray::Array2;

type C = Complex<f64>;
let a = Array2::<C>::from_elem((8, 6), Complex::new(0.0, 0.0));
let b = Array2::<C>::from_elem((6, 5), Complex::new(0.0, 0.0));

// plain A*B
let c = dot_cplx(&a, &b);

// conjugate A, accumulate into an existing C
let mut acc = Array2::<C>::from_elem((8, 5), Complex::new(0.0, 0.0));
gemm_cplx(
    Complex::new(1.0, 0.0),
    &a,
    true,  // conj_a
    &b,
    false, // conj_b
    Complex::new(0.0, 0.0),
    &mut acc,
    Parallelism::Serial,
);
```

With `complex` and `epilogue` both on, `gemm_cplx_fused` adds an optional `Bias` (added
verbatim, never conjugated) in the same pass. It takes no activation parameter. An
ordering activation like ReLU is undefined on complex numbers.

## Fused bias, activation, and maps (`epilogue`)

`gemm_fused` computes `C <- act(alpha*A*B + beta*C + bias)` in 1 pass. The bias is an
optional `Bias::PerRow` (length `A.rows`) or `Bias::PerCol` (length `B.cols`). The
activation is an optional `Relu` or `LeakyRelu(slope)`, applied last. With both set to
`None`, `gemm_fused` is exactly `gemm`.

```rust
use gemmkit_ndarray::{Activation, Bias, Parallelism, gemm_fused, gemm_map};
use ndarray::Array2;

let a = Array2::<f32>::zeros((12, 9));
let b = Array2::<f32>::zeros((9, 7));

// C <- ReLU(A*B + bias) in one pass; PerRow bias has length A.rows
let bias: Vec<f32> = vec![0.0; 12];
let mut c = Array2::<f32>::zeros((12, 7));
gemm_fused(
    1.0, &a, &b, 0.0, &mut c,
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::default(),
);

// arbitrary per-element closure f(value, row, col); here a relu6
let f = |v: f32, _r: usize, _c: usize| v.max(0.0).min(6.0);
let mut c2 = Array2::<f32>::zeros((12, 7));
gemm_map(1.0, &a, &b, 0.0, &mut c2, &f, Parallelism::default());
```

`Bias` and `Activation` are re-exported from `gemmkit_ndarray`, so you need not name
`gemmkit` for them. For `f32`/`f64`, `gemm_fused` is bit-identical to `gemm` followed by
the same scalar map, for every shape. For `f16`/`bf16`, the epilogue runs in `f32` before
the single narrowing. This is more precise than a separate narrow map, so the result is
not bitwise-equal to `gemm`-then-map for those types.

`gemm_map` is the general per-element extension point. The closure `f(value, row, col)`
sees each output element at its final value, with `(row, col)` in the user frame of `C`.
gemmkit calls it exactly once per element. It costs 1 indirect call per element. Prefer
`gemm_fused` for a plain bias or activation, because it vectorizes. Reach for `gemm_map`
instead for GELU, sigmoid, clamps, or position-dependent transforms. Here, `T` is
`f32`/`f64` only.

## Batched GEMM

This is the only operation with no plain-`gemm` analogue, and no counterpart in the
sibling adapters. It is a stack of independent products on a rank-3 `Array3`, with the
batch on axis 0. `a` is `(batch, m, k)`, `b` is `(batch, k, n)`, and `c` is
`(batch, m, n)`. Axis 0 is each operand's batch stride. Axes 1 and 2 are the element
strides. `gemm_batched` parallelizes across the batch. Each element runs on 1 worker, so
the result reproduces a loop of `gemm` calls exactly.

```rust
use gemmkit_ndarray::{Parallelism, dot_batched, gemm_batched};
use ndarray::Array3;

let a = Array3::<f32>::zeros((32, 8, 5)); // (batch, m, k)
let b = Array3::<f32>::zeros((32, 5, 6)); // (batch, k, n)

// stack of products
let c = dot_batched(&a, &b); // (32, 8, 6)

// general form into an existing accumulator
let mut acc = Array3::<f32>::zeros((32, 8, 6));
gemm_batched(0.7, &a, &b, 1.3, &mut acc, Parallelism::default());
```

The adapter reads only strides, so a permuted-axes or otherwise general-stride 3-D view
forwards without a copy. For example, `a.view().permuted_axes([0, 2, 1])` turns a
`(batch, k, m)` buffer into a `(batch, m, k)` view that batches straight through.

Under `epilogue`, `gemm_batched_fused` applies 1 shared `Bias`/`Activation` to every
element of the stack. This is the batched-linear-layer case. The bias is sized for a
single element (`PerRow` length `m`, `PerCol` length `n`), not the whole batch.

## Prepacked operands

When one operand is fixed and the other streams, pack the fixed side once and reuse it.
This skips the per-call repack. `prepack_rhs` returns a `PackedRhs<T>` for a reused `B`,
consumed by `gemm_packed_b`. `prepack_lhs` returns a `PackedLhs<T>` for a reused `A`,
consumed by `gemm_packed_a`. The prepack functions read strides directly, so `B` or `A`
may have any layout.

Each has 1 orientation constraint. `gemm_packed_b` needs a column-major-ish `C`
(`|col stride| >= |row stride|`). `gemm_packed_a` needs a row-major-ish `C`
(`|col stride| <= |row stride|`). The other orientation would swap the operands and
invalidate the packed handle, which gemmkit rejects. Use plain `gemm` for the layout
that does not fit.

The fused twins `gemm_packed_b_fused` and `gemm_packed_a_fused` accept the same handles
plus a bias and activation. This is exactly the fixed-weight inference layer. The
example below packs a weight matrix once as the LHS, and reuses it across inference
steps. It folds in a per-output-channel bias and a ReLU:

```rust
use gemmkit_ndarray::{Activation, Bias, Parallelism, gemm_packed_a_fused, prepack_lhs};
use ndarray::Array2;

let (out, in_features) = (256usize, 512usize);

// pack the fixed weight W: (out, in) once
let w = Array2::<f32>::zeros((out, in_features));
let packed = prepack_lhs(&w);
let bias: Vec<f32> = vec![0.0; out]; // per-output-channel, length C.rows

// each inference step: activations x (in, batch) -> y (out, batch)
let batch = 32;
let x = Array2::<f32>::zeros((in_features, batch));
let mut y = Array2::<f32>::zeros((out, batch)); // row-major (packed_a orientation)
gemm_packed_a_fused(
    1.0,
    &packed,
    &x,
    0.0,
    &mut y,
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::default(),
);
```

Pair the packed handle with the `_with` workspace variant, `gemm_packed_a_fused_with`. A
steady inference loop then allocates nothing after the first call. You specify the bias
axis in the user frame. The packed path forwards it unflipped for `gemm_packed_b_fused`,
and lets the core flip it for `gemm_packed_a_fused`. `PerRow` always means "1 value per
output row," regardless of which operand you packed.

## This adapter versus ndarray's own product

`ndarray` already multiplies matrices: `.dot()` for the plain product and
`general_mat_mul` for the in-place `alpha`/`beta` form. For a one-off `f32`/`f64`
product with no extra requirements, those functions work well and pull in 1 less
dependency. There is no reason to route through gemmkit out of habit.

Reach for this adapter when you want what `ndarray`'s built-in path does not offer.
gemmkit picks the fastest instruction set on the machine it runs on, at runtime. It does
not bake 1 choice in at compile time (see
[Runtime ISA Dispatch](../gemmkit-guide/Runtime_ISA_Dispatch.md)). It brings the wider
surface this page covers: fused bias and activation, `i8` and requantized inference,
complex with conjugation, batched products, and prepacking. It also exposes tuning
knobs, plus an install-time autotuner that calibrates blocking to the deployment machine
(see [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md)).

None of that changes the arrays you pass or the results you get back. The adapter is
the same zero-copy stride plumbing throughout. It only widens what you can ask for.
