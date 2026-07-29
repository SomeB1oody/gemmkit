# Fused Epilogues

A GEMM rarely stands alone. Its output usually feeds straight into a bias add, an
activation, or a quantization step. Done naively, that means a second full pass over `C`.
The GEMM writes `m*n` values, then a separate loop reads them all back, transforms them,
and writes them again.

A fused epilogue folds that transform into the GEMM's own store. It transforms each
output element in-register, at the moment the store writes it, so the extra pass over
memory disappears. Everything on this page lives behind the `epilogue` Cargo feature.

## Bias and activation

[`gemm_fused`](https://docs.rs/gemmkit) is the vectorized workhorse. It computes
`C <- act(alpha*A*B + beta*C + bias)` in one pass. The bias is a
[`Bias`](https://docs.rs/gemmkit) enum: either `Bias::PerRow(&[T])` (one value per output
row, length `m`) or `Bias::PerCol(&[T])` (one per output column, length `n`). gemmkit adds
that value to every element of the matching row or column, after the product. The
activation is an [`Activation`](https://docs.rs/gemmkit): `Relu` (`max(v, 0)`) or
`LeakyRelu(slope)`. Both arguments are `Option`. Passing `None` for both delegates
straight to plain `gemm`.

```rust
use gemmkit::{gemm_fused, Bias, Activation, MatRef, MatMut, Parallelism};

let bias = vec![0.0f32; m]; // one value per output row
gemm_fused(
    1.0,
    MatRef::from_row_major(&a, m, k),
    MatRef::from_col_major(&b, k, n),
    0.0,
    MatMut::from_col_major(&mut c, m, n),
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::Rayon(0),
);
```

The bias, the `LeakyRelu` slope, and the activation all apply in-register on the vector
fast path. So the fusion costs almost nothing over the raw GEMM.

## An arbitrary per-element map

When the transform is not a bias or a standard activation, use
[`gemm_map`](https://docs.rs/gemmkit). It takes a closure `f(value, row, col) -> value`.
gemmkit applies that closure to each output element, exactly once, at its final value,
fused into the store. It is the general extension point for an epilogue with no dedicated
fast path in gemmkit: GELU, sigmoid, a clamp, or a position-dependent transform.

```rust
use gemmkit::{gemm_map, MatRef, MatMut, Parallelism};

let f = |v: f32, _r: usize, _c: usize| v.tanh();
gemm_map(
    1.0,
    MatRef::from_row_major(&a, m, k),
    MatRef::from_col_major(&b, k, n),
    0.0,
    MatMut::from_col_major(&mut c, m, n),
    &f,
    Parallelism::Rayon(0),
);
```

The `(row, col)` handed to the closure are in the user frame of `C`. The closure may
capture its environment by reference. The bound is `+ Sync`, so gemmkit can share that
reference safely across the parallel workers, for example to borrow a lookup table.
`gemm_map` works for `f32`/`f64` only. It trades one indirect call per output element,
cheap next to the `O(k)` work per element, for total generality. For a plain bias or
activation, prefer `gemm_fused`, which vectorizes the transform.

## Integer requantization

Quantized inference wants the opposite of a widening GEMM. It takes `i8` inputs and
accumulates into `i32`. It produces an `i8` (or `u8`) output again, applying a scale and a
zero-point on the way down. [`gemm_i8_requant`](https://docs.rs/gemmkit) and
[`gemm_i8_requant_u8`](https://docs.rs/gemmkit) do the whole thing in one pass. That
deletes the full `m*n` `i32` materialization a separate `gemm_i8` call, followed by a
requantize step, would need. Both entries take a [`Requantize`](https://docs.rs/gemmkit)
struct:

```rust
use gemmkit::{gemm_i8_requant_u8, Requantize, RequantScale, MatRef, MatMut, Parallelism};

let req = Requantize {
    scale: RequantScale::PerRow(&per_channel_scales), // length m, per-channel
    zero_point: 128,
    bias: Some(&i32_bias),                             // optional per-row i32 bias, length m
};
gemm_i8_requant_u8(
    MatRef::from_row_major(&activations, m, k),
    MatRef::from_col_major(&weights, k, n),
    req,
    MatMut::from_col_major(&mut out_u8, m, n),
    Parallelism::Rayon(0),
);
```

The output is `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO,
HI)`, using round-half-to-even. `scale` is either a single `RequantScale::PerTensor(f32)`
or a per-row `RequantScale::PerRow(&[f32])`, the per-channel convention. The entry sets
the clamp band: `[-128, 127]` for `gemm_i8_requant`, `[0, 255]` for the `u8` twin. There
is no `alpha`, since it folds into `scale`. There is no `beta`, since accumulating into an
already-quantized `C` is not well defined. The requantize map is bit-exact across every
ISA (scalar, FMA, AVX-512F, VNNI) and across the vector and scalar store paths. So the
answer never depends on which kernel ran.

## Complex bias

Under the `complex` feature, `gemm_cplx_fused` adds a per-row or per-col bias to a
complex product: `C <- alpha*op(A)*op(B) + beta*C + bias`. It takes the same optional
operand conjugation as `gemm_cplx`. It is bias-only by design. An ordering-based
activation like ReLU has no definition on complex numbers. The `conj_a` and `conj_b`
flags conjugate the operands only. gemmkit adds the bias verbatim and never conjugates
it.

## What you can rely on

Every fused entry routes each shape through the same kernel plain `gemm` would pick: the
general driver, or one of the [special paths](Small_Shapes_and_GEMV.md). It fuses the
epilogue into that kernel's store without changing its accumulation order. So a fused
call is not a different algorithm. It runs the same GEMM and applies the map at store
time. The concrete guarantees:

- For `f32`/`f64`, a fused result is **bitwise identical** to plain `gemm` followed by the
  same scalar map. This holds for every shape, every layout, and every worker count.
  `gemm_map` gives the same guarantee for a per-element `f`. The complex bias entry gives
  the same guarantee against a complex `gemm_cplx` call followed by the bias add.
- For the narrow floats `f16`/`bf16` (feature `half`), there is one documented exception.
  gemmkit widens the bias and the slope exactly to `f32`, applies the epilogue in `f32`,
  and narrows to the output once, on store, with round-to-nearest-even. That is *more*
  precise than `gemm` followed by a separate map, which would round to the narrow type,
  widen, and round again. So for narrow types, the fused result is deliberately **not**
  bitwise-equal to that two-step form. Reproducibility and determinism are unchanged.
- Serial and parallel runs agree bit-for-bit today. The identity-fused case (`None`/`None`,
  or an absent bias) const-folds back to exactly plain `gemm`. The reproducibility contract
  covers only a fixed configuration, and the worker count is part of that configuration.

The payoff is the pass over `C` you no longer make. On a memory-bound epilogue, that
second pass can cost as much as the store itself. So fusing a bias or activation into the
GEMM is close to free, in a case where the two-step form is not.

The fused epilogues also compose with the other API tiers. `gemm_batched_fused` applies
one shared bias and activation to every element of a [batched GEMM](Batched_GEMM.md).
`gemm_packed_b_fused` and `gemm_packed_a_fused` fuse over a
[prepacked operand](Prepacked_Operands.md). Every checked entry has raw-pointer
`_unchecked` twins for adapters and FFI. Those twins carry the bias as a
`(ptr, BiasDim)` pair instead of the `Bias` enum. See
[The Unchecked Tier](The_Unchecked_Tier.md).
