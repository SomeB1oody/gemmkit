# Fused Epilogues

A GEMM rarely stands alone. Its output usually feeds straight into a bias add, an activation, or a quantization step before anything else looks at it. Done naively that is a second full pass over `C`: the GEMM writes `m*n` values, then a separate loop reads them all back, transforms them, and writes them again. A fused epilogue folds that transform into the GEMM's own store, so each output element is transformed in-register at the moment it is written and the extra pass over memory simply disappears. Everything on this page lives behind the `epilogue` Cargo feature.

## Bias and activation

[`gemm_fused`](https://docs.rs/gemmkit) is the vectorized workhorse: `C <- act(alpha*A*B + beta*C + bias)` in one pass. The bias is a [`Bias`](https://docs.rs/gemmkit) enum, either `Bias::PerRow(&[T])` (one value per output row, length `m`) or `Bias::PerCol(&[T])` (one per output column, length `n`), added to every element of that row or column after the product. The activation is an [`Activation`](https://docs.rs/gemmkit): `Relu` (`max(v, 0)`) or `LeakyRelu(slope)`. Both arguments are `Option`, and `None`/`None` delegates straight to plain `gemm`.

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

The bias, `LeakyRelu` slope, and activation apply in-register on the vector fast path, so the fusion costs almost nothing over the raw GEMM.

## An arbitrary per-element map

When the transform is not a bias or a standard activation, [`gemm_map`](https://docs.rs/gemmkit) takes a closure `f(value, row, col) -> value` and applies it to each output element at its final value, exactly once, fused into the store. It is the general extension point for epilogues gemmkit ships no fast path for, GELU, sigmoid, clamps, or anything position-dependent:

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

The `(row, col)` handed to the closure are in the user frame of `C`, and the closure may capture its environment by reference (the bound is `+ Sync`, so it is shared safely across the parallel workers, e.g. borrowing a lookup table). `gemm_map` is `f32`/`f64` only. It trades one indirect call per output element (cheap against the `O(k)` work per element) for total generality; for a plain bias or activation prefer `gemm_fused`, which vectorizes the transform.

## Integer requantization

Quantized inference wants the opposite of a widening GEMM: `i8` inputs, an `i32` accumulator, and an `i8` (or `u8`) output again, with a scale and zero-point applied on the way down. [`gemm_i8_requant`](https://docs.rs/gemmkit) and [`gemm_i8_requant_u8`](https://docs.rs/gemmkit) do the whole thing in one pass, deleting the full `m*n` `i32` materialization that a `gemm_i8` followed by a separate requantize step would pay. They take a [`Requantize`](https://docs.rs/gemmkit) struct:

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

The output is `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO, HI)` with round-half-to-even, where `scale` is either a single `RequantScale::PerTensor(f32)` or a per-row `RequantScale::PerRow(&[f32])` (the per-channel convention), and the clamp band is set by the entry: `[-128, 127]` for `gemm_i8_requant`, `[0, 255]` for the `u8` twin. There is no `alpha` (it folds into `scale`) and no `beta` (accumulating into a quantized `C` is ill-defined). The requantize map is bit-exact across every ISA (scalar, FMA, AVX-512F, VNNI) and across the vector and scalar store paths, so the answer never depends on which kernel ran.

## Complex bias

Under the `complex` feature, `gemm_cplx_fused` adds a per-row or per-col bias to a complex product `C <- alpha*op(A)*op(B) + beta*C + bias`, with the same optional operand conjugation as `gemm_cplx`. It is bias-only by design: an ordering-based activation like ReLU is undefined on complex numbers. The `conj_a` / `conj_b` flags conjugate the operands only; the bias is added verbatim, never conjugated.

## What you can rely on

Every fused entry routes each shape through the **same** kernel plain `gemm` would pick, the general driver or one of the [special paths](Small_Shapes_and_GEMV.md), and fuses the epilogue into that kernel's store without perturbing its accumulation order. So a fused call is not a different algorithm; it is the same GEMM with the map applied at store time. The concrete guarantees:

- For `f32`/`f64`, a fused result is **bitwise identical** to plain `gemm` followed by the same scalar map, for every shape, every layout, and every worker count. `gemm_map` gives the same guarantee against a per-element `f`, and the complex bias entry against a complex `gemm_cplx`-then-bias.
- For the narrow floats `f16`/`bf16` (feature `half`) there is one documented exception. The bias and slope are widened exactly to `f32`, the epilogue applies in `f32`, and the single round-to-nearest-even narrowing to the output happens once, on store. That is *more* precise than `gemm`-then-map (which would round to the narrow type, widen, and round again), so for narrow types the fused result is deliberately **not** bitwise-equal to the two-step form. Reproducibility and determinism are unchanged.
- Reproducibility holds throughout: serial and parallel agree bit-for-bit, and the identity-fused case (`None`/`None`, or an absent bias) const-folds back to exactly plain `gemm`.

The payoff is the pass over `C` you no longer make. On a memory-bound epilogue that second pass can cost as much as the store itself, so fusing a bias or activation into the GEMM is close to free where the two-step form is not.

The fused epilogues also compose with the other API tiers: `gemm_batched_fused` applies one shared bias and activation to every element of a [batched GEMM](Batched_GEMM.md), and `gemm_packed_b_fused` / `gemm_packed_a_fused` fuse over a [prepacked operand](Prepacked_Operands.md). Every checked entry has raw-pointer `_unchecked` twins for adapters and FFI, which carry the bias as a `(ptr, BiasDim)` pair instead of the `Bias` enum; see [The Unchecked Tier](The_Unchecked_Tier.md).
