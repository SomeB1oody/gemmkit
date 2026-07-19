# faer Adapter Advanced Usage

Beyond `gemm` and `dot`, the faer adapter mirrors the rest of gemmkit's surface: the extra element families, the fused epilogues, batched GEMM over a slice, and prepacked operands. Each is feature-gated and each reads raw pointers and strides straight out of faer's views, so transposed, sub-matrix, and reversed operands keep working exactly as they do on the plain path. This page walks the families one at a time and closes with an honest note on when the adapter earns its place next to faer's own matmul.

The [introductory page](Using_gemmkit_with_faer.md) covers installation, the zero-copy mechanism, `gemm`/`gemm_with`/`dot`, parallelism, and the workspace pattern; everything here builds on it. As on the plain path, every entry also has a `_with` twin that reuses a caller-owned `gemmkit::Workspace`.

## Integer GEMM (`int8`)

With the `int8` feature, `gemm_i8` and `dot_i8` take `i8` inputs and accumulate into an `i32` output. The input and output element types differ, which is why this is a separate entry from `gemm` rather than another instance of the generic; faer's view types are generic over the element, so an `i8` `MatRef` and an `i32` `MatMut` need no special handling.

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::{dot_i8, gemm_i8};

let a = Mat::<i8>::from_fn(16, 12, |i, j| ((i + j) as i8 % 7) - 3);
let b = Mat::<i8>::from_fn(12, 10, |i, j| ((i * 2 + j) as i8 % 5) - 2);
// i8 * i8 accumulated into a fresh Mat<i32>
let c = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());

// Mat::zeros is ComplexField-only, so integer outputs use from_fn
let mut acc = Mat::<i32>::from_fn(16, 10, |_, _| 0);
// c <- 3 * a * b + (-2) * c, all of alpha/beta/C in i32
gemm_i8(3, a.as_dyn_stride(), b.as_dyn_stride(), -2, acc.as_dyn_stride_mut(), Parallelism::Serial);
```

`alpha`, `beta`, and `C` are `i32`, and the arithmetic wraps on overflow, the conventional integer-GEMM semantics.

## Requantized output (`int8` + `epilogue`)

With both `int8` and `epilogue`, `gemm_i8_requant` fuses the requantize step into the kernel's store: `i8` inputs multiply into an `i32` accumulator, which is scaled, biased, rounded, and clamped down to an `i8` output in a single pass, without ever materializing the full `m*n` `i32` matrix. `gemm_i8_requant_u8` is the same but clamps to an unsigned `u8` output (the ONNX QLinearMatMul-style activation domain). There is no `alpha` (it folds into the scale) and no `beta` (accumulating into a quantized output is ill-defined).

The parameters come in a `Requantize`, re-exported from the crate so you need not depend on `gemmkit` for it. `scale` is a `RequantScale`, either `PerTensor(f32)` or a `PerRow(&[f32])` of per-channel scales; `zero_point` is joined in integer after rounding; and `bias` is an optional per-row `i32` vector added to the accumulator before scaling.

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_i8_requant, RequantScale, Requantize};

let (m, n) = (17, 13);
let bias: Vec<i32> = (0..m as i32).map(|i| 40 * i - 200).collect();
let mut c = Mat::<i8>::from_fn(m, n, |_, _| 0);
let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,
    bias: Some(&bias),
};
gemm_i8_requant(a.as_dyn_stride(), b.as_dyn_stride(), req, c.as_dyn_stride_mut(), Parallelism::Serial);
```

The output is `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO, HI)` with round-half-to-even, where `[LO, HI]` is `[-128, 127]` for the `i8` entry and `[0, 255]` for the `u8` entry. The adapter validates the requantize parameters before dispatch, reproducing gemmkit's own checked-entry wording (a non-finite or non-positive scale, a per-row scale slice of the wrong length or one overlapping `C`, a `zero_point` outside the output domain, or a bias of the wrong length or overlapping `C`). That validation is raw pointer math against `C`'s byte footprint; the adapter never fabricates a `C` slice, which is what lets it forward negative-stride views to the raw engine safely.

## Complex GEMM (`complex`)

With the `complex` feature, `gemm_cplx`, `gemm_cplx_with`, and `dot_cplx` operate on complex matrices with optional per-operand conjugation. The element type `T` is `Complex<f32>` or `Complex<f64>`. This is not a separate representation from faer's: faer 0.24's `c32` and `c64` are type aliases for `num_complex::Complex<f32>` and `num_complex::Complex<f64>`, the same types gemmkit re-exports as `gemmkit::Complex` and constrains its `ComplexScalar` bound over. So a faer complex `Mat` reaches the adapter with no conversion, just like a real one.

`gemm_cplx` is a separate entry from `gemm` because the conjugation flags do not fit the homogeneous surface. It computes `C <- alpha*op(A)*op(B) + beta*C` where `op(A) = conj(A)` when `conj_a` is set and `op(B) = conj(B)` when `conj_b` is set. The implementation in `cplx.rs` pulls the same raw parts as the real path and threads the two `bool` flags through to `gemm_cplx_unchecked`; nothing else differs, so transposed, sub-matrix, and reversed views work identically. `dot_cplx` is the non-conjugated `A*B` convenience.

```rust
use faer::Mat;
use gemmkit::{Complex, Parallelism};
use gemmkit_faer::gemm_cplx;

type C = Complex<f64>;
let a = Mat::<C>::from_fn(12, 9, |i, j| C::new(i as f64, j as f64));
let b = Mat::<C>::from_fn(9, 7, |i, j| C::new((i + j) as f64, 1.0));
let mut c = Mat::<C>::zeros(12, 7);
// C <- alpha * conj(A) * B + beta * C
gemm_cplx(
    C::new(1.3, -0.4),
    a.as_dyn_stride(), true,   // conjugate A
    b.as_dyn_stride(), false,  // leave B
    C::new(0.5, 0.7),
    c.as_dyn_stride_mut(),
    Parallelism::Serial,
);
```

Under `complex` + `epilogue` there is `gemm_cplx_fused`, which adds an optional bias in one pass: `C <- alpha*op(A)*op(B) + beta*C + bias`. The bias is a `Bias::PerRow` (length `A.rows`) or `Bias::PerCol` (length `B.cols`), added verbatim and never conjugated. There is deliberately no activation parameter: an ordering activation such as ReLU is undefined on complex numbers, so the fused complex entry carries a bias only.

## Fused bias and activation (`epilogue`)

With `epilogue`, `gemm_fused` computes `C <- act(alpha*A*B + beta*C + bias)` in a single pass. The optional `Bias` is `PerRow` or `PerCol`; the optional `Activation` is `Relu` or `LeakyRelu(slope)`, applied last. Passing `None` for both is exactly `gemm`. Both selectors are re-exported from the crate.

```rust
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_fused, Activation, Bias};

let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
// C <- relu(1.3 * A*B - 0.7 * C + rowbias)
gemm_fused(
    1.3, a.as_dyn_stride(), b.as_dyn_stride(), -0.7,
    c.as_dyn_stride_mut(),
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::Rayon(0),
);
```

For `f32`/`f64` the fused result is bit-identical to plain `gemm` followed by the same scalar map, for every shape and deterministic across thread counts, because the epilogue folds into the same kernel's store without perturbing the accumulation order. For `f16`/`bf16` (under `half`) it is *more* precise, not identical: the bias and slope widen exactly to `f32` and the epilogue applies in `f32` before the single round to the narrow output, so it avoids the double-rounding a separate narrow map would incur. The [Fused Epilogues](../gemmkit-guide/Fused_Epilogues.md) guide has the full contract.

For an arbitrary per-element function there is `gemm_map` (`f32`/`f64` only): `C[r,c] <- f(alpha*A*B + beta*C, r, c)`, with the closure applied once per output element at its final value and `(r, c)` in the user frame of `C`. Use it for GELU, sigmoid, clamps, or position-dependent transforms; prefer `gemm_fused` for a plain bias or ReLU because it vectorizes, whereas `gemm_map` pays one indirect call per element.

## Batched GEMM

faer has no rank-3 array type, so batched GEMM is expressed over slices: `gemm_batched` takes a `&[(MatRef, MatRef)]` of per-element `(A, B)` inputs paired positionally with a `&mut [MatMut]` of `C` outputs, sharing one `alpha`, `beta`, and `Parallelism`. The batch is parallelized *across elements*, whole GEMMs assigned to workers and each run serially and cache-hot, over gemmkit's pointer-array engine.

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::gemm_batched;

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
let mut c0 = Mat::<f64>::zeros(2, 2);
let mut c1 = Mat::<f64>::zeros(2, 2);
let ab = [
    (a.as_dyn_stride(), b.as_dyn_stride()),
    (a.as_dyn_stride(), b.as_dyn_stride()),
];
let mut c = [c0.as_dyn_stride_mut(), c1.as_dyn_stride_mut()];
gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
```

Element shapes may differ (a heterogeneous batch) as long as each element's own dimensions agree. The call panics if the input and output counts disagree, or if any element's dimensions are inconsistent, naming the offending element index. Each element re-dispatches through the full engine, so the batch reproduces a plain loop of `gemm` calls, is deterministic across thread counts, and is additionally bit-identical between serial and parallel because each element runs wholly on one worker. There is no batched fused entry here: the shared-epilogue batched form the ndarray adapter offers has no pointer-array analogue in the core. See [Batched GEMM](../gemmkit-guide/Batched_GEMM.md) for the scheduling policy.

## Prepacked operands

When one operand is fixed across many calls (weights against a stream of activations), pre-pack it once and skip the per-call repack. `prepack_rhs` turns a `B` into a reusable `PackedRhs`, consumed by `gemm_packed_b`; `prepack_lhs` turns an `A` into a `PackedLhs`, consumed by `gemm_packed_a`. Both handles are re-exported from the crate.

```rust
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_packed_b, prepack_rhs};

let packed = prepack_rhs(weights.as_dyn_stride()); // pack the fixed B once
for (act, mut out) in stream {
    // out must be column-major-ish (|col stride| >= |row stride|)
    gemm_packed_b(1.0, act.as_dyn_stride(), &packed, 0.0, out.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

The one constraint is output orientation. A prepacked `B` fixes the operand roles, so `gemm_packed_b` needs a column-major-ish `C` (`|col stride| >= |row stride|`); a row-major `C` would swap `A`/`B` and invalidate the packed RHS, which gemmkit rejects. Symmetrically `gemm_packed_a` needs a row-major-ish `C`. For a mismatched output layout, fall back to plain `gemm`. Under `epilogue` the prepacked entries have fused twins, `gemm_packed_b_fused` and `gemm_packed_a_fused`, taking the same `Bias`/`Activation` as `gemm_fused` off the same handle. The [Prepacked Operands](../gemmkit-guide/Prepacked_Operands.md) guide explains the reuse model.

## When to reach for this adapter

faer ships its own high-performance matmul, and for a plain `f32`/`f64` product of two faer matrices you should usually just use it. This adapter earns its place when you need something the core faer operator does not offer, on faer's own types and without leaving the ecosystem:

- **Extra element families**: `i8 -> i32` integer GEMM, and requantization that fuses straight down to an `i8` or `u8` output.
- **Fused epilogues**: bias and activation (or an arbitrary per-element closure) computed in the same pass as the product, rather than as a second sweep over `C`.
- **Prepacking across calls**: amortizing the pack of a fixed weight matrix over a long inference loop.
- **A shared tuning surface**: because all three gemmkit adapters sit on the same engine, one `GEMMKIT_*` environment profile from [gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md) applies uniformly. See [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md) for the knob surface.

If none of those apply, faer's built-in matmul is the simpler choice. The adapter is a supplement to it, not a replacement.
