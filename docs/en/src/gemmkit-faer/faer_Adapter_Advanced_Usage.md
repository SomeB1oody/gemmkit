# faer Adapter Advanced Usage

Beyond `gemm` and `dot`, the faer adapter mirrors the rest of gemmkit's surface. This
includes the extra element families, the fused epilogues, batched GEMM over a slice, and
prepacked operands. Each of these is feature-gated. Each also reads raw pointers and
strides straight out of faer's views. So transposed, sub-matrix, and reversed operands
keep working exactly as they do on the plain path. This page walks the families one at a
time. It closes with a note on when the adapter earns its place next to faer's own matmul.

The [introductory page](Using_gemmkit_with_faer.md) covers installation, the zero-copy
mechanism, `gemm`/`gemm_with`/`dot`, parallelism, and the workspace pattern. Everything
here builds on that page. As on the plain path, every entry also has a `_with` twin that
reuses a caller-owned `Workspace`.

## Integer GEMM (`int8`)

With the `int8` feature, `gemm_i8` and `dot_i8` take `i8` inputs and accumulate into an
`i32` output. The input and output element types differ. That is why this is a separate
entry from `gemm`, rather than another instance of the generic. faer's view types are
generic over the element. So an `i8` `MatRef` and an `i32` `MatMut` need no special
handling.

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, dot_i8, gemm_i8};

let a = Mat::<i8>::from_fn(16, 12, |i, j| ((i + j) as i8 % 7) - 3);
let b = Mat::<i8>::from_fn(12, 10, |i, j| ((i * 2 + j) as i8 % 5) - 2);
// i8 * i8 accumulated into a fresh Mat<i32>
let c = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());

// Mat::zeros is ComplexField-only, so integer outputs use from_fn
let mut acc = Mat::<i32>::from_fn(16, 10, |_, _| 0);
// c <- 3 * a * b + (-2) * c, all of alpha/beta/C in i32
gemm_i8(3, a.as_dyn_stride(), b.as_dyn_stride(), -2, acc.as_dyn_stride_mut(), Parallelism::Serial);
```

`alpha`, `beta`, and `C` are `i32`. The arithmetic wraps on overflow. This is the
conventional integer-GEMM semantics.

## Requantized output (`int8` + `epilogue`)

With both `int8` and `epilogue`, `gemm_i8_requant` fuses the requantize step into the
kernel's store. `i8` inputs multiply into an `i32` accumulator. The kernel scales, biases,
rounds, and clamps that accumulator down to an `i8` output in a single pass. It never
materializes the full `m*n` `i32` matrix. `gemm_i8_requant_u8` does the same, but clamps
to an unsigned `u8` output, the ONNX QLinearMatMul-style activation domain.

There is no `alpha`, because it folds into the scale. There is no `beta`, because
accumulating into a quantized output is ill-defined.

The parameters come in a `Requantize`. The crate re-exports this type, so you need not
depend on `gemmkit` for it. `scale` is a `RequantScale`, either `PerTensor(f32)` or a
`PerRow(&[f32])` of per-channel scales. `zero_point` joins in as an integer after
rounding. `bias` is an optional per-row `i32` vector, added to the accumulator before
scaling.

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, RequantScale, Requantize, gemm_i8_requant};

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

The output is `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO,
HI)`, with round-half-to-even. `[LO, HI]` is `[-128, 127]` for the `i8` entry and
`[0, 255]` for the `u8` entry.

The adapter validates the requantize parameters before dispatch. It reproduces gemmkit's
own checked-entry wording. The checks cover:

- A non-finite or non-positive scale.
- A per-row scale slice of the wrong length, or one that overlaps `C`.
- A `zero_point` outside the output domain.
- A bias of the wrong length, or one that overlaps `C`.

That validation is raw pointer math against `C`'s byte footprint. The adapter never
builds a `C` slice. This is what lets it forward negative-stride views to the raw engine
safely.

## Complex GEMM (`complex`)

With the `complex` feature, `gemm_cplx`, `gemm_cplx_with`, and `dot_cplx` operate on
complex matrices with optional per-operand conjugation. The element type `T` is
`Complex<f32>` or `Complex<f64>`.

This is not a separate representation from faer's own. faer 0.24's `c32` and `c64` are
type aliases for `num_complex::Complex<f32>` and `num_complex::Complex<f64>`. This crate
re-exports the same types as `Complex`, with the same `c32`/`c64` aliases, and constrains
its `ComplexScalar` bound over them. So a faer complex `Mat` reaches the adapter with no
conversion, just like a real one.

`gemm_cplx` is a separate entry from `gemm`, because the conjugation flags do not fit the
homogeneous surface. It computes `C <- alpha*op(A)*op(B) + beta*C`, where
`op(A) = conj(A)` when `conj_a` is set, and `op(B) = conj(B)` when `conj_b` is set.

The implementation in `cplx.rs` pulls the same raw parts as the real path. It threads the
2 `bool` flags through to `gemm_cplx_unchecked`. Nothing else differs, so transposed,
sub-matrix, and reversed views work identically. `dot_cplx` is the non-conjugated `A*B`
convenience.

```rust
use faer::Mat;
use gemmkit_faer::{Complex, Parallelism, gemm_cplx};

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

Under `complex` plus `epilogue`, there is `gemm_cplx_fused`. It adds an optional bias in
one pass: `C <- alpha*op(A)*op(B) + beta*C + bias`. The bias is a `Bias::PerRow` (length
`A.rows`) or a `Bias::PerCol` (length `B.cols`). gemmkit adds it verbatim, to every
element of that row or column, and never conjugates it.

There is deliberately no activation parameter. An ordering activation such as ReLU is
undefined on complex numbers, so the fused complex entry carries a bias only.

## Fused bias and activation (`epilogue`)

With `epilogue`, `gemm_fused` computes `C <- act(alpha*A*B + beta*C + bias)` in a single
pass. The optional `Bias` is `PerRow` or `PerCol`. The optional `Activation` is `Relu` or
`LeakyRelu(slope)`, applied last. Passing `None` for both gives exactly `gemm`. The crate
re-exports both selectors.

```rust
use gemmkit_faer::{Activation, Bias, Parallelism, gemm_fused};

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

For `f32`/`f64`, the fused result is bit-identical to plain `gemm` followed by the same
scalar map, for every shape. The epilogue folds into the same kernel's store without
perturbing the accumulation order. Serial and parallel runs also agree bit-for-bit today.
That agreement is a property of today's implementation, not a hard guarantee. The
reproducibility contract itself covers only a fixed configuration, and the worker count is
part of that configuration.

For `f16`/`bf16` (under `half`), the fused result is more precise instead of identical. A
separate `gemm()` call followed by a narrow map rounds to the narrow type, widens back,
and rounds again. The fused path skips that extra rounding. The bias and slope widen
exactly to `f32`, the epilogue applies in `f32`, and the result rounds once to the narrow
output. So for `f16`/`bf16` the fused result is more precise, though it is not
bit-identical to `gemm` followed by a narrow map. The `f32`/`f64` bitwise guarantee above
does not extend to these narrow types. Serial and parallel runs still agree bit-for-bit
for these types too, under the same fixed-configuration reproducibility contract. The
[Fused Epilogues](../gemmkit-guide/Fused_Epilogues.md) guide has the full contract.

For an arbitrary per-element function, there is `gemm_map` (`f32`/`f64` only):
`C[r,c] <- f(alpha*A*B + beta*C, r, c)`. The closure runs once per output element, at its
final value, with `(r, c)` in the user frame of `C`.

Use `gemm_map` for GELU, sigmoid, clamps, or position-dependent transforms. Prefer
`gemm_fused` instead for a plain bias or ReLU, because it vectorizes. `gemm_map` pays one
indirect call per element.

## Batched GEMM

faer has no rank-3 array type, so gemmkit-faer expresses batched GEMM over slices instead.
`gemm_batched`
takes a `&[(MatRef, MatRef)]` of per-element `(A, B)` inputs, paired positionally with a
`&mut [MatMut]` of `C` outputs. All elements share one `alpha`, `beta`, and `Parallelism`.

gemmkit's pointer-array engine parallelizes the batch across elements. Its scheduler
assigns whole GEMMs to workers. Each worker runs its GEMM serially and stays cache-hot for
it.

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, gemm_batched};

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

Element shapes may differ, a heterogeneous batch, as long as each element's own
dimensions agree. The call panics if the input and output counts disagree. It also
panics if any element's dimensions are inconsistent, naming the offending element index.

Each element re-dispatches through the full engine. So the batch reproduces a plain loop
of `gemm` calls. It is deterministic across thread counts, because each element runs
wholly on one worker. For the same reason, serial and batch-parallel output are
bit-identical.

There is no batched fused entry here. The ndarray adapter offers a shared-epilogue
batched form, but it has no pointer-array analogue in the core. See
[Batched GEMM](../gemmkit-guide/Batched_GEMM.md) for the scheduling policy.

## Prepacked operands

When one operand stays fixed across many calls, for example weights against a stream of
activations, pre-pack it once and skip the per-call repack. `prepack_rhs` turns a `B`
into a reusable `PackedRhs`, consumed by `gemm_packed_b`. `prepack_lhs` turns an `A` into
a `PackedLhs`, consumed by `gemm_packed_a`. The crate re-exports both handles.

```rust
use gemmkit_faer::{Parallelism, gemm_packed_b, prepack_rhs};

let packed = prepack_rhs(weights.as_dyn_stride()); // pack the fixed B once
for (act, mut out) in stream {
    // out must be column-major-ish (|col stride| >= |row stride|)
    gemm_packed_b(1.0, act.as_dyn_stride(), &packed, 0.0, out.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

The one constraint is output orientation. A prepacked `B` fixes the operand roles, so
`gemm_packed_b` needs a column-major-ish `C` (`|col stride| >= |row stride|`). A
row-major `C` would swap the `A`/`B` roles and invalidate the packed RHS, so gemmkit
rejects it. Symmetrically, `gemm_packed_a` needs a row-major-ish `C`. For a mismatched
output layout, fall back to plain `gemm`.

Under `epilogue`, the prepacked entries have fused twins: `gemm_packed_b_fused` and
`gemm_packed_a_fused`. Each takes the same `Bias`/`Activation` as `gemm_fused`, off the
same handle. The [Prepacked Operands](../gemmkit-guide/Prepacked_Operands.md) guide
explains the reuse model.

## When to reach for this adapter

faer ships its own matmul. For a plain `f32`/`f64` product of 2 faer matrices, use that
instead. This adapter earns its place when you need something the core faer operator
does not offer, on faer's own types, without leaving the faer ecosystem:

- **Extra element families**: `i8 -> i32` integer GEMM, and requantization that fuses
  straight down to an `i8` or `u8` output.
- **Fused epilogues**: the kernel computes bias and activation, or an arbitrary
  per-element closure, in the same pass as the product, not as a second sweep over `C`.
- **Prepacking across calls**: pack a fixed weight matrix once, then reuse it over a
  long inference loop.
- **A shared tuning surface**: all 3 gemmkit adapters sit on the same engine, so one
  `GEMMKIT_*` environment profile from
  [gemmkit-tune](../gemmkit-tune/Tuning_with_gemmkit-tune.md) applies to all of them. See
  [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md) for the knob surface.

If none of those apply, use faer's built-in matmul instead. It is the simpler choice.
This adapter supplements it. It does not replace it.
