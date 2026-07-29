# Using gemmkit with faer

`gemmkit-faer` is a thin, zero-copy bridge from faer's view types to the gemmkit GEMM
engine. It accepts a `MatRef<'_, T>` for each input and a `MatMut<'_, T>` for the output.
It reads the data pointer and the element-unit row and column strides straight out of the
view. Then it hands them to gemmkit's raw engine. The adapter does not transpose, copy, or
repack anything on the way in.

faer already stores its strides the way gemmkit's engine wants them. Because of this, a
faer `Mat`, a transposed view, an offset sub-matrix, and a reversed (negative-stride) view
all reach the kernel untouched.

The crate targets faer 0.24. It requires Rust 1.89.

## Installation and features

`gemmkit-faer` re-exports everything its own signatures name. Because of this, a direct
`gemmkit` dependency is not part of the normal setup.

```toml
[dependencies]
gemmkit-faer = "0.1"
faer = "0.24"
```

`gemmkit_faer` re-exports:

- The `Parallelism` selector, and the `Workspace` type every `_with` variant takes.
- The fused selectors `Bias` and `Activation`.
- The prepacked handles `PackedLhs` and `PackedRhs`.
- The requantization parameters `Requantize` and `RequantScale`.
- The element-type bounds `GemmScalar`, `FusedScalar`, `MapScalar`, and `ComplexScalar`.
  Name these when you write a wrapper generic over an entry.
- The element types `f16`, `bf16`, `Complex`, `c32`, and `c64`, each under its own feature.
  This keeps `half` and `num-complex` out of your manifest too.
- The `tuning` module.

Reach `tuning` through the adapter, not through a separate `gemmkit` dependency of your
own. The knobs are process-global atomics. A second, separately resolved `gemmkit` gives
you a different set of atomics, one the adapter never reads.

Every Cargo feature forwards to the same-named `gemmkit` feature. So when you enable an
element family or a fused entry here, the core turns it on too.

- `parallel` (default): rayon-based parallelism.
- `wasm_threads`: threading on `wasm32-wasip1-threads`. Also enables `parallel`.
- `half`: the `f16` and `bf16` element types, accumulated in `f32`.
- `complex`: the `c32` and `c64` element types.
- `int8`: `i8` inputs into an `i32` output.
- `epilogue`: the fused bias/activation, requantization, and per-element map entries.

The [advanced usage page](faer_Adapter_Advanced_Usage.md) covers the feature-gated
families and the fused entries. This page stays on the always-available `f32`/`f64` (plus
`f16`/`bf16` under `half`) surface.

## What zero-copy means here

Every entry routes through one small helper. This helper pulls the raw parts out of a
`MatRef`. faer reports strides in element units as `isize`, negative for a reversed view.
This is exactly the shape gemmkit's unchecked engine expects, so no conversion step exists
at all.

```rust
// gemmkit-faer/src/common.rs
pub(crate) fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (a.nrows(), a.ncols(), a.row_stride(), a.col_stride(), a.as_ptr())
}
```

The adapter validates the 3 shared dimensions itself. It then forwards the pointers
and strides to gemmkit's `_unchecked` engine inside a single `unsafe` block. The safety
argument is short. faer's view types guarantee that the pointer plus strides describe a
valid in-bounds layout. The output is a `MatMut`, an exclusive borrow, so `C` cannot alias
`A` or `B`. That is the entire adapter for the plain path.

All of gemmkit's cache blocking, ISA dispatch, packing, and parallel scheduling live in
the core. The core documents them there too. See the
[architecture chapter](../architecture/The_Layer_Stack.md) for the internals.

## gemm and dot

The 2 workhorses are `dot`, which returns a fresh product, and `gemm`, which updates an
output in place. Both are generic over `GemmScalar`: `f32` and `f64` always, plus `f16`
and `bf16` when the `half` feature is on.

```rust
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
// A*B into a fresh column-major Mat
let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
assert_eq!(c[(0, 0)], 19.0);
assert_eq!(c[(1, 1)], 50.0);
```

`dot(a, b)` computes `A*B` into a newly allocated column-major `Mat`. It runs with the
default parallelism, `Parallelism::Rayon(0)`, which auto-detects the thread count. Use
`dot` as the one-shot convenience. When you own the output buffer, or want the general
update, use `gemm` instead.

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, gemm};

let a = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
let b = Mat::<f64>::from_fn(3, 5, |i, j| (i as f64) * (j as f64));
let mut c = Mat::<f64>::zeros(4, 5);
// c <- 1.5 * a * b + 2.0 * c, single-threaded
gemm(1.5, a.as_dyn_stride(), b.as_dyn_stride(), 2.0, c.as_dyn_stride_mut(), Parallelism::Serial);
```

`gemm(alpha, a, b, beta, c, par)` computes `C <- alpha*A*B + beta*C` in place. With
`beta == 0`, `gemm` overwrites the prior contents of `C` and never reads them. This is
exactly what `dot` does internally. With a nonzero `beta`, the call accumulates onto what
`C` already holds.

The signatures are the ones you see above. The inputs are `MatRef<'_, T>`, the output is
`MatMut<'_, T>`, and `par` is a `Parallelism`. The `.as_dyn_stride()` and
`.as_dyn_stride_mut()` conversions turn faer's statically typed strides into the
dynamic-stride views the adapter accepts. They cost nothing at runtime.

## Layouts that pass through untouched

The adapter only ever reads a pointer and 2 strides. Because of this, any faer view works
without a copy or a fallback. A transposed operand is the common row-major-A case.
Transposing a column-major matrix yields a view whose row stride is non-unit, and that
view goes straight to the kernel.

```rust
// `at` is k x m column-major; `.transpose()` gives an m x k view with a non-unit
// row stride - read straight through, no copy
let a = at.as_dyn_stride().transpose();
let c = gemmkit_faer::dot(a, b.as_dyn_stride());
```

The same holds for an offset sub-matrix. `submatrix(...)` moves the base pointer and
keeps a non-contiguous column stride. It also holds for a reversed view. `reverse_rows()`
and `reverse_cols()` carry a negative stride.

gemmkit's unchecked path handles negative strides directly. So a reversed input
accumulates correctly under `beta`, just like any other input. See
[Matrix Views and Layouts](../gemmkit-guide/Matrix_Views_and_Layouts.md) for how the
engine treats general strides.

## Choosing parallelism

Every entry takes a `Parallelism`. `Parallelism::Serial` runs single-threaded.
`Parallelism::Rayon(n)` uses rayon with at most `n` threads. `Rayon(0)` auto-detects the
thread count.

gemmkit ramps the thread count with the workload, instead of jumping straight to every
core. For a fixed machine and a fixed configuration, a call gives reproducible results.
Serial and parallel runs also agree bit-for-bit today. That agreement is not a hard
guarantee, because the reproducibility contract covers a fixed configuration only, and the
worker count is part of that configuration. The
[Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md) guide covers the
scheduling model.

## Reusing a workspace across calls

`gemm` allocates its scratch space from a thread-local pool. Every entry also has a
`_with` twin. If you drive many GEMMs in a loop and want to own the scratch buffer
explicitly, use the `_with` twin instead. It takes a `&mut Workspace` as its first
argument and reuses that workspace across calls.

```rust
use gemmkit_faer::{Parallelism, Workspace, gemm_with};

let mut ws = Workspace::new();
for (a, b, mut c) in problems {
    // same result as `gemm`, but the scratch buffer is reused
    gemm_with(&mut ws, 1.0, a, b, 0.0, c.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

A single `Workspace` grows to fit the largest problem it has seen. After that, gemmkit
reuses it as-is. This matters most for a stream of similar small-to-medium GEMMs, where
allocation would otherwise show up in the profile.

## Panic behavior

The adapter checks the 3 shared dimensions before dispatching, and panics on a
mismatch. `A.cols` must equal `B.rows`. `A.rows` must equal `C.rows`. `B.cols` must equal
`C.cols`. The adapter prefixes each message with `gemmkit-faer:` and names the 2 conflicting
extents, for example `gemmkit-faer: A.cols (4) != B.rows (5)`. These are the only panics on the
plain `gemm`/`dot` path.

The feature-gated entries add a few more checks: bias length and overlap, requantize
parameters, and prepacked-`C` orientation. Those checks reproduce gemmkit's own
checked-entry wording. Each entry on the
[advanced usage page](faer_Adapter_Advanced_Usage.md) lists its own panics.
