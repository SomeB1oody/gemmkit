# Using gemmkit with faer

`gemmkit-faer` is a thin, zero-copy bridge from faer's view types to the gemmkit GEMM engine. It accepts a `MatRef<'_, T>` for each input and a `MatMut<'_, T>` for the output, reads the data pointer and the element-unit row and column strides straight out of the view, and hands them to gemmkit's raw engine. Nothing is transposed, copied, or repacked on the way in. Because faer already stores its strides the way gemmkit's engine wants them, a faer `Mat`, a transposed view, an offset sub-matrix, and a reversed (negative-stride) view all reach the kernel untouched.

The crate targets faer 0.24 and requires Rust 1.89.

## Installation and features

`gemmkit-faer` does not re-export `Parallelism` or `Workspace`, so depend on `gemmkit` too for those argument types.

```toml
[dependencies]
gemmkit-faer = "0.1"
gemmkit = "0.1" # for the Parallelism and Workspace argument types
faer = "0.24"
```

Every Cargo feature forwards to the same-named `gemmkit` feature, so you enable an element family or the fused entries here and the core turns them on underneath.

- `parallel` (default): rayon-based parallelism.
- `wasm_threads`: threading on `wasm32-wasip1-threads` (also enables `parallel`).
- `half`: the `f16` and `bf16` element types, accumulated in `f32`.
- `complex`: the `c32` and `c64` element types.
- `int8`: `i8` inputs into an `i32` output.
- `epilogue`: the fused bias/activation, requantization, and per-element map entries.

The feature-gated families and the fused entries are covered on the [advanced usage page](faer_Adapter_Advanced_Usage.md); this page stays on the always-available `f32`/`f64` (plus `f16`/`bf16` under `half`) surface.

## What zero-copy means here

Every entry routes through one small helper that pulls the raw parts out of a `MatRef`. faer reports strides in element units as `isize`, negative for a reversed view, which is exactly the shape gemmkit's unchecked engine takes, so there is no conversion step at all.

```rust
// gemmkit-faer/src/common.rs
pub(crate) fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (a.nrows(), a.ncols(), a.row_stride(), a.col_stride(), a.as_ptr())
}
```

The adapter validates the three shared dimensions itself, then forwards the pointers and strides to gemmkit's `_unchecked` engine inside a single `unsafe` block. The safety argument is short: faer's view types guarantee the pointer plus strides describe a valid in-bounds layout, and the output is a `MatMut` (an exclusive borrow), so `C` cannot alias `A` or `B`. That is the entire adapter for the plain path. All of gemmkit's cache blocking, ISA dispatch, packing, and parallel scheduling live in the core and are documented there; see the [architecture chapter](../architecture/The_Layer_Stack.md) if you want the internals.

## gemm and dot

The two workhorses are `dot`, which returns a fresh product, and `gemm`, which updates an output in place. Both are generic over `gemmkit::GemmScalar`: `f32` and `f64` always, plus `f16` and `bf16` when the `half` feature is on.

```rust
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
// A*B into a fresh column-major Mat
let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
assert_eq!(c[(0, 0)], 19.0);
assert_eq!(c[(1, 1)], 50.0);
```

`dot(a, b)` computes `A*B` into a newly allocated column-major `Mat` and runs with the default parallelism (`Parallelism::Rayon(0)`, auto-detected threads). It is the one-shot convenience; when you own the output buffer or want the general update, use `gemm`.

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::gemm;

let a = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
let b = Mat::<f64>::from_fn(3, 5, |i, j| (i as f64) * (j as f64));
let mut c = Mat::<f64>::zeros(4, 5);
// c <- 1.5 * a * b + 2.0 * c, single-threaded
gemm(1.5, a.as_dyn_stride(), b.as_dyn_stride(), 2.0, c.as_dyn_stride_mut(), Parallelism::Serial);
```

`gemm(alpha, a, b, beta, c, par)` computes `C <- alpha*A*B + beta*C` in place. With `beta == 0` the prior contents of `C` are overwritten and never read (this is exactly what `dot` does internally); with a nonzero `beta` the call accumulates onto what `C` already holds. The signatures are the ones you see above: inputs are `MatRef<'_, T>`, the output is `MatMut<'_, T>`, and `par` is a `gemmkit::Parallelism`. The `.as_dyn_stride()` / `.as_dyn_stride_mut()` conversions turn faer's statically-typed strides into the dynamic-stride views the adapter accepts; they cost nothing at runtime.

## Layouts that pass through untouched

Because the adapter only ever reads a pointer and two strides, any faer view works without a copy or a fallback. A transposed operand is the common "row-major A" case: transposing a column-major matrix yields a view whose row stride is non-unit, and it goes straight to the kernel.

```rust
// `at` is k x m column-major; `.transpose()` gives an m x k view with a non-unit
// row stride - read straight through, no copy
let a = at.as_dyn_stride().transpose();
let c = gemmkit_faer::dot(a, b.as_dyn_stride());
```

The same holds for an offset sub-matrix (`submatrix(...)`, which moves the base pointer and keeps a non-contiguous column stride) and for a reversed view (`reverse_rows()` / `reverse_cols()`, which carries a negative stride). gemmkit's unchecked path handles negative strides directly, so a reversed input accumulates correctly under `beta` just like any other. See [Matrix Views and Layouts](../gemmkit-guide/Matrix_Views_and_Layouts.md) for how the engine treats general strides.

## Choosing parallelism

Every entry takes a `gemmkit::Parallelism`. `Parallelism::Serial` runs single-threaded; `Parallelism::Rayon(n)` uses rayon with at most `n` threads, and `Rayon(0)` auto-detects. gemmkit ramps thread count with the workload rather than jumping to all cores, and the result is reproducible across thread counts for a fixed configuration, so switching between `Serial` and `Rayon` does not change the answer you get. The [Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md) guide covers the scheduling model.

## Reusing a workspace across calls

`gemm` allocates its scratch space from a thread-local pool. If you drive many GEMMs in a loop and want to own that scratch buffer explicitly, every entry has a `_with` twin that takes a `&mut gemmkit::Workspace` as its first argument and reuses it across calls.

```rust
use gemmkit::{Parallelism, Workspace};
use gemmkit_faer::gemm_with;

let mut ws = Workspace::new();
for (a, b, mut c) in problems {
    // same result as `gemm`, but the scratch buffer is reused
    gemm_with(&mut ws, 1.0, a, b, 0.0, c.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

A single `Workspace` grows to fit the largest problem it has seen and is then reused, which matters most for a stream of similar small-to-medium GEMMs where allocation would otherwise show up in the profile.

## Panic behavior

The adapter checks the three shared dimensions before dispatching and panics on a mismatch: `A.cols` must equal `B.rows`, `A.rows` must equal `C.rows`, and `B.cols` must equal `C.cols`. The messages are prefixed `gemmkit-faer:` and name the two conflicting extents, for example `gemmkit-faer: A.cols (4) != B.rows (5)`. These are the only panics on the plain `gemm`/`dot` path. The feature-gated entries add a few more (bias length and overlap, requantize parameters, prepacked-`C` orientation); those reproduce gemmkit's own checked-entry wording and are listed with each entry on the [advanced usage page](faer_Adapter_Advanced_Usage.md).
