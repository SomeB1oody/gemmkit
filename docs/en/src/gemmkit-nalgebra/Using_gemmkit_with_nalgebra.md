# Using gemmkit with nalgebra

`gemmkit-nalgebra` lets you drive the gemmkit engine straight from nalgebra matrices without copying them first. It targets nalgebra 0.35 and accepts `&Matrix<T, R, C, S>` for any storage `S: RawStorage<T, R, C>`: owned `DMatrix`, static `SMatrix`, and every view or slice type all qualify. The adapter reads the matrix's data pointer and its two strides and hands them to gemmkit's raw engine, so nothing about the input is reshaped, transposed into a scratch buffer, or duplicated on the way in.

nalgebra's natural layout is column-major, which is also gemmkit's preferred orientation, so the common case is the fast case. Row-major and general-stride views work too, at the same zero copies, because the engine consumes strides directly rather than assuming a layout.

## Adding it to a project

Three crates go into `Cargo.toml`. The adapter re-exports the epilogue and pack types it needs, but `Parallelism` and `Workspace` come from `gemmkit` itself, so you depend on the core crate as well.

```toml
[dependencies]
gemmkit-nalgebra = "0.1"
gemmkit = "0.1" # for the Parallelism and Workspace arguments, which are not re-exported
nalgebra = "0.35"
```

The default feature set enables `parallel`, which turns on rayon-based threading in the engine (`gemmkit/parallel`). Every adapter feature is a thin forward to the same-named feature on `gemmkit`: `half` adds `f16`/`bf16` inputs, `complex` adds `Complex<f32>`/`Complex<f64>`, `int8` adds the `i8 -> i32` path, `epilogue` adds fused bias/activation and the per-element map, and `wasm_threads` layers `parallel` onto `wasm32-wasip1-threads`. The feature-gated entries are covered in [nalgebra Adapter Advanced Usage](nalgebra_Adapter_Advanced_Usage.md); this page stays on the always-available real-scalar surface.

## The three real-scalar entries

The base surface is three functions, all generic over `gemmkit::GemmScalar` (which is `f32` and `f64` always, plus `f16` and `bf16` under the `half` feature). `gemm` is the accumulating multiply, `gemm_with` is the same call reusing a caller-owned workspace, and `dot` is a convenience wrapper that allocates its result.

Here is `gemm` verbatim, from `gemmkit-nalgebra/src/float.rs`:

```rust
pub fn gemm<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}
```

The ten generic parameters look heavy, but they say one simple thing: `A`, `B`, and `C` may each be any nalgebra matrix or view, with independent row-dimension, column-dimension, and storage types. `A` and `B` are read through `RawStorage`; the output `C` needs `RawStorageMut` because it is written in place. The operation is `C <- alpha*A*B + beta*C`. `gemm_with` takes the same arguments preceded by a `&mut Workspace`, and `dot(a, b) -> DMatrix<T>` computes `A*B` into a freshly allocated column-major matrix (it calls `gemm` internally with `beta == 0`, so the fresh buffer is never read before it is overwritten).

## A first multiply with DMatrix

```rust
use gemmkit::Parallelism;
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);

// dot: A*B into a fresh column-major DMatrix
let c = gemmkit_nalgebra::dot(&a, &b);
assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));

// gemm: accumulate C <- alpha*A*B + beta*C in place
let mut acc = DMatrix::<f32>::zeros(2, 2);
gemmkit_nalgebra::gemm(1.0, &a, &b, 0.0, &mut acc, Parallelism::default());
assert_eq!(acc, c);
```

`dot` is the right tool when you just want the product and are happy with a `DMatrix` back. `gemm` is the tool when you already own the destination, want to scale it (`beta`) or scale the product (`alpha`), or want to avoid the allocation `dot` performs. Note that `dot` always returns a `DMatrix<T>`, even for static inputs, because the output dimensions are only known at the value level inside the wrapper.

## Static and mixed-shape matrices

Static matrices go through the same functions with no special casing. Because the row, column, and storage generics are independent per operand, a static `A` can multiply a dynamic `B`, and `gemm` can write into a static `&mut SMatrix` output just as readily as into a `DMatrix`.

```rust
use nalgebra::{DMatrix, Matrix2, SMatrix};

// static x static
let a = Matrix2::new(1.0_f32, 2.0, 3.0, 4.0);
let b = Matrix2::new(5.0_f32, 6.0, 7.0, 8.0);
let c = gemmkit_nalgebra::dot(&a, &b); // -> DMatrix<f32>
assert_eq!(c[(0, 0)], 19.0);

// static A x dynamic B: the independent Dim generics allow it
let a34 = SMatrix::<f64, 3, 4>::from_fn(|i, j| (i as f64) - 0.5 * (j as f64) + 1.0);
let b = DMatrix::<f64>::from_element(4, 2, 0.25);
let c = gemmkit_nalgebra::dot(&a34, &b); // -> DMatrix<f64>, shape 3x2
```

## Layouts and zero-copy

The adapter never copies an operand. It pulls `(rows, cols, row-stride, col-stride)` out of the matrix and forwards the pointer plus strides to the engine, which means the source layout only decides which stride pair the engine sees, never whether an allocation happens. A column-major `DMatrix`, a row-major slice built with `from_slice_with_strides`, and a non-contiguous stepped view (say every other row of a larger matrix) are all read in place. nalgebra reports strides as non-negative element counts, and the adapter widens them to the signed strides the engine expects.

Whatever internal packing the engine does to feed its microkernel is independent of the source layout and happens regardless of where the data came from; that is a property of gemmkit, not a copy the adapter introduces. If you want to eliminate the repeated internal repacking of one reused operand, that is what the prepacked-operand path is for, described in the [advanced page](nalgebra_Adapter_Advanced_Usage.md).

## When it panics

The entries validate shapes and panic on a mismatch, with the offending dimensions in the message. `gemm` (and its siblings) check three equalities before touching any memory: `A.cols == B.rows`, `A.rows == C.rows`, and `B.cols == C.cols`. A mismatch on the inner dimension, for instance, aborts with `gemmkit-nalgebra: A.cols (k) != B.rows (kb)` rather than reading out of bounds. The output matrix must therefore already have the right shape; `gemm` writes into it but does not resize it. `dot`, which allocates the output itself, can only fail on the inner-dimension check.

## Choosing parallelism

Every call takes a `gemmkit::Parallelism` as its last argument. There are two variants: `Parallelism::Serial` runs single-threaded, and `Parallelism::Rayon(n)` runs on rayon with at most `n` threads, where `Rayon(0)` auto-detects the thread count. The `Default` is `Rayon(0)`, which is what `dot` uses internally. For small matrices, or when you are already inside a parallel region and want to avoid nested threading, pass `Parallelism::Serial`; for large multiplies on an otherwise idle machine, `Parallelism::Rayon(0)` lets the engine spread the work. The threading strategy and how the engine picks a thread count are covered in [Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md).

## Reusing a workspace

The engine needs scratch space to pack blocks of `A` and `B`. By default it borrows that space from a thread-local pool, so `gemm` and `dot` allocate nothing of their own per call in the steady state. When you run many multiplies in a tight loop and want full control over that buffer, `gemm_with` takes a `&mut Workspace` you own and reuse:

```rust
use gemmkit::{Parallelism, Workspace};
use nalgebra::DMatrix;

let mut ws = Workspace::new();
let a = DMatrix::<f64>::from_element(64, 64, 1.0);
let b = DMatrix::<f64>::from_element(64, 64, 2.0);

for _ in 0..1000 {
    let mut c = DMatrix::<f64>::zeros(64, 64);
    gemmkit_nalgebra::gemm_with(&mut ws, 1.0, &a, &b, 0.0, &mut c, Parallelism::default());
}
```

A single `Workspace` can back multiplies of different shapes across iterations; it grows to fit the largest one it has seen and keeps that capacity. `Workspace::new()` starts empty and costs nothing until the first call fills it. The `_with` form exists on all the accumulating entries, including the feature-gated ones, so the same reuse pattern carries over to integer, complex, and fused calls.
