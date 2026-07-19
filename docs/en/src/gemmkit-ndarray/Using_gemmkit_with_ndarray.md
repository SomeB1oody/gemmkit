# Using gemmkit with ndarray

`gemmkit-ndarray` is a thin bridge between `ndarray`'s two-dimensional arrays and the gemmkit engine. It does no numerical work of its own: each entry takes an `ArrayBase`, reads the base pointer and the two axis strides straight out of it, and hands those raw parts to gemmkit's unchecked engine. The whole crate is a stride-plumbing layer, so everything gemmkit knows how to do (runtime ISA selection, cache blocking, reproducible parallelism) applies unchanged, and the arrays never get reshaped or copied on the way in.

Because the entries accept `&ArrayBase<S, Ix2>` for any storage `S: Data`, both an owned `&Array2<T>` and a borrowed `ArrayView2<T>` work, along with `ArcArray`, `CowArray`, and slices of any of them. The one internal helper is worth seeing, since it is the whole of the adapter's data extraction:

`gemmkit-ndarray/src/common.rs`:

```rust
pub(crate) fn dims_strides<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix2>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}
```

That `(rows, cols, row_stride, col_stride)` tuple, plus `a.as_ptr()`, is everything gemmkit needs. The strides are signed `isize`, so a negative (reversed) stride forwards just like a positive one.

## Adding it to a project

The adapter re-exports the fused selectors (`Bias`, `Activation`) but not `Parallelism` or `Workspace`, so a typical project depends on both crates:

```toml
[dependencies]
gemmkit-ndarray = "0.1"
gemmkit = "0.1" # for Parallelism and Workspace
ndarray = "0.17.1"
```

Every feature on `gemmkit-ndarray` is a straight forward to the same-named feature on `gemmkit`, so you turn capabilities on here and they light up the matching entry points:

- `parallel` (default): rayon multithreading.
- `wasm_threads`: threading on `wasm32-wasip1-threads`; implies `parallel`.
- `half`: `f16` / `bf16` inputs with `f32` accumulation.
- `complex`: `Complex<f32>` / `Complex<f64>` matrices.
- `int8`: `i8` inputs accumulating into `i32`.
- `epilogue`: fused bias / activation, `i8` / `u8` requantization, and a user per-element map.

The default is `["parallel"]`; the feature-gated families (`gemm_cplx`, `gemm_i8`, `gemm_fused`, and the rest) are covered on the [advanced page](ndarray_Adapter_Advanced_Usage.md). The minimum `ndarray` is `0.17.1`.

## The core entries

Three functions cover the plain real path. `dot` is the convenience: it multiplies `A * B` into a freshly allocated row-major `Array2`, the way `ndarray`'s own `.dot()` reads.

```rust
use ndarray::array;

let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
let c = gemmkit_ndarray::dot(&a, &b);
assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
```

`dot` is generic over `T: GemmScalar`, which is `f32` and `f64` unconditionally, plus `f16` and `bf16` when the `half` feature is on. It parallelizes with `Parallelism::default()` and allocates its own output, so it is the right call for a one-off product where you do not already own the destination.

`gemm` writes the general form `C <- alpha*A*B + beta*C` in place, which is where `alpha`, `beta`, an existing accumulator, and an explicit `Parallelism` come in. Its signature is:

```rust
pub fn gemm<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
)
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>;
```

The output binds `SC: DataMut`, so `C` is a `&mut Array2` or an `ArrayViewMut2` and, like the inputs, may carry any layout. Here `A` is a row-major buffer transposed into a column-major view with no copy, and the multiply runs single-threaded:

```rust
use gemmkit::Parallelism;
use ndarray::{Array2, array};

// row-major storage, transposed into a column-major view with no copy
let a = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0])
    .unwrap()
    .reversed_axes();
let b = Array2::from_elem((2, 2), 1.0_f32);
let mut c = Array2::zeros((2, 2));
gemmkit_ndarray::gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Serial);
assert_eq!(c, array![[4.0, 4.0], [6.0, 6.0]]);
```

## Layouts that cost nothing

Because the adapter only ever reads strides, any two-dimensional view `ndarray` can express forwards without a copy. That includes the standard C-order (row-major) layout; an F-order (column-major) view from `.reversed_axes()`, `.t()`, or an array built with `.f()`; a windowed `.slice(...)` view with non-unit strides; and a reversed view from a negative-step slice such as `s![..;-1, ..]`, which produces a negative row stride. The destination `C` is just as free: `Array2::zeros((m, n).f())` gives a column-major output, and `gemm` fills it directly.

"Zero-copy" here means the *adapter* never copies to normalize a layout. gemmkit's engine still packs operands into its own scratch buffers when the microkernel needs contiguous panels; that internal packing is part of the algorithm, not a materialization of a transposed input. The point is that you never pay a `to_owned()` or a manual transpose to satisfy the call, whatever your arrays look like.

## Panics: shapes, not aliasing

The adapter validates shapes and nothing else. Each entry asserts that the inner dimensions line up and that `C` matches the product, panicking with a `gemmkit-ndarray:` message that names the offending dimensions, for instance `A.cols (k) != B.rows (kb)`, `A.rows (m) != C.rows (cm)`, or `B.cols (n) != C.cols (cn)`. A dimension mismatch is the only reason a plain `gemm` or `dot` panics.

Aliasing is not checked at runtime, and does not need to be: `C` arrives as `&mut ArrayBase<SC, _>`, an exclusive borrow, so the type system already guarantees it cannot overlap the shared `&` borrows of `A` and `B`. That is exactly the precondition gemmkit's `_unchecked` engine asks its caller to uphold, and the `&mut` signature upholds it for free. (The fused entries add one more runtime check, for a bias slice overlapping `C`; see the [advanced page](ndarray_Adapter_Advanced_Usage.md).)

## Choosing parallelism

`Parallelism` comes from `gemmkit`. `Parallelism::Serial` runs on the calling thread; `Parallelism::Rayon(n)` uses a rayon pool of at most `n` threads, and `Rayon(0)` auto-detects the machine's core count. `Parallelism::default()` is `Rayon(0)`, which is what `dot` uses, so `dot` is parallel out of the box. The threaded paths need the `parallel` feature (on by default); with it off, treat every call as serial. gemmkit keeps a serial and a parallel run bit-for-bit reproducible under a fixed input and configuration, so switching `par` never changes the numbers. The reasoning behind the thread counts lives in [Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md).

## Reusing a workspace

Every allocating entry borrows scratch from gemmkit's internal thread-local pool for the duration of the call, so a lone `gemm` never leaks an allocation into your steady state. When you run a hot loop of similar products, the `_with` variants let you own that scratch instead: pass a `&mut Workspace` as the first argument and it grows once to the largest size the loop needs, then gets reused with no further allocation.

```rust
use gemmkit::{Parallelism, Workspace};
use ndarray::Array2;

let mut ws = Workspace::new();
let par = Parallelism::default();
for &(m, k, n) in &[(256, 256, 256), (512, 128, 512)] {
    let a = Array2::<f32>::zeros((m, k));
    let b = Array2::<f32>::zeros((k, n));
    let mut c = Array2::<f32>::zeros((m, n));
    // reuses ws across iterations; allocates at most once
    gemmkit_ndarray::gemm_with(&mut ws, 1.0, &a, &b, 0.0, &mut c, par);
}
```

`gemm_with` is identical to `gemm` apart from the leading workspace and produces the same result. Every family in the adapter has a matching `_with` twin, so the pattern carries over to the integer, complex, fused, batched, and prepacked entries covered next. If you multiply a fixed weight matrix against a stream of activations, the workspace pairs naturally with the prepacked-operand path on the [advanced page](ndarray_Adapter_Advanced_Usage.md).
