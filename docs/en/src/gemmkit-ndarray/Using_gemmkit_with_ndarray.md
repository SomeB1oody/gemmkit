# Using gemmkit with ndarray

`gemmkit-ndarray` is a thin bridge between `ndarray`'s two-dimensional arrays and the
gemmkit engine. It does no numerical work of its own. Each entry takes an `ArrayBase`.
It reads the base pointer and the 2 axis strides straight out of it, and hands those
raw parts to gemmkit's unchecked engine. The whole crate is a stride-plumbing layer.
Everything gemmkit knows how to do, such as runtime ISA selection, cache blocking, and
reproducible parallelism, applies unchanged. The arrays never get reshaped or copied on
the way in.

The entries accept `&ArrayBase<S, Ix2>` for any storage `S: Data`. Both an owned
`&Array2<T>` and a borrowed `ArrayView2<T>` work, along with `ArcArray`, `CowArray`, and
slices of any of them. The only internal helper is worth seeing. It is the whole of the
adapter's data extraction:

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

That `(rows, cols, row_stride, col_stride)` tuple, plus `a.as_ptr()`, is everything
gemmkit needs. The strides are signed `isize`, so a negative (reversed) stride forwards
just like a positive one.

## Adding it to a project

2 crates, not 3. The adapter re-exports everything its own signatures name, so a direct
`gemmkit` dependency is not part of the normal setup:

```toml
[dependencies]
gemmkit-ndarray = "0.1"
ndarray = "0.17.1"
```

`gemmkit_ndarray` re-exports everything a caller needs, so a direct `gemmkit` dependency
is rarely necessary:

- The `Parallelism` selector and the `Workspace` type every `_with` variant takes.
- The fused selectors `Bias` and `Activation`.
- The prepacked handles `PackedLhs` and `PackedRhs`.
- The requantization parameters `Requantize` and `RequantScale`.
- The element-type bounds `GemmScalar`, `FusedScalar`, `MapScalar`, and `ComplexScalar`.
  Name these when you write a wrapper generic over an entry.
- The element types `f16`, `bf16`, `Complex`, `c32`, and `c64`, each under its own
  feature. This keeps `half` and `num-complex` out of your manifest too.
- The `tuning` module.

Reach for `tuning` through the adapter. Do not add a separate `gemmkit` dependency of
your own for it. The tuning knobs are process-global atomics. A second, separately
resolved `gemmkit` would give you a set of atomics the adapter never reads.

Every feature on `gemmkit-ndarray` forwards directly to the same-named feature on
`gemmkit`. Turn on a capability here, and the matching entry points light up:

- `parallel` (default): rayon multithreading.
- `wasm_threads`: threading on `wasm32-wasip1-threads`. Implies `parallel`.
- `half`: `f16` / `bf16` inputs with `f32` accumulation.
- `complex`: `Complex<f32>` / `Complex<f64>` matrices.
- `int8`: `i8` inputs accumulating into `i32`.
- `epilogue`: fused bias / activation, `i8` / `u8` requantization, and a user per-element
  map.

The default is `["parallel"]`. The [advanced page](ndarray_Adapter_Advanced_Usage.md)
covers the feature-gated families, such as `gemm_cplx`, `gemm_i8`, and `gemm_fused`.
The minimum `ndarray` version is `0.17.1`.

## The core entries

3 functions cover the plain real path. `dot` is the convenience entry. It multiplies
`A * B` into a freshly allocated row-major `Array2`, the way `ndarray`'s own `.dot()`
reads.

```rust
use ndarray::array;

let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
let c = gemmkit_ndarray::dot(&a, &b);
assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
```

`dot` is generic over `T: GemmScalar`, which is `f32` and `f64` unconditionally, plus
`f16` and `bf16` when the `half` feature is on. It parallelizes with
`Parallelism::default()` and allocates its own output. Use it for a one-off product
where you do not already own the destination.

`gemm` writes the general form `C <- alpha*A*B + beta*C` in place. This is where
`alpha`, `beta`, an existing accumulator, and an explicit `Parallelism` come in. Its
signature is:

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

The output binds `SC: DataMut`, so `C` is a `&mut Array2` or an `ArrayViewMut2` and, like
the inputs, may carry any layout. Here `A` is a row-major buffer transposed into a
column-major view with no copy. The multiply runs single-threaded:

```rust
use gemmkit_ndarray::Parallelism;
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

Because the adapter only reads strides, any two-dimensional view that `ndarray` can
express forwards without a copy. That includes:

- the standard C-order (row-major) layout
- an F-order (column-major) view from `.reversed_axes()`, `.t()`, or an array built with
  `.f()`
- a windowed `.slice(...)` view with non-unit strides
- a reversed view from a negative-step slice such as `s![..;-1, ..]`, which produces a
  negative row stride

The destination `C` is just as free. `Array2::zeros((m, n).f())` gives a column-major
output, and `gemm` fills it directly.

"Zero-copy" here means the *adapter* never copies to normalize a layout. gemmkit's
engine still packs operands into its own scratch buffers when the microkernel needs
contiguous panels. That internal packing is part of the algorithm, not a materialization
of a transposed input. The point is that you never pay a `to_owned()` or a manual
transpose to satisfy the call, whatever your arrays look like.

## Panics: shapes, not aliasing

The adapter validates shapes and nothing else. Each entry asserts that the inner
dimensions line up and that `C` matches the product. On a mismatch it panics with a
`gemmkit-ndarray:` message that names the offending dimensions, for instance
`A.cols (k) != B.rows (kb)`, `A.rows (m) != C.rows (cm)`, or `B.cols (n) != C.cols (cn)`.
A dimension mismatch is the only reason a plain `gemm` or `dot` panics.

The adapter does not check aliasing at runtime, and it does not need to. `C` arrives as
`&mut ArrayBase<SC, _>`, an exclusive borrow. The type system already guarantees it
cannot overlap the shared `&` borrows of `A` and `B`. That is exactly the precondition
gemmkit's `_unchecked` engine asks its caller to uphold, and the `&mut` signature upholds
it for free. The fused entries add 1 more runtime check, for a bias slice overlapping
`C`. See the [advanced page](ndarray_Adapter_Advanced_Usage.md) for that check.

## Choosing parallelism

`Parallelism` is re-exported by the adapter. `Parallelism::Serial` runs on the calling
thread. `Parallelism::Rayon(n)` uses a rayon pool of at most `n` threads, and `Rayon(0)`
auto-detects the machine's core count. `Parallelism::default()` is `Rayon(0)`, which is
what `dot` uses, so `dot` is parallel out of the box. The threaded paths need the
`parallel` feature, which is on by default. With that feature off, treat every call as
serial.

Blocking and job order do not depend on the thread count, so a fixed input and
configuration give a reproducible result. Serial and parallel runs agree bit-for-bit
today, because both walk the same blocking and the same kernel. That agreement is a
property of the current implementation, not a promise across every configuration. The
reasoning behind the thread counts lives in
[Parallelism in Practice](../gemmkit-guide/Parallelism_in_Practice.md).

## Reusing a workspace

Every allocating entry borrows scratch from gemmkit's internal thread-local pool for the
duration of the call. A lone `gemm` call never leaks an allocation into your steady
state. When you run a hot loop of similar products, the `_with` variants let you own
that scratch instead. Pass a `&mut Workspace` as the first argument. It grows once to
the largest size the loop needs, then gets reused with no further allocation.

```rust
use gemmkit_ndarray::{Parallelism, Workspace};
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

`gemm_with` is identical to `gemm` apart from the leading workspace, and it produces the
same result. Every family in the adapter has a matching `_with` twin. The pattern
carries over to the integer, complex, fused, batched, and prepacked entries covered
next. If you multiply a fixed weight matrix against a stream of activations, the
workspace pairs naturally with the prepacked-operand path on the
[advanced page](ndarray_Adapter_Advanced_Usage.md).
