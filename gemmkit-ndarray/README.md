# gemmkit-ndarray

A thin [`ndarray`] 0.17 adapter over the [`gemmkit`](/gemmkit/README.md) GEMM engine.

It accepts `&ArrayBase<S, Ix2>` for any storage `S: Data` — so both `ArrayView2`
and `&Array2` work — reads the pointer and strides straight out of the array, and
forwards to gemmkit's raw engine. C-order, F-order, general-stride, and reversed
(negative-stride) views therefore all work **without copying**.

```rust
use ndarray::array;

// .dot()-style convenience
let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
let c = gemmkit_ndarray::dot(&a, &b);
assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
```

```rust
use gemmkit::Parallelism;
use ndarray::Array2;
# let a = Array2::<f64>::zeros((8, 6));
# let b = Array2::<f64>::zeros((6, 5));
let mut c = Array2::<f64>::zeros((8, 5));
// C ← 1.5·A·B + 2.0·C, accepting views or owned arrays.
gemmkit_ndarray::gemm(1.5, &a.view(), &b, 2.0, &mut c, Parallelism::Rayon(0));
```

## API

- `gemm(alpha, a, b, beta, c, par)` — `C ← α·A·B + β·C`.
- `gemm_with(ws, alpha, a, b, beta, c, par)` — same, reusing a `gemmkit::Workspace`.
- `dot(a, b) -> Array2<T>` — `A·B` into a fresh array.

`T` is `f32` or `f64` (`gemmkit::GemmScalar`). `Parallelism` is re-exported from
`gemmkit`.

## Features

- `parallel` (default) — forwards to `gemmkit/parallel` (rayon).
- `half`, `complex`, `int8` — forward to the matching `gemmkit` feature.
- `epilogue` — fused epilogues: `gemm_fused` / `gemm_batched_fused` (bias/activation in one pass);
  requant `gemm_i8_requant` / `gemm_i8_requant_u8` needs `int8` + `epilogue`, complex-fused
  `gemm_cplx_fused` needs `complex` + `epilogue`, and `f16`/`bf16` fused ride `half`.

## License

MIT OR Apache-2.0.

[`ndarray`]: https://docs.rs/ndarray
