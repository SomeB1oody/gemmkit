# gemmkit-faer

A thin [`faer`] 0.24 adapter over the [`gemmkit`](/gemmkit/README.md) GEMM engine.

It accepts faer's view types — `MatRef<'_, T>` for inputs, `MatMut<'_, T>` for the
output — reads the data pointer and the element-unit `isize` row/column strides straight
out of the view, and forwards to gemmkit's raw engine. faer's natural column-major
layout, transposed views, sub-matrices, and reversed (negative-stride) views therefore
all work **without copying**.

```rust
use faer::Mat;

// .dot()-style convenience
let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
assert_eq!(c[(0, 0)], 19.0);
assert_eq!(c[(1, 1)], 50.0);
```

```rust
use faer::Mat;
use gemmkit::Parallelism;
# let a = Mat::<f64>::zeros(8, 6);
# let b = Mat::<f64>::zeros(6, 5);
let mut c = Mat::<f64>::zeros(8, 5);
// C ← 1.5·A·B + 2.0·C, accepting views or owned matrices.
gemmkit_faer::gemm(1.5, a.as_dyn_stride(), b.as_dyn_stride(), 2.0, c.as_dyn_stride_mut(), Parallelism::Rayon(0));
```

## API

- `gemm(alpha, a, b, beta, c, par)` — `C ← α·A·B + β·C`.
- `gemm_with(ws, alpha, a, b, beta, c, par)` — same, reusing a `gemmkit::Workspace`.
- `dot(a, b) -> Mat<T>` — `A·B` into a fresh column-major matrix.
- `prepack_rhs`/`prepack_lhs` + `gemm_packed_b`/`gemm_packed_a` — pre-pack one reused
  operand for the fixed-weight loop.
- `gemm_i8`/`dot_i8` (`int8` feature) and `gemm_cplx`/`dot_cplx` (`complex` feature).
- `gemm_fused`/`gemm_fused_with` (`epilogue` feature) — `C ← act(α·A·B + β·C + bias)` in one
  pass, an optional `Bias` + `Activation`. With `int8` + `epilogue`, `gemm_i8_requant` /
  `gemm_i8_requant_u8` (requantized `i8`/`u8` output). With `complex` + `epilogue`, the bias-only
  `gemm_cplx_fused`. (Like the plain entries, these read raw parts from the view and forward to
  gemmkit's raw engine, so reversed/negative-stride views work without copying.)

`T` is `f32` or `f64` (`gemmkit::GemmScalar`), plus `f16`/`bf16` under `half`. Complex
uses faer's `c32`/`c64` (`= num_complex::Complex<f32>`/`<f64>`), which are exactly
gemmkit's complex element types. `Parallelism` is re-exported from `gemmkit`. faer has no
3-D array type, so the ndarray adapter's batched entries have no analogue here.

## Features

- `parallel` (default) — forwards to `gemmkit/parallel` (rayon).
- `wasm_threads`, `half`, `complex`, `int8` — forward to the matching `gemmkit` feature.
- `epilogue` — fused epilogues: `gemm_fused` (bias/activation); requant `gemm_i8_requant` needs
  `int8` + `epilogue`, complex-fused `gemm_cplx_fused` needs `complex` + `epilogue`, and `f16`/`bf16`
  fused ride `half`.

## License

MIT OR Apache-2.0.

[`faer`]: https://docs.rs/faer
