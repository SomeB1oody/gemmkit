# gemmkit-nalgebra

A thin [`nalgebra`] 0.35 adapter over the [`gemmkit`](/gemmkit/README.md) GEMM engine.

It accepts `&Matrix<T, R, C, S>` for any storage `S: RawStorage` — so `DMatrix`,
static `SMatrix`, and every view type work — reads the pointer and strides straight
out of the matrix, and forwards to gemmkit's raw engine. Column-major (nalgebra's
natural layout), row-major, and general-stride views therefore all work **without
copying**.

```rust
use nalgebra::{DMatrix, Matrix2};

// .dot()-style convenience
let a = Matrix2::new(1.0_f32, 2.0, 3.0, 4.0);
let b = Matrix2::new(5.0_f32, 6.0, 7.0, 8.0);
let c = gemmkit_nalgebra::dot(&a, &b);
assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
```

```rust
use gemmkit::Parallelism;
use nalgebra::DMatrix;
# let a = DMatrix::<f64>::zeros(8, 6);
# let b = DMatrix::<f64>::zeros(6, 5);
let mut c = DMatrix::<f64>::zeros(8, 5);
// C ← 1.5·A·B + 2.0·C, accepting views or owned matrices.
gemmkit_nalgebra::gemm(1.5, &a, &b, 2.0, &mut c, Parallelism::Rayon(0));
```

## API

- `gemm(alpha, a, b, beta, c, par)` — `C ← α·A·B + β·C`.
- `gemm_with(ws, alpha, a, b, beta, c, par)` — same, reusing a `gemmkit::Workspace`.
- `dot(a, b) -> DMatrix<T>` — `A·B` into a fresh matrix.
- `prepack_rhs`/`prepack_lhs` + `gemm_packed_b`/`gemm_packed_a` — pre-pack one reused
  operand for the fixed-weight loop.
- `gemm_i8`/`dot_i8` (`int8` feature) and `gemm_cplx`/`dot_cplx` (`complex` feature).

`T` is `f32` or `f64` (`gemmkit::GemmScalar`), plus `f16`/`bf16` under `half`.
`Parallelism` is re-exported from `gemmkit`. nalgebra has no 3-D array type, so the
ndarray adapter's batched entries have no analogue here.

## Features

- `parallel` (default) — forwards to `gemmkit/parallel` (rayon).
- `wasm_threads`, `half`, `complex`, `int8` — forward to the matching `gemmkit` feature.

## License

MIT OR Apache-2.0.

[`nalgebra`]: https://docs.rs/nalgebra
