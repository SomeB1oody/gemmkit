[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-nalgebra/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-nalgebra/README.md)

# gemmkit-nalgebra

[![crates.io](https://img.shields.io/crates/v/gemmkit-nalgebra.svg)](https://crates.io/crates/gemmkit-nalgebra) [![docs.rs](https://img.shields.io/docsrs/gemmkit-nalgebra)](https://docs.rs/gemmkit-nalgebra)

Zero-copy [nalgebra](https://crates.io/crates/nalgebra) 0.35 adapter over the
[gemmkit](https://crates.io/crates/gemmkit) GEMM engine. Input operands are taken as
`&Matrix<T, R, C, S>` for any storage `S: RawStorage<T, R, C>`, so `DMatrix`, static
`SMatrix`, and every view type are accepted; the output is a `&mut Matrix` whose storage
is `RawStorageMut`. The adapter reads the pointer and strides straight out of the matrix
and forwards them to gemmkit's raw engine, so column-major (nalgebra's natural layout),
row-major, and general-stride views all work without copying.

The exposed surface mirrors the core engine. Real-scalar `gemm`, `gemm_with`, and `dot`
are generic over `gemmkit::GemmScalar` (`f32`/`f64`, plus `f16`/`bf16` under the `half`
feature). `prepack_lhs`/`prepack_rhs` build a reused pack handle for the
`gemm_packed_a`/`gemm_packed_b` fixed-operand loop. Feature-gated families add integer
(`gemm_i8`/`dot_i8`, `i8 -> i32`) and complex (`gemm_cplx`/`dot_cplx`) entries, and the
`epilogue` feature adds fused entries: `gemm_fused` (bias plus activation), the
per-element `gemm_map`, requantizing `gemm_i8_requant`/`gemm_i8_requant_u8`
(`int8` + `epilogue`), and the bias-only `gemm_cplx_fused` (`complex` + `epilogue`).
nalgebra has no rank-3 array type, so batched GEMM (`gemm_batched`) takes the batch as a
slice of per-element `(&A, &B)` inputs paired with a slice of `&mut C` outputs (over
gemmkit's pointer-array batched engine), with heterogeneous per-element shapes.

A step-by-step guide for this adapter lives in the
[gemmkit Guide](https://someb1oody.github.io/gemmkit/en/gemmkit-nalgebra/Using_gemmkit_with_nalgebra.html).

## Usage

```toml
[dependencies]
gemmkit-nalgebra = "0.1"
gemmkit = "0.1" # for the Parallelism argument, which is not re-exported
nalgebra = "0.35"
```

```rust
use nalgebra::DMatrix;

fn main() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);
    let c = gemmkit_nalgebra::dot(&a, &b);
    assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
}
```

`dot(a, b)` returns `A*B` in a fresh column-major `DMatrix`. For the accumulating form
`C <- alpha*A*B + beta*C`, call `gemm(alpha, &a, &b, beta, &mut c, par)` with a
`gemmkit::Parallelism` value.

## Feature flags

Each flag forwards to the same-named feature on `gemmkit`.

| Feature | Default | Effect |
| --- | --- | --- |
| `parallel` | Yes | rayon-based parallelism (`gemmkit/parallel`). |
| `wasm_threads` | No | Enables `parallel` and `gemmkit/wasm_threads` for `wasm32-wasip1-threads`. |
| `half` | No | `f16`/`bf16` inputs with `f32` accumulation. |
| `complex` | No | `Complex<f32>`/`Complex<f64>` entries with optional conjugation. |
| `int8` | No | `i8 -> i32` integer entries. |
| `epilogue` | No | Fused bias/activation, per-element map, and (with `int8`) `i8`/`u8` requantization. |

## Supported element types

The real `f32` and `f64` paths are always built; the rest are gated behind the
features above. Each type is read straight out of the nalgebra matrix, so
column-major (nalgebra's natural layout), row-major, and general-stride views all
work without conversion.

| Element type | Feature | Computes | Entry points |
| --- | --- | --- | --- |
| `f32`, `f64` | built-in | `C <- alpha*A*B + beta*C` | `gemm`, `dot`, `gemm_fused`, `gemm_map` |
| `f16`, `bf16` | `half` | same, output type in, `f32` accumulate | `gemm`, `dot`, `gemm_fused` |
| `i8` | `int8` | `i8 * i8 -> i32` | `gemm_i8`, `dot_i8` |
| `i8` (requantized) | `int8` + `epilogue` | `i8 * i8 ->` `i8` or `u8` | `gemm_i8_requant`, `gemm_i8_requant_u8` |
| `Complex<f32>`, `Complex<f64>` | `complex` | same, optional `conj(A)` / `conj(B)` | `gemm_cplx`, `dot_cplx`, `gemm_cplx_fused` |

Each entry also has a `_with` variant that reuses a caller-owned `Workspace`, and
the prepacked (`gemm_packed_a` / `gemm_packed_b`) and batched (`gemm_batched`, over
a slice of per-element inputs) paths carry the same element types.

## Related crates

- [gemmkit](https://crates.io/crates/gemmkit): the core engine. All algorithmic
  documentation lives there and on its [docs.rs](https://docs.rs/gemmkit) page, including
  the ISA backends and the `GEMMKIT_REQUIRE_ISA` pin.
- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) and
  [gemmkit-faer](https://crates.io/crates/gemmkit-faer): sibling zero-copy adapters for
  the ndarray and faer matrix types.
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune): install-time autotuner binary.

## License

Licensed under either of [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT)
or [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE), at your
option.

## Minimum supported Rust version

Rust 1.89.
