[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-faer/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-faer/README.md)

# gemmkit-faer

[![crates.io](https://img.shields.io/crates/v/gemmkit-faer.svg)](https://crates.io/crates/gemmkit-faer) [![docs.rs](https://img.shields.io/docsrs/gemmkit-faer)](https://docs.rs/gemmkit-faer)

Zero-copy [faer](https://crates.io/crates/faer) adapter for the [gemmkit](https://crates.io/crates/gemmkit) GEMM engine. It accepts faer's view types (`MatRef<'_, T>` for the inputs, `MatMut<'_, T>` for the output), reads the data pointer and the element-unit `isize` row and column strides straight out of each view, and forwards them to gemmkit's raw engine. faer's column-major layout, transposed views, sub-matrices, and reversed (negative-stride) views therefore all work without copying.

The entry points mirror the core gemmkit surface, including the feature-gated element families (`half`, `complex`, `int8`) and the fused epilogue entries (`epilogue`). faer has no rank-3 array type, so batched GEMM (`gemm_batched`) takes the batch as a slice of per-element `(A, B)` `MatRef` inputs paired with a slice of `&mut C` `MatMut` outputs (over gemmkit's pointer-array batched engine), with heterogeneous per-element shapes.

A step-by-step guide for this adapter lives in the [gemmkit Guide](https://someb1oody.github.io/gemmkit/en/gemmkit-faer/Using_gemmkit_with_faer.html).

## Usage

```toml
[dependencies]
gemmkit-faer = "0.1"
gemmkit = "0.1" # for the Parallelism argument, which is not re-exported
faer = "0.24"
```

```rust
use faer::Mat;

fn main() {
    let a = Mat::from_fn(2, 2, |i, j| [[1.0_f32, 2.0], [3.0, 4.0]][i][j]);
    let b = Mat::from_fn(2, 2, |i, j| [[5.0_f32, 6.0], [7.0, 8.0]][i][j]);
    let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
    assert_eq!(c[(0, 0)], 19.0);
    assert_eq!(c[(1, 1)], 50.0);
}
```

`dot` returns `A*B` in a fresh column-major `Mat`. For the general update `C <- alpha*A*B + beta*C` in place, use `gemm(alpha, a, b, beta, c, par)`, where `par` is a `gemmkit::Parallelism`; `gemm_with` runs the same call against a caller-owned `gemmkit::Workspace`. The element type `T` is `f32` or `f64`, plus `f16` and `bf16` under `half`.

Beyond `gemm` and `dot`, the crate exposes:

- Prepacked-operand reuse: `prepack_rhs` / `prepack_lhs` produce a reusable handle consumed by `gemm_packed_b` / `gemm_packed_a` for fixed-weight loops.
- Complex (`complex` feature): `gemm_cplx` / `dot_cplx` over faer's `c32` / `c64`, with optional per-operand conjugation.
- Integer (`int8` feature): `gemm_i8` / `dot_i8` take `i8` inputs and accumulate into an `i32` output.
- Fused epilogue (`epilogue` feature): `gemm_fused` computes `C <- act(alpha*A*B + beta*C + bias)` in one pass, `gemm_map` applies a user closure per output element (`f32` / `f64`), `gemm_i8_requant` / `gemm_i8_requant_u8` requantize the `i8` result (with `int8`), and `gemm_cplx_fused` adds a bias to a complex result (with `complex`). The prepacked entries have fused twins as well.

Each entry has a `_with` variant that reuses a caller-owned `Workspace`. See the [API docs](https://docs.rs/gemmkit-faer) for the complete list of signatures.

## Feature flags

Every feature forwards to the same-named `gemmkit` feature.

| Feature | Default | Effect |
| --- | --- | --- |
| `parallel` | Yes | rayon-based parallelism. |
| `wasm_threads` | No | Threading on `wasm32-wasip1-threads` (also enables `parallel`). |
| `half` | No | `f16` and `bf16` element types, accumulated in `f32`. |
| `complex` | No | `c32` and `c64` element types. |
| `int8` | No | `i8` inputs into an `i32` output. |
| `epilogue` | No | The fused bias/activation, requantization, and per-element map entries. |

## Supported element types

The real `f32` and `f64` paths are always built; the rest are gated behind the
features above. `T` is read straight out of faer's view, so every type below works
in faer's native `c32` / `c64` and `f16` / `bf16` spellings without conversion.

| Element type | Feature | Computes | Entry points |
| --- | --- | --- | --- |
| `f32`, `f64` | built-in | `C <- alpha*A*B + beta*C` | `gemm`, `dot`, `gemm_fused`, `gemm_map` |
| `f16`, `bf16` | `half` | same, output type in, `f32` accumulate | `gemm`, `dot`, `gemm_fused` |
| `i8` | `int8` | `i8 * i8 -> i32` | `gemm_i8`, `dot_i8` |
| `i8` (requantized) | `int8` + `epilogue` | `i8 * i8 ->` `i8` or `u8` | `gemm_i8_requant`, `gemm_i8_requant_u8` |
| `c32`, `c64` | `complex` | same, optional `conj(A)` / `conj(B)` | `gemm_cplx`, `dot_cplx`, `gemm_cplx_fused` |

Each entry also has a `_with` variant that reuses a caller-owned `Workspace`, and
the prepacked (`gemm_packed_a` / `gemm_packed_b`) and batched (`gemm_batched`)
paths carry the same element types.

## Related crates

- [gemmkit](https://crates.io/crates/gemmkit): the core engine. All algorithmic documentation (ISA dispatch, cache blocking, packing, and numeric semantics) lives there and on its [docs.rs](https://docs.rs/gemmkit) page.
- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) and [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra): sibling zero-copy adapters for other matrix libraries.
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune): install-time autotuner binary that emits a `GEMMKIT_*` environment profile for the target machine.

This adapter targets faer 0.24.

## Minimum supported Rust version

Rust 1.89.

## License

Licensed under either of [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) or [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE), at your option.
