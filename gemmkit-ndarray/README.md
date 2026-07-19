[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-ndarray/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-ndarray/README.md)

# gemmkit-ndarray

[![crates.io](https://img.shields.io/crates/v/gemmkit-ndarray.svg)](https://crates.io/crates/gemmkit-ndarray) [![docs.rs](https://img.shields.io/docsrs/gemmkit-ndarray)](https://docs.rs/gemmkit-ndarray)

Zero-copy [`ndarray`](https://crates.io/crates/ndarray) adapter for the [`gemmkit`](https://crates.io/crates/gemmkit) GEMM engine. Each entry accepts `&ArrayBase<S, Ix2>` for any storage `S: Data` (so both `ArrayView2` and `&Array2` work), reads the pointer and strides straight out of the array, and forwards them to gemmkit's engine. C-order, F-order, general-stride, and reversed (negative-stride) views therefore all work without copying, and the output `&mut ArrayBase<SC, Ix2>` may have any layout as well.

The adapter mirrors the full core surface: real `f32` / `f64` (plus `f16` / `bf16` under `half`), complex under `complex`, `i8 -> i32` under `int8`, and the fused-epilogue entries (bias, activation, `i8` / `u8` requantization, and a user per-element map) under `epilogue`, alongside the prepacked-operand reuse path (`prepack_lhs` / `prepack_rhs`). It is also the only adapter with a batched GEMM entry, mapped onto ndarray's rank-3 array type (`Ix3`, batch on axis 0). See the [API documentation](https://docs.rs/gemmkit-ndarray) for the full list of entry points.

A step-by-step guide for this adapter lives in the [gemmkit Guide](https://someb1oody.github.io/gemmkit/en/gemmkit-ndarray/Using_gemmkit_with_ndarray.html).

## Usage

```toml
[dependencies]
gemmkit-ndarray = "0.1"
gemmkit = "0.1" # for the Parallelism argument, which is not re-exported
ndarray = "0.17.1"
```

`dot` returns the product `A * B` in a fresh row-major `Array2`:

```rust
use ndarray::array;

fn main() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
    let c = gemmkit_ndarray::dot(&a, &b);
    assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
}
```

`gemm` writes the general `C <- alpha*A*B + beta*C` in place and takes any layout without copying, including a transposed (column-major) view:

```rust
use gemmkit::Parallelism;
use ndarray::{Array2, array};

fn main() {
    // A owns a row-major buffer, transposed into a column-major view with no copy
    let a = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .reversed_axes();
    let b = Array2::from_elem((2, 2), 1.0_f32);
    let mut c = Array2::zeros((2, 2));
    gemmkit_ndarray::gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Serial);
    assert_eq!(c, array![[4.0, 4.0], [6.0, 6.0]]);
}
```

The reusable-`Workspace` variants (`gemm_with`, and so on) and the feature-gated families follow the same shape; see the [API documentation](https://docs.rs/gemmkit-ndarray).

## Feature flags

Each flag forwards to the same-named `gemmkit` feature.

- `parallel` (default): rayon-based multithreading (`gemmkit/parallel`).
- `wasm_threads`: threading on `wasm32-wasip1-threads`; implies `parallel`.
- `half`: `f16` / `bf16` inputs with `f32` accumulation.
- `complex`: `Complex<f32>` / `Complex<f64>` matrices.
- `int8`: `i8` inputs accumulating into `i32`.
- `epilogue`: fused bias / activation, `i8` / `u8` requantization, and a user per-element map.

## Related crates

- [`gemmkit`](https://crates.io/crates/gemmkit): the core engine. All algorithmic documentation lives there and on its [docs.rs page](https://docs.rs/gemmkit); this adapter only maps ndarray types onto it.
- [`gemmkit-nalgebra`](https://crates.io/crates/gemmkit-nalgebra) and [`gemmkit-faer`](https://crates.io/crates/gemmkit-faer): the sibling adapters for nalgebra and faer.
- [`gemmkit-tune`](https://crates.io/crates/gemmkit-tune): the install-time autotuner binary.

Supported `ndarray` versions: `>= 0.17.1`.

## License

Licensed under either of [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) or [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE), at your option.

## Minimum supported Rust version

Rust 1.89.
