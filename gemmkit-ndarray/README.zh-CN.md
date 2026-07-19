[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-ndarray/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-ndarray/README.md)

# gemmkit-ndarray

[![crates.io](https://img.shields.io/crates/v/gemmkit-ndarray.svg)](https://crates.io/crates/gemmkit-ndarray) [![docs.rs](https://img.shields.io/docsrs/gemmkit-ndarray)](https://docs.rs/gemmkit-ndarray)

面向 [`gemmkit`](https://crates.io/crates/gemmkit) GEMM 引擎的零拷贝 [`ndarray`](https://crates.io/crates/ndarray) 适配器。每个入口都接受 `&ArrayBase<S, Ix2>`，其中存储类型 `S: Data` 任意（因此 `ArrayView2` 和 `&Array2` 都可用），直接从数组中读出指针和步长，再转交给 gemmkit 的引擎。因此 C 序、F 序、一般步长以及反转（负步长）视图都无需拷贝即可工作，输出参数 `&mut ArrayBase<SC, Ix2>` 同样可以是任意布局。

该适配器完整镜像了核心 API：实数 `f32` / `f64`（在 `half` 下还有 `f16` / `bf16`）、`complex` 下的复数、`int8` 下的 `i8 -> i32`，以及 `epilogue` 下的融合尾部运算（fused epilogue）入口（偏置、激活、`i8` / `u8` 重量化，以及用户自定义的逐元素映射），另外还有预打包操作数的复用路径（`prepack_lhs` / `prepack_rhs`）。它也是唯一带有批量 GEMM 入口的适配器，映射到 ndarray 的三维数组类型（`Ix3`，批次维为 0 轴）。完整的入口点列表见 [API 文档](https://docs.rs/gemmkit-ndarray)。

## 用法

```toml
[dependencies]
gemmkit-ndarray = "0.1"
gemmkit = "0.1" # for the Parallelism argument, which is not re-exported
ndarray = "0.17.1"
```

`dot` 在一个新建的行主序 `Array2` 中返回乘积 `A * B`：

```rust
use ndarray::array;

fn main() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
    let c = gemmkit_ndarray::dot(&a, &b);
    assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
}
```

`gemm` 就地写入通用形式 `C <- alpha*A*B + beta*C`，并且接受任意布局而无需拷贝，包括转置后的（列主序）视图：

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

可复用 `Workspace` 的变体（`gemm_with` 等）以及各个 feature 门控的类型族都遵循同样的形式，详见 [API 文档](https://docs.rs/gemmkit-ndarray)。

## Cargo feature

每个 flag 都转发到 `gemmkit` 中的同名 feature。

- `parallel`（默认）：基于 rayon 的多线程（`gemmkit/parallel`）。
- `wasm_threads`：在 `wasm32-wasip1-threads` 上启用线程；隐含 `parallel`。
- `half`：`f16` / `bf16` 输入，以 `f32` 累加。
- `complex`：`Complex<f32>` / `Complex<f64>` 矩阵。
- `int8`：`i8` 输入，累加进 `i32`。
- `epilogue`：融合的偏置 / 激活、`i8` / `u8` 重量化，以及用户自定义的逐元素映射。

## 相关 crate

- [`gemmkit`](https://crates.io/crates/gemmkit)：核心引擎。所有算法相关的文档都在那里以及它的 [docs.rs 页面](https://docs.rs/gemmkit)上；本适配器只负责把 ndarray 的类型映射过去。
- [`gemmkit-nalgebra`](https://crates.io/crates/gemmkit-nalgebra) 和 [`gemmkit-faer`](https://crates.io/crates/gemmkit-faer)：面向 nalgebra 与 faer 的同类适配器。
- [`gemmkit-tune`](https://crates.io/crates/gemmkit-tune)：安装期自动调优二进制程序。

支持的 `ndarray` 版本：`>= 0.17.1`。

## 许可证

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或 [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE) 双许可，由你任选其一。

## 最低支持的 Rust 版本

Rust 1.89。
