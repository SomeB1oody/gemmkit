[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-faer/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-faer/README.md)

# gemmkit-faer

[![crates.io](https://img.shields.io/crates/v/gemmkit-faer.svg)](https://crates.io/crates/gemmkit-faer) [![docs.rs](https://img.shields.io/docsrs/gemmkit-faer)](https://docs.rs/gemmkit-faer)

面向 [gemmkit](https://crates.io/crates/gemmkit) GEMM 引擎的零拷贝 [faer](https://crates.io/crates/faer) 适配器。它接受 faer 的视图类型（输入用 `MatRef<'_, T>`，输出用 `MatMut<'_, T>`），直接从每个视图中读出数据指针以及以元素为单位的 `isize` 行步长和列步长，再转交给 gemmkit 的底层引擎。因此 faer 的列主序布局、转置视图、子矩阵以及反转（负步长）视图都无需拷贝即可工作。

各入口点镜像了 gemmkit 的核心 API，包括 feature 门控的元素类型族（`half`、`complex`、`int8`）和融合尾部运算（fused epilogue）入口（`epilogue`）。faer 没有三维数组类型，所以批量 GEMM（`gemm_batched`）以每个批次元素的 `(A, B)` `MatRef` 输入切片搭配 `&mut C` `MatMut` 输出切片的形式接收批次（走 gemmkit 的指针数组批量引擎），且每个批次元素可以有不同的形状。

## 用法

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

`dot` 在一个新建的列主序 `Mat` 中返回 `A*B`。若要就地执行通用更新 `C <- alpha*A*B + beta*C`，请使用 `gemm(alpha, a, b, beta, c, par)`，其中 `par` 是一个 `gemmkit::Parallelism`；`gemm_with` 执行同样的调用，但使用调用方持有的 `gemmkit::Workspace`。元素类型 `T` 为 `f32` 或 `f64`，在 `half` 下还有 `f16` 和 `bf16`。

除 `gemm` 和 `dot` 之外，本 crate 还提供：

- 预打包操作数复用：`prepack_rhs` / `prepack_lhs` 产出可复用的句柄，由 `gemm_packed_b` / `gemm_packed_a` 在固定权重的循环中消费。
- 复数（`complex` feature）：`gemm_cplx` / `dot_cplx` 作用于 faer 的 `c32` / `c64`，可对每个操作数分别选择是否共轭。
- 整数（`int8` feature）：`gemm_i8` / `dot_i8` 接受 `i8` 输入并累加进 `i32` 输出。
- 融合尾部运算（`epilogue` feature）：`gemm_fused` 一趟算出 `C <- act(alpha*A*B + beta*C + bias)`，`gemm_map` 对每个输出元素应用用户闭包（`f32` / `f64`），`gemm_i8_requant` / `gemm_i8_requant_u8` 对 `i8` 结果做重量化（需配合 `int8`），`gemm_cplx_fused` 为复数结果加上偏置（需配合 `complex`）。预打包的各入口同样有对应的融合版本。

每个入口都有一个复用调用方持有的 `Workspace` 的 `_with` 变体。完整的签名列表见 [API 文档](https://docs.rs/gemmkit-faer)。

## Cargo feature

每个 feature 都转发到 `gemmkit` 中的同名 feature。

- `parallel`（默认）：基于 rayon 的并行。
- `wasm_threads`：在 `wasm32-wasip1-threads` 上启用线程（同时启用 `parallel`）。
- `half`：`f16` 和 `bf16` 元素类型，以 `f32` 累加。
- `complex`：`c32` 和 `c64` 元素类型。
- `int8`：`i8` 输入进入 `i32` 输出。
- `epilogue`：融合的偏置/激活、重量化以及逐元素映射入口。

## 相关 crate

- [gemmkit](https://crates.io/crates/gemmkit)：核心引擎。所有算法相关的文档（ISA 分发、缓存分块、打包以及数值语义）都在那里以及它的 [docs.rs](https://docs.rs/gemmkit) 页面上。
- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) 和 [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra)：面向其他矩阵库的同类零拷贝适配器。
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune)：安装期自动调优二进制程序，为目标机器生成一份 `GEMMKIT_*` 环境变量配置。

本适配器面向 faer 0.24。

## 最低支持的 Rust 版本

Rust 1.89。

## 许可证

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或 [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE) 双许可，由你任选其一。
