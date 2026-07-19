[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-nalgebra/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-nalgebra/README.md)

# gemmkit-nalgebra

[![crates.io](https://img.shields.io/crates/v/gemmkit-nalgebra.svg)](https://crates.io/crates/gemmkit-nalgebra) [![docs.rs](https://img.shields.io/docsrs/gemmkit-nalgebra)](https://docs.rs/gemmkit-nalgebra)

架设在 [gemmkit](https://crates.io/crates/gemmkit) GEMM 引擎之上的零拷贝
[nalgebra](https://crates.io/crates/nalgebra) 0.35 适配器。输入操作数以
`&Matrix<T, R, C, S>` 的形式接收，存储类型 `S: RawStorage<T, R, C>` 任意，因此
`DMatrix`、静态的 `SMatrix` 以及各种视图类型都被接受；输出是存储类型为
`RawStorageMut` 的 `&mut Matrix`。适配器直接从矩阵中读出指针和步长并转交给 gemmkit
的底层引擎，因此列主序（nalgebra 的自然布局）、行主序以及一般步长的视图都无需拷贝
即可工作。

对外暴露的 API 镜像了核心引擎。实数标量的 `gemm`、`gemm_with` 和 `dot` 对
`gemmkit::GemmScalar` 泛型（`f32`/`f64`，在 `half` feature 下还有 `f16`/`bf16`）。
`prepack_lhs`/`prepack_rhs` 构建可复用的打包句柄，供
`gemm_packed_a`/`gemm_packed_b` 的固定操作数循环使用。feature 门控的类型族补充了整数
（`gemm_i8`/`dot_i8`，`i8 -> i32`）和复数（`gemm_cplx`/`dot_cplx`）入口，而
`epilogue` feature 则加入了融合尾部运算（fused epilogue）入口：`gemm_fused`（偏置加
激活）、逐元素的 `gemm_map`、做重量化的 `gemm_i8_requant`/`gemm_i8_requant_u8`
（`int8` + `epilogue`），以及仅带偏置的 `gemm_cplx_fused`（`complex` + `epilogue`）。
nalgebra 没有三维数组类型，所以批量 GEMM（`gemm_batched`）以每个批次元素的
`(&A, &B)` 输入切片搭配 `&mut C` 输出切片的形式接收批次（走 gemmkit 的指针数组批量
引擎），且每个批次元素可以有不同的形状。

本适配器的分步教程见
[gemmkit 指南](https://someb1oody.github.io/gemmkit/zh-Hans/gemmkit-nalgebra/在nalgebra中使用gemmkit.html)。

## 用法

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

`dot(a, b)` 在一个新建的列主序 `DMatrix` 中返回 `A*B`。若需要累加形式
`C <- alpha*A*B + beta*C`，请调用 `gemm(alpha, &a, &b, beta, &mut c, par)`，其中 `par`
为一个 `gemmkit::Parallelism` 值。

## Cargo feature

每个 flag 都转发到 `gemmkit` 上的同名 feature。

- `parallel`（默认）：基于 rayon 的并行（`gemmkit/parallel`）。
- `wasm_threads`：为 `wasm32-wasip1-threads` 启用 `parallel` 和 `gemmkit/wasm_threads`。
- `half`：`f16`/`bf16` 输入，以 `f32` 累加。
- `complex`：`Complex<f32>`/`Complex<f64>` 入口，可选做共轭。
- `int8`：`i8 -> i32` 整数入口。
- `epilogue`：融合的偏置/激活、逐元素映射，以及（配合 `int8`）`i8`/`u8` 重量化。

## 相关 crate

- [gemmkit](https://crates.io/crates/gemmkit)：核心引擎。所有算法相关的文档都在那里以及
  它的 [docs.rs](https://docs.rs/gemmkit) 页面上，包括各个 ISA 后端和
  `GEMMKIT_REQUIRE_ISA` 固定开关。
- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) 和
  [gemmkit-faer](https://crates.io/crates/gemmkit-faer)：面向 ndarray 与 faer 矩阵类型的
  同类零拷贝适配器。
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune)：安装期自动调优二进制程序。

## 许可证

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或
[Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE) 双许可，
由你任选其一。

## 最低支持的 Rust 版本

Rust 1.89。
