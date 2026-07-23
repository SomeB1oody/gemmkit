[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/README.md)

# gemmkit

[![CI](https://github.com/SomeB1oody/gemmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/SomeB1oody/gemmkit/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/gemmkit.svg)](https://crates.io/crates/gemmkit) [![docs.rs](https://img.shields.io/docsrs/gemmkit)](https://docs.rs/gemmkit)

一个纯 Rust 实现的通用矩阵乘法（GEMM）工作空间：在带步长（stride）的视图（或裸指针）上计算
`C <- alpha*A*B + beta*C`，并在运行时选择当前可用的最优指令集。

核心引擎开箱即用地支持 `f32` 和 `f64`；在 Cargo feature 之后，还支持
`f16`/`bf16`（以 `f32` 累加的混合精度）、`i8 -> i32` 整数以及 `c32`/`c64` 复数数据。
运行时 ISA 分发覆盖 x86-64 FMA 与 AVX-512F（`int8` 走 AVX-512 VNNI，`bf16` 走
AVX-512 BF16）、aarch64 NEON 和 wasm32 `simd128`，并以可移植的标量回退路径兜底；
`GEMMKIT_REQUIRE_ISA` 环境变量可以锁定或禁用某个后端。多线程是可选的（基于
rayon），且在输入与配置固定时结果可复现。关闭默认 feature 后，核心可在 `no_std`
下构建（仅依赖 `core` + `alloc`）。除了普通的 GEMM，它还提供融合尾部运算（fused
epilogue，包括偏置、激活、`i8`/`u8` 重量化以及用户自定义的逐元素映射）、面向定权重
内层循环的预打包操作数复用、批量 GEMM，以及针对矩阵-向量乘和小尺寸问题的自动带宽受限路径。

## Crates

本工作空间包含五个 crate，共享 0.1.0 版本号并同步发布。

| Crate | 说明 |
| --- | --- |
| [gemmkit](https://crates.io/crates/gemmkit) | 核心 GEMM 引擎：步长视图与裸指针入口、运行时 ISA 分发、`no_std` 支持 |
| [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) | 面向 `ndarray` 矩阵视图的零拷贝适配器 |
| [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra) | 面向 `nalgebra` 矩阵视图的零拷贝适配器 |
| [gemmkit-faer](https://crates.io/crates/gemmkit-faer) | 面向 `faer` 矩阵视图的零拷贝适配器 |
| [gemmkit-tune](https://crates.io/crates/gemmkit-tune) | 安装期自动调优器程序：在目标机器上扫描运行时旋钮，并输出一份 `GEMMKIT_*` 环境变量配置 |

这些适配器分别封装 `ndarray >= 0.17.1`、`nalgebra 0.35` 和 `faer 0.24`，并把
`parallel` / `wasm_threads` / `half` / `complex` / `int8` / `epilogue` 各个 feature
转发到 `gemmkit` 中的同名 feature。

## 快速上手

```toml
[dependencies]
gemmkit = "0.1"
```

```rust
use gemmkit::{gemm, MatMut, MatRef, Parallelism};

fn main() {
    // 2x3 times 3x2 = 2x2, all row-major
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0_f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 3, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}
```

转置通过步长来表达（`from_col_major`，或在 `MatRef::new` 中显式给出 `rs`/`cs`），
因此转置操作数无需任何拷贝。

## 元素类型与后端

下表中的每一类元素类型，在每个后端上都有对应的 SIMD 实现，并以随处可用的标量回退路径兜底。

| 类型族 | Feature | 累加器 |
| --- | --- | --- |
| `f32`, `f64` | （内置） | 同类型 |
| `f16`, `bf16` | `half` | `f32` |
| `i8 -> i32` | `int8` | `i32` |
| `c32`, `c64` | `complex` | 同类型 |

运行时选择的后端（也可用 `GEMMKIT_REQUIRE_ISA` 锁定）：

- 标量：可移植的回退路径，不要求任何目标特性
- x86-64 FMA
- x86-64 AVX-512F，其中 `int8` 使用 AVX-512 VNNI（`vpdpbusd`），`bf16` 使用 AVX-512 BF16（`vdpbf16ps`）
- aarch64 NEON
- wasm32 `simd128`（编译期特性检测）

`gemmkit` 的 Cargo feature 包括 `std` 和 `parallel`（两者均默认开启）、`wasm_threads`
（用于 `wasm32-wasip1-threads`）、`complex`、`half`、`int8`，以及 `epilogue`（融合的
偏置/激活、`i8`/`u8` 重量化和用户自定义逐元素映射）。关闭 `std` 后 crate 即为
`no_std`；`parallel` 隐含开启 `std`。

## 文档

- [gemmkit 指南](https://someb1oody.github.io/gemmkit/zh-Hans/)：完整的指南书
  （使用指南、适配器指南以及深入的架构说明）
- API 参考：[docs.rs/gemmkit](https://docs.rs/gemmkit)
- [ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md)：内部设计与扩展接缝
- [CHANGELOG.md](https://github.com/SomeB1oody/gemmkit/blob/master/CHANGELOG.md)：发布说明

## 许可协议

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或
[Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE)
双许可，由你任选其一。

最低支持的 Rust 版本：1.89（edition 2024）。
