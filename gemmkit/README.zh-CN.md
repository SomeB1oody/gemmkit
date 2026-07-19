[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/README.md)

# gemmkit

[![crates.io](https://img.shields.io/crates/v/gemmkit.svg)](https://crates.io/crates/gemmkit) [![docs.rs](https://img.shields.io/docsrs/gemmkit)](https://docs.rs/gemmkit)

gemmkit 是一个通用矩阵乘法（GEMM）引擎，在带步长（stride）的 `&[T]` 视图上为 f32
和 f64 计算 `C <- alpha*A*B + beta*C`，且不依赖任何矩阵库。它在运行时选择当前可用的
最优指令集，并以可移植的标量路径覆盖没有向量后端的目标平台。转置通过步长来表达
（转置视图只是交换行步长与列步长，无需拷贝）；当 `beta == 0` 时输出 `C` 不会被读取，
因此它可以是未初始化的。

入口函数 `gemm` 接受带检查的 `MatRef`/`MatMut` 切片视图，并在执行任何 unsafe 代码之前
就对形状、边界或别名重叠错误 panic。另有两档 API 用检查换控制力：`*_with` 变体接受由
调用方持有的 `Workspace`，从而避免每次调用都分配内存；`*_unchecked` 入口则直接操作裸
指针和 `isize` 步长（包括负步长），供自行校验输入的调用方使用。

除了基本的矩阵乘积，gemmkit 还提供：

- 运行时 ISA 分发：x86-64 FMA 与 AVX-512（int8 走 AVX-512 VNNI，bf16 走 AVX-512
  BF16）、aarch64 NEON、wasm32 simd128（编译期特性检测），以及标量回退路径。
  `GEMMKIT_REQUIRE_ISA` 环境变量可以锁定或禁用某个后端。
- 位于 cargo feature 之后的可选元素类型族：以 f32 累加的 f16/bf16、i8 到 i32，以及
  支持逐操作数共轭的 c32/c64。
- 预打包操作数：`prepack_rhs`/`prepack_lhs` 构建可复用的打包缓冲区，供
  `gemm_packed_b`/`gemm_packed_a` 消费，适用于共享同一固定操作数的一系列乘积。
- 批量 GEMM（`gemm_batched`），在一组彼此独立的问题上执行。
- `epilogue` feature 之后的融合尾部运算（fused epilogue）：`gemm_fused`（偏置与
  激活）、`gemm_i8_requant`（整数重量化）和 `gemm_map`（用户自定义的逐元素闭包）。
- 针对带宽受限形状（gemv、小 k 以及小 m,n）的自动特殊路径，在同样的入口之后自动选用。
- 基于 rayon 的并行，且在输入与配置固定时结果可复现；关闭默认 feature 后还可在
  `no_std` 下运行（仅需 `core` 和 `alloc`）。

## 用法

```toml
[dependencies]
gemmkit = "0.1"
```

```rust
use gemmkit::{gemm, MatMut, MatRef, Parallelism};

fn main() {
    // A 2x3 times a 3x2, both row-major, into a 2x2 result
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c: Vec<f32> = vec![0.0; 4];
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

## Feature 说明

- `std`（默认）：运行时 CPU 特性与缓存检测、`GEMMKIT_REQUIRE_ISA` 及各项调优环境
  旋钮，以及线程局部的工作区池。关闭后 crate 即为 `no_std`，仅需 `core` 和 `alloc`。
- `parallel`（默认，隐含开启 `std`）：rayon 多线程。关闭后一切照常编译并以单线程运行。
- `wasm_threads`：在 `wasm32-wasip1-threads` 目标上启用 rayon 并行。
- `complex`：c32/c64 复数 GEMM，支持对 A 或 B 的可选共轭（`gemm_cplx`）；引入
  `num-complex` 依赖。
- `half`：以 f32 累加的 f16/bf16 混合精度 GEMM；引入 `half` 依赖。
- `int8`：i8 到 i32 的整数 GEMM（`gemm_i8`）；算术在溢出时回绕。
- `epilogue`：融合尾部运算（偏置与激活、逐元素映射，以及在开启 `int8` 时的 i8/u8
  重量化）。默认关闭，因此纯 GEMM 构建不会为它的代码生成付出任何代价。

## 调优

每一个启发式阈值的解析顺序为：每次调用传入的参数，其次是编程式 setter，再次是
`GEMMKIT_*` 环境变量，最后是编译期默认值。
[gemmkit-tune](https://crates.io/crates/gemmkit-tune) 程序会在目标机器上扫描这些
旋钮，并输出一份可直接 source 的环境变量配置；各个旋钮的说明见
[docs.rs](https://docs.rs/gemmkit)。

## 相关 crate

- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray)：面向 `ndarray` 矩阵
  视图的零拷贝适配器。
- [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra)：面向 `nalgebra`
  矩阵视图的零拷贝适配器。
- [gemmkit-faer](https://crates.io/crates/gemmkit-faer)：面向 `faer` 矩阵视图的零
  拷贝适配器。
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune)：安装期自动调优器程序。

引擎的设计细节参见
[ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md)。

## 最低支持的 Rust 版本

gemmkit 需要 Rust 1.89 或更新版本。

## 许可协议

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或
[Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE)
双许可，由你任选其一。
