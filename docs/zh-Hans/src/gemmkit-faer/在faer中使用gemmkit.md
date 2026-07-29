# 在faer中使用gemmkit

`gemmkit-faer` 是一层很薄的零拷贝桥梁，把 faer 的视图类型接到 gemmkit 的 GEMM 引擎上。它对每个输入接受一个 `MatRef<'_, T>`，对输出接受一个 `MatMut<'_, T>`。它直接从视图里读出数据指针，以及以元素为单位的行、列步长，再把它们交给 gemmkit 的底层引擎。适配器在入口处不做任何转置、拷贝或重打包。

faer 存放步长的方式，正好就是 gemmkit 引擎所需要的。因此，一个 faer `Mat`、一个转置视图、一个带偏移的子矩阵，以及一个反转（负步长）视图，都会原样抵达内核。

本 crate 面向 faer 0.24，需要 Rust 1.89。

## 安装与 feature

`gemmkit-faer` 把自己签名里出现的所有类型都重新导出了。因此，常规配置里并不需要直接依赖 `gemmkit`。

```toml
[dependencies]
gemmkit-faer = "0.1"
faer = "0.24"
```

`gemmkit_faer` 重新导出了：

- `Parallelism` 选择器，以及每个 `_with` 变体都要用到的 `Workspace` 类型。
- 融合选择器 `Bias` 与 `Activation`。
- 预打包句柄 `PackedLhs` 与 `PackedRhs`。
- 重量化参数 `Requantize` 与 `RequantScale`。
- 元素类型约束 `GemmScalar`、`FusedScalar`、`MapScalar` 与 `ComplexScalar`。当你要写一个对某个入口泛型的封装时，需要用到它们。
- 对应 feature 下的元素类型 `f16`、`bf16`、`Complex`、`c32`、`c64`。这样 `half` 与 `num-complex` 也不必进入你的 manifest。
- `tuning` 模块。

请通过适配器来用 `tuning`，而不要自己再单独依赖一份 `gemmkit`。这些旋钮是进程级的全局原子量。另一份单独解析出来的 `gemmkit` 会给你一组不同的原子量，一组适配器根本不会读取的原子量。

每个 Cargo feature 都会转发到 `gemmkit` 中的同名 feature。所以你在这里启用某个元素类型族或某个融合入口时，底层核心也会一并启用它。

- `parallel`（默认）：基于 rayon 的并行。
- `wasm_threads`：在 `wasm32-wasip1-threads` 上启用线程，同时也会启用 `parallel`。
- `half`：`f16` 与 `bf16` 元素类型，以 `f32` 累加。
- `complex`：`c32` 与 `c64` 元素类型。
- `int8`：`i8` 输入进入 `i32` 输出。
- `epilogue`：融合的偏置/激活、重量化，以及逐元素映射入口。

feature 门控的类型族与融合入口，在[进阶用法页](faer适配器进阶用法.md)中介绍。本页只讲始终可用的 `f32`/`f64`（以及 `half` 下的 `f16`/`bf16`）这层接口。

## 这里的“零拷贝”指什么


每个入口都会经过同一个小小的辅助函数，从 `MatRef` 里取出原始的组成部分。faer 已经以 `isize` 按元素为单位报告步长，反转视图为负值，这正是 gemmkit 非检查引擎所期望的形状，所以完全不需要任何转换步骤。

```rust
// gemmkit-faer/src/common.rs
pub(crate) fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (a.nrows(), a.ncols(), a.row_stride(), a.col_stride(), a.as_ptr())
}
```

适配器自己校验三个共享维度，然后在一个 `unsafe` 块内，把指针和步长转发给 gemmkit 的 `_unchecked` 引擎。安全性论证很简短：faer 的视图类型保证指针加步长描述的是一个合法的、边界内的布局。输出是一个 `MatMut`，也就是独占借用，所以 `C` 不可能与 `A` 或 `B` 存在别名。对于普通路径来说，这就是适配器的全部内容。

gemmkit 的缓存分块、ISA 分发、打包，以及并行调度，全都在核心里实现，核心那边也有相应的文档。内部机理见[架构章节](../architecture/分层结构.md)。

## gemm 与 dot

两个主力函数是 `dot` 和 `gemm`：`dot` 返回一个全新的乘积，`gemm` 则就地更新一个已有的输出。二者都对 `GemmScalar` 泛型：始终支持 `f32` 与 `f64`，在启用 `half` feature 时还支持 `f16` 与 `bf16`。

```rust
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
// A*B into a fresh column-major Mat
let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
assert_eq!(c[(0, 0)], 19.0);
assert_eq!(c[(1, 1)], 50.0);
```

`dot(a, b)` 把 `A*B` 算进一个新分配的列主序 `Mat` 中。它以默认并行度运行，也就是 `Parallelism::Rayon(0)`，会自动探测线程数。把 `dot` 当作一次性的便捷接口来用。当你自己持有输出缓冲区，或者想做通用更新时，改用 `gemm`。

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, gemm};

let a = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
let b = Mat::<f64>::from_fn(3, 5, |i, j| (i as f64) * (j as f64));
let mut c = Mat::<f64>::zeros(4, 5);
// c <- 1.5 * a * b + 2.0 * c, single-threaded
gemm(1.5, a.as_dyn_stride(), b.as_dyn_stride(), 2.0, c.as_dyn_stride_mut(), Parallelism::Serial);
```

`gemm(alpha, a, b, beta, c, par)` 就地计算 `C <- alpha*A*B + beta*C`。当 `beta == 0` 时，`gemm` 会覆盖 `C` 原有的内容，且完全不读取它们。这正是 `dot` 内部所做的事。当 `beta` 非零时，调用会在 `C` 已有的值上累加。

签名就是上面看到的样子：输入是 `MatRef<'_, T>`，输出是 `MatMut<'_, T>`，`par` 是一个 `Parallelism`。`.as_dyn_stride()` 与 `.as_dyn_stride_mut()` 这两个转换，把 faer 静态类型化的步长变成适配器所接受的动态步长视图。它们在运行时没有任何开销。

## 无需拷贝即可直通的布局

适配器始终只读取一个指针和 2 个步长。正因如此，任何 faer 视图都无需拷贝、也无需退化路径即可工作。转置操作数是常见的“行主序 A”情形：把一个列主序矩阵转置，得到的视图行步长非单位，这个视图会直接送进内核。

```rust
// `at` is k x m column-major; `.transpose()` gives an m x k view with a non-unit
// row stride - read straight through, no copy
let a = at.as_dyn_stride().transpose();
let c = gemmkit_faer::dot(a, b.as_dyn_stride());
```

带偏移的子矩阵同理：`submatrix(...)` 会移动基指针，并保留非连续的列步长。反转视图也一样：`reverse_rows()` 与 `reverse_cols()` 带有负步长。

gemmkit 的非检查路径直接处理负步长，所以一个反转的输入在 `beta` 下也会正确累加，和其他任何输入没有区别。关于引擎如何处理一般步长，见[矩阵视图与内存布局](../gemmkit-guide/矩阵视图与内存布局.md)。

## 选择并行度

每个入口都接受一个 `Parallelism`。`Parallelism::Serial` 单线程运行。`Parallelism::Rayon(n)` 用 rayon 以至多 `n` 个线程运行。`Rayon(0)` 会自动探测线程数。

gemmkit 让线程数随负载渐进增长，而不是一上来就用满每一个核心。对于固定的机器和固定的配置，同一次调用会给出可复现的结果。今天，串行与并行的运行结果也恰好逐位一致。但这种一致并不是硬性保证：可复现性契约本身只覆盖固定配置这一种情形，而线程数正是配置的一部分。调度模型见[并行实践](../gemmkit-guide/并行实践.md)指南。

## 跨调用复用工作区

`gemm` 从一个线程局部池中分配它的临时空间。每个入口也都有一个 `_with` 孪生版本。如果你在循环里驱动大量 GEMM，并想显式持有那块临时缓冲区，就改用 `_with` 版本。它把 `&mut Workspace` 作为第一个参数，并在多次调用之间复用这个工作区。

```rust
use gemmkit_faer::{Parallelism, Workspace, gemm_with};

let mut ws = Workspace::new();
for (a, b, mut c) in problems {
    // same result as `gemm`, but the scratch buffer is reused
    gemm_with(&mut ws, 1.0, a, b, 0.0, c.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

一个 `Workspace` 会先增长到能容纳它见过的最大问题，此后 gemmkit 就直接复用它。这对一连串规模相近的中小型 GEMM 最有意义，否则分配开销原本会显现在性能剖析中。

## panic 行为

适配器在分发之前会先校验三个共享维度，遇到不匹配就 panic：`A.cols` 必须等于 `B.rows`，`A.rows` 必须等于 `C.rows`，`B.cols` 必须等于 `C.cols`。适配器会给每条消息加上 `gemmkit-faer:` 前缀，并点名两个冲突的维度，例如 `gemmkit-faer: A.cols (4) != B.rows (5)`。这些是普通 `gemm`/`dot` 路径上仅有的 panic。

feature 门控的入口还会再加几种检查：偏置的长度与重叠、重量化参数，以及预打包 `C` 的朝向。这些检查复刻了 gemmkit 自身检查入口的措辞。[进阶用法页](faer适配器进阶用法.md)上的每个入口都各自列出了自己的 panic。
