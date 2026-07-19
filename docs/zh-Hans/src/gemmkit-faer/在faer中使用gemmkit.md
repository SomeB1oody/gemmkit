# 在faer中使用gemmkit

`gemmkit-faer` 是一层很薄的、零拷贝的桥梁，把 faer 的视图类型接到 gemmkit 的 GEMM 引擎上。它对每个输入接受一个 `MatRef<'_, T>`，对输出接受一个 `MatMut<'_, T>`，直接从视图里读出数据指针以及以元素为单位的行、列步长，再交给 gemmkit 的底层引擎。入口处不做任何转置、拷贝或重打包。由于 faer 存放步长的方式恰好就是 gemmkit 引擎想要的，一个 faer `Mat`、一个转置视图、一个带偏移的子矩阵，以及一个反转（负步长）视图，都会原样抵达内核。

本 crate 面向 faer 0.24，需要 Rust 1.89。

## 安装与 feature

`gemmkit-faer` 不会重新导出 `Parallelism` 或 `Workspace`，因此还需要依赖 `gemmkit` 来获取这两个参数类型。

```toml
[dependencies]
gemmkit-faer = "0.1"
gemmkit = "0.1" # 用于 Parallelism 与 Workspace 参数类型
faer = "0.24"
```

每个 Cargo feature 都转发到 `gemmkit` 中的同名 feature，所以你在这里启用某个元素类型族或融合入口，底层的核心也会随之打开它。

- `parallel`（默认）：基于 rayon 的并行。
- `wasm_threads`：在 `wasm32-wasip1-threads` 上启用线程（同时启用 `parallel`）。
- `half`：`f16` 与 `bf16` 元素类型，以 `f32` 累加。
- `complex`：`c32` 与 `c64` 元素类型。
- `int8`：`i8` 输入进入 `i32` 输出。
- `epilogue`：融合的偏置/激活、重量化以及逐元素映射入口。

feature 门控的类型族以及融合入口在[进阶用法页](faer适配器进阶用法.md)介绍；本页只讲始终可用的 `f32`/`f64`（以及 `half` 下的 `f16`/`bf16`）这一层接口。

## 这里的“零拷贝”指什么

每个入口都走同一个小辅助函数，把原始部件从 `MatRef` 里取出来。faer 以 `isize` 按元素单位报告步长，反转视图为负值，这正是 gemmkit 非检查引擎需要的形状，所以中间根本没有任何转换步骤。

```rust
// gemmkit-faer/src/common.rs
pub(crate) fn ref_parts<T>(a: MatRef<'_, T>) -> (usize, usize, isize, isize, *const T) {
    (a.nrows(), a.ncols(), a.row_stride(), a.col_stride(), a.as_ptr())
}
```

适配器自己校验三个共享维度，然后在一个 `unsafe` 块里把指针和步长转发给 gemmkit 的 `_unchecked` 引擎。安全性论证很短：faer 的视图类型保证指针加步长描述的是一个合法的、在界内的布局，而输出是 `MatMut`（一个独占借用），所以 `C` 不可能与 `A` 或 `B` 别名。对于普通路径，这就是适配器的全部。gemmkit 的缓存分块、ISA 分发、打包以及并行调度都在核心里，也都在那里有文档；若想看内部机理，见[架构章节](../architecture/分层结构.md)。

## gemm 与 dot

两个主力是 `dot`（返回一个全新的乘积）和 `gemm`（就地更新一个输出）。二者都对 `gemmkit::GemmScalar` 泛型：始终支持 `f32` 与 `f64`，在启用 `half` feature 时再加上 `f16` 与 `bf16`。

```rust
use faer::Mat;

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
// A*B 写入一个新建的列主序 Mat
let c = gemmkit_faer::dot(a.as_dyn_stride(), b.as_dyn_stride());
assert_eq!(c[(0, 0)], 19.0);
assert_eq!(c[(1, 1)], 50.0);
```

`dot(a, b)` 把 `A*B` 算进一个新分配的列主序 `Mat`，并以默认并行度运行（`Parallelism::Rayon(0)`，自动探测线程数）。它是一次性的便捷接口；当你自己持有输出缓冲区，或想做通用更新时，请改用 `gemm`。

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::gemm;

let a = Mat::<f64>::from_fn(4, 3, |i, j| (i + j) as f64);
let b = Mat::<f64>::from_fn(3, 5, |i, j| (i as f64) * (j as f64));
let mut c = Mat::<f64>::zeros(4, 5);
// c <- 1.5 * a * b + 2.0 * c，单线程
gemm(1.5, a.as_dyn_stride(), b.as_dyn_stride(), 2.0, c.as_dyn_stride_mut(), Parallelism::Serial);
```

`gemm(alpha, a, b, beta, c, par)` 就地计算 `C <- alpha*A*B + beta*C`。当 `beta == 0` 时，`C` 原有内容被覆盖、绝不读取（这正是 `dot` 内部所做的）；当 `beta` 非零时，调用会在 `C` 已有的值上累加。签名就是上面看到的样子：输入是 `MatRef<'_, T>`，输出是 `MatMut<'_, T>`，`par` 是一个 `gemmkit::Parallelism`。`.as_dyn_stride()` / `.as_dyn_stride_mut()` 转换把 faer 静态类型化的步长变成适配器所接受的动态步长视图；它们在运行时没有任何开销。

## 无需拷贝即可直通的布局

因为适配器只读取一个指针加两个步长，任何 faer 视图都无需拷贝或退化路径即可工作。转置操作数是常见的“行主序 A”情形：把一个列主序矩阵转置，得到的视图行步长非单位，它会直接送进内核。

```rust
// `at` 是 k x m 列主序；`.transpose()` 给出一个 m x k 视图，行步长非单位
// 直通读取，无拷贝
let a = at.as_dyn_stride().transpose();
let c = gemmkit_faer::dot(a, b.as_dyn_stride());
```

带偏移的子矩阵（`submatrix(...)`，它移动基指针并保持非连续的列步长）以及反转视图（`reverse_rows()` / `reverse_cols()`，它带有负步长）同理。gemmkit 的非检查路径直接处理负步长，因此一个反转的输入在 `beta` 下会像其他任何输入一样正确累加。关于引擎如何处理一般步长，见[矩阵视图与内存布局](../gemmkit-guide/矩阵视图与内存布局.md)。

## 选择并行度

每个入口都取一个 `gemmkit::Parallelism`。`Parallelism::Serial` 单线程运行；`Parallelism::Rayon(n)` 用 rayon 以至多 `n` 个线程运行，`Rayon(0)` 则自动探测。gemmkit 让线程数随负载渐进增长，而不是一上来就用满所有核心，并且在固定配置下结果对线程数可复现，因此在 `Serial` 与 `Rayon` 之间切换不会改变你得到的答案。调度模型见[并行实践](../gemmkit-guide/并行实践.md)指南。

## 跨调用复用工作区

`gemm` 从一个线程局部池分配它的临时空间。如果你在循环里驱动大量 GEMM，并想显式持有那块临时缓冲区，每个入口都有一个 `_with` 孪生版本，把 `&mut gemmkit::Workspace` 作为第一个参数并在多次调用间复用它。

```rust
use gemmkit::{Parallelism, Workspace};
use gemmkit_faer::gemm_with;

let mut ws = Workspace::new();
for (a, b, mut c) in problems {
    // 结果与 `gemm` 相同，但复用了临时缓冲区
    gemm_with(&mut ws, 1.0, a, b, 0.0, c.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

一个 `Workspace` 会增长到能容下它见过的最大问题，此后便被复用；这对一连串规模相近的中小型 GEMM 最有意义，否则分配开销会显现在性能剖析里。

## panic 行为

适配器在分发前校验三个共享维度，遇到不匹配就 panic：`A.cols` 必须等于 `B.rows`，`A.rows` 必须等于 `C.rows`，`B.cols` 必须等于 `C.cols`。消息以 `gemmkit-faer:` 为前缀并点名两个冲突的尺寸，例如 `gemmkit-faer: A.cols (4) != B.rows (5)`。这些是普通 `gemm`/`dot` 路径上仅有的 panic。feature 门控的入口会再加几种（偏置长度与重叠、重量化参数、预打包 `C` 的朝向）；它们复刻 gemmkit 自身检查入口的措辞，并在[进阶用法页](faer适配器进阶用法.md)随各入口列出。
