# 在nalgebra中使用gemmkit

`gemmkit-nalgebra` 让你直接用 nalgebra 的矩阵驱动 gemmkit 引擎，不必先拷贝一份。它面向 nalgebra 0.35，接收 `&Matrix<T, R, C, S>`，其中存储类型满足 `S: RawStorage<T, R, C>` 即可。拥有所有权的 `DMatrix`、静态的 `SMatrix`，以及各种视图和切片类型都满足要求。适配器读出矩阵的数据指针和它的 2 个步长，交给 gemmkit 的底层引擎。引擎不会重排输入，不会把它转置进临时缓冲区，也不会复制一份。

nalgebra 的自然布局是列主序，这也正是 gemmkit 偏好的朝向，所以最常见的情形恰好就是最快的情形。行主序视图和一般步长视图同样能用，拷贝次数照样为零，因为引擎直接读取步长，而不是假定某种布局。

## 加入项目

`Cargo.toml` 里需要 2 个 crate。适配器把自己签名里出现的类型都重新导出了，所以常规配置不需要直接依赖 `gemmkit`。

```toml
[dependencies]
gemmkit-nalgebra = "0.1"
nalgebra = "0.35"
```

`gemmkit_nalgebra` 重新导出了好几种类型，因此项目不需要为它们直接依赖 `gemmkit`：

- `Parallelism` 选择器，以及每个 `_with` 变体所需的 `Workspace`
- 融合选择器 `Bias` 与 `Activation`
- 预打包句柄 `PackedLhs` 与 `PackedRhs`
- 重量化参数 `Requantize` 与 `RequantScale`
- 元素类型约束 `GemmScalar`、`FusedScalar`、`MapScalar`、`ComplexScalar`。当你要写一个对某个入口泛型的封装时，需要用到它们
- 对应 feature 下的元素类型 `f16`、`bf16`、`Complex`、`c32`、`c64`，因此 `half` 和 `num-complex` 也不必进入你的 manifest
- `tuning` 模块

请通过适配器去用 `tuning`，而不是自己再依赖一份 `gemmkit`。这些调优旋钮是进程级的全局原子量。第二份单独解析出来的 `gemmkit` 会给你一组适配器根本不会读取的原子量。

默认 feature 集启用了 `parallel`，它会打开引擎里基于 rayon 的线程化（`gemmkit/parallel`）。适配器的每个 feature 都只是对 `gemmkit` 上同名 feature 的一层转发：

- `half` 增加 `f16`/`bf16` 输入
- `complex` 增加 `Complex<f32>`/`Complex<f64>`
- `int8` 增加 `i8 -> i32` 路径
- `epilogue` 增加融合的偏置/激活以及逐元素映射
- `wasm_threads` 在 `wasm32-wasip1-threads` 上叠加 `parallel`

这些 feature 门控的入口在 [nalgebra 适配器进阶用法](nalgebra适配器进阶用法.md) 中讲解。本页只谈始终可用的实数标量接口。

## 3 个实数标量入口

基础接口是 3 个函数，都对 `GemmScalar` 泛型（它始终是 `f32` 和 `f64`，在 `half` feature 下另加 `f16` 和 `bf16`）。`gemm` 是做累加的乘法，`gemm_with` 是复用调用方持有的工作区的同一调用，`dot` 则是一个自行分配结果的便捷封装。

下面是 `gemm` 的原样签名，摘自 `gemmkit-nalgebra/src/float.rs`：

```rust
pub fn gemm<T, R1, C1, S1, R2, C2, S2, RC, CC, SC>(
    alpha: T,
    a: &Matrix<T, R1, C1, S1>,
    b: &Matrix<T, R2, C2, S2>,
    beta: T,
    c: &mut Matrix<T, RC, CC, SC>,
    par: Parallelism,
) where
    T: GemmScalar,
    R1: Dim,
    C1: Dim,
    S1: RawStorage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: RawStorage<T, R2, C2>,
    RC: Dim,
    CC: Dim,
    SC: RawStorageMut<T, RC, CC>,
{
    gemm_common(None, alpha, a, b, beta, c, par);
}
```

10 个泛型参数看着吓人，其实只说了一件简单的事。`A`、`B`、`C` 各自可以是任意 nalgebra 矩阵或视图，行维度、列维度和存储类型三者相互独立。`A` 和 `B` 通过 `RawStorage` 读取。`C` 需要 `RawStorageMut`，因为 gemm 会就地写入它。运算是 `C <- alpha*A*B + beta*C`。

`gemm_with` 的参数相同，只是最前面多一个 `&mut Workspace`。`dot(a, b) -> DMatrix<T>` 把 `A*B` 算进一块新分配的列主序矩阵。它内部以 `beta == 0` 调用 `gemm`，所以那块新缓冲区在被覆盖之前从不会被读取。

## 用 DMatrix 做第一次乘法

```rust
use gemmkit_nalgebra::Parallelism;
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);

// dot: A*B 写进一块新的列主序 DMatrix
let c = gemmkit_nalgebra::dot(&a, &b);
assert_eq!(c, DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));

// gemm: 就地累加 C <- alpha*A*B + beta*C
let mut acc = DMatrix::<f32>::zeros(2, 2);
gemmkit_nalgebra::gemm(1.0, &a, &b, 0.0, &mut acc, Parallelism::default());
assert_eq!(acc, c);
```

当你只想要乘积、也乐意拿回一个 `DMatrix` 时，`dot` 是趁手的工具。当你已经拥有目标矩阵时，`gemm` 才是该用的工具。用它来缩放目标矩阵（`beta`）、缩放乘积（`alpha`），或者省掉 `dot` 那次分配。`dot` 始终返回 `DMatrix<T>`，即便输入是静态矩阵也一样，因为封装只有在值的层面才知道输出维度。

## 静态矩阵与混合形状

静态矩阵走同样的函数，没有任何特判。由于每个操作数的行、列、存储泛型相互独立，静态的 `A` 可以乘一个动态的 `B`。`gemm` 写入静态的 `&mut SMatrix` 输出，也和写入 `DMatrix` 一样自然。

```rust
use nalgebra::{DMatrix, Matrix2, SMatrix};

// 静态 x 静态
let a = Matrix2::new(1.0_f32, 2.0, 3.0, 4.0);
let b = Matrix2::new(5.0_f32, 6.0, 7.0, 8.0);
let c = gemmkit_nalgebra::dot(&a, &b); // -> DMatrix<f32>
assert_eq!(c[(0, 0)], 19.0);

// 静态 A x 动态 B：相互独立的 Dim 泛型使其成立
let a34 = SMatrix::<f64, 3, 4>::from_fn(|i, j| (i as f64) - 0.5 * (j as f64) + 1.0);
let b = DMatrix::<f64>::from_element(4, 2, 0.25);
let c = gemmkit_nalgebra::dot(&a34, &b); // -> DMatrix<f64>，形状 3x2
```

## 布局与零拷贝

适配器从不拷贝操作数。它从矩阵中取出 `(rows, cols, row-stride, col-stride)`，把指针连同步长转交给引擎。源布局只决定引擎看到的是哪一组步长，不决定是否发生分配。一个列主序的 `DMatrix` 是就地读取的，一个用 `from_slice_with_strides` 构造的行主序切片也是。一个非连续的跳步视图，比如某个更大矩阵的每隔一行，同样是就地读取的。nalgebra 以非负的元素个数报告步长，适配器把它们拓宽成引擎所需的带符号步长。

引擎为喂饱其微内核所做的内部打包，与源布局无关，无论数据从哪来都会发生。这是 gemmkit 的性质，不是适配器引入的拷贝。如果你想消除对某个被反复使用的操作数的重复内部打包，就改用预打包操作数路径。它在[进阶用法页](nalgebra适配器进阶用法.md)中有说明。

## 何时 panic

这些入口会校验形状，遇到不一致就 panic，并在消息里带上出错的维度。`gemm`（及其同族）在触碰任何内存之前，先检查 3 个等式：`A.cols == B.rows`、`A.rows == C.rows`、`B.cols == C.cols`。比如内维不匹配时，会以 `gemmkit-nalgebra: A.cols (k) != B.rows (kb)` 中止，而不是越界读取。因此输出矩阵必须事先具备正确形状。`gemm` 会写入它，但不会调整其大小。而自行分配输出的 `dot` 只可能在内维检查上失败。

## 选择并行方式

每个调用都以一个 `Parallelism` 作为最后一个参数。它有 2 个变体：`Parallelism::Serial` 单线程运行，`Parallelism::Rayon(n)` 在 rayon 上以至多 `n` 个线程运行，其中 `Rayon(0)` 自动探测线程数。`Default` 是 `Rayon(0)`，也正是 `dot` 内部所用的。

对于小矩阵，或者当你已经身处一个并行区域、想避免嵌套线程化时，就传 `Parallelism::Serial`。对于空闲机器上的大乘法，`Parallelism::Rayon(0)` 能让引擎把工作铺开。线程化策略，以及引擎如何挑选线程数，见[并行实践](../gemmkit-guide/并行实践.md)。

## 复用工作区

引擎需要暂存空间来打包 `A` 和 `B` 的分块。默认情况下，它从一个线程局部的池子借用这块空间，因此在稳态下 `gemm` 和 `dot` 每次调用都不会自行分配。当你在紧凑循环里做很多次乘法、想完全掌控那块缓冲区时，`gemm_with` 会接收一个你自己持有并复用的 `&mut Workspace`：

```rust
use gemmkit_nalgebra::{Parallelism, Workspace};
use nalgebra::DMatrix;

let mut ws = Workspace::new();
let a = DMatrix::<f64>::from_element(64, 64, 1.0);
let b = DMatrix::<f64>::from_element(64, 64, 2.0);

for _ in 0..1000 {
    let mut c = DMatrix::<f64>::zeros(64, 64);
    gemmkit_nalgebra::gemm_with(&mut ws, 1.0, &a, &b, 0.0, &mut c, Parallelism::default());
}
```

同一个 `Workspace` 可以在各次迭代里支撑不同形状的乘法。它会增长到能容纳所见过的最大那次，并保留该容量。`Workspace::new()` 从空开始，在第一次调用把它填满之前不花任何代价。`_with` 形式在所有做累加的入口上都有，包括 feature 门控的那些。同样的复用模式也适用于整数、复数和融合调用。
