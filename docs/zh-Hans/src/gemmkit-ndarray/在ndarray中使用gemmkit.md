# 在ndarray中使用gemmkit

`gemmkit-ndarray` 是 `ndarray` 二维数组与 gemmkit 引擎之间的一层薄桥。它本身不做任何数值计算：每个入口接受一个 `ArrayBase`，直接从中读出基指针和两个轴的步长，再把这些原始参数交给 gemmkit 的 unchecked 引擎。整个 crate 就是一层步长转接，所以 gemmkit 会的一切（运行时 ISA 选择、缓存分块、可复现的并行）都原样适用，而数组在进入时既不会被重排也不会被拷贝。

由于这些入口接受任意存储类型 `S: Data` 的 `&ArrayBase<S, Ix2>`，无论是拥有所有权的 `&Array2<T>` 还是借用的 `ArrayView2<T>` 都能用，`ArcArray`、`CowArray` 以及它们的切片同样可以。唯一的内部辅助函数值得一看，因为它就是适配器全部的数据提取逻辑：

`gemmkit-ndarray/src/common.rs`：

```rust
pub(crate) fn dims_strides<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix2>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}
```

这个 `(rows, cols, row_stride, col_stride)` 元组，再加上 `a.as_ptr()`，就是 gemmkit 需要的全部。步长是带符号的 `isize`，所以负（反转）步长和正步长一样直接转发。

## 加入项目

适配器重新导出了 fused 选择器（`Bias`、`Activation`），但不重新导出 `Parallelism` 和 `Workspace`，所以一个典型项目会同时依赖两个 crate：

```toml
[dependencies]
gemmkit-ndarray = "0.1"
gemmkit = "0.1" # 需要 Parallelism 和 Workspace
ndarray = "0.17.1"
```

`gemmkit-ndarray` 上的每个 feature 都直接转发到 `gemmkit` 中的同名 feature，因此你在这里打开某项能力，对应的入口点就会随之启用：

- `parallel`（默认）：rayon 多线程。
- `wasm_threads`：在 `wasm32-wasip1-threads` 上启用线程；隐含 `parallel`。
- `half`：`f16` / `bf16` 输入，以 `f32` 累加。
- `complex`：`Complex<f32>` / `Complex<f64>` 矩阵。
- `int8`：`i8` 输入，累加进 `i32`。
- `epilogue`：融合的偏置 / 激活、`i8` / `u8` 重量化，以及用户自定义的逐元素映射。

默认 feature 是 `["parallel"]`；受 feature 门控的各个类型族（`gemm_cplx`、`gemm_i8`、`gemm_fused` 等）在[进阶页](ndarray适配器进阶用法.md)中介绍。支持的最低 `ndarray` 版本是 `0.17.1`。

## 核心入口

三个函数覆盖了实数的普通路径。`dot` 是便捷入口：它把 `A * B` 算进一个新分配的行主序 `Array2`，读起来和 `ndarray` 自带的 `.dot()` 一样。

```rust
use ndarray::array;

let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
let c = gemmkit_ndarray::dot(&a, &b);
assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
```

`dot` 对 `T: GemmScalar` 泛型，也就是无条件支持的 `f32` 和 `f64`，再加上开启 `half` feature 后的 `f16` 和 `bf16`。它用 `Parallelism::default()` 并行，并自行分配输出，所以在你尚未持有目标矩阵、只想算一次乘积时是最合适的调用。

`gemm` 就地写入通用形式 `C <- alpha*A*B + beta*C`，`alpha`、`beta`、一个已有的累加器以及显式的 `Parallelism` 都在这里出场。它的签名是：

```rust
pub fn gemm<T, S1, S2, SC>(
    alpha: T,
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    beta: T,
    c: &mut ArrayBase<SC, Ix2>,
    par: Parallelism,
)
where
    T: GemmScalar,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    SC: DataMut<Elem = T>;
```

输出约束为 `SC: DataMut`，所以 `C` 是 `&mut Array2` 或 `ArrayViewMut2`，并且和输入一样可以是任意布局。下面 `A` 是一块行主序缓冲区，被无拷贝地转置成列主序视图，乘法以单线程运行：

```rust
use gemmkit::Parallelism;
use ndarray::{Array2, array};

// 行主序存储，无拷贝地转置成列主序视图
let a = Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0])
    .unwrap()
    .reversed_axes();
let b = Array2::from_elem((2, 2), 1.0_f32);
let mut c = Array2::zeros((2, 2));
gemmkit_ndarray::gemm(1.0, &a, &b, 0.0, &mut c, Parallelism::Serial);
assert_eq!(c, array![[4.0, 4.0], [6.0, 6.0]]);
```

## 零代价的布局

由于适配器只读步长，`ndarray` 能表达的任何二维视图都能无拷贝转发。这包括标准的 C 序（行主序）布局；来自 `.reversed_axes()`、`.t()` 或用 `.f()` 构造的数组的 F 序（列主序）视图；带非单位步长的 `.slice(...)` 窗口视图；以及来自负步长切片（如 `s![..;-1, ..]`）的反转视图，后者会产生一个负的行步长。目标 `C` 同样自由：`Array2::zeros((m, n).f())` 给出一个列主序输出，`gemm` 会直接填充它。

这里的“零拷贝”指的是*适配器*从不为规整布局而拷贝。当微内核需要连续的 panel 时，gemmkit 引擎仍会把操作数打包进自己的临时缓冲区；那种内部打包是算法的一部分，不是对转置输入的物化。关键在于：无论你的数组长什么样，你都不必为了满足调用而付出一次 `to_owned()` 或手动转置。

## panic：只查形状，不查别名

适配器只校验形状，别的一概不查。每个入口都断言内维对齐、且 `C` 与乘积匹配，不满足时以 `gemmkit-ndarray:` 开头的信息 panic 并指出出错的维度，例如 `A.cols (k) != B.rows (kb)`、`A.rows (m) != C.rows (cm)` 或 `B.cols (n) != C.cols (cn)`。维度不匹配是普通 `gemm` 或 `dot` panic 的唯一原因。

别名不在运行时检查，也无需检查：`C` 以 `&mut ArrayBase<SC, _>` 传入，这是一个独占借用，所以类型系统已经保证它不会与 `A`、`B` 的共享 `&` 借用重叠。这正是 gemmkit 的 `_unchecked` 引擎要求调用方维持的前提条件，而 `&mut` 签名免费地维持了它。（fused 入口多一项运行时检查——偏置切片是否与 `C` 重叠，见[进阶页](ndarray适配器进阶用法.md)。）

## 选择并行度

`Parallelism` 来自 `gemmkit`。`Parallelism::Serial` 在调用线程上运行；`Parallelism::Rayon(n)` 使用一个至多 `n` 线程的 rayon 线程池，而 `Rayon(0)` 会自动探测机器的核数。`Parallelism::default()` 就是 `Rayon(0)`，也是 `dot` 所用的，因此 `dot` 开箱即并行。并行路径需要 `parallel` feature（默认开启）；关掉它后，可把每次调用都当作串行。gemmkit 让串行与并行的运行在固定输入和配置下逐位可复现，所以切换 `par` 从不改变数值。线程数背后的取舍见[并行实践](../gemmkit-guide/并行实践.md)。

## 复用工作区

每个会分配的入口在调用期间都从 gemmkit 的内部线程本地池借用临时空间，所以单次 `gemm` 绝不会把一次分配泄漏进你的稳态。当你跑一个形状相近的热循环时，`_with` 变体让你转而自己持有这块临时空间：把一个 `&mut Workspace` 作为首个参数传入，它会一次性增长到循环所需的最大尺寸，之后被复用而不再分配。

```rust
use gemmkit::{Parallelism, Workspace};
use ndarray::Array2;

let mut ws = Workspace::new();
let par = Parallelism::default();
for &(m, k, n) in &[(256, 256, 256), (512, 128, 512)] {
    let a = Array2::<f32>::zeros((m, k));
    let b = Array2::<f32>::zeros((k, n));
    let mut c = Array2::<f32>::zeros((m, n));
    // 跨迭代复用 ws，至多分配一次
    gemmkit_ndarray::gemm_with(&mut ws, 1.0, &a, &b, 0.0, &mut c, par);
}
```

除了首个工作区参数，`gemm_with` 与 `gemm` 完全一致，给出相同结果。适配器里的每个类型族都有对应的 `_with` 孪生入口，所以这个模式同样适用于接下来介绍的整数、复数、fused、批量和预打包入口。如果你要用一个固定的权重矩阵去乘一串激活，工作区会与[进阶页](ndarray适配器进阶用法.md)上的预打包操作数路径自然搭配。
