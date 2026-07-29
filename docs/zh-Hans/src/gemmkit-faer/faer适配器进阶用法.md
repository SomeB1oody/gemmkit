# faer适配器进阶用法

除了 `gemm` 与 `dot`，faer 适配器还镜像了 gemmkit 其余的接口，包括额外的元素类型族、融合尾部运算、基于切片的批量 GEMM，以及预打包操作数。它们每一项都由 feature 门控。每一项也都直接从 faer 视图中读出原始指针与步长，所以转置、子矩阵与反转操作数的表现，同普通路径完全一致。本页逐一走过这些类型族，最后给出一段说明：在 faer 已自带 matmul 的前提下，这个适配器什么时候才值得使用。

[入门页](在faer中使用gemmkit.md)介绍了安装、零拷贝机制、`gemm`/`gemm_with`/`dot`、并行度，以及工作区模式。本页的内容都建立在它之上。与普通路径一样，每个入口也都有一个复用调用方持有的 `Workspace` 的 `_with` 孪生版本。

## 整数 GEMM（`int8`）

在 `int8` feature 下，`gemm_i8` 与 `dot_i8` 接受 `i8` 输入，并累加进 `i32` 输出。输入与输出的元素类型不同。这正是它作为独立入口、而非泛型的又一个实例的原因。faer 的视图类型对元素是泛型的，所以一个 `i8` 的 `MatRef` 和一个 `i32` 的 `MatMut` 无需任何特殊处理。

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, dot_i8, gemm_i8};

let a = Mat::<i8>::from_fn(16, 12, |i, j| ((i + j) as i8 % 7) - 3);
let b = Mat::<i8>::from_fn(12, 10, |i, j| ((i * 2 + j) as i8 % 5) - 2);
// i8 * i8 accumulated into a fresh Mat<i32>
let c = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());

// Mat::zeros is ComplexField-only, so integer outputs use from_fn
let mut acc = Mat::<i32>::from_fn(16, 10, |_, _| 0);
// c <- 3 * a * b + (-2) * c, all of alpha/beta/C in i32
gemm_i8(3, a.as_dyn_stride(), b.as_dyn_stride(), -2, acc.as_dyn_stride_mut(), Parallelism::Serial);
```

`alpha`、`beta` 和 `C` 都是 `i32`。算术运算在溢出时回绕。这就是整数 GEMM 的常规语义。

## 重量化输出（`int8` + `epilogue`）

在同时启用 `int8` 与 `epilogue` 时，`gemm_i8_requant` 把重量化这一步融进了内核的写回。`i8` 输入相乘后进入一个 `i32` 累加器。内核在一趟之内，把这个累加器缩放、加偏置、取整，并夹取为 `i8` 输出。它从不物化完整的 `m*n` 个 `i32`。`gemm_i8_requant_u8` 做的是同一件事，只是夹取到无符号的 `u8` 输出，也就是 ONNX QLinearMatMul 风格的激活值域。

这里没有 `alpha`，因为它已经折进了 scale。这里也没有 `beta`，因为在一个量化输出上累加是没有意义的。

参数装在一个 `Requantize` 里。crate 已经重新导出了这个类型，所以你无需为它单独依赖 `gemmkit`。`scale` 是一个 `RequantScale`，可以是 `PerTensor(f32)`，也可以是逐通道的 `PerRow(&[f32])`。`zero_point` 在取整之后以整数形式并入。`bias` 是一个可选的逐行 `i32` 向量，在缩放之前加到累加器上。

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, RequantScale, Requantize, gemm_i8_requant};

let (m, n) = (17, 13);
let bias: Vec<i32> = (0..m as i32).map(|i| 40 * i - 200).collect();
let mut c = Mat::<i8>::from_fn(m, n, |_, _| 0);
let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,
    bias: Some(&bias),
};
gemm_i8_requant(a.as_dyn_stride(), b.as_dyn_stride(), req, c.as_dyn_stride_mut(), Parallelism::Serial);
```

输出为 `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO, HI)`，采用向偶数取整。`[LO, HI]` 在 `i8` 入口是 `[-128, 127]`，在 `u8` 入口是 `[0, 255]`。

适配器在分发之前会校验重量化参数，复刻 gemmkit 自身检查入口的措辞。校验覆盖：

- scale 非有限或非正。
- 逐行 scale 切片长度不对，或者与 `C` 重叠。
- `zero_point` 超出输出值域。
- 偏置长度不对，或者与 `C` 重叠。

这套校验是针对 `C` 字节足迹的原始指针运算。适配器从不构造 `C` 切片。这正是它能把负步长视图安全转发给底层引擎的原因。

## 复数 GEMM（`complex`）

在 `complex` feature 下，`gemm_cplx`、`gemm_cplx_with` 与 `dot_cplx` 作用于复数矩阵，并可对每个操作数分别选择是否取共轭。元素类型 `T` 是 `Complex<f32>` 或 `Complex<f64>`。

这并非 faer 之外的另一套表示。faer 0.24 的 `c32` 与 `c64` 就是 `num_complex::Complex<f32>` 与 `num_complex::Complex<f64>` 的类型别名。本 crate 重新导出的正是这两个类型，命名为 `Complex`，并带有同名的 `c32`/`c64` 别名，`ComplexScalar` 约束也建立在它们之上。因此，一个 faer 复数 `Mat` 抵达适配器时不需要任何转换，就像实数一样。

`gemm_cplx` 之所以独立于 `gemm`，是因为共轭标志放不进同质的接口。它计算 `C <- alpha*op(A)*op(B) + beta*C`，其中当 `conj_a` 置位时 `op(A) = conj(A)`，当 `conj_b` 置位时 `op(B) = conj(B)`。

`cplx.rs` 里的实现取出与实数路径相同的原始部件，并把两个 `bool` 标志一路传给 `gemm_cplx_unchecked`。除此之外没有任何区别，所以转置、子矩阵和反转视图的表现完全一致。`dot_cplx` 是非共轭 `A*B` 的便捷接口。

```rust
use faer::Mat;
use gemmkit_faer::{Complex, Parallelism, gemm_cplx};

type C = Complex<f64>;
let a = Mat::<C>::from_fn(12, 9, |i, j| C::new(i as f64, j as f64));
let b = Mat::<C>::from_fn(9, 7, |i, j| C::new((i + j) as f64, 1.0));
let mut c = Mat::<C>::zeros(12, 7);
// C <- alpha * conj(A) * B + beta * C
gemm_cplx(
    C::new(1.3, -0.4),
    a.as_dyn_stride(), true,   // conjugate A
    b.as_dyn_stride(), false,  // leave B
    C::new(0.5, 0.7),
    c.as_dyn_stride_mut(),
    Parallelism::Serial,
);
```

在 `complex` 加 `epilogue` 下，还有 `gemm_cplx_fused`。它在一趟之内加上一个可选偏置：`C <- alpha*op(A)*op(B) + beta*C + bias`。偏置是 `Bias::PerRow`（长度为 `A.rows`）或 `Bias::PerCol`（长度为 `B.cols`）。gemmkit 会把它原样加到该行或该列的每个元素上，绝不取共轭。

这里刻意没有激活参数。像 ReLU 这样基于序的激活，在复数上是没有定义的，所以融合的复数入口只带一个偏置。

## 融合偏置与激活（`epilogue`）

在 `epilogue` 下，`gemm_fused` 在一趟之内计算 `C <- act(alpha*A*B + beta*C + bias)`。可选的 `Bias` 是 `PerRow` 或 `PerCol`。可选的 `Activation` 是 `Relu` 或 `LeakyRelu(slope)`，最后应用。两者都传 `None`，就恰好等于 `gemm`。crate 把这两个选择子都重新导出了。

```rust
use gemmkit_faer::{Activation, Bias, Parallelism, gemm_fused};

let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
// C <- relu(1.3 * A*B - 0.7 * C + rowbias)
gemm_fused(
    1.3, a.as_dyn_stride(), b.as_dyn_stride(), -0.7,
    c.as_dyn_stride_mut(),
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::Rayon(0),
);
```

对 `f32`/`f64` 来说，任何形状下，融合结果都与先做普通 `gemm` 再做同样的标量映射逐位相同。尾部运算折进了同一个内核的写回，不扰动累加的顺序。今天，串行与并行的运行结果也恰好逐位一致。但这种一致只是当前实现的一个特性，不是硬性保证：可复现性契约本身只覆盖固定配置这一种情形，而线程数正是配置的一部分。

对 `f16`/`bf16`（在 `half` 下）来说，融合结果更精确，而不是逐位相同。单独调用 `gemm()` 再做窄类型映射，会先取整到窄类型，再拓宽回来，然后再取整一次。融合路径省掉了这次多余的取整：偏置与斜率精确地拓宽到 `f32`，尾部运算在 `f32` 中进行，结果只对窄输出取整一次。所以对 `f16`/`bf16`，融合结果更精确，但它和先 `gemm` 再做窄类型映射并不逐位相同。上面对 `f32`/`f64` 的逐位保证，并不延伸到这些窄类型上。不过，串行与并行的运行结果，对这些类型今天也依然逐位一致，同样只在固定配置的可复现性契约之下成立。完整契约见[融合 Epilogue](../gemmkit-guide/融合Epilogue.md)指南。

对任意的逐元素函数，还有 `gemm_map`（仅 `f32`/`f64`）：`C[r,c] <- f(alpha*A*B + beta*C, r, c)`。闭包在每个输出元素的最终值上恰好运行一次，`(r, c)` 处于 `C` 的用户坐标系中。

用 `gemm_map` 来做 GELU、sigmoid、夹取，或者与位置相关的变换。普通的偏置或 ReLU，则优先用 `gemm_fused`，因为它会向量化。`gemm_map` 每个元素都要付一次间接调用的代价。

## 批量 GEMM

faer 没有三维数组类型，所以 gemmkit-faer 改用切片来表达批量 GEMM。`gemm_batched` 取一个 `&[(MatRef, MatRef)]`，也就是逐元素的 `(A, B)` 输入，与一个 `&mut [MatMut]` 的 `C` 输出按位置配对。所有元素共享同一个 `alpha`、`beta` 和 `Parallelism`。

gemmkit 的指针数组引擎把这个批次跨元素并行。它的调度器把整个 GEMM 分配给工作线程。每个工作线程串行运行自己的 GEMM，并为它保持缓存热度。

```rust
use faer::Mat;
use gemmkit_faer::{Parallelism, gemm_batched};

let a = Mat::from_fn(2, 2, |i, j| [[1.0_f64, 2.0], [3.0, 4.0]][i][j]);
let b = Mat::from_fn(2, 2, |i, j| [[5.0_f64, 6.0], [7.0, 8.0]][i][j]);
let mut c0 = Mat::<f64>::zeros(2, 2);
let mut c1 = Mat::<f64>::zeros(2, 2);
let ab = [
    (a.as_dyn_stride(), b.as_dyn_stride()),
    (a.as_dyn_stride(), b.as_dyn_stride()),
];
let mut c = [c0.as_dyn_stride_mut(), c1.as_dyn_stride_mut()];
gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
```

只要每个元素自身的维度自洽，各元素的形状可以不同，也就是一个异构批次。如果输入与输出的数量不一致，调用就会 panic。如果任何元素的维度不自洽，调用也会 panic，并点名出错的元素下标。

每个元素都会重新经过完整引擎分发。所以这个批次复现的是一次普通的 `gemm` 循环。它对线程数是确定的，因为每个元素都完整地运行在一个工作线程上。出于同样的原因，串行与批量并行的输出逐位相同。

这里没有融合的批量入口：ndarray 适配器提供了共享尾部运算的批量形式，但核心里没有它的指针数组对应物。调度策略见[批量 GEMM](../gemmkit-guide/批量GEMM.md)。

## 预打包操作数

当一个操作数在多次调用之间保持固定，比如权重面对一连串激活值时，把它预打包一次，就能省掉每次调用的重打包。`prepack_rhs` 把一个 `B` 变成可复用的 `PackedRhs`，由 `gemm_packed_b` 消费。`prepack_lhs` 把一个 `A` 变成 `PackedLhs`，由 `gemm_packed_a` 消费。crate 把这两个句柄都重新导出了。

```rust
use gemmkit_faer::{Parallelism, gemm_packed_b, prepack_rhs};

let packed = prepack_rhs(weights.as_dyn_stride()); // pack the fixed B once
for (act, mut out) in stream {
    // out must be column-major-ish (|col stride| >= |row stride|)
    gemm_packed_b(1.0, act.as_dyn_stride(), &packed, 0.0, out.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

唯一的约束是输出的朝向。预打包的 `B` 固定了操作数的角色，所以 `gemm_packed_b` 需要一个偏列主序的 `C`（`|col stride| >= |row stride|`）。行主序的 `C` 会交换 `A`/`B` 的角色，使打包好的 RHS 失效，因此 gemmkit 会拒绝它。对称地，`gemm_packed_a` 需要一个偏行主序的 `C`。如果输出布局不匹配，就退回去用普通的 `gemm`。

在 `epilogue` 下，预打包入口还有融合的孪生版本：`gemm_packed_b_fused` 与 `gemm_packed_a_fused`。它们各自在同一个句柄上，接受与 `gemm_fused` 相同的 `Bias`/`Activation`。复用模型见[预打包操作数](../gemmkit-guide/预打包操作数.md)指南。

## 何时该动用这个适配器

faer 自带了它自己的 matmul。对于两个 faer 矩阵的普通 `f32`/`f64` 乘积，直接用它就够了。这个适配器在你需要核心 faer 算子没有提供的能力时才值得使用，前提是要用在 faer 自己的类型上，并且不想离开 faer 生态：

- **额外的元素类型族**：`i8 -> i32` 的整数 GEMM，以及一路融合到 `i8` 或 `u8` 输出的重量化。
- **融合尾部运算**：内核在同一趟里计算偏置与激活，或任意的逐元素闭包，而不是对 `C` 再扫一遍。
- **跨调用预打包**：把一个固定权重矩阵打包一次，然后在一段长推理循环中复用它。
- **共享的调优面**：gemmkit 的三个适配器都坐在同一个引擎上，所以来自 [gemmkit-tune](../gemmkit-tune/使用gemmkit-tune调优.md) 的一份 `GEMMKIT_*` 环境配置，对它们全部适用。旋钮面见[调优旋钮](../gemmkit-guide/调优旋钮.md)。

如果以上都不适用，就改用 faer 内置的 matmul。它是更简单的选择。这个适配器是它的补充，不是替代。
