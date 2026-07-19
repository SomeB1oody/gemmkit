# faer适配器进阶用法

除了 `gemm` 与 `dot`，faer 适配器还镜像了 gemmkit 其余的接口：额外的元素类型族、融合尾部运算、基于切片的批量 GEMM，以及预打包操作数。每一项都由 feature 门控，每一项都直接从 faer 视图里读出原始指针与步长，因此转置、子矩阵和反转操作数的表现，与普通路径上完全一致。本页逐一走过这些类型族，最后给出一段坦诚的说明：在 faer 已自带高性能 matmul 的前提下，何时才该动用这个适配器。

[入门页](在faer中使用gemmkit.md)介绍了安装、零拷贝机制、`gemm`/`gemm_with`/`dot`、并行度以及工作区模式；本页在其之上展开。与普通路径一样，每个入口也都有一个复用调用方持有的 `gemmkit::Workspace` 的 `_with` 孪生版本。

## 整数 GEMM（`int8`）

在 `int8` feature 下，`gemm_i8` 与 `dot_i8` 接受 `i8` 输入并累加进 `i32` 输出。输入与输出的元素类型不同，这正是它作为独立入口、而非泛型的又一实例的原因；faer 的视图类型对元素泛型，所以一个 `i8` 的 `MatRef` 和一个 `i32` 的 `MatMut` 无需任何特殊处理。

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::{dot_i8, gemm_i8};

let a = Mat::<i8>::from_fn(16, 12, |i, j| ((i + j) as i8 % 7) - 3);
let b = Mat::<i8>::from_fn(12, 10, |i, j| ((i * 2 + j) as i8 % 5) - 2);
// i8 * i8 累加进一个新建的 Mat<i32>
let c = dot_i8(a.as_dyn_stride(), b.as_dyn_stride());

// Mat::zeros 只支持 ComplexField，整数输出改用 from_fn
let mut acc = Mat::<i32>::from_fn(16, 10, |_, _| 0);
// c <- 3 * a * b + (-2) * c，alpha/beta/C 均为 i32
gemm_i8(3, a.as_dyn_stride(), b.as_dyn_stride(), -2, acc.as_dyn_stride_mut(), Parallelism::Serial);
```

`alpha`、`beta` 和 `C` 都是 `i32`，算术在溢出时回绕，即整数 GEMM 的惯例语义。

## 重量化输出（`int8` + `epilogue`）

同时启用 `int8` 与 `epilogue` 时，`gemm_i8_requant` 把重量化步骤融进内核的写回：`i8` 输入乘进一个 `i32` 累加器，随后在一趟内被缩放、加偏置、取整并夹取为 `i8` 输出，全程不必物化完整的 `m*n` 个 `i32`。`gemm_i8_requant_u8` 与之相同，但夹取到无符号 `u8` 输出（ONNX QLinearMatMul 风格的激活值域）。这里没有 `alpha`（它折进 scale），也没有 `beta`（在量化输出上累加是无定义的）。

参数装在一个 `Requantize` 里，crate 已重新导出它，因此你无需为它依赖 `gemmkit`。`scale` 是一个 `RequantScale`，可为 `PerTensor(f32)` 或逐通道的 `PerRow(&[f32])`；`zero_point` 在取整之后以整数并入；`bias` 是一个可选的逐行 `i32` 向量，在缩放前加到累加器上。

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_i8_requant, RequantScale, Requantize};

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

输出为 `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO, HI)`，采用向偶数取整，其中 `[LO, HI]` 对 `i8` 入口是 `[-128, 127]`，对 `u8` 入口是 `[0, 255]`。适配器在分发前校验重量化参数，复刻 gemmkit 自身检查入口的措辞（非有限或非正的 scale、长度不对或与 `C` 重叠的逐行 scale 切片、超出输出值域的 `zero_point`，或长度不对或与 `C` 重叠的偏置）。这套校验是针对 `C` 字节足迹的原始指针运算；适配器绝不构造 `C` 切片，正因如此它才能安全地把负步长视图转发给底层引擎。

## 复数 GEMM（`complex`）

在 `complex` feature 下，`gemm_cplx`、`gemm_cplx_with` 与 `dot_cplx` 作用于复数矩阵，可对每个操作数分别选择是否共轭。元素类型 `T` 是 `Complex<f32>` 或 `Complex<f64>`。这并非 faer 之外的另一套表示：faer 0.24 的 `c32` 与 `c64` 就是 `num_complex::Complex<f32>` 与 `num_complex::Complex<f64>` 的类型别名，与 gemmkit 重新导出为 `gemmkit::Complex`、并作为 `ComplexScalar` 约束的类型完全相同。因此一个 faer 复数 `Mat` 抵达适配器时不需要任何转换，就像实数一样。

`gemm_cplx` 之所以独立于 `gemm`，是因为共轭标志放不进同质接口。它计算 `C <- alpha*op(A)*op(B) + beta*C`，其中当 `conj_a` 置位时 `op(A) = conj(A)`，当 `conj_b` 置位时 `op(B) = conj(B)`。`cplx.rs` 里的实现取出与实数路径相同的原始部件，并把两个 `bool` 标志一路传给 `gemm_cplx_unchecked`；除此之外没有区别，所以转置、子矩阵和反转视图表现一致。`dot_cplx` 是非共轭 `A*B` 的便捷接口。

```rust
use faer::Mat;
use gemmkit::{Complex, Parallelism};
use gemmkit_faer::gemm_cplx;

type C = Complex<f64>;
let a = Mat::<C>::from_fn(12, 9, |i, j| C::new(i as f64, j as f64));
let b = Mat::<C>::from_fn(9, 7, |i, j| C::new((i + j) as f64, 1.0));
let mut c = Mat::<C>::zeros(12, 7);
// C <- alpha * conj(A) * B + beta * C
gemm_cplx(
    C::new(1.3, -0.4),
    a.as_dyn_stride(), true,   // 对 A 取共轭
    b.as_dyn_stride(), false,  // B 保持原样
    C::new(0.5, 0.7),
    c.as_dyn_stride_mut(),
    Parallelism::Serial,
);
```

在 `complex` + `epilogue` 下有 `gemm_cplx_fused`，它一趟加上一个可选偏置：`C <- alpha*op(A)*op(B) + beta*C + bias`。偏置是 `Bias::PerRow`（长度 `A.rows`）或 `Bias::PerCol`（长度 `B.cols`），原样相加、绝不共轭。这里刻意没有激活参数：ReLU 这类基于序的激活在复数上无定义，所以融合复数入口只带偏置。

## 融合偏置与激活（`epilogue`）

在 `epilogue` 下，`gemm_fused` 一趟计算 `C <- act(alpha*A*B + beta*C + bias)`。可选的 `Bias` 是 `PerRow` 或 `PerCol`；可选的 `Activation` 是 `Relu` 或 `LeakyRelu(slope)`，最后应用。两者都传 `None` 就恰好是 `gemm`。两个选择子都由 crate 重新导出。

```rust
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_fused, Activation, Bias};

let bias: Vec<f64> = (0..m).map(|i| 0.5 * i as f64 - 2.0).collect();
// C <- relu(1.3 * A*B - 0.7 * C + 逐行偏置)
gemm_fused(
    1.3, a.as_dyn_stride(), b.as_dyn_stride(), -0.7,
    c.as_dyn_stride_mut(),
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::Rayon(0),
);
```

对 `f32`/`f64`，融合结果与先做普通 `gemm` 再做同样的标量映射逐位相同，对任何形状皆然，且对线程数确定，因为尾部运算折进同一个内核的写回而不扰动累加顺序。对 `f16`/`bf16`（在 `half` 下）它则更精确、而非相同：偏置与斜率精确地拓宽到 `f32`，尾部运算在 `f32` 中进行，之后才对窄输出做唯一一次取整，从而避免了单独窄映射会引入的二次取整。完整契约见[融合 Epilogue](../gemmkit-guide/融合Epilogue.md)指南。

对任意的逐元素函数则有 `gemm_map`（仅 `f32`/`f64`）：`C[r,c] <- f(alpha*A*B + beta*C, r, c)`，闭包在每个输出元素的最终值上恰好应用一次，`(r, c)` 处于 `C` 的用户坐标系。用它来做 GELU、sigmoid、夹取或与位置相关的变换；普通偏置或 ReLU 则优先用 `gemm_fused`，因为它会向量化，而 `gemm_map` 每个元素要付一次间接调用。

## 批量 GEMM

faer 没有三维数组类型，所以批量 GEMM 以切片表达：`gemm_batched` 取一个 `&[(MatRef, MatRef)]` 的逐元素 `(A, B)` 输入，与一个 `&mut [MatMut]` 的 `C` 输出按位配对，共享同一个 `alpha`、`beta` 和 `Parallelism`。批次跨元素并行——整个 GEMM 被指派给工作线程，各自串行、缓存热地运行——走 gemmkit 的指针数组引擎。

```rust
use faer::Mat;
use gemmkit::Parallelism;
use gemmkit_faer::gemm_batched;

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

只要每个元素自身的维度自洽，各元素的形状可以不同（异构批次）。若输入与输出的数量不一致，或任何元素的维度不自洽，调用会 panic 并点名出错的元素下标。每个元素都重新经全引擎分发，因此批次复现一次普通的 `gemm` 循环，对线程数确定，并且由于每个元素完整地在一个工作线程上运行，串行与并行之间还额外逐位相同。这里没有融合的批量入口：ndarray 适配器提供的共享尾部批量形式在核心里没有指针数组对应物。调度策略见[批量 GEMM](../gemmkit-guide/批量GEMM.md)。

## 预打包操作数

当一个操作数在多次调用间固定（权重对着一串激活值）时，把它预打包一次，就能省掉每次调用的重打包。`prepack_rhs` 把一个 `B` 变成可复用的 `PackedRhs`，由 `gemm_packed_b` 消费；`prepack_lhs` 把一个 `A` 变成 `PackedLhs`，由 `gemm_packed_a` 消费。两个句柄都由 crate 重新导出。

```rust
use gemmkit::Parallelism;
use gemmkit_faer::{gemm_packed_b, prepack_rhs};

let packed = prepack_rhs(weights.as_dyn_stride()); // 把固定的 B 打包一次
for (act, mut out) in stream {
    // out 必须偏列主序（|列步长| >= |行步长|）
    gemm_packed_b(1.0, act.as_dyn_stride(), &packed, 0.0, out.as_dyn_stride_mut(), Parallelism::Rayon(0));
}
```

唯一的约束是输出朝向。预打包的 `B` 固定了操作数的角色，所以 `gemm_packed_b` 需要一个偏列主序的 `C`（`|列步长| >= |行步长|`）；行主序的 `C` 会交换 `A`/`B` 并使打包好的 RHS 失效，gemmkit 会拒绝它。对称地，`gemm_packed_a` 需要一个偏行主序的 `C`。若输出布局不匹配，退回到普通 `gemm`。在 `epilogue` 下，预打包入口有融合孪生版本 `gemm_packed_b_fused` 与 `gemm_packed_a_fused`，在同一个句柄上接受与 `gemm_fused` 相同的 `Bias`/`Activation`。复用模型见[预打包操作数](../gemmkit-guide/预打包操作数.md)指南。

## 何时该动用这个适配器

faer 自带高性能 matmul，对于两个 faer 矩阵的普通 `f32`/`f64` 乘积，通常就该直接用它。当你需要核心 faer 算子没有提供的东西、又想在 faer 自己的类型上完成、且不离开这套生态时，这个适配器才有价值：

- **额外的元素类型族**：`i8 -> i32` 的整数 GEMM，以及一路融合到 `i8` 或 `u8` 输出的重量化。
- **融合尾部运算**：偏置与激活（或任意逐元素闭包）与乘积在同一趟里算完，而不是对 `C` 再扫一遍。
- **跨调用预打包**：把一个固定权重矩阵的打包成本摊到一段长推理循环上。
- **共享的调优面**：由于 gemmkit 三个适配器都坐在同一引擎上，[gemmkit-tune](../gemmkit-tune/使用gemmkit-tune调优.md) 产出的一份 `GEMMKIT_*` 环境配置对它们一致适用。旋钮面见[调优旋钮](../gemmkit-guide/调优旋钮.md)。

若以上都不适用，faer 内置的 matmul 是更简单的选择。这个适配器是它的补充，而非替代。
