# nalgebra适配器进阶用法

除了 [在 nalgebra 中使用 gemmkit](在nalgebra中使用gemmkit.md) 里讲的实数标量 `gemm`/`gemm_with`/`dot`，适配器还在 Cargo feature 之后暴露了引擎的完整接口：整数 GEMM、复数 GEMM、融合 epilogue、重量化输出、逐元素映射、预打包操作数以及批量。每个入口都恪守适配器的核心承诺——直接读取 nalgebra 的指针和步长、无拷贝——并镜像一个同名的 gemmkit 核心函数。每个类型族都被门控，你只为开启的部分付出代价。

| Feature | 增加 |
|---|---|
| `half` | 让 `f16`/`bf16` 走同一套 `gemm`/`gemm_fused` 泛型 |
| `int8` | `gemm_i8`、`gemm_i8_with`、`dot_i8`（`i8 -> i32`） |
| `complex` | `gemm_cplx`、`gemm_cplx_with`、`dot_cplx` |
| `epilogue` | `gemm_fused`、`gemm_map`（及预打包融合孪生） |
| `int8` + `epilogue` | `gemm_i8_requant`、`gemm_i8_requant_u8` |
| `complex` + `epilogue` | `gemm_cplx_fused` |

被重新导出的辅助类型（`Bias`、`Activation`、`RequantScale`、`Requantize`、`PackedLhs`、`PackedRhs`）来自适配器 crate 本身，所以你无需点名 `gemmkit` 即可使用它们。`Parallelism`、`Workspace` 和 `Complex` 仍来自 `gemmkit`。

## 整数 GEMM

在 `int8` 下，`gemm_i8` 把两个 `i8` 矩阵相乘成一个 `i32` 输出。输入是 `i8`；`alpha`、`beta` 和 `C` 是 `i32`，因为 `i8*i8` 的乘积需要更宽的累加器。算术在溢出时回绕，这是整数 GEMM 的惯例语义。它之所以是独立于 `gemm` 的入口，正是因为输入与输出的元素类型不同。

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{dot_i8, gemm_i8};
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 3, &[1_i8, 2, 3, 4, 5, 6]);
let b = DMatrix::from_row_slice(3, 2, &[1_i8, 0, 0, 1, 1, 1]);

// dot_i8：A*B 写进一块新的 DMatrix<i32>
let c = dot_i8(&a, &b);

// gemm_i8：缩放并累加进 i32 输出
let mut acc = DMatrix::<i32>::zeros(2, 2);
gemm_i8(1, &a, &b, 0, &mut acc, Parallelism::Serial);
assert_eq!(acc, c);
```

`dot_i8(a, b) -> DMatrix<i32>` 是做分配的便捷形式，而 `gemm_i8_with` 为定成本的量化推理循环复用调用方持有的 `Workspace`。

## 重量化输出

重量化把量化推理里“反量化—缩放—取整—夹取”这一步折进 GEMM，于是那个 `m*n` 的 `i32` 累加器从不必被完整物化。`gemm_i8_requant` 接收 `i8` 输入、写出 `i8` 输出；`gemm_i8_requant_u8` 写出无符号的 `u8` 输出（ONNX QLinearMatMul 的约定）。两者都需要 `int8` + `epilogue`。没有 `alpha`（它折进 scale）也没有 `beta`（往一个量化后的 `C` 上累加是没有良好定义的）。参数装在一个被重新导出的 `Requantize` 里：

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{RequantScale, Requantize, gemm_i8_requant};
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 3, &[10_i8, -4, 7, 3, 8, -2]);
let b = DMatrix::from_row_slice(3, 2, &[2_i8, 1, -1, 5, 4, 0]);
let bias = [100_i32, -50]; // 逐行，长度为 A.rows

let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,        // i8 输出取值域 [-128, 127]
    bias: Some(&bias),
};
let mut c = DMatrix::from_element(2, 2, 0_i8);
gemm_i8_requant(&a, &b, req, &mut c, Parallelism::Serial);
```

`RequantScale::PerTensor(s)` 对每个元素施加同一个 scale；`RequantScale::PerRow(&[f32])` 给每个输出行（每个输出通道，即标准的逐通道约定）一个 scale，长度为 `A.rows`。每个 scale 都必须有限且 `> 0`。`zero_point` 在取整之后以整数加入，且必须落在所选入口的输出域内：`gemm_i8_requant` 为 `[-128, 127]`，`gemm_i8_requant_u8` 为 `[0, 255]`。可选的逐行 `i32` 偏置，长度为 `A.rows`，在缩放之前加到累加器上。适配器会校验以上全部，并在任何违规时以核心引擎的措辞 panic（scale 非有限或非正、逐行 scale 或 bias 长度不对、`zero_point` 越界，或者 scale/bias 切片与 `C` 重叠）。

## 复数 GEMM

在 `complex` 下，`gemm_cplx` 对 `T = Complex<f32>` 或 `Complex<f64>` 计算 `C <- alpha*op(A)*op(B) + beta*C`，其中当置起 `conj_a` 标志时 `op(A) = conj(A)`，`conj_b` 同理。这两个共轭标志正是复数需要独立入口的原因：它们塞不进同质的实数标量签名。`dot_cplx(a, b)` 是不做共轭的 `A*B` 便捷形式；要做共轭乘积就直接用 `gemm_cplx`。

```rust
use gemmkit::{Complex, Parallelism};
use gemmkit_nalgebra::{dot_cplx, gemm_cplx};
use nalgebra::DMatrix;

type C = Complex<f64>;
let a = DMatrix::from_element(2, 2, C::new(1.0, 1.0));
let b = DMatrix::from_element(2, 2, C::new(0.0, -1.0));

// 普通乘积
let p = dot_cplx(&a, &b);

// 对 A 取共轭、B 不变，累加进已有的 C
let mut acc = DMatrix::from_element(2, 2, C::new(0.0, 0.0));
gemm_cplx(C::new(1.0, 0.0), &a, true, &b, false,
          C::new(0.0, 0.0), &mut acc, Parallelism::Serial);
```

## 带融合偏置的复数

`gemm_cplx_fused`（需要 `complex` + `epilogue`）在复数乘积的同一趟里加上偏置：`C <- alpha*op(A)*op(B) + beta*C + bias`。偏置是被重新导出的 `Bias`，要么 `Bias::PerRow`（长度 `A.rows`），要么 `Bias::PerCol`（长度 `B.cols`），并原样相加，绝不共轭。这里刻意没有激活参数：像 ReLU 这样带次序的激活在复数上没有定义。当 `bias == None` 时，该调用就是 `gemm_cplx`。

## 融合 epilogue

`epilogue` feature 增加了 `gemm_fused`，它在 `f32`/`f64` 上以单趟计算 `C <- act(alpha*A*B + beta*C + bias)`（开启 `half` 时也支持 `f16`/`bf16`，其 epilogue 在 `f32` 中求值、之后只做一次收窄存储）。可选的 `Bias` 是 `PerRow`（长度 `A.rows`）或 `PerCol`（长度 `B.cols`）；可选的 `Activation` 是 `Relu` 或 `LeakyRelu(slope)`，最后施加。两者都传 `None` 时，逐位等同于普通 `gemm`。

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{Activation, Bias, gemm_fused};
use nalgebra::DMatrix;

let a = DMatrix::<f32>::from_element(12, 9, 0.5);
let b = DMatrix::<f32>::from_element(9, 7, -0.25);
let bias: Vec<f32> = (0..12).map(|i| 0.5 * i as f32 - 2.0).collect();
let mut c = DMatrix::<f32>::zeros(12, 7);

gemm_fused(1.3, &a, &b, -0.7, &mut c,
           Some(Bias::PerRow(&bias)), Some(Activation::Relu), Parallelism::Serial);
```

融合这一趟不只是图方便：它省掉了对 `C` 的第二次扫描，以及分开做偏置加法和激活时要付的那趟内存往返。对 `f32`/`f64`，其结果与先跑 `gemm`、再逐元素施加同样的偏置和激活是逐位一致的，所以你可以放心采用而不改变数值结果。它背后的设计见 [融合 Epilogue](../gemmkit-guide/融合Epilogue.md)。

## 逐元素映射

对于套不进“偏置加激活”形状的 epilogue，`gemm_map` 把一个任意闭包施加到每个算完的输出元素上：`C[r, c] <- f(alpha*A*B + beta*C, r, c)`，每个元素恰好触发一次，其中 `(r, c)` 处于 `C` 的用户坐标系。`T` 只能是 `f32`/`f64`。闭包是 `&(dyn Fn(T, usize, usize) -> T + Sync)`，所以它必须是 `Sync` 才能并行运行，并且可以按引用捕获数据。

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::gemm_map;
use nalgebra::DMatrix;

let a = DMatrix::<f64>::from_element(8, 6, 0.3);
let b = DMatrix::<f64>::from_element(6, 5, 0.4);
let mut c = DMatrix::<f64>::zeros(8, 5);

// sigmoid，忽略位置
let sigmoid = |v: f64, _r: usize, _c: usize| 1.0 / (1.0 + (-v).exp());
gemm_map(1.0, &a, &b, 0.0, &mut c, &sigmoid, Parallelism::Serial);
```

对普通的偏置或 ReLU 优先用 `gemm_fused`，因为它会向量化；`gemm_map` 是通用的扩展点（GELU、sigmoid、夹取、依赖位置的变换），代价是每个输出元素一次间接调用。和融合入口一样，它在 `f32`/`f64` 上的结果与先跑 `gemm`、再施加同一映射逐位一致，而 `gemm_map_with` 复用 `Workspace`。

## 预打包操作数

当某个操作数在许多次乘法中固定不变——例如一个权重矩阵对着一串激活服务时——把它预打包一次就能免掉每次调用的重打包。`prepack_rhs(b) -> PackedRhs<T>` 打包右操作数；`gemm_packed_b` 随后以该句柄代替 `B`。`prepack_lhs`/`gemm_packed_a` 对固定的左操作数做镜像。

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::{gemm_packed_b, prepack_rhs};
use nalgebra::DMatrix;

let weights = DMatrix::<f32>::from_fn(64, 32, |i, j| 0.01 * (i as f32 - j as f32));
let packed = prepack_rhs(&weights); // 把固定的 B 打包一次

for step in 0..100 {
    let x = DMatrix::<f32>::from_element(16, 64, step as f32); // 一批激活
    let mut y = DMatrix::<f32>::zeros(16, 32);                 // 列主序输出
    gemm_packed_b(1.0, &x, &packed, 0.0, &mut y, Parallelism::default());
}
```

这里有一个朝向约束。`gemm_packed_b` 需要偏列主序的 `C`（`|列步长| >= |行步长|`）；行主序的 `C` 会迫使引擎在内部交换 `A` 与 `B`，从而使预打包的 RHS 失效，因此 gemmkit 会拒绝它。`gemm_packed_a` 则相反：它需要偏行主序的 `C`，并拒绝列主序的。对于朝向不对的 `C`，退回到普通 `gemm`。每个预打包入口都有做工作区复用的 `_with` 孪生；在 `epilogue` 下，融合孪生 `gemm_packed_b_fused` 和 `gemm_packed_a_fused` 在同一句柄上再加偏置和激活。`PackedRhs` 和 `PackedLhs` 暴露 `.rows()` 和 `.cols()`，方便你复核维度。底层的复用模型见 [预打包操作数](../gemmkit-guide/预打包操作数.md)。

## 批量 GEMM

nalgebra 没有三维数组类型，所以批量 GEMM 不像 ndarray 适配器那样接收一个三维张量。取而代之，`gemm_batched` 以一个逐元素 `(&A, &B)` 输入对的切片，搭配一个 `&mut C` 输出的切片，按位置配对，在一次调用里对每个元素运行 `C_e <- alpha*A_e*B_e + beta*C_e`，走 gemmkit 的指针数组引擎。`alpha`、`beta`、`par` 由整个批次共享。

```rust
use gemmkit::Parallelism;
use gemmkit_nalgebra::gemm_batched;
use nalgebra::DMatrix;

let a = DMatrix::from_row_slice(2, 2, &[1.0_f32, 2.0, 3.0, 4.0]);
let b = DMatrix::from_row_slice(2, 2, &[5.0_f32, 6.0, 7.0, 8.0]);
let mut c = vec![DMatrix::<f32>::zeros(2, 2), DMatrix::<f32>::zeros(2, 2)];

let ab = [(&a, &b), (&a, &b)];
gemm_batched(1.0, &ab, 0.0, &mut c, Parallelism::Serial);
assert_eq!(c[0], DMatrix::from_row_slice(2, 2, &[19.0, 22.0, 43.0, 50.0]));
```

各元素的形状可以不同（异构批次），只要每个元素自身的维度自洽——`A_e.cols == B_e.rows` 等等；共享的存储类型承载变化的运行期维度，所以用 `DMatrix` 或动态步长视图就能覆盖异构形状和混合布局。输入数量与输出数量必须相等（`ab.len() == c.len()`）；数量不符、或任一元素维度不符，都会 panic。并行是跨批次铺开的——整个 GEMM 被分派给各个 worker，每个都串行、缓存命中地跑完——正因如此，其结果重现了一个普通的 `gemm` 调用循环，跨线程数确定，且串行与并行逐位一致。由于 C 切片是单一存储类型，单个共享融合 epilogue 没有指针数组的对应物，所以不同于 ndarray 适配器，这里没有 `gemm_batched_fused`。批量模型的进一步讨论见 [批量 GEMM](../gemmkit-guide/批量GEMM.md)。

## 它与 nalgebra 自带乘法的分工

nalgebra 本来就会做矩阵乘法：`&a * &b`、`a.mul_to(&b, &mut c)` 以及其余的运算符接口，都返回类型规整的矩阵，并与它的常量泛型维度融为一体。对于一次普通的 `f32`/`f64` 乘积，尤其是小的静态矩阵，那些才是地道的选择，没有理由去动这个适配器。适配器的价值在于 nalgebra 的运算符所不提供的东西：引擎的运行时 SIMD 分发，它在运行期而非编译期挑选机器上可用的最佳指令集；`i8 -> i32` 与重量化的整数路径；单趟内的融合偏置/激活与逐元素 epilogue；对许多小问题的批量乘法；以及供固定权重跨调用复用的预打包操作数。当你的矩阵本就在 nalgebra 里、又需要以上任意一项时，适配器让你不离开 nalgebra 的类型、也不做拷贝，就拿到引擎的吞吐与功能。
