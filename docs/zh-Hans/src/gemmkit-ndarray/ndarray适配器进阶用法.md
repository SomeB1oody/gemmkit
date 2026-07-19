# ndarray适配器进阶用法

除了普通的实数乘积，适配器还镜像了 gemmkit 的全部 API：整数 GEMM、面向量化推理的重量化输出、带可选共轭的复数乘积、融合的偏置与激活、用户自定义的逐元素映射、在三维数组上的批量乘法，以及预打包操作数。每个类型族都由与之同名的 Cargo feature 门控，并且都保持[快速上手页](在ndarray中使用gemmkit.md)里那些普通入口的形态：直接从数组读步长、转发给 gemmkit、仅在维度不匹配时 panic（对 fused 入口还多一条：偏置切片与 `C` 重叠）。每个入口也都有一个携带调用方自有 `Workspace` 的 `_with` 孪生。

## 整数 GEMM（`int8`）

`gemm_i8` 把 `i8` 输入乘进一个 `i32` 累加器：`C(i32) <- alpha*A(i8)*B(i8) + beta*C`，其中 `alpha`、`beta` 和 `C` 都是 `i32`。它之所以是独立于 `gemm` 的入口，正是因为输入与输出的元素类型不同。算术在溢出时回绕，这是整数 GEMM 的惯例语义。`dot_i8` 是其便捷孪生，返回一个新建的 `Array2<i32>`。

```rust
use gemmkit::Parallelism;
use gemmkit_ndarray::{dot_i8, gemm_i8};
use ndarray::Array2;

let a = Array2::<i8>::zeros((16, 12));
let b = Array2::<i8>::zeros((12, 10));

// i8 输入，i32 累加器
let c: Array2<i32> = dot_i8(&a, &b);

// 带 i32 alpha/beta 的通用形式，累加进已有累加器
let mut acc = Array2::<i32>::zeros((16, 10));
gemm_i8(2, &a, &b, 1, &mut acc, Parallelism::Serial);
```

## 重量化输出（`int8` + `epilogue`）

量化推理很少想要原始的 `i32` 累加器，它要的是一个 8 位张量。`gemm_i8_requant` 把乘法和重量化融进一趟，直接把 `i32` 累加器折叠成 `i8` 输出，全程不物化完整的 `m*n` 中间结果。它没有 `alpha`（折进 scale）也没有 `beta`（累加进量化输出没有良好定义）。参数装在一个 `Requantize` 里：

```rust
use gemmkit::Parallelism;
use gemmkit_ndarray::{RequantScale, Requantize, gemm_i8_requant, gemm_i8_requant_u8};
use ndarray::Array2;

let a = Array2::<i8>::zeros((16, 12));
let b = Array2::<i8>::zeros((12, 10));

// i8 输出，范围 [-128, 127]，单个 per-tensor scale，per-row 偏置（长度 A.rows）
let bias: Vec<i32> = vec![0; 16];
let mut c = Array2::<i8>::zeros((16, 10));
let req = Requantize {
    scale: RequantScale::PerTensor(0.05),
    zero_point: -7,
    bias: Some(&bias),
};
gemm_i8_requant(&a, &b, req, &mut c, Parallelism::default());

// u8 输出，范围 [0, 255]，per-channel scale，无偏置
let scales: Vec<f32> = vec![0.02; 16]; // 每个输出行 / 通道一个
let mut cu = Array2::<u8>::zeros((16, 10));
gemm_i8_requant_u8(
    &a,
    &b,
    Requantize { scale: RequantScale::PerRow(&scales), zero_point: 128, bias: None },
    &mut cu,
    Parallelism::default(),
);
```

输出为 `clamp(zero_point + round_ne(scale * (accumulator + bias[i])), LO, HI)`，采用四舍六入五成双，其中 `scale` 是 per-tensor 的那个值或 per-row 的 `scale_i`。`u8` 变体就是 ONNX-QLinearMatMul 风格的激活：除了输出域 `[0, 255]` 和 `zero_point` 的取值范围外，与 `gemm_i8_requant` 完全相同。两者都会拒绝非有限或非正的 scale、长度不等于 `A.rows` 的 per-row scale 或偏置、与 `C` 重叠的切片，以及超出该入口取值域的 `zero_point`。

## 复数 GEMM（`complex`）

复数乘积有自己的入口，因为那两个共轭标志放不进同构的实数签名。`gemm_cplx` 计算 `C <- alpha*op(A)*op(B) + beta*C`，其中设置 `conj_a` 时 `op(A)` 为 `conj(A)`，设置 `conj_b` 时 `op(B)` 为 `conj(B)`，元素类型为 `Complex<f32>` 或 `Complex<f64>`。`dot_cplx` 是不做共轭的便捷入口。

```rust
use gemmkit::{Complex, Parallelism};
use gemmkit_ndarray::{dot_cplx, gemm_cplx};
use ndarray::Array2;

type C = Complex<f64>;
let a = Array2::<C>::from_elem((8, 6), Complex::new(0.0, 0.0));
let b = Array2::<C>::from_elem((6, 5), Complex::new(0.0, 0.0));

// 普通 A*B
let c = dot_cplx(&a, &b);

// 对 A 取共轭，累加进已有 C
let mut acc = Array2::<C>::from_elem((8, 5), Complex::new(0.0, 0.0));
gemm_cplx(
    Complex::new(1.0, 0.0),
    &a,
    true,  // conj_a
    &b,
    false, // conj_b
    Complex::new(0.0, 0.0),
    &mut acc,
    Parallelism::Serial,
);
```

同时开启 `complex` 与 `epilogue` 后，`gemm_cplx_fused` 在同一趟里加上一个可选的 `Bias`（原样相加，绝不共轭）。它不接受激活参数：像 ReLU 这样带序关系的激活在复数上没有定义。

## 融合偏置、激活与映射（`epilogue`）

`gemm_fused` 一趟算出 `C <- act(alpha*A*B + beta*C + bias)`。偏置是可选的 `Bias::PerRow`（长度 `A.rows`）或 `Bias::PerCol`（长度 `B.cols`）；激活是可选的 `Relu` 或 `LeakyRelu(slope)`，最后施加。两者都为 `None` 时，它就是 `gemm`。

```rust
use gemmkit::Parallelism;
use gemmkit_ndarray::{Activation, Bias, gemm_fused, gemm_map};
use ndarray::Array2;

let a = Array2::<f32>::zeros((12, 9));
let b = Array2::<f32>::zeros((9, 7));

// 一趟算出 C <- ReLU(A*B + bias)；PerRow 偏置长度为 A.rows
let bias: Vec<f32> = vec![0.0; 12];
let mut c = Array2::<f32>::zeros((12, 7));
gemm_fused(
    1.0, &a, &b, 0.0, &mut c,
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::default(),
);

// 任意逐元素闭包 f(value, row, col)；这里是一个 relu6
let f = |v: f32, _r: usize, _c: usize| v.max(0.0).min(6.0);
let mut c2 = Array2::<f32>::zeros((12, 7));
gemm_map(1.0, &a, &b, 0.0, &mut c2, &f, Parallelism::default());
```

`Bias` 和 `Activation` 由 `gemmkit_ndarray` 重新导出，所以你无需为它们再点名 `gemmkit`。对 `f32`/`f64`，`gemm_fused` 对任意形状都与“先 `gemm` 再做同样的标量映射”逐位相同；对 `f16`/`bf16`，尾部运算在单次窄化之前以 `f32` 进行，这比单独的窄化映射*更*精确，因此对窄类型而言不与“先 `gemm` 再映射”逐位相等。`gemm_map` 是通用的逃生口：闭包 `f(value, row, col)` 看到的是每个输出元素的最终值，`(row, col)` 处在 `C` 的用户坐标系里，每个元素恰好触发一次。它对每个元素要付一次间接调用，所以普通的偏置或激活优先用 `gemm_fused`（它会向量化），而把 `gemm_map` 留给 GELU、sigmoid、clamp 或依赖位置的变换。这里的 `T` 只能是 `f32`/`f64`。

## 批量 GEMM

这是唯一没有普通 `gemm` 对应、也在同类适配器里没有对手的运算：一叠彼此独立的乘积，承载在三维 `Array3` 上，批次维在 0 轴。`a` 是 `(batch, m, k)`，`b` 是 `(batch, k, n)`，`c` 是 `(batch, m, n)`；0 轴是各操作数的批次步长，1、2 轴是元素步长。它在批次上并行，每个元素在一个 worker 上串行运行，因此结果精确复现一个 `gemm` 调用循环。

```rust
use gemmkit::Parallelism;
use gemmkit_ndarray::{dot_batched, gemm_batched};
use ndarray::Array3;

let a = Array3::<f32>::zeros((32, 8, 5)); // (batch, m, k)
let b = Array3::<f32>::zeros((32, 5, 6)); // (batch, k, n)

// 一叠乘积
let c = dot_batched(&a, &b); // (32, 8, 6)

// 累加进已有累加器的通用形式
let mut acc = Array3::<f32>::zeros((32, 8, 6));
gemm_batched(0.7, &a, &b, 1.3, &mut acc, Parallelism::default());
```

由于只读步长，一个换轴的、或本就一般步长的三维视图可以无拷贝转发：`a.view().permuted_axes([0, 2, 1])` 把一块 `(batch, k, m)` 缓冲区变成 `(batch, m, k)` 视图并直接批量转发。在 `epilogue` 下，`gemm_batched_fused` 对这叠里的每个元素施加同一个共享的 `Bias`/`Activation`，即批量线性层的情形；偏置按单个元素定尺寸（`PerRow` 长度 `m`，`PerCol` 长度 `n`），而非整批。

## 预打包操作数

当一个操作数固定、另一个成流而来时，把固定那一侧打包一次并复用，就省掉了每次调用的重复打包。`prepack_rhs` 为复用的 `B` 返回一个 `PackedRhs<T>`，由 `gemm_packed_b` 消费；`prepack_lhs` 为复用的 `A` 返回一个 `PackedLhs<T>`，由 `gemm_packed_a` 消费。打包函数直接读步长，所以 `B` 或 `A` 可以是任意布局。各自有一条朝向约束：`gemm_packed_b` 需要一个偏列主序的 `C`（`|列步长| >= |行步长|`），`gemm_packed_a` 需要一个偏行主序的 `C`（`|列步长| <= |行步长|`）；另一种朝向会交换操作数并使打包句柄失效，gemmkit 会拒绝。不合适的布局请用普通 `gemm`。

融合孪生 `gemm_packed_b_fused` 和 `gemm_packed_a_fused` 接受同样的句柄，再加上偏置和激活，这正是固定权重的推理层。下面把一个权重矩阵作为 LHS 打包一次并在各推理步之间复用，融进一个 per-output-channel 的偏置和一个 ReLU：

```rust
use gemmkit::Parallelism;
use gemmkit_ndarray::{Activation, Bias, gemm_packed_a_fused, prepack_lhs};
use ndarray::Array2;

let (out, in_features) = (256usize, 512usize);

// 把固定权重 W: (out, in) 打包一次
let w = Array2::<f32>::zeros((out, in_features));
let packed = prepack_lhs(&w);
let bias: Vec<f32> = vec![0.0; out]; // per-output-channel，长度 C.rows

// 每个推理步：激活 x (in, batch) -> y (out, batch)
let batch = 32;
let x = Array2::<f32>::zeros((in_features, batch));
let mut y = Array2::<f32>::zeros((out, batch)); // 行主序（packed_a 朝向）
gemm_packed_a_fused(
    1.0,
    &packed,
    &x,
    0.0,
    &mut y,
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::default(),
);
```

把打包句柄与 `_with` 工作区变体（`gemm_packed_a_fused_with`）搭配，一个稳定的推理循环在首次调用后就不再分配。偏置的轴以用户坐标系给出；`gemm_packed_b_fused` 不翻转地转发它，而 `gemm_packed_a_fused` 让核心去翻转，所以无论打包的是哪个操作数，`PerRow` 始终表示“每个输出行一个值”。

## 本适配器与 ndarray 自带乘积的取舍

`ndarray` 本就能做矩阵乘法：`.dot()` 给出普通乘积，`general_mat_mul` 给出就地的 `alpha`/`beta` 形式。对一次没有额外需求的 `f32`/`f64` 乘积，它们完全够用，还少拉一个依赖，所以没必要出于习惯就走 gemmkit。

当你需要 `ndarray` 内建路径给不了的东西时，再选本适配器。gemmkit 会在真正运行的机器上、于运行时挑选最快的指令集，而不是在编译期就把某个选择写死（见[运行时 ISA 分发](../gemmkit-guide/运行时ISA分发.md)）。它带来本页走过的更宽 API：融合偏置与激活、`i8` 与重量化推理、带共轭的复数、批量乘积以及预打包。它还暴露调优旋钮，外加一个把分块校准到部署机器的安装期自动调优器（见[调优旋钮](../gemmkit-guide/调优旋钮.md)）。这些都不改变你传入的数组或拿回的结果，因为适配器自始至终是同一套零拷贝步长转接；它只是拓宽了你能提出的请求。
