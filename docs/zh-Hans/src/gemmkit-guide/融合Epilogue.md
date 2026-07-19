# 融合Epilogue

GEMM 很少独自出场。它的输出通常紧接着就喂进一个偏置加法、一个激活，或一步量化，之后才有别的东西去看它。若按朴素写法，这就是对 `C` 的第二遍完整扫描：GEMM 写出 `m*n` 个值，然后一个单独的循环把它们全部读回、变换、再写回去。融合 epilogue 把这个变换折进 GEMM 自己的存储里，于是每个输出元素在写出的那一刻就在寄存器里被变换掉，那次额外的内存扫描干脆消失了。本页所有内容都位于 `epilogue` 这个 Cargo feature 之后。

## 偏置与激活

[`gemm_fused`](https://docs.rs/gemmkit) 是向量化的主力：一趟完成 `C <- act(alpha*A*B + beta*C + bias)`。偏置是一个 [`Bias`](https://docs.rs/gemmkit) 枚举，要么 `Bias::PerRow(&[T])`（每个输出行一个值，长度 `m`），要么 `Bias::PerCol(&[T])`（每列一个，长度 `n`），在乘积之后加到该行或该列的每个元素上。激活是一个 [`Activation`](https://docs.rs/gemmkit)：`Relu`（`max(v, 0)`）或 `LeakyRelu(slope)`。两个参数都是 `Option`，`None`/`None` 直接委托给普通 `gemm`。

```rust
use gemmkit::{gemm_fused, Bias, Activation, MatRef, MatMut, Parallelism};

let bias = vec![0.0f32; m]; // 每个输出行一个值
gemm_fused(
    1.0,
    MatRef::from_row_major(&a, m, k),
    MatRef::from_col_major(&b, k, n),
    0.0,
    MatMut::from_col_major(&mut c, m, n),
    Some(Bias::PerRow(&bias)),
    Some(Activation::Relu),
    Parallelism::Rayon(0),
);
```

偏置、`LeakyRelu` 斜率和激活都在向量快路径上于寄存器内施加，所以融合相比裸 GEMM 几乎不花代价。

## 任意逐元素映射

当变换既不是偏置也不是标准激活时，[`gemm_map`](https://docs.rs/gemmkit) 接受一个闭包 `f(value, row, col) -> value`，把它施加到每个输出元素的最终值上，恰好一次，融合进存储里。它是 gemmkit 没有内置快路径的那些 epilogue 的通用扩展点——GELU、sigmoid、clamp，或任何与位置相关的变换：

```rust
use gemmkit::{gemm_map, MatRef, MatMut, Parallelism};

let f = |v: f32, _r: usize, _c: usize| v.tanh();
gemm_map(
    1.0,
    MatRef::from_row_major(&a, m, k),
    MatRef::from_col_major(&b, k, n),
    0.0,
    MatMut::from_col_major(&mut c, m, n),
    &f,
    Parallelism::Rayon(0),
);
```

交给闭包的 `(row, col)` 是 `C` 的用户坐标系；闭包可以按引用捕获它的环境（约束是 `+ Sync`，所以能安全地在并行 worker 间共享，比如借用一张查找表）。`gemm_map` 只支持 `f32`/`f64`。它用每个输出元素一次间接调用（相对每元素 `O(k)` 的工作量而言很便宜）换来完全的通用性；对于普通的偏置或激活，优先选会把变换向量化的 `gemm_fused`。

## 整数重量化

量化推理想要的恰是加宽 GEMM 的反面：`i8` 输入、`i32` 累加器，输出又回到 `i8`（或 `u8`），并在降位途中施加一个 scale 和一个 zero-point。[`gemm_i8_requant`](https://docs.rs/gemmkit) 与 [`gemm_i8_requant_u8`](https://docs.rs/gemmkit) 一趟做完整件事，省掉了 `gemm_i8` 再接一步单独重量化所要付的、对完整 `m*n` 个 `i32` 的物化。它们接受一个 [`Requantize`](https://docs.rs/gemmkit) 结构体：

```rust
use gemmkit::{gemm_i8_requant_u8, Requantize, RequantScale, MatRef, MatMut, Parallelism};

let req = Requantize {
    scale: RequantScale::PerRow(&per_channel_scales), // 长度 m，逐通道
    zero_point: 128,
    bias: Some(&i32_bias),                             // 可选的逐行 i32 偏置，长度 m
};
gemm_i8_requant_u8(
    MatRef::from_row_major(&activations, m, k),
    MatRef::from_col_major(&weights, k, n),
    req,
    MatMut::from_col_major(&mut out_u8, m, n),
    Parallelism::Rayon(0),
);
```

输出为 `C[i,j] = clamp(zero_point + round_ne(scale * (sum_k A*B + bias[i])), LO, HI)`，采用四舍六入五成双（round-half-to-even），其中 `scale` 要么是单个 `RequantScale::PerTensor(f32)`，要么是逐行的 `RequantScale::PerRow(&[f32])`（逐通道约定），钳位区间由入口决定：`gemm_i8_requant` 为 `[-128, 127]`，`u8` 孪生为 `[0, 255]`。没有 `alpha`（并入 `scale`），也没有 `beta`（往量化后的 `C` 里累加是没有良定义的）。这个重量化映射在每种 ISA（scalar、FMA、AVX-512、VNNI）上、以及向量与标量存储路径之间都是逐位精确的，所以答案绝不取决于跑了哪个内核。

## 复数偏置

在 `complex` feature 之下，`gemm_cplx_fused` 给复数乘积加上逐行或逐列偏置，`C <- alpha*op(A)*op(B) + beta*C + bias`，并带有与 `gemm_cplx` 相同的可选操作数共轭。它按设计只支持偏置：像 ReLU 这样基于序的激活在复数上没有定义。`conj_a` / `conj_b` 标志只共轭操作数；偏置是原样加上的，绝不共轭。

## 你可以依赖的保证

每个融合入口都把每个形状路由到普通 `gemm` 会选的**同一个**内核——通用 driver 或某条[特殊路径](小形状与GEMV.md)——并把 epilogue 融进那个内核的存储里，而不扰动它的累加次序。所以融合调用不是另一种算法；它就是同一个 GEMM，只是在存储时施加了映射。具体的保证是：

- 对 `f32`/`f64`，融合结果与普通 `gemm` 后接同一个标量映射**逐位一致**，对每个形状、每种布局、每个 worker 数都成立。`gemm_map` 对逐元素的 `f` 给出同样的保证，复数偏置入口对 `gemm_cplx`-再加偏置给出同样的保证。
- 对窄浮点 `f16`/`bf16`（`half` feature）有一个明确记录的例外。偏置和斜率被精确加宽到 `f32`，epilogue 在 `f32` 中施加，向输出的那一次四舍五入取偶的收窄只在存储时发生一次。这比 `gemm`-再映射*更*精确（后者会先舍入到窄类型、再加宽、再舍入一次），所以对窄类型，融合结果有意地**不**与两步式逐位相等。可复现性与确定性不受影响。
- 可复现性自始至终成立：串行与并行逐位一致，而恒等融合的情形（`None`/`None`，或没有偏置）会常量折叠回严格的普通 `gemm`。

回报就是你不再做的那趟 `C` 扫描。在一个内存受限的 epilogue 上，那第二趟扫描的代价可能不亚于存储本身，所以把偏置或激活融进 GEMM，在两步式并不便宜的场景下几乎是免费的。

融合 epilogue 也和其它 API 档次组合：`gemm_batched_fused` 对一次[批量 GEMM](批量GEMM.md) 的每个元素施加同一份共享的偏置和激活，而 `gemm_packed_b_fused` / `gemm_packed_a_fused` 在[预打包操作数](预打包操作数.md)之上融合。每个带检查入口都有裸指针的 `_unchecked` 孪生，供适配器与 FFI 使用，它们用 `(ptr, BiasDim)` 对来携带偏置，而非 `Bias` 枚举；见 [Unchecked 层](Unchecked层.md)。
