# Epilogue融合

GEMM 的输出很少以原始形态离开例程：推理层要加偏置和激活，量化流水线要把 `i32` 累加器重量化成一个字节。朴素做法里，这些都是对 `C` 的第二次完整遍历——每个元素写入内存、被逐出、读回、变换、再写一次。`epilogue` feature（`gemmkit/src/kernel/epilogue.rs`）改为把变换融合进 microkernel 的存储：元素在它本就占据的寄存器（或暂存槽）里、在它本就要被存储的那一刻完成变换，第二次遍历就此消失。对重量化收益还要更大——非融合流程必须先把整个 `m x n` 矩阵以 `i32` 物化出来再收窄。

## 接缝

接缝是 `Epilogue` trait，经由 `KernelFamily::microkernel_epi` 穿进每个 family 的存储点，driver 对此一无所知：

```rust
// gemmkit/src/kernel/epilogue.rs (trimmed)
pub trait Epilogue<Fam: KernelFamily>: Copy + Send + Sync {
    /// true => every hook const-folds away; the kernel is bit-identical to non-fused
    const IS_IDENTITY: bool = false;
    /// true => apply_reg is implemented, enabling the fast vector store path
    const VECTOR: bool = false;
    /// true => apply_store is implemented (the Out != Acc requantize pattern)
    const VECTOR_STORE: bool = false;

    /// Scalar transform at absolute (row, col) in the oriented problem frame
    unsafe fn apply(&self, v: Fam::Acc, row: usize, col: usize) -> Fam::Out;
    /// Vector transform of LANES consecutive rows; MUST agree with apply bit-for-bit
    unsafe fn apply_reg<S>(&self, simd: S, v: ..., row: usize, col: usize) -> ...;
    /// Vector store-transform from Acc scratch to Out; same bit-agreement contract
    unsafe fn apply_store<S>(&self, simd: S, src: *const Fam::Acc, dst: *mut Fam::Out, ...);
}
```

两条不变量撑起整个设计。第一，**零开销恒等**：普通 `gemm` 传入 `Identity` epilogue，其 `IS_IDENTITY = true` 让每个钩子在编译期折叠掉，单态化出的非融合内核与接缝存在之前逐位一致——不用融合时，融合不花一分钱。第二，**恰好触发一次**：driver 把 `last_k` 标志交给 microkernel，epilogue 只在最后一个深度 panel 上应用；之前的 panel 存储原始的 `Acc` 部分和，与非融合内核完全一样。`OUT_IS_ACC = false` 的 family（窄 `f16`/`bf16` 输出）从构造上就以单个 `kc = k` panel 跑完整个收缩，`last_k` 在那里结构性为真——而会破坏这一单 panel 保证的[深 K 孪生](点积内核与深K孪生.md)在融合路径上刻意从不启用。特殊路径天然只触发一次：它们的每个输出元素都是一次完整归约加一次存储。

## 内置 epilogue

随库发布三个 epilogue，各有自己的公开入口（feature `epilogue`，其中重量化入口还需同时开启 `int8`；面向用户的视角见[融合 Epilogue](../gemmkit-guide/融合Epilogue.md)）：

**`FusedEpi`** 是运行期组合的“偏置 + 激活”epilogue：先加按行或按列的偏置（`Bias::PerRow` / `Bias::PerCol`），再做 `Relu` 或 `LeakyRelu(slope)`。一次单态化覆盖所有组合——枚举分支只是每个 tile 上几次可预测的判断，被 `mr*nr*kc` 的 FMA 循环摊薄——所以融合内核的数量不会随 epilogue 种类相乘。它支撑 `gemm_fused` 及其整个家族：`gemm_batched_fused`（整批共享一份偏置和激活）、预打包孪生 `gemm_packed_b_fused` / `gemm_packed_a_fused`，以及复数入口 `gemm_cplx_fused`——后者只有偏置，因为基于大小比较的激活在复数上没有数学定义。它设 `VECTOR = true`：快路径上偏置加法与激活以寄存器操作运行（`max(v, 0)` 等），SIMD `max`/`min` 的 NaN 契约经过挑选，使向量与标量形态严格一致（两边都是 `ReLU(NaN) = 0`）。

**`MapEpi`** 是逃生门：`gemm_map` 把任意用户闭包 `f(value, row, col) -> value` 应用到每个输出元素的最终值上，`(row, col)` 位于用户坐标系。闭包是借用的 `&dyn Fn + Sync`——每个 `(类型, ISA)` 一次单态化，而不是每个闭包一次——以标量方式每元素调用一次，由每个元素背后 `O(k)` 的浮点运算摊销。仅限 `f32`/`f64`：窄类型将不得不先舍入到 `N`、应用 `N` 域闭包、再舍入一次，破坏下述逐位契约。

**`KRequantize`** 实现量化推理的存储：`C[r,c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`，从 `i32` 累加器到 `i8`（`gemm_i8_requant`，值域 `[-128, 127]`）或 `u8`（`gemm_i8_requant_u8`，值域 `[0, 255]`，即 ONNX QLinearMatMul 惯例）。scale 可以按张量或按行（`RequantScale`），zero point 在舍入之后以整数并入，可选的 `i32` 偏置在唯一一次 `f64` 舍入之前以整数并入，舍入是 round-half-to-even，用 `no_std` 安全的 `2^52` 技巧实现（`round_ne_f64`）。没有 `alpha`（折进 scale）也没有 `beta`（向已量化的 `C` 累加没有良定义）。

## 正确性契约

契约之所以写得这么精确，是因为 epilogue 测试逐位钉住的就是它。三个要素合成：

1. **相同的路由。** 融合调用把每个形状送进普通 `gemm` 会用的同一个内核：通用 driver、gemv、small-k、small-mn 各有融合形态（见[特殊路径](特殊路径.md)），融合分发入口与普通的门一一镜像。没有哪个形状因为要了偏置就得付 driver 的开销。（唯一刻意的路由例外是混合 `f16`/`bf16` 的融合 gemv：出于特殊路径一页解释的舍入原因，它留在 driver 上——窄类型本来就在下述逐位契约之外。）
2. **与 epilogue 无关的引擎。** 分块、调度、打包与累加顺序都不依赖穿入的是哪个 epilogue；epilogue 只碰存储。
3. **逐位一致的两条应用路径。** 完整的列主序 tile 走向量路径（`apply_reg` / `apply_store`）；边缘或跨步 tile 经暂存走标量 `apply`；同一个输出矩阵自由混用两者——所以 trait 契约要求同一 token 下两者逐位一致。

三者合起来给出头条保证：对 `f32`/`f64`，`gemm_fused`（以及 `gemm_map`、批量与预打包的融合入口）等于 `gemm()` 再接同一个标量映射，**逐位一致，对每一种形状成立**。`MapEpi` 展示了这有多刻意：它设 `VECTOR = true` 不是为了向量化闭包（做不到），而是让内核做出与普通 `gemm` *相同的路径选择*——快路径融合的 `beta*C + alpha*AB` 存储在一般 `beta` 下与标量路径的非融合算式差一个 ULP，只走暂存的 epilogue 会把普通 `gemm` 从未写出过的值交给闭包。`apply_reg` 于是把寄存器排空到栈缓冲、逐 lane 调用同一个标量 `apply`，`f` 看到的永远是普通 `gemm` 的精确位。

文档化的例外是 `f16`/`bf16`：窄类型的 blanket 实现在唯一一次 round-to-nearest-even 收窄*之前*、在 `f32` 累加器上应用偏置和激活。这刻意**比** `gemm()` 再映射**更精确**——后者要先舍入到窄类型、拓宽回来、再舍入一次——所以窄类型的融合入口*不*与 gemm-再-映射逐位相等，文档直说这一点而不是削弱语义去凑。融合运行内部，向量与标量路径仍逐位一致（都在 `f32` 中算 `act(bias(v))`、恰好舍入一次），跨 worker 数的可复现性不变。

`KRequantize` 的向量路径以另一种方式挣得资格。x86 token 实现 `KernelSimd::requant_store`——向量化的拓宽到 `f64`、乘 scale、硬件 round-to-nearest-even、clamp、写低字节——其文档带着逐情形证明：每个 lane 等于标量的 `clamp(zp + round_ne(scale*v), lo, hi)`——`i32 -> f64` 与 `f32 -> f64` 拓宽精确、`2^52` 技巧在 `2^52` 以下与硬件舍入重合、以上时饱和行为一致、NaN 不可能出现（API 校验 scale 有限且为正）。按行 scale 逐 lane 变化，改走逐 lane 标量映射；非 x86 token 保持 `REQUANT_VECTOR = false`，全程标量映射。模块内的一致性扫描（`gemmkit/src/simd.rs` 中的 `requant_store` 测试）在每个具备能力的 token 上检查这一逐位相等，契约里的“已证明”是被强制执行的，不是愿景。

最后一个角落：当 `A*B` 项消失（`k == 0` 或 `alpha == 0`）时，融合入口仍欠一笔 `C <- act(beta*C + bias)`。这个退化映射在用户坐标系里逐元素运行（`gemmkit/src/dispatch/float.rs` 的 `fused_degenerate`，窄类型的同胞在 `f32` 中合成并只收窄一次），因此连乘积为空的情形也遵守与完整内核相同的语义。
