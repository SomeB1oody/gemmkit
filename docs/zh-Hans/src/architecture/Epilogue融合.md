# Epilogue融合

GEMM 的输出很少以原始形态离开例程。推理层要加偏置和激活，量化流水线要把 `i32` 累加器重量化成一个字节。朴素地做，这些操作每一个都是对 `C` 的第二次完整遍历：每个元素都要写入内存、被逐出、读回、变换、再写一次。

`epilogue` feature（`gemmkit/src/kernel/epilogue.rs`）改为把这个变换直接融合进 microkernel 的存储步骤。它在元素本就占据的寄存器（或暂存槽位）里完成变换，时机就是 microkernel 本来要存储它的那一刻，第二次遍历因此完全消失。对重量化而言，省下的开销还要更大：非融合流程必须先把整个 `m x n` 矩阵以 `i32` 物化出来，再收窄它。

## 接缝

这条接缝是 `Epilogue` trait。它经由 `KernelFamily::microkernel_epi` 穿入，让每个 family 的存储点都能应用它，而 driver 完全不需要知道它的存在：

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
    /// Vector transform of a whole MR_REG x NR register tile; defaults to a loop over
    /// apply_reg. Overridden to hoist a runtime discriminant out of the unrolled pass
    unsafe fn apply_tile<S, const MR_REG: usize, const NR: usize>(&self, simd: S, acc: ..., row0: usize, col0: usize) -> ...;
    /// Vector store-transform from Acc scratch to Out; same bit-agreement contract
    unsafe fn apply_store<S>(&self, simd: S, src: *const Fam::Acc, dst: *mut Fam::Out, ...);
}
```

整个设计靠两条不变量撑起来。第一条是**零开销恒等**。普通 `gemm` 传入的是 `Identity` epilogue，它的 `IS_IDENTITY = true` 让每个钩子都在编译期折叠掉。单态化出来的非融合内核，和这条接缝存在之前完全逐位一致。调用者不用融合时，融合不花一分钱。

第二条是**恰好触发一次**的语义。driver 把一个 `last_k` 标志交给 microkernel，epilogue 只在最后一个深度 panel 上应用。更早的 panel 存储的是原始的 `Acc` 部分和，和非融合内核完全一样。`OUT_IS_ACC = false` 的 family（比如窄类型 `f16`/`bf16` 的输出）从构造上就以单个 `kc = k` panel 跑完整个收缩，所以 `last_k` 在那里天然为真。[深 K 孪生](点积内核与深K孪生.md)会破坏这个单 panel 的保证，因此它在融合路径上刻意从不启用。特殊路径也天然只触发一次，因为它们的每个输出元素本身就是一次完整归约加一次存储。

## 内置 epilogue

随库发布三个 epilogue，各自有自己的公开入口（feature `epilogue`）。重量化入口还额外需要 `int8`。面向用户的视角见[融合 Epilogue](../gemmkit-guide/融合Epilogue.md)：

**`FusedEpi`** 是运行期组合出来的"偏置加激活"epilogue。它先加上按行或按列的偏置（`Bias::PerRow` / `Bias::PerCol`），再应用 `Relu` 或 `LeakyRelu(slope)`。一次单态化就覆盖了所有组合，所以融合内核的数量不会随 epilogue 种类的数目相乘。但这一点成立，仅仅是因为枚举分支*每个 tile 只解码一次*，在 `apply_tile` 的覆写里完成，而不是每个累加器解码一次。

这个区别不是什么微优化。tile 的 const 泛型把内核的存储遍展开。如果用逐寄存器的钩子，两个 `match` 都会在 tile 的每一个累加器槽位上被复制一遍。在一个宽 tile 上，由此形成的分支网会让编译器付出整个累加器 tile 的代价：它不再把 `acc` 留在寄存器里，而是从 `kc` 循环内部就把每个值写穿到栈上，而 epilogue 在那里根本不会执行。把解码提到循环外面、改成每个 tile 只做一次，就能完全避免这次溢出。`tests/perf/fused.rs` 钉住了融合速率与普通速率之间的比值，这样以后的改动就不会在无人察觉的情况下让这个问题重新出现。

`FusedEpi` 支撑着 `gemm_fused` 以及它的整个家族。这包括在整批之间共享同一份偏置和激活的 `gemm_batched_fused`、预打包孪生函数 `gemm_packed_b_fused` 和 `gemm_packed_a_fused`，以及复数入口 `gemm_cplx_fused`。复数入口只有偏置，因为基于大小比较的激活函数在复数上没有数学定义。`FusedEpi` 把 `VECTOR` 设为 `true`。在快路径上，偏置加法和激活都以寄存器操作的形式运行，比如 `max(v, 0)`。它在 SIMD `max`/`min` 上的 NaN 约定经过精心选择，使向量形态与标量形态严格一致：两者都算出 `ReLU(NaN) = 0`。

**`MapEpi`** 是逃生舱口。`gemm_map` 把一个任意的用户闭包 `f(value, row, col) -> value` 应用到每个输出元素的最终值上，`(row, col)` 使用用户坐标系。这个闭包是借用的 `&dyn Fn + Sync`，所以每个 `(类型, ISA)` 只需一次单态化，而不是每个闭包一次。它以标量方式运行，每个元素调用一次，由每个元素背后 `O(k)` 次浮点运算摊销这个开销。`MapEpi` 只支持 `f32`/`f64`。窄类型将不得不先舍入到 `N`、应用 `N` 域上的闭包、再舍入一次，这会破坏下文描述的逐位契约。

**`KRequantize`** 实现了量化推理的存储步骤：`C[r,c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`。它把 `i32` 累加器映射到 `i8`（`gemm_i8_requant`，值域 `[-128, 127]`）或 `u8`（`gemm_i8_requant_u8`，值域 `[0, 255]`，即 ONNX QLinearMatMul 的惯例）。scale 可以按张量取，也可以按行取（`RequantScale`）。zero point 在舍入步骤之后以整数形式并入。可选的 `i32` 偏置在唯一一次 `f64` 舍入步骤之前以整数形式并入。舍入本身是 round-half-to-even，通过一个 `no_std` 安全的 `2^52` 技巧实现（`round_ne_f64`）。`KRequantize` 没有 `alpha`，因为它已经折进了 scale。它也没有 `beta`，因为在一个已经量化过的 `C` 上继续累加没有清晰的含义。

## 正确性契约

这份契约之所以写得如此精确，正是因为 epilogue 的测试逐位钉住的就是它。三个要素合成了它：

1. **相同的路由。** 一次融合调用把每个形状都送进普通 `gemm` 会用的同一个内核。通用 driver、gemv、small-k、small-mn 各自都有对应的融合形态（见[特殊路径](特殊路径.md)），融合分发入口与普通的门一一镜像。没有哪个形状会仅仅因为要了个偏置就得多付 driver 的开销。唯一刻意的例外是混合 `f16`/`bf16` 的融合 gemv，出于特殊路径一章解释的舍入原因，它留在 driver 上。反正窄类型本来就在下文的逐位契约之外。
2. **与 epilogue 无关的引擎。** 分块、调度、打包和累加顺序都不依赖穿入的是哪个 epilogue，epilogue 只触碰存储这一步。
3. **逐位一致的两条应用路径。** 完整的列主序 tile 走向量路径存储，即 `apply_tile`（默认实现是 `apply_reg`），或者走 `apply_store`。边缘或跨步的 tile 则改为经暂存走标量 `apply`。同一个输出矩阵可以自由混用这两条路径。所以 trait 契约要求两者在同一个 token 下逐位一致。`apply_tile` 的覆写继承了这条义务，它必须逐元素留下与 `apply_reg` 完全相同的结果。

这三点合起来给出了头条保证：对 `f32`/`f64` 而言，`gemm_fused`、`gemm_map`，以及批量和预打包的融合入口，都等于 `gemm()` 后接同一个标量映射，**逐位相等，对每一种形状都成立**。

`MapEpi` 展示了这份保证有多刻意。它把 `VECTOR` 设为 `true`，不是为了向量化闭包（做不到），而是为了让内核走上与普通 `gemm` *完全相同的路径选择*。快路径融合的 `beta*C + alpha*AB` 存储，在一般的 `beta` 下，与标量路径的非融合算式相差 1 个 ULP。因此只走暂存的 epilogue，会把普通 `gemm` 实际上从未写出过的值交给闭包。`apply_reg` 转而把寄存器排空到一个栈缓冲区，再逐 lane 调用同一个标量 `apply`，这样 `f` 看到的永远是普通 `gemm` 产生的那些精确位。

文档化的例外是 `f16`/`bf16`。窄类型的 blanket 实现在唯一一次 round-to-nearest-even 收窄*之前*，就在 `f32` 累加器上应用了偏置和激活。这刻意比"`gemm()` 再映射"**更精确**，因为后者要先舍入到窄类型、再拓宽回来、再舍入一次。所以对窄类型而言，融合入口*并不*与"gemm 再映射"逐位相等，文档直接说明这一点，而不是削弱融合语义去迁就那个精度更差的替代方案。在同一次融合运行内部，向量路径和标量路径仍然逐位一致，因为两者都在 `f32` 中计算 `act(bias(v))`，并且都只舍入一次。跨 worker 数的可复现性也保持不变。

`KRequantize` 的向量路径是以另一种方式获得资格的。x86 token 实现了 `KernelSimd::requant_store`：向量化的拓宽到 `f64`、乘 scale、硬件 round-to-nearest-even、clamp，再写出低字节。它的文档带着逐情形的证明，说明每个 lane 都等于标量的 `clamp(zp + round_ne(scale*v), lo, hi)`。`i32 -> f64` 与 `f32 -> f64` 的拓宽都是精确的，`2^52` 技巧在 `2^52` 以下与硬件舍入一致，饱和行为在其上也一致，而 NaN 不可能出现，因为 API 已经校验过每个 scale 都是有限且为正的。按行取值的 scale 会逐 lane 变化，这种情况改走逐 lane 的标量映射。非 x86 token 保持 `REQUANT_VECTOR = false`，全程使用标量映射。一个模块内的一致性扫描，也就是 `gemmkit/src/simd.rs` 里的 `requant_store` 测试，会在每个具备这项能力的 token 上检查这种逐位相等。契约里说的"已证明"，因此是靠测试强制执行的，而不只是一个愿景。

最后还有一个角落。当 `A*B` 项消失时，也就是 `k == 0` 或 `alpha == 0`，融合入口仍然欠着一笔 `C <- act(beta*C + bias)`。这个退化映射在用户坐标系里逐元素运行。`gemmkit/src/dispatch/float.rs` 里的 `fused_degenerate` 处理这种情况，它还有一个窄类型的同胞版本，在 `f32` 中合成结果、只收窄一次。因此，即便是乘积项为空的情形，也遵守与完整内核相同的语义。
