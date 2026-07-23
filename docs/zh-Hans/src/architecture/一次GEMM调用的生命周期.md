# 一次GEMM调用的生命周期

上一页描述的是静止的栈，这一页跟随一次调用穿过它。样本是 crate 文档里的快速上手示例：

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

// 2x3 乘 3x2 = 2x2，全部行主序
let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
let mut c = [0.0_f32; 4];
gemm(
    1.0,
    MatRef::from_row_major(&a, 2, 3),
    MatRef::from_row_major(&b, 3, 2),
    0.0,
    MatMut::from_row_major(&mut c, 2, 2),
    Parallelism::Serial,
);
assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
```

这个玩具形状会在下文的某个早退口离开主路，所以整个走读同时记住两个问题：上面的 2x2x3，以及一台 AVX-512 机器上一路走到底的 2048x2048x2048 `f32` 乘积。路线压缩成一张图：

```
gemm(alpha, A, B, beta, C, par)
  |  validate_gemm_views：形状、边界、别名          [api.rs]
  v
Task<T>：裸指针 + isize 步长
  |  m == 0 || n == 0        -> 直接返回             [dispatch.rs]
  |  k == 0 || alpha == 0    -> C <- beta*C，结束
  v
记忆化的按类型内核（OnceLock 函数指针）
  |  gemv 形状 (m==1||n==1)  -> special/gemv.rs      [dispatch/float.rs]
  |  方向归一化：行主序倾向的 C -> 计算 C^T = B^T*A^T
  |  小 m,n 且长 k           -> special/small_mn.rs
  |  k <= small_k_threshold  -> special/small_k.rs
  v
driver::run                                          [driver.rs]
  jc 遍历 NC -> pc 遍历 KC（永不并行）
    -> 扁平作业列表（ic 行块 x jt 列 tile），
       工作线程从共享 JobCursor 拉取，自适应打包 A/B
  v
Fam::microkernel_epi：寄存器中的 MR x NR tile        [kernel/float.rs]
  alpha/beta epilogue 存储（向量快路径 | scratch 中转）
```

## 第一站：校验与降解

`gemm` 本身只有一行：借出线程本地工作区，转发给 `gemm_with`（`gemmkit/src/api.rs`）。`gemm_with` 运行 `validate_gemm_views`——即[设计目标](设计目标与总体图景.md)里的完整 panic 清单：形状一致、每个视图不越出自己的切片、`C` 对每个 `(i, j)` 寻址唯一、`C` 与两个输入都不重叠。然后视图就消解了。这一点以下的所有代码只讲 `Task<T>`：一个 `Copy` 结构体，装着 `m, k, n`、`alpha`/`beta`，以及三个带 `isize` 行/列步长的裸指针。转置从来不是一个标志位——转置视图不过是交换了步长——而当 `beta == 0` 时契约规定 `C` 永远不被读取，所以它可以是未初始化的。`unsafe` 边界恰好在这里跨越，由刚刚跑完的校验背书；`gemm_unchecked` 则晚一步进场，由调用方自己扛起这份背书。

## 第二站：分发层的早退口

`dispatch::execute`（`gemmkit/src/dispatch.rs`）趁元素类型还是具体的，先处理掉退化的代数情形：

```rust
if task.m == 0 || task.n == 0 {
    return;
}
// k == 0 or alpha == 0 => the A*B term vanishes: C <- beta*C only
if task.k == 0 || task.alpha == T::ZERO {
    T::scale_c(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
    return;
}
T::dispatch(task, par, ws);
```

输出为空意味着无事可做。`A*B` 项消失（`k == 0` 或 `alpha == 0`）时，调用退化成一次从不读取 `A`/`B` 的 `C <- beta*C` 缩放——其中 `beta == 0` 又直接写零而不读 `C`，兑现未初始化 C 的契约。只有真正的乘积才会到达 `T::dispatch`，它读取该类型的 `OnceLock` 槽：首次使用时选择阶梯探测 CPU 特性（尊重 `GEMMKIT_REQUIRE_ISA` 的钉选，不满足时 panic 而非回退），缓存胜出的单态化入口和 tile 几何；此后每次调用都只是一次间接调用。在那台 AVX-512 机器上，`f32` 解析为 `run_typed::<f32, Avx512F, 2, 12>`——32x12 的 tile。

## 第三站：`run_typed` 里的路由

`run_typed`（`gemmkit/src/dispatch/float.rs`）是一串简短的闸门，每道闸门都把寄存器分块驱动伺候不好的形状改道送走。

先是 gemv：若 `m == 1 || n == 1`（且该路径未被 `GEMMKIT_GEMV_THRESHOLD` 封顶关闭），调用立即去 `special/gemv.rs`——注意是在方向归一化**之前**、在用户原始坐标系里，因为 gemv 自己解决方向问题：`m == 1` 的情形被视作转置后的 `rows x k` 乘 `k` 向量问题，输出行的划分由它自己完成。

其余的都经过 `orient_transpose` 做方向归一化：若 `C` 是行主序倾向（`|csc| < |rsc|`），分发层把问题改写成它的转置——`C^T = B^T * A^T`，交换 `m` 与 `n`、`A`/`B` 的指针与步长、`rsc` 与 `csc`。这个恒等式是免费的（数据一字节不动，只改描述符），换来一条强不变量：此后输出的**行**步长是小的那个（对完全连续的 C 即 `rsc == 1`），输出的每一列在内存中连续，内核得以沿连续的列往下走。微内核的快速存储路径恰恰要求这一点（`rsc == 1`——用向量存储写一列中 `LANES` 个连续行），而下面的每一层都只需为一种方向优化而不是两种。我们那个全行主序的 2048 立方就命中了这次交换：引擎实际算的是 `C^T`，而分发层以下无人知晓。

然后是归一化任务上的另外两道闸门。小 `m,n` 形状（两个维度都不超过 `small_mn_dim`，收缩长度超过 `small_k_threshold`）去 `special/small_mn.rs`，每个输出元素是一次水平 SIMD 点积——两个操作数都沿 `k` 单位步长流动时零拷贝直读；有一个带步长时（`k > small_mn_pack_min_k`）走打包档，只拷贝不合格的那个操作数。小 `k` 形状（`k <= small_k_threshold`，默认 x86 为 16、aarch64 为 8）去 `special/small_k.rs`，把整个乘积当作一个就地读取的深度面板直接过微内核，没有任何分块和打包开销。闯过全部闸门的——我们的 2048 立方就是——带着驱动声明的前置条件进入 `driver::run`：`m, n, k > 0`、`alpha != 0`、方向已归一化。

## 第四站：驱动的循环嵌套

`driver::run` 携带零成本的 `Identity` epilogue 转发给 `run_inner`（`gemmkit/src/driver.rs`）——融合入口带着真 epilogue 落进同一个函数。驱动对家族和 ISA 令牌泛型；对我们这次调用是 `FloatGemm<f32>` 与 `Avx512F`，`mr = MR_REG * LANES = 32`，`nr = 12`。

先算分块：`cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k)` 按 BLIS 缓存模型给出 `(MC, KC, NC)`，尺寸以**打包输入**元素计（`sizeof(Lhs)` 而非累加器——窄类型因此得到更深的块）。随后循环嵌套按 BLIS 顺序展开：

- **`jc` 遍历 `NC`**——列块，尺寸保证打包后的 B 宏面板驻留 L3。
- **`pc` 遍历 `KC`**——深度切片，这层循环**永不并行**。所有深度切片都累加进同一批 C tile，把深度并行化就意味着对 C 做同步的读-改-写或者拆分归约；让深度保持串行，才使每个输出元素由同一个工作线程从头归约到尾——这是可复现性契约的一半。`beta` 只在第一个切片（`pc == 0`）参与；之后的切片以等效 beta 为一累加。混合精度家族（`OUT_IS_ACC = false`）恰好只有一个切片——`kc = k`——运行中的部分和永远不会经过窄输出类型的舍入。
- **一份扁平的一维作业列表**——每个深度切片内，剩下的工作是 `n_mc` 个行块乘 `n_nt` 个列 tile，压平成 `n_jobs = n_mc * n_nt` 个下标。工作线程按需从共享的无锁 `JobCursor` 拉取连续区块：没有静态划分，快核自然吸收更多工作，区块粒度对线程数过采样（`job_grain`；打包 LHS 路径用与行块对齐的 `packed_block_grain`，区块永不跨越打包边界）。线程数本身来自 `par.resolve(m*n*k, n_jobs)`——以工作量为准，随总 flops（`m*n*k` 除以每 worker 下限）扩展，而不是一步跳到全部核心。若这个线程数会让作业列表浅到每个 worker 分不到几个块，驱动器会先缩小 `mc`——这只会切出更多、更小的行块，因而不会移动任何结果比特——把列表加深，再交给游标发放。

打包是自适应的，两侧各自决策。**B** 在 `m` 越过 `rhs_pack_threshold` 时每个深度切片打包一次——打包面板被全部 `n_mc` 个行块复用，只有复用足够高拷贝才划算；否则 B 按原始步长就地读取。真要打包时打包本身也是并行的：工作线程从游标拉取 `nr` 宽的列面板，`for_each_worker` 的汇合就是先写后读的屏障，因为打包后的 B 是所有计算线程共享的唯一缓冲。**A** 有三种模式：每个工作线程把手头的行块打包进自己的私有工作区区域（`rsa != 1` 或行块不是 `mr` 的整倍数时强制；否则由每线程列复用量或 TLB 不友好的列步长决定是否值得）；大型并行问题上，一个共享预打包把每个行块恰好打包一次到按块分配的区域，靠自己的屏障同步（`shared_lhs_mnk` 门限），消除各线程的重复打包；复用低到任何拷贝都摊不平时，A 就地读取。这些区域的尺寸经由 `Workspace::regions` 预先切好，带着前文说过的“失败即封闭”溢出检查；完全不打包的路线甚至不会碰工作区。

对每个作业，工作线程解析出自己的 A 面板（已打包或就地）、定位 B 面板（本次调用打包的、预打包缓冲里的、或就地的），然后对块内每条 `mr` 行的条带调用微内核——全程在 `simd.vectorize` 之内，整个条带都在 target-feature 代码生成上下文中执行。

## 第五站：微内核与它的存储

`Fam::microkernel_epi`（`gemmkit/src/kernel/float.rs` 的 `microkernel_impl`）计算一个 `MR x NR` 的 tile——对我们这次调用是 32x12 个 `f32`，以 `[[Reg; MR_REG]; NR]` 数组的形式驻留在 24 个 ZMM 累加寄存器里。满宽 tile 走 `SimdOps::accumulate_tile`，即升序 `k` 的融合乘加调度（这条接缝允许像 NEON 这样受载入约束的 ISA 换上软件流水的变体——只重排载入，绝不重排算术）；列方向的边缘 tile 走运行时定界的循环，恰好读 `nr_eff` 列，保证未打包的 B 永远不会被读过最后一个真实列。然后 `alpha` 折进累加器——`alpha == 1` 时整段跳过，靠的是驱动预先算好的 `AlphaStatus`。

存储是 `beta` 和 epilogue 的居所，有两条路。快路径在满 tile 且输出行步长为一时触发（`mr_eff == mr && nr_eff == NR && rsc == 1`——第三站的方向归一化正是让这个条件变得常见的原因）：每个累加寄存器直接与 `C` 结合——`beta == 0` 原样存储（不读 C）、`beta == 1` 相加、一般 `beta` 融合乘加——再用向量存储写回。边缘 tile 和带步长的输出走通用路径：所有累加器先倒进栈上的 scratch tile（工作线程栈帧里的 `SCRATCH_LEN` 数组，零分配），再由标量循环沿 `C` 的任意步长逐元素做同样的 beta 运算。

普通 `gemm` 让 `Identity` epilogue 穿过这一切，而每个 epilogue 挂钩都由 `!E::IS_IDENTITY` 把守——这是关联 `const`，守卫在单态化时折叠殆尽，产出的内核与无 epilogue 的代码逐字节相同。融合调用（`gemm_fused`、`gemm_map`、重量化）跑同一个引擎，带一个只在 `last_k` 为真时点火的真 epilogue——在最后一个深度切片上，每个输出元素恰好一次。这条线索在 [Epilogue 融合](Epilogue融合.md)里继续。

## 抄近路回家

我们的 2x2x3 示例没见过上面大部分风景：它带着均为正的 `m, n, k` 和 `alpha == 1` 进入 `execute`，到达 `run_typed`，没过 gemv 闸门（`n != 1`、`m != 1`），被方向交换，没过小 `m,n` 闸门（`k = 3` 算不上长收缩），然后以 `k = 3 <= 16` 进了 `special/small_k.rs`：同一个微内核上的一个就地深度面板，没有分块，没有打包，工作区一次都没碰。2048 立方则走完了带并行 B 打包的完整驱动，在 `Parallelism::Rayon(0)` 下线程数随其总工作量扩展。同一个入口，同一份结果契约，两段截然不同的旅程——下面的层替调用者做了所有决定。各站的深层机制见[分块与缓存模型](分块与缓存模型.md)、[打包与工作区](打包与工作区.md)、[并行执行](并行执行.md)和[特殊路径](特殊路径.md)。
