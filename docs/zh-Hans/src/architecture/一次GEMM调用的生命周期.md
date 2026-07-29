# 一次GEMM调用的生命周期

上一页描述的是静止的栈。这一页跟随一次调用穿过它。样本是 crate 文档里的快速上手示例：

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

// 2x3 * 3x2 = 2x2, all row-major
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

这个玩具形状会在下文的某个早退口离开主路。整个走读同时记住两个问题：上面的 2x2x3，
以及一台 AVX-512 机器上的 2048x2048x2048 `f32` 乘积。后面这个更大的乘积会一路走到底，
穿过每一层。下面是压缩过的路线：

```
gemm(alpha, A, B, beta, C, par)
  |  validate_gemm_views: shapes, bounds, aliasing     [api.rs]
  v
Task<T>: raw pointers + isize strides
  |  m == 0 || n == 0        -> return                 [dispatch.rs]
  |  k == 0 || alpha == 0    -> C <- beta*C, done
  v
memoized per-type kernel (OnceLock fn pointer)
  |  gemv shape (m==1||n==1) -> special/gemv.rs        [dispatch/float.rs]
  |  orient: row-major-ish C -> compute C^T = B^T*A^T
  |  small m,n + long k      -> special/small_mn.rs
  |  k <= small_k_threshold  -> special/small_k.rs
  v
driver::run                                            [driver.rs]
  jc over NC -> pc over KC (never parallel)
    -> flat job list (ic row-block x jt column-tile),
       workers drain a shared JobCursor, pack A/B adaptively
  v
Fam::microkernel_epi: MR x NR tile in registers        [kernel/float.rs]
  alpha/beta epilogue store (vector fast path | scratch drain)
```

## 第一站：校验与降解

`gemm` 本身只有一行。它借出线程本地工作区，转发给 `gemm_with`（`gemmkit/src/api.rs`）。
`gemm_with` 会运行 `validate_gemm_views`，也就是[设计目标](设计目标与总体图景.md)里的
完整 panic 清单。形状必须一致。每个视图都必须留在自己的切片以内。`C` 必须对每个
`(i, j)` 寻址唯一。`C` 不能与任何一个输入重叠。

然后视图就消解了。这一点以下的所有代码只讲 `Task<T>`：一个 `Copy` 结构体，装着
`m, k, n`、`alpha`/`beta`，以及三个带 `isize` 行/列步长的裸指针。转置从来不是一个标志
位，转置视图不过是交换了步长。当 `beta == 0` 时，契约规定 `C` 永远不会被读取，所以它
可以是未初始化的。

`unsafe` 边界恰好在这里跨越，由刚刚跑完的校验背书。`gemm_unchecked` 则晚一步进场，改由
调用方自己扛起这份背书。

## 第二站：分发层的早退口

`dispatch::execute`（`gemmkit/src/dispatch.rs`）会趁元素类型还是具体的，先处理掉退化
的代数情形：

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

输出为空意味着无事可做。`A*B` 项消失（`k == 0` 或 `alpha == 0`）时，调用会退化成一次
`C <- beta*C` 缩放。这次缩放从不读取 `A` 或 `B`。其中 `beta == 0` 会直接写零而不读
`C`，兑现未初始化 C 的契约。

只有真正的乘积才会到达 `T::dispatch`，它读取该类型的 `OnceLock` 槽。首次使用时，选择
阶梯会探测 CPU 特性，尊重 `GEMMKIT_REQUIRE_ISA` 的钉选（不满足时 panic 而非回退）。
随后它会缓存胜出的单态化入口，连同 tile 几何一起。此后每次调用都只是一次间接调用。在
那台 AVX-512 机器上，`f32` 解析为 `run_typed::<f32, Avx512F, 2, 12>`，也就是 32x12
的 tile。

## 第三站：`run_typed` 里的路由

`run_typed`（`gemmkit/src/dispatch/float.rs`）是一串简短的闸门，每道闸门都把寄存器
分块驱动伺候不好的形状改道送走。

先是 gemv。若 `m == 1 || n == 1`，且该路径未被 `GEMMKIT_GEMV_THRESHOLD` 封顶关闭，
调用会直接去 `special/gemv.rs`。这发生在方向归一化**之前**，用的是用户的原始坐标系。
gemv 自己解决方向问题。它把 `m == 1` 的情形当作转置后的 `rows x k` 问题来处理，输出行
的划分也由它自己完成。

其余的都会经过 `orient_transpose` 做方向归一化。若 `C` 是行主序倾向
（`|csc| < |rsc|`），分发层会把问题改写成它的转置：`C^T = B^T * A^T`。这会交换 `m` 与 `n`、
`A`/`B` 的指针与步长，以及 `rsc` 与 `csc`。

这个恒等式是免费的。数据一字节不动，只改描述符。它换来一条强不变量。此后，输出的
**行**步长是小的那个（对完全连续的 C 即 `rsc == 1`）。输出的每一列因此在内存中连续，
内核得以沿连续的列往下走。

微内核的快速存储路径恰恰要求这一点：`rsc == 1`，这样它就能用向量存储写一列中 `LANES`
个连续行。下面的每一层都只需为一种方向优化，而不是两种。那个全行主序的 2048 立方
就命中了这次交换。引擎实际算的是 `C^T`，而分发层以下无人知晓。

接下来是归一化任务上的另外两道闸门。

小 `m,n` 形状会去 `special/small_mn.rs`。两个维度都必须不超过 `small_mn_dim`，且收缩
长度要超过 `small_k_threshold`。在那里，每个输出元素是一次水平 SIMD 点积。当两个操作
数都沿 `k` 单位步长流动时，这是零拷贝的。当某个操作数带步长时（`k >
small_mn_pack_min_k`），会走一个打包档，只拷贝不合格的那个操作数。

小 `k` 形状（`k <= small_k_threshold`，默认 x86 为 16、aarch64 为 8）会去
`special/small_k.rs`。它把整个乘积当作一个就地读取的深度面板，直接过微内核，没有任何
分块和打包开销。

闯过全部闸门的（那个 2048 立方就是）会进入 `driver::run`。驱动声明的前置条件是：
`m, n, k > 0`、`alpha != 0`、方向已归一化。

## 第四站：驱动的循环嵌套

`driver::run` 带着零成本的 `Identity` epilogue 转发给 `run_inner`
（`gemmkit/src/driver.rs`）。融合入口会带着真正的 epilogue 落进同一个函数。驱动对
家族和 ISA 令牌泛型。对这次调用，那就是 `FloatGemm<f32>` 与 `Avx512F`，
`mr = MR_REG * LANES = 32`，`nr = 12`。

分块要先算。`cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k)` 按 BLIS 缓存
模型给出 `(MC, KC, NC)`。它们以**打包输入**元素计（`sizeof(Lhs)`，而非累加器），所以
窄类型会得到更深的块。随后循环嵌套按 BLIS 顺序展开：

- **`jc` 遍历 `NC`**：列块，尺寸保证打包后的 B 宏面板驻留 L3。
- **`pc` 遍历 `KC`**：深度切片。这层循环**永不并行**。所有深度切片都累加进同一批 C
  tile。把深度并行化就意味着对 C 做同步的读-改-写，或者拆分归约。让深度保持串行，才
  使每个输出元素由同一个工作线程从头归约到尾，这是可复现性契约的一半。`beta` 只在第
  一个切片（`pc == 0`）参与。之后的切片以等效 beta 为一累加。混合精度家族
  （`OUT_IS_ACC = false`）恰好只有一个切片，`kc = k`，运行中的部分和因此永远不会经过
  窄输出类型的舍入。
- **一份扁平的一维作业列表**：每个深度切片内，剩下的工作是 `n_mc` 个行块乘 `n_nt`
  个列 tile。它们压平成 `n_jobs = n_mc * n_nt` 个下标。工作线程按需从共享的无锁
  `JobCursor` 拉取连续区块。没有静态划分，快核自然吸收更多工作。区块粒度对线程数过
  采样（`job_grain`）。打包 LHS 路径改用与行块对齐的 `packed_block_grain`，区块因此
  永不跨越打包边界。线程数本身来自 `par.resolve(m*n*k, n_jobs)`。这是以工作量为准：
  随总工作量 `m*n*k` 除以每 worker 下限来扩展，而不是一步跳到全部核心。若这个线程数
  会让作业列表浅到每个 worker 分不到几个块，驱动器会先缩小 `mc`。这只会切出更多、更
  小的行块，因而不会移动任何结果比特。缩小 `mc` 会先把列表加深，再交给游标发放。

打包是自适应的，两侧各自独立决策。

B 会在 `m` 越过 `rhs_pack_threshold` 时每个深度切片打包一次。打包面板会被全部 `n_mc`
个行块复用，只有复用足够高，这次拷贝才划算。否则 B 按原始步长就地读取。真要打包时，
打包本身也是并行的。工作线程从游标拉取 `nr` 宽的列面板。`for_each_worker` 的汇合就是
先写后读的屏障。打包后的 B 是所有计算线程共享的唯一缓冲，这道屏障因此很关键。

A 有三种模式。每个工作线程都可以把手头的行块打包进自己的私有工作区区域。`rsa != 1`
或者行块不是 `mr` 的整倍数时，这是强制的。其余情况下，是否值得打包取决于每线程列复
用量，或者 TLB 不友好的列步长。大型并行问题上，一个共享预打包可以代替这一步：把每个
行块恰好打包一次。它打包进按块分配的区域，靠自己的屏障同步（`shared_lhs_mnk` 门限）。
这消除了各线程的重复打包。复用低到任何拷贝都摊不平时，A 就地读取。

这些区域的尺寸经由 `Workspace::regions` 预先切好，带着前文说过的"失败即封闭"溢出
检查。完全不打包的路线甚至不会碰工作区。

对每个作业，工作线程会解析出自己的 A 面板，已打包或就地。它会定位 B 面板：本次调用
打包的、预打包缓冲里的，或者就地的。随后它对块内每条 `mr` 行的条带调用微内核。这一切
都在 `simd.vectorize` 之内进行，整个条带因此都在 target-feature 代码生成上下文中
执行。

## 第五站：微内核与它的存储

`Fam::microkernel_epi`（`gemmkit/src/kernel/float.rs` 的 `microkernel_impl`）计算
一个 `MR x NR` 的 tile。对这次调用，那是 32x12 个 `f32` 值，以
`[[Reg; MR_REG]; NR]` 数组的形式驻留在 24 个 ZMM 累加寄存器里。

满宽 tile 走 `SimdOps::accumulate_tile`，即升序 `k` 的融合乘加调度。这是一条接缝，
像 NEON 这样受载入约束的 ISA 会换上一个软件流水的变体，只重排载入，绝不重排算术。
列方向的边缘 tile 则走一个运行时定界的循环。这个循环恰好读 `nr_eff` 列，保证未打包的
B 永远不会被读过最后一个真实列。

然后 `alpha` 会折进累加器。`alpha == 1` 时这一步整段跳过，靠的是驱动预先算好的
`AlphaStatus`。

存储是 `beta` 和 epilogue 的居所。它有两条路。

快路径在满 tile 且输出行步长为一时触发：`mr_eff == mr && nr_eff == NR && rsc ==
1`。第三站的方向归一化正是让这个条件变得常见的原因。每个累加寄存器直接与 `C` 结合。
`beta == 0` 时原样存储（不读 C），`beta == 1` 时相加，一般 `beta` 时融合乘加。结果
再用向量存储写回。

边缘 tile 和带步长的输出走通用路径。所有累加器先倒进栈上的 scratch tile，即工作线程
栈帧里的 `SCRATCH_LEN` 数组，零分配。随后一个标量循环沿 `C` 的任意步长逐元素做同样的
beta 运算。

普通 `gemm` 让 `Identity` epilogue 穿过这一切。每个 epilogue 挂钩都由
`!E::IS_IDENTITY` 把守，这是一个关联 `const`。守卫在单态化时折叠殆尽，产出的内核与
无 epilogue 的代码逐字节相同。

融合调用，比如 `gemm_fused`、`gemm_map`、重量化，跑的是同一个引擎，带一个真正的
epilogue。这个 epilogue 只在 `last_k` 为真时点火，也就是在最后一个深度切片上，每个
输出元素恰好一次。这条线索在 [Epilogue 融合](Epilogue融合.md)里继续。

## 抄近路回家

那个 2x2x3 示例没见过上面大部分风景。它带着均为正的 `m, n, k` 和 `alpha == 1` 进入
`execute`。它到达 `run_typed`，没过 gemv 闸门（`n != 1`、`m != 1`）。它被方向交换，
随后没过小 `m,n` 闸门，因为 `k = 3` 算不上长收缩。以 `k = 3 <= 16`，它进了
`special/small_k.rs`。这条路线是同一个微内核上的一个就地深度面板，没有分块，没有
打包，工作区一次都没碰。

2048 立方走完了带并行 B 打包的完整驱动。在 `Parallelism::Rayon(0)` 下，它的线程数随
总工作量扩展。

同一个入口，同一份结果契约，两段截然不同的旅程。下面的层替调用者做了所有决定，调用者
从不需要操心。各站的深层机制见[分块与缓存模型](分块与缓存模型.md)、
[打包与工作区](打包与工作区.md)、[并行执行](并行执行.md)和[特殊路径](特殊路径.md)。
