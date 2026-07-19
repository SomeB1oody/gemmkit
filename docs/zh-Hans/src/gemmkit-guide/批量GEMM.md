# 批量GEMM

一个大 GEMM 单靠自己就能把现代 CPU 喂满；一堆小 GEMM 却做不到。注意力头、分组卷积、逐样本的线性层、块对角求解，都会产出许多彼此独立的小乘积，而把它们当成一个普通的 `gemm` 调用循环来跑，会让机器大部分闲着：每次调用都小到无法有效并行，可循环却要么每个元素付一次 fork/join，要么干脆串行。批量入口一次性接下整组，把它当作一个整体来调度——把整个 GEMM 分派给 worker，于是一批小矩阵真正把核心填满。

批量 GEMM 是一层编排，而非一个新内核。每个元素都重新经由完整的单 GEMM 引擎分发，所以一次批量调用会自动与 driver、gemv 路径以及[小形状路径](小形状与GEMV.md)组合：一批 `1 x 1 x k` 的乘积会在每个元素内部跑水平点积，一批普通形状会跑寄存器分块 driver。每个元素都是一个独立的 GEMM，整批在任意 worker 数下都可复现。

## 带步长形式

当各元素以规则的步长排布——一个矩阵接一个矩阵地放在一块扁平缓冲区里——时，[`gemm_batched`](https://docs.rs/gemmkit) 只接收一次单元素的形状和步长，外加 `A`、`B`、`C` 各自的一个批步长。元素 `b` 基于 `A + b*a_batch_stride`、`B + b*b_batch_stride`、`C + b*c_batch_stride`，全部共享同一个形状：

```rust
use gemmkit::{gemm_batched, MatRef, MatMut, Parallelism};

// batch 个独立的 m x k 乘 k x n 乘积，连续排布
gemm_batched(
    batch,
    1.0,
    MatRef::new(&a, m, k, 1, m as isize), (m * k) as isize, // A 单元素 + 批步长
    MatRef::new(&b, k, n, 1, k as isize), (k * n) as isize, // B 单元素 + 批步长
    0.0,
    MatMut::new(&mut c, m, n, 1, m as isize), (m * n) as isize, // C 单元素 + 批步长
    Parallelism::Rayon(0),
);
```

批步长为 `0` 会把一个操作数在整批上广播，对只读的 `A` 或 `B` 合法（同一个共享权重矩阵去乘一批输入），但对 `C` 绝不合法——它的各元素是并发写入的，必须互不重叠。结果精确复现一个 `gemm` 调用循环。在 `epilogue` feature 之下，`gemm_batched_fused` 对每个元素施加**一份**共享偏置和**一个**共享激活，即批量线性层的情形，并复现一个 [`gemm_fused`](融合Epilogue.md) 调用循环；那份偏置向量是按单个元素定尺寸的，不是按整批。

## 切片形式：逐元素形状

当各元素形状不同，或者根本不落在固定步长上时，[`gemm_batched_slice`](https://docs.rs/gemmkit) 接收一个 [`BatchProblem`](https://docs.rs/gemmkit) 切片，每个都携带自己的 `alpha`、`A`、`B`、`beta`，以及一个独立的 `&mut` `C` 视图：

```rust
use gemmkit::{gemm_batched_slice, BatchProblem, MatRef, MatMut, Parallelism};

let mut problems: Vec<BatchProblem<'_, f32>> = /* 每个乘积一个，各有各的形状 */;
gemm_batched_slice(&mut problems, Parallelism::Rayon(0));
```

因为每个 `C` 都是一个独立的 `&mut`，各输出天然两两不相交、也不可能与输入别名，所以校验只需检查逐元素的形状一致和步长在界内。当你的矩阵本就以一个视图 `Vec` 的形式存在时，这就是该用的形式。它的裸对应版本 `gemm_batched_ptr_unchecked` 接收一个 `GemmProblem` 切片，把同样的逐元素形状以裸指针给出，供自行校验输入、可能使用任意或负步长的 FFI 和适配器使用；两者都在 [Unchecked 层](Unchecked层.md)中展开。

## 一批是怎么调度的

真正有意思的决定，是这些活怎么在核心间铺开，而引擎会在每次调用时、根据共享形状和批大小做一次这个决定。共有三种调度：

- **批级并行（batch-parallel）。** 各 worker 从一个共享游标里领取互不相交的元素区段，每个元素在一个 worker 上*串行*、缓存驻留地跑。这正是这套 API 的意义所在：对许多小矩阵，它为整批只付一次 fork/join，而非每个元素一次，并且让每个核心都忙在完整的 GEMM 上。因为没有任何元素被跨 worker 切分，这种调度对任意 worker 数都与串行运行逐位一致。
- **串行（serial）。** 整批在调用线程上跑，每个元素单线程。当总工作量太小、不值得一次 fork/join 时选用。
- **顺序 + 内部并行（sequential with internal parallelism）。** 对于元素少而大、受 DRAM 支配的情形，把这批循环起来，逐个把*完整*的引擎并行度交给每个元素。当元素大到单个就能饱和内存带宽时，把单个元素铺满所有核心，胜过同时跑好几个而互相冲刷缓存。这种调度只用于 `m, n > 1` 的形状，其路由会把每个输出的归约收在一个 worker 内，所以它仍然可复现。

对你而言，结论很简单：把整批交给引擎，让它去挑。许多小而独立的乘积，正是批处理胜过手写循环之处，因为循环做不出「为所有元素只付一次 fork/join」的选择——它要么并行每个小 GEMM（几乎全是开销），要么串行地跑它们。对于少数几个大乘积，批量调用会收敛到普通循环本就做得不错的行为，所以批处理在那里既帮不上多少也不会拖后腿。

确定性贯穿三种调度。各元素相互独立，所以在固定配置下整批对 worker 数可复现；串行与批级并行调度因每个元素都串行运行，还额外对 worker 数逐位一致，而元素少而大的调度则继承它所跑那条路由的「串行等于并行」行为。零长度的批是一个空操作。

和 API 的其余部分一样，每个批量入口都有一个 `_with` 变体，复用调用方持有的 `Workspace` 以避免每次调用分配内存。有一个细节值得知道：在批级并行调度下，打包无法走单个共享的 `Workspace`（并发的 worker 会在它上面相撞），所以那种调度改为让每个 worker 走自己那份持久的线程局部池，其复用方式与你的 `Workspace` 一样。
