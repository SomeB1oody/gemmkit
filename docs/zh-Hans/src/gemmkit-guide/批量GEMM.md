# 批量GEMM

一个大 GEMM 单靠自己就能把现代 CPU 喂满，一堆小 GEMM 却做不到。注意力头、分组卷积、逐样本的线性层、块对角求解，都会产出许多彼此独立的小乘积。把它们当成一个普通的 `gemm` 调用循环来跑，会让机器大部分时间闲着：每次调用都小到无法有效并行，可循环却仍然要么每个元素付一次 fork/join，要么干脆串行。批量入口一次性接下整组乘积，把它当作一个整体来调度：把整个 GEMM 分派给 worker，于是一批小矩阵真正能把核心填满。

批量 GEMM 是一层编排，而不是一个新内核。每个元素都会重新经由完整的单 GEMM 引擎来分发，所以一次批量调用会自动与 driver、gemv 路径以及[小形状路径](小形状与GEMV.md)组合起来：一批 `1 x 1 x k` 的乘积会在每个元素内部跑水平点积，一批普通形状则跑寄存器分块 driver。每个元素都是一个独立的 GEMM，整批在任意 worker 数下都可复现。

## 带步长形式

当各元素以规则的步长排布时（一个矩阵接一个矩阵地放在一块扁平缓冲区里），[`gemm_batched`](https://docs.rs/gemmkit) 只需接收一次单元素的形状和步长，外加 `A`、`B`、`C` 各自的一个批步长。元素 `b` 基于 `A + b*a_batch_stride`、`B + b*b_batch_stride`、`C + b*c_batch_stride`，所有元素共享同一个形状：

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

批步长为 `0` 会把一个操作数在整批上广播。这对只读的 `A` 或 `B` 是合法的，比如让同一个共享权重矩阵去乘一批输入，但对 `C` 绝不合法，因为 `C` 的各元素是并发写入的，必须互不重叠。这样得到的结果精确复现了一个 `gemm` 调用循环。

在 `epilogue` feature 之下，`gemm_batched_fused` 对每个元素施加同一份共享偏置和同一个共享激活，也就是批量线性层的情形。它复现了一个 [`gemm_fused`](融合Epilogue.md) 调用循环。这份偏置向量是按单个元素定尺寸的，不是按整批。

## 切片形式：逐元素形状

当各元素形状不同，或者根本不落在固定步长上时，用 [`gemm_batched_slice`](https://docs.rs/gemmkit)。它接收一个 [`BatchProblem`](https://docs.rs/gemmkit) 切片，每个元素都携带自己的 `alpha`、`A`、`B`、`beta`，以及一个独立的 `&mut` `C` 视图：

```rust
use gemmkit::{gemm_batched_slice, BatchProblem, MatRef, MatMut, Parallelism};

let mut problems: Vec<BatchProblem<'_, f32>> = /* 每个乘积一个，各有各的形状 */;
gemm_batched_slice(&mut problems, Parallelism::Rayon(0));
```

因为每个 `C` 都是一个独立的 `&mut`，各输出天然两两不相交，也不可能与输入产生别名。所以校验只需检查逐元素的形状是否一致、步长是否在界内。当你的矩阵本就以一个视图 `Vec` 的形式存在时，就该用这个形式。它的裸对应版本 `gemm_batched_ptr_unchecked` 接收一个 `GemmProblem` 切片，把同样的逐元素形状以裸指针的形式给出。它服务于自行校验输入、可能使用任意或负步长的 FFI 和适配器。两者都在 [Unchecked 层](Unchecked层.md)中有详细说明。

## 一批是怎么调度的

真正有意思的决定，是这些工作怎么在核心间铺开。引擎会在每次调用时，根据共享形状和批大小做一次这个决定。一共有 3 种调度：

- **批级并行（batch-parallel）。** 各 worker 从一个共享游标里领取互不相交的元素区段，每个元素在一个 worker 上*串行*、缓存驻留地跑。这正是这套 API 的核心意义所在：对许多小矩阵，它为整批只付一次 fork/join，而不是每个元素一次，并且让每个核心都忙在完整的 GEMM 上。这种调度从不把某个元素拆到多个 worker 上，所以在任意 worker 数下都与串行运行逐位一致。
- **串行（serial）。** 整批在调用线程上跑，每个元素单线程执行。当总工作量太小、不值得付一次 fork/join 时，引擎会选用这种调度。
- **顺序 + 内部并行（sequential with internal parallelism）。** 对于元素少而大、受内存带宽支配的情形，引擎会把这批循环起来，逐个把*完整*的引擎并行度交给每个元素。当一个元素大到自己就能饱和内存带宽时，把它铺满所有核心，胜过同时跑好几个元素而互相冲刷缓存。这种调度只用于 `m, n > 1` 的形状，因为这些形状的路由本就会把每个输出的归约收在一个 worker 内完成，所以它仍然可复现。

对你而言，结论很简单：把整批交给引擎，让它去挑选调度方式。许多小而独立的乘积，正是批处理胜过手写循环之处。手写循环做不出「为所有元素只付一次 fork/join」这种选择，它要么并行每个小 GEMM（这几乎全是开销），要么干脆串行地跑。对于少数几个大乘积，批量调用会收敛到普通循环本就做得不错的行为，所以批处理在那里既帮不上多少忙，也不会拖后腿。

确定性贯穿这 3 种调度。每个元素都是独立的，所以整批在不同 worker 数下都可复现。串行与批级并行这两种调度更进一步：因为每个元素都只在一个 worker 上完整运行，它们在任意 worker 数下都逐位一致。元素少而大的那种调度，则继承了它所跑的那条路由自身的串行与并行行为。零长度的批是一个空操作。

和 API 的其余部分一样，每个批量入口都有一个 `_with` 变体，复用调用方持有的 `Workspace`，以避免每次调用都分配内存。有一个细节值得了解：在批级并行调度下，打包无法走单个共享的 `Workspace`，因为并发的 worker 会在它上面互相冲突。所以这种调度改为让每个 worker 走自己那份持久的线程局部池，其复用方式与你自己的 `Workspace` 是一样的。
