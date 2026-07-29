# Unchecked层

安全入口（`gemm`、`gemm_fused` 等等）在触碰内存之前会先校验输入。形状必须相符。每个跨步视图都必须落在自己的切片之内。输出必须一次性寻址每个元素，并且不能与输入重叠。

在每一层这样的检查之下，坐着的都是同一个引擎。它通过一个裸指针、`isize` 步长、完全不做检查的接口触达。这就是 unchecked 层。它是为那些已经持有安全 API 本会重新推导出的那些不变量的调用者而存在的。

## 它面向谁

这里住着三类调用者。封装其他矩阵库的**适配器**，比如 `ndarray`、`nalgebra`、`faer`，直接从宿主类型里就已经拿到了一个校验过的指针和步长。再检查一遍边界，就是在库已经保证过的数据上做冗余功。从 C 或其他语言过来的 **FFI 调用者**，手里只有一个指针和步长，根本没有 Rust 切片可供边界检查。代码库自有的**自定义矩阵类型**，可以降解为指针并直接调用引擎，而不必先拷进一个 `MatRef`。这几种情形里，知道内存有效的都是调用者自己，所以检查就搬到了那份知识所在之处。

如果以上都不是你的情况，就用安全 API。对单次调用而言，unchecked 层并不更快。相对于乘法本身，校验的代价很便宜。它存在的意义，是让已经持有不变量的调用者，不必再证明一遍。

## 目录清单

每个安全入口都有一个裸孪生，命名方式是加后缀 `_unchecked`，多数还提供一个接收调用者自有工作区的 `_with` 形式（见下一节）。按家族划分，完整的裸接口如下：

| 家族 | 裸入口 | Feature |
| --- | --- | --- |
| 普通 GEMM | `gemm_unchecked`、`gemm_unchecked_with` | 核心（`f32`/`f64`，`half` 下另加 `f16`/`bf16`） |
| 复数 | `gemm_cplx_unchecked`、`gemm_cplx_unchecked_with` | `complex` |
| 整数 | `gemm_i8_unchecked`、`gemm_i8_unchecked_with` | `int8` |
| 融合偏置/激活 | `gemm_fused_unchecked`、`gemm_fused_unchecked_with` | `epilogue` |
| Map（逐元素闭包） | `gemm_map_unchecked`、`gemm_map_unchecked_with` | `epilogue` |
| 复数融合 | `gemm_cplx_fused_unchecked`、`gemm_cplx_fused_unchecked_with` | `complex` + `epilogue` |
| 重量化 | `gemm_i8_requant_unchecked`、`gemm_i8_requant_u8_unchecked`（及 `_with`） | `int8` + `epilogue` |
| 跨步批量 | `gemm_batched_unchecked`、`gemm_batched_unchecked_with` | 核心 |
| 指针数组批量 | `gemm_batched_ptr_unchecked` | 核心 |
| 批量融合 | `gemm_batched_fused_unchecked`、`gemm_batched_fused_unchecked_with` | `epilogue` |
| 预打包 | `prepack_rhs_unchecked`、`prepack_lhs_unchecked`、`prepack_rhs_i8_unchecked` | 核心 / `int8` |
| 消费预打包 | `gemm_packed_a_unchecked`、`gemm_packed_b_unchecked`（及 `_with`、`_fused_`） | 核心 / `epilogue` |
| 消费预打包（i8） | `gemm_i8_packed_b_unchecked`（及 `_with`） | `int8` |

指针数组批量形式值得单拎出来说。`gemm_batched_ptr_unchecked` 接收一个 `GemmProblem<T>` 的切片，每个元素都有自己的形状、自己的指针。于是一个批次既能混合不同尺寸，也能把操作数散落在内存任意处。它没有同样形状的安全对应物。表达“一组相互独立的裸问题”正是裸层的用途所在。nalgebra 和 faer 适配器就把它们的批量 GEMM 搭建在它之上。

## 安全契约

调入 unchecked 层，意味着为安全 API 本会检查的东西逐项签字：

- **有效的指针与步长。** 对由维度和步长隐含的每个 `(i, j)`，`a` 与 `b` 对读有效，`c` 对读写有效。没有任何东西会为此做边界检查。越界的步长是未定义行为，而非 panic。
- **一个唯一寻址的输出。** `C` 的步长必须把每个不同的 `(i, j)` 映射到不同的位置。并行驱动器假定输出 tile 互不相交，并会并发写入它们。一个自别名的 `C`（例如 `rsc == 0`）就会因此成为一场数据竞争。输入则可以自由地自别名，因为它们只被读取。所以一个零步长的广播式 `A` 或 `B` 没有问题。
- **`C` 与 `A`/`B` 不重叠。** 输出会被写入。如果它与某个输入重叠，结果就会是垃圾。

有一处放宽是随之而来的。当 `beta == 0` 时，输出不被读取，因此 `C` 无需初始化。这里还有一项安全 API 不给的能力：**负步长**，以及指向缓冲区中部的指针，二者都是允许的。一个反向视图（`rs < 0`），或者一个从最后一个元素往回寻址的操作数，正是安全的 `MatRef` 会拒绝的那类布局。裸引擎则接纳它。这也是为什么封装那些会产出反向步长的库的适配器要转发到这一层。

## 复用工作区

每个裸入口有两种分配风格。普通形式 `gemm_unchecked` 借用线程本地打包池。它每线程至多分配一次。`_with` 形式 `gemm_unchecked_with` 则改为接收一个你自有的 `&mut Workspace`：

```rust
use gemmkit::{Workspace, Parallelism};

let mut ws = Workspace::new();
// each iteration reuses `ws`; after the first large call it does no heap work
for _ in 0..iters {
    // SAFETY: pointers/strides valid, c uniquely addressed, c disjoint from a/b
    unsafe {
        gemmkit::gemm_unchecked_with(
            &mut ws, m, k, n,
            1.0_f32, a, rsa, csa, b, rsb, csb, 0.0_f32, c, rsc, csc,
            Parallelism::Serial,
        );
    }
}
```

工作区会长到能容纳它服务过的最大问题。此后它复用那块分配，所以一个 GEMM 热循环能达到零稳态分配。这正是 `no_std` 构建赖以复用的机制，因为它们没有线程本地池。在 `std` 下，对于想把分配挪出热路径的实时或延迟敏感循环，它同样好用。

## 一个完整示例：自定义 tile 类型

假设你的代码已经带着自己的稠密行主序矩阵，你想把两个相乘，而不必先拷进一个 `MatRef`：

```rust
use gemmkit::{gemm_unchecked, Parallelism};

// a dense row-major matrix the caller already owns
struct Tile {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

// c = a * b for row-major tiles
fn matmul(a: &Tile, b: &Tile, c: &mut Tile) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);
    // row-major: row stride = cols, column stride = 1
    // SAFETY: shapes checked above; each tile owns a dense rows*cols buffer, so
    // every addressed element is in bounds; c is a distinct &mut, so it cannot
    // alias a or b, and a dense layout addresses each (i, j) once
    unsafe {
        gemm_unchecked(
            a.rows, a.cols, b.cols,
            1.0_f32,
            a.data.as_ptr(), a.cols as isize, 1,
            b.data.as_ptr(), b.cols as isize, 1,
            0.0_f32,
            c.data.as_mut_ptr(), c.cols as isize, 1,
            Parallelism::Serial,
        );
    }
}
```

`assert_eq!` 的形状检查与 `&mut Tile` 借用，两者合在一起就结清了整份契约。稠密存储让每个偏移都落在界内，也让每个 `(i, j)` 都各不相同。对 `c` 的独占借用则排除了它与 `a` 或 `b` 的重叠。这就是该采用的范式：在你自己类型的边界处证明不变量，然后把裸指针交给引擎。

## 适配器就是参照

把这件事做好的最干净的例子，就是适配器 crate 自身。每一个都从原生视图里抠出指针和步长：C 序、F 序、一般步长，或反向步长，全都零拷贝。然后各自转发到 `*_unchecked` 引擎，并在每个调用点附上一段简短的安全论证。如果你在封装自己的矩阵类型，读一读某个适配器章节，照着它的结构来。[nalgebra](../gemmkit-nalgebra/在nalgebra中使用gemmkit.md) 一章是个不错的起点。至于上面目录里的预打包入口，它们服务的定权重复用范式，在[预打包操作数](预打包操作数.md)中有介绍。
