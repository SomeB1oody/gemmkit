# SIMD令牌与ISA分发

gemmkit 在运行时选择指令集，这个决定与 Rust 编译 SIMD 内建函数的方式相冲突。AVX 和 AVX-512 内建函数只有在 target feature 已启用的上下文中才能正确生成代码。这个上下文通常来自外层函数上的 `#[target_feature(enable = "...")]` 属性。程序只有在真正运行于某颗具体 CPU 上时，才知道哪些 feature 可以安全启用。微内核是所有指令集共用的同一个泛型函数，所以没有一个属性可以单独钉在它身上。

本页讲两件事。第一，L0 SIMD 层（`gemmkit/src/simd.rs` 与 `gemmkit/src/simd/`）如何用零尺寸 ISA 令牌加一个蹦床函数化解这对矛盾。第二，L7 分发层（`gemmkit/src/dispatch.rs` 与 `gemmkit/src/dispatch/`）如何选出并缓存获胜的内核。

## ISA 令牌与 vectorize 蹦床

ISA 令牌是一个零尺寸类型，代表一种指令集选择。x86 上的令牌是 `Fma`（AVX2 + FMA）和 `Avx512F`，外加点积内核变体 `Avx512Vnni` 与 `Avx512Bf16`。aarch64 上的令牌是 `Neon`。wasm32 上的令牌是 `Simd128`。`ScalarTok` 存在于每个平台，是可移植的兜底。

每个令牌都实现 `Simd` trait。它唯一的方法是 `vectorize`：在该令牌的 target feature 已启用的情况下运行一个闭包。下面的代码展示了整个机制，摘自 `gemmkit/src/simd/fma.rs`。

```rust
/// AVX2 + FMA ISA token
#[derive(Copy, Clone, Default)]
pub struct Fma;

impl Simd for Fma {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: the caller of `vectorize` (the runtime dispatcher) guarantees
        // the CPU supports avx2+fma(+f16c); `inner` then establishes the codegen
        // context, and `f` inlines into it
        unsafe { inner(f) }
    }
}
```

诀窍在于内联的方向。`inner` 是一个带 `#[target_feature]` 属性的极小函数。闭包 `f` 被内联进 `inner`。`f` 装着打包循环和微内核调用，这些代码全部由 `#[inline(always)]` 原语构成。于是每一条内建指令都落在 feature 已启用的代码生成上下文里，泛型内核本身却从未被任何属性触碰。

`unsafe` 契约只有一条义务：调用者必须保证 CPU 确实支持该令牌的 feature。运行时分发器在每个进程里只确认一次这件事。

这与 pulp 和 faer 使用的模式相同。它对串行路径和 rayon 工作线程闭包同样适用。驱动层把每个列条带的微内核调用整体包进 `simd.vectorize(|| ...)`，蹦床的开销就这样摊薄到许多个 tile 上。

`ScalarTok` 的 `vectorize` 就是一句 `f()`，无需启用任何东西。这正是标量路径能在任何地方运行、包括在 Miri 下运行的原因。

## SimdOps：按元素类型展开的指令词汇表

L0 一共定义了 3 个 trait，不是 2 个。`Simd` 就是上文的 ISA 令牌 trait。`SimdOps<T>` 是本节要讲的、按元素类型展开的词汇表。`KernelSimd<L, R, A, O>` 是第三个 trait：当一个家族的输入类型、累加器类型和输出类型并不完全相同时，它负责把加载值加宽、把存储值收窄。本节末尾会讲到它。

令牌本身对元素类型一无所知。所有实际运算都放在 `SimdOps<T>` 上，按 `(ISA, T)` 对分别实现一次。它给出寄存器类型 `Reg`、通道数 `LANES`，以及微内核需要的每一条原语。令牌与元素类型是解耦的，所以 `LANES` 随这个二元组变化。`f32` 在 `Fma` 下是 8 通道，在 `Avx512F` 下是 16 通道。`f64` 的通道数是同一令牌下 `f32` 的一半。

这份词汇表刻意做得很厚。基础操作有 `zero`、`splat`、`loadu`、`storeu`、`mul`、`add`，以及融合乘加 `mul_add`。它的减法搭档是 `fnma`，计算 `c - a*b`。复数内核的某个累加项要靠 `fnma` 才能算。词汇表里还有水平求和 `reduce_sum`，供 gemv 与点积 epilogue 使用。

在这些之上还有几个原语。`max` 与 `min` 只有实数浮点令牌才实现，供融合 ReLU 和 clip epilogue 使用。`LANE_FMA` 标志和它的 `fma_bvec` 方法给 NEON 提供了一条按通道索引的 FMA 路径：把一段 RHS 列作为一个向量整体加载，替代逐列发出 splat。`accumulate_tile` 就是 GEMM 的内层循环本体。它的可移植默认调度在任何乱序核心上都已经会被直接编译成教科书式的寄存器分块内核。

复数拆分内核在这里也有自己的接缝，叫 `cplx_microkernel`。点积内核也有自己的接缝，叫 `dot_accumulate`，它长在 `KernelSimd` 上，而不是 `SimdOps` 上。当一个家族的输入类型、累加器类型和输出类型并不全部相等时，`KernelSimd<L, R, A, O>` 就是它要驱动的那个接缝，比如 `f16` 输入配 `f32` 累加器。它把较窄的输入加载值加宽成累加器类型，再把累加器的值收窄后写出。同构家族的这 4 个类型全部相等，靠一个覆盖一切（blanket）实现就能免费获得 `KernelSimd`，完全不需要任何专属某个 ISA 的代码。这两个接缝的细节见[标量与内核家族](标量与内核家族.md)和[点积内核与深K孪生](点积内核与深K孪生.md)。

厚，正是设计意图所在。matrixmultiply 的 per-ISA trait 很薄，逼着每个指令集都从头重写一遍内核。这里内核需要的每一条原语都在 `SimdOps` 背后，于是微内核成了横跨所有 ISA 的同一个泛型函数。新增一个指令集的成本是一个新令牌、它的 `SimdOps` 实现，加上每条分发阶梯里的一行。`simd` 模块只依赖 `crate::scalar` 和 `core`，对内核、驱动、缓存层没有任何反向依赖，所以整个抽象可以原封不动搬进一个独立的 crate。

## 分发层

分发层把"用哪个令牌"这个问题变成一次性的决定。每个可分发的元素类型都拥有一个 `OnceLock` 槽位，存放一个 `Dispatched<T>` 描述符。`f32` 和 `f64` 用 `gemmkit/src/dispatch/float.rs`。`f16` 和 `bf16` 用 `dispatch/mixed.rs`。`i8` 用 `dispatch/int.rs`，它有自己的 `IntDispatched` 与 `IntRequantDispatched` 形态，因为这些类型是异构的。`c32` 和 `c64` 用 `dispatch/complex.rs`。下面这段代码摘自 `dispatch/float.rs`，略有删节。

```rust
#[derive(Copy, Clone)]
pub(super) struct Dispatched<T> {
    pub(super) run: GemmFn<T>,
    pub(super) run_packed: PackedFn<T>,
    #[cfg(feature = "epilogue")]
    pub(super) run_fused: FusedFn<T>,
    #[cfg(feature = "epilogue")]
    pub(super) run_packed_fused: PackedFusedFn<T>,
    pub(super) mr: usize,
    pub(super) nr: usize,
    pub(super) depth_multiple: usize,
}
```

这个槽位缓存了获胜的单态化入口：普通内核、预打包 RHS 内核，以及（在 `epilogue` feature 下）它们的融合孪生版本。它同时缓存微铺块几何 `(mr, nr)` 和该家族的 `depth_multiple`。缓存几何信息是为了让 `prepack_rhs` 能用与后续消费调用相同的 ISA 选择来确定缓冲区尺寸。`depth_multiple` 让 bf16 预打包路径把打包深度取整，匹配点积内核的布局。这里的一切都是带类型的函数指针，没有 `transmute`，也没有 `AtomicPtr<()>`。

一次调用沿着固定的链路走：`gemm` 调用 `dispatch::execute`，它先处理退化情形。`dispatch::execute` 再调用 `T::dispatch`，读取记忆化的槽位。槽位解析为一次间接调用，进入像 `gemm_f32_avx512f` 这样的包装函数，这个包装函数把共享的泛型入口实例化为 `run_typed::<f32, Avx512F, 2, 12>`。

选择只运行一次，就在 `OnceLock` 的初始化器里。它先处理 `GEMMKIT_REQUIRE_ISA` 锁定（见下文）。之后，自动阶梯在 x86 上先探测 `avx512f`，再探 `avx2` 加 `fma`，最后落到标量。aarch64 上 NEON 是基线，架构规定 NEON 必备，所以这里无需探测。

wasm32 上根本没有运行时特性检测。`simd128` 在编译期由 `cfg(target_feature = "simd128")` 决定。构建必须传 `-C target-feature=+simd128`，否则就会拿到标量内核。标量在所有架构上都是地板。

各类型的阶梯在同一骨架上加自己的门槛。`f16` 的 FMA 分支还额外要求 `f16c`，因为 `vcvtph2ps` 和 `vcvtps2ph` 转换需要它。`bf16` 阶梯在普通 AVX-512F 之前先试 `avx512bf16` 点积内核。`i8` 阶梯在加宽内核之前先试 `avx512vnni`（连同 `avx512bw`）。

有两个构建模式的细节值得知道。有 `std` 时，特性检测走 `is_x86_feature_detected!`，结果记忆化在 `OnceLock` 里。没有 `std` 时不存在运行时 CPU 检测，因为 `raw-cpuid` 由 `std` 门控。探测宏退化为 `cfg!(target_feature = ...)`，`GEMMKIT_REQUIRE_ISA` 的解析退化为 `Auto`，select 函数每次调用都会执行。不过其中每个分支此时都是编译期常量，所以会直接折叠成一个确定的选择。`no_std` 构建就只跑其编译期 target feature 保证的那条路径。参见 [no_std 与 WebAssembly](../gemmkit-guide/no_std与WebAssembly.md)。

## 作为 const 泛型的微铺块几何

除指令编码之外，真正随 `(类型, ISA)` 变化的只有微铺块形状。它表达为在分发点选定的一对 const 泛型 `(MR_REG, NR)`，从来不是新类型、新 trait 或宏。`MR_REG` 是铺块的寄存器高度，行数即 `MR = MR_REG * LANES`。下表以 `f32` 为例。

| ISA | `(MR_REG, NR)` | `LANES` | 铺块 `MR x NR` | 寄存器预算 |
|---|---|---|---|---|
| AVX-512F | `(2, 12)` | 16 | 32 x 12 | 24 累加 + 2 lhs + 1 rhs = 27 个 ZMM |
| FMA（AVX2） | `(2, 6)` | 8 | 16 x 6 | 12 累加 + 2 lhs + 1 rhs = 15 个 YMM |
| NEON | `(4, 4)` | 4 | 16 x 4 | 16 累加 + 4 lhs + 1 rhs = 21/32 个向量寄存器 |
| simd128 | `(2, 4)` | 4 | 8 x 4 | 8 累加 + 2 lhs + 1 rhs = 11 个活跃 `v128` |
| 标量 | `(4, 4)` | 1 | 4 x 4 | 普通局部变量 |

`f64` 的通道数减半，同样的 `(MR_REG, NR)` 组合于是给出 16x12（AVX-512F）、8x6（FMA）、8x4（NEON）、4x4（simd128）。这些预算不是巧合。NEON 刻意留出约 11 个空闲寄存器，给宽乱序核心留出重命名余量，让它把下一步的加载与当前的 FMA 重叠起来。simd128 停在 11 个活跃向量，因为 LLVM 的 wasm 后端在大约 16 个之后就会开始溢出。这些注释就写在 `dispatch/float.rs` 里包装函数的旁边，所以上表就是代码本身，不是一个愿望。

## 用 GEMMKIT_REQUIRE_ISA 锁定内核

默认情况下，最优可用的 ISA 胜出。设置环境变量 `GEMMKIT_REQUIRE_ISA` 会强制锁定唯一一个内核，不再自动选择。它接受下面这些取值（不区分大小写）：

- `scalar`
- `fma`（别名 `avx2`）
- `avx512f`
- `avx512vnni`（别名 `vnni`）
- `avx512bf16`（别名 `bf16`）
- `neon`
- `simd128`（别名 `wasm`）
- `auto`

未设置或空串等同于 `auto`。无法识别的值会直接 panic，这样 CI 配置里的拼写错误就不可能悄悄选中别的东西。`avx512vnni` 锁定 `i8` 的 `vpdpbusd` 点积内核。`avx512bf16` 锁定 `bf16` 的 `vdpbf16ps` 点积内核。对其余元素类型，两者都会解析为普通 AVX-512F 路径。

这里最核心的行为是：锁定永不回退。如果 CPU（或 Intel SDE 这类模拟器）没有报告所需的特性，选择就会 panic。如果所请求的 ISA 在目标架构上根本不存在，比如非 aarch64 上的 `neon`，或非 x86 上的 `fma`/`avx512*`，选择同样会 panic。panic 消息会指明缺失的特性。这么做是为了 CI 的诚实性：一个想要检验某个内核的任务，必须大声失败，而不是默默测了另一个内核。

`simd128` 的锁定在 wasm 上同样有用。那个 target feature 是一个极易被遗忘的编译期开关。锁定把"忘了传开关"从静默回退到标量，变成了拒绝运行的构建。

取值只读一次。选择结果记忆化在各类型的 `OnceLock` 里，所以必须在第一次 GEMM 调用之前，在进程环境中设好这个变量。之后再修改，在进程的生命周期内都不再生效。锁定的使用侧内容，包括 CI 配方以及它与调优旋钮的配合，见[运行时 ISA 分发](../gemmkit-guide/运行时ISA分发.md)。
