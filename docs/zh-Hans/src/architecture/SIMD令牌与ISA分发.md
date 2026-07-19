# SIMD令牌与ISA分发

gemmkit 在运行时选择指令集，而这个决定与 Rust 编译 SIMD 内建函数的方式正面相撞。AVX 和 AVX-512 的内建函数必须在对应 target feature 已启用的上下文中生成代码——通常靠外层函数上的 `#[target_feature(enable = "...")]` 属性。但哪些 feature 可以安全启用，只有程序真正跑在某颗具体 CPU 上时才知道；而微内核是所有指令集共用的同一个泛型函数，根本没有一个可以钉死在它身上的属性。本页讲 L0 SIMD 层（`gemmkit/src/simd.rs` 与 `gemmkit/src/simd/`）如何用零尺寸 ISA 令牌加一个蹦床（trampoline）化解这对矛盾，以及 L7 分发层（`gemmkit/src/dispatch.rs` 与 `gemmkit/src/dispatch/`）如何选出并缓存获胜的内核。

## ISA 令牌与 vectorize 蹦床

ISA 令牌是一个零尺寸类型，代表一种指令集选择：x86 上有 `Fma`（AVX2 + FMA）和 `Avx512`，外加点积内核变体 `Avx512Vnni` 与 `Avx512Bf16`；aarch64 上是 `Neon`；wasm32 上是 `Simd128`；而 `ScalarTok` 在任何平台都存在，是可移植的兜底。每个令牌实现 `Simd` trait，它唯一的方法是 `vectorize`：在该令牌的 target feature 已启用的上下文中运行一个闭包。整个机制就这么大，摘自 `gemmkit/src/simd/fma.rs`：

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

诀窍在于内联的方向。`inner` 是一个带 `#[target_feature]` 标注的极小函数，而闭包 `f`——打包循环和微内核调用，全部由 `#[inline(always)]` 的原语拼成——被内联*进* `inner`。于是每一条内建指令都落在 feature 已启用的代码生成上下文里，尽管泛型内核本身从未沾过任何属性。`unsafe` 契约只有一条义务：调用者必须保证 CPU 确实支持该令牌的 feature，而运行时分发器在每个进程里只确认一次。这是 pulp/faer 已验证过的模式，对串行路径和 rayon 工作线程闭包同样成立；驱动层把每个列条带的微内核调用整体包进 `simd.vectorize(|| ...)`，蹦床开销摊薄到许多个 tile 上。`ScalarTok` 的 `vectorize` 就是一句 `f()`——无需启用任何东西——这正是标量路径处处可跑、连 Miri 下也能跑的原因。

## SimdOps：按元素类型展开的指令词汇表

令牌刻意对元素类型一无所知。所有实际运算都放在第二个 trait `SimdOps<T>` 上，按 `(ISA, T)` 对分别实现：它给出寄存器类型 `Reg`、通道数 `LANES`，以及微内核需要的每一条原语。因为令牌与元素类型解耦，`LANES` 随二元组变化——`f32` 在 `Fma` 下是 8 通道、在 `Avx512` 下是 16 通道，`f64` 减半。

这份词汇表刻意做得很厚。基础操作有 `zero`、`splat`、`loadu`、`storeu`、`mul`、`add`、融合乘加 `mul_add` 及其减法搭档 `fnma`（`c - a*b`，复数内核需要），还有 gemv 与点积 epilogue 用的水平求和 `reduce_sum`。在此之上还有 `max`/`min`（只有实数浮点令牌覆写，供融合 ReLU/clip epilogue 使用）、`LANE_FMA` 标志与 `fma_bvec`（NEON 的按通道索引 FMA 路径：把一段 RHS 列作为一个向量整体加载，替代逐列 splat），以及 `accumulate_tile`——GEMM 的内层循环本体，其可移植默认调度在任何乱序核心上都会被 LLVM 直接降低成教科书式的寄存器分块内核。复数拆分内核在这里也有自己的接缝（`cplx_microkernel`），点积内核亦然（伴生 trait `KernelSimd` 上的 `dot_accumulate`）——细节见[标量与内核家族](标量与内核家族.md)和[点积内核与深K孪生](点积内核与深K孪生.md)。

厚，正是设计意图。matrixmultiply 的薄 per-ISA trait 逼着每个指令集重写一遍内核；这里内核需要的*每一条*原语都在 `SimdOps` 背后，于是微内核是横跨所有 ISA 的同一个泛型函数，新增一个指令集的成本是一个新令牌、它的 `SimdOps` 实现、加上每条分发阶梯里的一行。`simd` 模块只依赖 `crate::scalar` 和 `core`——对内核、驱动、缓存层零反向依赖——整个抽象可以原封不动拆成独立 crate。

## 分发层

分发层把"用哪个令牌"变成一次性决定。每个可分发的元素类型拥有一个 `OnceLock` 槽位，存放一个 `Dispatched<T>` 描述符：`f32`/`f64` 在 `gemmkit/src/dispatch/float.rs`，`f16`/`bf16` 在 `dispatch/mixed.rs`，`i8` 在 `dispatch/int.rs`（类型异构，所以有自己的 `IntDispatched`/`IntRequantDispatched` 形态），`c32`/`c64` 在 `dispatch/complex.rs`。摘自 `dispatch/float.rs`，略有删节：

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

槽位缓存获胜的单态化入口——普通内核、预打包 RHS 内核，以及（`epilogue` feature 下）它们的融合孪生——外加微铺块几何 `(mr, nr)` 和家族的 `depth_multiple`。缓存几何是为了让 `prepack_rhs` 用与后续消费调用*相同*的 ISA 选择来确定缓冲区尺寸；`depth_multiple` 则让 bf16 预打包路径把打包深度取整到点积内核的布局。一切都是带类型的函数指针：没有 `transmute`，没有 `AtomicPtr<()>`。一次调用的路径是 `gemm` → `dispatch::execute`（退化情形在此处理）→ `T::dispatch` → 记忆化的槽位 → 一次间接调用进入 `gemm_f32_avx512` 这样的包装函数，后者把共享泛型入口实例化为 `run_typed::<f32, Avx512, 2, 12>`。

选择只跑一次，就在 `OnceLock` 的初始化器里。在优先处理 `GEMMKIT_REQUIRE_ISA` 锁定（见下文）之后，自动阶梯在 x86 上先探测 `avx512f`，再探 `avx2` + `fma`，最后落到标量。aarch64 上 NEON 是基线——架构规定必备，无需探测。wasm32 上根本没有运行时特性检测：`simd128` 在编译期由 `cfg(target_feature = "simd128")` 决定，构建必须传 `-C target-feature=+simd128`，否则拿到标量内核。标量在所有架构上都是地板。各类型的阶梯在同一骨架上加自己的门槛：`f16` 的 FMA 分支额外要求 `f16c`（`vcvtph2ps`/`vcvtps2ph` 转换需要），`bf16` 阶梯在普通 AVX-512 之前先试 `avx512bf16` 点积内核，`i8` 阶梯在加宽内核之前先试 `avx512vnni`（连同 `avx512bw`）。

两个构建模式细节值得知道。有 `std` 时，特性检测走 `is_x86_feature_detected!`，结果记忆化在 `OnceLock` 里。没有 `std` 时不存在运行时 CPU 检测（`raw-cpuid` 由 `std` 门控）：探测宏退化为 `cfg!(target_feature = ...)`，`GEMMKIT_REQUIRE_ISA` 解析退化为 `Auto`，select 函数每次调用都会执行——但其中每个分支此时都是编译期常量，直接折叠成一个确定选择。`no_std` 构建就跑其编译期 target feature 保证的那条路径；参见 [no_std 与 WebAssembly](../gemmkit-guide/no_std与WebAssembly.md)。

## 作为 const 泛型的微铺块几何

除指令编码之外，真正随 `(类型, ISA)` 变化的只有微铺块形状，而它表达为在分发点选定的一对 const 泛型 `(MR_REG, NR)`——从来不是新类型、新 trait 或宏。`MR_REG` 是铺块的寄存器高度，行数即 `MR = MR_REG * LANES`。以 `f32` 为例：

| ISA | `(MR_REG, NR)` | `LANES` | 铺块 `MR x NR` | 寄存器预算 |
|---|---|---|---|---|
| AVX-512 | `(2, 12)` | 16 | 32 x 12 | 24 累加 + 2 lhs + 1 rhs = 27 个 ZMM |
| FMA（AVX2） | `(2, 6)` | 8 | 16 x 6 | 12 累加 + 2 lhs + 1 rhs = 15 个 YMM |
| NEON | `(4, 4)` | 4 | 16 x 4 | 16 累加 + 4 lhs + 1 rhs = 21/32 个向量寄存器 |
| simd128 | `(2, 4)` | 4 | 8 x 4 | 8 累加 + 2 lhs + 1 rhs = 11 个活跃 `v128` |
| 标量 | `(4, 4)` | 1 | 4 x 4 | 普通局部变量 |

`f64` 通道数减半，同样的 `(MR_REG, NR)` 组合于是给出 16x12（AVX-512）、8x6（FMA）、8x4（NEON）、4x4（simd128）。这些预算不是巧合：NEON 刻意留出约 11 个空闲寄存器，给宽乱序核心以重命名余量，让下一步的加载与当前的 FMA 重叠；simd128 停在 11 个活跃向量，因为 LLVM 的 wasm 后端在大约 16 个之后开始溢出。这些注释就写在 `dispatch/float.rs` 的包装函数旁边——上表是代码本身，不是愿望。

## 用 GEMMKIT_REQUIRE_ISA 端到端锁定内核

默认由最优可用 ISA 胜出。设置环境变量 `GEMMKIT_REQUIRE_ISA` 则强制锁定唯一一个内核。接受的取值（不区分大小写）为 `scalar`、`fma`（别名 `avx2`）、`avx512`（别名 `avx512f`）、`avx512vnni`（别名 `vnni`）、`avx512bf16`（别名 `bf16`）、`neon`、`simd128`（别名 `wasm`）和 `auto`；未设置或空串等同 `auto`，无法识别的值直接 panic，这样 CI 配置里的拼写错误不可能悄悄选中别的东西。`avx512vnni` 锁定 `i8` 的 `vpdpbusd` 点积内核，`avx512bf16` 锁定 `bf16` 的 `vdpbf16ps` 点积内核；对其余元素类型，两者都解析为普通 AVX-512 路径。

决定性的行为是：锁定永不回退。如果 CPU（或 Intel SDE 这类模拟器）没有报告所需特性，或者所请求的 ISA 在目标架构上根本不存在——非 aarch64 上的 `neon`、非 x86 上的 `fma`/`avx512*`——分发会带着指明缺失特性的消息 panic。理由是 CI 的诚实性：一个想要检验某个内核的任务必须大声失败，而不是默默测了另一个内核。`simd128` 的锁定在 wasm 上同理见效：那个 target feature 是个极易遗忘的编译期开关，锁定把"忘了传开关"从静默回退标量变成拒绝运行。

取值只读一次。选择结果记忆化在各类型的 `OnceLock` 里，因此必须在第一次 GEMM 调用之前于进程环境中设好；之后再改在进程生命周期内不再生效。锁定的使用侧——CI 配方以及与调优旋钮的配合——见[运行时 ISA 分发](../gemmkit-guide/运行时ISA分发.md)。
