# no_std与WebAssembly

gemmkit 的核心不需要操作系统。关掉默认 feature，crate 就是 `#![no_std]` 的，只需要 `core` 和 `alloc`，此外不依赖任何东西。这让它可用于内核、嵌入式固件以及 WebAssembly——而同一条代码路径也恰好是 wasm SIMD 后端的构建方式。本页讲清楚 `no_std` 构建放弃了什么、保留了什么，以及 wasm 目标额外需要哪些步骤。

## no_std 核心

`std` feature 默认开启（作为 `default = ["std", "parallel"]` 的一部分）；把默认关掉，你就走上了只带 `alloc` 的路径：

```toml
[dependencies]
gemmkit = { version = "0.1", default-features = false }
```

`alloc` 始终是必需的，因为两种构建里打包暂存都由堆支撑。除此之外，这种配置下 crate 什么也不拉。每个可选 feature 至多增加一个依赖：`std` 拉 `raw-cpuid`（仅 x86，用于 CPUID 缓存与特性探测），`parallel` 拉 `rayon`，`half` 拉 `half`，`complex` 拉 `num-complex`。`int8` 与 `epilogue` 完全不增加依赖。元素类型 feature 与 `no_std` 自由组合，所以 `default-features = false, features = ["half", "int8"]` 就是一个合法、零依赖的 `f16`/`bf16`/`i8` 引擎。

注意 `parallel` 蕴含 `std`（rayon 需要标准库），所以 `no_std` 构建总是单线程的。一切照常编译、照常运行，只是跑在一个线程上。

## 没有 std 时会变什么

有三样东西从运行期挪到编译期，或者从自动变成显式：

**特性探测变成编译期的。** 有 `std` 时，x86 分发调用 `is_x86_feature_detected!`，挑运行中 CPU 所报告的最优内核。没有 `std` 时就没有运行时 CPU 探测（那部分在受 `std` 门控的 `raw-cpuid` 里），于是 ISA 阶梯回退到 `cfg!(target_feature = ...)`：构建跑的是它编译期目标特性所保证的东西。想从 `no_std` 构建里得到加速的 x86 内核，你必须为之编译，比如用 `-C target-cpu=native` 或显式的 `-C target-feature=+avx512f`；否则你拿到的是标量兜底。在 aarch64 和 wasm 上，选择本就是这样工作的，所以那里没有任何损失。

**环境旋钮关闭。** 读取环境变量需要 `std`。没有它，`GEMMKIT_REQUIRE_ISA` 从不被查询（分发总是自动选择），每一个 `GEMMKIT_*` 调优旋钮都直接解析到它的编译期默认值。编程式的 `tuning::set_*` setter 仍然有效，所以你在代码里、而非通过环境去重调一个 `no_std` 构建。setter 这一层见[调优旋钮](调优旋钮.md)。

**每次调用的工作区取代线程池。** 默认的线程本地打包池是一个 `std` 构造。没有 `std` 就没有池，每次调用为其暂存分配一个新的 `Workspace`，返回时释放。这是正确的，但每次调用都分配；要拿到零分配的稳态，就创建一个 [`Workspace`](Unchecked层.md) 并把它穿过 `*_with` 入口（`gemm_with`，以及每个家族的 `_with` 变体）。第一次足够大的调用之后，这些入口会复用缓冲区，不再有进一步的堆流量。

## 为 WebAssembly 构建

wasm32 没有运行时特性探测，所以 `simd128` 后端由编译期 `cfg` 选定，而构建必须显式打开那个目标特性。如果你忘了它，wasm 构建会正确编译、正确运行，但跑在标量兜底上，慢好几倍。通过 `RUSTFLAGS` 传入这个标志：

```sh
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --target wasm32-wasip1 --no-default-features --features std
```

要运行产物，你需要一个 wasm 运行时；gemmkit 的 CI 用 `wasmtime`。把 Cargo 的目标 runner 指向它。当你想确认 SIMD 路径确实在跑、而不是悄悄用了标量时，就把 ISA 钉住：`GEMMKIT_REQUIRE_ISA=simd128` 会把缺失的 `+simd128` 变成一次 panic，而不是无声的回退——这正是测试作业想要的（这个 pin 需要 `std`，而 wasm 构建里 `std` 是开着的）。

```sh
RUSTFLAGS="-C target-feature=+simd128" \
CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime --env GEMMKIT_REQUIRE_ISA=simd128" \
  cargo test --target wasm32-wasip1 --no-default-features --features std
```

基线 `wasm32-wasip1` 没有线程。如果你为基线 wasm 目标开着 `parallel` 构建，gemmkit 不会 trap：一个内部守卫会让 rayon 在那里不可用，`Parallelism::Rayon(_)` 会降级到串行循环。所以一个可移植的 wasm 二进制可以带着 `parallel` feature，只是单线程运行，无需为目标专门构建。

## 带线程的 wasm

wasm 上真正的多线程需要支持线程的目标以及对应的 feature：

```sh
RUSTFLAGS="-C target-feature=+simd128" \
CARGO_TARGET_WASM32_WASIP1_THREADS_RUNNER="wasmtime -W threads=y -W shared-memory=y -S threads=y" \
  cargo test --target wasm32-wasip1-threads \
  --no-default-features --features std,parallel,wasm_threads
```

`wasm_threads` feature（它蕴含 `parallel`）面向 `wasm32-wasip1-threads`，并打开 gemmkit 专用的 wasm rayon 线程池。由于 wasm 运行时无法报告核数，池的宽度不是自动推导的；它来自 `GEMMKIT_WASM_THREADS` 旋钮，默认 8，它既给自动线程数封顶，也给池定尺寸。把它设成与你的运行时实际配备的工作线程数相符。其余一切——分块、工作清单、可复现性——的表现与原生带线程构建完全一致。
