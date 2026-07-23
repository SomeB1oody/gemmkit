[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/fuzz/README.md)

# gemmkit fuzzing harness

gemmkit 的 [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)（libFuzzer + AddressSanitizer）模糊测试（fuzzing）框架。**仅限 nightly**（需要 `-Z build-std` 与 sanitizer），且**被排除在稳定版工作空间之外** —— 它是自成一体的工作空间根（`Cargo.toml` 中带 `[workspace]` 表），拥有独立的 `Cargo.lock` 与 `target/`，因此 `cargo test/clippy/fmt --workspace` 以及 MSRV-1.89 构建都不会碰到它。

这里的内容都不会提交到 git（`corpus/`、`artifacts/`、`target/`、`Cargo.lock` 均已 `.gitignore`）。

## 前置条件

```sh
rustup toolchain install nightly           # provides rust-src for build-std
rustup component add rust-src --toolchain nightly
cargo install cargo-fuzz --locked          # 0.13.2 verified
```

下文所有命令都要**在本目录下**（`gemmkit/fuzz`）运行，并显式带上 `+nightly`（项目规则禁止全局默认使用 nightly，因此这里刻意不放 `rust-toolchain.toml`）。

### 环境变量卫生（可复现性）

`GEMMKIT_*` 环境变量每个旋钮只解析一次并被缓存，`GEMMKIT_REQUIRE_ISA` 则按进程记忆化。在任何一次运行之前：

```sh
env | grep '^GEMMKIT_'    # must print nothing, except a deliberate ISA pin (below)
```

若导出了调优得到的配置文件，会悄悄扭曲 `fuzz_gemm`/`fuzz_batched`/`fuzz_prepack`，并让崩溃产物在别处无法复现。（`fuzz_knobs` 不受影响 —— 它对每个输入都无条件设置所有旋钮，优先级高于环境变量。）

## 五个 fuzz 目标

| target | what it fuzzes | panic policy |
|---|---|---|
| `fuzz_gemm` | 合法的 `gemm`/`gemm_i8`/`gemm_cplx`（f32/f64/f16/bf16/i8/c32/c64），涵盖所有布局（含广播 A/B）、`beta==0` 时的 NaN-C 契约、可选的调用方 `Workspace` 复用；并与 f64/i32/复数参考实现做差分对比 | 任何 panic 都是 bug |
| `fuzz_knobs` | 每个输入都把全部 22 个进程级全局调优旋钮设为对抗性取值类别，然后运行一个小场景（plain / gemv / small-mn / prepack-B / prepack-A / i8 / batched）；算术溢出的主要发现者 | 任何 panic 都是 bug |
| `fuzz_api_validation` | 把对抗性维度（含 `2^33`、`usize::MAX`）与 `isize` 步长（含 `isize::MIN/MAX`）喂给**带校验的** `gemm`/`gemm_i8`/`gemm_cplx`/`gemm_batched`/`prepack_*` 入口 | 已文档化的 `gemmkit:` 前缀 panic 可接受；其余一律是 bug |
| `fuzz_batched` | 合法的跨步批量 `gemm_batched`（广播 A/B、合法的批次步长）与 `gemm_batched_slice`；逐元素差分对比 | 任何 panic 都是 bug |
| `fuzz_prepack` | `prepack_rhs`→`gemm_packed_b` 与 `prepack_lhs`→`gemm_packed_a` 的往返（f32/f64/bf16）；与参考实现按容差（而非逐位）比较 | 任何 panic 都是 bug |

每个使用合法输入的目标还会在 C 的底层缓冲区中，向非视图（交错 / 填充 / 元素间）位置写入**金丝雀哨兵值**，并在调用结束后断言它们未被改动 —— 即使越界写没有落在 ASan 可见的边界上，也能被暴露出来。

`fuzz_api_validation` 在 `catch_unwind` 下运行，并安装静默的 panic hook；它把带 `gemmkit:` 前缀的 panic 视为可接受的拒绝，对其他任何 panic（索引越界、算术溢出等）调用 `abort()`（即真实发现）。它只跳过那些*本会*完全通过校验、随后执行无界计算的方案（`WORK_CAP` 为 2^24 次 MAC / 某个维度巨大 / 批次循环巨大）—— 所有拒绝路径都仍在完整 fuzz 之列。

## 冒烟测试（每个目标约 45–60 秒 —— CI 规模）

```sh
for t in fuzz_gemm fuzz_knobs fuzz_api_validation fuzz_batched fuzz_prepack; do
  cargo +nightly fuzz run "$t" -- \
    -max_total_time=45 -max_len=512 -timeout=60 -malloc_limit_mb=1024 -print_final_stats=1
done
```

`-malloc_limit_mb=1024`：在这些极小的维度上出现 >1 GB 的单次分配，*本身就是*旋钮健壮性的 bug —— 应把这类崩溃产物当作真实发现，只有在定位分析证明其无害后才提高上限。`-timeout=60`：一个通过校验但退化为巨大维度 / 巨大批次而空转的输入会表现为超时 —— 同样按真实发现定位分析。

## 长时间浸泡测试（通宵运行，交由用户执行）

```sh
# process-parallel, shared corpus (prefer -jobs/-workers over -fork under ASan+threads)
cargo +nightly fuzz run fuzz_knobs -- \
  -max_total_time=14400 -max_len=512 -timeout=60 -malloc_limit_mb=1024 -jobs=4 -workers=4

# per-ISA passes: the dispatch pin is once-per-process, so use SEPARATE processes
for isa in scalar fma avx512f avx512vnni avx512bf16; do
  GEMMKIT_REQUIRE_ISA=$isa \
    cargo +nightly fuzz run fuzz_gemm -- -max_total_time=3600 -max_len=512 -timeout=60
done

# corpus maintenance afterwards
cargo +nightly fuzz cmin fuzz_gemm
```

`corpus/<target>/` 中的语料库会自动创建，并随多次运行不断增长；无需手工准备种子（方案由 `int_in_range` 驱动，因此即使是空输入也能解码出一个最小的合法方案）。

### 基于累积语料库的覆盖率报告

`cargo fuzz coverage` 默认以 `--build-std=false` 构建（组合起来更干净），随后渲染一份 llvm-cov 覆盖率报告，用于找出方案从未触及的分发路由 / 场景：

```sh
cargo +nightly fuzz coverage fuzz_gemm
# then render (the exact profdata/binary paths are printed by the command):
$(rustc +nightly --print target-libdir)/../bin/llvm-cov show \
  target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/fuzz_gemm \
  -instr-profile=coverage/fuzz_gemm/coverage.profdata \
  -Xdemangler=rustfilt -format=html > /tmp/fuzz_gemm-cov.html
```

## 崩溃 → 最小化 → Miri 重放 → 稳定版回归测试

当某个目标崩溃时，libFuzzer 会写出 `artifacts/<target>/crash-<sha>`（或 `timeout-…`/`oom-…`）并打印方案的 `Debug` 输出。

1. **复现**
   ```sh
   cargo +nightly fuzz run <target> artifacts/<target>/crash-<sha>
   ```
2. **最小化输入**
   ```sh
   cargo +nightly fuzz tmin <target> artifacts/<target>/crash-<sha>
   ```
3. **解码为测试参数。** 由于每个方案保存的都是*已解析*的取值（手写的 `Arbitrary`），其 `Debug` 输出就是维度 / 步长布局 / alpha-beta 索引 / 旋钮数组 / 并行度本身：
   ```sh
   cargo +nightly fuzz fmt <target> artifacts/<target>/<minimized>
   ```
4. **（可选）Miri 重放。** ASan 会漏掉 Miri 能捕获的未初始化读取 / provenance 类 bug，而 gemmkit 的内核保持了 Miri 兼容性（`tests/correctness.rs` 中的 `cfg(miri)` 路径）。把解码出的方案改写成一个极小的 `#[test]`，并**在稳定版工作空间中**用 Miri 运行（切勿依赖这个仅限 nightly 的 fuzz crate）：
   ```sh
   cargo +nightly miri test -p gemmkit --test <file> <testname>
   ```
5. 在 `gemmkit/tests/` 中**手写一个与平台无关的稳定版回归测试**（断言行为，绝不断言机器相关的常数）：
   - 旋钮类崩溃 → `tests/tuning.rs`，持有 `knob_guard()` 并恢复所有被改动的旋钮（`KNOB_LOCK` 模式）；
   - 环境变量契约崩溃 → 仿照 `tests/env.rs` 新建一个「每个二进制一个测试」的文件；
   - 形状 / 校验类崩溃 → `tests/correctness.rs`（若某个校验缺口被提升为已文档化的 panic，则用 `#[should_panic]` —— 先例：`panic_extent_overflow_view`），或新建 `tests/fuzz_regressions.rs`。
6. **在稳定版上验证：**
   ```sh
   cargo test -p gemmkit --all-features --test <file>
   ```
   并确认该 fuzz 目标在这个崩溃产物上不再崩溃。

## 计算量上限策略（`fuzz_api_validation`）

prepack 入口只跳过那些打包结果*可表示但极其庞大*的方案（元素个数能放进 `usize`，却超过了 `WORK_CAP`）—— 即使行为完全正确，运行它们也会 OOM。空操作数（prepack 会短路）以及打包尺寸溢出 `usize` 的情况（已文档化的 `gemmkit: … too large` 拒绝）仍在 fuzz 之列；这一类溢出的回归测试位于 `gemmkit/tests/props_packed.rs`（`prepack_*`）与 `gemmkit/tests/props_api.rs`（`mixed_huge_k_fails_closed`）。
