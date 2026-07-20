# 使用gemmkit-tune调优

`gemmkit-tune` 是一个小巧的命令行自动调优器。你在将要运行 gemmkit 工作负载的那台机器上运行它一次，它便会扫描 gemmkit 的运行时 `GEMMKIT_*` 旋钮，为每个旋钮测量一组有代表性的矩阵形状，并写出一个由 `export GEMMKIT_*=...` 行组成的 shell 配置文件。在启动 gemmkit 二进制前 source 这个文件，已经编译好的二进制就会针对当前主机重新调优——不用重新编译，不用改代码，也不依赖二进制是怎么分发过来的。

## 它为什么存在

gemmkit 编译期内置的默认值并非凭空而来。[`gemmkit::tuning`](https://docs.rs/gemmkit) 里的每一个阈值都是依据真实测量手工校准的——但那些测量来自同一台机器，一颗 Ryzen 9950X（Zen5）。在那台机器上最优的取值，编码的是它的缓存大小、核心数和访存带宽与算力之比。换一颗 CPU，L2 不同、核心数不同、内存带宽不同，各个转折点也就落在了别处：打包开始划算的那个 `k`、共享预打包胜过每个 worker 各自打包的那个问题规模、带宽受限的 gemv 应当保持单线程的那个字节下限。`gemmkit-tune` 会在眼前这块芯片上重新找出这些转折点并把它们钉死。

由此有一个很有用的推论。在一台 9950X 上运行本工具，它基本会重新选出出厂默认值；报告会显示每个旋钮都保持了默认，整体加速比约为 1.0x。这是正确的结果，而不是令人失望的结果——它验证了工具本身。真正的收益出现在与校准机器*不同*的主机上：一台笔记本、一台共享插槽的云实例、一颗宽核的 Graviton、一颗带共享 cluster-L2 而没有 L3 的 Apple 芯片。部署主机离 9950X 越远，可争取的空间就越大。

扫描过程不含任何随机性。给定同一台机器和同一组标志，它每次都产出相同的配置文件，因此一份配置文件就是一个可复现的产物，可以随部署一起提交进版本库。

## 为什么在部署主机上运行，而绝不放进 build.rs

这些旋钮是针对*执行*计算的那颗 CPU 校准的，所以工具必须在那里运行。由此有两点。

其一，不要放进 `build.rs`。构建主机通常并非部署主机——你在 CI runner 或开发者笔记本上编译，再把二进制发到完全不同的机器上——而 `build.rs` 里的自动调优测量的是构建机，随后会把构建机的转折点烤进一个将在缓存层级截然不同的机器上运行的二进制里。更糟的是，交叉编译的构建根本无法执行目标机的代码。整件事的要点就是测量真实的芯片，这意味着必须在其上运行。

其二，由于这些旋钮就是进程启动时读取一次的普通运行时环境变量，应用一份配置文件并不需要重新构建。同一个发布出来的二进制会读取它所处环境里的任何 `GEMMKIT_*` 取值。调优主机、source 配置文件、启动——引擎便会自行重新配置。

## 安装与首次运行

安装该二进制并在目标机器上运行：

```sh
cargo install gemmkit-tune
gemmkit-tune
```

一次完整运行需要一两分钟，过程中会打印报告。默认情况下它会在当前目录写入 `gemmkit-tune.env`。在启动应用的 shell 中 source 它：

```sh
source gemmkit-tune.env
./your-gemmkit-app
```

整个流程就这些。下面都是细化：如何限定运行时长、如何配合你的部署方式、以及何时需要再做一次。

## 各个标志

`gemmkit-tune` 不接受位置参数，全部行为都由五个标志控制。

### `--threads <n>`

按此工作线程数进行调优。每个并行探测都在 `Parallelism::Rayon(n)` 下运行，因此调度类旋钮（各个 oversample 因子、自动 worker 数爬坡）都是针对该宽度优化的。默认值是机器的可用并行度。它会被机器宽度封顶——你无法针对多于机器实有核心数的 worker 调优，且标注的 worker 数始终真实。

经验法则：用你的应用实际会使用的 worker 数来跑扫描。如果你的应用固定用 `Parallelism::Rayon(8)`，就传 `--threads 8`；针对 32 个 worker 调优的配置文件，其调度粒度可能与针对 8 个的不同，而这种错配是要付出代价的。

### `--time-budget <dur>`

限定扫描时长并相应放粗。可写成 `30s`、`2m`、`1h`，或直接给出秒数。在预算之下，工具会为每次估算减少计时重复次数（无预算时为 7 次，90 秒以下为 5 次，30 秒以下为 3 次），并且一旦超过截止时刻就停止扫描，把剩余旋钮列为因“time budget exhausted”而跳过。当安装时长必须受限时用它；想要最可靠的配置文件就不要设。若预算小到连一个旋钮都测不完，报告会明说并提示你调大它。

### `--large-matrices <GiB>`

选择性地探测两个内存开销大的旋钮 `GEMMKIT_K_STREAM_MAX` 和 `GEMMKIT_SHARED_LHS_MNK`，以给定的 GiB 数作为大型 gemv 矩阵的预算。这两个旋钮只有在开销昂贵的区间才起作用——一个需要 gemv 输出溢出末级缓存（数 GB 的矩阵），另一个需要越过共享预打包转折点的极高 FLOP 形状——所以默认关闭。如果你给的预算装不下所需探测，工具会干净地跳过该旋钮，并打印出应当重跑时使用的确切 GiB 数。其中 `GEMMKIT_K_STREAM_MAX` 探测仅在 64 位下进行；`GEMMKIT_SHARED_LHS_MNK` 在 32 位上照常扫描。先从 `4` 或 `8` 起步，若它要求更多就照办。这两个探测究竟做什么，见[深入扫描过程](深入扫描过程.md)。

### `--out <path>`

把配置文件写到 `./gemmkit-tune.env` 以外的位置。

### `--dry-run`

跑完整扫描并打印报告，但不写出配置文件。适合在落盘之前预览一台机器会选出什么。`-h` / `--help` 打印用法。

## 生成的配置文件剖析

文件由一段头部注释、每个被扫描旋钮一行 `export`、以及一段列出未扫描项的尾注组成。它长这样（取值完全取决于主机）：

```sh
# gemmkit-tune profile — source before running a gemmkit app: `source <this file>`
# generated 2026-07-19 14:12:03 UTC by gemmkit-tune 0.1.0
# host: 16 logical cores; L1d 32 KiB, L2 1024 KiB, L3 32 MiB; page 4 KiB
# tuned for 16 worker(s)

export GEMMKIT_MC_REG_PANELS=8  # default (1.00x)
export GEMMKIT_LHS_PACK_THRESHOLD=256  # tuned (1.07x)
export GEMMKIT_PAR_MNK_PER_WORKER=4000000  # tuned (1.03x)

# not swept on this host:
#   GEMMKIT_PARALLEL_THRESHOLD: serial/parallel break-even is strongly shape-dependent ...
#   GEMMKIT_DEEP_KC_BYTES: narrow-only (f16/bf16 deep-contraction twin); no narrow probe here ...
```

头部记录了它所调优的主机：逻辑核心数、三级缓存、页大小、worker 数，以及工具版本和 UTC 时间戳。几个月后，正是这段标注告诉你一份配置文件是否还与你手上这台机器相符。每一行 `export` 都带一段行尾注释，标为 `default`（胜出值等于出厂默认）或 `tuned`（胜出值发生了移动），并附上相对默认的实测加速比。保持默认的旋钮也照样写出，所以配置文件是对每个决策的完整、自带说明的记录，而不只是差异项。

取值一律是原始整数——“无上界”的胜出值会写成它的数值，绝不写成 `MAX` 别名——因为 gemmkit 的环境变量解析器读的是普通十进制整数。格式错误的 `GEMMKIT_*` 取值不会致命：gemmkit 会在 stderr 上警告一次并回退到编译期默认，所以手工编辑出的笔误只会退化为默认，而不会崩溃。

## 部署这份配置文件

gemmkit 对每个 `GEMMKIT_*` 变量只在首次访问时读取一次，随后在进程整个生命周期内缓存。因此配置文件必须在第一次 GEMM 调用*之前*就位于环境中——在启动前 source 恰好能保证这一点。这件事通常有三种形态。

**shell 配置或启动脚本。** 最直接的情形：在运行二进制的那个 shell 里、或在服务的启动脚本里 `source gemmkit-tune.env`。完整的 shell 语义适用，所以文件原样即可用。

**容器 entrypoint。** 把配置文件打进镜像，并在 entrypoint 里 `exec` 你的应用之前 source 它，这样每个容器一启动就已调优就绪。要在与容器运行时硬件相符的主机上调优，而不是在镜像构建机上。

**systemd `EnvironmentFile`。** 这可行，但有一个注意点：systemd 的 `EnvironmentFile` 解析器要的是裸的 `NAME=value` 行——它不认识 `export` 关键字，也不认识行尾的 `# tuned (...)` 注释。先转换配置文件，例如 `grep '^export' gemmkit-tune.env | sed -e 's/^export //' -e 's/[[:space:]]*#.*$//' > gemmkit.env`，再让 `EnvironmentFile=` 指向结果。`#` 注释头部留着无妨，只有赋值行需要这一步转换。

一点优先级说明：`GEMMKIT_*` 环境变量会被应用中的 `tuning::set_*` 程序化调用覆盖。如果你的应用在代码里调优了某个旋钮，配置文件就不会改动该旋钮（这是刻意为之——自调优的代码胜过部署配置）。希望配置文件生效的应用只要不调用这些 setter 即可。完整的优先级次序见[调优旋钮](../gemmkit-guide/调优旋钮.md)。

## 在干净的环境里运行

调优所在 shell 里已经设置的任何 `GEMMKIT_*` 变量都会扭曲扫描，因为 gemmkit 在测量基准时会读取它。工具会中和它所扫描的那些旋钮，并在发现有 `GEMMKIT_*` 被设置时向你警告，但可靠的做法是从一个没有任何这类变量的 shell 里调优。不要 source 一份旧的 `gemmkit-tune.env` 之后又在同一个 shell 里重跑工具——那正是警告所指的被污染基准。

## 何时需要重新调优

当配置文件所标注的对象发生变化时就要重调：换了部署机器（CPU 不同、缓存大小不同——这正是工具存在的全部理由），或换了 worker 数（针对 8 个 worker 调优的配置文件不适用于 32 个）。gemmkit 或 gemmkit-tune 的版本升级也可能新增旋钮，所以升级后要重新生成。一份不再与其头部标注相符的配置文件，就是一份该丢弃并重新生成的配置文件。

要理解扫描究竟在测量什么、如何为候选打分、以及如何拿一份配置文件对照你自己的工作负载做校验，请读[深入扫描过程](深入扫描过程.md)。
