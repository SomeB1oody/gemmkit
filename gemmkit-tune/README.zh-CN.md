[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-tune/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit-tune/README.md)

# gemmkit-tune

[![crates.io](https://img.shields.io/crates/v/gemmkit-tune.svg)](https://crates.io/crates/gemmkit-tune)

面向 [`gemmkit`](https://crates.io/crates/gemmkit) GEMM 引擎的安装期自动调优器。它在将要运行你的 gemmkit 应用的那台机器上执行，为每个 `GEMMKIT_*` 运行时调优旋钮测量一组有代表性的矩阵形状，以此扫描各旋钮取值，并输出一个由 `export GEMMKIT_*=...` 行组成的环境变量配置文件。启动 gemmkit 二进制前 source 该文件，即可无需重新编译地针对当前主机重新调优。

请在部署机器上运行它，而不是放进 `build.rs`：构建主机通常并非部署主机，而这些旋钮是针对真正执行计算的 CPU 校准的。gemmkit 编译期内置的默认值是在某一台机器（Ryzen 9950X）上校准的，因此在同一台机器上运行基本只会重新得到相同的取值；真正的收益出现在换一台机器时。

## 用法

安装该二进制并在目标机器上运行：

```sh
cargo install gemmkit-tune
gemmkit-tune
```

默认情况下，它会在当前目录写入一个 `gemmkit-tune.env` 文件。在启动应用的 shell 中 source 它：

```sh
source gemmkit-tune.env
./your-gemmkit-app
```

生成的文件是一组 `export GEMMKIT_*=<value>` 行，并带有一段头部注释，记录调优所针对的主机和工作线程数。gemmkit 在运行时读取这些变量，因此无需重新构建。该工具还会在终端打印一份报告，逐个旋钮列出扫描的默认值、最终选定的取值，以及哪些旋钮被跳过、为何被跳过。

调优所在的 shell 中不应设置任何 `GEMMKIT_*` 变量：它们会影响测得的基准，工具在发现这类变量时会发出警告。

## 选项

`gemmkit-tune` 不接受位置参数。可用的标志如下：

- `--threads <n>`：按该工作线程数进行调优（默认：可用并行度）。
- `--time-budget <dur>`：限制扫描时长并相应放粗测量精度，可写成 `30s`、`2m`、`1h`，或直接给出秒数。
- `--large-matrices <GiB>`：额外探测内存开销较大的旋钮（`GEMMKIT_K_STREAM_MAX`、`GEMMKIT_SHARED_LHS_MNK`），大型 gemv 探测矩阵最多使用给定的 GiB 预算。默认关闭。
- `--out <path>`：输出配置文件的路径（默认：`gemmkit-tune.env`）。
- `--dry-run`：只打印报告，不写出配置文件。
- `-h`、`--help`：打印用法说明。

## Feature 标志

无。`gemmkit-tune` 是一个二进制 crate，自身不提供任何 Cargo feature；它在构建时会启用 `gemmkit` 的 `complex`、`half` 和 `int8` 类型族，以便探测每一种元素类型。

## 相关 crate

- [`gemmkit`](https://crates.io/crates/gemmkit)：本工具所调优的运行时旋钮所属的 GEMM 引擎。生成的配置文件同样会影响零拷贝适配器 crate（[`gemmkit-ndarray`](https://crates.io/crates/gemmkit-ndarray)、[`gemmkit-nalgebra`](https://crates.io/crates/gemmkit-nalgebra)、[`gemmkit-faer`](https://crates.io/crates/gemmkit-faer)），因为它们都转发到同一个引擎。

## 许可

采用 [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT) 或 [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE) 双许可，由你任选其一。

最低支持的 Rust 版本：1.89。
