# Introduction

gemmkit is a pure-Rust workspace for GEMM, the general matrix multiply `C <- alpha*A*B + beta*C`. The core crate works over strided views or raw pointers. It selects the fastest instruction set available on the machine at runtime. Under a fixed input and configuration, it keeps its results reproducible from run to run. Around that core sit 3 zero-copy adapters, one each for `ndarray`, `nalgebra`, and `faer`. An install-time autotuner calibrates the engine for the machine it will actually run on.

This book is the narrative documentation for the whole workspace. The API reference on [docs.rs](https://docs.rs/gemmkit) remains the authority on exact signatures and item-level details. The book explains the pieces in context: the reasoning, the trade-offs, and the corners of the API that a reference page cannot cover in depth.

## What is in the book

The **gemmkit user guide** covers the core crate. It starts with the first multiply and goes on to parts most users never need:

- matrix views and layouts
- the optional element types (`f16`/`bf16`, `i8`, complex)
- parallel execution
- prepacked operands
- fused epilogues
- batched GEMM
- small shapes and GEMV
- instruction-set pinning
- the tuning knobs
- `no_std` and WebAssembly builds
- the unchecked raw-pointer tier

The **adapter guides** show how to drive the engine straight from `ndarray`, `nalgebra`, and `faer` types with no copies. Each adapter gets a chapter with a getting-started page and an advanced page. The advanced page covers the full surface: fused operations, integer and complex GEMM, batching, and prepacking, all in the host library's native types.

The **gemmkit-tune guide** explains the autotuner. It covers how to run the autotuner on a deployment machine, what the emitted profile contains, and how the sweep behind it works.

The **architecture chapter** walks through the inside of the engine, layer by layer. It covers how a call travels from the public API down to the microkernel, and how instruction sets and element types stay pluggable without macros. It also covers how blocking is derived from the cache hierarchy, and how the whole thing is tested. It is a more detailed, more approachable companion to the compact [ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md) in the repository, written to be read front to back.

## How to read it

If you just want fast matrix multiplication in an application, start with [Getting Started](gemmkit-guide/Getting_Started.md). Read the user guide as far as your use case demands.

If your matrices already live in `ndarray`, `nalgebra`, or `faer`, jump straight to that adapter's chapter. When you need the underlying concepts, fall back to the user guide. The adapters forward to the same engine and share its semantics.

If you are curious how the engine works, or you plan to contribute, the architecture chapter is the intended path. It assumes you have skimmed the user guide, but not that you know BLIS. It explains the design decisions, not just the code.

## Conventions and resources

Code examples target Rust edition 2024 and the workspace MSRV of 1.89. Examples that need an optional Cargo feature say so where they appear. Repository paths like `gemmkit/src/driver.rs` are relative to the [repository root](https://github.com/SomeB1oody/gemmkit).

Related resources include the [API reference](https://docs.rs/gemmkit), the [CHANGELOG](https://github.com/SomeB1oody/gemmkit/blob/master/CHANGELOG.md), and the crates on crates.io: [gemmkit](https://crates.io/crates/gemmkit), [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray), [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra), [gemmkit-faer](https://crates.io/crates/gemmkit-faer), [gemmkit-tune](https://crates.io/crates/gemmkit-tune).

本书也有[简体中文版](https://someb1oody.github.io/gemmkit/zh-Hans/)。
