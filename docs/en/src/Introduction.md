# Introduction

gemmkit is a pure-Rust workspace for GEMM, the general matrix multiply `C <- alpha*A*B + beta*C`. The core crate works over strided views or raw pointers, selects the fastest instruction set available on the machine at runtime, and keeps its results reproducible from run to run under a fixed input and configuration. Around that core sit three zero-copy adapters, one each for `ndarray`, `nalgebra`, and `faer`, and an install-time autotuner that calibrates the engine for the machine it will actually run on.

This book is the narrative documentation for the whole workspace. The API reference on [docs.rs](https://docs.rs/gemmkit) remains the authority on exact signatures and item-level details; the book is where the pieces get explained in context, with room for the reasoning, the trade-offs, and the corners of the API that a reference page cannot do justice to.

## What is in the book

The **gemmkit user guide** covers the core crate, from the first multiply to the parts most users never need: matrix views and layouts, the optional element types (`f16`/`bf16`, `i8`, complex), parallel execution, prepacked operands, fused epilogues, batched GEMM, instruction-set pinning, the tuning knobs, `no_std` and WebAssembly builds, and the unchecked raw-pointer tier.

The **adapter guides** show how to drive the engine straight from `ndarray`, `nalgebra`, and `faer` types with no copies. Each adapter gets a chapter with a getting-started page and an advanced page covering the full surface: fused operations, integer and complex GEMM, batching, and prepacking, all in the host library's native types.

The **gemmkit-tune guide** explains the autotuner: how to run it on a deployment machine, what the emitted profile contains, and how the sweep behind it actually works.

The **architecture chapter** walks through the inside of the engine, layer by layer: how a call travels from the public API down to the microkernel, how instruction sets and element types stay pluggable without macros, how blocking is derived from the cache hierarchy, and how the whole thing is tested. It is a more detailed, more approachable companion to the compact [ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md) in the repository, written to be read front to back.

## How to read it

If you just want fast matrix multiplication in an application, start with [Getting Started](gemmkit-guide/Getting_Started.md) and read the user guide as far as your use case demands. If your matrices already live in `ndarray`, `nalgebra`, or `faer`, jump straight to that adapter's chapter and fall back to the user guide when you need the underlying concepts, since the adapters forward to the same engine and share its semantics.

If you are curious how the engine works, or you plan to contribute, the architecture chapter is the intended path. It assumes you have skimmed the user guide but not that you know BLIS, and it explains the design decisions rather than just describing the code.

## Conventions and resources

Code examples target Rust edition 2024 and the workspace MSRV of 1.89. Examples that need an optional Cargo feature say so where they appear. Repository paths like `gemmkit/src/driver.rs` are relative to the [repository root](https://github.com/SomeB1oody/gemmkit).

Related resources: the [API reference](https://docs.rs/gemmkit), the [CHANGELOG](https://github.com/SomeB1oody/gemmkit/blob/master/CHANGELOG.md), and the crates on crates.io ([gemmkit](https://crates.io/crates/gemmkit), [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray), [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra), [gemmkit-faer](https://crates.io/crates/gemmkit-faer), [gemmkit-tune](https://crates.io/crates/gemmkit-tune)).

本书也有[简体中文版](https://someb1oody.github.io/gemmkit/zh-Hans/)。
