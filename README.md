[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/README.md)

# gemmkit

[![CI](https://github.com/SomeB1oody/gemmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/SomeB1oody/gemmkit/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/gemmkit.svg)](https://crates.io/crates/gemmkit) [![docs.rs](https://img.shields.io/docsrs/gemmkit)](https://docs.rs/gemmkit)

A pure-Rust workspace for GEMM (general matrix multiply): it computes
`C <- alpha*A*B + beta*C` over strided views (or raw pointers) and picks the
best available instruction set at runtime.

The core engine works on `f32` and `f64` out of the box, and, behind Cargo
features, on `f16`/`bf16` (mixed precision with `f32` accumulation), `i8 -> i32`
integer, and `c32`/`c64` complex data. Runtime ISA dispatch covers x86-64 FMA and
AVX-512 (with AVX-512 VNNI for `int8` and AVX-512 BF16 for `bf16`), aarch64 NEON,
and wasm32 `simd128`, over a portable scalar fallback; the `GEMMKIT_REQUIRE_ISA`
environment variable pins or forbids a backend. Multithreading is optional
(rayon) and produces run-to-run reproducible results for a fixed input and
configuration. With default features off the core builds under `no_std` (only
`core` + `alloc`). Beyond plain GEMM it offers fused epilogues (bias, activation,
`i8`/`u8` requantization, and a user per-element map), prepacked-operand reuse for
fixed-weight inner loops, batched GEMM, and automatic bandwidth-bound paths for
matrix-vector and small shapes.

## Crates

The workspace ships five crates that share version 0.1.0 and release in lockstep.

| Crate | Description |
| --- | --- |
| [gemmkit](https://crates.io/crates/gemmkit) | Core GEMM engine: strided-view and raw-pointer entry points, runtime ISA dispatch, `no_std` support |
| [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray) | Zero-copy adapter over `ndarray` matrix views |
| [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra) | Zero-copy adapter over `nalgebra` matrix views |
| [gemmkit-faer](https://crates.io/crates/gemmkit-faer) | Zero-copy adapter over `faer` matrix views |
| [gemmkit-tune](https://crates.io/crates/gemmkit-tune) | Install-time autotuner binary: sweeps the runtime knobs on the target machine and emits a `GEMMKIT_*` env profile |

The adapters wrap `ndarray >= 0.17.1`, `nalgebra 0.35`, and `faer 0.24`, and
forward each `parallel` / `wasm_threads` / `half` / `complex` / `int8` /
`epilogue` feature to the same-named `gemmkit` feature.

## Quick start

```toml
[dependencies]
gemmkit = "0.1"
```

```rust
use gemmkit::{gemm, MatMut, MatRef, Parallelism};

fn main() {
    // 2x3 times 3x2 = 2x2, all row-major
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0_f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 3, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
    assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
}
```

Transposition is expressed through strides (`from_col_major`, or explicit `rs`/`cs`
in `MatRef::new`), so a transposed operand needs no copy.

## Element types and backends

Every element-type family below has a SIMD implementation on every backend, over
the scalar fallback that runs anywhere.

| Family | Feature | Accumulator |
| --- | --- | --- |
| `f32`, `f64` | (built in) | same type |
| `f16`, `bf16` | `half` | `f32` |
| `i8 -> i32` | `int8` | `i32` |
| `c32`, `c64` | `complex` | same type |

Backends, selected at runtime (or pinned with `GEMMKIT_REQUIRE_ISA`):

- Scalar: portable fallback, no target features required
- x86-64 FMA
- x86-64 AVX-512, with AVX-512 VNNI (`vpdpbusd`) for `int8` and AVX-512 BF16 (`vdpbf16ps`) for `bf16`
- aarch64 NEON
- wasm32 `simd128` (compile-time feature detection)

The `gemmkit` Cargo features are `std` and `parallel` (both default), `wasm_threads`
(for `wasm32-wasip1-threads`), `complex`, `half`, `int8`, and `epilogue` (fused
bias/activation, `i8`/`u8` requantization, and the user per-element map). With
`std` off the crate is `no_std`; `parallel` implies `std`.

## Documentation

- The [gemmkit Guide](https://someb1oody.github.io/gemmkit/en/): the full book
  (user guide, adapter guides, and an in-depth architecture walkthrough)
- API reference: [docs.rs/gemmkit](https://docs.rs/gemmkit)
- [ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md): the internal design and extension seams
- [CHANGELOG.md](https://github.com/SomeB1oody/gemmkit/blob/master/CHANGELOG.md): release notes

## License

Licensed under either of [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT)
or [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE)
at your option.

Minimum supported Rust version: 1.89 (edition 2024).
