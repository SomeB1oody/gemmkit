[简体中文](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/README.zh-CN.md) | [English](https://github.com/SomeB1oody/gemmkit/blob/master/gemmkit/README.md)

# gemmkit

[![crates.io](https://img.shields.io/crates/v/gemmkit.svg)](https://crates.io/crates/gemmkit) [![docs.rs](https://img.shields.io/docsrs/gemmkit)](https://docs.rs/gemmkit)

gemmkit is a general matrix multiply (GEMM) engine that computes
`C <- alpha*A*B + beta*C` for f32 and f64 over strided `&[T]` views, with no
dependency on any matrix library. The best available instruction set is selected
at runtime, and a portable scalar path covers targets with no vector backend.
Transposition is expressed through strides (a transposed view swaps its row and
column stride, no copy), and when `beta == 0` the output `C` is not read, so it
may be uninitialized.

The entry point, `gemm`, takes checked `MatRef`/`MatMut` slice views and panics
on a shape, bounds, or aliasing error before running any unsafe code. Two further
API tiers trade checks for control: the `*_with` variants take a caller-owned
`Workspace` to avoid per-call allocation, and the `*_unchecked` entries operate
on raw pointers and `isize` strides (negative strides included) for callers that
validate their own inputs.

Beyond the plain product, gemmkit provides:

- Runtime ISA dispatch: x86-64 FMA and AVX-512 (AVX-512 VNNI for int8, AVX-512
  BF16 for bf16), aarch64 NEON, wasm32 simd128 (compile-time feature detection),
  and a scalar fallback. The `GEMMKIT_REQUIRE_ISA` env var pins or forbids a
  backend.
- Optional element families behind cargo features: f16/bf16 with f32
  accumulation, i8 to i32, and c32/c64 with per-operand conjugation.
- Prepacked operands: `prepack_rhs`/`prepack_lhs` build a reusable packed buffer
  that `gemm_packed_b`/`gemm_packed_a` consume, for products that share a fixed
  operand.
- Batched GEMM (`gemm_batched`) over an array of independent problems.
- Fused epilogues behind the `epilogue` feature: `gemm_fused` (bias and
  activation), `gemm_i8_requant` (integer requantization), and `gemm_map` (a
  user per-element closure).
- Automatic special paths for bandwidth-bound shapes (gemv, small-k, and small
  m,n), selected behind the same entry points.
- rayon parallelism with results reproducible for a fixed input and config, plus
  `no_std` operation when the default features are off (needs only `core` and
  `alloc`).

## Usage

```toml
[dependencies]
gemmkit = "0.1"
```

```rust
use gemmkit::{gemm, MatMut, MatRef, Parallelism};

fn main() {
    // A 2x3 times a 3x2, both row-major, into a 2x2 result
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c: Vec<f32> = vec![0.0; 4];
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

## Feature flags

- `std` (default): runtime CPU-feature and cache detection, the
  `GEMMKIT_REQUIRE_ISA` and tuning env knobs, and the thread-local workspace
  pool. With it off the crate is `no_std`, needing only `core` and `alloc`.
- `parallel` (default, implies `std`): rayon multithreading. With it off,
  everything compiles and runs single-threaded.
- `wasm_threads`: rayon parallelism on the `wasm32-wasip1-threads` target.
- `complex`: c32/c64 complex GEMM with optional conjugation of A or B
  (`gemm_cplx`); pulls in `num-complex`.
- `half`: f16/bf16 mixed-precision GEMM with f32 accumulation; pulls in `half`.
- `int8`: i8 to i32 integer GEMM (`gemm_i8`); arithmetic wraps on overflow.
- `epilogue`: fused epilogues (bias and activation, per-element map, and, with
  `int8`, i8/u8 requantization). Off by default, so a plain-GEMM build pays for
  none of its codegen.

## Tuning

Every heuristic threshold resolves per-call argument, then programmatic setter,
then a `GEMMKIT_*` env var, then a compile-time default. The
[gemmkit-tune](https://crates.io/crates/gemmkit-tune) binary sweeps these knobs
on the target machine and emits a ready-to-source env profile; the individual
knobs are documented on [docs.rs](https://docs.rs/gemmkit).

## Related crates

- [gemmkit-ndarray](https://crates.io/crates/gemmkit-ndarray): zero-copy adapter
  over `ndarray` matrix views.
- [gemmkit-nalgebra](https://crates.io/crates/gemmkit-nalgebra): zero-copy
  adapter over `nalgebra` matrix views.
- [gemmkit-faer](https://crates.io/crates/gemmkit-faer): zero-copy adapter over
  `faer` matrix views.
- [gemmkit-tune](https://crates.io/crates/gemmkit-tune): install-time autotuner
  binary.

For the engine design, see
[ARCHITECTURE.md](https://github.com/SomeB1oody/gemmkit/blob/master/ARCHITECTURE.md).

## Minimum supported Rust version

gemmkit requires Rust 1.89 or newer.

## License

Licensed under either of [MIT](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-MIT)
or [Apache-2.0](https://github.com/SomeB1oody/gemmkit/blob/master/LICENSE-APACHE),
at your option.
