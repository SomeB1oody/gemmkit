# Getting Started

gemmkit computes `C <- alpha*A*B + beta*C` over strided views of ordinary Rust slices, and picks the fastest instruction set your CPU actually has at runtime. There is no build-time ISA choice to make and no BLAS to link: you add one dependency, hand it three matrices, and call `gemm`.

## Adding the dependency

The core crate is `gemmkit`. For plain `f32`/`f64` work on a normal (std) target, that single line is all you need:

```toml
[dependencies]
gemmkit = "0.1"
```

This pulls in the two default features, `std` and `parallel`. `std` gives you runtime cache and CPU-feature detection, the `GEMMKIT_REQUIRE_ISA` and `GEMMKIT_*` tuning knobs, and a thread-local workspace pool that makes repeated same-size calls allocation-free. `parallel` adds rayon multithreading and implies `std`. The optional element-type families (`half`, `complex`, `int8`) and the `epilogue` capability are off by default, so a plain float build pays for none of their codegen or dependencies. To build the crate `no_std` (only `core` + `alloc`), turn the defaults off with `default-features = false`; see [no_std and WebAssembly](no_std_and_WebAssembly.md).

## A first complete example

A `2x3` times a `3x2`, all row-major, run single-threaded:

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

The arguments are exactly the terms of `C <- alpha*A*B + beta*C`: the scalar `alpha`, the two input views, the scalar `beta`, the output view, and a [`Parallelism`](Parallelism_in_Practice.md) selector. `MatRef::from_row_major(&a, 2, 3)` says "read `a` as a 2-by-3 row-major matrix"; the shapes have to line up (`A.cols == B.rows`, and `C` is `A.rows` by `B.cols`) or the call panics before touching memory. Transposition never needs a copy because it is a stride change, not a data move: `MatRef::from_col_major(&b, 3, 2)` reads the same buffer as a column-major matrix, and [`MatRef::new`](Matrix_Views_and_Layouts.md) lets you set the row and column strides directly.

## What happened under the hood

The `gemm` entry does a small amount of work before any arithmetic. First it validates: it checks that the inner dimensions agree, that each view stays inside its slice, that `C` addresses every `(i, j)` at a distinct offset, and that `C`'s storage does not overlap `A`'s or `B`'s. Any of those failing is a panic with a specific message, raised before a single unsafe operation runs. Only then does it lower the three views to raw pointers and strides and hand them to the dispatch layer.

Dispatch resolves which kernel to run. The very first GEMM call for a given element type runs CPU feature detection once, records the winning entry point in a `OnceLock`, and returns; every later call is a plain indirect call through that cached pointer, with no re-detection. So the runtime ISA choice is a one-time cost amortized across the whole process. You can override the automatic choice, or pin a specific backend for testing, with the `GEMMKIT_REQUIRE_ISA` environment variable, read once and memoized the same way; see [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md). The full path from call to microkernel is walked in [Life of a GEMM Call](../architecture/Life_of_a_GEMM_Call.md).

## alpha and beta, precisely

`alpha` scales the product `A*B`; `beta` scales the incoming contents of `C`. The one subtlety worth internalizing is what happens at the edges.

When `beta == 0`, `C` is **not read** at all: the engine overwrites it with `alpha*A*B`. That is what makes the `let mut c = [0.0_f32; 4]` above correct even though you could have left the buffer uninitialized. Concretely, a `beta == 0` output slice may hold garbage (or, through the unchecked tier, be genuinely uninitialized memory) and the result is still well-defined. When `beta == 1`, the existing `C` is left untouched and the product is accumulated onto it; any other `beta` multiplies `C` through first.

There is also a degenerate fast path. If `k == 0` (an empty contraction) or `alpha == 0` (the product vanishes), the call reduces to `C <- beta*C` and never touches `A` or `B` at all: it just scales the output in place. Combined with the `beta == 0` rule, `alpha == 0, beta == 0` zeroes `C`, and `k == 0, beta == 1` is a no-op. Narrow types scale in `f32` and round back on the store, so the degenerate path rounds exactly as the full kernel would.

## The Cargo features

| Feature | Default | Unlocks | Pulls in |
| --- | --- | --- | --- |
| `std` | yes | runtime cache/CPU detection, env knobs, thread-local workspace pool; off = `no_std` (`core` + `alloc`) | `raw-cpuid` (x86 only) |
| `parallel` | yes | rayon multithreading (`Parallelism::Rayon`); implies `std` | `rayon` |
| `wasm_threads` | no | a sized rayon pool for `wasm32-wasip1-threads`; implies `parallel` | (via `parallel`) |
| `half` | no | `f16`/`bf16` mixed-precision GEMM, `f32` accumulate | `half` |
| `complex` | no | `c32`/`c64` GEMM with conjugation (`gemm_cplx`) | `num-complex` |
| `int8` | no | `i8 -> i32` integer GEMM (`gemm_i8`) | (none) |
| `epilogue` | no | fused bias/activation, `i8`/`u8` requantization, per-element map | (none) |

The element-type and capability features compose: `half` + `epilogue` gives fused f16 GEMM, `int8` + `epilogue` gives the requantizing entries, and so on. Each is covered in [Element Types](Element_Types.md) and [Fused Epilogues](Fused_Epilogues.md).

## Version requirements

gemmkit targets **Rust 1.89** on **edition 2024**. It is licensed MIT OR Apache-2.0. The API reference is on [docs.rs/gemmkit](https://docs.rs/gemmkit); this book is the long-form companion.

## Where to next

- [Matrix Views and Layouts](Matrix_Views_and_Layouts.md) - how `MatRef`/`MatMut`, strides, transposition, and submatrices work, and exactly what the safe API validates.
- [Element Types](Element_Types.md) - `f16`/`bf16`, `i8`, and complex, with their accuracy characteristics.
- [Parallelism in Practice](Parallelism_in_Practice.md) - what `Rayon(0)` auto actually does and when `Serial` is the right call.
- [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md) and [The Unchecked Tier](The_Unchecked_Tier.md) - overriding the backend, and the raw-pointer engine.
- Working with an existing array library? See the adapters: [ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md), [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md), [faer](../gemmkit-faer/Using_gemmkit_with_faer.md).
