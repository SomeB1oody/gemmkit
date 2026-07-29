# Getting Started

gemmkit computes `C <- alpha*A*B + beta*C` over strided views of ordinary Rust slices.
It picks the fastest instruction set your CPU actually has, at runtime. There is no
build-time ISA choice to make and no BLAS to link. You add 1 dependency, hand it 3
matrices, and call `gemm`.

## Adding the dependency

The core crate is `gemmkit`. For plain `f32`/`f64` work on a normal (std) target, this
single line is all you need:

```toml
[dependencies]
gemmkit = "0.1"
```

This pulls in the 2 default features, `std` and `parallel`. `std` gives you runtime
cache and CPU feature detection. It also gives you the `GEMMKIT_REQUIRE_ISA` and
`GEMMKIT_*` tuning knobs, and a thread-local workspace pool. The pool makes repeated
same-size calls allocation-free. `parallel` adds rayon multithreading, and implies
`std`.

The optional element-type families (`half`, `complex`, `int8`) and the `epilogue`
capability are off by default. A plain float build pays for none of their codegen or
dependencies. To build the crate as `no_std` (only `core` + `alloc`), turn the defaults
off with `default-features = false`. See [no_std and WebAssembly](no_std_and_WebAssembly.md)
for that setup.

## A first complete example

This example computes a `2x3` matrix times a `3x2` matrix, all row-major, run
single-threaded:

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

The arguments are exactly the terms of `C <- alpha*A*B + beta*C`. They are the scalar
`alpha`, the 2 input views, the scalar `beta`, the output view, and a
[`Parallelism`](Parallelism_in_Practice.md) selector.

`MatRef::from_row_major(&a, 2, 3)` reads `a` as a 2-by-3 row-major matrix. The shapes
must line up: `A.cols` must equal `B.rows`, and `C` must be `A.rows` by `B.cols`. If
they do not line up, the call panics before it touches memory.

Transposition never needs a copy. It is a stride change, not a data move.
`MatRef::from_col_major(&b, 3, 2)` reads the same buffer as a column-major matrix.
[`MatRef::new`](Matrix_Views_and_Layouts.md) lets you set the row and column strides
directly.

## What happened under the hood

The `gemm` entry does a small amount of work before it does any arithmetic.

First, it validates the call. It checks that the inner dimensions agree. It checks that
each view stays inside its slice. It checks that `C` addresses every `(i, j)` at a
distinct offset, and that `C`'s storage does not overlap `A`'s or `B`'s storage. Any
failed check raises a panic with a specific message, before a single unsafe operation
runs. Only then does it lower the 3 views to raw pointers and strides, and hand them to
the dispatch layer.

Dispatch resolves which kernel to run. The first GEMM call for a given element type
runs CPU feature detection once. It records the winning entry point in a `OnceLock`,
and returns. Every later call is a plain indirect call through that cached pointer, with
no repeat detection. So the runtime ISA choice is a one-time cost, amortized across the
whole process.

You can override the automatic choice with the `GEMMKIT_REQUIRE_ISA` environment
variable. You can also use it to pin a specific backend for testing. gemmkit reads the
variable once and memoizes it the same way. See [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md)
for details. [Life of a GEMM Call](../architecture/Life_of_a_GEMM_Call.md) walks the
full path from call to microkernel.

## alpha and beta, precisely

`alpha` scales the product `A*B`. `beta` scales the incoming contents of `C`. The one
detail to remember is what happens at the edges.

When `beta == 0`, the engine **does not read** `C` at all. It overwrites `C` with
`alpha*A*B`. That rule is what makes `let mut c = [0.0_f32; 4]` correct above, even
though you could have left the buffer uninitialized. In concrete terms, a `beta == 0`
output slice may hold garbage. Through the unchecked tier it may even be genuinely
uninitialized memory, and the result is still well defined.

When `beta == 1`, the engine leaves the existing `C` untouched and accumulates the
product onto it. Any other `beta` value first multiplies `C` through.

There is also a degenerate fast path. If `k == 0` (an empty contraction) or `alpha == 0`
(the product vanishes), the call reduces to `C <- beta*C`. It never touches `A` or `B`
at all, and just scales the output in place. Combined with the `beta == 0` rule,
`alpha == 0, beta == 0` zeroes `C`, and `k == 0, beta == 1` is a no-op. Narrow types
scale in `f32` and round back on the store. So the degenerate path rounds exactly as the
full kernel would.

## The Cargo features

| Feature | Default | Unlocks | Pulls in |
| --- | --- | --- | --- |
| `std` | yes | runtime cache/CPU detection, env knobs, thread-local workspace pool. Off = `no_std` (`core` + `alloc`) | `raw-cpuid` (x86 only) |
| `parallel` | yes | rayon multithreading (`Parallelism::Rayon`). Implies `std` | `rayon` |
| `wasm_threads` | no | a sized rayon pool for `wasm32-wasip1-threads`. Implies `parallel` | (via `parallel`) |
| `half` | no | `f16`/`bf16` mixed-precision GEMM, `f32` accumulate | `half` |
| `complex` | no | `c32`/`c64` GEMM with conjugation (`gemm_cplx`) | `num-complex` |
| `int8` | no | `i8 -> i32` integer GEMM (`gemm_i8`) | (none) |
| `epilogue` | no | fused bias/activation, `i8`/`u8` requantization, per-element map | (none) |

The element-type and capability features compose. `half` + `epilogue` gives fused f16
GEMM. `int8` + `epilogue` gives the requantizing entries, and so on. See
[Element Types](Element_Types.md) and [Fused Epilogues](Fused_Epilogues.md) for each
combination.

## Version requirements

gemmkit targets **Rust 1.89** on **edition 2024**. It is licensed MIT OR Apache-2.0. The
API reference is on [docs.rs/gemmkit](https://docs.rs/gemmkit). This book is the
long-form companion to that reference.

## Where to next

- [Matrix Views and Layouts](Matrix_Views_and_Layouts.md): how `MatRef`/`MatMut`,
  strides, transposition, and submatrices work, and exactly what the safe API
  validates.
- [Element Types](Element_Types.md): `f16`/`bf16`, `i8`, and complex, with their
  accuracy characteristics.
- [Parallelism in Practice](Parallelism_in_Practice.md): what `Rayon(0)` auto actually
  does and when `Serial` is the right call.
- [Runtime ISA Dispatch](Runtime_ISA_Dispatch.md) and [The Unchecked Tier](The_Unchecked_Tier.md):
  overriding the backend, and the raw-pointer engine.
- Working with an existing array library? See the adapters:
  [ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md),
  [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md),
  [faer](../gemmkit-faer/Using_gemmkit_with_faer.md).
