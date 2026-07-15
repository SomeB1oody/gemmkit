# gemmkit

The core GEMM engine — **zero ndarray dependency**. Computes `C ← α·A·B + β·C` for
`f32`/`f64` — plus optional `f16`/`bf16`, `i8`, and `c32`/`c64` families behind Cargo
features — over a data-type-agnostic `&[T]` + stride API, selecting the best
instruction set at runtime: x86 (scalar / AVX2+FMA / AVX-512) and AArch64 (NEON,
baseline on the architecture).

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

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
```

## API

Two layers over the same engine:

- **Safe** — [`gemm`] and [`gemm_with`] take `MatRef`/`MatMut` slice + stride
  views. Shape mismatches, out-of-bounds strides, and a `C` that aliases `A`/`B`
  all **panic** before any unsafe work runs.
- **Unchecked** — [`gemm_unchecked`] / [`gemm_unchecked_with`] are the raw
  pointer + `isize` stride engine for callers that validate their own inputs (and
  may use negative strides).

Semantics are exactly `C ← α·A·B + β·C`. Transposition is expressed through strides
(a transposed view swaps `rs`/`cs`, no copy). When `β == 0`, `C` is **not read**, so
it may be uninitialized.

## Workspace

The default `gemm` path uses a transparent thread-local pool, so it allocates at
most once per thread. For hot loops of small products or real-time code, create a
[`Workspace`] and pass it to `gemm_with`: its second and later uses perform **zero**
heap allocation.

```rust
use gemmkit::{gemm_with, MatRef, MatMut, Parallelism, Workspace};
let mut ws = Workspace::new();
# let (a, b) = (vec![0.0f32; 12], vec![0.0f32; 12]);
# let mut c = vec![0.0f32; 9];
for _ in 0..1000 {
    gemm_with(&mut ws, 1.0,
        MatRef::from_col_major(&a, 3, 4),
        MatRef::from_col_major(&b, 4, 3),
        0.0, MatMut::from_col_major(&mut c, 3, 3), Parallelism::Serial);
}
```

## Features

- `std` (default) — runtime cache/feature detection and the thread-local pool.
- `parallel` (default) — rayon multithreading. With it off, everything still
  compiles and runs single-threaded.
- `half` — `f16`/`bf16` mixed-precision GEMM with f32 accumulation.
- `int8` — `i8 → i32` integer GEMM (`gemm_i8`); arithmetic wraps on overflow.
- `complex` — `c32`/`c64` complex GEMM with optional conjugation of A/B (`gemm_cplx`).
- `epilogue` — fused epilogues: bias/activation (`gemm_fused*`, `gemm_batched_fused*`,
  and with `complex` `gemm_cplx_fused*`); a user-defined per-element closure (`gemm_map*`,
  `f32`/`f64`); and, with `int8`, requantized `i8`/`u8` output
  (`gemm_i8_requant*`). Off by default; a plain-GEMM build pays for none of its codegen.

## Tuning

Every heuristic threshold lives in [`tuning`], each resolving *per-call argument >
programmatic setter > `GEMMKIT_*` env var > compile-time default* (calibrated on a
Ryzen 9950X). For example `GEMMKIT_PARALLEL_THRESHOLD`, `GEMMKIT_LHS_PACK_THRESHOLD`.

## Forcing a kernel (testing / CI)

By default the best available ISA is selected at runtime. Set
`GEMMKIT_REQUIRE_ISA` to `scalar`, `fma`, `avx512`, or `neon` to **force** exactly
that kernel through the public API; if the CPU (or an emulator like Intel SDE)
does not report the feature — or the ISA does not exist on this target (`neon` is
aarch64-only, `fma`/`avx512` x86-only) — dispatch **panics** instead of falling
back, so CI can run the whole suite against each kernel and a green run proves that
kernel really executed. Unset (or `auto`) is the normal auto-selecting behavior.
See `.github/workflows/ci.yml` for the matrix (scalar / FMA natively, AVX-512
under SDE, NEON on a macOS arm64 runner).

## Extending it

The variation points are traits, all public:

- a new **ISA** → implement [`simd::Simd`] + [`simd::SimdOps`] for a new token, add
  one `vectorize` trampoline and one dispatch line;
- a new **element type** → implement [`Scalar`] (and `Float`-like arithmetic);
- a new **operation family** (complex, integer) → implement [`kernel::KernelFamily`]
  and drive it through the unchanged [`driver::run`].

See [`ARCHITECTURE.md`](../ARCHITECTURE.md).

[`gemm`]: https://docs.rs/gemmkit
[`gemm_with`]: https://docs.rs/gemmkit
[`gemm_unchecked`]: https://docs.rs/gemmkit
[`gemm_unchecked_with`]: https://docs.rs/gemmkit
[`Workspace`]: https://docs.rs/gemmkit
[`tuning`]: https://docs.rs/gemmkit
[`Scalar`]: https://docs.rs/gemmkit
[`simd::Simd`]: https://docs.rs/gemmkit
[`simd::SimdOps`]: https://docs.rs/gemmkit
[`kernel::KernelFamily`]: https://docs.rs/gemmkit
[`driver::run`]: https://docs.rs/gemmkit
