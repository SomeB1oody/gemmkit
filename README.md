# gemmkit

A clean, extensible, high-performance **GEMM** (general matrix multiply) engine
for Rust, in two crates:

- [**`gemmkit`**](/gemmkit/README.md) — the core engine. Zero ndarray dependency; a
  data-type-agnostic `&[T]` + stride API (plus a raw-pointer engine). Picks the
  best instruction set at runtime — x86 (scalar / AVX2+FMA / AVX-512) and
  AArch64 (NEON).
- [**`gemmkit-ndarray`**](/gemmkit-ndarray/README.md) — a thin [`ndarray`] 0.17 adapter.

It computes `C ← α·A·B + β·C` for `f32` and `f64`.

```rust
use gemmkit::{gemm, MatRef, MatMut, Parallelism};

let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 row-major
let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2 row-major
let mut c = [0.0_f32; 4];
gemm(
    1.0,
    MatRef::from_row_major(&a, 2, 3),
    MatRef::from_row_major(&b, 3, 2),
    0.0,
    MatMut::from_row_major(&mut c, 2, 2),
    Parallelism::Rayon(0), // 0 = auto
);
assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
```

```rust
use ndarray::array;
let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
let c = gemmkit_ndarray::dot(&a, &b);
assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
```

## Why another GEMM?

Two goals, pursued together:

**Performance.** On the calibration machine (AMD Ryzen 9950X, Zen5, AVX-512), the
single-threaded kernel reaches ~90% of the hardware f32 peak and, at equal ISA, is
**95–100% of the `gemm` crate** (faer's backend); against `matrixmultiply`
(ndarray's current backend) it is ~**2.2×**. Because gemmkit uses *stable*
AVX-512 while `gemm 0.18` needs a nightly feature for it, gemmkit's default build
runs **1.4–1.9×** faster than `gemm`'s default build.

**Architecture.** Cleaner and more extensible than the `gemm` crate: no
macro-generated kernels, no `transmute` dispatch, a single crate covering multiple
types, and all extension points collapsed onto traits. Adding an instruction set
is a `SimdOps` implementation plus two one-line entries; adding an operation family
(complex, integer) leaves the driver, packing, cache model, and parallelism
untouched. See [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Status

v1 ships f32/f64 over scalar + AVX2/FMA + AVX-512 on x86-64 and NEON on AArch64
(Apple Silicon, with `sysctl` cache detection). f16/bf16, complex, and integer
families are designed-for but not yet implemented. The AArch64 kernel uses a
lane-indexed FMA fast path (`vfmaq_laneq`, packed RHS) on top of the portable
`splat`+`vfmaq` path; single-threaded it runs at ~80% of the `gemm` crate on an
M-series core. The remaining gap is in the analytical `k`-blocking/tile, not the
FMA primitive, and is left as tuning work.

## Build, test, bench

```sh
cargo test --workspace                 # correctness, architecture, alloc, ndarray
cargo clippy --workspace --all-targets # zero warnings
cargo test -p gemmkit --release --test perf -- --ignored --nocapture   # quick perf table
cargo bench -p gemmkit                 # criterion vs gemm + matrixmultiply
```

MSRV: Rust 1.89 (stable AVX-512 intrinsics). Edition 2024.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at
your option. Contributions are dual-licensed accordingly; see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

[`ndarray`]: https://docs.rs/ndarray
