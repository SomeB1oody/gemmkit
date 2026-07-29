# Prepacked Operands

Before the microkernel can touch `A` and `B`, the engine copies each into a cache-friendly
micropanel layout. This layout holds contiguous tiles, and the microkernel walks each tile
with unit strides. For a one-shot product, that copy is pure setup. The engine pays for it
once and never uses it again.

Many workloads multiply the same matrix over and over. A linear layer applies one fixed
weight matrix to a stream of activation batches. A solver applies one fixed operator to
many right-hand sides. Repacking a fixed operand on every call throws away work the engine
already did. The prepacked-operand API lets you pay for the pack once, then reuse the
result across every product that shares that operand.

## Packing the right-hand side

The common case fixes `B` (the weights) and streams a series of differently sized `A`
matrices (the activations). Call [`prepack_rhs`](https://docs.rs/gemmkit) once to turn a
`k x n` `B` into a [`PackedRhs`](https://docs.rs/gemmkit) handle. Then feed that handle to
`gemm_packed_b` for each product:

```rust
use gemmkit::{prepack_rhs, gemm_packed_b, MatRef, MatMut, Parallelism};

// fixed weights: a k x n matrix reused across many activation batches
let (k, n) = (512, 256);
let weights = vec![0.0f32; k * n];
let packed = prepack_rhs(MatRef::from_col_major(&weights, k, n));

// per activation batch: an m x k input, sharing the packed weights
let mut c = vec![0.0f32; m * n];
gemm_packed_b(
    1.0,
    MatRef::from_row_major(&input, m, k),
    &packed,
    0.0,
    MatMut::from_col_major(&mut c, m, n),
    Parallelism::Rayon(0),
);
```

`prepack_rhs` accepts any layout of `B` and reads it through its strides. A row-major,
column-major, or transposed view all pack the same way. The pack runs once, inside
`prepack_rhs`, on a single thread. Every later call to `gemm_packed_b` skips it.

The buffer records its own blocking geometry: `nr`, `kc`, and `nc`. Every
consuming call reads that same geometry back, so a panel is always read against its own
tiling. The buffer is read-only for the whole GEMM, so gemmkit never writes to it after
the pack. A single `PackedRhs` is therefore safe to share across threads and across
concurrent calls, with no extra synchronization. `PackedRhs::rows()` reports the original
`k`. `PackedRhs::cols()` reports the original `n`.

The handle works for any product whose `(k, n)` match the packed `B`, as long as the
output `C` is column-major-ish (`|csc| >= |rsc|`). That constraint is the one surprise in
this API. A row-major `C` would force the engine to swap `A` and `B` internally, to keep
its stores contiguous. A prepacked `B` cannot move into the `A` role. So `gemm_packed_b`
panics on a row-major `C` and points you back to plain `gemm` for that layout. Only `C` is
pinned this way. `A`'s layout is unconstrained.

Under a fixed configuration, `gemm_packed_b` reproduces plain `gemm` and is deterministic
across worker counts. There is one narrow caveat. It applies to a small product, where
both `m` and `n` are at or below the `small_mn_dim` knob (16 by default, 32 on aarch64).
It also applies to a gemv-shaped product, where `m == 1` or `n == 1`. In both cases the
2 calls may differ in the last ULP. The reason is routing, not error. Plain `gemm`
reroutes those shapes to a [special path](Small_Shapes_and_GEMV.md), while the prepacked
entry always drives the general packed kernel. Both answers are correct. They sum in a
slightly different order, only on the shapes where the special paths would otherwise take
over.

## The left-hand-side mirror

The symmetric case fixes `A` and streams a series of varying `B` matrices.
[`prepack_lhs`](https://docs.rs/gemmkit) produces a [`PackedLhs`](https://docs.rs/gemmkit)
handle, and `gemm_packed_a` consumes it. This mirrors the RHS pair exactly, with the axes
relabeled. `PackedLhs::rows()` is the original `m`. `PackedLhs::cols()` is the shared `k`.

Internally, the LHS pack is not a separate code path. By the engine's A/B symmetry, a
prepacked `A` is exactly the prepacked `B` of the transposed product `C^T = B^T A^T`. So
`prepack_lhs` lays down the identical micropanel buffer, and only records the dimensions
in LHS terms.

This has one visible consequence: the `C`-layout constraint flips. `gemm_packed_a`
requires a row-major-ish `C` (`|csc| <= |rsc|`), the exact opposite of the RHS entry. A
column-major `C` would keep `A` in the genuine LHS role, and a transposed-RHS buffer
cannot fill that role. Pick the packed-`A` entry when your `C` is row-major. Pick the
packed-`B` entry when your `C` is column-major. Together the 2 entries cover both
orientations.

## Fused variants

Each packed entry has a fused twin, under the `epilogue` feature. `gemm_packed_b_fused`
and `gemm_packed_a_fused` add a per-row or per-col bias, plus an optional activation, in
the same store the packed kernel already runs. See [Fused Epilogues](Fused_Epilogues.md)
for the bias and activation types.

The same `PackedRhs` or `PackedLhs` handle serves both the plain entry and the fused
entry. The epilogue applies only at the store, so it never touches the pack. Build the
buffer once, then choose per call whether to fuse.

2 details are specific to the packed path. First, unlike plain `gemm_fused`, the packed
fused entries never reroute to the gemv, small-`m,n`, or small-`k` kernels. They always
drive the general packed kernel, the same divergence the plain packed entries document.
Second, gemmkit always gives the per-row or per-col bias in the natural user frame.
`gemm_packed_a_fused` handles the internal transpose for you, so a `PerRow` bias has
length `A.rows`, no matter which entry you call.

## Prepacking i8 weights

Under the `int8` feature, the same pattern extends to quantized inference. `prepack_rhs_i8`
packs a fixed `i8` weight matrix into a `PackedRhs<i8>`, and `gemm_i8_packed_b` consumes
it. It takes `i8` inputs and produces an `i32` output.

Prepacking is a bigger win here than for floats, for a structural reason. The AVX-512
VNNI kernel (`vpdpbusd`) reads its RHS from a k-quad-interleaved layout. The engine cannot
produce that layout in place, so this kernel's RHS pack is mandatory on every call. At
small `m`, that per-call `O(k*n)` pack easily dominates the `O(m*k*n)` compute. Prepacking
removes it from the hot loop entirely.

The packed buffer also pins the kernel choice. It is laid out for whichever integer kernel
the process's dispatch selected, either the VNNI interleave or the widen kernel's plain
panels. `gemm_i8_packed_b` always runs that same family, so the buffer is never misread.

Integer accumulation is exact and does not depend on the ISA. So the packed and plain
paths agree bit-for-bit for every valid shape, with no small-shape caveat at all.

```rust
use gemmkit::{prepack_rhs_i8, gemm_i8_packed_b, MatRef, MatMut, Parallelism};

let packed = prepack_rhs_i8(MatRef::from_col_major(&weights_i8, k, n));
let mut c = vec![0i32; m * n];
gemm_i8_packed_b(
    1,
    MatRef::from_row_major(&input_i8, m, k),
    &packed,
    0,
    MatMut::from_col_major(&mut c, m, n),
    Parallelism::Rayon(0),
);
```

## When prepacking pays

Prepacking trades one upfront `O(k*n)` copy for a saved repack on every later product
against that operand. It pays off exactly when you reuse the operand enough to amortize
that copy. A single product, or an operand that changes on every call, gains nothing. The
pack is then pure overhead, and plain `gemm` is the right tool.

Be aware that the float path does not always pack `B` in the first place. For small `m`,
plain `gemm` reads `B` in place, unpacked, a choice governed by the `rhs_pack_threshold`
[knob](Tuning_Knobs.md). So prepacking a lightly reused float `B` can even lose.

The clearest wins are the fixed-weight inference loops this API is named for. Above all,
that means the `i8` VNNI path, whose RHS pack is otherwise unavoidable on every single
call. When in doubt, measure the loop both ways. The crossover point depends on your
reuse count and your machine, not on a fixed rule.

The raw-pointer forms (`prepack_rhs_unchecked`, `gemm_packed_b_unchecked`, and their
`_with`, LHS, and `i8` counterparts) exist for adapters and FFI that validate their own
inputs. See [The Unchecked Tier](The_Unchecked_Tier.md).
