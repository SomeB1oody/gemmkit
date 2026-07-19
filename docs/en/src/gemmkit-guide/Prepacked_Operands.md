# Prepacked Operands

Before the microkernel can touch A and B, the engine copies each into a cache-friendly micropanel layout: contiguous tiles the microkernel walks with unit strides. For a one-shot product that copy is pure setup, paid once and forgotten. But a great many workloads multiply the *same* matrix over and over: a linear layer applies one fixed weight matrix to a stream of activation batches, a solver applies one operator to many right-hand sides. Repacking that fixed operand on every call throws away work you already did. The prepacked-operand API lets you pay the pack once and reuse the result across as many products as share that operand.

## Packing the right-hand side

The common case is a fixed `B` (the weights) against a stream of differently-sized `A` (the activations). Call [`prepack_rhs`](https://docs.rs/gemmkit) once to turn a `k x n` `B` into a [`PackedRhs`](https://docs.rs/gemmkit) handle, then feed that handle to `gemm_packed_b` for each product:

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

`prepack_rhs` accepts any layout of `B` and reads it through its strides, so a row-major, column-major, or transposed view all pack the same way. The pack runs once, single-threaded, inside `prepack_rhs`; every later `gemm_packed_b` skips it. The buffer records the blocking geometry (`nr`, `kc`, `nc`) it was built for, and the consuming call reads that geometry back verbatim, so a panel is always interpreted against its own tiling. Because the buffer is only read during the GEMM, never written, one `PackedRhs` is safely shared across threads and across concurrent calls with no synchronization. `PackedRhs::rows()` reports the original `k`, `cols()` the original `n`.

The handle is valid for any product whose `(k, n)` match the packed `B` and whose output `C` is **column-major-ish** (`|csc| >= |rsc|`). That last constraint is the one surprise. A row-major `C` would make the engine internally swap `A` and `B` to keep its stores contiguous, and a prepacked `B` cannot move into the `A` role, so `gemm_packed_b` panics on a row-major `C` and points you back to plain `gemm` for that layout. `A`'s layout is unconstrained; only `C` is pinned.

Under a fixed configuration `gemm_packed_b` **reproduces** plain `gemm`, and is deterministic across worker counts, with one narrow caveat: for very small products (both `m` and `n` at or below `tiny_block_dim`, default 64) and for gemv-shaped products (`m == 1` or `n == 1`) the two may differ in the last ULP. The reason is routing, not error: plain `gemm` reroutes those shapes to a [special path](Small_Shapes_and_GEMV.md), whereas the prepacked entry always drives the general packed kernel. Both answers are correct; they just sum in a slightly different order on exactly the shapes where the special paths would have taken over.

## The left-hand-side mirror

The symmetric case, one fixed `A` against a stream of varying `B`, is served by [`prepack_lhs`](https://docs.rs/gemmkit) producing a [`PackedLhs`](https://docs.rs/gemmkit), consumed by `gemm_packed_a`. It mirrors the RHS pair exactly, with the axes relabelled: `PackedLhs::rows()` is the original `m`, `cols()` the shared `k`.

Internally the LHS pack is not a separate code path. By the engine's A/B symmetry, a prepacked `A` is precisely the prepacked `B` of the transposed product `C^T = B^T A^T`, so `prepack_lhs` lays down the identical micropanel buffer and only records the dimensions in LHS terms. The visible consequence is that the `C`-layout constraint flips: `gemm_packed_a` requires a **row-major-ish** `C` (`|csc| <= |rsc|`), the exact opposite of the RHS entry, because a column-major `C` would keep `A` in the genuine LHS role that a transposed-RHS buffer cannot fill. Pick the packed-`A` entry when your `C` is row-major and the packed-`B` entry when it is column-major; they cover the two orientations between them.

## Fused variants

Each packed entry has a fused twin, under the `epilogue` feature: `gemm_packed_b_fused` and `gemm_packed_a_fused` add a per-row or per-col bias and an optional activation in the same store the packed kernel already runs (see [Fused Epilogues](Fused_Epilogues.md) for the bias and activation types). The **same** `PackedRhs` or `PackedLhs` handle serves both the plain and the fused entry: the epilogue is store-side only and never touches the pack, so you build the buffer once and choose per call whether to fuse. Two details are specific to the packed path. First, unlike plain `gemm_fused`, the packed fused entries are never rerouted to the gemv / small-`m,n` / small-`k` kernels; they always drive the general packed kernel (the same divergence the plain packed entries document). Second, the per-row / per-col bias is always given in the natural **user** frame: `gemm_packed_a_fused` handles the internal transpose for you, so a `PerRow` bias is length `A.rows` regardless of which entry you call.

## Prepacking i8 weights

Under the `int8` feature the pattern extends to quantized inference: `prepack_rhs_i8` packs a fixed `i8` weight matrix into a `PackedRhs<i8>`, and `gemm_i8_packed_b` consumes it (`i8` inputs, `i32` output). Prepacking is a larger win here than for floats, for a structural reason. The AVX-512 VNNI kernel (`vpdpbusd`) reads its RHS from a k-quad-interleaved layout that cannot be produced in place, so its RHS pack is **mandatory on every call**; at small `m` that per-call `O(k*n)` pack easily dominates the `O(m*k*n)` compute. Prepacking deletes it from the hot loop entirely. The packed buffer also **pins the kernel choice**: it is laid out for whichever integer kernel the process's dispatch selected (the VNNI interleave, or the widen kernel's plain panels), and `gemm_i8_packed_b` always runs that same family, so the buffer is never misread. Integer accumulation is exact and ISA-independent, so the packed and plain paths agree **bit-for-bit** for every valid shape, with no small-shape caveat at all.

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

Prepacking trades one upfront `O(k*n)` copy for a saved repack on every subsequent product against that operand, so it pays exactly when the operand is reused enough to amortize the copy. A single product, or an operand that changes every call, gains nothing: the pack is then pure overhead, and plain `gemm` is the right tool. Be aware, too, that the float path does not always pack `B` in the first place: for small `m` a plain `gemm` reads `B` in place unpacked (governed by the `rhs_pack_threshold` [knob](Tuning_Knobs.md)), so prepacking a lightly-reused float `B` can even lose. The clearest wins are the fixed-weight inference loops the API is named for, and above all the `i8` VNNI path, whose RHS pack is otherwise unavoidable on every single call. When in doubt, measure the loop both ways; the crossover is a property of your reuse count and your machine, not a fixed rule.

The raw-pointer forms (`prepack_rhs_unchecked`, `gemm_packed_b_unchecked`, and their `_with`, LHS, and `i8` counterparts) exist for adapters and FFI that validate their own inputs; see [The Unchecked Tier](The_Unchecked_Tier.md).
