# The Unchecked Tier

The safe entries (`gemm`, `gemm_fused`, and the rest) validate their inputs before touching memory. Shapes must agree. Every strided view must stay inside its slice. The output must address each element once, and it must not overlap the inputs.

Under every one of those checks sits the same engine. It is reached through a raw-pointer, `isize`-stride interface with no checks at all. That is the unchecked tier. It exists for callers who already hold the invariants that the safe API would otherwise re-derive.

## Who it is for

3 kinds of caller live here. **Adapters** over other matrix libraries, such as `ndarray`, `nalgebra`, and `faer`, already have a validated pointer and strides straight out of the host type. Re-checking bounds would be redundant work on data the library already guarantees. **FFI callers** arriving from C or another language have a pointer and strides, and no Rust slice to bound-check against. **Custom matrix types** that a codebase owns can lower to pointers and call the engine directly, instead of copying into a `MatRef`. In each case, the caller is the party that knows the memory is valid, so the checks move to where that knowledge lives.

If none of that describes you, use the safe API. The unchecked tier is not faster for a single call. The validation cost is cheap relative to the multiply itself. It exists to let a caller that already owns the invariants avoid proving them again.

## The catalog

Every safe entry has a raw twin, named by appending `_unchecked`, and most also offer a `_with` form that takes a caller-owned workspace (next section). The full raw surface, by family:

| Family | Raw entries | Feature |
| --- | --- | --- |
| Plain GEMM | `gemm_unchecked`, `gemm_unchecked_with` | core (`f32`/`f64`, plus `f16`/`bf16` under `half`) |
| Complex | `gemm_cplx_unchecked`, `gemm_cplx_unchecked_with` | `complex` |
| Integer | `gemm_i8_unchecked`, `gemm_i8_unchecked_with` | `int8` |
| Fused bias/activation | `gemm_fused_unchecked`, `gemm_fused_unchecked_with` | `epilogue` |
| Map (per-element closure) | `gemm_map_unchecked`, `gemm_map_unchecked_with` | `epilogue` |
| Complex fused | `gemm_cplx_fused_unchecked`, `gemm_cplx_fused_unchecked_with` | `complex` + `epilogue` |
| Requantize | `gemm_i8_requant_unchecked`, `gemm_i8_requant_u8_unchecked` (+ `_with`) | `int8` + `epilogue` |
| Strided batched | `gemm_batched_unchecked`, `gemm_batched_unchecked_with` | core |
| Pointer-array batched | `gemm_batched_ptr_unchecked` | core |
| Batched fused | `gemm_batched_fused_unchecked`, `gemm_batched_fused_unchecked_with` | `epilogue` |
| Prepack | `prepack_rhs_unchecked`, `prepack_lhs_unchecked`, `prepack_rhs_i8_unchecked` | core / `int8` |
| Consume prepacked | `gemm_packed_a_unchecked`, `gemm_packed_b_unchecked` (+ `_with`, `_fused_`) | core / `epilogue` |
| Consume prepacked (i8) | `gemm_i8_packed_b_unchecked` (+ `_with`) | `int8` |

The pointer-array batched form is worth calling out. `gemm_batched_ptr_unchecked` takes a slice of `GemmProblem<T>`, each with its own shape and its own pointers. So a batch can mix sizes, and it can scatter its operands anywhere in memory. It has no safe counterpart of the same shape. Expressing "an array of independent raw problems" is precisely what the raw tier is for. The nalgebra and faer adapters build their batched GEMM on top of it.

## The safety contract

Calling into the unchecked tier means signing, per call, for what the safe API would otherwise check:

- **Valid pointers and strides.** For every `(i, j)` implied by the dimensions and strides, `a` and `b` are valid for reads and `c` is valid for read and write. Nothing bounds-checks this. An out-of-range stride is undefined behavior, not a panic.
- **A uniquely-addressed output.** `C`'s strides must map every distinct `(i, j)` to a distinct location. The parallel driver assumes output tiles are disjoint, and it writes them concurrently. A self-aliasing `C` (for example `rsc == 0`) would then be a data race. The inputs may alias themselves freely, since they are only read. So a broadcast, zero-stride, `A` or `B` is fine.
- **No overlap between `C` and `A`/`B`.** The output is written. If it overlapped an input, the result would be garbage.

One relaxation comes with the territory. When `beta == 0`, the output is not read, so `C` need not be initialized. One capability the safe API withholds is available here too: **negative strides**, and pointers into the middle of a buffer, are both allowed. A reversed view (`rs < 0`), or an operand addressed backward from its last element, is exactly the kind of layout the safe `MatRef` refuses. The raw engine accepts it instead. This is why adapters over libraries that produce reversed strides forward to this tier.

## Reusing a workspace

Each raw entry comes in 2 allocation flavors. The plain form, `gemm_unchecked`, borrows the thread-local packing pool. It allocates at most once per thread. The `_with` form, `gemm_unchecked_with`, takes a `&mut Workspace` that you own instead:

```rust
use gemmkit::{Workspace, Parallelism};

let mut ws = Workspace::new();
// each iteration reuses `ws`; after the first large call it does no heap work
for _ in 0..iters {
    // SAFETY: pointers/strides valid, c uniquely addressed, c disjoint from a/b
    unsafe {
        gemmkit::gemm_unchecked_with(
            &mut ws, m, k, n,
            1.0_f32, a, rsa, csa, b, rsb, csb, 0.0_f32, c, rsc, csc,
            Parallelism::Serial,
        );
    }
}
```

The workspace grows to fit the largest problem it has served. It reuses that allocation thereafter, so a hot loop of GEMMs reaches zero steady-state allocation. This is the mechanism that `no_std` builds rely on for reuse, since they have no thread-local pool. It is equally useful under `std`, for real-time or latency-sensitive loops where you want allocation off the hot path.

## A worked example: a custom tile type

Suppose your code already carries its own dense row-major matrix, and you want to multiply two of them without copying into a `MatRef`:

```rust
use gemmkit::{gemm_unchecked, Parallelism};

// a dense row-major matrix the caller already owns
struct Tile {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

// c = a * b for row-major tiles
fn matmul(a: &Tile, b: &Tile, c: &mut Tile) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);
    // row-major: row stride = cols, column stride = 1
    // SAFETY: shapes checked above; each tile owns a dense rows*cols buffer, so
    // every addressed element is in bounds; c is a distinct &mut, so it cannot
    // alias a or b, and a dense layout addresses each (i, j) once
    unsafe {
        gemm_unchecked(
            a.rows, a.cols, b.cols,
            1.0_f32,
            a.data.as_ptr(), a.cols as isize, 1,
            b.data.as_ptr(), b.cols as isize, 1,
            0.0_f32,
            c.data.as_mut_ptr(), c.cols as isize, 1,
            Parallelism::Serial,
        );
    }
}
```

The `assert_eq!` shape checks and the `&mut Tile` borrow together discharge the whole contract. Dense storage makes every offset land in bounds, and it makes every `(i, j)` distinct. The exclusive borrow of `c` then rules out overlap with `a` or `b`. This is the pattern to reach for: prove the invariants at the boundary of your own type, then hand raw pointers to the engine.

## The adapters are the reference

The cleanest examples of doing this well are the adapter crates themselves. Each one pulls a pointer and strides out of a native view: C-order, F-order, general strides, or reversed strides, all with no copies. Each then forwards to the `*_unchecked` engine, with a short safety argument at each call site. If you are wrapping a matrix type of your own, read one of the adapter chapters and mirror its structure. The [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md) chapter is a good place to start. For the prepacked entries in the catalog above, the fixed-weight reuse pattern they serve is covered in [Prepacked Operands](Prepacked_Operands.md).
