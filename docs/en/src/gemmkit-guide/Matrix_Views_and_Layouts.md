# Matrix Views and Layouts

Every gemmkit call takes its operands as *views*: a slice plus a shape plus two strides. `MatRef<'a, T>` is the immutable input view, `MatMut<'a, T>` the mutable output. Neither owns its data; both borrow a slice you already have. The whole layout vocabulary of the library - row-major, column-major, transposed, submatrix, broadcast - is expressed in the two stride numbers, so the same buffer can be read a dozen ways without ever being copied.

## The two strides

Element `(i, j)` of a view lives at slice offset `i*rs + j*cs`, where `rs` is the row stride and `cs` the column stride. **Strides are counted in elements, not bytes** - `rs` of 4 means "the next row is 4 elements further along the slice." That single offset formula is the entire model; everything else is a choice of `rs` and `cs`.

Three constructors cover the common cases, each on both `MatRef` and `MatMut`:

```rust
use gemmkit::MatRef;

let data = [0.0_f32; 12];
let row_major = MatRef::from_row_major(&data, 3, 4); // rs = cols = 4, cs = 1
let col_major = MatRef::from_col_major(&data, 3, 4); // rs = 1, cs = rows = 3
let general   = MatRef::new(&data, 3, 4, 4, 1);      // explicit rs, cs (here == row-major)
```

`from_row_major(data, rows, cols)` sets `rs = cols, cs = 1`: rows are contiguous, the classic C order. `from_col_major(data, rows, cols)` sets `rs = 1, cs = rows`: columns are contiguous, Fortran order. `new(data, rows, cols, rs, cs)` takes the strides verbatim, which is what you reach for when neither canonical layout matches - a submatrix, or a view whose leading dimension differs from its logical width. `MatRef` and `MatMut` also expose `.rows()` and `.cols()`.

## Transposition is a stride swap

Because `(i, j)` maps through `i*rs + j*cs`, swapping the roles of the two strides (and the two dimensions) transposes the view in place. Say `a` holds an `m x k` matrix in row-major order (`rs = k, cs = 1`). Its transpose is the `k x m` matrix whose `(i, j)` is the original `(j, i)`, at offset `j*k + i` - which is exactly `rs = 1, cs = k` over the same slice:

```rust
use gemmkit::MatRef;

// `a` is m x k row-major
let (m, k) = (2, 3);
let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

let a_rowmajor = MatRef::from_row_major(&a, m, k); // m x k
let a_transposed = MatRef::from_col_major(&a, k, m); // k x m, same bytes, no copy
```

So `from_col_major` over a row-major buffer *is* the transpose, and vice versa; `new` with `rs`/`cs` swapped does the same for any layout. A transposed operand therefore costs nothing at the API level - the kernel walks the strides you give it. This is how you feed `A^T * B` or `A * B^T` without materializing a transpose.

## Submatrices and strided views

A submatrix is a view whose leading dimension (the distance between successive rows or columns) is larger than its logical extent. You build one by slicing the buffer so the block's top-left element sits at the start of the slice, then handing over the *parent's* strides. For the top-left `2 x 2` block of a `4 x 4` row-major matrix starting at row 1, column 1:

```rust
use gemmkit::MatRef;

let parent = [0.0_f32; 16]; // 4x4 row-major, leading dimension 4
let block = MatRef::new(&parent[1 * 4 + 1..], 2, 2, 4, 1); // rs stays 4, cs stays 1
```

The row stride is still 4 (the parent's width), so consecutive rows of the block skip over the columns you excluded, and the slice begins at offset `5`, the block's `(0, 0)`. The safe API verifies the tail slice is long enough to reach the block's far corner.

The same mechanism expresses a broadcast input: a stride of `0` makes a dimension repeat one element. A `1 x n` row broadcast down `m` rows is `MatRef::new(row, m, n, 0, 1)` - every logical row reads the same storage. Broadcasts are allowed for the read-only inputs `A` and `B`, but never for the output (see below).

## What the safe API accepts, and what it rejects

The safe entries (`gemm`, `gemm_i8`, `gemm_cplx`, the fused variants) accept **non-negative strides only**, including `0` for a broadcast input. A negative stride, or a base pointer that sits in the middle of a buffer rather than at element `(0, 0)`, is outside what a `&[T]` view can describe safely; those live in [The Unchecked Tier](The_Unchecked_Tier.md), the raw-pointer engine the adapters use to express arbitrary layouts.

Before any arithmetic, the safe entries run one validation prologue over the `(A, B, C)` trio, and every failure is a panic raised ahead of the first unsafe operation:

- **Shape agreement.** `A.cols == B.rows`, `A.rows == C.rows`, `B.cols == C.cols`. A mismatch panics with the offending pair, e.g. `gemmkit: A.cols (3) != B.rows (4)`.
- **In-bounds views.** For each view the engine computes the highest slice offset it will touch and checks it against the slice length. Too small a slice panics with `gemmkit: A view of 3x4 (strides 4,1) needs 12 elements but slice has 8`. A view whose strides are negative or so large the addressing overflows `usize` panics with `... has negative strides or is too large to address; use gemm_unchecked`.
- **`C` addresses each element uniquely.** The output is written, so two distinct `(i, j)` must never land on the same offset. A self-aliasing `C` - a zero row/column stride, or strides that collide - would become a data race in parallel mode, reachable from entirely safe code, so it panics: `gemmkit: C view aliases itself (...); C must address each (i,j) uniquely`. This is why broadcast strides are fine for `A`/`B` (read-only) but forbidden for `C`.
- **`C` does not overlap `A` or `B`.** The output's byte range must be disjoint from each input's, compared as byte ranges so the heterogeneous integer API (`i8` inputs, `i32` output) is exact. Overlap panics with `gemmkit: C aliases A or B`. In fully safe Rust the borrow checker already forbids an overlapping `&mut`/`&` pair; this is a defensive backstop that also covers the raw-lowered paths.

These messages are stable - the correctness suite asserts on their wording - so you can rely on them in tests.

## Zero-sized dimensions

A view with a zero dimension is legal and validates cleanly: a `0 x k`, `m x 0`, or `m x n x (k = 0)` shape is accepted, and any slice (even an empty one) satisfies the in-bounds check because such a view addresses nothing. If `m == 0` or `n == 0` the call is a no-op - there is no output to write. If only `k == 0`, the contraction is empty and the call reduces to `C <- beta*C`, the same scale-only path `alpha == 0` takes; see [Getting Started](Getting_Started.md) for that degenerate rule.

## Where to next

- [Element Types](Element_Types.md) - the same views over `f16`/`bf16`, `i8`, and complex data.
- [The Unchecked Tier](The_Unchecked_Tier.md) - negative strides, interior base pointers, and the raw-pointer engine.
- The adapters ([ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md), [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md), [faer](../gemmkit-faer/Using_gemmkit_with_faer.md)) build these views for you from each library's native matrix types.
