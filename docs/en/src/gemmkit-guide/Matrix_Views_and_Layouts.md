# Matrix Views and Layouts

Every gemmkit call takes its operands as *views*: a slice, a shape, and 2 strides.
`MatRef<'a, T>` is the immutable input view. `MatMut<'a, T>` is the mutable output
view. Neither view owns its data. Both borrow a slice you already have.

The library's whole layout vocabulary lives in those 2 stride numbers: row-major,
column-major, transposed, submatrix, and broadcast. So the same buffer can be read a
dozen ways without ever being copied.

## The 2 strides

Element `(i, j)` of a view lives at slice offset `i*rs + j*cs`, where `rs` is the row
stride and `cs` is the column stride. **Strides are counted in elements, not bytes.** An
`rs` of 4 means the next row sits 4 elements further along the slice. That single offset
formula is the entire model. Everything else is a choice of `rs` and `cs`.

3 constructors cover the common cases. Each exists on both `MatRef` and `MatMut`:

```rust
use gemmkit::MatRef;

let data = [0.0_f32; 12];
let row_major = MatRef::from_row_major(&data, 3, 4); // rs = cols = 4, cs = 1
let col_major = MatRef::from_col_major(&data, 3, 4); // rs = 1, cs = rows = 3
let general   = MatRef::new(&data, 3, 4, 4, 1);      // explicit rs, cs (here == row-major)
```

`from_row_major(data, rows, cols)` sets `rs = cols, cs = 1`. Rows are contiguous: the
classic C order. `from_col_major(data, rows, cols)` sets `rs = 1, cs = rows`. Columns
are contiguous: Fortran order.

`new(data, rows, cols, rs, cs)` takes the strides verbatim. Reach for it when neither
canonical layout matches, for example a submatrix, or a view whose leading dimension
differs from its logical width. `MatRef` and `MatMut` also expose `.rows()` and
`.cols()`.

## Transposition is a stride swap

Because `(i, j)` maps through `i*rs + j*cs`, swapping the roles of the 2 strides (and
the 2 dimensions) transposes the view in place. Say `a` holds an `m x k` matrix in
row-major order (`rs = k, cs = 1`). Its transpose is the `k x m` matrix whose `(i, j)`
is the original `(j, i)`, at offset `j*k + i`. That offset is exactly `rs = 1, cs = k`
over the same slice:

```rust
use gemmkit::MatRef;

// `a` is m x k row-major
let (m, k) = (2, 3);
let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

let a_rowmajor = MatRef::from_row_major(&a, m, k); // m x k
let a_transposed = MatRef::from_col_major(&a, k, m); // k x m, same bytes, no copy
```

So `from_col_major` over a row-major buffer *is* the transpose, and the reverse also
holds. `new` with `rs`/`cs` swapped does the same for any layout. A transposed operand
therefore costs nothing at the API level: the kernel walks the strides you give it.
This is how you feed `A^T * B` or `A * B^T` without materializing a transpose.

## Submatrices and strided views

A submatrix is a view whose leading dimension (the distance between successive rows or
columns) is larger than its logical extent. Build one by slicing the buffer so the
block's top-left element sits at the start of the slice, then handing over the
*parent's* strides. Here is the top-left `2 x 2` block of a `4 x 4` row-major matrix,
starting at row 1, column 1:

```rust
use gemmkit::MatRef;

let parent = [0.0_f32; 16]; // 4x4 row-major, leading dimension 4
let block = MatRef::new(&parent[1 * 4 + 1..], 2, 2, 4, 1); // rs stays 4, cs stays 1
```

The row stride is still 4, the parent's width. So consecutive rows of the block skip
over the columns you excluded. The slice begins at offset `5`, the block's `(0, 0)`. The
safe API verifies the tail slice is long enough to reach the block's far corner.

The same mechanism expresses a broadcast input. A stride of `0` makes a dimension
repeat 1 element. A `1 x n` row broadcast down `m` rows is `MatRef::new(row, m, n, 0, 1)`.
Every logical row then reads the same storage. gemmkit allows broadcasts for the
read-only inputs `A` and `B`, but never for the output. The next section explains why.

## What the safe API accepts, and what it rejects

The safe entries (`gemm`, `gemm_i8`, `gemm_cplx`, and the fused variants) accept
**non-negative strides only**, including `0` for a broadcast input. A negative stride
is outside what a `&[T]` view can describe safely. So is a base pointer that sits in the
middle of a buffer, rather than at element `(0, 0)`. Those cases live in
[The Unchecked Tier](The_Unchecked_Tier.md), the raw-pointer engine the adapters use to
express arbitrary layouts.

Before any arithmetic, the safe entries run one validation prologue over the
`(A, B, C)` trio. Every failure is a panic, raised ahead of the first unsafe operation:

- **Shape agreement.** `A.cols == B.rows`, `A.rows == C.rows`, `B.cols == C.cols`. A
  mismatch panics with the offending pair, for example `gemmkit: A.cols (3) != B.rows (4)`.
- **In-bounds views.** For each view, the engine computes the highest slice offset it
  will touch, and checks it against the slice length. Too small a slice panics with
  `gemmkit: A view of 3x4 (strides 4,1) needs 12 elements but slice has 8`. A view whose
  strides are negative, or so large the addressing overflows `usize`, panics with
  `... has negative strides or is too large to address; use gemm_unchecked`.
- **`C` addresses each element uniquely.** gemmkit writes the output, so 2 distinct
  `(i, j)` must never land on the same offset. A self-aliasing `C`, such as a zero row
  or column stride, or strides that collide, would become a data race in parallel mode.
  That is reachable from entirely safe code, so it panics:
  `gemmkit: C view aliases itself (...); C must address each (i,j) uniquely`. This is why
  broadcast strides are fine for `A`/`B` (read-only) but forbidden for `C`.
- **`C` does not overlap `A` or `B`.** The output's byte range must be disjoint from each
  input's. gemmkit compares byte ranges, not element counts, so the heterogeneous
  integer API (`i8` inputs, `i32` output) stays exact. Overlap panics with
  `gemmkit: C aliases A or B`. In fully safe Rust, the borrow checker already forbids an
  overlapping `&mut`/`&` pair. This check is a defensive backstop that also covers the
  raw-lowered paths.

These messages are stable. The correctness suite asserts on their wording, so you can
rely on them in tests.

## Zero-sized dimensions

A view with a zero dimension is legal and validates cleanly. gemmkit accepts a `0 x k`,
`m x 0`, or `m x n x (k = 0)` shape. Any slice, even an empty one, satisfies the
in-bounds check, because such a view addresses nothing.

If `m == 0` or `n == 0`, the call is a no-op: there is no output to write. If only
`k == 0`, the contraction is empty, and the call reduces to `C <- beta*C`. That is the
same scale-only path `alpha == 0` takes. See [Getting Started](Getting_Started.md) for
that degenerate rule.

## Where to next

- [Element Types](Element_Types.md): the same views over `f16`/`bf16`, `i8`, and
  complex data.
- [The Unchecked Tier](The_Unchecked_Tier.md): negative strides, interior base pointers,
  and the raw-pointer engine.
- The adapters ([ndarray](../gemmkit-ndarray/Using_gemmkit_with_ndarray.md),
  [nalgebra](../gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md),
  [faer](../gemmkit-faer/Using_gemmkit_with_faer.md)) build these views for you from
  each library's native matrix types.
