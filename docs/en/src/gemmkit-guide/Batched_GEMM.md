# Batched GEMM

A single large GEMM saturates a modern CPU on its own. A crowd of tiny ones does not.
Attention heads, grouped convolutions, per-sample linear layers, and block-diagonal solves
all produce many small independent products. Running them as a plain loop of `gemm` calls
leaves most of the machine idle. Each call is too small to parallelize usefully. Yet the
loop still pays a fork/join, or serializes, once per element. The batched entries take the
whole set in one call and schedule it as a unit. They assign whole GEMMs to workers, so a
batch of small matrices actually fills the cores.

Batched GEMM is an orchestration layer, not a new kernel. Each element re-dispatches
through the full single-GEMM engine. So a batch composes automatically with the driver,
the gemv path, and the [small-shape paths](Small_Shapes_and_GEMV.md). A batch of
`1 x 1 x k` products runs the horizontal dot inside each element. A batch of ordinary
shapes runs the register-blocked driver. Every element is an independent GEMM, and the
whole batch is reproducible across worker counts.

## The strided form

The elements might sit at a regular stride, one matrix after another in a flat buffer. In
that case, [`gemm_batched`](https://docs.rs/gemmkit) takes the single-element shape and
strides once, plus a batch stride for each of `A`, `B`, and `C`. Element `b` sits at
`A + b*a_batch_stride`, `B + b*b_batch_stride`, and `C + b*c_batch_stride`. All elements
share the same shape:

```rust
use gemmkit::{gemm_batched, MatRef, MatMut, Parallelism};

// `batch` independent m x k times k x n products, packed contiguously
gemm_batched(
    batch,
    1.0,
    MatRef::new(&a, m, k, 1, m as isize), (m * k) as isize, // A element + batch stride
    MatRef::new(&b, k, n, 1, k as isize), (k * n) as isize, // B element + batch stride
    0.0,
    MatMut::new(&mut c, m, n, 1, m as isize), (m * n) as isize, // C element + batch stride
    Parallelism::Rayon(0),
);
```

A batch stride of `0` broadcasts one operand across the whole batch. This is valid for
the read-only `A` or `B`, for example one shared weight matrix against a batch of inputs,
but never for `C`. Workers write `C`'s elements concurrently, so those elements must stay
disjoint. The result reproduces a loop of `gemm` calls exactly.

Under the `epilogue` feature, `gemm_batched_fused` applies one shared bias and one shared
activation to every element, the batched-linear-layer case. It reproduces a loop of
[`gemm_fused`](Fused_Epilogues.md) calls. gemmkit sizes the single bias vector for one
element, not the whole batch.

## The slice form: per-element shapes

When the elements differ in shape, or simply do not sit at a fixed stride, use
[`gemm_batched_slice`](https://docs.rs/gemmkit). It takes a slice of
[`BatchProblem`](https://docs.rs/gemmkit), each carrying its own `alpha`, `A`, `B`,
`beta`, and a distinct `&mut` `C` view:

```rust
use gemmkit::{gemm_batched_slice, BatchProblem, MatRef, MatMut, Parallelism};

let mut problems: Vec<BatchProblem<'_, f32>> = /* one per product, each its own shape */;
gemm_batched_slice(&mut problems, Parallelism::Rayon(0));
```

Because every `C` is a distinct `&mut`, the outputs are pairwise disjoint and cannot alias
the inputs by construction. So validation only checks per-element shape agreement and
in-bounds strides. Reach for this form when your matrices already live as a `Vec` of
views. Its raw counterpart, `gemm_batched_ptr_unchecked` over a slice of `GemmProblem`,
takes the same per-element shapes as bare pointers. It serves FFI and adapters that
validate their own inputs and may use arbitrary or negative strides.
[The Unchecked Tier](The_Unchecked_Tier.md) covers both.

## How the batch is scheduled

The interesting decision is how the work spreads across cores. The engine makes that
decision once per call, from the shared shape and the batch size. There are 3 schedules:

- **Batch-parallel.** Workers pull disjoint ranges of elements from a shared cursor. Each
  element runs *serially* on one worker, cache-hot. This is the whole point of the API.
  For many small matrices, it pays a single fork/join for the entire batch instead of one
  per element. It also keeps every core busy on complete GEMMs. This schedule never splits
  an element across workers, so it is bit-identical to the serial run at any worker count.
- **Serial.** The whole batch runs on the calling thread, each element single-threaded.
  The engine picks this when there is too little total work to justify a fork/join.
- **Sequential with internal parallelism.** For the few-but-large, memory-bound regime,
  the engine loops the batch and hands each element the *full* engine parallelism in
  turn. When an element is big enough to saturate memory bandwidth on its own, spreading
  it across all cores wins. Running several elements at once instead just thrashes the
  cache. This schedule runs only for `m, n > 1` shapes, whose routes already reduce each
  output within one worker. So it stays reproducible.

The upshot for you is simple: hand the engine the whole batch and let it choose. Many
small independent products are exactly where batching wins over a hand-written loop. A
hand-written loop cannot make the fork-join-once-for-all-elements choice. It either
parallelizes each tiny GEMM, which is mostly overhead, or runs them serially. For a
handful of large products, the batched call converges to what a plain loop already does
well. So batching neither helps much nor hurts there.

Determinism spans all 3 schedules. Every element is independent, so the whole batch is
reproducible across worker counts. The serial and batch-parallel schedules go further:
they are bit-identical across worker counts, because each element runs on exactly one
worker. The few-but-large schedule inherits whichever serial-versus-parallel behavior its
own per-element route has. A zero-length batch is a no-op.

Like the rest of the API, every batched entry has a `_with` variant that reuses a
caller-owned `Workspace`, to avoid a per-call allocation. One detail is worth knowing.
Under the batch-parallel schedule, the packing cannot go through a single shared
`Workspace`, because concurrent workers would collide on it. So that schedule packs
through each worker's own persistent per-thread pool instead, reused across calls the
same way your `Workspace` is.
