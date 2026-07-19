# Batched GEMM

A single large GEMM saturates a modern CPU on its own; a crowd of tiny ones does not. Attention heads, grouped convolutions, per-sample linear layers, and block-diagonal solves all produce many small independent products, and running them as a plain loop of `gemm` calls leaves most of the machine idle: each call is too small to parallelize usefully, yet the loop pays a fork/join, or serializes, once per element. The batched entries take the whole set in one call and schedule it as a unit, assigning whole GEMMs to workers so a batch of small matrices actually fills the cores.

Batched GEMM is an orchestration layer, not a new kernel. Each element re-dispatches through the full single-GEMM engine, so a batch composes automatically with the driver, the gemv path, and the [small-shape paths](Small_Shapes_and_GEMV.md): a batch of `1 x 1 x k` products runs the horizontal dot inside each element, a batch of ordinary shapes runs the register-blocked driver. Every element is an independent GEMM, and the whole batch is reproducible across worker counts.

## The strided form

When the elements are laid out at a regular stride, one matrix after another in a flat buffer, [`gemm_batched`](https://docs.rs/gemmkit) takes the single-element shape and strides once, plus a batch stride for each of `A`, `B`, and `C`. Element `b` is based at `A + b*a_batch_stride`, `B + b*b_batch_stride`, `C + b*c_batch_stride`, all sharing the same shape:

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

A batch stride of `0` broadcasts one operand across the whole batch, valid for the read-only `A` or `B` (one shared weight matrix against a batch of inputs) but never for `C`, whose elements are written concurrently and must stay disjoint. The result reproduces a loop of `gemm` calls exactly. Under the `epilogue` feature, `gemm_batched_fused` applies **one** shared bias and **one** shared activation to every element, the batched-linear-layer case, and reproduces a loop of [`gemm_fused`](Fused_Epilogues.md) calls; the single bias vector is sized for one element, not the whole batch.

## The slice form: per-element shapes

When the elements differ in shape, or simply do not sit at a fixed stride, [`gemm_batched_slice`](https://docs.rs/gemmkit) takes a slice of [`BatchProblem`](https://docs.rs/gemmkit), each carrying its own `alpha`, `A`, `B`, `beta`, and a distinct `&mut` `C` view:

```rust
use gemmkit::{gemm_batched_slice, BatchProblem, MatRef, MatMut, Parallelism};

let mut problems: Vec<BatchProblem<'_, f32>> = /* one per product, each its own shape */;
gemm_batched_slice(&mut problems, Parallelism::Rayon(0));
```

Because every `C` is a distinct `&mut`, the outputs are pairwise disjoint and cannot alias the inputs by construction, so validation only checks per-element shape agreement and in-bounds strides. This is the form to reach for when your matrices already live as a `Vec` of views. Its raw counterpart, `gemm_batched_ptr_unchecked` over a slice of `GemmProblem`, takes the same per-element shapes as bare pointers for FFI and adapters that validate their own inputs and may use arbitrary or negative strides; both are covered in [The Unchecked Tier](The_Unchecked_Tier.md).

## How the batch is scheduled

The interesting decision is how the work is spread across cores, and the engine makes it once per call from the shared shape and the batch size. There are three schedules:

- **Batch-parallel.** Workers pull disjoint ranges of elements from a shared cursor, and each element runs *serially* on one worker, cache-hot. This is the whole point of the API: for many small matrices it pays a single fork/join for the entire batch instead of one per element, and it keeps every core busy on complete GEMMs. Because no element is ever split across workers, this schedule is bit-identical to the serial run for any worker count.
- **Serial.** The whole batch runs on the calling thread, each element single-threaded. Chosen when there is too little total work to justify a fork/join.
- **Sequential with internal parallelism.** For the few-but-large, DRAM-bound regime, the batch is looped and each element is handed the *full* engine parallelism in turn. When the elements are big enough that one already saturates memory bandwidth, spreading a single element across all cores beats running several at once and thrashing the cache. This schedule is used only for `m, n > 1` shapes, whose routes reduce each output within one worker, so it stays reproducible.

The upshot for you is simple: hand the engine the whole batch and let it choose. Many small independent products are exactly where batching wins over a hand-written loop, because the loop cannot make that fork/join-once-for-all-elements choice, it either parallelizes each tiny GEMM (mostly overhead) or runs them serially. For a handful of large products the batched call converges to what a plain loop already does well, so batching neither helps much nor hurts there.

Determinism spans all three schedules. Elements are independent, so the batch is reproducible across worker counts under a fixed configuration; the serial and batch-parallel schedules are additionally bit-identical across worker counts because each element runs serially, and the few-but-large schedule inherits the per-element serial-equals-parallel behavior of the route it runs. A zero-length batch is a no-op.

Like the rest of the API, every batched entry has a `_with` variant that reuses a caller-owned `Workspace` to avoid per-call allocation. One detail is worth knowing: under the batch-parallel schedule the packing cannot go through a single shared `Workspace` (concurrent workers would collide on it), so that schedule packs through each worker's own persistent per-thread pool instead, reused across calls the same way your `Workspace` is.
