# Small Shapes and GEMV

The register-blocking driver at the heart of gemmkit needs enough work per output tile. It must amortize packing, cache blocking, and a full `MR x NR` accumulator.

Some shapes break that premise outright. A matrix-vector product has no tile reuse at all. A `k = 4` contraction finishes before the pack pays for itself. An `8 x 8 x 100000` product would spend most of the driver's effort multiplying padding. For shapes like these, the driver is the wrong tool, so the engine quietly routes around it.

There is nothing to opt into. The same `gemm` entry (and `gemm_i8`, `gemm_fused`, `gemm_map`) inspects the shape and strides at the top of dispatch. It reroutes to a special-case kernel when one fits the shape, and it falls through to the general driver otherwise. Every reroute stays behind the same public entry.

Each reroute keeps the library's reproducibility contract: the same call, on the same machine, with the same configuration, returns the same result. Each reroute is also gated by a [tuning knob](Tuning_Knobs.md) that you can move or disable without recompiling. You never call these paths directly. You benefit from writing the shape naturally, instead of hand-rolling a dot loop.

## gemv: the memory-bound edge

A shape with `m == 1` or `n == 1` is a matrix-vector product. Each output element needs only `2k` floating-point operations, but reading it takes `k` matrix elements. The arithmetic is trivial, so the whole problem is minimizing DRAM traffic. The dedicated gemv path handles both cases with one core routine. It views the matrix as `rows x k` times a `k`-vector, transposing the matrix first when `m == 1`. This path is correct for every layout, and it vectorizes the contiguous ones.

This also covers the degenerate `m == n == 1` dot product. A `1 x k` view has both strides equal to 1, so it fits the column-major strategy and the row-major strategy at once. The column-major strategy vectorizes over output rows, but there is only one output row here, too few to fill a SIMD register. So the route sends any sweep shorter than one register to the `k`-vectorizing strategy instead. There is nothing to opt into and nothing to know. A dot product built from `MatRef::from_col_major(a, 1, k)`, the shape a column-major library hands you, runs on the same fast path as the row-major spelling.

2 properties matter to a caller. First, gemv follows the library's general reproducibility contract. The same call, on the same machine, with the same configuration, always returns the same result. Each output element is reduced in a single pass over `k` by one worker.

Splitting the rows across workers changes only which worker does the work, not how the work is done. The library does not extend gemv's guarantee to bitwise agreement across different worker counts. The worker count is part of the configuration, the same as everywhere else in gemmkit.

Second, gemv has its own parallel policy. It is bandwidth-bound, so the worker count comes from a bandwidth model instead of the compute ramp the general driver uses. Past the few cores that saturate DRAM, more workers stop helping, and only add fork/join overhead and shared-cache contention. The count follows a ladder over the bytes a call touches. Below a floor, the matrix fits one core's private cache, so the call stays serial. Past that floor, the width climbs in steps, and it never reaches the full machine width.

4 knobs expose this policy:

- `gemv_parallel_bytes` sets the byte floor below which gemv stays single-threaded.
- `gemv_tier_step` sets how many bytes apart the ladder's steps sit.
- `gemv_thread_cap` sets a flat width that replaces the ladder outright.
- `gemv_threshold` sits alongside the other 3. Since a gemv shape always has `min(m, n) == 1`, this knob works as an on/off switch rather than a graduated cap.

## The small-k path

The contraction `k` can also be too small for packing to pay off. The threshold is `small_k_threshold`, with a default of 16 on x86 and 8 on aarch64. At or below that depth, the whole product is a single depth panel, and every packed element would be read exactly once. Packing has nothing to amortize, so the driver's packing step becomes pure overhead.

The small-`k` route covers these skinny, low-depth shapes: gevv, rank-`k` updates, and tall-skinny products. It computes `C` directly with the family's microkernel, reading `A` and `B` **in place**, unpacked, in one pass. This route inherits the family's widen, bias, conjugate, and rounding behavior for free, and it is bit-identical to the serial run for any worker count. It needs a column-major `A`, meaning unit-stride rows (`rsa == 1`). When `A` does not have that layout, packing rarely amortizes at such a small `k` anyway. The route then defers to the general driver, which still computes the correct result.

## The small-m,n path

The mirror case is a tiny output with a long contraction. Both `m` and `n` sit far below the microtile, at or below `small_mn_dim` (default 16 on x86, 32 on aarch64), while `k` is long. The driver would pad the tiny row and column tiles up to a full microtile. It would then spend most of its work on that padding.

This route instead computes each output as a horizontal dot, `C[i,j] = alpha * <A[i,:], B[:,j]> + beta * C[i,j]`. It streams SIMD along the contraction with no blocking or orientation machinery. It also register-blocks the small output grid, so several independent FMA chains stay in flight.

The horizontal kernel needs both operands unit-stride along `k`. That means `A`'s rows must be contiguous (`csa == 1`, a **row-major A**) and `B`'s columns must be contiguous (`rsb == 1`, a **column-major B**). When both hold, the route reads `A` and `B` in place with zero copies. This is the fast path.

The 2 most common layouts each miss exactly one side. An all-row-major pair fails `rsb`, and an all-column-major pair fails `csa`. When that happens, an internal pre-pack step copies **only the failing operand** once into `k`-contiguous scratch, then runs the same horizontal dot over it. That copy reads roughly `m*k` elements, or `n*k` for the other operand.

That is a fraction of about `1/n` (or `1/m`) of the `m*n*k` work in the product itself. It costs far less than what the horizontal route saves, so a strided small-`m,n` shape still beats falling back to the driver's padded microtile.

A flop count understates that copy. The copy does no arithmetic per byte it moves. The dots do about 2. The copy therefore has less to hide memory latency behind, and it takes a much larger share of the *time* than of the work.

The copy runs across workers on its own, apart from the dots. A small `m, n` leaves a tiny output grid for the dots to split. The copy splits the depth instead, and a deep contraction makes the depth long. You configure nothing. On the reference machine, a column-major small-`m,n` shape with a deep contraction got 1.1-3.1x faster (f32, auto width). The larger gains land where the packed operand still fits cache.

The pre-pack tier engages once `k` clears its own knob, `small_mn_pack_min_k` (default 16), separate from the zero-copy tier's `small_k_threshold`. This path is likewise bit-identical to the serial run at any worker count. It computes each output as one fixed-order reduction on a disjoint tile.

## Practical guidance

Prefer the ordinary entries over a hand-written dot loop. Your problem might be a matrix-vector product, a rank-`k` update, or a grid of small outputs over a long contraction. In each case, `gemm` already carries a kernel tuned for that shape. It comes with the bandwidth-aware threading and the reproducibility guarantees that a hand-written loop would have to reinvent.

Layout is the one lever in your hands. For the horizontal small-`m,n` path, a **row-major A and a column-major B** stream unit-stride and hit the zero-copy fast path. For the small-`k` path, a **column-major A** stays on the in-place route. Any other layout still works, but it pays either the small-`m,n` pre-pack copy or a fall-back to the general driver.

Every threshold named in this chapter is a tunable knob. Each one resolves per call from an argument, a programmatic setter, a `GEMMKIT_*` environment variable, or a calibrated compile-time default, in that order. If a reroute is miscalibrated for your machine, or if you want to force a shape onto the general driver, see the [Tuning Knobs](Tuning_Knobs.md) chapter. It covers `gemv_threshold`, `small_k_threshold`, `small_mn_dim`, `small_mn_pack_min_k`, `gemv_parallel_bytes`, `gemv_tier_step`, `gemv_thread_cap`, and `k_stream_max` in full. For the special-path internals, meaning why each kernel is shaped the way it is, see the architecture chapter's [Special Paths](../architecture/Special_Paths.md).
