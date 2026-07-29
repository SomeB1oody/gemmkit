# Special Paths

The register-tiling driver described in [Life of a GEMM Call](Life_of_a_GEMM_Call.md) is built around 1 assumption. It assumes there is enough work per output tile to amortize packing, blocking, and a full `MR x NR` register accumulator. Some shapes break that assumption badly. A matrix-vector product has no tile reuse at all. A `k = 4` product finishes before the pack pays for itself. An `8 x 8 x 100000` contraction would spend most of the driver's effort multiplying zero padding.

Layer L6 (`gemmkit/src/special/`) reroutes exactly these shapes to dedicated kernels. Every reroute holds 3 properties. It hides behind the same public entries, so `gemm` and its siblings never expose which path ran. It is threshold-gated through [tuning knobs](../gemmkit-guide/Tuning_Knobs.md), so a mis-calibrated gate can move or turn off without a recompile. It also keeps the library's reproducibility contract: the same machine and the same configuration give a reproducible result. Most of the routes below go further and stay bit-identical across worker counts for a fixed shape. gemv is the exception. Its own section below explains why.

The gates live at the top of each per-type dispatch entry, in a fixed order. This is `run_typed` from `gemmkit/src/dispatch/float.rs`, trimmed for clarity:

```rust
// gemmkit/src/dispatch/float.rs (run_typed, trimmed)
if (t.n == 1 || t.m == 1) && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold() {
    gemv::run_typed_epi::<T, S, Identity>(/* user frame, before orientation */);
    return;
}
orient_transpose(&mut t);
if small_mn_eligible(&t) || small_mn_pack_eligible(&t) {
    small_mn::run_epi::<T, S, Identity>(/* horizontal dot kernel */);
    return;
}
if t.k <= tuning::small_k_threshold() {
    small_k::run::<FloatGemm<T>, S, MR_REG, NR>(/* one depth panel, in place */);
    return;
}
driver::run::<FloatGemm<T>, S, MR_REG, NR>(/* the general blocked driver */);
```

gemv fires before orientation normalization, in the user's coordinate frame. The other gates run on the oriented problem. Each special path also exists in a fused-epilogue form, so a `gemm_fused` call takes the same route its plain twin would. That contract is the subject of [Epilogue Fusion](Epilogue_Fusion.md).

## gemv: the memory-bound edge

A shape with `m == 1` or `n == 1` (`gemmkit/src/special/gemv.rs`) does `2k` flops per output element, against `k` matrix elements read once. This makes it memory-bound. The whole design question is how to cut DRAM traffic, not how to schedule FMAs. Both orientations reduce to 1 core routine. It views the matrix, transposed when `m == 1`, as a `rows x k` block times a `k`-vector.

The gate checks shape, not size. `min(m, n)` is 1 for any gemv shape, so the `GEMMKIT_GEMV_THRESHOLD` knob it compares against works as an on/off switch rather than a size cap. Set it to 0 to force gemv shapes onto the general driver instead, which still produces a correct result.

Parallelism follows a bandwidth model, not the compute ramp. `Parallelism::resolve_bandwidth` stays serial below a cache-derived byte floor. Below that floor, the matrix fits inside 1 core's private cache, and that core saturates it alone. Above the floor, the resolver steps straight to a width sized for the bytes touched. That width climbs a ladder over the exact-fit pool tiers. It tops out at half the logical cores, because a handful of workers is the worst point on a bandwidth scaling curve.

`row_sweep` partitions output rows across workers in panels whose grain is a multiple of the SIMD width. Each row's full `k`-reduction stays inside 1 worker, and no worker ever merges a partial result from another. This keeps gemv reproducible at a fixed worker count, the same bar the rest of the engine holds to. Splitting the rows across more or fewer workers does not, by itself, promise bitwise agreement across those different worker counts. small-k, small-mn, and batched, below, go further than that.

Inside a worker's row range, the code picks 1 of 4 strategies based on layout. A column-major matrix takes the axpy shape. It offers 2 variants that are deliberately bit-identical and differ only in memory traffic.

The **register-blocked output** form holds a panel of output rows in SIMD registers across the whole `k`-sweep. It reads the matrix and the output exactly once each. The **plain column-outer** form re-reads the output every few columns, but it streams the matrix as 1 contiguous read. The choice between them depends on output cache residency, computed in `output_register_block`. The route picks register-blocking under 2 conditions. First, the output (`rows * sizeof`) must outgrow a fraction of the last-level cache, so the plain form's re-reads would otherwise reach DRAM. Second, `k` must sit at or below `GEMMKIT_K_STREAM_MAX` (default 32). Past that `k`, the register-blocked form's many concurrent column streams start to thrash the hardware prefetcher. Both variants run the same ascending-`k` fused accumulation per element, and the same per-row SIMD-versus-scalar split. Switching between them never changes a single output bit. It only changes speed.

A row-major matrix takes a dot form instead. It register-blocks rows in groups of 4 to overlap FMA latency chains, while each row still runs the shared, fixed-order `dot_contiguous` reduction. Fully strided operands fall back to a scalar loop.

One shape fits 2 of those classifications at once, and the way that tie breaks decides whether anything vectorizes at all. The axpy forms vectorize over **output rows**, holding `lanes` of them in a register. The dot form vectorizes over **`k`** instead. A matrix of a single row has both its strides equal to 1, so column-major and row-major describe the same bytes equally well. This is the pure dot product, `m == n == 1`.

Handing that shape to the axpy form leaves its vector loop unreachable, since `while i + lanes <= e` never holds when `e == 1`. The whole reduction then falls onto the scalar remainder. `axpy_yields_to_dot` avoids this. It gives the sweep to the dot form whenever the row count is short of 1 SIMD register, and the dot form's own strides hold. The dot form also carries a wider accumulator tree, so it is the more accurate of the two. This choice matters in practice. Column-major adapter libraries such as nalgebra and faer describe a row vector exactly this way. A caller reaching for a dot product would otherwise land on the slow classification by default.

Whether the rows split across workers is a separate decision from which strategy computes them. For a column-major matrix the answer is usually no. The output-row axis is that matrix's **inner, fastest-varying memory axis**. Cutting it hands every worker a strided walk over the entire matrix, where each worker consumes only its own slice of every column. The serial route instead makes 1 sequential pass: `row_sweep` short-circuits to a single `body(0, rows)` call, with no blocking at all. That pass already runs near the achievable single-stream rate. Below some row count, extra workers have little to win and a great deal of sequentiality to lose.

`GEMMKIT_GEMV_AXPY_PAR_MIN_ROWS` holds that row-count floor. Below it, the axpy split stays serial regardless of the requested worker count. This floor deliberately exempts 2 routes. A row-major matrix gives each worker whole `k`-contiguous rows, so its stream stays sequential even when split, and splitting it keeps winning at every size. The mixed-precision twin's widening axpy is compute-bound enough to scale on the same column-major stream, so it keeps splitting worthwhile too.

The mixed-precision twin, `run_mixed` (feature `half`), serves `f16`/`bf16` gemv. It uses the same row partition and the same reproducibility argument as the float routine above. Every load widens to `f32` through the `KernelSimd<N, N, f32, N>` seam, and the reduction runs in `f32`. The result rounds to the narrow type exactly once, at the store.

A single asymmetry follows from that single-rounding rule. The mixed axpy always register-blocks the output. The plain column-outer form would re-read and re-write the narrow output every column group. That would round it once per group, instead of once per element.

The mixed *fused* gemv does not route here at all, on purpose. The float fused gemv fuses by re-reading each stored output and mapping it in place. That is bit-exact only because the float output *is* the accumulator. A narrow output has already rounded once at the store, so reading it back and mapping it again would round it twice. Instead of threading the epilogue through every widening store, the mixed fused entry keeps gemv shapes on the general driver. The driver already applies the epilogue in `f32`, before its single narrowing step (`gemmkit/src/dispatch/mixed.rs`).

## small-k: one depth panel, nothing to amortize

At a tiny `k` (`gemmkit/src/special/small_k.rs`), the whole product is a single depth panel. The driver's cache-blocking model, its workspace carving, and above all its A/B packing would all be pure setup. Every packed element would be read only once anyway. This route instead computes `C <- alpha*A*B + beta*C` directly over the family's microkernel, with `kc = k`. It reads A and B **in place**: no packing, no blocking, no workspace traffic. It still inherits the family's widen and rounding semantics for free, because it stays generic over `KernelFamily`.

The gate is `k <= GEMMKIT_SMALL_K_THRESHOLD`. Its default splits by architecture: 16 on x86, 8 on aarch64. The narrower NEON microkernel tile packs cheaply enough that the driver wins sooner there, hence the lower default.

The in-place read needs 3 preconditions. When any of them fails, the route defers to the driver instead, and the result stays correct, just differently scheduled. First, the microkernel needs unit-stride LHS rows, so A must be column-major (`rsa == 1`). Second, a `FORCE_PACK_*` family, such as complex, transforms the data into a planar form while packing, so it cannot be read in place by construction. Third, `k` past a hard `SMALL_K_MAX = 32` would overflow the one stack buffer this route uses: a zero-padded panel for the bottom, partial row-tile. That panel still needs packing, because the microkernel always loads a full `mr` rows.

The route partitions work over output tiles, one full `k`-pass per tile, with 1 worker per tile. The bandwidth model caps the worker count itself, since the `m*n` output write dominates at a small `k`. Every tile is a complete reduction owned by a single worker. So the result stays bit-identical across worker counts, a stronger property than gemv holds to.

## small-mn: horizontal dots for tiny outputs

Both `m` and `n` can sit far below the microtile while `k` stays long (`gemmkit/src/special/small_mn.rs`). There, the driver would pad the tiny row and column tiles up to full `MR x NR` microtiles, and compute mostly padding. This route instead computes each output element as 1 horizontal SIMD dot, `C[i,j] = alpha*<A[i,:], B[:,j]> + beta*C[i,j]`, streaming along the contraction. The output is register-blocked into `4 x 4` tiles of accumulators, so 16 independent FMA chains stay in flight across the `k`-sweep. Each A-row and B-column loads only once per tile. This is the same latency-hiding trick as gemv's dot form, generalized to a small grid.

The dims gate is `m, n <= GEMMKIT_SMALL_MN_DIM` (default 16, 32 on aarch64), together with `k` above the small-k threshold. The cap is arch-split because the point where the driver's padding starts to cost more than this route's horizontal dots differs by machine. The 2 small-shape routes split the `k` axis between them this way.

The kernel needs both operands unit-stride along `k`. It wants A's rows contiguous (`csa == 1`, row-major A) and B's columns contiguous (`rsb == 1`, column-major B). This is the zero-copy tier. The 2 most common layouts each fail exactly 1 side: all-row-major fails B, and all-column-major fails A. For these, a second, mutually exclusive gate (`k > GEMMKIT_SMALL_MN_PACK_MIN_K`, default 16) engages a pre-pack tier. `prepack_operands` copies **only the failing operand** into a `k`-contiguous workspace scratch buffer, then runs the identical kernel over it with unit strides.

The copy touches `m*k` (or `n*k`) elements, against the `m*n*k` work of the dot itself. This is a small tax next to the horizontal kernel's win, so a strided small-`m,n` shape still beats falling back to the driver's padded microtiles. The scratch buffer rounds each line up to an *odd* number of cache lines (`packed_line_stride`). A natural stride of exactly `k` would map every packed line to the same L1 set whenever `k` is a power of 2. That collapses the benefit of the re-reads, so the odd-line rounding avoids it.

That tax is small in *flops*, but flops are the wrong unit for it. The copy does no arithmetic per byte it moves. The dots do about 2. The copy therefore has less to hide memory latency behind. On a long `k` it takes a far larger share of the *time* than of the work. On the calling thread it dominated the route.

The copy now runs across workers itself. It forks after its traffic clears the cache-derived byte floor that the bandwidth-bound routes share (`GEMMKIT_GEMV_PARALLEL_BYTES`). Below that floor the copy stays serial. It resolves its own worker count, apart from the tile sweep that follows. The 2 axes offer different amounts of parallelism. The `MT x NT` output grid caps the sweep, and a small `m, n` makes that grid tiny. The depth caps the copy, and a long `k` makes the depth large.

The copy splits the depth, never the `lead` axis. A contiguous `t` range lets each worker read whole depth lines, `lead` consecutive elements per step. A split of the few `lead` lines would instead take a single element per step from each line, across the whole operand.

Measured on the Zen5 reference machine (f32, auto width, column-major `A`): `8x8x524288` 3.1x, `16x16x262144` 2.0x, `4x4x1048576` 1.8x, `8x8x2097152` 1.7x, `16x16x1048576` 1.1x. Footprint is the one trend that holds across these shapes. At a fixed `m,n`, the smaller packed operand gains more. The order across `m,n` at a fixed footprint is not monotonic. This page therefore claims no mechanism beyond the copy's share of the serial time.

The pre-pack step is a pure reorder: the same values, in the same per-line order. So the packed route stays bit-identical to the eligible-layout route. Each worker writes every cell once, with the value a serial copy writes. A split of the copy across workers therefore moves no byte either. This route has 2 siblings. One is mixed (`f16`/`bf16`, widen to `f32`, round once per cell). The other is integer (`i8 -> i32`, wrapping, and therefore bit-exact against the driver). Both share the same tiling, the same pre-pack helper, and the same reproducibility argument.

## batched: orchestration, not a kernel

Batched GEMM (`gemmkit/src/special/batched.rs`) is deliberately not a new kernel. Every element re-dispatches through the full single-GEMM engine, so a batch composes with the driver, gemv, small-k, and small-mn automatically. What this layer adds is a schedule, chosen once per call by `Parallelism::resolve_batch`:

- **`BatchParallel`**: chosen when there are enough elements to fill the workers. Each worker runs whole GEMMs serially and cache-hot, and the batch pays 1 fork/join instead of 1 per element. This is the model for the motivating workload: many small matrices.
- **`SequentialInternal`**: chosen for few, large, DRAM-bound elements. It loops the batch on 1 thread and gives each element the full engine parallelism in turn. On x86 the split engages once an element spills the per-core L2. aarch64 shares an L2 cache across a cluster and has high unified bandwidth. There, it engages once the per-batch-worker share, `elem_bytes / batch`, exceeds `GEMMKIT_SEQ_INTERNAL_BYTES_PER_WORKER` (default 128 KiB). This plan splits 1 element's own work across workers, so it is gated to `m, n > 1` shapes. The driver, small-k, and small-mn routes all reduce every output within 1 worker. So they agree bit-for-bit between serial and parallel, under the current, thread-independent blocking. gemv is held only to the base reproducibility promise, not to that bitwise agreement, so this plan excludes it.
- **Serial**: chosen below the total-work gate, or when no threads are usable.

Every element is independent, and the serial and batch-parallel plans never split one across workers. So a batch stays bit-identical across worker counts under those 2 plans, whatever the per-element route. The strided-batched entries, `gemm_batched` and `gemm_batched_fused`, thread 1 shared epilogue through the same skeleton and share a single schedule implementation. The plain and fused forms cannot drift apart.

For a batch whose elements differ in shape, the pointer-array form `gemm_batched_ptr_unchecked` takes a slice of `GemmProblem` descriptors instead. Each descriptor carries its own dimensions, strides, and pointers. This form uses the simpler `resolve_batch_flat` policy instead. It hands whole GEMMs to workers, and never splits a single element, since there is no uniform per-element residency to test. `gemm_batched_slice` is its safe, validated twin.

[Batched GEMM](../gemmkit-guide/Batched_GEMM.md) and [Small Shapes and GEMV](../gemmkit-guide/Small_Shapes_and_GEMV.md) cover the user-facing view of all this. [Parallel Execution](Parallel_Execution.md) describes the worker-count machinery these paths depend on.
