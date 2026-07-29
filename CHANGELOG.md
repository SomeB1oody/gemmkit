# Changelog

All notable changes to the gemmkit workspace are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 5
workspace crates (`gemmkit`, `gemmkit-ndarray`, `gemmkit-nalgebra`, `gemmkit-faer`,
`gemmkit-tune`) share one version and release in lockstep. So releases are recorded
once, with per-crate subsections where a change is crate-specific.

## [Unreleased]

### gemmkit

#### Added

- `GEMMKIT_GEMV_TIER_STEP` (`set_gemv_tier_step`, default auto): the byte spacing
  between the rungs of the new gemv/gevv worker ladder
- `GEMMKIT_GEMV_AXPY_PAR_MIN_ROWS` (`set_gemv_axpy_par_min_rows`, default 16384 on
  x86, 1024 on aarch64): the output-row floor below which a column-major gemv stays
  serial instead of splitting its rows

#### Changed

- The tiny-matrix shortcut's depth ceiling (`GEMMKIT_KC`) is deeper on x86, and it now
  scales with the element size. The x86 default goes from 512 to 2048. The ceiling
  counts 4-byte elements, and the shortcut divides it by the packed element size. So
  `f64` takes half the depth of `f32`, and `int8` takes 4 times as much, while the
  packed panel bytes stay the same. The aarch64 default is unchanged, and so is the
  `f32` depth it gives.

  A shape with both `m` and `n` at or below `GEMMKIT_TINY_BLOCK_DIM` (default 64) runs
  `k` in slices of this depth. Each extra slice re-reads and re-writes `C`, re-enters
  the driver, and forks the workers once more. A long `k` made that per-slice cost
  dominate. A deeper slice also grows the packed A and B panels, which need to stay in
  a private L2, so the depth cannot grow without a limit. The new x86 budget puts the
  widest shortcut shape's panels at about 1.1 MiB, which is one Zen5 L2.

  Measured on the Zen5 reference machine over 8 shapes in the band, against the old
  512-element budget. The parallel path is 1.06 to 2.3 times faster for `f32`, `f64`,
  `f16`, and `int8`. The serial path is neutral to 11 percent faster, except at `m` and
  `n` of 64, where it loses 2 to 4 percent. One step deeper would cost the serial path
  12 to 20 percent for `f32` and `f64`.

  The 1 element-size rule replaces a single depth shared by every family. That shared
  depth over-blocked `f64` by 2 times and under-blocked `int8` by 4 times. A tuner
  sweep run on `f32` now transfers to the other families. gemmkit-tune's own
  `GEMMKIT_KC` probe changed with it. Its candidates now straddle the default, and its
  shapes are deep enough to hold 8 or more slices. It also scores the serial and the
  parallel mode together, because the 2 modes pull the ceiling in opposite directions.

  Blocking still depends only on the cache geometry and the problem shape, never on
  the worker count. So output remains bit-identical across worker counts. The depth
  slicing sets the summation order, so an affected shape's result bits change from the
  previous release

- The small-`m,n` route's pre-pack copy now runs across workers. Before, it ran on
  the calling thread. The copy resolves its own worker count, apart from the tile
  sweep that follows. The 2 axes offer different amounts of parallelism. The
  `MT x NT` output grid caps the sweep, and a small `m, n` makes that grid tiny.
  The depth caps the copy, and a long `k` makes the depth large. The copy splits
  the depth, not the few `lead` lines, so each worker reads whole depth lines.
  Below the cache-derived byte floor that the bandwidth-bound routes share
  (`GEMMKIT_GEMV_PARALLEL_BYTES`), the copy still runs on the calling thread.
  Measured on the Zen5 reference machine, f32 at the auto width, with a
  column-major `A`: `8x8x524288` 3.1x, `16x16x262144` 2.0x, `4x4x1048576` 1.8x,
  `8x8x2097152` 1.7x, `16x16x1048576` 1.1x. The gain tracks the copy's share of
  the route's serial time. Footprint is the one trend that holds across these
  shapes. At a fixed `m,n`, the smaller packed operand gains more. A pack writes
  each cell once, with the value a serial copy writes. No result bit moves, and
  the route stays bit-identical across worker counts

- The auto worker count for a bandwidth-bound gemv/gevv is now a ladder over the
  touched bytes, not a flat fraction of the core count. Its rungs are the
  exact-fit thread-pool tiers that the compute path already snaps to, so a gemv
  width always has a pool sized to it. It climbs one tier per
  `GEMMKIT_GEMV_TIER_STEP` factor of bytes above the serial floor, and stops at the
  largest tier instead of the full machine width. `GEMMKIT_GEMV_THREAD_CAP` is
  unchanged. A non-zero value still pins one flat width, now bypassing the ladder.
  gemv output remains bit-identical across worker counts

#### Fixed

- A column-major gemv no longer splits its output rows across workers below that
  floor. For a column-major matrix, the output-row axis is the *inner*,
  fastest-varying memory axis. So any row split gives every worker a strided walk
  over the whole matrix, while the serial route makes one sequential pass. That
  made parallelism a net loss over most of the practical range. The floor restores
  the serial route below the crossover, and leaves everything above it unchanged.

  Row-major gemv is untouched. It splits into `k`-contiguous rows, and stays faster
  in parallel at every size. So does the `half` mixed twin. No result bit moves
  either way. The floor decides only whether the rows are split, and gemv output
  is bit-identical across worker counts, as before. The aarch64 default is 1024,
  an order of magnitude below the x86 one, because the same regression covers a
  much narrower band there. On that target the crossover also follows the bytes in
  one column rather than the row count, so an `f64`-dominated workload there
  wants 512
- A gemv sweep with fewer output rows than one SIMD register no longer takes the
  axpy strategy. That strategy vectorizes over output rows, so below one register
  its vector loops were unreachable, and the whole reduction ran on the scalar
  remainder instead.

  This case only ever arose where both classifications fit at once: a single-row
  matrix, whose row and column strides are both 1. That is the pure dot-product
  shape `m == n == 1`. It is also how the column-major libraries (nalgebra, faer)
  naturally describe a row vector, so the adapters hit it by default. Such a sweep
  now takes the dot strategy instead, which vectorizes over `k`. The dot form's
  wider accumulator tree is also the more accurate of the 2.

  The affected shapes change bits, since the summation order changes. gemv output
  remains bit-identical across worker counts, and no other shape changes route.
  The same fix applies to the `half` mixed-precision twin

### gemmkit-ndarray, gemmkit-nalgebra, gemmkit-faer

#### Added

- Each adapter now re-exports every gemmkit item that its own signatures name, so
  an adapter user no longer needs a direct `gemmkit` dependency. New in this
  release:
  - the `GemmScalar`, `FusedScalar`, `MapScalar`, and `ComplexScalar` element-type
    bounds, which appear in the adapters' own public signatures. A caller could
    not name them from outside before this. So a caller could call `gemm`, but
    could not write a wrapper generic over it
  - the feature-gated element types `f16`/`bf16` (`half`) and `Complex`/`c32`/`c64`
    (`complex`), so `half` and `num-complex` also stay out of the caller's
    manifest
  - the `tuning` module

  Reaching `tuning` through the adapter is the correct route, not only the
  convenient one. The knobs are process-global atomics. A `set_*` call made
  through a separately resolved second `gemmkit` writes a copy that the adapter
  never reads. That failure is silent: the knob appears set, but has no effect

### gemmkit-tune

#### Changed

- The `GEMMKIT_GEMV_THREAD_CAP` sweep gained a small cache-resident probe shape.
  Its 2 previous shapes both touched a working set too large to fit in cache. So
  the sweep could not observe the size-dependent behavior it is meant to
  calibrate

## [0.1.1] - 2026-07-24

### gemmkit

#### Changed

- Recalibrated the bandwidth-bound (gemv/gevv) parallelism defaults. The serial
  floor dropped from half the per-core L3 to the per-core private L2. So a
  repeatedly scanned matrix in the L2-to-L3 band now parallelizes, instead of
  running single-threaded. The auto worker cap rose from a quarter to half the
  logical core count. Both stay auto-derived, and overridable with
  `GEMMKIT_GEMV_PARALLEL_BYTES` and `GEMMKIT_GEMV_THREAD_CAP`. gemv output
  remains bit-identical across worker counts

### gemmkit-ndarray, gemmkit-nalgebra, gemmkit-faer

#### Added

- Re-export `Parallelism` and `Workspace` from each adapter. Callers no longer
  need a direct `gemmkit` dependency to name the parallelism argument that every
  entry takes, or the workspace that the `*_with` variants reuse

## [0.1.0] - 2026-07-24

Initial release.

### gemmkit

#### Added

- f32/f64 GEMM (`C <- alpha*A*B + beta*C`) over strided views. 3 API tiers:
  checked slice entries (`gemm`), explicit-workspace variants (`*_with`), and raw
  pointer entries that accept negative strides (`*_unchecked`, `*_unchecked_with`)
- Runtime ISA dispatch with a portable scalar fallback. It covers x86-64 FMA and
  AVX-512F, aarch64 NEON, and wasm32 `simd128` (a compile-time feature). AVX-512F
  also adds AVX-512 VNNI `vpdpbusd` for `int8` and AVX-512 BF16 `vdpbf16ps` for
  `half`. The `GEMMKIT_REQUIRE_ISA` env knob pins or forbids a kernel end to end
- Element-type families behind cargo features:
  - `half` (`f16`/`bf16` with f32 accumulation)
  - `int8` (`i8 -> i32` with documented wrapping semantics)
  - `complex` (`c32`/`c64` split-layout kernel with per-operand conjugation)
- `epilogue` feature, adding:
  - fused bias plus activation (`gemm_fused*`, batched and prepacked variants)
  - integer requantization to `i8`/`u8` with per-tensor or per-row scales
    (`gemm_i8_requant*`)
  - a user-supplied per-element closure (`gemm_map*`)
  - bias-only complex fusion (`gemm_cplx_fused*`)

  Fused entries are bitwise-identical to the equivalent unfused call followed by
  a map. The exception is the `f16`/`bf16` entries, which apply the epilogue
  before the final narrowing step, for one rounding instead of two (documented
  on the entries)
- Prepacked operand reuse: `prepack_rhs`/`prepack_lhs` with
  `gemm_packed_b`/`gemm_packed_a` consumers for fixed-weight inner loops, plus
  the `int8` twin `prepack_rhs_i8`/`gemm_i8_packed_b`. It is bit-identical to
  plain `gemm_i8`. Its layout is pinned to the selected integer kernel, so the
  VNNI `vpdpbusd` path skips its otherwise-mandatory per-call RHS repack
- Deep-contraction reblocking for `f16`/`bf16`. At large `k`, the narrow single
  depth panel (`kc = k`) streams an L2-overflowing RHS micropanel from L3 or
  DRAM. Above an auto-derived engage gate (`GEMMKIT_DEEP_KC_BYTES`, default half
  the detected L2), the dispatch instead runs an f32-output twin
  (`MixedGemmF32`/`Bf16DotGemmF32`). That twin re-blocks `K` at the cache-model
  `kc` into an f32 scratch, and narrows once. It is byte-for-byte the single
  panel for `beta in {0, 1}`, and held to tolerance otherwise. Shallow `k` is
  unchanged
- Batched GEMM (`gemm_batched*`) with an internal per-batch parallel policy
- Bandwidth-bound special paths (gemv/gevv, small-k, and the small-m,n
  inner-product route), selected automatically behind the same entry points. The
  small-m,n route also covers strided layouts (all-row-major, all-col-major).
  Above `GEMMKIT_SMALL_MN_PACK_MIN_K`, it copies only the operand strided along
  `k` into a padded, `k`-contiguous scratch, and runs the same horizontal dot.
  That packed route is bit-identical to the unit-stride layout
- `parallel` feature (rayon, default), with reproducible run-to-run results for a
  fixed input and configuration, plus `wasm_threads` for `wasm32-wasip1-threads`
- `no_std` operation with default features off (needs only `core` + `alloc`)
- Cache-topology detection (x86 CPUID, Linux sysfs) feeding BLIS-style
  analytical blocking, `GEMMKIT_*` env tuning knobs, and reusable packing
  workspaces (`Workspace`)

### gemmkit-ndarray, gemmkit-nalgebra, gemmkit-faer

#### Added

- Zero-copy adapters over each library's native matrix views (C-order, F-order,
  general and reversed strides). They mirror the full core surface, including
  the `half`/`int8`/`complex` families and the `epilogue` fused entries.

  Batched GEMM is exposed in the shape each library's types allow. The ndarray
  adapter has a 3-D strided `gemm_batched`/`dot_batched` (and the fused twin).
  The nalgebra and faer adapters instead take `gemm_batched` over a slice of
  per-element `(A, B)` inputs, paired with a slice of `&mut C` outputs. That form
  runs over gemmkit's pointer-array batched engine, with heterogeneous
  per-element shapes, since neither library has a rank-3 type

### gemmkit-tune

#### Added

- Install-time autotuner binary: sweeps the runtime knobs on the target machine
  and emits a ready-to-source `GEMMKIT_*` env profile
