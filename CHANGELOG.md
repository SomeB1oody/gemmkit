# Changelog

All notable changes to the gemmkit workspace are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The 5
workspace crates (`gemmkit`, `gemmkit-ndarray`, `gemmkit-nalgebra`, `gemmkit-faer`,
`gemmkit-tune`) share one version and release in lockstep. So releases are recorded
once, with per-crate subsections where a change is crate-specific.

## [0.1.2] - 2026-07-29

### gemmkit

#### Added

- `GEMMKIT_GEMV_TIER_STEP` (`set_gemv_tier_step`, default auto): the byte spacing
  between the rungs of the new gemv/gevv worker ladder
- `GEMMKIT_GEMV_AXPY_PAR_MIN_ROWS` (`set_gemv_axpy_par_min_rows`, default 16384 on
  x86, 1024 on aarch64): the output-row floor below which a column-major gemv stays
  serial instead of splitting its rows
- `Epilogue::apply_tile`, a hook that transforms a whole `MR_REG x NR` register tile
  in 1 call. It defaults to a loop over `apply_reg`, so an existing impl is
  unaffected
- `Debug` on `Bias`, `Activation`, `RequantScale`, and `Requantize`, plus `Copy` and
  `Clone` on the latter 2, so an epilogue config can be built once and reused by
  value

#### Changed

- The tiny-matrix shortcut's depth ceiling (`GEMMKIT_KC`) now resolves against the
  packed element size, and is deeper on x86 (default 512 to 2048). It counts 4-byte
  elements, so `f16` gets 2 times the `f32` depth, `int8` 4 times, and `c64` 1
  quarter, at constant packed panel bytes. On aarch64 the divisor stops at 4 bytes, so
  a wide element keeps the whole ceiling, and the `f32` depth there is unchanged.
  Output stays bit-identical across worker counts, but an affected shape's result bits
  change from the previous release
- The small-`m,n` route's pre-pack copy now runs across workers instead of on the
  calling thread, splitting the depth rather than the few `lead` lines. It resolves
  its own worker count, and stays serial below `GEMMKIT_GEMV_PARALLEL_BYTES`. No
  result bit moves
- The auto worker count for a bandwidth-bound gemv/gevv is now a ladder over the
  touched bytes, not a flat fraction of the core count. Its rungs are the exact-fit
  thread-pool tiers, and it climbs 1 tier per `GEMMKIT_GEMV_TIER_STEP` factor of
  bytes above the serial floor. A non-zero `GEMMKIT_GEMV_THREAD_CAP` still pins 1
  flat width, bypassing the ladder. gemv output remains bit-identical across worker
  counts

#### Fixed

- A column-major gemv no longer splits its output rows across workers below the new
  floor. The output-row axis is the inner memory axis there, so a row split gave
  every worker a strided walk over the whole matrix, and lost to the serial pass
  over most of the practical range. Row-major gemv and the `half` mixed twin are
  untouched, and no result bit moves
- A gemv with fewer output rows than 1 SIMD register no longer takes the axpy
  strategy, whose vector loops were unreachable there, leaving the whole reduction
  on the scalar remainder. This is the `m == n == 1` unit-stride shape, which is how
  the column-major adapters describe a row vector. It now takes the dot strategy,
  which vectorizes over `k`. Those shapes change bits, since the summation order
  changes
- A fused GEMM no longer runs at a fraction of the plain rate. `FusedEpi` decodes
  its bias and activation enums once per output tile, in an `apply_tile` override,
  instead of once per accumulator register, where the branch web spilled the
  accumulator tile to the stack from inside the `kc` loop. A perf test now pins the
  ratio. No result bit moves
- The wasm32 `simd128` `max` and `min` no longer pass their operands in the order
  that inverts the trait contract, which returned NaN for a NaN 1st operand and
  `-0.0` for `max(-0.0, +0.0)`. So a vectorized `ReLU` on wasm maps NaN to 0 and
  `-0.0` to `+0.0`, as the scalar path and every other backend already did

### gemmkit-ndarray, gemmkit-nalgebra, gemmkit-faer

#### Added

- Each adapter now re-exports every gemmkit item its own signatures name, so an
  adapter user no longer needs a direct `gemmkit` dependency. New in this release:
  the `GemmScalar`, `FusedScalar`, `MapScalar`, and `ComplexScalar` element-type
  bounds, the feature-gated element types `f16`/`bf16` and `Complex`/`c32`/`c64`,
  and the `tuning` module. The last one matters beyond convenience: the knobs are
  process-global atomics, so a `set_*` call made through a separately resolved 2nd
  `gemmkit` silently has no effect on the adapter

### gemmkit-tune

#### Changed

- The `GEMMKIT_GEMV_THREAD_CAP` sweep gained a small cache-resident probe shape.
  Both of its previous shapes overflowed cache, so it could not observe the
  size-dependent behavior it calibrates

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
