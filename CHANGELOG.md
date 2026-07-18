# Changelog

All notable changes to the gemmkit workspace are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The five
workspace crates (`gemmkit`, `gemmkit-ndarray`, `gemmkit-nalgebra`, `gemmkit-faer`,
`gemmkit-tune`) share one version and release in lockstep, so releases are recorded
once with per-crate subsections where a change is crate-specific.

## [0.1.0] - Unreleased

Initial release.

### gemmkit

#### Added

- f32/f64 GEMM (`C <- alpha*A*B + beta*C`) over strided views, in 3 API tiers:
  checked slice entries (`gemm`), explicit-workspace variants (`*_with`), and raw
  pointer entries accepting negative strides (`*_unchecked`, `*_unchecked_with`)
- Runtime ISA dispatch with a portable scalar fallback: x86-64 FMA and AVX-512
  (plus AVX-512 VNNI `vpdpbusd` for `int8` and AVX-512 BF16 `vdpbf16ps` for
  `half`), aarch64 NEON, and wasm32 `simd128` (compile-time feature); the
  `GEMMKIT_REQUIRE_ISA` env knob pins or forbids a kernel end to end
- Element-type families behind cargo features: `half` (`f16`/`bf16` with f32
  accumulation), `int8` (`i8 -> i32` with documented wrapping semantics), and
  `complex` (`c32`/`c64` split-layout kernel with per-operand conjugation)
- `epilogue` feature: fused bias + activation (`gemm_fused*`, batched and
  prepacked variants), integer requantization to `i8`/`u8` with per-tensor or
  per-row scales (`gemm_i8_requant*`), a user-supplied per-element closure
  (`gemm_map*`), and bias-only complex fusion (`gemm_cplx_fused*`); fused
  entries are bitwise-identical to the equivalent unfused call followed by a
  map, except the `f16`/`bf16` ones, which apply the epilogue before the final
  narrowing (one rounding, more precise; documented on the entries)
- Prepacked operand reuse: `prepack_rhs`/`prepack_lhs` with
  `gemm_packed_b`/`gemm_packed_a` consumers for fixed-weight inner loops, plus
  the `int8` twin `prepack_rhs_i8`/`gemm_i8_packed_b` (bit-identical to plain
  `gemm_i8`; the layout is pinned to the selected integer kernel, so the VNNI
  `vpdpbusd` path skips its otherwise-mandatory per-call RHS repack)
- Deep-contraction reblocking for `f16`/`bf16`: at large `k` the narrow single
  depth panel (`kc = k`) streams an L2-overflowing RHS micropanel from L3/DRAM,
  so above an auto-derived engage gate (`GEMMKIT_DEEP_KC_BYTES`, default half the
  detected L2) the dispatch runs an f32-output twin (`MixedGemmF32`/
  `Bf16DotGemmF32`) that re-blocks `K` at the cache-model `kc` into an f32 scratch
  and narrows once. Byte-for-byte the single panel for `beta in {0, 1}`, held to
  tolerance otherwise; measured 2.8x-3.6x (`f16`) and up to +32% (`bf16`) at
  `k = 32768`/`65536` on the Zen5 9950X, with shallow `k` unchanged
- Batched GEMM (`gemm_batched*`) with an internal per-batch parallel policy
- Bandwidth-bound special paths (gemv/gevv, small-k, and the small-m,n
  inner-product route) selected automatically behind the same entry points. The
  small-m,n route also covers strided layouts (all-row-major, all-col-major):
  above `GEMMKIT_SMALL_MN_PACK_MIN_K` it copies only the operand strided along
  `k` into a padded `k`-contiguous scratch and runs the same horizontal dot,
  bit-identical to the unit-stride layout and measured 1.4x-6.8x over the driver
  fallback it replaces (Zen5 9950X, `4x4`-`16x16`)
- `parallel` feature (rayon; default) with reproducible run-to-run results for a
  fixed input/config, plus `wasm_threads` for `wasm32-wasip1-threads`
- `no_std` operation with default features off (needs only `core` + `alloc`)
- Cache-topology detection (x86 CPUID, Linux sysfs) feeding BLIS-style
  analytical blocking, `GEMMKIT_*` env tuning knobs, and reusable packing
  workspaces (`Workspace`)

### gemmkit-ndarray, gemmkit-nalgebra, gemmkit-faer

#### Added

- Zero-copy adapters over each library's native matrix views (C-order, F-order,
  general and reversed strides), mirroring the full core surface including the
  `half`/`int8`/`complex` families and the `epilogue` fused entries; batched
  GEMM is exposed in the shape each library's types allow: the ndarray adapter's
  3-D strided `gemm_batched`/`dot_batched` (and the fused twin), and the
  nalgebra/faer `gemm_batched` over a slice of per-element `(A, B)` inputs paired
  with a slice of `&mut C` outputs (gemmkit's pointer-array batched engine, with
  heterogeneous per-element shapes; neither library has a rank-3 type)

### gemmkit-tune

#### Added

- Install-time autotuner binary: sweeps the runtime knobs on the target machine
  and emits a ready-to-source `GEMMKIT_*` env profile
