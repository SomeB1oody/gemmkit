# Extension Points

gemmkit's variation points are traits, const generics, and typed function pointers in `OnceLock` slots — no macros, no `transmute`. That discipline exists for exactly one reason: so the four kinds of growth the library expects (a new instruction set, a new element type, a new dot-product instruction, a new fused transform) each land as additive code with a short, checkable list of touch points, leaving the driver, the packing routines, and the blocking model untouched. This page turns the four recipes into walkthroughs. They are written for someone extending the crate itself; the seams are public enough that the key one — driving the generic driver with your own kernel family — works from outside the crate too, and a test proves it.

## A new ISA backend

An ISA backend is a zero-sized *token* plus vocabulary implementations. The wasm `simd128` backend (`gemmkit/src/simd/wasm.rs`) is the most recent complete example and worth reading end to end — it is one file plus a handful of dispatch lines.

The token's only inherent behavior is `Simd::vectorize`, the `#[target_feature]` trampoline: runtime CPU detection cannot pair with a fixed `#[target_feature]` on the generic kernel, so every kernel invocation runs inside a tiny annotated function into which the `#[inline(always)]` primitives fold, landing all intrinsics in feature-enabled codegen:

```rust
// gemmkit/src/simd/wasm.rs
impl Simd for Simd128 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "simd128")]
        fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        inner(f)
    }
}
```

The checklist:

1. **The token**: a `Copy + Send + Sync + 'static` zero-sized struct in a new `gemmkit/src/simd/` module, `cfg`-gated to its architecture, with the `vectorize` trampoline above.
2. **`SimdOps<T>` impls** for each element type the ISA accelerates: register type, `LANES`, and the primitive vocabulary (load/store/splat/mul/add/`mul_add`/`fnma`/`reduce_sum`, plus `max`/`min` if the fused float epilogue should vectorize). The vocabulary is deliberately thick so the microkernel stays one generic function; you implement primitives, never a kernel. Honor the documented contracts — the simd128 impl uses `f32x4_pmax`, not `f32x4_max`, because the trait's `max` requires NaN-in-`a` to yield `b` (the `ReLU(NaN) = 0` agreement between vector and scalar epilogues), and it spells `mul_add` as unfused `mul` + `add` because wasm has no hardware FMA and the relaxed-SIMD alternative is spec-nondeterministic, which would break reproducibility.
3. **Tile geometry**: pick `(MR_REG, NR)` for each type and encode it as the const generics of the per-ISA wrapper functions in the dispatch modules. This is the *only* per-`(type, ISA)` knob. Budget registers explicitly: simd128 runs 2x4 for `f32` (8 accumulators + 2 LHS + 1 RHS = 11 live `v128`) because LLVM's wasm backend spills past ~16 live vectors; NEON runs 4x4 with ~11 registers spare on purpose.
4. **A `Dispatched` descriptor and one arm per `select_*` ladder**: the memoized selection ladders live in `gemmkit/src/dispatch/` — `select_f32`/`select_f64` (float), `select_f16`/`select_bf16` (mixed), `select_i8` (int), `select_c32`/`select_c64` (complex), plus the map-epilogue selectors. Each ladder arm bundles the plain, prepacked, and fused entry points with the tile geometry, so adding the ISA is one descriptor constant and one match arm per type it accelerates.
5. **A `GEMMKIT_REQUIRE_ISA` name**: a `ForcedIsa` variant and its parse string in `gemmkit/src/dispatch/isa.rs` (current values: `scalar`, `fma`, `avx512`, `avx512vnni`, `avx512bf16`, `neon`, `simd128`, `auto`), with the fail-loudly rule — if the pinned ISA is unsupported, dispatch panics rather than falling back, so a CI job that means to exercise your kernel cannot silently pass on another one.
6. **Tests ride along mostly for free**: `tests/simd_conformance.rs` constructs tokens directly and checks every primitive against scalar references, and an `env_isa_*` pin binary plus a CI job make the dispatch route itself testable (see [Testing and Verification](Testing_and_Verification.md)).

What you do not touch: `driver.rs`, the kernel families, `pack.rs`, `cache.rs`. The simd128 backend changed none of them.

## A new element type

Element types vary along two small traits (see [Scalars and Kernel Families](Scalars_and_Kernel_Families.md)). `Scalar` (`gemmkit/src/scalar.rs`) declares only the identity constants and the accumulator type `Acc` — the single most consequential decision, since it fixes the rounding story (`f16` chose `Acc = f32`; `i8` chose `Acc = i32`, making integer GEMM exact). `KernelFamily` (`gemmkit/src/kernel.rs`) bundles everything that distinguishes the operation: the `Lhs`/`Rhs`/`Acc`/`Out` types, the pack layout, and the microkernel.

Often you do not need a new family at all. If the type is a narrow input over an existing accumulator, implement the `KernelSimd<L, R, A, O>` widen/narrow seam on the capable tokens — widening loads, one narrowing store — and reuse the generic microkernel the way `MixedGemm<f16>`/`MixedGemm<bf16>` do; the homogeneous case is a blanket impl, and the mixed impls cannot overlap it. A genuinely new operation shape (the planar complex kernel, the requantizing integer families) gets its own `KernelFamily`.

Wiring it into the public API is a dispatch module under `gemmkit/src/dispatch/` with its own `OnceLock` slot per type: feature detection runs once, the winning monomorphized entry points are cached, and every later call is one indirect call. The pattern to copy is `dispatch/mixed.rs` (two types, gemv/small-mn/small-k reroutes, a dot-kernel selection wrinkle) or `dispatch/int.rs` (heterogeneous task type).

The open/closed property is not folklore; it is enforced by `gemmkit/tests/open_closed.rs`, which defines `NaiveFloat` — an independently written family with its own packing and a plain scalar microkernel, using only public items — and drives the *unchanged* public `driver::run` with it, checking the result against an `f64` reference. If a driver change breaks the family seam, that test fails to compile. It is also the template to start a new family from.

## A dot-product instruction

Instructions like `vpdpbusd` and `vdpbf16ps` fold several depth steps into one operation, which *reshapes the accumulation rounding* — so they must not arrive as a clever override of the portable tile loop. The seam is split accordingly:

- The family declares `DEPTH_MULTIPLE = Q` (> 1) and packs through `pack_kgroup_panels` (`gemmkit/src/pack.rs`), which interleaves `Q` consecutive depth steps contiguously per lane; the driver rounds panel depths up to `Q` and keeps k-groups from straddling slice boundaries.
- The capable token overrides `KernelSimd::dot_accumulate`, consuming whole instruction groups from those panels; the packed layout is a private contract between the family's packers and the overriding token. Any signedness correction (VNNI's `+128` trick with its column-sum compensation) lives inside the override, so the accumulator holds the true sum on return.
- `SimdOps::accumulate_tile` overrides are reserved for *scheduling* changes that keep the rounding shape — an in-order core that needs explicit software pipelining, a scalable-vector ISA whose length is not a compile-time constant. Its documentation is explicit: rounding-reshaping instructions are out of scope for that seam and arrive as a new family with the dot seam instead. An `accumulate_tile` override must stay deterministic and round consistently with the edge path; the default already saturates the FMA pipes on any wide out-of-order core, so prove an override pays before keeping it.

`IntGemmVnni` (bit-exact against the widen path, because integer arithmetic is associative) and `Bf16DotGemm` (held to a tolerance, within the reproducibility contract) are the two worked examples; [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md) covers them in depth.

## A new fused transform

A fused transform is one `Epilogue` impl (`gemmkit/src/kernel/epilogue.rs`); the driver's `last_k` plumbing, the zero-cost `Identity` default, and the routing through every special path come for free (see [Epilogue Fusion](Epilogue_Fusion.md)). The design work is choosing the application path and honoring one hard rule: **the vector and scalar paths must agree bit-for-bit**, because full tiles take the vector path, edge and strided tiles take the scalar path, and one output matrix mixes them.

- A transform on `Acc`-typed values with a natural register form sets `VECTOR = true` and implements `apply_reg` (the `FusedEpi` pattern; mind NaN and signed-zero semantics — `LeakyRelu` is written as the identical `max + slope*min` composition in both forms).
- A transform that narrows `Acc` to a different `Out` sets `VECTOR_STORE = true` and implements `apply_store` (the `KRequantize` pattern), with the bit-equality argued case by case and pinned by a conformance sweep.
- A transform with no profitable vector form keeps both flags `false` and routes everything through scratch and the scalar `apply` — correct for any tile shape. But if the scalar value could differ by an ULP from the fast path's fused store, borrow `MapEpi`'s trick: set `VECTOR = true` and implement `apply_reg` as a drain-to-stack-and-apply-per-lane, so the transform always sees exactly the bits plain `gemm` would have stored.

Whatever the path, add the transform's own gemm-then-map equivalence test next to the existing ones in `gemmkit/tests/epilogue/`; that suite is where the bitwise contract is actually held to.
