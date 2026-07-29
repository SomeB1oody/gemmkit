# Extension Points

gemmkit's variation points are traits, const generics, and typed function pointers stored in `OnceLock` slots. There are no macros and no `transmute`. That discipline exists for 1 reason. The library expects 4 kinds of growth: a new instruction set, a new element type, a new dot-product instruction, and a new fused transform. Each one should land as additive code, with a short, checkable list of touch points. Each one should also leave the driver, the packing routines, and the blocking model untouched.

This page turns those 4 recipes into walkthroughs. It is written for someone extending the crate itself. The seams stay public enough that the most important one, driving the generic driver with your own kernel family, also works from outside the crate. A test proves it.

## A new ISA backend

An ISA backend is a zero-sized *token*, plus a set of vocabulary implementations. The wasm `simd128` backend (`gemmkit/src/simd/wasm.rs`) is the most recent complete example. It is worth reading end to end, since it is only 1 file plus a handful of dispatch lines.

The token's only inherent behavior is `Simd::vectorize`, the `#[target_feature]` trampoline. Runtime CPU detection cannot pair with a fixed `#[target_feature]` attribute on the generic kernel. So every kernel invocation runs inside a tiny, annotated function. The `#[inline(always)]` primitives fold into that function, so every intrinsic lands in feature-enabled codegen:

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

1. **The token.** Add a `Copy + Send + Sync + 'static` zero-sized struct in a new `gemmkit/src/simd/` module. `cfg`-gate it to its architecture, and give it the `vectorize` trampoline shown above.
2. **`SimdOps<T>` implementations.** Add 1 for each element type the ISA accelerates. Each one needs a register type and `LANES`. It also needs the primitive vocabulary: load, store, splat, mul, add, `mul_add`, `fnma`, `reduce_sum`, plus `max`/`min` if the fused float epilogue should vectorize. The vocabulary stays deliberately thick, so the microkernel can stay 1 generic function. You implement primitives, never a kernel.

   Honor the documented contracts here. The simd128 implementation uses `f32x4_pmax`, not `f32x4_max`, because the trait's `max` requires a NaN in `a` to yield `b`. This is the `ReLU(NaN) = 0` agreement between the vector and scalar epilogues. It also passes the operands *reversed*, as `f32x4_pmax(b, a)`, because `pmax(x, y)` computes `x < y ? y : x`. The natural argument order would return NaN for a NaN `a`, and `-0.0` for `max(-0.0, +0.0)`, the opposite of the contract in both cases. It also spells `mul_add` as an unfused `mul` followed by `add`, because wasm has no hardware FMA. The relaxed-SIMD alternative is nondeterministic by specification, which would break reproducibility.
3. **Tile geometry.** Pick `(MR_REG, NR)` for each type, and encode it as the const generics of the per-ISA wrapper functions in the dispatch modules. This is the *only* per-`(type, ISA)` knob. Budget the registers explicitly. simd128 runs 2x4 for `f32`: 8 accumulators, 2 LHS registers, and 1 RHS register, for 11 live `v128` values. LLVM's wasm backend starts to spill past about 16 live vectors, which is why simd128 stays at that width. NEON runs 4x4 instead, leaving registers spare on purpose.
4. **A `Dispatched` descriptor, and 1 arm per `select_*` ladder.** The memoized selection ladders live in `gemmkit/src/dispatch/`. They are `select_f32`/`select_f64` for float, `select_f16`/`select_bf16` for mixed, `select_i8` for int, and `select_c32`/`select_c64` for complex, plus the map-epilogue selectors. Each ladder arm bundles the plain, prepacked, and fused entry points together with the tile geometry. So adding the ISA costs 1 descriptor constant and 1 match arm per type it accelerates.
5. **A `GEMMKIT_REQUIRE_ISA` name.** Add a `ForcedIsa` variant and its parse string in `gemmkit/src/dispatch/isa.rs`. The current values are `scalar`, `fma`, `avx512f`, `avx512vnni`, `avx512bf16`, `neon`, `simd128`, and `auto`. Follow the fail-loudly rule. If the pinned ISA is unsupported, dispatch must panic rather than fall back. This way, a CI job that means to exercise your kernel cannot silently pass on a different one instead.
6. **Tests ride along mostly for free.** `tests/simd_conformance.rs` constructs tokens directly and checks every primitive against scalar references. An `env_isa_*` pin binary, plus a CI job, makes the dispatch route itself testable too (see [Testing and Verification](Testing_and_Verification.md)).

You should not need to touch `driver.rs`, any kernel family, `pack.rs`, or `cache.rs`. The simd128 backend changed none of them.

## A new element type

Element types vary along 2 small traits (see [Scalars and Kernel Families](Scalars_and_Kernel_Families.md)). `Scalar` (`gemmkit/src/scalar.rs`) declares only the identity constants and the accumulator type `Acc`. Choosing `Acc` is the single most consequential decision here, since it fixes the rounding story. `f16` chose `Acc = f32`. `i8` chose `Acc = i32`, which makes integer GEMM exact. `KernelFamily` (`gemmkit/src/kernel.rs`) bundles everything else that distinguishes the operation: the `Lhs`/`Rhs`/`Acc`/`Out` types, the pack layout, and the microkernel.

Often a new family is not needed at all. If the type is a narrow input over an existing accumulator, implement the `KernelSimd<L, R, A, O>` widen/narrow seam on the capable tokens instead. That means widening loads, plus 1 narrowing store. Then reuse the generic microkernel, the way `MixedGemm<f16>` and `MixedGemm<bf16>` do. The homogeneous case is a blanket implementation, and the mixed implementations cannot overlap it. A genuinely new operation shape, such as the planar complex kernel or the requantizing integer families, gets its own `KernelFamily` instead.

Wiring the new type into the public API means adding a dispatch module under `gemmkit/src/dispatch/`, with its own `OnceLock` slot per type. Feature detection runs once, the winning monomorphized entry points get cached, and every later call becomes 1 indirect call. Copy `dispatch/mixed.rs` as a pattern for 2 types, gemv/small-mn/small-k reroutes, and a dot-kernel selection wrinkle. Copy `dispatch/int.rs` instead for a heterogeneous task type.

The open/closed property here is not folklore. `gemmkit/tests/open_closed.rs` enforces it directly. That test defines `NaiveFloat`, an independently written family with its own packing and a plain scalar microkernel, built using only public items. It then drives the *unchanged* public `driver::run` with `NaiveFloat`, and checks the result against an `f64` reference. If a driver change ever breaks the family seam, that test fails to compile. It also works as the template to start a new family from.

## A dot-product instruction

Instructions such as `vpdpbusd` and `vdpbf16ps` fold several depth steps into 1 operation. That *reshapes the accumulation rounding*, so they must never arrive as a clever override of the portable tile loop. The seam is split to keep that distinction clear:

- The family declares `DEPTH_MULTIPLE = Q` (greater than 1), and packs through `pack_kgroup_panels` (`gemmkit/src/pack.rs`). That function interleaves `Q` consecutive depth steps contiguously per lane. The driver rounds panel depths up to `Q`, and keeps k-groups from straddling slice boundaries.
- The capable token overrides `KernelSimd::dot_accumulate`, and consumes whole instruction groups from those panels. The packed layout is a private contract between the family's packers and the overriding token. Any signedness correction, such as VNNI's `+128` trick with its column-sum compensation, lives inside the override. So the accumulator holds the true sum when it returns.
- `SimdOps::accumulate_tile` overrides are reserved for *scheduling* changes that keep the rounding shape unchanged. 2 examples are an in-order core that needs explicit software pipelining, and a scalable-vector ISA whose length is not a compile-time constant. Its documentation is explicit that rounding-reshaping instructions are out of scope for this seam. Those should arrive as a new family with the dot seam instead. An `accumulate_tile` override must stay deterministic, and must round consistently with the edge path. The default implementation already saturates the FMA pipes on any wide out-of-order core, so prove an override earns its keep before adding one.

`IntGemmVnni` and `Bf16DotGemm` are the 2 worked examples. `IntGemmVnni` is bit-exact against the widen path, because integer arithmetic is associative. `Bf16DotGemm` is held to a tolerance instead, within the reproducibility contract. [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md) covers both in depth.

## A new fused transform

A fused transform is just 1 `Epilogue` implementation (`gemmkit/src/kernel/epilogue.rs`). The driver's `last_k` plumbing, the zero-cost `Identity` default, and the routing through every special path all come for free (see [Epilogue Fusion](Epilogue_Fusion.md)). The design work is choosing an application path, and honoring 1 hard rule. **The vector and scalar paths must agree bit-for-bit.** Full tiles take the vector path, edge and strided tiles take the scalar path, and a single output matrix can mix the two.

- A transform on `Acc`-typed values with a natural register form sets `VECTOR = true` and implements `apply_reg`. This is the `FusedEpi` pattern. Mind the NaN and signed-zero semantics here. `LeakyRelu`, for instance, is written as the identical `max + slope*min` composition in both forms.
- A transform that narrows `Acc` to a different `Out` sets `VECTOR_STORE = true` and implements `apply_store`. This is the `KRequantize` pattern. Argue the bit-equality case by case, and pin it with a conformance sweep.
- A transform with no profitable vector form keeps both flags `false`, and routes everything through scratch and the scalar `apply`. This stays correct for any tile shape. If the scalar value could differ by 1 ULP from the fast path's fused store, though, borrow `MapEpi`'s trick instead. Set `VECTOR = true`, and implement `apply_reg` as a drain-to-stack-and-apply-per-lane. This way the transform always sees exactly the bits plain `gemm` would have stored.

Whatever path a new transform takes, add its own gemm-then-map equivalence test next to the existing ones in `gemmkit/tests/epilogue/`. That suite is where the bitwise contract actually gets enforced.
