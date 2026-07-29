# SIMD Tokens and ISA Dispatch

gemmkit picks its instruction set at runtime. That decision collides with how Rust compiles SIMD intrinsics. AVX and AVX-512 intrinsics compile correctly only inside a context where the target feature is enabled. Normally that context comes from a `#[target_feature(enable = "...")]` attribute on the enclosing function. The program only knows which features are safe to enable once it runs on a concrete CPU. The microkernel is one generic function shared by every instruction set, so no single attribute can sit on it.

This page explains 2 things. First, how the L0 SIMD layer (`gemmkit/src/simd.rs` and `gemmkit/src/simd/`) resolves that tension with zero-sized ISA tokens and a trampoline function. Second, how the L7 dispatch layer (`gemmkit/src/dispatch.rs` and `gemmkit/src/dispatch/`) selects and caches the winning kernel.

## ISA tokens and the vectorize trampoline

An ISA token is a zero-sized type that stands for one instruction-set choice. On x86 the tokens are `Fma` (AVX2 + FMA) and `Avx512F`, plus the dot-kernel variants `Avx512Vnni` and `Avx512Bf16`. On aarch64 the token is `Neon`. On wasm32 the token is `Simd128`. `ScalarTok` exists on every platform as the portable floor.

Each token implements the `Simd` trait. Its only method is `vectorize`, which runs a closure with this token's target features enabled. The code below shows the entire mechanism, from `gemmkit/src/simd/fma.rs`.

```rust
/// AVX2 + FMA ISA token
#[derive(Copy, Clone, Default)]
pub struct Fma;

impl Simd for Fma {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: the caller of `vectorize` (the runtime dispatcher) guarantees
        // the CPU supports avx2+fma(+f16c); `inner` then establishes the codegen
        // context, and `f` inlines into it
        unsafe { inner(f) }
    }
}
```

The trick is the direction of inlining. `inner` is a tiny function with a `#[target_feature]` attribute. The closure `f` inlines into it. `f` holds the packing loops and the microkernel calls, and every one of those is built from `#[inline(always)]` primitives. Every intrinsic therefore lands in a codegen context where the feature is enabled. No attribute ever touches the generic kernel itself.

The `unsafe` contract has exactly one obligation. The caller must guarantee the CPU really supports the token's features. The runtime dispatcher establishes this once per process.

This is the same pattern pulp and faer use. It works the same way for the serial path and for rayon worker closures. The driver wraps each column strip of microkernel calls in `simd.vectorize(|| ...)`. This amortizes the trampoline overhead over many tiles.

`ScalarTok`'s `vectorize` is just `f()`. It has nothing to enable. This is what makes the scalar path run everywhere, including under Miri.

## SimdOps: the per-type vocabulary

L0 builds on 3 traits, not 2. `Simd` is the ISA token trait above. `SimdOps<T>` is the per-element-type vocabulary this section covers. `KernelSimd<L, R, A, O>` is a third trait. It widens loads and narrows stores when a family's input type, accumulator type, and output type are not all the same. This section covers it near the end.

The token itself knows nothing about element types. All the raw operations live on `SimdOps<T>`, implemented once per `(ISA, T)` pair. It names the register type `Reg`, the lane count `LANES`, and every primitive the microkernel needs. The token and the element type are decoupled, so `LANES` varies with the pair. `f32` gets 8 lanes under `Fma` and 16 lanes under `Avx512F`. `f64` gets half as many lanes as `f32` on the same token.

The vocabulary is deliberately thick. The basic operations are `zero`, `splat`, `loadu`, `storeu`, `mul`, `add`, and the fused `mul_add`. Its subtractive partner is `fnma`, which computes `c - a*b`. The complex kernel needs `fnma` for one of its accumulation terms. The vocabulary also has the horizontal `reduce_sum`, used by the gemv and dot-product epilogues.

On top of those sit a few more primitives. `max` and `min` exist only on the real-float tokens, for the fused ReLU and clip epilogues. The `LANE_FMA` flag and its `fma_bvec` method give NEON a lane-indexed FMA path. It loads a block of RHS columns as one vector, instead of issuing a separate splat per column. `accumulate_tile` is the GEMM inner loop itself. Its portable default schedule already compiles down to the canonical register-blocked kernel on any out-of-order core.

The complex split kernel has its own seam here too, `cplx_microkernel`. The dot kernels have theirs as well, `dot_accumulate`, which lives on `KernelSimd` rather than on `SimdOps`. `KernelSimd<L, R, A, O>` is the seam a family drives on when its input types, accumulator type, and output type are not all equal. One example is an `f16` input with an `f32` accumulator. It widens a narrow input load into the accumulator type and narrows an accumulator value back down on store. A homogeneous family has all 4 types equal. It gets a `KernelSimd` implementation for free through a blanket impl. It never needs any per-ISA code of its own. See [Scalars and Kernel Families](Scalars_and_Kernel_Families.md) and [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md) for more on both seams.

The thickness is the point. matrixmultiply's per-ISA trait is thin, so each instruction set has to reimplement the kernel from scratch. Here every primitive the kernel needs sits behind `SimdOps`, so the microkernel is one generic function over every ISA. Adding an instruction set costs a new token, its `SimdOps` impls, and one line in each dispatch ladder. The `simd` module depends only on `crate::scalar` and `core`. It has no reverse dependency on the kernel, driver, or cache layers, so the whole abstraction could move into its own crate unchanged.

## The dispatch layer

Dispatch turns the question "which token applies" into a one-time decision. Each dispatched element type owns one `OnceLock` slot holding a `Dispatched<T>` descriptor. `f32` and `f64` use `gemmkit/src/dispatch/float.rs`. `f16` and `bf16` use `dispatch/mixed.rs`. `i8` uses `dispatch/int.rs`, with its own `IntDispatched` and `IntRequantDispatched` shapes, because those types are heterogeneous. `c32` and `c64` use `dispatch/complex.rs`. The code below, lightly trimmed, is from `dispatch/float.rs`.

```rust
#[derive(Copy, Clone)]
pub(super) struct Dispatched<T> {
    pub(super) run: GemmFn<T>,
    pub(super) run_packed: PackedFn<T>,
    #[cfg(feature = "epilogue")]
    pub(super) run_fused: FusedFn<T>,
    #[cfg(feature = "epilogue")]
    pub(super) run_packed_fused: PackedFusedFn<T>,
    pub(super) mr: usize,
    pub(super) nr: usize,
    pub(super) depth_multiple: usize,
}
```

The slot caches the winning monomorphized entry points: the plain kernel, the prepacked-RHS kernel, and, under the `epilogue` feature, their fused twins. It also caches the microtile geometry `(mr, nr)` and the family's `depth_multiple`. The geometry is cached so `prepack_rhs` can size a buffer through the same ISA choice the consuming call will make. `depth_multiple` lets the bf16 prepack path round its packed depth to match the dot kernel's layout. Everything here is a typed function pointer. There is no `transmute` and no `AtomicPtr<()>`.

A call flows through a fixed chain. `gemm` calls `dispatch::execute`, which handles the degenerate cases. `dispatch::execute` calls `T::dispatch`, which reads the memoized slot. The slot resolves to one indirect call into a wrapper, such as `gemm_f32_avx512f`. That wrapper instantiates the shared generic entry as `run_typed::<f32, Avx512F, 2, 12>`.

Selection runs once, inside the `OnceLock` initializer. It first honors any `GEMMKIT_REQUIRE_ISA` pin, covered below. After that, the auto ladder on x86 probes `avx512f` first, then `avx2` plus `fma`, and falls back to scalar last. On aarch64, NEON is the baseline. The architecture makes NEON mandatory, so no probe is needed there.

On wasm32 there is no runtime feature detection at all. `simd128` is chosen at compile time, through `cfg(target_feature = "simd128")`. The build must pass `-C target-feature=+simd128`, or it gets the scalar kernel instead. Scalar is the floor on every architecture.

Each per-type ladder adds its own gate on the same skeleton. The `f16` FMA arm also requires `f16c`, for the `vcvtph2ps` and `vcvtps2ph` conversions. The `bf16` ladder tries the `avx512bf16` dot kernel before plain AVX-512F. The `i8` ladder tries `avx512vnni` (with `avx512bw`) before the widen kernel.

2 build-mode details are worth knowing. Under `std`, feature detection uses `is_x86_feature_detected!`, and the result is memoized in the `OnceLock`. Without `std`, there is no runtime CPU detection, because `raw-cpuid` is gated on `std`. The probe macro degrades to `cfg!(target_feature = ...)`, `GEMMKIT_REQUIRE_ISA` parsing degrades to `Auto`, and the select function runs on every call. Every branch inside it is now a compile-time constant, so it folds down to a direct choice. A `no_std` build simply runs whatever its compile-time target features guarantee. See [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md).

## Tile geometry as const generics

Besides the instruction encoding, the one thing that genuinely varies per `(type, ISA)` is the microtile shape. It is expressed as a pair of const generics, `(MR_REG, NR)`, chosen at the dispatch site. It is never a new type, trait, or macro. `MR_REG` is how many registers tall the tile is, so the row count is `MR = MR_REG * LANES`. The table below covers `f32`.

| ISA | `(MR_REG, NR)` | `LANES` | Tile `MR x NR` | Register budget |
|---|---|---|---|---|
| AVX-512F | `(2, 12)` | 16 | 32 x 12 | 24 acc + 2 lhs + 1 rhs = 27 ZMM |
| FMA (AVX2) | `(2, 6)` | 8 | 16 x 6 | 12 acc + 2 lhs + 1 rhs = 15 YMM |
| NEON | `(4, 4)` | 4 | 16 x 4 | 16 acc + 4 lhs + 1 rhs = 21 of 32 vregs |
| simd128 | `(2, 4)` | 4 | 8 x 4 | 8 acc + 2 lhs + 1 rhs = 11 live `v128` |
| scalar | `(4, 4)` | 1 | 4 x 4 | plain locals |

`f64` halves the lane count. The same `(MR_REG, NR)` pairs then yield 16x12 on AVX-512F, 8x6 on FMA, 8x4 on NEON, and 4x4 on simd128. The budgets are not accidents. NEON deliberately leaves about 11 registers free, as rename headroom for a wide out-of-order core to overlap loads with FMAs. simd128 stays at 11 live vectors, because LLVM's wasm backend starts spilling past roughly 16. These comments live next to the wrappers in `dispatch/float.rs`, so the table above is the code, not an aspiration.

## Pinning with GEMMKIT_REQUIRE_ISA

By default the best available ISA wins. Setting the environment variable `GEMMKIT_REQUIRE_ISA` forces exactly one kernel instead of the automatic choice. It accepts these values, case-insensitive:

- `scalar`
- `fma` (alias `avx2`)
- `avx512f`
- `avx512vnni` (alias `vnni`)
- `avx512bf16` (alias `bf16`)
- `neon`
- `simd128` (alias `wasm`)
- `auto`

Unset or empty means `auto`. An unrecognized value causes a hard panic, so a typo in a CI configuration cannot silently select the wrong thing. `avx512vnni` pins the `i8` `vpdpbusd` dot kernel. `avx512bf16` pins the `bf16` `vdpbf16ps` dot kernel. For every other element type, both resolve to the plain AVX-512F path.

The defining behavior is that a pin never falls back. Selection panics if the CPU, or an emulator such as Intel SDE, does not report the required feature. It also panics if the requested ISA does not exist on the target architecture at all, such as `neon` off aarch64 or `fma`/`avx512*` off x86. The panic message names the missing feature. The rationale is CI honesty. A job that means to exercise a given kernel must fail loudly, rather than silently test a different one.

The `simd128` pin earns its keep the same way on wasm. The target feature is an easily forgotten compile-time flag. Pinning turns a dropped flag from a silent scalar fallback into a build that refuses to run.

The value is read once. Selection is memoized in the per-type `OnceLock`, so the variable must be set in the process environment before the first GEMM call. Changing it afterward has no effect for the life of the process. See [Runtime ISA Dispatch](../gemmkit-guide/Runtime_ISA_Dispatch.md) for the user-facing side of pinning, including CI recipes and how it interacts with the tuning knobs.
