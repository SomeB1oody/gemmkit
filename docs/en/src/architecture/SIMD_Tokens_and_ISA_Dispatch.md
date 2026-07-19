# SIMD Tokens and ISA Dispatch

gemmkit picks its instruction set at runtime, and that decision collides head-on with how Rust compiles SIMD intrinsics. AVX and AVX-512 intrinsics must be code-generated inside a context where the corresponding target feature is enabled — normally a `#[target_feature(enable = "...")]` attribute on the enclosing function. But which features are safe to enable is only known once the program is running on a concrete CPU, and the microkernel is one generic function shared by every instruction set: there is no single attribute you could pin on it. This page explains how the L0 SIMD layer (`gemmkit/src/simd.rs` and `gemmkit/src/simd/`) resolves that tension with zero-sized ISA tokens and a trampoline, and how the L7 dispatch layer (`gemmkit/src/dispatch.rs` and `gemmkit/src/dispatch/`) selects and caches the winning kernel.

## ISA tokens and the vectorize trampoline

An ISA token is a zero-sized type that stands for one instruction-set choice: `Fma` (AVX2 + FMA) and `Avx512` on x86, plus the dot-kernel variants `Avx512Vnni` and `Avx512Bf16`; `Neon` on aarch64; `Simd128` on wasm32; and `ScalarTok` everywhere as the portable floor. Each token implements the `Simd` trait, whose only method is `vectorize`: run a closure with this token's target features enabled. Here is the entire mechanism, from `gemmkit/src/simd/fma.rs`:

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

The trick is inlining direction. `inner` is a tiny `#[target_feature]`-annotated function, and the closure `f` — the packing loops and microkernel calls, all built from `#[inline(always)]` primitives — inlines *into* it. Every intrinsic therefore lands in a codegen context where the feature is enabled, even though no attribute ever touches the generic kernel itself. The `unsafe` contract is exactly one obligation: the caller must guarantee the CPU really supports the token's features, which the runtime dispatcher establishes once per process. This is the proven pulp/faer pattern, and it works identically for the serial path and for rayon worker closures; the driver wraps each column strip of microkernel calls in `simd.vectorize(|| ...)`, so the trampoline overhead is amortized over many tiles. `ScalarTok`'s `vectorize` is just `f()` — nothing to enable — which is what makes the scalar path runnable everywhere, including under Miri.

## SimdOps: the per-type vocabulary

The token deliberately knows nothing about element types. All actual operations live on a second trait, `SimdOps<T>`, implemented per `(ISA, T)` pair: it names the register type `Reg`, the lane count `LANES`, and every primitive the microkernel needs. Because the token and the element type are decoupled, `LANES` varies with the pair — `f32` is 8 lanes under `Fma` and 16 under `Avx512`, and `f64` halves both.

The vocabulary is deliberately thick. The basics are `zero`, `splat`, `loadu`, `storeu`, `mul`, `add`, the fused `mul_add`, its subtractive partner `fnma` (`c - a*b`, needed by the complex kernel), and the horizontal `reduce_sum` used by gemv and dot epilogues. On top of those sit `max`/`min` (overridden only by the real-float tokens, for the fused ReLU/clip epilogues), the `LANE_FMA` flag with `fma_bvec` (NEON's lane-indexed FMA path, which loads a block of RHS columns as one vector instead of issuing a splat per column), and `accumulate_tile` — the GEMM inner loop itself, with a portable default schedule that LLVM already lowers to the canonical register-blocked kernel on any out-of-order core. The complex split kernel gets its own seam here too (`cplx_microkernel`), as do the dot kernels (`dot_accumulate` on the `KernelSimd` companion trait) — those are covered in [Scalars and Kernel Families](Scalars_and_Kernel_Families.md) and [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md).

The thickness is the point. matrixmultiply's thin per-ISA trait forces each instruction set to reimplement the kernel; here *every* primitive the kernel needs is behind `SimdOps`, so the microkernel is one generic function over all ISAs, and adding an instruction set costs a new token, its `SimdOps` impls, and one line in each dispatch ladder. The `simd` module depends only on `crate::scalar` and `core` — no reverse dependency on the kernel, driver, or cache layers — so the whole abstraction could be split into its own crate unchanged.

## The dispatch layer

Dispatch turns "which token do we use?" into a one-time decision. Each dispatched element type owns one `OnceLock` slot holding a `Dispatched<T>` descriptor: `f32`/`f64` in `gemmkit/src/dispatch/float.rs`, `f16`/`bf16` in `dispatch/mixed.rs`, `i8` in `dispatch/int.rs` (its own `IntDispatched`/`IntRequantDispatched` shapes, since the types are heterogeneous), and `c32`/`c64` in `dispatch/complex.rs`. From `dispatch/float.rs`, lightly trimmed:

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

The slot caches the winning monomorphized entry points — the plain kernel, the prepacked-RHS kernel, and (under the `epilogue` feature) their fused twins — plus the microtile geometry `(mr, nr)` and the family's `depth_multiple`. The geometry is cached so `prepack_rhs` can size a buffer through the *same* ISA choice the consuming call will make, and `depth_multiple` lets the bf16 prepack path round its packed depth to match the dot kernel's layout. Everything is a typed function pointer: no `transmute`, no `AtomicPtr<()>`. A call flows as `gemm` → `dispatch::execute` (degenerate cases handled here) → `T::dispatch` → the memoized slot → one indirect call into a wrapper like `gemm_f32_avx512`, which instantiates the shared generic entry as `run_typed::<f32, Avx512, 2, 12>`.

Selection runs once, inside the `OnceLock` initializer. After honoring any `GEMMKIT_REQUIRE_ISA` pin (below), the auto ladder on x86 probes `avx512f` first, then `avx2` + `fma`, then falls to scalar. On aarch64 NEON is baseline — mandatory in the architecture, so no probe is needed. On wasm32 there is no runtime feature detection at all: `simd128` is chosen at compile time via `cfg(target_feature = "simd128")`, so the build must pass `-C target-feature=+simd128` or it gets the scalar kernel. Scalar is the floor on every architecture. Per-type ladders add their own gates on the same skeleton: the `f16` FMA arm additionally requires `f16c` (for the `vcvtph2ps`/`vcvtps2ph` conversions), the `bf16` ladder tries the `avx512bf16` dot kernel before plain AVX-512, and the `i8` ladder tries `avx512vnni` (with `avx512bw`) before the widen kernel.

Two build-mode wrinkles are worth knowing. Under `std`, feature detection is `is_x86_feature_detected!` and the result is memoized in the `OnceLock`. Without `std` there is no runtime CPU detection (`raw-cpuid` is `std`-gated): the probe macro degrades to `cfg!(target_feature = ...)`, `GEMMKIT_REQUIRE_ISA` parsing degrades to `Auto`, and the select function runs on each call — but every branch in it is now a compile-time constant, so it folds to a direct choice. A `no_std` build simply runs whatever its compile-time target features guarantee; see [no_std and WebAssembly](../gemmkit-guide/no_std_and_WebAssembly.md).

## Tile geometry as const generics

The one thing that genuinely varies per `(type, ISA)` besides the instruction encoding is the microtile shape, and it is expressed as a pair of const generics `(MR_REG, NR)` chosen at the dispatch site — never a new type, trait, or macro. `MR_REG` is how many registers tall the tile is, so the row count is `MR = MR_REG * LANES`. For `f32`:

| ISA | `(MR_REG, NR)` | `LANES` | Tile `MR x NR` | Register budget |
|---|---|---|---|---|
| AVX-512 | `(2, 12)` | 16 | 32 x 12 | 24 acc + 2 lhs + 1 rhs = 27 ZMM |
| FMA (AVX2) | `(2, 6)` | 8 | 16 x 6 | 12 acc + 2 lhs + 1 rhs = 15 YMM |
| NEON | `(4, 4)` | 4 | 16 x 4 | 16 acc + 4 lhs + 1 rhs = 21 of 32 vregs |
| simd128 | `(2, 4)` | 4 | 8 x 4 | 8 acc + 2 lhs + 1 rhs = 11 live `v128` |
| scalar | `(4, 4)` | 1 | 4 x 4 | plain locals |

`f64` halves the lane count, so the same `(MR_REG, NR)` pairs yield 16x12 (AVX-512), 8x6 (FMA), 8x4 (NEON), and 4x4 (simd128). The budgets are not accidents: NEON deliberately leaves ~11 spare registers as rename headroom for a wide out-of-order core to overlap loads with FMAs, and simd128 stays at 11 live vectors because LLVM's wasm backend starts spilling past roughly 16. These comments live next to the wrappers in `dispatch/float.rs`, so the table above is the code, not an aspiration.

## Pinning with GEMMKIT_REQUIRE_ISA

By default the best available ISA wins. Setting the environment variable `GEMMKIT_REQUIRE_ISA` forces exactly one kernel instead. Accepted values (case-insensitive) are `scalar`, `fma` (alias `avx2`), `avx512` (alias `avx512f`), `avx512vnni` (alias `vnni`), `avx512bf16` (alias `bf16`), `neon`, `simd128` (alias `wasm`), and `auto`; unset or empty means `auto`, and an unrecognized value is a hard panic so a typo in CI configuration cannot silently select the wrong thing. `avx512vnni` pins the `i8` `vpdpbusd` dot kernel and `avx512bf16` the `bf16` `vdpbf16ps` dot kernel; for every other element type both resolve to the plain AVX-512 path.

The defining behavior is that a pin never falls back. If the CPU (or an emulator such as Intel SDE) does not report the required feature, or the requested ISA does not exist on the target architecture — `neon` off aarch64, `fma`/`avx512*` off x86 — dispatch panics with a message naming the missing feature. The rationale is CI honesty: a job that means to exercise a given kernel must fail loudly rather than silently test a different one. The `simd128` pin earns its keep the same way on wasm: the target feature is an easily forgotten compile-time flag, and pinning turns "the flag was dropped" from a silent scalar-fallback into a build that refuses to run.

The value is read once. Selection is memoized in the per-type `OnceLock`, so the variable must be set in the process environment before the first GEMM call; changing it afterwards has no effect for the life of the process. The user-facing side of pinning — CI recipes and interaction with the tuning knobs — is covered in [Runtime ISA Dispatch](../gemmkit-guide/Runtime_ISA_Dispatch.md).
