# Scalars and Kernel Families

gemmkit multiplies several element types: `f32`, `f64`, `f16`, `bf16`, `i8`, `Complex<f32>`, and `Complex<f64>`. `u8` also appears, but only as a requantize output. Every one of these types flows through the same driver, the same packing framework, the same cache model, and the same parallel scheduler.

2 traits carry all the variation. `Scalar`, at L0 (`gemmkit/src/scalar.rs`), answers what a type is and what it accumulates in. `KernelFamily`, at L4 (`gemmkit/src/kernel.rs`), answers what makes this kind of GEMM different from the others. The driver is generic over the family. It never branches on element type. This page walks through that split, and why it falls where it does.

## Scalar: constants and an accumulator, nothing else

`Scalar` is deliberately tiny. The whole trait, from `gemmkit/src/scalar.rs`:

```rust
pub trait Scalar: Copy + Send + Sync + PartialEq + 'static {
    /// The type in which products are accumulated. `Self` for `f32`/`f64`
    type Acc: Scalar<Acc = Self::Acc>;
    /// The additive identity
    const ZERO: Self;
    /// The multiplicative identity
    const ONE: Self;
}
```

`Scalar` has no `Add`, no `Mul`, and no conversions. It has only the identity constants and the associated accumulator type. This omission is deliberate. All vectorized arithmetic lives in `SimdOps` (see [SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md)). The scalar arithmetic that an epilogue needs lives instead on narrow side traits, and only the types that need them implement those traits. `Float` covers `f32` and `f64`, and, through `num-complex`'s operators, the complex types too. `NarrowFloat` covers the `f16`/`bf16` widen and narrow conversions. `ComplexFloat` covers the real and imaginary accessors of the split complex kernel.

If `Scalar` itself carried arithmetic, every new element type would owe a full set of operations it may not actually have. `i8` is the clearest case. It needs no arithmetic trait at all, because its kernel does everything through the SIMD seam and exact `i32` integer operations.

`Acc` is the mixed-precision seam. The table is short.

| Element type | Accumulates in |
|---|---|
| `f32`, `f64` | itself |
| `f16`, `bf16` | `f32` |
| `i8` (and the output-only `u8`) | `i32` |
| `Complex<f32>`, `Complex<f64>` | itself |

The recursive bound `Acc: Scalar<Acc = Self::Acc>` pins the chain after one step: `f16 -> f32 -> f32 -> ...`. Generic code can then name the accumulator's accumulator without caring how narrow the input was. For the homogeneous types, the `Acc = Self` branch collapses at compile time and costs nothing.

## KernelFamily: everything that distinguishes one GEMM from another

`KernelFamily` bundles the rest. It carries the 4 element types (`Lhs`, `Rhs`, `Acc`, `Out`), the pack layout (`pack_lhs`/`pack_rhs`, which write micropanel-major panels), and the microkernel.

3 associated constants shape how the driver treats a family. `OUT_IS_ACC` says whether a running partial sum may round-trip through `C` between depth panels. This is the pivotal constant, covered below. `FORCE_PACK_LHS` and `FORCE_PACK_RHS` are set when packing performs a transform the kernel depends on, such as complex conjugation or dot-kernel interleaving. In that case, the driver must never read the operand in place. `DEPTH_MULTIPLE` is the instruction-group depth of a dot kernel. It is `1` for every other family.

A family overrides exactly one of 2 microkernel methods. A non-fusing family overrides the plain `microkernel` and inherits the default `microkernel_epi`. That default forwards to `microkernel`, after it asserts `E::IS_IDENTITY`, a fail-closed guard. A real epilogue reaching a family that cannot fuse panics, instead of being silently dropped. A fusing family, such as the float, mixed, and requantizing families, overrides `microkernel_epi` instead. It threads the epilogue through its own store, and its plain `microkernel` method is then dead code, keeping the default `unreachable!` body.

Tile geometry is deliberately not on the trait. `(MR_REG, NR)` is a pair of const generics, chosen per `(family, ISA)` at the dispatch site. A new tile is therefore a new instantiation of that pair, not a new type.

The payoff shows up in the driver's signature. `driver::run::<Fam, S, MR_REG, NR>` is generic over the family and the ISA token. It calls `Fam::pack_lhs`, `Fam::pack_rhs`, and `Fam::microkernel_epi`, and it contains not one `if` on an element type. Adding a kind of GEMM means writing a new family. It never means touching the driver.

## The family roster

10 family types ship today. They fall into generations: homogeneous, widen, dot, requantize, and complex. Reading them in that order makes the seams visible.

| Family | Types (`Lhs`/`Rhs` -> `Acc` -> `Out`) | `OUT_IS_ACC` | `DEPTH_MULTIPLE` | Notes |
|---|---|---|---|---|
| `FloatGemm<T>` | `T -> T -> T` for `f32`/`f64` | `true` | 1 | The baseline: one generic microkernel for every ISA |
| `MixedGemm<N>` | `N -> f32 -> N` for `f16`/`bf16` | `false` | 1 | Widen-FMA through the `KernelSimd` seam |
| `Bf16DotGemm` | `bf16 -> f32 -> bf16` | `false` | 2 | `vdpbf16ps` dot kernel. Both operands force-packed, k-pair-interleaved |
| `MixedGemmF32<N>` / `Bf16DotGemmF32` | `N -> f32 -> f32` | `true` | 1 / 2 | The f32-output deep-k twins: same accumulation, `f32` store |
| `IntGemm` | `i8 -> i32 -> i32` | `true` | 1 | Exact, wrapping. Sign-extend on load |
| `IntGemmVnni` | `i8 -> i32 -> i32` | `true` | 4 | `vpdpbusd` dot kernel, `+128` signedness correction, bit-identical to `IntGemm` |
| `IntGemmQ<O>` / `IntGemmVnniQ<O>` | `i8 -> i32 -> i8` or `u8` | `false` | 1 / 4 | Requantizing variants (feature `epilogue`) |
| `ComplexGemm<T, CONJ_A, CONJ_B>` | `T -> T -> T` for `c32`/`c64` | `true` | 1 | Split (SoA) kernel. Both operands force-packed planar. Conj is a pack-time sign flip |

`FloatGemm` is the reference point. It is homogeneous, with one generic `microkernel_impl` shared by every ISA and every tile.

The mixed and integer families introduce `Acc != Lhs`. They lean entirely on the widen/narrow seam covered below. The dot families, `Bf16DotGemm` and `IntGemmVnni`, go further. Each swaps in an interleaved pack layout and a hardware dot instruction. The f32-output twins exist so a deep contraction can re-block. [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md) covers all of this.

The requantizing variants bolt an exact `i32 -> i8`/`u8` requantize onto the integer accumulation. They are part of the fusion story in [Epilogue Fusion](Epilogue_Fusion.md).

`ComplexGemm` keeps `Acc = T`, so complex `alpha`/`beta` thread through the driver unchanged. Its hot loop instead runs on the real component, through a dedicated seam. [The Complex Split Kernel](The_Complex_Split_Kernel.md) covers that seam.

This page stays at the roster level. The deep dives live in those other pages.

## KernelSimd: the widen/narrow seam

The driver's bound on the ISA token is `S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>` (`gemmkit/src/simd.rs`). `KernelSimd<L, R, A, O>` extends `SimdOps<A>`, so it accumulates in `A`. It adds 4 moves a family needs at the type boundary.

`load_lhs` loads `LANES` LHS values and widens them to an `A` register. `splat_rhs` widens one RHS scalar and broadcasts it. `load_out` widens output values for the `beta != 0` read of `C`. `store_out` narrows an `A` register to `LANES` output values, and rounds to nearest-even when it actually narrows.

The homogeneous case costs nothing. A blanket impl, `KernelSimd<A, A, A, A> for S: SimdOps<A>`, forwards all 4 methods to plain `loadu`, `splat`, and `storeu`. So `FloatGemm<f32>` and its relatives need zero per-ISA code.

A mixed family instead adds per-ISA impls. Its loads genuinely widen, such as `f16 -> f32` through `vcvtph2ps`, or `i8 -> i32` through sign extension. Its `store_out` genuinely narrows. Coherence comes free here. The all-equal blanket and a mixed impl with `L != A` can never describe the same types.

2 further impl groups are derived, rather than hand-written per ISA. The requant blankets cover `Out = i8` or `u8`. They forward the accumulate side to the `<i8, i8, i32, i32>` impl. The f32-output twins cover `<N, N, f32, f32>` for `N = f16` or `bf16`. Those are written as 2 concrete heads, rather than one generic blanket over `N`. A generic blanket could not rule out colliding with the homogeneous blanket at `N = f32`.

`KernelSimd` also hosts 2 more seams. `dot_accumulate` is the dot seam. Only dot-capable tokens override it, and its default is `unreachable!`. `requant_store` is the vectorized requantize store, following the same pattern.

The constant that ties this seam to the driver's blocking is `OUT_IS_ACC`. The driver normally accumulates across `k` by splitting it into `kc` panels. The partial sum round-trips through `C`, with `beta = 1` after the first panel. That round-trip is exact only when `Out == Acc`.

When the output is narrower than the accumulator, that round-trip would round to 16 bits at every panel boundary. So a narrow family declares `OUT_IS_ACC = false`. The driver then responds with `kc = k`, one depth panel that spans the entire contraction. The whole contraction accumulates in `f32` registers, and the result rounds to the narrow output exactly once, at the end.

That single-rounding guarantee is what makes the mixed-precision results defensible. It has one cost. A single panel means its RHS micropanel can outgrow the L2 cache when `k` is large. The f32-output twins exist to pay that cost down. [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md) covers how they do it.

## The open/closed proof

The claim that the family seam is open for extension is not just prose. `gemmkit/tests/open_closed.rs` enforces it. This is an integration test that lives outside the crate, so it sees only the public API.

The test declares `NaiveFloat`, a deliberately naive second float family that shares nothing with `FloatGemm`. It reimplements micropanel packing from scratch, because the crate's internal `pack` helper is not visible to it. This is exactly the situation a third party would be in. `NaiveFloat` also supplies a plain scalar triple-loop `microkernel`.

The test then drives the unchanged generic driver, `driver::run::<NaiveFloat, ScalarTok, 4, 4>`, on a 40x33x28 problem. It checks the result against an `f64` reference.

The test's main value is that it compiles at all. A second family drove the driver with no edit to `driver.rs` or `pack.rs`. It used nothing but public items: `gemmkit::kernel::KernelFamily`, `gemmkit::simd::ScalarTok`, `gemmkit::driver::run`, `Workspace`, and `Parallelism`.

Any refactor that closes the seam breaks this file first. Examples include a driver branch on a concrete family, a newly required private helper, or a leaked internal type in the trait's signature. Such a refactor breaks this test before it breaks a downstream user.

[Testing and Verification](Testing_and_Verification.md) covers the wider testing story, including how the real families are cross-checked against oracles. [Extension Points](Extension_Points.md) covers what third parties can build on this seam.
