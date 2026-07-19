# Scalars and Kernel Families

Every element type gemmkit multiplies — `f32`, `f64`, `f16`, `bf16`, `i8`, `Complex<f32>`, `Complex<f64>`, with `u8` appearing only as a requantize output — flows through the same driver, the same packing framework, the same cache model, and the same parallel scheduler. Two traits carry all of the variation: `Scalar` at L0 (`gemmkit/src/scalar.rs`) answers "what is this type and what does it accumulate in", and `KernelFamily` at L1 (`gemmkit/src/kernel.rs`) answers "what makes this *kind* of GEMM different from the others". The driver is generic over the family and never branches on element type; this page walks the split and why it is drawn where it is.

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

No `Add`, no `Mul`, no conversions — only the identity constants and the associated accumulator type. The omission is the design: all vectorized arithmetic lives in `SimdOps` (see [SIMD Tokens and ISA Dispatch](SIMD_Tokens_and_ISA_Dispatch.md)), and the scalar arithmetic that epilogues need lives on narrow side-traits that only the types which need them implement — `Float` for `f32`/`f64` (and, via `num-complex`'s operators, the complex types), `NarrowFloat` for the `f16`/`bf16` widen/narrow conversions, `ComplexFloat` for the re/im accessors of the split complex kernel. If `Scalar` itself carried arithmetic, every new element type would owe a full set of operations it may not meaningfully have; `i8` needs no arithmetic trait at all, because its kernel does everything through the SIMD seam and exact `i32` integer ops.

`Acc` is the mixed-precision seam, and the table is short:

| Element type | Accumulates in |
|---|---|
| `f32`, `f64` | itself |
| `f16`, `bf16` | `f32` |
| `i8` (and the output-only `u8`) | `i32` |
| `Complex<f32>`, `Complex<f64>` | itself |

The recursive bound `Acc: Scalar<Acc = Self::Acc>` pins the chain after one step (`f16 -> f32 -> f32 -> ...`), so generic code can name "the accumulator's accumulator" without ever caring how narrow the input was. For the homogeneous types the `Acc = Self` branch collapses at compile time and costs nothing.

## KernelFamily: everything that distinguishes one GEMM from another

`KernelFamily` bundles the rest: the four element types (`Lhs`, `Rhs`, `Acc`, `Out`), the pack layout (`pack_lhs`/`pack_rhs`, which write micropanel-major panels), and the microkernel. Three associated constants shape how the driver treats a family: `OUT_IS_ACC` (whether a running partial sum may round-trip through `C` between depth panels — the pivotal one, discussed below), `FORCE_PACK_LHS`/`FORCE_PACK_RHS` (set when packing performs a transform the kernel depends on, such as complex conjugation or dot-kernel interleaving, so the driver must never read the operand in place), and `DEPTH_MULTIPLE` (the instruction-group depth of a dot kernel; `1` for everything else).

A family implements exactly one of two microkernel methods. A non-fusing family overrides the plain `microkernel` and inherits `microkernel_epi`, whose default forwards to it after asserting `E::IS_IDENTITY` — a fail-closed guard, so a real epilogue reaching a family that cannot fuse panics instead of being silently dropped. A fusing family (the float, mixed, and requantizing families) overrides `microkernel_epi` instead and threads the epilogue through its store; the plain method is then dead and keeps its `unreachable!` default. Tile geometry is deliberately *not* on the trait: `(MR_REG, NR)` is a pair of const generics chosen per `(family, ISA)` at the dispatch site, so a new tile is a new instantiation rather than a new type.

The payoff is the driver's signature: `driver::run::<Fam, S, MR_REG, NR>` is generic over the family and the ISA token, calls `Fam::pack_lhs`/`Fam::pack_rhs`/`Fam::microkernel_epi`, and contains not one `if` on an element type. Adding a kind of GEMM means writing a family, not touching the driver.

## The family roster

Ten family types ship today, and reading them in generations — homogeneous, widen, dot, requantize, complex — makes the seams visible:

| Family | Types (`Lhs`/`Rhs` -> `Acc` -> `Out`) | `OUT_IS_ACC` | `DEPTH_MULTIPLE` | Notes |
|---|---|---|---|---|
| `FloatGemm<T>` | `T -> T -> T` for `f32`/`f64` | `true` | 1 | The baseline: one generic microkernel for every ISA |
| `MixedGemm<N>` | `N -> f32 -> N` for `f16`/`bf16` | `false` | 1 | Widen-FMA through the `KernelSimd` seam |
| `Bf16DotGemm` | `bf16 -> f32 -> bf16` | `false` | 2 | `vdpbf16ps` dot kernel; both operands force-packed k-pair-interleaved |
| `MixedGemmF32<N>` / `Bf16DotGemmF32` | `N -> f32 -> f32` | `true` | 1 / 2 | The f32-output deep-k twins: same accumulation, `f32` store |
| `IntGemm` | `i8 -> i32 -> i32` | `true` | 1 | Exact, wrapping; sign-extend on load |
| `IntGemmVnni` | `i8 -> i32 -> i32` | `true` | 4 | `vpdpbusd` dot kernel, `+128` signedness correction, bit-identical to `IntGemm` |
| `IntGemmQ<O>` / `IntGemmVnniQ<O>` | `i8 -> i32 -> i8` or `u8` | `false` | 1 / 4 | Requantizing variants (feature `epilogue`) |
| `ComplexGemm<T, CONJ_A, CONJ_B>` | `T -> T -> T` for `c32`/`c64` | `true` | 1 | Split (SoA) kernel; both operands force-packed planar; conj is a pack-time sign flip |

`FloatGemm` is the reference point: homogeneous, one generic `microkernel_impl` shared by every ISA and tile. The mixed and integer families introduce `Acc != Lhs` and lean entirely on the widen/narrow seam below; the dot families (`Bf16DotGemm`, `IntGemmVnni`) additionally swap in an interleaved pack layout and a hardware dot instruction, and the f32-output twins exist so a deep contraction can re-block — all covered in [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md). The requantizing variants bolt an exact `i32 -> i8`/`u8` requantize onto the integer accumulation and are part of the fusion story in [Epilogue Fusion](Epilogue_Fusion.md). `ComplexGemm` keeps `Acc = T` so complex `alpha`/`beta` thread through the driver unchanged, but runs its hot loop on the real component through a dedicated seam — the subject of [The Complex Split Kernel](The_Complex_Split_Kernel.md). This page stays at the roster level; the deep dives live in those pages.

## KernelSimd: the widen/narrow seam

The driver's bound on the ISA token is `S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>` (`gemmkit/src/simd.rs`). `KernelSimd<L, R, A, O>` extends `SimdOps<A>` — accumulate in `A` — with the four moves a family needs at the type boundary: `load_lhs` (load `LANES` LHS values, widening to an `A` register), `splat_rhs` (widen one RHS scalar and broadcast), `load_out` (widen output values for the `beta != 0` read of `C`), and `store_out` (narrow an `A` register to `LANES` output values, rounding to nearest-even when narrowing).

The homogeneous case costs nothing: a blanket impl `KernelSimd<A, A, A, A> for S: SimdOps<A>` forwards the four methods to plain `loadu`/`splat`/`storeu`, so `FloatGemm<f32>` and friends need zero per-ISA code. A mixed family adds per-ISA impls whose loads genuinely widen (`f16 -> f32` via `vcvtph2ps`, `i8 -> i32` sign-extension) and whose `store_out` genuinely narrows. Coherence comes free: the all-equal blanket and a mixed impl with `L != A` can never describe the same types. Two further impl groups are derived rather than hand-written per ISA: the requant blankets (`Out = i8`/`u8`, forwarding the accumulate side to the `<i8, i8, i32, i32>` impl) and the f32-output twins (`<N, N, f32, f32>` for `N = f16`/`bf16`, written as two concrete heads because a blanket generic in `N` could not be ruled out of colliding with the homogeneous blanket at `N = f32`). `KernelSimd` also hosts the dot seam (`dot_accumulate`, overridden only by dot-capable tokens, default `unreachable!`) and the vectorized requantize store (`requant_store`, same pattern).

The constant that ties the seam to the driver's blocking is `OUT_IS_ACC`. The driver normally accumulates across `k` by splitting it into `kc` panels and letting the partial sum round-trip through `C` (`beta = 1` after the first panel) — exact when `Out == Acc`. When the output is narrower than the accumulator, that round-trip would round to 16 bits at every panel boundary, so a narrow family declares `OUT_IS_ACC = false` and the driver responds with `kc = k`: one depth panel, the whole contraction accumulated in `f32` registers, and exactly one rounding to the narrow output at the end. That single-rounding guarantee is what makes the mixed results defensible; its cost — a single panel whose RHS micropanel can outgrow L2 at large `k` — is what the f32-output twins exist to pay down, over in [Dot Kernels and the Deep-K Twin](Dot_Kernels_and_the_Deep-K_Twin.md).

## The open/closed proof

The claim that the family seam is open for extension is not left as prose — it is enforced by `gemmkit/tests/open_closed.rs`, an integration test that lives outside the crate and therefore sees only the public API. The test declares `NaiveFloat`, a deliberately naive second float family sharing nothing with `FloatGemm`: it re-implements micropanel packing from scratch (the crate's internal `pack` helper is not visible to it, exactly the third-party situation) and supplies a plain scalar triple-loop `microkernel`. It then drives the *unchanged* generic driver — `driver::run::<NaiveFloat, ScalarTok, 4, 4>` — on a 40x33x28 problem and checks the result against an `f64` reference.

The test's value is mostly that it compiles: a second family drove the driver with no edit to `driver.rs` or `pack.rs`, using nothing but public items (`gemmkit::kernel::KernelFamily`, `gemmkit::simd::ScalarTok`, `gemmkit::driver::run`, `Workspace`, `Parallelism`). Any refactor that closes the seam — a driver branch on a concrete family, a required private helper, a leaked internal type in the trait's signature — breaks this file before it breaks a downstream user. The wider testing story, including how the real families are cross-checked against oracles, is in [Testing and Verification](Testing_and_Verification.md); what third parties can build on the seam is in [Extension Points](Extension_Points.md).
