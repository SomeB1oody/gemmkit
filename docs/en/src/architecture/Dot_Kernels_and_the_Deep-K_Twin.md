# Dot Kernels and the Deep-K Twin

Most kernel families consume the contraction one depth step at a time. Each step loads a column of the packed A panel, and broadcasts one element of packed B. It then issues one FMA, or widening multiply-add, per accumulator register.

2 AVX-512 extensions break that rhythm. Each folds several depth steps into a single instruction. VNNI's `vpdpbusd` multiplies 4 consecutive `i8` depth steps into each of 16 `i32` lanes. AVX-512 BF16's `vdpbf16ps` folds 2 consecutive `bf16` depth steps into each `f32` lane.

An instruction that consumes several depth steps at once wants those steps adjacent in memory. For floats, it also changes how the accumulation rounds. So gemmkit gives dot kernels their own kernel families and ISA tokens, instead of hiding them behind a branch in the shared microkernel.

This page walks through 3 things. First, the 2 seams that carry dot kernels. Second, the 2 concrete dot kernels themselves. Third, the deep-contraction route that both narrow families share.

## Why dot instructions get their own families

A dot kernel differs from its widen sibling in exactly 2 places. Each difference lands on a different extension axis of the engine. See [Scalars and Kernel Families](Scalars_and_Kernel_Families.md) for the family and token split.

The pack layout is a family concern. `KernelFamily::pack_lhs` and `pack_rhs` take no ISA parameter, so a different interleave must key off the family instead. This is why `Bf16DotGemm` is a sibling of `MixedGemm<bf16>`, rather than a branch inside it.

The inner loop is a token concern. Only a CPU that actually has `vpdpbusd` or `vdpbf16ps` can run it. So the instruction lives behind a `KernelSimd` method, and only dot-capable tokens override that method.

The concrete dot families are `IntGemmVnni`, and its requantizing variant `IntGemmVnniQ`, with 4 depth steps per instruction. `Bf16DotGemm` has 2 depth steps per instruction. 2 f32-output twins, covered at the end of this page, round out the set.

## DEPTH_MULTIPLE and the k-group pack

A family that folds `Q` depth steps per instruction declares `const DEPTH_MULTIPLE: usize = Q`. The default is `1`. The contract, spelled out in `gemmkit/src/kernel.rs`, works as follows. The family's packers write panels of `width * kc.next_multiple_of(Q)` elements, and pad the depth tail. The driver strides packed panels by that same padded depth, so both sides stay in lockstep. For every ordinary family, `DEPTH_MULTIPLE = 1`, and every `next_multiple_of` call degenerates to an identity.

The layout itself comes from one shared routine, `pack_kgroup_panels` in `gemmkit/src/pack.rs`. It is the single source of truth for the interleave index math.

Plain `pack_panels` stores a panel depth-major. At each depth step, it stores `width` contiguous leading elements: `mr` rows for the LHS, `nr` columns for the RHS.

`pack_kgroup_panels` instead groups the depth axis `Q` at a time, so one lane's `Q` consecutive depth values become contiguous. Within a panel, group `g`, lane `i`, in-group position `t` lands at offset `g*width*Q + i*Q + t`.

That is exactly the shape one dot instruction reads. A 64-byte A register covers `LANES` rows times `Q` contiguous depth elements. A B group broadcasts `Q` contiguous depth values of one column, as a single 32-bit load.

2 more details of the shared packer matter. It takes a per-element transform, `xform`. That transform is the identity for bf16, and the `+128` bias for VNNI's A operand. The packer fills every pad position with `xform(0)`. Pad positions are the leading positions past the block, and the depth positions past `kc`. This keeps the pad always consistent with the live elements.

The interleaved layout cannot be read in place. So every dot family sets `FORCE_PACK_LHS` and `FORCE_PACK_RHS`, overriding the driver's cost-based pack decision. A dot kernel always pays the pack cost. This is precisely the cost the gates described below hedge against.

A byte-level oracle in `pack.rs`'s tests reimplements the layout naively. It checks that the real routine reproduces it bit-for-bit, across width tails, depth tails, and strided sources.

On the consuming side, `KernelSimd::dot_accumulate` is the seam the families call, instead of the widen-FMA loop. Its default body is `unreachable!`. Only a dot-capable token overrides it, and only a dot family ever calls it.

`Avx512Vnni` and `Avx512Bf16` are distinct tokens from `Avx512F`, because `#[target_feature]` works per token. `_mm512_dpbusd_epi32` needs an `avx512vnni` codegen context. `Avx512F::vectorize` only establishes `avx512f`, so it cannot provide that context.

The method receives the real, unpadded `kc`. It reads `ceil(kc / Q)` instruction groups from the depth-padded panels. Any signedness or bias correction is applied internally, so the accumulators hold the true `sum_k(A*B)` on return.

The fold lives on this dedicated seam, rather than on the generic `accumulate_tile`, for a documented reason. Fusing depth steps reshapes the accumulation rounding, and `accumulate_tile`'s contract forbids that.

## i8 through vpdpbusd

`vpdpbusd` computes an unsigned-times-signed dot. It takes u8 from the first operand and i8 from the second. GEMM wants signed-times-signed instead.

The fix is algebraic, not per-element. The LHS pack offsets every byte by `+128`, into the unsigned domain. This transform is `vnni_a_xform` in `gemmkit/src/kernel/int.rs`. It uses the constant `VNNI_A_BIAS = 128`, defined once in `gemmkit/src/simd.rs`, so the pack and the correction can never drift apart.

`sum_k((A+128)*B) = sum_k(A*B) + 128*sum_k(B)`. So the kernel recovers the true product by subtracting a per-column correction, `128 * sum_k(B[k][j])`. `Avx512Vnni::dot_accumulate` computes those column sums with a small scalar pass over the signed packed B panel, before the vector loop. It then subtracts the splatted correction from every accumulator at the end.

The pads cooperate with this scheme. The A pad is `xform(0) = 128`, and the correction cancels its contribution exactly. The B pad is `0`, so it contributes nothing to the product or to the column sums.

`i32` accumulation wraps, and wrapping addition is associative modulo `2^32`. So regrouping the sum into quads, plus the bias correction, equals the ascending-`k` widen sum bit-for-bit. `IntGemmVnni` and the widen `IntGemm` produce identical output on every input.

The ISA choice can therefore never change an `i8` result. This is a stronger property than the [reproducibility contract](Design_Goals_and_the_Big_Picture.md) requires. That contract only promises reproducible results on a fixed machine and configuration, not bitwise agreement across kernel choices.

That freedom to swap kernels mid-flight is what the small-parallel fallback gate exploits. The VNNI pack is mandatory on both operands. On a small multi-threaded problem, that pack barrier can dominate the compute it is meant to save.

3 conditions trigger the fallback. The ISA selection must be automatic. The parallelism must be `Rayon(n)` with `n != 1`. And `m*n*k` must fall below `GEMMKIT_I8_VNNI_MIN_PAR_MNK`, whose default is `768^3`. When all 3 hold, `dispatch/int.rs` hands the call to the in-place widen kernel instead.

Serial runs and large parallel runs keep VNNI. A forced `GEMMKIT_REQUIRE_ISA=avx512vnni` disables the gate entirely, because a forced pin must run exactly that kernel.

The prepacked-RHS path also bypasses the gate, for 2 reasons. A k-quad-interleaved buffer is only consumable by the VNNI family. The pack barrier the gate hedges against was already amortized once, at prepack time. VNNI's RHS pack is otherwise mandatory on every call. So prepacking is a bigger win there than for any kernel that can read its operand in place.

## bf16 through vdpbf16ps

`Bf16DotGemm` is the floating-point sibling. Its `DEPTH_MULTIPLE` is `2`. Both operands are packed k-pair-interleaved, with each pair stored as one 32-bit `__m512bh` element. `dot_accumulate` issues one `vdpbf16ps` per accumulator, per pair-step.

Everything downstream of the accumulation is shared verbatim with `MixedGemm<bf16>`. This includes the alpha fold and the widen-read, narrow-store epilogue, through the common `mixed_epilogue` helper. The family keeps `OUT_IS_ACC = false`, so the whole contraction accumulates in `f32` and rounds to bf16 exactly once.

The numeric story differs from VNNI in one essential way. `vdpbf16ps`'s fused 2-term dot rounds differently from 2 separate widen-FMAs. So the dot kernel's result is only tolerance-equal to the widen path, not bitwise equal.

That is exactly what the engine's consistency bar allows. Results must be reproducible under a fixed input, environment, and configuration. They need not be bitwise-identical across kernel choices. The dot kernel is fully deterministic. Serial, parallel, and prepacked runs all share the same kernel and pack layout, so they reproduce each other bit-for-bit.

There is no size gate on this path. Auto-selection prefers `Bf16DotGemm` whenever the CPU reports `avx512bf16`, because it is a structural win over the plain widen path. Unlike VNNI, there is no small-parallel fallback here.

3 special-path reroutes are the only exceptions. gemv, `small_mn`, and small-`k` shapes deliberately stay on `MixedGemm<bf16>`'s widen seam. A tiny or degenerate output folds nothing, and the dot pack's depth padding is pure loss there. The `i8` dispatch reroutes its own tiny shapes to the widen kernel, for the same reason.

## The deep-K problem

`OUT_IS_ACC = false` buys single-rounding at a structural price. The driver runs `kc = k`, one depth panel spanning the entire contraction. This replaces the cache-model `kc` slices every homogeneous family gets (see [Blocking and the Cache Model](Blocking_and_the_Cache_Model.md)).

The RHS micropanel a microtile call reads is then `nr * k * sizeof(N)` bytes. Once that micropanel outgrows the L2 cache, every one of the `m/mr` microtile calls in a column strip streams it from L3 or DRAM instead. The even larger `mr * k` LHS micropanel streams from there too. The cliff this creates is sharp. Throughput stays near peak while the micropanel fits in L2, and falls once it does not.

The engage gate in `gemmkit/src/dispatch/mixed.rs` compares that micropanel size against a byte threshold:

```rust
let engage_deep_k = NR
    .checked_mul(t.k)
    .and_then(|x| x.checked_mul(core::mem::size_of::<N>()))
    .is_some_and(|bytes| bytes > crate::cache::deep_k_engage_bytes());
if engage_deep_k {
    run_deep_k_twin::<N, Fam::Twin, S, MR_REG, NR>(simd, &t, par, ws);
    return;
}
```

The threshold is the `GEMMKIT_DEEP_KC_BYTES` knob, taken verbatim when it is non-zero. The default, `0`, derives the threshold as half the effective per-worker L2 capacity.

Half of L2, rather than the whole of it, is a deliberate choice. A gate set to the full L2 size would engage too late. It would fire well past the point where the micropanel no longer fits, and so it would miss the cliff. Half of L2 leaves room for the rest of the working set, and engages the twin while there is still time to avoid the cliff.

The `checked_mul` chain fails closed. A broadcast operand can pass validation with a logically absurd `k`. An overflowing size must fall through to the single panel instead. That panel's own pack sizing then rejects the problem, rather than let a twin multi-slice that `k` forever.

## The f32-output twin

Above the gate, dispatch does not run the narrow family at all. A small `DeepKTwin` trait maps each narrow family to its f32-output twin. `MixedGemm<N>` maps to `MixedGemmF32<N>`, and `Bf16DotGemm` maps to `Bf16DotGemmF32`. The only type change in each twin is `Out = f32 = Acc`.

That one change flips `OUT_IS_ACC` back to its default of `true`. The driver's ordinary multi-slice K blocking then applies unchanged, and every slice's panels are L2-resident again. That is the entire point of the twin.

The pack layout and the accumulation loop are the narrow family's, verbatim. `MixedGemmF32` reuses `pack_panels` and the shared widen-FMA helper. `Bf16DotGemmF32` reuses `pack_kgroup_panels` and `dot_accumulate`. The accumulation helpers only touch the input side of the `KernelSimd` seam. So the accumulator they produce is byte-for-byte what the narrow family would compute.

The twin runs with `alpha = 1` and `beta = 0`, into an `m x n` column-major `f32` scratch buffer. That buffer is drawn from a dedicated `Workspace`. Deep-K is by definition a large-`k` regime, so one `m*n` f32 allocation is negligible. Keeping it separate also leaves the pooled packing workspace free for the twin driver to use.

Afterward, one vectorized sweep computes `narrow(alpha*scratch + beta*widen(C))`. This replicates `mixed_epilogue`'s arithmetic operation for operation, including the same `store_out` narrowing.

What makes the route more than an approximation is how the slices chain together. The twin's microkernel seeds its accumulator registers from the scratch buffer, through a third `KernelSimd<N, N, f32, f32>` seam, `twin_seed` in `gemmkit/src/kernel/mixed.rs`. On an accumulate slice, it loads the running partial into the registers and continues the ascending-`k` chain. It never sums a slice from zero and adds the result afterward.

A store and reload of an `f32` is exact. So the multi-slice sum is exactly the single-panel sum, merely split at slice boundaries. For `beta` in `{0, 1}`, the deep-K result is byte-for-byte the single-panel result.

For a general `beta`, the result holds only to tolerance instead. The reason is mundane. The single panel fuses `beta*C + AB` in one FMA on full tiles, but combines the same terms unfused on edge tiles. No single sweep can match both cases at once.

Serial and parallel runs remain bit-identical in every case. The twin driver's blocking does not depend on the thread count, and the final sweep is elementwise.

The dot twin has one extra alignment rule. The driver rounds the blocking `kc` up to `DEPTH_MULTIPLE`, so an interior slice boundary never splits a k-pair. A split pair would zero-pad mid-contraction and regroup the fused dot incorrectly. With this rule, only the final short tail is ever padded, exactly as in the single-panel case.

3 routes deliberately keep the single panel. They are shallow `k` below the gate, the fused-epilogue entries, and the prepacked-RHS path. On the prepacked path, a `DEPTH_MULTIPLE > 1` buffer requires the whole contraction to stay one depth slice. The driver enforces this with a hard assert, because violating it would misaddress micropanels silently.

The parity claims are tested directly. `gemmkit/tests/deep_k_narrow.rs` toggles `GEMMKIT_DEEP_KC_BYTES`. A value of `1` forces the twin at any `k`, and `usize::MAX` forces the single panel. The test checks byte-for-byte equality for `beta` in `{0, 1}`, and tolerance for general `beta`, on whichever ISA the host selects. [Tuning Knobs](../gemmkit-guide/Tuning_Knobs.md) documents this knob along with the rest.
