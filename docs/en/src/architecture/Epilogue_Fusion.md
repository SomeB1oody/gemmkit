# Epilogue Fusion

A GEMM output rarely leaves the routine raw: inference layers add a bias and an activation, quantized pipelines requantize the `i32` accumulator down to a byte. Done naively, each of those is a second full pass over `C` — every element written to memory, evicted, read back, transformed, written again. The `epilogue` feature (`gemmkit/src/kernel/epilogue.rs`) fuses the transform into the microkernel's store instead: the element is transformed in the register (or scratch slot) it already occupies, at the moment it would have been stored anyway, and the second pass disappears. For requantization the saving is even larger — the unfused flow would have to materialize the entire `m x n` matrix in `i32` before narrowing it.

## The seam

The seam is the `Epilogue` trait, threaded through `KernelFamily::microkernel_epi` so that every family's store site can apply it without the driver knowing anything about it:

```rust
// gemmkit/src/kernel/epilogue.rs (trimmed)
pub trait Epilogue<Fam: KernelFamily>: Copy + Send + Sync {
    /// true => every hook const-folds away; the kernel is bit-identical to non-fused
    const IS_IDENTITY: bool = false;
    /// true => apply_reg is implemented, enabling the fast vector store path
    const VECTOR: bool = false;
    /// true => apply_store is implemented (the Out != Acc requantize pattern)
    const VECTOR_STORE: bool = false;

    /// Scalar transform at absolute (row, col) in the oriented problem frame
    unsafe fn apply(&self, v: Fam::Acc, row: usize, col: usize) -> Fam::Out;
    /// Vector transform of LANES consecutive rows; MUST agree with apply bit-for-bit
    unsafe fn apply_reg<S>(&self, simd: S, v: ..., row: usize, col: usize) -> ...;
    /// Vector store-transform from Acc scratch to Out; same bit-agreement contract
    unsafe fn apply_store<S>(&self, simd: S, src: *const Fam::Acc, dst: *mut Fam::Out, ...);
}
```

Two invariants carry the whole design. First, **zero-cost identity**: plain `gemm` passes the `Identity` epilogue, whose `IS_IDENTITY = true` makes every hook const-fold away, so the monomorphized non-fused kernel is bit-identical to what it was before the seam existed — fusion costs nothing when you do not use it. Second, **fire-once semantics**: the driver hands the microkernel a `last_k` flag, and the epilogue applies only on the final depth panel; earlier panels store raw `Acc` partials, exactly as the non-fused kernel would. Families with `OUT_IS_ACC = false` (the narrow `f16`/`bf16` outputs) run the whole contraction as a single `kc = k` panel by construction, so `last_k` is structurally true there — and the [deep-K twin](Dot_Kernels_and_the_Deep-K_Twin.md), which would break that single-panel guarantee, deliberately never engages on the fused path. The special paths fire once for free: each of their output elements is a single complete reduction with a single store.

## The built-ins

Three epilogues ship, each behind its own public entry (feature `epilogue` — the requantize entries additionally need `int8`; see [Fused Epilogues](../gemmkit-guide/Fused_Epilogues.md) for the user-facing view):

**`FusedEpi`** is the runtime-composed bias-plus-activation epilogue: a per-row or per-column bias (`Bias::PerRow` / `Bias::PerCol`), then `Relu` or `LeakyRelu(slope)`. One monomorphization covers every combination — the enum branches are a couple of predictable tests per tile, amortized over the `mr*nr*kc` FMA loop — so the fused kernel count is not multiplied by the number of epilogue kinds. It backs `gemm_fused` and its whole constellation: `gemm_batched_fused` (one shared bias and activation across the batch), the prepacked twins `gemm_packed_b_fused` / `gemm_packed_a_fused`, and the complex entry `gemm_cplx_fused`, which is bias-only because an ordering-based activation is mathematically undefined on complex numbers. It sets `VECTOR = true`: on the fast path the bias add and activation run as register operations (`max(v, 0)` and friends), with a NaN contract on the SIMD `max`/`min` chosen so the vector and scalar forms agree exactly (`ReLU(NaN) = 0` on both).

**`MapEpi`** is the escape hatch: `gemm_map` applies an arbitrary user closure `f(value, row, col) -> value` to each output element at its final value, with `(row, col)` in the user frame. The closure is a borrowed `&dyn Fn + Sync` — one monomorphization per `(type, ISA)`, not per closure — and is invoked scalar, once per element, amortized by the `O(k)` flops behind each element. It is `f32`/`f64` only: a narrow type would have to round to `N`, apply the `N`-domain closure, and round again, breaking the bitwise contract below.

**`KRequantize`** implements the quantized-inference store: `C[r,c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`, from the `i32` accumulator to `i8` (`gemm_i8_requant`, band `[-128, 127]`) or `u8` (`gemm_i8_requant_u8`, band `[0, 255]`, the ONNX QLinearMatMul convention). The scale is per-tensor or per-row (`RequantScale`), the zero point joins in integer after the rounding, the optional `i32` bias joins in integer before the single `f64` rounding step, and the rounding is round-half-to-even via a `no_std`-safe `2^52` trick (`round_ne_f64`). There is no `alpha` (it folds into the scale) and no `beta` (accumulating into an already-quantized `C` is ill-defined).

## The correctness contract

The contract is stated precisely because it is what the epilogue tests pin down, bit by bit. Three ingredients compose:

1. **Identical routing.** A fused call routes every shape through the same kernel plain `gemm` would: the general driver, gemv, small-k, and small-mn each exist in fused form (see [Special Paths](Special_Paths.md)), and the fused dispatch entries mirror the plain gates one for one. No shape pays the driver's overhead just because it asked for a bias. (The one deliberate routing exception is the mixed `f16`/`bf16` fused gemv, which stays on the driver for a rounding reason explained in Special Paths — narrow types sit outside the bitwise contract below anyway.)
2. **An epilogue-independent engine.** Blocking, scheduling, packing, and the accumulation order do not depend on which epilogue is threaded through; the epilogue only touches the store.
3. **Bit-agreeing apply paths.** A full column-major tile stores through the vector path (`apply_reg` / `apply_store`); an edge or strided tile drains through scratch and the scalar `apply`; a single output matrix mixes the two freely — so the trait contract requires them to agree bit-for-bit under the same token.

Together these give the headline guarantee: for `f32`/`f64`, `gemm_fused` (and `gemm_map`, and the batched and prepacked fused entries) equals `gemm()` followed by the same scalar map, **bitwise, for every shape**. `MapEpi` illustrates how deliberate that is: it sets `VECTOR = true` not to vectorize the closure (it cannot) but so the kernel takes the *same path selection* plain `gemm` does — the fast path's fused `beta*C + alpha*AB` store differs from the scalar path's unfused arithmetic by an ULP for general `beta`, so a scratch-only epilogue would hand the closure a value plain `gemm` never wrote. Instead `apply_reg` drains the register to a stack buffer and calls the same scalar `apply` per lane, so `f` always sees exactly the plain-`gemm` bits.

The documented exception is `f16`/`bf16`: the narrow blanket impl applies bias and activation in `f32`, on the accumulator, *before* the single round-to-nearest-even narrowing to the output. That is deliberately **more precise** than `gemm()`-then-map, which would round to the narrow type, widen back, and round again — so for narrow types the fused entries are *not* bitwise-equal to gemm-then-map, and the docs say so rather than weaken the semantics to match. Within a fused run the vector and scalar paths still agree bit-for-bit (both compute `act(bias(v))` in `f32` and round exactly once), and reproducibility across worker counts is unchanged.

`KRequantize` earns its vector path differently. The x86 tokens implement `KernelSimd::requant_store`, a vectorized widen-to-`f64`, scale, hardware round-to-nearest-even, clamp, and low-byte store, and its documentation carries a case-by-case proof that every lane equals the scalar `clamp(zp + round_ne(scale*v), lo, hi)` — exact `i32 -> f64` and `f32 -> f64` widenings, the `2^52` trick coinciding with the hardware rounding below `2^52`, saturation agreeing above it, NaN impossible because the API validates scales finite and positive. A per-row scale varies per lane and takes the per-lane scalar map instead; non-x86 tokens keep `REQUANT_VECTOR = false` and use the scalar map throughout. An in-module conformance sweep (`requant_store` tests in `gemmkit/src/simd.rs`) checks the bit-equality on every capable token, so the "proven" in the contract is enforced, not aspirational.

One last corner: when the `A*B` term vanishes (`k == 0` or `alpha == 0`), the fused entries still owe `C <- act(beta*C + bias)`. That degenerate map runs element-wise in the user frame (`fused_degenerate` in `gemmkit/src/dispatch/float.rs`, with a narrow sibling that combines in `f32` and narrows once), so even the no-op-product case honors the same semantics as the full kernel.
