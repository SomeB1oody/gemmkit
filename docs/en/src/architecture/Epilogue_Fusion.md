# Epilogue Fusion

A GEMM output rarely leaves the routine raw. Inference layers add a bias and an activation. Quantized pipelines requantize the `i32` accumulator down to a byte. Done naively, each of those is a second full pass over `C`: every element gets written to memory, evicted, read back, transformed, and written again.

The `epilogue` feature (`gemmkit/src/kernel/epilogue.rs`) fuses the transform into the microkernel's store instead. It transforms the element in the register, or scratch slot, it already occupies, at the moment the microkernel would have stored it anyway. The second pass disappears. The saving is even larger for requantization. An unfused flow would have to materialize the entire `m x n` matrix in `i32` before narrowing it.

## The seam

The seam is the `Epilogue` trait. It threads through `KernelFamily::microkernel_epi`, so every family's store site can apply it, and the driver never needs to know it exists:

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
    /// Vector transform of a whole MR_REG x NR register tile; defaults to a loop over
    /// apply_reg. Overridden to hoist a runtime discriminant out of the unrolled pass
    unsafe fn apply_tile<S, const MR_REG: usize, const NR: usize>(&self, simd: S, acc: ..., row0: usize, col0: usize) -> ...;
    /// Vector store-transform from Acc scratch to Out; same bit-agreement contract
    unsafe fn apply_store<S>(&self, simd: S, src: *const Fam::Acc, dst: *mut Fam::Out, ...);
}
```

The whole design rests on 2 invariants. The first is **zero-cost identity**. Plain `gemm` passes the `Identity` epilogue, whose `IS_IDENTITY = true` makes every hook const-fold away. The monomorphized, non-fused kernel stays bit-identical to what it was before the seam existed. Fusion costs nothing when a caller does not use it.

The second is **fire-once semantics**. The driver hands the microkernel a `last_k` flag, and the epilogue applies only on the final depth panel. Earlier panels store raw `Acc` partials, exactly as the non-fused kernel would. A family with `OUT_IS_ACC = false`, such as the narrow `f16`/`bf16` outputs, runs the whole contraction as a single `kc = k` panel, by construction. So `last_k` is structurally true there. The [deep-K twin](Dot_Kernels_and_the_Deep-K_Twin.md) would break that single-panel guarantee, so it deliberately never engages on the fused path. The special paths fire once for free too, since each of their output elements is a single complete reduction with a single store.

## The built-ins

The library ships 3 epilogues, each behind its own public entry point (feature `epilogue`). The requantize entries also need `int8`. See [Fused Epilogues](../gemmkit-guide/Fused_Epilogues.md) for the user-facing view:

**`FusedEpi`** is the runtime-composed bias-plus-activation epilogue. It adds a per-row or per-column bias (`Bias::PerRow` / `Bias::PerCol`), then applies `Relu` or `LeakyRelu(slope)`. A single monomorphization covers every combination, so the fused kernel count does not multiply by the number of epilogue kinds. That only holds because the enum branches decode *once per tile*, in the `apply_tile` override, rather than once per accumulator.

That distinction is not a micro-optimization. The tile const generics unroll the kernel's store pass. A per-register hook would replicate both `match` statements in every one of the tile's accumulator slots. On a wide tile, the resulting branch web costs the compiler the accumulator tile itself. It stops keeping `acc` in registers. Instead, it writes every value through to the stack from inside the `kc` loop, where the epilogue does not even run. Hoisting the decode out of the loop, so it runs once per tile instead, avoids that spill entirely. `tests/perf/fused.rs` pins the ratio between the fused and the plain rate, so a future change cannot let this regression back in unnoticed.

`FusedEpi` backs `gemm_fused` and its whole constellation. This includes `gemm_batched_fused`, which shares 1 bias and 1 activation across the whole batch, the prepacked twins `gemm_packed_b_fused` and `gemm_packed_a_fused`, and the complex entry `gemm_cplx_fused`. The complex entry is bias-only, because an ordering-based activation has no mathematical definition on complex numbers. `FusedEpi` sets `VECTOR = true`. On the fast path, the bias add and the activation run as register operations, such as `max(v, 0)`. Its NaN contract on the SIMD `max`/`min` is chosen so the vector and scalar forms agree exactly: both compute `ReLU(NaN) = 0`.

**`MapEpi`** is the escape hatch. `gemm_map` applies an arbitrary user closure, `f(value, row, col) -> value`, to each output element at its final value. It uses `(row, col)` in the user frame. The closure is a borrowed `&dyn Fn + Sync`, so there is 1 monomorphization per `(type, ISA)`, not 1 per closure. It runs scalar, once per element, amortized by the `O(k)` flops behind each element. `MapEpi` supports `f32`/`f64` only. A narrow type would have to round to `N`, apply the `N`-domain closure, then round again, which would break the bitwise contract described below.

**`KRequantize`** implements the quantized-inference store: `C[r,c] = clamp(zp + round_ne(scale*(acc + bias)), LO, HI)`. It maps the `i32` accumulator down to `i8` (`gemm_i8_requant`, band `[-128, 127]`) or `u8` (`gemm_i8_requant_u8`, band `[0, 255]`, the ONNX QLinearMatMul convention). The scale can be per-tensor or per-row (`RequantScale`). The zero point joins in as an integer after the rounding step. An optional `i32` bias joins in as an integer before the single `f64` rounding step. The rounding itself is round-half-to-even, through a `no_std`-safe `2^52` trick (`round_ne_f64`). `KRequantize` has no `alpha`, since that folds into the scale, and no `beta`, since accumulating into an already-quantized `C` has no clear meaning.

## The correctness contract

This contract is stated precisely because it is exactly what the epilogue tests pin down, bit by bit. It is composed of 3 ingredients:

1. **Identical routing.** A fused call routes every shape through the same kernel plain `gemm` would use. The general driver, gemv, small-k, and small-mn each exist in a fused form (see [Special Paths](Special_Paths.md)). The fused dispatch entries mirror the plain gates one for one. No shape pays the driver's overhead just because it asked for a bias. The one deliberate exception is the mixed `f16`/`bf16` fused gemv, which stays on the driver for a rounding reason explained in Special Paths. Narrow types sit outside the bitwise contract below anyway.
2. **An epilogue-independent engine.** Blocking, scheduling, packing, and the accumulation order do not depend on which epilogue is threaded through. The epilogue only touches the store.
3. **Bit-agreeing apply paths.** A full column-major tile stores through the vector path, `apply_tile`, itself defaulting to `apply_reg`, or through `apply_store`. An edge or strided tile drains through scratch and the scalar `apply` instead. A single output matrix can freely mix the two. So the trait contract requires both paths to agree bit-for-bit under the same token. An `apply_tile` override inherits that obligation. It must leave exactly what `apply_reg` would have, element for element.

Together these 3 give the headline guarantee. For `f32`/`f64`, `gemm_fused`, `gemm_map`, and the batched and prepacked fused entries all equal `gemm()` followed by the same scalar map, **bitwise, for every shape**.

`MapEpi` shows how deliberate that guarantee is. It sets `VECTOR = true`. This is not to vectorize the closure, which it cannot do. It is so the kernel takes the *same path selection* plain `gemm` does. The fast path's fused `beta*C + alpha*AB` store differs from the scalar path's unfused arithmetic by 1 ULP for a general `beta`. A scratch-only epilogue would therefore hand the closure a value plain `gemm` never actually wrote. Instead, `apply_reg` drains the register to a stack buffer, and calls the same scalar `apply` once per lane. So `f` always sees exactly the bits plain `gemm` produced.

The documented exception is `f16`/`bf16`. The narrow blanket implementation applies the bias and the activation in `f32`, on the accumulator, *before* the single round-to-nearest-even narrowing to the output. That is deliberately **more precise** than `gemm()`-then-map, which would round to the narrow type, widen it back, and round again. So for narrow types the fused entries are *not* bitwise equal to gemm-then-map. The documentation states this plainly, rather than weaken the fused semantics to match the less precise alternative. Within 1 fused run, the vector and scalar paths still agree bit-for-bit, since both compute `act(bias(v))` in `f32` and round exactly once. Reproducibility across worker counts stays unchanged too.

`KRequantize` earns its vector path differently. The x86 tokens implement `KernelSimd::requant_store`: a vectorized widen-to-`f64`, scale, hardware round-to-nearest-even, clamp, and low-byte store. Its documentation carries a case-by-case proof that every lane equals the scalar `clamp(zp + round_ne(scale*v), lo, hi)`. The `i32 -> f64` and `f32 -> f64` widenings are exact. The `2^52` trick agrees with the hardware rounding below `2^52`, and saturation agrees above it. A NaN cannot occur, because the API validates that every scale is finite and positive. A per-row scale varies per lane, so that case takes the per-lane scalar map instead. Non-x86 tokens keep `REQUANT_VECTOR = false` and use the scalar map throughout. An in-module conformance sweep, the `requant_store` tests in `gemmkit/src/simd.rs`, checks this bit-equality on every capable token. The word "proven" in this contract is therefore enforced by a test, not just an aspiration.

One last corner remains. When the `A*B` term vanishes, because `k == 0` or `alpha == 0`, the fused entries still owe `C <- act(beta*C + bias)`. That degenerate map runs element-wise in the user frame. `fused_degenerate`, in `gemmkit/src/dispatch/float.rs`, handles this, with a narrow sibling that combines in `f32` and narrows once. So even the no-op-product case honors the same semantics as the full kernel.
