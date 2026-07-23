//! `f32`/`f64` homogeneous-float dispatch: driver entries for the plain, fused-bias,
//! user-map, and prepacked-RHS routes, the per-ISA wrapper functions, the memoized
//! descriptors, ISA selection, and the `GemmScalar`/`FusedScalar`/`MapScalar` impls that
//! plug into the dispatch layer

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::{
    GemmScalar, PackedConsume, Task, orient_transpose, scale_c_float, small_mn_eligible,
    small_mn_pack_eligible,
};
use crate::driver;
use crate::kernel::FloatGemm;
use crate::kernel::epilogue::{Epilogue, Identity};
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{FusedEpi, MapEpi};
use crate::parallel::Parallelism;
use crate::scalar::Float;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::simd::{ScalarTok, SimdOps};
use crate::special::{gemv, small_k, small_mn};
use crate::tuning;
use crate::workspace::Workspace;

/// The single route-priority ladder for the float family (`f32`/`f64`): route one concrete
/// `(type, ISA, tile)` GEMM through gemv, the small-`m,n` horizontal kernel, the small-`k` panel
/// kernel, or the general driver, whichever the shape and tuning gates select, with the fused
/// [`Epilogue`] `E` threaded into every route. The named entries below ([`run_typed`],
/// [`run_typed_fused`], [`run_typed_map`]) are thin epilogue-choice wrappers over this one body,
/// mirroring the `Identity`-wrapper pattern the special layer and driver already use: with
/// `E = Identity` every hook const-folds away, so the plain route is bit-identical to the
/// non-fused kernel; for a real epilogue each route stores exactly the bits plain `gemm` would and
/// applies the same scalar map exactly once per element (the vector fast path agrees bitwise with
/// the scalar map by the [`Epilogue::apply_reg`] contract), so a fused / map result is `gemm()`
/// then that map for every shape
///
/// Concrete typing here (`T: Float<Acc = T>`) gives the special paths the `Float` bound the fully
/// generic driver entry intentionally lacks
///
/// Route-frame semantics: gemv dispatches **before** orientation normalization, in the user frame,
/// so `epi` still speaks the caller's original coordinates (gemv resolves its own per-row / per-col
/// ambiguity through `swap_rc`, so a [`FusedEpi`] needs no bias-axis flip and a [`MapEpi`] stays
/// `swapped = false`). Every route after it runs in the **oriented** frame, and `on_orient_swap`
/// re-orients the epilogue's frame-dependent state exactly once on a swap - a field write flipping
/// a bias axis or flagging a coordinate transpose, not a new monomorphization
///
/// # Safety
/// As [`crate::dispatch::execute`], plus `epi`'s interior pointers valid for the (pre-swap) problem's `m`/`n`
#[inline]
unsafe fn run_typed_epi<T, S, E, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<T>,
    mut epi: E,
    par: Parallelism,
    ws: &mut Workspace,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
    E: Epilogue<FloatGemm<T>>,
{
    unsafe {
        // gemv/gevv shape, skipped if tuning::gemv_threshold() has been lowered below the true
        // minimum dimension; either way falls through correctly to the general driver. gemv
        // dispatches before orientation normalization, so epi stays in the user frame: it resolves
        // the per-row / per-col coordinate itself from its own n == 1 / m == 1 swap_rc branch (no
        // bias-axis flip, a MapEpi stays swapped == false)
        if (t.n == 1 || t.m == 1) && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold() {
            gemv::run_typed_epi::<T, S, E>(
                simd, t.m, t.k, t.n, par, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                t.c, t.rsc, t.csc, &epi,
            );
            return;
        }

        // Orientation normalization transposes the engine frame for the routes below (they all
        // consume the oriented epi): a row-major-ish C computes C^T = B^T*A^T (swapping m<->n), so
        // on_orient_swap re-orients epi's frame-dependent state once - flipping a FusedEpi bias
        // axis (per-row becomes per-col) or flagging a MapEpi (row, col) transpose (a no-op for
        // Identity)
        if orient_transpose(&mut t) {
            epi.on_orient_swap();
        }

        // Small m,n with a long contraction: the driver would pad the tiny row/col tiles to a
        // full microtile and pack mostly padding, where the horizontal path computes each output
        // as a direct SIMD dot over k instead, applying the epilogue at that cell's single store
        // The zero-copy gate needs k above small_k_threshold and both operands unit-stride along k
        // (csa == 1, rsb == 1); the pack gate needs k above its own small_mn_pack_min_k floor and
        // covers the rest by copying only the failing operand into k-contiguous scratch first (the
        // epilogue still fires on the same per-cell store either way). Short k instead takes the
        // small_k route below
        if small_mn_eligible(&t) || small_mn_pack_eligible(&t) {
            small_mn::run_epi::<T, S, E>(
                simd, t.m, t.k, t.n, par, ws, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb,
                t.beta, t.c, t.rsc, t.csc, &epi,
            );
            return;
        }
        // Low-depth shape: the whole product fits in one depth panel, so the driver's
        // blocking/packing setup would be pure overhead; read A/B in place instead and apply the
        // epilogue at that single per-tile store (last_k is structurally true here)
        if t.k <= tuning::small_k_threshold() {
            small_k::run_epi::<FloatGemm<T>, S, E, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, &epi, par, ws,
            );
            return;
        }
        driver::run_epilogue::<FloatGemm<T>, S, E, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, &epi, par, ws,
        );
    }
}

/// Route one concrete `(type, ISA, tile)` plain GEMM: the `E = Identity` choice of
/// [`run_typed_epi`], so every epilogue hook const-folds away and the monomorphization is
/// bit-identical to a non-fused kernel
///
/// # Safety
/// As [`crate::dispatch::execute`]
#[inline]
unsafe fn run_typed<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: Task<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe { run_typed_epi::<T, S, Identity, MR_REG, NR>(simd, t, Identity, par, ws) }
}

/// Fused-epilogue driver entry for a concrete `(type, ISA, tile)`: the [`FusedEpi`] choice of
/// [`run_typed_epi`]. A fused shape takes the same kernel plain `gemm` would (gemv / small_mn /
/// small_k / general driver) rather than paying the driver's pack/blocking overhead on a shape one
/// of the special paths wins, and each route stores exactly the bits plain `gemm` would and applies
/// the same scalar map exactly once per element, so the fused result is bit-identical to `gemm()`
/// followed by that map for every shape (the vector fast path agrees bitwise with the scalar map by
/// the [`Epilogue::apply_reg`] contract). The orientation swap flips the bias axis through
/// `on_orient_swap` (see [`run_typed_epi`] for the route-frame semantics)
///
/// # Safety
/// As [`run_typed`], plus `epi`'s interior pointers valid for the (pre-swap) problem's `m`/`n`
#[cfg(feature = "epilogue")]
#[inline]
unsafe fn run_typed_fused<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: Task<T>,
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    // FusedEpi<T>: Epilogue<FloatGemm<T>> needs Float<Acc = T> + PartialOrd; FusedScalar itself
    // does not imply them, since it also covers the narrow f16/bf16 types, which route through
    // run_typed_mixed_fused instead
    T: Float<Acc = T> + PartialOrd,
    S: SimdOps<T>,
{
    unsafe { run_typed_epi::<T, S, FusedEpi<T>, MR_REG, NR>(simd, t, epi, par, ws) }
}

/// User-defined map-epilogue driver entry for a concrete `(type, ISA, tile)`: the borrowed-closure
/// [`MapEpi`] choice of [`run_typed_epi`]. A `gemm_map` shape takes the same kernel plain `gemm`
/// would (gemv / small_mn / small_k / general driver), each route storing exactly the bits plain
/// `gemm` would and then applying the closure exactly once per element. [`MapEpi`] sets
/// `VECTOR = true`, so the microkernel takes the same path selection plain `gemm` does (fast vector
/// store for a full column-major tile, scratch for an edge), and the value handed to the closure is
/// bit-for-bit the plain-`gemm` store value on every path, so `gemm_map` is `gemm()` then the
/// per-element `f` for every `f32`/`f64` shape. The orientation swap flags `MapEpi`'s coordinate
/// transpose through `on_orient_swap`, so [`MapEpi::apply`] flips `(row, col)` back to the user
/// frame for the closure (see [`run_typed_epi`] for the route-frame semantics)
///
/// # Safety
/// As [`run_typed`]; the closure in `epi` is total (it is called on every stored element)
#[cfg(feature = "epilogue")]
#[inline]
unsafe fn run_typed_map<'u, T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: Task<T>,
    epi: MapEpi<'u, T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    // The real-float map epilogue (`MapEpi<T>: Epilogue<FloatGemm<T>>`) and the special paths need
    // `Float<Acc = T>`; the closure never compares, so (unlike `FusedEpi`) no `PartialOrd`
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe { run_typed_epi::<T, S, MapEpi<'u, T>, MR_REG, NR>(simd, t, epi, par, ws) }
}

/// Prepacked-RHS driver entry for a concrete `(type, ISA, tile)`. No gemv route and no
/// orientation swap: the API guarantees a column-major-ish C (`|csc| >= |rsc|`), so the
/// prepacked buffer is always the genuine RHS
///
/// # Safety
/// As [`run_typed`], plus `req.packed` valid for the recorded geometry
#[inline]
unsafe fn run_packed_typed<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        // The driver reads panels with the buffer's own (kc, nc), so nothing is re-derived
        // nr is structural (the panel width is this kernel's NR); one process's memoized ISA
        // choice guarantees they agree
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<FloatGemm<T>, S, MR_REG, NR>(
            simd, req.m, req.k, req.n, req.alpha, req.a, req.rsa, req.csa, req.packed, req.kc,
            req.nc, req.beta, req.c, req.rsc, req.csc, par, ws,
        );
    }
}

/// Fused-epilogue prepacked-RHS driver entry for a concrete `(type, ISA, tile)`: the mirror of
/// [`run_packed_typed`] threading `epi` into the prepacked driver entry
/// ([`driver::run_packed_rhs_epilogue`]). Like the plain prepacked path there is no gemv route
/// and no orientation swap in the driver: the consume frame is the frame the buffer was packed
/// for, so `epi` is applied verbatim (the `gemm_packed_a_fused` entry has already flipped the
/// bias axis where its transposed consume requires it). Because the engine
/// (blocking/scheduling/panel bytes) is epilogue-independent, the fused result is bit-identical
/// to a plain prepacked GEMM then the same scalar map for `f32`/`f64`
///
/// # Safety
/// As [`run_packed_typed`], plus `epi`'s interior pointers valid for the problem's `m`/`n`
#[cfg(feature = "epilogue")]
#[inline]
unsafe fn run_typed_packed_fused<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<T>,
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    // FusedEpi<T>: Epilogue<FloatGemm<T>> needs Float<Acc = T> + PartialOrd (as run_typed_fused)
    T: Float<Acc = T> + PartialOrd,
    S: SimdOps<T>,
{
    unsafe {
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs_epilogue::<FloatGemm<T>, S, FusedEpi<T>, MR_REG, NR>(
            simd, req.m, req.k, req.n, req.alpha, req.a, req.rsa, req.csa, req.packed, req.kc,
            req.nc, req.beta, req.c, req.rsc, req.csc, &epi, par, ws,
        );
    }
}

// per-type, per-ISA monomorphized entry points (the dispatch slots)
//
// Tile geometry (MR_REG, NR) is the only per-(type, ISA) knob; everything else is
// shared generic code. MR = MR_REG * LANES

unsafe fn gemm_f32_scalar(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed::<f32, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
unsafe fn gemm_f64_scalar(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed::<f64, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_fma(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 6 -> 12 acc + 2 lhs + 1 rhs = 15 of 16 YMM
    unsafe { run_typed::<f32, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_fma(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*4 = 8, NR = 6, same 15-YMM budget as f32
    unsafe { run_typed::<f64, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_avx512(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*16 = 32, NR = 12 -> 24 acc + 2 lhs + 1 rhs = 27 of 32 ZMM
    unsafe { run_typed::<f32, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_avx512(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 12, same 27-ZMM budget as f32
    unsafe { run_typed::<f64, Avx512, 2, 12>(Avx512, t, par, ws) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f32_neon(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 4*4 = 16, NR = 4 -> 16 acc + 4 lhs + 1 rhs = 21 of the 32 v0-v31 vector
    // registers (NR == LANES, so 1 loaded RHS vector feeds all 4 columns). The ~11
    // spare registers are deliberate: they give the wide out-of-order window rename
    // headroom to overlap the next step's loads with the current FMAs
    unsafe { run_typed::<f32, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f64_neon(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 4*2 = 8, NR = 4 -> 16 acc + 4 lhs + 2 rhs = 22 vregs, same low-pressure
    // tile as f32
    unsafe { run_typed::<f64, Neon, 4, 4>(Neon, t, par, ws) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f32_simd128(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*4 = 8, NR = 4 -> 8 acc + 2 lhs + 1 rhs = 11 live v128. wasm has no
    // hardware FMA (LANE_FMA is false) and LLVM's wasm backend spills past ~16 live
    // vectors, so the wider NEON-style 4x4 tile would over-subscribe
    unsafe { run_typed::<f32, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f64_simd128(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*2 = 4, NR = 4 -> 8 acc + 2 lhs + 1 rhs = 11 live v128, same tile shape
    // as f32 (f64 just packs 2 lanes per register)
    unsafe { run_typed::<f64, Simd128, 2, 4>(Simd128, t, par, ws) }
}

// prepacked-RHS entry points: one per (type, ISA), reusing the plain wrappers' tiles

unsafe fn gemm_f32_scalar_packed(r: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f32, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
unsafe fn gemm_f64_scalar_packed(r: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f64, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_fma_packed(r: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f32, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_fma_packed(r: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f64, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_avx512_packed(r: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f32, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_avx512_packed(r: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f64, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f32_neon_packed(r: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f32, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f64_neon_packed(r: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f64, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f32_simd128_packed(r: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f32, Simd128, 2, 4>(Simd128, r, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f64_simd128_packed(r: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed::<f64, Simd128, 2, 4>(Simd128, r, par, ws) }
}

// fused-epilogue entry points: one per (f32/f64, ISA), reusing the plain wrappers' tiles
// (the epilogue fuses into the existing store, so the register budget is unchanged)

#[cfg(feature = "epilogue")]
unsafe fn gemm_f32_scalar_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
#[cfg(feature = "epilogue")]
unsafe fn gemm_f64_scalar_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_fma_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_fma_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_avx512_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_avx512_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f32_neon_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f64_neon_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f32_simd128_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f64_simd128_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}

// prepacked-RHS fused-epilogue entry points: one per (f32/f64, ISA), reusing the plain
// prepacked wrappers' tiles (the epilogue fuses into the existing store, so the register
// budget is unchanged). Each is cfg-gated exactly like its plain prepacked sibling plus epilogue

#[cfg(feature = "epilogue")]
unsafe fn gemm_f32_scalar_packed_fused(
    r: PackedConsume<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f32, ScalarTok, 4, 4>(ScalarTok, r, epi, par, ws) }
}
#[cfg(feature = "epilogue")]
unsafe fn gemm_f64_scalar_packed_fused(
    r: PackedConsume<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f64, ScalarTok, 4, 4>(ScalarTok, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_fma_packed_fused(
    r: PackedConsume<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f32, Fma, 2, 6>(Fma, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_fma_packed_fused(
    r: PackedConsume<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f64, Fma, 2, 6>(Fma, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_avx512_packed_fused(
    r: PackedConsume<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f32, Avx512, 2, 12>(Avx512, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_avx512_packed_fused(
    r: PackedConsume<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f64, Avx512, 2, 12>(Avx512, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f32_neon_packed_fused(
    r: PackedConsume<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f32, Neon, 4, 4>(Neon, r, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f64_neon_packed_fused(
    r: PackedConsume<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f64, Neon, 4, 4>(Neon, r, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f32_simd128_packed_fused(
    r: PackedConsume<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f32, Simd128, 2, 4>(Simd128, r, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f64_simd128_packed_fused(
    r: PackedConsume<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_packed_fused::<f64, Simd128, 2, 4>(Simd128, r, epi, par, ws) }
}

/// The sealed element-type bound for the fused-epilogue public API: the real floats `f32`/`f64`
/// and, under the `half` feature, the narrow floats `f16`/`bf16`. A superset of [`GemmScalar`]
/// (for dispatch), sealed (a private supertrait) so downstream crates cannot widen the fused
/// surface. It does not require `Float<Acc = Self> + PartialOrd`: the real-float [`FusedEpi`]
/// arithmetic keeps those bounds on its own `Epilogue<FloatGemm<T>>` impl, and the narrow types
/// are not `Float` (they widen to `f32`). What every fused type must provide is the finiteness
/// test used to validate a `LeakyRelu` slope, and the degenerate `C <- act(beta*C + bias)` map
/// (type-specific: real floats compute in `T`, narrow types in `f32`, narrowing once)
#[cfg(feature = "epilogue")]
pub trait FusedScalar: GemmScalar + sealed::Sealed {
    /// `true` iff `self` is finite. `f32`/`f64` use the inherent test; `f16`/`bf16` widen to
    /// `f32` first. `core`-only, so it is `no_std`-safe
    #[doc(hidden)]
    fn finite(self) -> bool;

    /// The degenerate fused epilogue `C[i,j] <- apply(beta*C[i,j], i, j)` in the user frame, run
    /// when the `A*B` term vanishes (`k == 0` or `alpha == 0`)
    ///
    /// # Safety
    /// `c` valid for the `m x n` region; `epi`'s bias valid for the problem's `m`/`n`
    #[doc(hidden)]
    unsafe fn fused_degenerate(t: &Task<Self>, epi: &FusedEpi<Self>);
}

// Sealed supertrait: only this crate can implement FusedScalar
#[cfg(feature = "epilogue")]
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    #[cfg(feature = "half")]
    impl Sealed for half::f16 {}
    #[cfg(feature = "half")]
    impl Sealed for half::bf16 {}
}

#[cfg(feature = "epilogue")]
impl FusedScalar for f32 {
    #[inline]
    fn finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    unsafe fn fused_degenerate(t: &Task<f32>, epi: &FusedEpi<f32>) {
        unsafe { fused_degenerate_float::<f32>(t, epi) }
    }
}
#[cfg(feature = "epilogue")]
impl FusedScalar for f64 {
    #[inline]
    fn finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    unsafe fn fused_degenerate(t: &Task<f64>, epi: &FusedEpi<f64>) {
        unsafe { fused_degenerate_float::<f64>(t, epi) }
    }
}

/// Top-level fused entry (called by the API layer): handle the degenerate case in the user
/// frame (before orientation), then run the ISA-dispatched fused kernel
///
/// # Safety
/// `task`'s pointers must be valid; `c` must not alias `a`/`b`, and `epi`'s bias slice must
/// not overlap `c` (the API validates this)
#[cfg(feature = "epilogue")]
pub(crate) unsafe fn execute_fused<T: FusedScalar>(
    task: Task<T>,
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if task.m == 0 || task.n == 0 {
            return;
        }
        // A*B vanishes (k == 0 or alpha == 0): C <- act(beta*C + bias), element-wise in the
        // user frame (bias axes as the caller specified). This is type-specific (narrow types
        // combine in f32 and narrow once), so it is a FusedScalar method that each type
        // implements on its own
        if task.k == 0 || task.alpha == T::ZERO {
            T::fused_degenerate(&task, &epi);
            return;
        }
        T::dispatch_fused(task, epi, par, ws);
    }
}

/// Top-level prepacked-RHS fused entry (called by the API layer): the prepacked twin of
/// [`execute_fused`] that also mirrors [`crate::dispatch::execute_packed`]'s degenerate
/// handling. Handles the degenerate case in the prepacked buffer's oriented frame, which is
/// where `req` and `epi` already live, then runs the ISA-dispatched prepacked-fused kernel.
/// `req.packed` is never read on the degenerate path (the `A*B` term vanishes)
///
/// # Safety
/// As [`crate::dispatch::execute_packed`], plus `epi`'s bias valid for the problem's `m`/`n` and
/// disjoint from `c` (the API validates this)
#[cfg(feature = "epilogue")]
pub(crate) unsafe fn execute_packed_fused<T: FusedScalar>(
    req: PackedConsume<T>,
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if req.m == 0 || req.n == 0 {
            return;
        }
        // A*B vanishes (k == 0 or alpha == 0): C <- act(beta*C + bias), element-wise in the
        // consume (oriented) frame. fused_degenerate reads only the output geometry
        // (m/n/c/strides/beta) and epi, so a Task built from req (with a null, unread b) drives
        // it at the right coordinates; epi is already oriented (the gemm_packed_a entry
        // pre-flipped the bias axis), so the degenerate matches the compute path's bias axis
        if req.k == 0 || req.alpha == T::ZERO {
            let task = Task {
                m: req.m,
                k: req.k,
                n: req.n,
                alpha: req.alpha,
                a: req.a,
                rsa: req.rsa,
                csa: req.csa,
                b: core::ptr::null(),
                rsb: 0,
                csb: 0,
                beta: req.beta,
                c: req.c,
                rsc: req.rsc,
                csc: req.csc,
            };
            T::fused_degenerate(&task, &epi);
            return;
        }
        T::dispatch_packed_fused(req, epi, par, ws);
    }
}

/// The degenerate fused epilogue `C[i,j] <- apply(beta*C[i,j], i, j)` in the user frame, for the
/// real floats (`f32`/`f64`): all arithmetic is in `T`. The narrow (`f16`/`bf16`) sibling (which
/// combines in `f32` and narrows once) lives in [`crate::dispatch`]'s `mixed` module
///
/// # Safety
/// `c` valid for the `m x n` region; `epi`'s bias valid for the problem's `m`/`n`
#[cfg(feature = "epilogue")]
pub(super) unsafe fn fused_degenerate_float<T: Float<Acc = T> + PartialOrd>(
    t: &Task<T>,
    epi: &FusedEpi<T>,
) {
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                let p = t.c.offset(i as isize * t.rsc + j as isize * t.csc);
                let base = if t.beta == T::ZERO {
                    T::ZERO
                } else if t.beta == T::ONE {
                    *p
                } else {
                    t.beta * *p
                };
                *p = epi.apply(base, i, j);
            }
        }
    }
}

type GemmFn<T> = unsafe fn(Task<T>, Parallelism, &mut Workspace);
type PackedFn<T> = unsafe fn(PackedConsume<T>, Parallelism, &mut Workspace);
/// The fused-epilogue kernel entry: a plain [`Task`] plus the runtime-composed [`FusedEpi`].
/// Every [`FusedScalar`] type (`f32`/`f64` here, `f16`/`bf16` in the `mixed` module) supplies
/// one, so the slot is non-optional. `pub(super)` so `dispatch/mixed.rs` can name it (as with
/// [`Dispatched`])
#[cfg(feature = "epilogue")]
pub(super) type FusedFn<T> = unsafe fn(Task<T>, FusedEpi<T>, Parallelism, &mut Workspace);
/// The prepacked-RHS fused-epilogue kernel entry: a [`PackedConsume`] plus the runtime-composed
/// [`FusedEpi`]. Like [`FusedFn`], every [`FusedScalar`] type supplies one, so the slot is
/// non-optional. `pub(super)` so `dispatch/mixed.rs` can name it
#[cfg(feature = "epilogue")]
pub(super) type PackedFusedFn<T> =
    unsafe fn(PackedConsume<T>, FusedEpi<T>, Parallelism, &mut Workspace);

/// The memoized dispatch slot for one element type: the plain kernel, the prepacked-RHS
/// kernel, the fused-epilogue kernels, and the microtile `(mr, nr)` they all share. Bundling
/// them keeps adding an ISA a single `select_*` ladder arm. `mr`/`nr` mirror the tile constants
/// in the wrappers above and feed `prepack_rhs` (via `rhs_tile`) so the buffer and the consume
/// path agree on the blocking geometry
#[derive(Copy, Clone)]
pub(super) struct Dispatched<T> {
    pub(super) run: GemmFn<T>,
    pub(super) run_packed: PackedFn<T>,
    /// Fused-epilogue entry (bias/activation). Every dispatched type supplies one (`f32`/`f64`
    /// and, under `half`, `f16`/`bf16`), so it is non-optional
    #[cfg(feature = "epilogue")]
    pub(super) run_fused: FusedFn<T>,
    /// Prepacked-RHS fused-epilogue entry: the fused twin of `run_packed`. Every dispatched type
    /// supplies one, so it is non-optional (like `run_fused`)
    #[cfg(feature = "epilogue")]
    pub(super) run_packed_fused: PackedFusedFn<T>,
    pub(super) mr: usize,
    pub(super) nr: usize,
    /// The dispatched kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`]: `1` for
    /// every widen/homogeneous kernel, `2` for the bf16 `vdpbf16ps` dot kernel. The prepack
    /// constructor rounds the packed depth up to it (via [`GemmScalar`]). Read only by the
    /// `bf16` prepack path, so it is dead code without the `half` feature
    #[cfg_attr(not(feature = "half"), allow(dead_code))]
    pub(super) depth_multiple: usize,
}

// One descriptor per (type, ISA). mr = MR_REG*LANES, nr = NR: mirrors the tile in each
// wrapper's comment above (scalar 4x4; FMA 16x6 / f64 8x6; AVX-512 32x12 / f64 16x12;
// NEON 16x4 / f64 8x4; simd128 8x4 / f64 4x4)
const DISP_F32_SCALAR: Dispatched<f32> = Dispatched {
    run: gemm_f32_scalar,
    run_packed: gemm_f32_scalar_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f32_scalar_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f32_scalar_packed_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
const DISP_F64_SCALAR: Dispatched<f64> = Dispatched {
    run: gemm_f64_scalar,
    run_packed: gemm_f64_scalar_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f64_scalar_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f64_scalar_packed_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_FMA: Dispatched<f32> = Dispatched {
    run: gemm_f32_fma,
    run_packed: gemm_f32_fma_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f32_fma_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f32_fma_packed_fused,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_FMA: Dispatched<f64> = Dispatched {
    run: gemm_f64_fma,
    run_packed: gemm_f64_fma_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f64_fma_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f64_fma_packed_fused,
    mr: 8,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_AVX512: Dispatched<f32> = Dispatched {
    run: gemm_f32_avx512,
    run_packed: gemm_f32_avx512_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f32_avx512_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f32_avx512_packed_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_AVX512: Dispatched<f64> = Dispatched {
    run: gemm_f64_avx512,
    run_packed: gemm_f64_avx512_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f64_avx512_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f64_avx512_packed_fused,
    mr: 16,
    nr: 12,
    depth_multiple: 1,
};

#[cfg(target_arch = "aarch64")]
const DISP_F32_NEON: Dispatched<f32> = Dispatched {
    run: gemm_f32_neon,
    run_packed: gemm_f32_neon_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f32_neon_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f32_neon_packed_fused,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(target_arch = "aarch64")]
const DISP_F64_NEON: Dispatched<f64> = Dispatched {
    run: gemm_f64_neon,
    run_packed: gemm_f64_neon_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f64_neon_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f64_neon_packed_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F32_SIMD128: Dispatched<f32> = Dispatched {
    run: gemm_f32_simd128,
    run_packed: gemm_f32_simd128_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f32_simd128_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f32_simd128_packed_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F64_SIMD128: Dispatched<f64> = Dispatched {
    run: gemm_f64_simd128,
    run_packed: gemm_f64_simd128_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f64_simd128_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f64_simd128_packed_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

fn select_f32() -> Dispatched<f32> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F32_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_F32_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F32_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F32_NEON, // aarch64 guarantees NEON, so this arm never panics
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_F32_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => {
            panic!(
                "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
            )
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return DISP_F32_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_F32_FMA;
        }
    }
    // aarch64 has no lower ISA tier: NEON is the only fallback
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F32_NEON
    }
    // wasm32: simd128 if enabled, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_F32_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_F32_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_F32_SCALAR
    }
}

fn select_f64() -> Dispatched<f64> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F64_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_F64_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F64_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F64_NEON, // aarch64 guarantees NEON, so this arm never panics
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_F64_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => {
            panic!(
                "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
            )
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return DISP_F64_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_F64_FMA;
        }
    }
    // aarch64 has no lower ISA tier: NEON is the only fallback
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F64_NEON
    }
    // wasm32: simd128 if enabled, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_F64_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_F64_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_F64_SCALAR
    }
}

memoized_select!(
    GEMM_F32,
    dispatched_f32,
    Dispatched<f32>,
    select_f32,
    "The memoized dispatch descriptor for `f32` (selection runs once)."
);
memoized_select!(
    GEMM_F64,
    dispatched_f64,
    Dispatched<f64>,
    select_f64,
    "The memoized dispatch descriptor for `f64` (selection runs once)."
);

/// Emit the `GemmScalar` impl for a homogeneous float type (`f32` / `f64`): `Out == Acc`
/// (`OUT_IS_ACC = true`), in-place `scale_c`, packing through `FloatGemm<$t>`, and the
/// always-present fused path. `$disp` is the memoized dispatch accessor. The 2 float impls
/// are pure type substitutions, so this macro keeps them from drifting apart. The narrow
/// `f16`/`bf16` impls differ too much to share it (narrow scale, `MixedGemm`, no fused path,
/// bf16's depth-multiple pack switch), so they stay hand-written in `dispatch/mixed.rs`
macro_rules! float_gemm_scalar {
    ($t:ty, $disp:ident) => {
        impl GemmScalar for $t {
            const OUT_IS_ACC: bool = true;
            #[inline]
            unsafe fn scale_c(beta: $t, c: *mut $t, m: usize, n: usize, rsc: isize, csc: isize) {
                unsafe { scale_c_float(beta, c, m, n, rsc, csc) }
            }
            #[inline]
            unsafe fn pack_rhs_full(
                dst: *mut $t,
                b: *const $t,
                rsb: isize,
                csb: isize,
                k: usize,
                n: usize,
                kc: usize,
                nc: usize,
                nr: usize,
            ) {
                unsafe {
                    driver::pack_rhs_full::<FloatGemm<$t>>(dst, b, rsb, csb, k, n, kc, nc, nr)
                }
            }
            #[inline]
            unsafe fn dispatch(task: Task<$t>, par: Parallelism, ws: &mut Workspace) {
                unsafe { ($disp().run)(task, par, ws) }
            }
            #[inline]
            unsafe fn dispatch_packed(
                req: PackedConsume<$t>,
                par: Parallelism,
                ws: &mut Workspace,
            ) {
                unsafe { ($disp().run_packed)(req, par, ws) }
            }
            #[cfg(feature = "epilogue")]
            #[inline]
            unsafe fn dispatch_fused(
                t: Task<$t>,
                epi: FusedEpi<$t>,
                par: Parallelism,
                ws: &mut Workspace,
            ) {
                unsafe { ($disp().run_fused)(t, epi, par, ws) }
            }
            #[cfg(feature = "epilogue")]
            #[inline]
            unsafe fn dispatch_packed_fused(
                req: PackedConsume<$t>,
                epi: FusedEpi<$t>,
                par: Parallelism,
                ws: &mut Workspace,
            ) {
                unsafe { ($disp().run_packed_fused)(req, epi, par, ws) }
            }
            #[inline]
            fn rhs_tile() -> (usize, usize) {
                let d = $disp();
                (d.mr, d.nr)
            }
        }
    };
}

float_gemm_scalar!(f32, dispatched_f32);
float_gemm_scalar!(f64, dispatched_f64);

// user-defined per-element map epilogue (gemm_map)
//
// Kept off the shared Dispatched<T> descriptor on purpose: that struct is reused by the narrow
// (f16/bf16) mixed module, which has no valid map path (a T-domain closure after the f32
// accumulate would double-round). Adding a run_map field would force a bogus f16 map wrapper
// into every mixed descriptor. Instead the map entry rides its own memoized table (MAP_F32 /
// MAP_F64), built only for f32/f64, so Dispatched and the mixed module stay untouched and there
// is no unreachable narrow map path anywhere

/// The map-epilogue kernel entry: a [`Task`] plus the borrowed-closure [`MapEpi`]. Higher-ranked
/// over the closure lifetime `'u` so one memoized function pointer serves every call. Only
/// `f32`/`f64` supply one (the [`MapScalar`] seal)
#[cfg(feature = "epilogue")]
type MapFn<T> = for<'u> unsafe fn(Task<T>, MapEpi<'u, T>, Parallelism, &mut Workspace);

// map-epilogue entry points: one per (f32/f64, ISA), reusing the plain wrappers' tiles (the
// closure fuses into the existing store, so the register budget is unchanged). Each is
// cfg-gated exactly like its plain sibling plus epilogue

#[cfg(feature = "epilogue")]
unsafe fn gemm_f32_scalar_map(
    t: Task<f32>,
    epi: MapEpi<'_, f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f32, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
#[cfg(feature = "epilogue")]
unsafe fn gemm_f64_scalar_map(
    t: Task<f64>,
    epi: MapEpi<'_, f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f64, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_fma_map(
    t: Task<f32>,
    epi: MapEpi<'_, f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f32, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_fma_map(
    t: Task<f64>,
    epi: MapEpi<'_, f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f64, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f32_avx512_map(
    t: Task<f32>,
    epi: MapEpi<'_, f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f32, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f64_avx512_map(
    t: Task<f64>,
    epi: MapEpi<'_, f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f64, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f32_neon_map(
    t: Task<f32>,
    epi: MapEpi<'_, f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f32, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f64_neon_map(
    t: Task<f64>,
    epi: MapEpi<'_, f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f64, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f32_simd128_map(
    t: Task<f32>,
    epi: MapEpi<'_, f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f32, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}
#[cfg(all(
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f64_simd128_map(
    t: Task<f64>,
    epi: MapEpi<'_, f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_map::<f64, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}

/// Select the `f32` map-epilogue entry for the running CPU, mirroring [`select_f32`]'s ISA
/// ladder exactly (same `GEMMKIT_REQUIRE_ISA` pins, same detection order) but returning the
/// map wrapper instead
#[cfg(feature = "epilogue")]
fn select_map_f32() -> MapFn<f32> {
    match forced_isa() {
        ForcedIsa::Scalar => return gemm_f32_scalar_map,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return gemm_f32_fma_map;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return gemm_f32_avx512_map;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return gemm_f32_neon_map,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return gemm_f32_simd128_map,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return gemm_f32_avx512_map;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return gemm_f32_fma_map;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        gemm_f32_neon_map
    }
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            gemm_f32_simd128_map
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            gemm_f32_scalar_map
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        gemm_f32_scalar_map
    }
}

/// Select the `f64` map-epilogue entry for the running CPU (mirrors [`select_f64`])
#[cfg(feature = "epilogue")]
fn select_map_f64() -> MapFn<f64> {
    match forced_isa() {
        ForcedIsa::Scalar => return gemm_f64_scalar_map,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return gemm_f64_fma_map;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return gemm_f64_avx512_map;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return gemm_f64_neon_map,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return gemm_f64_simd128_map,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return gemm_f64_avx512_map;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return gemm_f64_fma_map;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        gemm_f64_neon_map
    }
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            gemm_f64_simd128_map
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            gemm_f64_scalar_map
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        gemm_f64_scalar_map
    }
}

memoized_select!(
    MAP_F32,
    map_dispatched_f32,
    MapFn<f32>,
    select_map_f32,
    "The memoized `f32` map-epilogue dispatch entry (selection runs once).",
    "epilogue"
);
memoized_select!(
    MAP_F64,
    map_dispatched_f64,
    MapFn<f64>,
    select_map_f64,
    "The memoized `f64` map-epilogue dispatch entry (selection runs once).",
    "epilogue"
);

/// The sealed element-type bound for the user-defined map-epilogue public API
/// ([`crate::gemm_map`]): the real floats `f32`/`f64` only. A superset of [`GemmScalar`] (for
/// dispatch), sealed by the private `sealed::Sealed` supertrait so downstream crates cannot
/// widen the surface
///
/// The narrow floats (`f16`/`bf16`) are excluded on purpose: a `T`-domain closure applied after
/// the `f32` accumulate would double-round (narrow, then the closure re-widens/re-narrows),
/// breaking the `gemm_map == gemm()`-then-`f` bitwise contract the fused-in-`f32` convention
/// relies on. Complex and integer are likewise out of scope for v1 (no `apply` seam is wired for
/// a per-element closure). For bias/activation use [`crate::gemm_fused`] (it vectorizes);
/// `gemm_map` is the general per-element extension point
#[cfg(feature = "epilogue")]
pub trait MapScalar: GemmScalar + sealed::Sealed {
    /// Run the ISA-dispatched per-element map kernel for this type
    ///
    /// # Safety
    /// `task`'s pointers valid and `c` not aliasing `a`/`b`; the closure in `epi` is total
    #[doc(hidden)]
    unsafe fn dispatch_map(
        task: Task<Self>,
        epi: MapEpi<'_, Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );

    /// The degenerate map `C[i,j] <- f(beta*C[i,j], i, j)` in the user frame, run when the `A*B`
    /// term vanishes (`k == 0` or `alpha == 0`)
    ///
    /// # Safety
    /// `c` valid for the `m x n` region
    #[doc(hidden)]
    unsafe fn map_degenerate(t: &Task<Self>, epi: &MapEpi<'_, Self>);
}

#[cfg(feature = "epilogue")]
impl MapScalar for f32 {
    #[inline]
    unsafe fn dispatch_map(
        task: Task<f32>,
        epi: MapEpi<'_, f32>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (map_dispatched_f32())(task, epi, par, ws) }
    }
    #[inline]
    unsafe fn map_degenerate(t: &Task<f32>, epi: &MapEpi<'_, f32>) {
        unsafe { map_degenerate_float::<f32>(t, epi) }
    }
}
#[cfg(feature = "epilogue")]
impl MapScalar for f64 {
    #[inline]
    unsafe fn dispatch_map(
        task: Task<f64>,
        epi: MapEpi<'_, f64>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (map_dispatched_f64())(task, epi, par, ws) }
    }
    #[inline]
    unsafe fn map_degenerate(t: &Task<f64>, epi: &MapEpi<'_, f64>) {
        unsafe { map_degenerate_float::<f64>(t, epi) }
    }
}

/// Top-level map entry (called by the API layer): handle the degenerate case in the user frame
/// (before orientation, so `epi.swapped` is still `false`), then run the ISA-dispatched map kernel
///
/// # Safety
/// `task`'s pointers must be valid; `c` must not alias `a`/`b`, and the closure in `epi` is total
#[cfg(feature = "epilogue")]
pub(crate) unsafe fn execute_map<T: MapScalar>(
    task: Task<T>,
    epi: MapEpi<'_, T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if task.m == 0 || task.n == 0 {
            return;
        }
        // A*B vanishes (k == 0 or alpha == 0): C[i,j] <- f(beta*C[i,j], i, j), element-wise in
        // the user frame (the same degenerate contract as plain gemm, then the closure). This
        // runs before any orientation swap, so the coordinates are already user-frame
        if task.k == 0 || task.alpha == T::ZERO {
            T::map_degenerate(&task, &epi);
            return;
        }
        T::dispatch_map(task, epi, par, ws);
    }
}

/// The degenerate map `C[i,j] <- f(beta*C[i,j], i, j)` in the user frame, for the real floats
/// (`f32`/`f64`): all arithmetic is in `T`, then the closure. `epi.swapped` is `false` here (the
/// degenerate is handled before orientation), so `apply` passes `(i, j)` straight through
///
/// # Safety
/// `c` valid for the `m x n` region
#[cfg(feature = "epilogue")]
pub(super) unsafe fn map_degenerate_float<T: Float<Acc = T>>(t: &Task<T>, epi: &MapEpi<'_, T>) {
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                let p = t.c.offset(i as isize * t.rsc + j as isize * t.csc);
                let base = if t.beta == T::ZERO {
                    T::ZERO
                } else if t.beta == T::ONE {
                    *p
                } else {
                    t.beta * *p
                };
                *p = epi.apply(base, i, j);
            }
        }
    }
}
