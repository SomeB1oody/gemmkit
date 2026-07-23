//! `f16`/`bf16` mixed-precision dispatch, accumulating in `f32`: the narrow-output
//! degenerate scale, the deep-contraction f32-twin route, driver entries, per-ISA
//! wrappers, memoized descriptors, selection, and the `GemmScalar`/`FusedScalar` impls

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::float::Dispatched;
#[cfg(feature = "epilogue")]
use super::float::FusedScalar;
use super::isa::{ForcedIsa, forced_isa};
use super::{
    GemmScalar, PackedConsume, Task, orient_transpose, small_mn_eligible, small_mn_pack_eligible,
};
use crate::driver::{self, alpha_status, beta_status};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::Bf16DotGemm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::Bf16DotGemmF32;
use crate::kernel::KernelFamily;
use crate::kernel::MixedGemmF32;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::FusedEpi;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::kernel::{AlphaStatus, BetaStatus, MixedGemm};
use crate::parallel::Parallelism;
use crate::scalar::NarrowFloat;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::Avx512Bf16;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
use crate::simd::ScalarTok;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512F, Fma};
use crate::simd::{KernelSimd, SimdOps};
use crate::special::{gemv, small_k, small_mn};
use crate::tuning;
use crate::workspace::Workspace;
use half::{bf16, f16};

/// `C <- beta*C` for a **narrow** type (`f16`/`bf16`): widen each element to `f32`, scale,
/// and narrow back once. `beta == 0` overwrites with zero rather than multiplying, matching
/// the mixed kernel's own epilogue precision
#[cfg(feature = "half")]
unsafe fn scale_c_narrow<N: NarrowFloat>(
    beta: N,
    c: *mut N,
    m: usize,
    n: usize,
    rsc: isize,
    csc: isize,
) {
    unsafe {
        let b = beta.widen();
        for j in 0..n {
            for i in 0..m {
                let p = c.offset(i as isize * rsc + j as isize * csc);
                if beta == N::ZERO {
                    *p = N::ZERO;
                } else if beta != N::ONE {
                    *p = N::narrow(b * (*p).widen());
                }
            }
        }
    }
}

/// Maps a narrow-output family to its f32-output **deep-k twin** (`Out = f32 = Acc`, so
/// `OUT_IS_ACC` stays at its `true` default): the family [`run_deep_k_twin`] drives for a
/// large-`k` narrow GEMM, since it lets the driver multi-slice K instead of running one
/// L2-overflowing depth panel. Keyed off the family (not the ISA), so [`run_typed_mixed`] picks
/// the right twin from the same `Fam` its wrappers already pass: `MixedGemm<N> -> MixedGemmF32<N>`,
/// `Bf16DotGemm -> Bf16DotGemmF32`
#[cfg(feature = "half")]
trait DeepKTwin: KernelFamily {
    /// The f32-output twin family: same `Lhs`/`Rhs`/`Acc`, `Out = f32`
    type Twin: KernelFamily<Lhs = Self::Lhs, Rhs = Self::Rhs, Acc = Self::Acc, Out = f32>;
}
#[cfg(feature = "half")]
impl<N: NarrowFloat> DeepKTwin for MixedGemm<N> {
    type Twin = MixedGemmF32<N>;
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
impl DeepKTwin for Bf16DotGemm {
    type Twin = Bf16DotGemmF32;
}

/// Deep-contraction route: re-block a large-`k` narrow GEMM through its f32-output twin `Tw`, then
/// narrow once. The single-panel narrow family (`OUT_IS_ACC = false`, `kc = k`) rounds the output
/// only once, but at large `k` its RHS micropanel outgrows L2 and every microtile call streams it
/// from L3/DRAM. The twin instead blocks `K` at the cache-model `kc` (panels stay L2-resident),
/// accumulating into an `m x n` **f32 scratch** with `alpha = 1`, `beta = 0`; a single vectorized
/// sweep then applies the real `alpha`/`beta` and narrows to `N`
///
/// **Bit-identical to the single panel** for the common `beta in {0, 1}`: the twin seeds each
/// slice's accumulators from the f32 scratch (see `kernel::mixed::twin_seed`), so every output's
/// ascending-`k` FMA/dot chain is the single-panel one merely split at slice boundaries (an f32
/// store/reload is exact); the dot twin's `kc` is rounded to `DEPTH_MULTIPLE` so a pair never
/// straddles a boundary. The narrowing sweep replicates `mixed_epilogue`'s arithmetic (fold
/// `alpha` with `mul`, combine `beta` with `add`, narrow with the same `store_out`), so the result
/// matches the single-panel path byte-for-byte when `beta in {0, 1}`; a general `beta` is accurate
/// only to tolerance (the single panel fuses `beta*C + AB` on a full tile but not on an edge tile,
/// so no one sweep formula matches both cases). Serial and parallel stay bit-identical throughout
/// (the twin driver's blocking is thread-count independent, and the sweep is elementwise)
///
/// The f32 scratch comes from a **dedicated `Workspace`**: deep-k is a large-`k` regime, so the
/// single `m*n` f32 allocation is negligible next to the contraction, and this keeps the hot
/// packing buffer (`ws`, threaded into the twin driver) pooled and reused rather than displaced.
/// `Workspace::regions` carries the same fail-closed overflow guard as the driver's own sizing
///
/// # Safety
/// As [`run_typed_mixed`]; `t` is already orientation-normalized (`c` column-major-ish)
#[cfg(feature = "half")]
#[inline]
unsafe fn run_deep_k_twin<N, Tw, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: &Task<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Tw: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = f32>,
    S: KernelSimd<N, N, f32, N> + KernelSimd<N, N, f32, f32>,
{
    unsafe {
        let (m, n, k) = (t.m, t.n, t.k);
        // f32 scratch, contiguous column-major m x n (rsc = 1, csc = m). A dedicated Workspace
        // keeps the pooled `ws` free for the twin driver's own packing; `regions` fail-closes on
        // the element -> byte overflow the same way the driver's own sizing does
        let mut scratch_ws = Workspace::new();
        let scratch = scratch_ws.regions::<f32>(m.saturating_mul(n), 1, 0).a_base;
        // Pure sum(A*B) into the scratch (alpha = 1, beta = 0). Out = f32 = Acc, so the driver
        // multi-slices at the cache-model kc; the twin seeds each slice from the scratch, so the
        // accumulation is the single panel's sum split at slice boundaries. The first (beta = 0)
        // slice writes every scratch element before any later slice reads it (the pc-loop
        // fork-joins per slice), so the scratch is never read uninitialized
        driver::run::<Tw, S, MR_REG, NR>(
            simd, m, k, n, 1.0f32, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, 0.0f32, scratch, 1,
            m as isize, par, ws,
        );

        // Narrowing sweep: c = narrow(alpha*scratch + beta*widen(c_old)), replicating
        // mixed_epilogue op-for-op so beta in {0, 1} reproduces the single panel bitwise
        let alpha = t.alpha.widen();
        let beta = t.beta.widen();
        let ash = alpha_status(alpha);
        let bst = beta_status(beta);
        let (c, rsc, csc) = (t.c, t.rsc, t.csc);
        simd.vectorize(|| {
            let lanes = <S as SimdOps<f32>>::LANES;
            let av = simd.splat(alpha);
            let bv = simd.splat(beta);
            for j in 0..n {
                let sc = scratch.add(j * m); // f32 scratch column, contiguous
                let cc = c.offset(j as isize * csc); // narrow-C column
                if rsc == 1 {
                    let mut i = 0;
                    while i + lanes <= m {
                        let mut r = simd.loadu(sc.add(i));
                        if ash == AlphaStatus::Other {
                            r = simd.mul(r, av);
                        }
                        r = match bst {
                            BetaStatus::Zero => r,
                            BetaStatus::One => {
                                let cv = <S as KernelSimd<N, N, f32, N>>::load_out(simd, cc.add(i));
                                simd.add(cv, r)
                            }
                            BetaStatus::Other => {
                                let cv = <S as KernelSimd<N, N, f32, N>>::load_out(simd, cc.add(i));
                                simd.mul_add(cv, bv, r)
                            }
                        };
                        <S as KernelSimd<N, N, f32, N>>::store_out(simd, cc.add(i), r);
                        i += lanes;
                    }
                    while i < m {
                        let mut r = *sc.add(i);
                        if ash == AlphaStatus::Other {
                            r *= alpha;
                        }
                        r = match bst {
                            BetaStatus::Zero => r,
                            BetaStatus::One => (*cc.add(i)).widen() + r,
                            BetaStatus::Other => beta * (*cc.add(i)).widen() + r,
                        };
                        *cc.add(i) = N::narrow(r);
                        i += 1;
                    }
                } else {
                    for i in 0..m {
                        let cp = cc.offset(i as isize * rsc);
                        let mut r = *sc.add(i);
                        if ash == AlphaStatus::Other {
                            r *= alpha;
                        }
                        r = match bst {
                            BetaStatus::Zero => r,
                            BetaStatus::One => (*cp).widen() + r,
                            BetaStatus::Other => beta * (*cp).widen() + r,
                        };
                        *cp = N::narrow(r);
                    }
                }
            }
        });
    }
}

/// The single route-priority ladder for the mixed-precision families (narrow-in / `f32`-accumulate):
/// route one concrete `(narrow type, family, ISA, tile)` GEMM through gemv, the small-`m,n`
/// horizontal kernel, the small-`k` panel kernel, the deep-`k` f32-twin reblocking, or the general
/// driver, with the fused [`Epilogue`] `E` threaded into every route and `alpha`/`beta` **widened to
/// the `f32` accumulator** before the driver call. [`run_typed_mixed`] and [`run_typed_mixed_fused`]
/// are thin epilogue-choice wrappers over this one body, mirroring the `Identity`-wrapper pattern the
/// special layer and driver already use. `Fam` selects the general-driver kernel (`MixedGemm<N>` for
/// the widen path, `Bf16DotGemm` for the `vdpbf16ps` dot path), while the `small_mn` / small-`k`
/// reroutes deliberately stay on the widen path (`MixedGemm<N>`'s `KernelSimd` seam): both bypass any
/// dot kernel, since a tiny output folds nothing and the dot pack's `DEPTH_MULTIPLE` would be pure
/// loss there
///
/// 2 routes fire **only on the plain (`E::IS_IDENTITY`) path** and const-fold away for a real
/// epilogue - the gemv reroute and the deep-`k` twin - each with its own rationale at its branch. The
/// remaining routes take the epilogue: a fused mixed call applies `E` to the `f32` accumulator
/// **before** the single round-to-nearest-even narrowing to `N`, which is *more* precise than
/// `gemm()` then a separate narrow map (that would round to `N`, widen back, and round again), so a
/// fused mixed result is *not* bitwise-equal to `gemm`-then-map, unlike the `f32`/`f64` every-shape
/// contract. Reproducibility and determinism are unaffected (serial still equals parallel bitwise on
/// these routes). The orientation swap flips the bias axis through `on_orient_swap` (a no-op for
/// `Identity`)
///
/// # Safety
/// As `run_typed`, plus `epi`'s interior pointers valid for the (pre-swap) problem's `m`/`n`
#[cfg(feature = "half")]
#[inline]
unsafe fn run_typed_mixed_epi<N, Fam, S, const MR_REG: usize, const NR: usize, E>(
    simd: S,
    mut t: Task<N>,
    mut epi: E,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N> + DeepKTwin,
    S: KernelSimd<N, N, f32, N> + KernelSimd<N, N, f32, f32>,
    E: Epilogue<Fam> + Epilogue<MixedGemm<N>>,
{
    unsafe {
        // gemv shape (unless the dedicated path is disabled via tuning): the widen matrix*vector, in
        // the user frame before orientation normalization (mirrors run_typed's float gate),
        // alpha/beta widened to the f32 accumulator. An m == 1 / n == 1 shape is a bandwidth-bound
        // matrix*vector the general driver would pad up to a full microtile (mostly zero FMAs);
        // gemv::run_mixed reads N in place, widens each load to f32, accumulates in f32, and rounds
        // to N once at the store
        //
        // Plain (Identity) path only: mixed fused gemv stays on the general driver deliberately. The
        // float fused gemv fuses by re-reading each stored output and mapping it, bit-exact only
        // because the float output IS the accumulator (OUT_IS_ACC = true); for a narrow output the
        // store has already rounded, so re-reading and mapping would double-round. Applying the
        // epilogue to the f32 accumulator before the single narrowing (the mixed discipline) would
        // instead mean threading it through each widen gemv strategy kernel's f32 -> N store, a large
        // diff for the rare fused-decode shape. The general driver already applies the epilogue in
        // f32 before narrowing, so a fused mixed gemv rides it correctly, just without the bandwidth
        // win
        // E is Epilogue for 2 families here (Fam and MixedGemm<N>), so IS_IDENTITY needs a
        // fully-qualified path; both impls agree (Identity sets it on the blanket, FusedEpi never
        // does), so either family answers the same
        if <E as Epilogue<Fam>>::IS_IDENTITY
            && (t.n == 1 || t.m == 1)
            && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold()
        {
            gemv::run_mixed::<N, S>(
                simd,
                t.m,
                t.k,
                t.n,
                par,
                t.alpha.widen(),
                t.a,
                t.rsa,
                t.csa,
                t.b,
                t.rsb,
                t.csb,
                t.beta.widen(),
                t.c,
                t.rsc,
                t.csc,
            );
            return;
        }
        // Orientation normalization transposes the engine frame for the routes below (they all
        // consume the oriented epi): a row-major-ish C computes C^T = B^T*A^T (swapping m<->n), so
        // on_orient_swap flips a FusedEpi bias axis, same as the float run_typed_epi (a no-op for
        // Identity)
        if orient_transpose(&mut t) {
            // Fully qualified for the same reason as IS_IDENTITY above: E is Epilogue for 2
            // families, and both impls' on_orient_swap agree (FusedEpi flips the bias axis either
            // way, Identity is the no-op default)
            <E as Epilogue<Fam>>::on_orient_swap(&mut epi);
        }
        // Small m,n + long k: the horizontal path, widening N -> f32 on load and applying the
        // epilogue to each f32 cell before the single narrowing (mirrors run_typed's float gate;
        // deliberately MixedGemm<N> even on the dot path). A contiguous-along-k layout reads in
        // place; a strided operand is packed into k-contiguous scratch first, as narrow N, widened
        // on load
        if small_mn_eligible(&t) || small_mn_pack_eligible(&t) {
            small_mn::run_mixed_epi::<N, S, E>(
                simd,
                t.m,
                t.k,
                t.n,
                par,
                ws,
                t.alpha.widen(),
                t.a,
                t.rsa,
                t.csa,
                t.b,
                t.rsb,
                t.csb,
                t.beta.widen(),
                t.c,
                t.rsc,
                t.csc,
                &epi,
            );
            return;
        }
        // Skinny / low-depth shape through the widen microkernel (deliberately MixedGemm<N> even on
        // the dot path, mirroring the small_mn reroute rationale), the epilogue applied to each f32
        // cell before the single narrowing
        if t.k <= tuning::small_k_threshold() {
            small_k::run_epi::<MixedGemm<N>, S, E, MR_REG, NR>(
                simd,
                t.m,
                t.k,
                t.n,
                t.alpha.widen(),
                t.a,
                t.rsa,
                t.csa,
                t.b,
                t.rsb,
                t.csb,
                t.beta.widen(),
                t.c,
                t.rsc,
                t.csc,
                &epi,
                par,
                ws,
            );
            return;
        }
        // Deep-contraction reblocking (see run_deep_k_twin): a narrow family runs kc = k (one depth
        // panel) so the output rounds once, but at large k its RHS micropanel (nr * k * sizeof(N))
        // outgrows L2 and every microtile call streams it from L3/DRAM. Once that micropanel exceeds
        // the engage gate, run the f32-output twin (multi-slice, panels L2-resident) into an f32
        // scratch and narrow once. checked_mul so an overflowing micropanel size (a broadcast operand
        // can pass validation with a logically huge k) does NOT engage: it falls through to the
        // single panel instead, whose pack sizing then fails closed with the "too large" guard,
        // rather than having the twin multi-slice that k forever
        //
        // Plain (Identity) path only: the twin narrows through a dedicated f32 sweep, not the
        // epilogue, so a fused shape stays on the single-panel general driver below, which applies
        // the epilogue in f32 before the single narrowing (IS_IDENTITY fully qualified as above)
        if <E as Epilogue<Fam>>::IS_IDENTITY {
            let engage_deep_k = NR
                .checked_mul(t.k)
                .and_then(|x| x.checked_mul(core::mem::size_of::<N>()))
                .is_some_and(|bytes| bytes > crate::cache::deep_k_engage_bytes());
            if engage_deep_k {
                run_deep_k_twin::<N, Fam::Twin, S, MR_REG, NR>(simd, &t, par, ws);
                return;
            }
        }
        driver::run_epilogue::<Fam, S, E, MR_REG, NR>(
            simd,
            t.m,
            t.k,
            t.n,
            t.alpha.widen(),
            t.a,
            t.rsa,
            t.csa,
            t.b,
            t.rsb,
            t.csb,
            t.beta.widen(),
            t.c,
            t.rsc,
            t.csc,
            &epi,
            par,
            ws,
        );
    }
}

/// Mixed-precision driver entry for a concrete `(narrow type, family, ISA, tile)`: the plain
/// (`E = Identity`) choice of [`run_typed_mixed_epi`], so every epilogue hook const-folds away and
/// the routes reduce to the raw narrowing store. `Fam` selects the general-driver kernel
/// (`MixedGemm<N>` widen path, `Bf16DotGemm` dot path); the special-path reroutes stay on the widen
/// path (see [`run_typed_mixed_epi`])
///
/// # Safety
/// As `run_typed`
#[cfg(feature = "half")]
#[inline]
unsafe fn run_typed_mixed<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: Task<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N> + DeepKTwin,
    S: KernelSimd<N, N, f32, N> + KernelSimd<N, N, f32, f32>,
{
    unsafe { run_typed_mixed_epi::<N, Fam, S, MR_REG, NR, Identity>(simd, t, Identity, par, ws) }
}

/// **Fused-epilogue** mixed-precision driver entry for a concrete `(narrow type, family, ISA,
/// tile)`: the [`FusedEpi`] choice of [`run_typed_mixed_epi`]. The bias vector and `LeakyRelu`
/// slope are the narrow type `N`, widened **exactly** to `f32`; the epilogue applies in `f32` to
/// the `f32` accumulator **before** the single round-to-nearest-even narrowing to `N`. This is
/// *more* precise than `gemm()` then a separate narrow map (which rounds to `N`, widens back, and
/// rounds again), so it is *not* bitwise-equal to `gemm`-then-map, unlike the `f32`/`f64`
/// every-shape contract. Reproducibility and determinism are unaffected (serial still equals
/// parallel bitwise on these routes)
///
/// There is **no gemv route** and **no deep-`k` twin** here, unlike the plain [`run_typed_mixed`]:
/// both are gated to the `E::IS_IDENTITY` path inside [`run_typed_mixed_epi`] (the rationale for
/// each lives at its branch there). The `small_mn` / small-`k` reroutes stay on `MixedGemm<N>`
/// (both bypass any dot kernel), and the orientation swap flips the bias axis, same as the float
/// `run_typed_fused`
///
/// # Safety
/// As [`run_typed_mixed`], plus `epi`'s interior pointers valid for the (pre-swap) `m`/`n`
#[cfg(all(feature = "half", feature = "epilogue"))]
#[inline]
unsafe fn run_typed_mixed_fused<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    t: Task<N>,
    epi: FusedEpi<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N> + DeepKTwin,
    S: KernelSimd<N, N, f32, N> + KernelSimd<N, N, f32, f32>,
    // The narrow blanket Epilogue impl supplies both (for the general-driver Fam and for the
    // MixedGemm<N> reroutes); naming them keeps the generic Fam well-formed
    FusedEpi<N>: Epilogue<Fam> + Epilogue<MixedGemm<N>>,
{
    unsafe { run_typed_mixed_epi::<N, Fam, S, MR_REG, NR, FusedEpi<N>>(simd, t, epi, par, ws) }
}

/// The degenerate fused epilogue for a **narrow** type (`f16`/`bf16`): `C[i,j] <- apply(beta*C[i,j],
/// i, j)` in the user frame, `beta*C` combined in `f32` and narrowed once by the epilogue (`apply`
/// returns `N`). The narrow sibling of the real-float `fused_degenerate_float`
///
/// # Safety
/// `c` valid for the `m x n` region; `epi`'s bias valid for the problem's `m`/`n`
#[cfg(all(feature = "half", feature = "epilogue"))]
unsafe fn fused_degenerate_mixed<N>(t: &Task<N>, epi: &FusedEpi<N>)
where
    N: NarrowFloat,
    FusedEpi<N>: Epilogue<MixedGemm<N>>,
{
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                let p = t.c.offset(i as isize * t.rsc + j as isize * t.csc);
                // beta*C combined in f32 (beta and C widened exactly), then the epilogue narrows once
                let base: f32 = if t.beta == N::ZERO {
                    0.0
                } else if t.beta == N::ONE {
                    (*p).widen()
                } else {
                    t.beta.widen() * (*p).widen()
                };
                *p = <FusedEpi<N> as Epilogue<MixedGemm<N>>>::apply(epi, base, i, j);
            }
        }
    }
}

/// Prepacked-RHS mixed-precision entry (mirror of `run_packed_typed` for a narrow-in /
/// `f32`-accumulate family `Fam`): no orientation swap, `alpha`/`beta` widened to `f32`
///
/// # Safety
/// As `run_packed_typed`
#[cfg(feature = "half")]
#[inline]
unsafe fn run_packed_typed_mixed<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<Fam, S, MR_REG, NR>(
            simd,
            req.m,
            req.k,
            req.n,
            req.alpha.widen(),
            req.a,
            req.rsa,
            req.csa,
            req.packed,
            req.kc,
            req.nc,
            req.beta.widen(),
            req.c,
            req.rsc,
            req.csc,
            par,
            ws,
        );
    }
}

/// **Fused-epilogue** prepacked-RHS mixed-precision entry: the mirror of the float sibling
/// `run_typed_packed_fused`, and of [`run_packed_typed_mixed`] with the epilogue threaded in. No
/// orientation swap, `alpha`/`beta` widened to `f32`, the [`FusedEpi`] applied in `f32` before the
/// single narrowing on store. No small-`m,n` / small-`k` reroute exists on the prepacked path, so
/// only `FusedEpi<N>: Epilogue<Fam>` is needed here, not the extra `Epilogue<MixedGemm<N>>` the
/// general-driver mixed-fused entry names for its reroutes
///
/// # Safety
/// As [`run_packed_typed_mixed`], plus `epi`'s interior pointers valid for the problem's `m`/`n`
#[cfg(all(feature = "half", feature = "epilogue"))]
#[inline]
unsafe fn run_typed_mixed_packed_fused<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<N>,
    epi: FusedEpi<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
    FusedEpi<N>: Epilogue<Fam>,
{
    unsafe {
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs_epilogue::<Fam, S, FusedEpi<N>, MR_REG, NR>(
            simd,
            req.m,
            req.k,
            req.n,
            req.alpha.widen(),
            req.a,
            req.rsa,
            req.csa,
            req.packed,
            req.kc,
            req.nc,
            req.beta.widen(),
            req.c,
            req.rsc,
            req.csc,
            &epi,
            par,
            ws,
        );
    }
}

// mixed-precision (f16 / bf16) entry points: same tiles as f32, since the accumulator
// is f32 too, so the register budget matches

#[cfg(feature = "half")]
unsafe fn gemm_f16_scalar(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_bf16_scalar(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_f16_scalar_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_bf16_scalar_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        run_packed_typed_mixed::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(ScalarTok, r, par, ws)
    }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_fma(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator: MR = 2*8 = 16, NR = 6, the f32 FMA tile
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_fma(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_fma_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_fma_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_avx512f(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator: MR = 2*16 = 32, NR = 12, the f32 AVX-512F tile
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Avx512F, 2, 12>(Avx512F, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512f(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Avx512F, 2, 12>(Avx512F, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_avx512f_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Avx512F, 2, 12>(Avx512F, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512f_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Avx512F, 2, 12>(Avx512F, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512bf16(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    // bf16 dot: f32 accumulator, same tile as plain AVX-512F (MR = 2*16 = 32, NR = 12). The
    // Bf16DotGemm family swaps in the vdpbf16ps pack and inner loop; the shared
    // run_typed_mixed still routes small_mn / small_k through MixedGemm<bf16>
    unsafe { run_typed_mixed::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512bf16_packed(
    r: PackedConsume<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_packed_typed_mixed::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, r, par, ws)
    }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f16_simd128(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_bf16_simd128(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f16_simd128_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_bf16_simd128_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(Simd128, r, par, ws) }
}

// fused-epilogue mixed entry points: same tiles as the plain wrappers, since the epilogue is
// tile-local so the f32-accumulator register budget is unchanged. Each is cfg-gated exactly like
// its plain sibling

#[cfg(all(feature = "half", feature = "epilogue"))]
unsafe fn gemm_f16_scalar_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws)
    }
}
#[cfg(all(feature = "half", feature = "epilogue"))]
unsafe fn gemm_bf16_scalar_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_f16_fma_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_fma_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_f16_avx512f_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<f16, MixedGemm<f16>, Avx512F, 2, 12>(Avx512F, t, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512f_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<bf16, MixedGemm<bf16>, Avx512F, 2, 12>(Avx512F, t, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512bf16_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    // bf16 dot family for the general driver; small_mn / small_k reroute through MixedGemm<bf16>
    unsafe {
        run_typed_mixed_fused::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, t, epi, par, ws)
    }
}
#[cfg(all(feature = "half", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(feature = "half", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f16_simd128_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_bf16_simd128_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(Simd128, t, epi, par, ws)
    }
}

// prepacked-RHS fused-epilogue mixed entry points: same tiles as the plain prepacked wrappers
// Each is cfg-gated exactly like its plain prepacked sibling plus epilogue; the small_mn /
// small_k reroute question does not arise on the prepacked path, so the bf16 dot variant drives
// Bf16DotGemm throughout

#[cfg(all(feature = "half", feature = "epilogue"))]
unsafe fn gemm_f16_scalar_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(
            ScalarTok, r, epi, par, ws,
        )
    }
}
#[cfg(all(feature = "half", feature = "epilogue"))]
unsafe fn gemm_bf16_scalar_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(
            ScalarTok, r, epi, par, ws,
        )
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_f16_fma_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, r, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_fma_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, r, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_f16_avx512f_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, Avx512F, 2, 12>(
            Avx512F, r, epi, par, ws,
        )
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512f_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, Avx512F, 2, 12>(
            Avx512F, r, epi, par, ws,
        )
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512bf16_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(
            Avx512Bf16, r, epi, par, ws,
        )
    }
}
#[cfg(all(feature = "half", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, r, epi, par, ws)
    }
}
#[cfg(all(feature = "half", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, r, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_f16_simd128_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, r, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_bf16_simd128_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(
            Simd128, r, epi, par, ws,
        )
    }
}

#[cfg(feature = "half")]
const DISP_F16_SCALAR: Dispatched<f16> = Dispatched {
    run: gemm_f16_scalar,
    run_packed: gemm_f16_scalar_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_scalar_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_scalar_packed_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(feature = "half")]
const DISP_BF16_SCALAR: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_scalar,
    run_packed: gemm_bf16_scalar_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_scalar_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_scalar_packed_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_FMA: Dispatched<f16> = Dispatched {
    run: gemm_f16_fma,
    run_packed: gemm_f16_fma_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_fma_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_fma_packed_fused,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_FMA: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_fma,
    run_packed: gemm_bf16_fma_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_fma_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_fma_packed_fused,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_AVX512F: Dispatched<f16> = Dispatched {
    run: gemm_f16_avx512f,
    run_packed: gemm_f16_avx512f_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_avx512f_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_avx512f_packed_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512F: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512f,
    run_packed: gemm_bf16_avx512f_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_avx512f_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_avx512f_packed_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512BF16: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512bf16,
    run_packed: gemm_bf16_avx512bf16_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_avx512bf16_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_avx512bf16_packed_fused,
    mr: 32,
    nr: 12,
    // k-pair-interleaved pack: the prepack buffer rounds its depth up to a multiple of 2
    depth_multiple: 2,
};

#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_F16_NEON: Dispatched<f16> = Dispatched {
    run: gemm_f16_neon,
    run_packed: gemm_f16_neon_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_neon_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_neon_packed_fused,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_BF16_NEON: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_neon,
    run_packed: gemm_bf16_neon_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_neon_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_neon_packed_fused,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F16_SIMD128: Dispatched<f16> = Dispatched {
    run: gemm_f16_simd128,
    run_packed: gemm_f16_simd128_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_simd128_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_simd128_packed_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_BF16_SIMD128: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_simd128,
    run_packed: gemm_bf16_simd128_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_simd128_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_simd128_packed_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

/// `f16` ISA selection. The FMA path additionally needs **F16C**
/// (`vcvtph2ps`/`vcvtps2ph`), checked here so an FMA selection on an F16C-less part falls
/// back instead of faulting. AVX-512F covers `f16` conversion within `avx512f` itself
#[cfg(feature = "half")]
fn select_f16() -> Dispatched<f16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma") && x86_isa_detected!("f16c"),
                "GEMMKIT_REQUIRE_ISA=fma for f16, but this CPU does not report avx2+fma+f16c"
            );
            return DISP_F16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
            );
            return DISP_F16_AVX512F;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_F16_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return DISP_F16_AVX512F;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") && x86_isa_detected!("f16c") {
            return DISP_F16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F16_NEON
    }
    // simd128 on wasm32 (when compiled in), scalar everywhere else
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_F16_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_F16_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_F16_SCALAR
    }
}

/// `bf16` ISA selection: auto-selects the `vdpbf16ps` dot kernel first (needs `avx512bf16`),
/// then plain AVX-512F (`avx512f`), then FMA. The FMA path uses only AVX2 integer ops
/// (shift / pack) to widen `bf16 -> f32`, so it needs no F16C, unlike the `f16` ladder
#[cfg(feature = "half")]
fn select_bf16() -> Dispatched<bf16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_BF16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_BF16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
            );
            return DISP_BF16_AVX512F;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512bf16, but this CPU/emulator does not report avx512f+bf16"
            );
            return DISP_BF16_AVX512BF16;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_BF16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_BF16_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // bf16 dot kernel first: vdpbf16ps is a structural win over the plain AVX-512F widen path
        if x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512BF16;
        }
        if x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512F;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_BF16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_BF16_NEON
    }
    // simd128 on wasm32 (when compiled in), scalar everywhere else
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_BF16_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_BF16_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_BF16_SCALAR
    }
}

memoized_select!(
    GEMM_F16,
    dispatched_f16,
    Dispatched<f16>,
    select_f16,
    "The memoized dispatch descriptor for `f16` (selection runs once).",
    "half"
);
memoized_select!(
    GEMM_BF16,
    dispatched_bf16,
    Dispatched<bf16>,
    select_bf16,
    "The memoized dispatch descriptor for `bf16` (selection runs once).",
    "half"
);

#[cfg(feature = "half")]
impl GemmScalar for f16 {
    const OUT_IS_ACC: bool = false;
    #[inline]
    unsafe fn scale_c(beta: f16, c: *mut f16, m: usize, n: usize, rsc: isize, csc: isize) {
        unsafe { scale_c_narrow(beta, c, m, n, rsc, csc) }
    }
    #[inline]
    unsafe fn pack_rhs_full(
        dst: *mut f16,
        b: *const f16,
        rsb: isize,
        csb: isize,
        k: usize,
        n: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_rhs_full::<MixedGemm<f16>>(dst, b, rsb, csb, k, n, kc, nc, nr) }
    }
    #[inline]
    unsafe fn dispatch(task: Task<f16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f16().run)(task, par, ws) }
    }
    #[inline]
    unsafe fn dispatch_packed(req: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f16().run_packed)(req, par, ws) }
    }
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_fused(
        t: Task<f16>,
        epi: FusedEpi<f16>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (dispatched_f16().run_fused)(t, epi, par, ws) }
    }
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_packed_fused(
        req: PackedConsume<f16>,
        epi: FusedEpi<f16>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (dispatched_f16().run_packed_fused)(req, epi, par, ws) }
    }
    #[inline]
    fn rhs_tile() -> (usize, usize) {
        let d = dispatched_f16();
        (d.mr, d.nr)
    }
}

#[cfg(feature = "half")]
impl GemmScalar for bf16 {
    const OUT_IS_ACC: bool = false;
    #[inline]
    unsafe fn scale_c(beta: bf16, c: *mut bf16, m: usize, n: usize, rsc: isize, csc: isize) {
        unsafe { scale_c_narrow(beta, c, m, n, rsc, csc) }
    }
    #[inline]
    unsafe fn pack_rhs_full(
        dst: *mut bf16,
        b: *const bf16,
        rsb: isize,
        csb: isize,
        k: usize,
        n: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe {
            // The dot kernel packs k-pair-interleaved; pack through *its* family so the
            // prepacked layout matches what the consuming call reads. Detected via the
            // depth multiple, which is > 1 only for the bf16 dot descriptor
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if dispatched_bf16().depth_multiple > 1 {
                driver::pack_rhs_full::<Bf16DotGemm>(dst, b, rsb, csb, k, n, kc, nc, nr);
                return;
            }
            driver::pack_rhs_full::<MixedGemm<bf16>>(dst, b, rsb, csb, k, n, kc, nc, nr);
        }
    }
    #[inline]
    unsafe fn dispatch(task: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_bf16().run)(task, par, ws) }
    }
    #[inline]
    unsafe fn dispatch_packed(req: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_bf16().run_packed)(req, par, ws) }
    }
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_fused(
        t: Task<bf16>,
        epi: FusedEpi<bf16>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (dispatched_bf16().run_fused)(t, epi, par, ws) }
    }
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_packed_fused(
        req: PackedConsume<bf16>,
        epi: FusedEpi<bf16>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        unsafe { (dispatched_bf16().run_packed_fused)(req, epi, par, ws) }
    }
    #[inline]
    fn rhs_tile() -> (usize, usize) {
        let d = dispatched_bf16();
        (d.mr, d.nr)
    }
    #[inline]
    fn rhs_depth_multiple() -> usize {
        dispatched_bf16().depth_multiple
    }
}

// FusedScalar (the fused-epilogue element bound) for the narrow floats: finite widens exactly
// to f32 then tests; fused_degenerate combines beta*C in f32 and narrows once, the narrow
// sibling of the real-float degenerate path

#[cfg(all(feature = "half", feature = "epilogue"))]
impl FusedScalar for f16 {
    #[inline]
    fn finite(self) -> bool {
        self.widen().is_finite()
    }
    #[inline]
    unsafe fn fused_degenerate(t: &Task<f16>, epi: &FusedEpi<f16>) {
        unsafe { fused_degenerate_mixed::<f16>(t, epi) }
    }
}

#[cfg(all(feature = "half", feature = "epilogue"))]
impl FusedScalar for bf16 {
    #[inline]
    fn finite(self) -> bool {
        self.widen().is_finite()
    }
    #[inline]
    unsafe fn fused_degenerate(t: &Task<bf16>, epi: &FusedEpi<bf16>) {
        unsafe { fused_degenerate_mixed::<bf16>(t, epi) }
    }
}
