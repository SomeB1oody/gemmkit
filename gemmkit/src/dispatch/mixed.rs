//! `f16`/`bf16` mixed-precision dispatch (`Acc = f32`): narrow scale, driver
//! entries, per-ISA wrappers, descriptors, selection, and the `GemmScalar` impls

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::float::Dispatched;
#[cfg(feature = "epilogue")]
use super::float::FusedScalar;
use super::isa::{ForcedIsa, forced_isa};
use super::{GemmScalar, PackedConsume, Task, orient_transpose, small_mn_eligible};
use crate::driver;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::Bf16DotGemm;
use crate::kernel::KernelFamily;
use crate::kernel::MixedGemm;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{BiasSpec, Epilogue, FusedEpi};
use crate::parallel::Parallelism;
use crate::scalar::NarrowFloat;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::Avx512Bf16;
use crate::simd::KernelSimd;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
use crate::simd::ScalarTok;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::special::{gemv, small_k, small_mn};
use crate::tuning;
use crate::workspace::Workspace;
use half::{bf16, f16};

/// `C <- beta*C` for a **narrow** type (`f16`/`bf16`): widen each element to `f32`,
/// scale, and round back. Matches the mixed kernel's epilogue precision
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

/// Mixed-precision driver entry for a concrete `(narrow type, family, ISA, tile)`. Mirror
/// of [`run_typed`] driving a narrow-in / `f32`-accumulate family: the gemv reroute, the same
/// orientation swap, and `alpha`/`beta` **widened to the `f32` accumulator** before the driver
/// call. `Fam` selects the general-driver kernel (`MixedGemm<N>` for the widen path,
/// `Bf16DotGemm` for the `vdpbf16ps` dot path) while the gemv / `small_mn` / small-`k` reroutes
/// deliberately stay on the widen path (`MixedGemm<N>`'s `KernelSimd` seam): all 3 special
/// paths bypass any dot kernel (a tiny/degenerate output folds nothing and the dot pack's
/// `DEPTH_MULTIPLE` is pure loss there)
///
/// The gemv reroute is the mixed twin of the float gate in [`run_typed`]: an `m == 1` / `n == 1`
/// shape is a bandwidth-bound matrix*vector, which the general driver would pad up to a full
/// microtile (mostly zero FMAs). The widen [`gemv::run_mixed`] reads `N` in place, widens each
/// load to `f32`, accumulates in `f32`, and rounds to `N` once at the store. It routes in the
/// **user** frame before orientation normalization (as the float gate does)
///
/// # Safety
/// As [`run_typed`]
#[cfg(feature = "half")]
#[inline]
unsafe fn run_typed_mixed<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        // gemv shape (unless the dedicated path is disabled via tuning): the widen matrix*vector,
        // in the user frame before orientation normalization (mirrors the float gate in
        // [`run_typed`]). `alpha`/`beta` widened to the `f32` accumulator
        if (t.n == 1 || t.m == 1) && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold() {
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
        orient_transpose(&mut t);
        // Small `m,n` + long `k` + contiguous-along-`k` layout: the horizontal path, widening
        // `N -> f32` on load and accumulating in `f32` (see [`run_typed`]'s float gate)
        if small_mn_eligible(&t) {
            small_mn::run_mixed::<N, S>(
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
        // Skinny / low-depth shape through the widen microkernel (see [`run_typed`])
        if t.k <= tuning::small_k_threshold() {
            small_k::run::<MixedGemm<N>, S, MR_REG, NR>(
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
                par,
                ws,
            );
            return;
        }
        driver::run::<Fam, S, MR_REG, NR>(
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
            par,
            ws,
        );
    }
}

/// **Fused-epilogue** mixed-precision driver entry for a concrete `(narrow type, family, ISA,
/// tile)`: the mirror of [`run_typed_mixed`] with the fused [`FusedEpi`] threaded into each route.
/// The bias vector and `LeakyRelu` slope are the narrow type `N`, widened **exactly** to `f32`; the
/// epilogue applies in `f32` to the `f32` accumulator **before** the single round-to-nearest-even
/// narrowing to `N`. This is *more* precise than `gemm()` then a separate narrow map (which rounds
/// to `N`, widens back, and rounds again), so it is *not* bitwise-equal to `gemm`-then-map (unlike
/// the `f32`/`f64` every-shape contract). Reproducibility/determinism are unchanged (serial ==
/// parallel bitwise on these routes)
///
/// There is **no gemv route** here, unlike the plain [`run_typed_mixed`]: mixed fused gemv stays on
/// the general driver deliberately. The float fused gemv fuses by re-reading each stored output and
/// mapping it ([`gemv::run_typed_epi`]'s final in-place sweep), which is bit-exact only because the
/// float output *is* the accumulator (`OUT_IS_ACC = true`); for a narrow output the store has already
/// rounded, so re-reading and mapping would round twice. Applying the epilogue to the `f32`
/// accumulator *before* the single narrowing (the mixed discipline) would instead mean threading it
/// through each widen gemv strategy kernel's `f32 -> N` store (the vectorized axpy narrow especially),
/// a large diff for the rare fused-decode shape. The general driver already applies the epilogue in
/// `f32` before narrowing, so fused mixed gemv rides it correctly (just without the bandwidth win).
/// Like [`run_typed_mixed`], the `small_mn` / small-`k` reroutes stay on `MixedGemm<N>` (both bypass
/// any dot kernel). `alpha`/`beta` are widened to the `f32` accumulator; the orientation swap flips
/// the bias axis (same as the float `run_typed_fused`)
///
/// # Safety
/// As [`run_typed_mixed`], plus `epi`'s interior pointers valid for the (pre-swap) `m`/`n`
#[cfg(all(feature = "half", feature = "epilogue"))]
#[inline]
unsafe fn run_typed_mixed_fused<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<N>,
    mut epi: FusedEpi<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
    // The narrow blanket `Epilogue` impl supplies both (for the general-driver `Fam` and for the
    // `MixedGemm<N>` reroutes); naming them keeps the generic `Fam` well-formed
    FusedEpi<N>: Epilogue<Fam> + Epilogue<MixedGemm<N>>,
{
    unsafe {
        // No gemv route (see the doc). Orientation normalization flips the bias axis for the routes
        // below (they consume the oriented `epi`): a row-major-ish C computes `C^T = B^T*A^T`
        let swap = orient_transpose(&mut t);
        if swap {
            epi.bias = match epi.bias {
                BiasSpec::None => BiasSpec::None,
                BiasSpec::Row(p) => BiasSpec::Col(p),
                BiasSpec::Col(p) => BiasSpec::Row(p),
            };
        }

        // Small `m,n` + long `k` + contiguous-along-`k`: the horizontal path, widening `N -> f32` on
        // load and applying the epilogue to each `f32` cell before the single narrowing
        if small_mn_eligible(&t) {
            small_mn::run_mixed_epi::<N, S, FusedEpi<N>>(
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
                &epi,
            );
            return;
        }
        // Skinny / low-depth shape through the widen microkernel (deliberately `MixedGemm<N>` even
        // on the dot path, mirroring `run_typed_mixed`'s reroute rationale)
        if t.k <= tuning::small_k_threshold() {
            small_k::run_epi::<MixedGemm<N>, S, FusedEpi<N>, MR_REG, NR>(
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
        driver::run_epilogue::<Fam, S, FusedEpi<N>, MR_REG, NR>(
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

/// The degenerate fused epilogue for a **narrow** type (`f16`/`bf16`): `C[i,j] <- apply(beta*C[i,j],
/// i, j)` in the user frame, combined in `f32` and narrowed once by the epilogue (`apply` returns
/// `N`). The `f32` sibling of the real-float `fused_degenerate_float`
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
                // beta*C combined in `f32` (beta and C widened exactly), then the epilogue narrows once
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

/// Prepacked-RHS mixed-precision entry (mirror of [`run_packed_typed`] for a narrow-in /
/// `f32`-accumulate family `Fam`); no swap, `alpha`/`beta` widened to `f32`
///
/// # Safety
/// As [`run_packed_typed`]
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

/// **Fused-epilogue** prepacked-RHS mixed-precision entry (mirror of [`run_typed_mixed_packed_fused`]'s
/// float sibling `run_typed_packed_fused`, and of [`run_packed_typed_mixed`] with the epilogue
/// threaded): no swap, `alpha`/`beta` widened to `f32`, the [`FusedEpi`] applied in `f32` before the
/// single narrowing on store. No small-`m,n` / small-`k` reroute exists on the prepacked path, so
/// only `FusedEpi<N>: Epilogue<Fam>` is needed (not the extra `Epilogue<MixedGemm<N>>` the
/// general-driver mixed-fused entry names for its reroutes)
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

// mixed-precision (f16 / bf16) entry points: same tiles as f32 (the
// accumulator is f32, so the register budget matches)

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
    // f32 accumulator -> MR = 2*8 = 16, NR = 6 (the f32 FMA tile)
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
unsafe fn gemm_f16_avx512(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator -> MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile)
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_avx512_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_dot(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    // bf16 dot: f32 accumulator -> MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile). The
    // `Bf16DotGemm` family swaps in the `vdpbf16ps` pack + inner loop; the shared
    // `run_typed_mixed` routes small_mn / small_k through `MixedGemm<bf16>` as before
    unsafe { run_typed_mixed::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_dot_packed(
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

// fused-epilogue mixed entry points: same tiles as the plain wrappers (the epilogue is
// tile-local, so the f32-accumulator register budget is unchanged). Each is cfg-gated exactly like
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
unsafe fn gemm_f16_avx512_fused(
    t: Task<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_mixed_fused::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_fused::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(Avx512, t, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512_dot_fused(
    t: Task<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    // bf16 dot family for the general driver; small_mn / small_k reroute through `MixedGemm<bf16>`
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

// prepacked-RHS fused-epilogue mixed entry points: same tiles as the plain prepacked wrappers.
// Each is cfg-gated exactly like its plain prepacked sibling plus `epilogue`; the small_mn / small_k
// reroute question does not arise on the prepacked path, so the bf16 dot variant drives `Bf16DotGemm`
// throughout

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
unsafe fn gemm_f16_avx512_packed_fused(
    r: PackedConsume<f16>,
    epi: FusedEpi<f16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, r, epi, par, ws)
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512_packed_fused(
    r: PackedConsume<bf16>,
    epi: FusedEpi<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_typed_mixed_packed_fused::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(
            Avx512, r, epi, par, ws,
        )
    }
}
#[cfg(all(
    feature = "half",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_bf16_avx512_dot_packed_fused(
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
const DISP_F16_AVX512: Dispatched<f16> = Dispatched {
    run: gemm_f16_avx512,
    run_packed: gemm_f16_avx512_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_f16_avx512_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_f16_avx512_packed_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512,
    run_packed: gemm_bf16_avx512_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_avx512_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_avx512_packed_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512_DOT: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512_dot,
    run_packed: gemm_bf16_avx512_dot_packed,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_bf16_avx512_dot_fused,
    #[cfg(feature = "epilogue")]
    run_packed_fused: gemm_bf16_avx512_dot_packed_fused,
    mr: 32,
    nr: 12,
    // k-pair-interleaved pack -> the prepack buffer rounds its depth up to 2
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
/// (`vcvtph2ps`/`vcvtps2ph`): checked here so an FMA selection on an F16C-less part
/// falls back rather than faulting. AVX-512 covers `f16` within `avx512f`
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
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F16_AVX512;
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
            return DISP_F16_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") && x86_isa_detected!("f16c") {
            return DISP_F16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F16_NEON
    }
    // `simd128` on wasm32, else scalar
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

/// `bf16` ISA selection. The FMA path uses only AVX2 integer ops (shift / pack), so
/// no F16C is required; AVX-512 covers `bf16` within `avx512f`
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
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_BF16_AVX512;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512bf16, but this CPU/emulator does not report avx512f+bf16"
            );
            return DISP_BF16_AVX512_DOT;
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
        // bf16 dot kernel first - `vdpbf16ps` ~doubles bf16
        if x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512_DOT;
        }
        if x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_BF16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_BF16_NEON
    }
    // `simd128` on wasm32, else scalar
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
            // prepacked layout matches what the consuming call reads. Identified by the
            // depth multiple (> 1 only for the bf16 dot descriptor)
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

// FusedScalar (the fused-epilogue element bound) for the narrow floats
// `finite` widens exactly to `f32` then tests; `fused_degenerate` combines `beta*C` in `f32` and
// narrows once (the narrow degenerate sibling of the real-float path)

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
