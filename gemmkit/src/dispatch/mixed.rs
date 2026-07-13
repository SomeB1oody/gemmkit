//! `f16`/`bf16` mixed-precision dispatch (`Acc = f32`): narrow scale, driver
//! entries, per-ISA wrappers, descriptors, selection, and the `GemmScalar` impls.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::float::Dispatched;
use super::isa::{ForcedIsa, forced_isa};
use super::{GemmScalar, PackedConsume, Task, orient_transpose, small_mn_eligible};
use crate::driver;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::Bf16DotGemm;
use crate::kernel::KernelFamily;
use crate::kernel::MixedGemm;
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
use crate::special::{small_k, small_mn};
use crate::tuning;
use crate::workspace::Workspace;
use half::{bf16, f16};

/// `C <- beta·C` for a **narrow** type (`f16`/`bf16`): widen each element to `f32`,
/// scale, and round back. Matches the mixed kernel's epilogue precision.
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
/// of [`run_typed`] driving a narrow-in / `f32`-accumulate family: no gemv special path (the
/// general driver handles those shapes), the same orientation swap, and `alpha`/`beta`
/// **widened to the `f32` accumulator** before the driver call. `Fam` selects the general-
/// driver kernel — `MixedGemm<N>` for the widen path, `Bf16DotGemm` for the `vdpbf16ps` dot
/// path — while the `small_mn` / small-`k` reroutes deliberately stay on `MixedGemm<N>`
/// (both special paths bypass any dot kernel: a tiny output folds nothing and the dot pack's
/// `DEPTH_MULTIPLE` is pure loss there).
///
/// # Safety
/// As [`run_typed`].
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
        orient_transpose(&mut t);
        // Small `m,n` + long `k` + contiguous-along-`k` layout: the horizontal path, widening
        // `N → f32` on load and accumulating in `f32` (see [`run_typed`]'s float gate).
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
        // Skinny / low-depth shape through the widen microkernel (see [`run_typed`]).
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

/// Prepacked-RHS mixed-precision entry (mirror of [`run_packed_typed`] for a narrow-in /
/// `f32`-accumulate family `Fam`); no swap, `alpha`/`beta` widened to `f32`.
///
/// # Safety
/// As [`run_packed_typed`].
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

// ---- mixed-precision (f16 / bf16) entry points: same tiles as f32 (the
// accumulator is f32, so the register budget matches) ----

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
    // f32 accumulator → MR = 2*8 = 16, NR = 6 (the f32 FMA tile).
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
    // f32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile).
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
    // bf16 dot: f32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile). The
    // `Bf16DotGemm` family swaps in the `vdpbf16ps` pack + inner loop; the shared
    // `run_typed_mixed` routes small_mn / small_k through `MixedGemm<bf16>` as before.
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

#[cfg(feature = "half")]
const DISP_F16_SCALAR: Dispatched<f16> = Dispatched {
    run: gemm_f16_scalar,
    run_packed: gemm_f16_scalar_packed,
    run_fused: None,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(feature = "half")]
const DISP_BF16_SCALAR: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_scalar,
    run_packed: gemm_bf16_scalar_packed,
    run_fused: None,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_FMA: Dispatched<f16> = Dispatched {
    run: gemm_f16_fma,
    run_packed: gemm_f16_fma_packed,
    run_fused: None,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_FMA: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_fma,
    run_packed: gemm_bf16_fma_packed,
    run_fused: None,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_AVX512: Dispatched<f16> = Dispatched {
    run: gemm_f16_avx512,
    run_packed: gemm_f16_avx512_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512,
    run_packed: gemm_bf16_avx512_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512_DOT: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512_dot,
    run_packed: gemm_bf16_avx512_dot_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    // k-pair-interleaved pack → the prepack buffer rounds its depth up to 2.
    depth_multiple: 2,
};

#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_F16_NEON: Dispatched<f16> = Dispatched {
    run: gemm_f16_neon,
    run_packed: gemm_f16_neon_packed,
    run_fused: None,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_BF16_NEON: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_neon,
    run_packed: gemm_bf16_neon_packed,
    run_fused: None,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F16_SIMD128: Dispatched<f16> = Dispatched {
    run: gemm_f16_simd128,
    run_packed: gemm_f16_simd128_packed,
    run_fused: None,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_BF16_SIMD128: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_simd128,
    run_packed: gemm_bf16_simd128_packed,
    run_fused: None,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

/// `f16` ISA selection. The FMA path additionally needs **F16C**
/// (`vcvtph2ps`/`vcvtps2ph`) — checked here so an FMA selection on an F16C-less part
/// falls back rather than faulting. AVX-512 covers `f16` within `avx512f`.
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
/// no F16C is required; AVX-512 covers `bf16` within `avx512f`.
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
    unsafe fn pack_lhs_full(
        dst: *mut f16,
        a: *const f16,
        rsa: isize,
        csa: isize,
        m: usize,
        k: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_lhs_full::<MixedGemm<f16>>(dst, a, rsa, csa, m, k, kc, nc, nr) }
    }
    #[inline]
    unsafe fn dispatch(task: Task<f16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f16().run)(task, par, ws) }
    }
    #[inline]
    unsafe fn dispatch_packed(req: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f16().run_packed)(req, par, ws) }
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
            // depth multiple (> 1 only for the bf16 dot descriptor).
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if dispatched_bf16().depth_multiple > 1 {
                driver::pack_rhs_full::<Bf16DotGemm>(dst, b, rsb, csb, k, n, kc, nc, nr);
                return;
            }
            driver::pack_rhs_full::<MixedGemm<bf16>>(dst, b, rsb, csb, k, n, kc, nc, nr);
        }
    }
    #[inline]
    unsafe fn pack_lhs_full(
        dst: *mut bf16,
        a: *const bf16,
        rsa: isize,
        csa: isize,
        m: usize,
        k: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if dispatched_bf16().depth_multiple > 1 {
                driver::pack_lhs_full::<Bf16DotGemm>(dst, a, rsa, csa, m, k, kc, nc, nr);
                return;
            }
            driver::pack_lhs_full::<MixedGemm<bf16>>(dst, a, rsa, csa, m, k, kc, nc, nr);
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
