//! `c32`/`c64` complex GEMM dispatch (optional conjA/conjB): the runtime->compile-time
//! conj branch, per-ISA wrappers, descriptors, selection, and the `ComplexScalar` impls.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::{Task, orient_transpose, scale_c_float};
use crate::driver;
use crate::parallel::Parallelism;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::simd::{KernelSimd, ScalarTok};
use crate::workspace::Workspace;

/// `c32` / `c64` element-type aliases (the complex-GEMM dispatch types).
#[cfg(feature = "complex")]
type C32 = num_complex::Complex<f32>;
#[cfg(feature = "complex")]
type C64 = num_complex::Complex<f64>;

// ===========================================================================
// Complex GEMM (c32 / c64, with optional conjA / conjB).
// ===========================================================================

/// Run a complex GEMM for a concrete `(complex type, ISA, tile)`: do the
/// orientation swap (which also **swaps the conj flags**, since
/// `(A̅·B)ᵀ = Bᵀ·A̅ᵀ` puts old-A's conj on the new RHS), then dispatch the now-fixed
/// `(conj_a, conj_b)` to the matching const-generic `ComplexGemm` variant — the
/// runtime→compile-time conj branch lives here, never in the hot loop.
///
/// # Safety
/// `t`'s pointers valid; `c` not aliasing `a`/`b`. Run after the degenerate check.
#[cfg(feature = "complex")]
#[inline]
unsafe fn run_complex<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    conj_a: bool,
    conj_b: bool,
    mut t: Task<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    T: crate::scalar::ComplexFloat,
    S: KernelSimd<T, T, T, T>,
{
    use crate::kernel::ComplexGemm;
    unsafe {
        let (mut ca, mut cb) = (conj_a, conj_b);
        if orient_transpose(&mut t) {
            core::mem::swap(&mut ca, &mut cb);
        }
        match (ca, cb) {
            (false, false) => driver::run::<ComplexGemm<T, false, false>, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            ),
            (true, false) => driver::run::<ComplexGemm<T, true, false>, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            ),
            (false, true) => driver::run::<ComplexGemm<T, false, true>, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            ),
            (true, true) => driver::run::<ComplexGemm<T, true, true>, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            ),
        }
    }
}

/// Complex element types gemmkit can dispatch (`Complex<f32>` / `Complex<f64>`).
/// Separate from [`GemmScalar`](super::GemmScalar) because complex carries the conj op-family. The
/// [`crate::scalar::ComplexFloat`] supertrait supplies the real component type and the
/// re/im split the SoA kernel and its epilogue need.
#[cfg(feature = "complex")]
pub trait ComplexScalar: crate::scalar::ComplexFloat {
    /// Dispatch a complex GEMM (with conj flags) to the best ISA.
    ///
    /// # Safety
    /// `t`'s pointers valid; `c` not aliasing `a`/`b`.
    #[doc(hidden)]
    unsafe fn dispatch_complex(
        conj_a: bool,
        conj_b: bool,
        t: Task<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );
}

/// Top-level complex entry: degenerate cases (`C <- beta·C`) then the ISA dispatch.
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` not aliasing `a`/`b`.
#[cfg(feature = "complex")]
pub(crate) unsafe fn execute_complex<T: ComplexScalar>(
    conj_a: bool,
    conj_b: bool,
    t: Task<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        if t.k == 0 || t.alpha == T::ZERO {
            scale_c_float(t.beta, t.c, t.m, t.n, t.rsc, t.csc);
            return;
        }
        T::dispatch_complex(conj_a, conj_b, t, par, ws);
    }
}

#[cfg(feature = "complex")]
type CplxFn<T> = unsafe fn(bool, bool, Task<T>, Parallelism, &mut Workspace);

#[cfg(feature = "complex")]
#[derive(Copy, Clone)]
struct CplxDispatched<T> {
    run: CplxFn<T>,
}

#[cfg(feature = "complex")]
unsafe fn gemm_c32_scalar(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_complex::<C32, ScalarTok, 4, 4>(ScalarTok, ca, cb, t, par, ws) }
}
#[cfg(feature = "complex")]
unsafe fn gemm_c64_scalar(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_complex::<C64, ScalarTok, 4, 4>(ScalarTok, ca, cb, t, par, ws) }
}
// SoA tiles use *real*-lane geometry: `LANES = SimdOps<real>::LANES` (real lanes =
// complex rows), and the kernel needs `2·MR_REG·NR` accumulator registers (re + im
// banks) plus `2·MR_REG` A-plane regs — so the tiles are smaller than the old AoS ones.
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_fma(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 FMA: real LANES = 8, MR = 1*8 = 8 complex rows, NR = 5 → 10 acc + 2 A + 2 B
    // splat = 14 of 16 YMM. The 2 spare matter: a full 16/16 tile (NR = 6) spills
    // accumulators and roughly halves throughput.
    unsafe { run_complex::<C32, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_fma(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 FMA: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 5 (same 14-YMM budget).
    unsafe { run_complex::<C64, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_avx512(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 AVX-512: real LANES = 16, MR = 2*16 = 32, NR = 6 → 24 acc + 4 A + 2 B = 30 ZMM.
    unsafe { run_complex::<C32, Avx512, 2, 6>(Avx512, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_avx512(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 AVX-512: real LANES = 8, MR = 2*8 = 16, NR = 6 (same 30-ZMM budget).
    unsafe { run_complex::<C64, Avx512, 2, 6>(Avx512, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c32_neon(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 NEON: real LANES = 4, MR = 2*4 = 8 complex rows, NR = 5 → 20 acc + 4 A + 2 B
    // splat = 26 of the 32 v0–v31, leaving room for the in-flight load/lane temporaries.
    unsafe { run_complex::<C32, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c64_neon(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 NEON: real LANES = 2, MR = 2*2 = 4 complex rows, NR = 5 (same 26-vreg budget and
    // the same MR_REG=2 / NR=5 rationale as c32 above).
    unsafe { run_complex::<C64, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
// wasm simd128 complex
// real `Reg` = v128
// The SoA kernel needs `2·MR_REG·NR` accumulators (re+im) + `2·MR_REG` A regs + 2 B splats
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c32_simd128(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 simd128: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 4.
    unsafe { run_complex::<C32, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c64_simd128(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 simd128: real LANES = 2, MR = 1*2 = 2 complex rows, NR = 4 (same 12-v128 budget).
    unsafe { run_complex::<C64, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}

#[cfg(feature = "complex")]
const CDISP_C32_SCALAR: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_scalar,
};
#[cfg(feature = "complex")]
const CDISP_C64_SCALAR: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_scalar,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C32_FMA: CplxDispatched<C32> = CplxDispatched { run: gemm_c32_fma };
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C64_FMA: CplxDispatched<C64> = CplxDispatched { run: gemm_c64_fma };
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C32_AVX512: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_avx512,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C64_AVX512: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_avx512,
};
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
const CDISP_C32_NEON: CplxDispatched<C32> = CplxDispatched { run: gemm_c32_neon };
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
const CDISP_C64_NEON: CplxDispatched<C64> = CplxDispatched { run: gemm_c64_neon };
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
const CDISP_C32_SIMD128: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_simd128,
};
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
const CDISP_C64_SIMD128: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_simd128,
};

/// `c32` ISA selection (the complex multiply uses only AVX2/AVX-512 float ops).
#[cfg(feature = "complex")]
fn select_c32() -> CplxDispatched<C32> {
    match forced_isa() {
        ForcedIsa::Scalar => return CDISP_C32_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return CDISP_C32_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return CDISP_C32_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return CDISP_C32_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return CDISP_C32_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return CDISP_C32_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return CDISP_C32_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        CDISP_C32_NEON
    }
    // `simd128` on wasm32, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            CDISP_C32_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            CDISP_C32_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        CDISP_C32_SCALAR
    }
}

/// `c64` ISA selection.
#[cfg(feature = "complex")]
fn select_c64() -> CplxDispatched<C64> {
    match forced_isa() {
        ForcedIsa::Scalar => return CDISP_C64_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return CDISP_C64_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return CDISP_C64_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return CDISP_C64_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return CDISP_C64_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return CDISP_C64_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return CDISP_C64_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        CDISP_C64_NEON
    }
    // `simd128` on wasm32, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            CDISP_C64_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            CDISP_C64_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        CDISP_C64_SCALAR
    }
}

memoized_select!(
    GEMM_C32,
    dispatched_c32,
    CplxDispatched<C32>,
    select_c32,
    "The memoized `Complex<f32>` dispatch descriptor (selection runs once).",
    "complex"
);
memoized_select!(
    GEMM_C64,
    dispatched_c64,
    CplxDispatched<C64>,
    select_c64,
    "The memoized `Complex<f64>` dispatch descriptor (selection runs once).",
    "complex"
);

#[cfg(feature = "complex")]
impl ComplexScalar for C32 {
    #[inline]
    unsafe fn dispatch_complex(
        ca: bool,
        cb: bool,
        t: Task<C32>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        let d = dispatched_c32();
        unsafe { (d.run)(ca, cb, t, par, ws) }
    }
}
#[cfg(feature = "complex")]
impl ComplexScalar for C64 {
    #[inline]
    unsafe fn dispatch_complex(
        ca: bool,
        cb: bool,
        t: Task<C64>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        let d = dispatched_c64();
        unsafe { (d.run)(ca, cb, t, par, ws) }
    }
}
