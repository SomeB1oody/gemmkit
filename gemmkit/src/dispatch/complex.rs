//! `c32`/`c64` complex GEMM dispatch (optional conjA/conjB): the runtime->compile-time
//! conj branch, per-ISA wrappers, descriptors, selection, and the `ComplexScalar` impls

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::{Task, orient_transpose, scale_c_float};
use crate::driver;
#[cfg(all(feature = "complex", feature = "epilogue"))]
use crate::kernel::epilogue::{BiasSpec, Epilogue, FusedEpi};
use crate::parallel::Parallelism;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::simd::{KernelSimd, ScalarTok};
use crate::workspace::Workspace;

/// `c32` / `c64` element-type aliases (the complex-GEMM dispatch types)
#[cfg(feature = "complex")]
type C32 = num_complex::Complex<f32>;
#[cfg(feature = "complex")]
type C64 = num_complex::Complex<f64>;

// Complex GEMM (c32 / c64, with optional conjA / conjB)

/// Run a complex GEMM for a concrete `(complex type, ISA, tile)`: do the
/// orientation swap (which also **swaps the conj flags**, since
/// `(conj(A)*B)^T = B^T*conj(A)^T` puts old-A's conj on the new RHS), then dispatch the now-fixed
/// `(conj_a, conj_b)` to the matching const-generic `ComplexGemm` variant: the
/// runtime->compile-time conj branch lives here, never in the hot loop
///
/// # Safety
/// `t`'s pointers valid; `c` not aliasing `a`/`b`. Run after the degenerate check
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

/// **Fused-bias** complex GEMM for a concrete `(complex type, ISA, tile)`: the mirror of
/// [`run_complex`] with the bias epilogue threaded into the driver's store. The orientation swap
/// swaps **both** the conj flags (`(conj(A)*B)^T = B^T*conj(A)^T` puts old-A's conj on the new
/// RHS) **and** the bias axis (a row-major-ish C makes the engine compute `C^T = B^T*A^T`,
/// swapping `m<->n`, so a user per-row bias becomes per-col in the oriented frame: the same
/// `C^T` reasoning as the float path; a field write, not a new monomorphization)
///
/// Complex has **no** special paths to mirror: [`run_complex`] routes everything to
/// [`driver::run`] (no gemv / small_mn / small_k arms), so this single [`driver::run_epilogue`]
/// mirror is complete. The complex family stores exactly the bits plain `gemm_cplx` would and the
/// epilogue's tile-local post-pass maps them in place on the final depth panel, so the result is
/// bit-identical to `gemm_cplx` then the same element-wise bias add, for every shape and conj
/// combination
///
/// # Safety
/// `t`'s pointers valid; `c` not aliasing `a`/`b`; `epi`'s bias valid for the (pre-swap) problem's
/// `m`/`n`. Run after the degenerate check
#[cfg(all(feature = "complex", feature = "epilogue"))]
#[inline]
unsafe fn run_complex_fused<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    conj_a: bool,
    conj_b: bool,
    mut t: Task<T>,
    mut epi: FusedEpi<T>,
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
            // The `C^T = B^T*A^T` swap of `m<->n` flips the bias axis too (per-row <-> per-col)
            epi.bias = match epi.bias {
                BiasSpec::None => BiasSpec::None,
                BiasSpec::Row(p) => BiasSpec::Col(p),
                BiasSpec::Col(p) => BiasSpec::Row(p),
            };
        }
        match (ca, cb) {
            (false, false) => {
                driver::run_epilogue::<ComplexGemm<T, false, false>, S, FusedEpi<T>, MR_REG, NR>(
                    simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                    t.c, t.rsc, t.csc, &epi, par, ws,
                )
            }
            (true, false) => {
                driver::run_epilogue::<ComplexGemm<T, true, false>, S, FusedEpi<T>, MR_REG, NR>(
                    simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                    t.c, t.rsc, t.csc, &epi, par, ws,
                )
            }
            (false, true) => {
                driver::run_epilogue::<ComplexGemm<T, false, true>, S, FusedEpi<T>, MR_REG, NR>(
                    simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                    t.c, t.rsc, t.csc, &epi, par, ws,
                )
            }
            (true, true) => {
                driver::run_epilogue::<ComplexGemm<T, true, true>, S, FusedEpi<T>, MR_REG, NR>(
                    simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                    t.c, t.rsc, t.csc, &epi, par, ws,
                )
            }
        }
    }
}

/// Complex element types gemmkit can dispatch (`Complex<f32>` / `Complex<f64>`).
/// Separate from [`GemmScalar`](super::GemmScalar) because complex carries the conj op-family. The
/// [`crate::scalar::ComplexFloat`] supertrait supplies the real component type and the
/// re/im split the SoA kernel and its epilogue need
#[cfg(feature = "complex")]
pub trait ComplexScalar: crate::scalar::ComplexFloat {
    /// Dispatch a complex GEMM (with conj flags) to the best ISA
    ///
    /// # Safety
    /// `t`'s pointers valid; `c` not aliasing `a`/`b`
    #[doc(hidden)]
    unsafe fn dispatch_complex(
        conj_a: bool,
        conj_b: bool,
        t: Task<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );

    /// Dispatch a **fused-bias** complex GEMM (with conj flags) to the best ISA
    ///
    /// # Safety
    /// `t`'s pointers valid; `c` not aliasing `a`/`b`; `epi`'s bias valid for the problem's
    /// `m`/`n`
    #[doc(hidden)]
    #[cfg(feature = "epilogue")]
    unsafe fn dispatch_complex_fused(
        conj_a: bool,
        conj_b: bool,
        t: Task<Self>,
        epi: FusedEpi<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );
}

/// Top-level complex entry: degenerate cases (`C <- beta*C`) then the ISA dispatch
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` not aliasing `a`/`b`
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

/// The degenerate fused complex epilogue `C[i,j] <- beta*C[i,j] + bias`, in the **user** frame,
/// run when the `A*B` term vanishes (`k == 0` or `alpha == 0`). Mirrors `fused_degenerate_float`
/// but the epilogue is **bias-only** (complex has no activation), so it reuses `epi.apply`, which
/// for the complex family is exactly the bias add. The conj flags do not appear: with no `A*B`
/// term there is nothing to conjugate, so they are irrelevant here
///
/// # Safety
/// `c` valid for the `m x n` region; `epi`'s bias valid for the problem's `m`/`n`
#[cfg(all(feature = "complex", feature = "epilogue"))]
unsafe fn fused_degenerate_cplx<T>(t: &Task<T>, epi: &FusedEpi<T>)
where
    T: crate::scalar::ComplexFloat,
    FusedEpi<T>: Epilogue<crate::kernel::ComplexGemm<T, false, false>>,
{
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                let p = t.c.offset(i as isize * t.rsc + j as isize * t.csc);
                // `beta*C` combined in `T`; then the (bias-only) epilogue adds the bias. Exactly
                // the real-float `fused_degenerate_float` shape, complex-typed
                let base = if t.beta == T::ZERO {
                    T::ZERO
                } else if t.beta == T::ONE {
                    *p
                } else {
                    t.beta * *p
                };
                *p = <FusedEpi<T> as Epilogue<crate::kernel::ComplexGemm<T, false, false>>>::apply(
                    epi, base, i, j,
                );
            }
        }
    }
}

/// Top-level fused-bias complex entry: the degenerate cases (`C <- beta*C + bias`) in the **user**
/// frame, then the ISA dispatch
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` not aliasing `a`/`b`; `epi`'s bias valid for
/// the problem's `m`/`n` (the API validates this)
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub(crate) unsafe fn execute_complex_fused<T: ComplexScalar>(
    conj_a: bool,
    conj_b: bool,
    t: Task<T>,
    epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        // The `A*B` term vanishes (`k == 0` or `alpha == 0`): `C <- beta*C + bias`, element-wise
        // in the user frame; conj is irrelevant (no product to conjugate)
        if t.k == 0 || t.alpha == T::ZERO {
            fused_degenerate_cplx(&t, &epi);
            return;
        }
        T::dispatch_complex_fused(conj_a, conj_b, t, epi, par, ws);
    }
}

#[cfg(feature = "complex")]
type CplxFn<T> = unsafe fn(bool, bool, Task<T>, Parallelism, &mut Workspace);

/// The fused-bias complex kernel entry: a plain [`Task`] plus the runtime-composed [`FusedEpi`]
/// (bias-only for complex). Non-optional: every dispatched complex type supplies one
#[cfg(all(feature = "complex", feature = "epilogue"))]
type CplxFusedFn<T> = unsafe fn(bool, bool, Task<T>, FusedEpi<T>, Parallelism, &mut Workspace);

#[cfg(feature = "complex")]
#[derive(Copy, Clone)]
struct CplxDispatched<T> {
    run: CplxFn<T>,
    /// Fused-bias entry (same tile as `run`; the epilogue is tile-local)
    #[cfg(feature = "epilogue")]
    run_fused: CplxFusedFn<T>,
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
// complex rows), and the kernel needs `2*MR_REG*NR` accumulator registers (re + im
// banks) plus `2*MR_REG` A-plane regs, so the tiles are smaller than the old AoS ones
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_fma(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 FMA: real LANES = 8, MR = 1*8 = 8 complex rows, NR = 5 -> 10 acc + 2 A + 2 B
    // splat = 14 of 16 YMM. The 2 spare matter: a full 16/16 tile (NR = 6) spills
    // accumulators and roughly halves throughput
    unsafe { run_complex::<C32, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_fma(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 FMA: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 5 (same 14-YMM budget)
    unsafe { run_complex::<C64, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_avx512(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 AVX-512: real LANES = 16, MR = 2*16 = 32, NR = 6 -> 24 acc + 4 A + 2 B = 30 ZMM
    unsafe { run_complex::<C32, Avx512, 2, 6>(Avx512, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_avx512(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 AVX-512: real LANES = 8, MR = 2*8 = 16, NR = 6 (same 30-ZMM budget)
    unsafe { run_complex::<C64, Avx512, 2, 6>(Avx512, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c32_neon(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 NEON: real LANES = 4, MR = 2*4 = 8 complex rows, NR = 5 -> 20 acc + 4 A + 2 B
    // splat = 26 of the 32 v0-v31, leaving room for the in-flight load/lane temporaries
    unsafe { run_complex::<C32, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c64_neon(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 NEON: real LANES = 2, MR = 2*2 = 4 complex rows, NR = 5 (same 26-vreg budget and
    // the same MR_REG=2 / NR=5 rationale as c32 above)
    unsafe { run_complex::<C64, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
// wasm simd128 complex
// real `Reg` = v128
// The SoA kernel needs `2*MR_REG*NR` accumulators (re+im) + `2*MR_REG` A regs + 2 B splats
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c32_simd128(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 simd128: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 4
    unsafe { run_complex::<C32, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c64_simd128(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 simd128: real LANES = 2, MR = 1*2 = 2 complex rows, NR = 4 (same 12-v128 budget)
    unsafe { run_complex::<C64, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}

// fused-bias entry points: one per (c32/c64, ISA), same tiles as the plain wrappers (the
// bias epilogue is a tile-local post-pass, so the accumulator-register budget is unchanged)

#[cfg(all(feature = "complex", feature = "epilogue"))]
unsafe fn gemm_c32_scalar_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, ScalarTok, 4, 4>(ScalarTok, ca, cb, t, epi, par, ws) }
}
#[cfg(all(feature = "complex", feature = "epilogue"))]
unsafe fn gemm_c64_scalar_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, ScalarTok, 4, 4>(ScalarTok, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_c32_fma_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, Fma, 1, 5>(Fma, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_c64_fma_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, Fma, 1, 5>(Fma, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_c32_avx512_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, Avx512, 2, 6>(Avx512, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_c64_avx512_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, Avx512, 2, 6>(Avx512, ca, cb, t, epi, par, ws) }
}
#[cfg(all(feature = "complex", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_c32_neon_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, Neon, 2, 5>(Neon, ca, cb, t, epi, par, ws) }
}
#[cfg(all(feature = "complex", feature = "epilogue", target_arch = "aarch64"))]
unsafe fn gemm_c64_neon_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, Neon, 2, 5>(Neon, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    feature = "epilogue",
    target_feature = "simd128"
))]
unsafe fn gemm_c32_simd128_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, Simd128, 1, 4>(Simd128, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    feature = "epilogue",
    target_feature = "simd128"
))]
unsafe fn gemm_c64_simd128_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, Simd128, 1, 4>(Simd128, ca, cb, t, epi, par, ws) }
}

#[cfg(feature = "complex")]
const CDISP_C32_SCALAR: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_scalar,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_scalar_fused,
};
#[cfg(feature = "complex")]
const CDISP_C64_SCALAR: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_scalar,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_scalar_fused,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C32_FMA: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_fma,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_fma_fused,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C64_FMA: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_fma,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_fma_fused,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C32_AVX512: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_avx512,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_avx512_fused,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C64_AVX512: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_avx512,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_avx512_fused,
};
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
const CDISP_C32_NEON: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_neon,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_neon_fused,
};
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
const CDISP_C64_NEON: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_neon,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_neon_fused,
};
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
const CDISP_C32_SIMD128: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_simd128,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_simd128_fused,
};
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
const CDISP_C64_SIMD128: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_simd128,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_simd128_fused,
};

/// `c32` ISA selection (the complex multiply uses only AVX2/AVX-512 float ops)
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

/// `c64` ISA selection
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
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_complex_fused(
        ca: bool,
        cb: bool,
        t: Task<C32>,
        epi: FusedEpi<C32>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        let d = dispatched_c32();
        unsafe { (d.run_fused)(ca, cb, t, epi, par, ws) }
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
    #[cfg(feature = "epilogue")]
    #[inline]
    unsafe fn dispatch_complex_fused(
        ca: bool,
        cb: bool,
        t: Task<C64>,
        epi: FusedEpi<C64>,
        par: Parallelism,
        ws: &mut Workspace,
    ) {
        let d = dispatched_c64();
        unsafe { (d.run_fused)(ca, cb, t, epi, par, ws) }
    }
}
