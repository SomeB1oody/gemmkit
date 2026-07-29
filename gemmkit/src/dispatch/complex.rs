//! `Complex<f32>`/`Complex<f64>` GEMM dispatch
//!
//! Resolves the runtime `conj_a`/`conj_b` flags to a compile-time
//! `ComplexGemm<T, CONJ_A, CONJ_B>` monomorphization. Holds the per-ISA wrappers, the
//! memoized descriptors, ISA selection, and the `ComplexScalar` impls that plug into the
//! dispatch layer

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::{Task, orient_transpose, scale_c_float};
use crate::driver;
#[cfg(all(feature = "complex", feature = "epilogue"))]
use crate::kernel::epilogue::{Epilogue, FusedEpi};
use crate::parallel::Parallelism;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512F, Fma};
use crate::simd::{KernelSimd, ScalarTok};
use crate::workspace::Workspace;

/// c32 dispatch alias
#[cfg(feature = "complex")]
type C32 = num_complex::Complex<f32>;
/// c64 dispatch alias
#[cfg(feature = "complex")]
type C64 = num_complex::Complex<f64>;

// c32 / c64 complex GEMM: conj_a/conj_b dispatch

/// Run a complex GEMM for one concrete `(T, ISA, tile)`
///
/// Applies the orientation swap, then dispatches the resulting `(conj_a, conj_b)` to the
/// matching const-generic `ComplexGemm<T, CONJ_A, CONJ_B>` monomorphization. This turns the
/// runtime conj flags into a compile-time branch here, not inside the hot loop
///
/// The orientation swap also exchanges the conj flags. `(conj(A)*B)^T = B^T*conj(A)^T` moves
/// the original A conjugation onto what is now the RHS operand
///
/// # Safety
/// `t`'s pointers are valid and `c` does not alias `a` or `b`. Call this after the degenerate
/// check
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

/// Fused-bias complex GEMM for one concrete `(T, ISA, tile)`
///
/// Mirrors [`run_complex`] but threads `epi` into [`driver::run_epilogue`] instead of
/// [`driver::run`]
///
/// The orientation swap exchanges both the conj flags, as in [`run_complex`], and the bias
/// axis. A row-major-ish `C` makes the engine compute `C^T = B^T*A^T`, which swaps `m<->n`.
/// A per-row bias becomes a per-col bias in the oriented frame
///
/// Complex has no special paths to mirror. [`run_complex`] always calls [`driver::run`], with
/// no gemv, small_mn, or small_k arms, so this is the only fused entry point needed. The
/// complex kernel stores the same bits plain `gemm_cplx` would. On the final depth panel only,
/// it sweeps the tile and applies the bias in place. So the fused result is bit-identical to
/// `gemm_cplx` followed by the same element-wise bias add, for every shape and conj combination
///
/// # Safety
/// `t`'s pointers are valid. `c` does not alias `a` or `b`. `epi`'s bias is valid for the
/// problem's `m` and `n` from before the orientation swap. Call this after the degenerate check
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
            // row-major-ish C computes C^T = B^T*A^T (m<->n swap), so bias axis flips too
            epi.flip_bias();
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

/// Complex element types gemmkit can dispatch: `Complex<f32>` and `Complex<f64>`
///
/// Separate from [`GemmScalar`](super::GemmScalar) because complex carries the conj op-family.
/// The [`crate::scalar::ComplexFloat`] supertrait supplies the real component type and the
/// re/im split the SoA kernel and its epilogue need
#[cfg(feature = "complex")]
pub trait ComplexScalar: crate::scalar::ComplexFloat {
    /// Dispatch a complex GEMM (with conj flags) to the best available ISA
    ///
    /// # Safety
    /// `t`'s pointers are valid and `c` does not alias `a` or `b`
    #[doc(hidden)]
    unsafe fn dispatch_complex(
        conj_a: bool,
        conj_b: bool,
        t: Task<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );

    /// Dispatch a fused-bias complex GEMM (with conj flags) to the best available ISA
    ///
    /// # Safety
    /// `t`'s pointers are valid, `c` does not alias `a` or `b`, and `epi`'s bias is valid for
    /// the problem's `m` and `n`
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

/// Top-level complex entry, called by the API layer
///
/// Handles the degenerate case, `C <- beta*C` when `k == 0` or `alpha == 0`, then runs the
/// ISA dispatch
///
/// # Safety
/// `t`'s pointers are valid for the implied regions and `c` does not alias `a` or `b`
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

/// The degenerate fused complex epilogue is `C[i,j] <- beta*C[i,j] + bias`, in the user frame
///
/// It runs when the `A*B` term vanishes, when `k == 0` or `alpha == 0`
///
/// This mirrors `fused_degenerate_float`, but the epilogue here is bias-only since complex has
/// no activation. It reuses `epi.apply`, which for the complex family is exactly the bias add.
/// The conj flags never appear here, since there is no `A*B` term to conjugate
///
/// # Safety
/// `c` is valid for the `m x n` region and `epi`'s bias is valid for the problem's `m` and `n`
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
                // Same beta-then-bias shape as the real-float degenerate path, complex-typed
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

/// Top-level fused-bias complex entry, called by the API layer
///
/// Handles the degenerate case, `C <- beta*C + bias` in the user frame, then runs the ISA
/// dispatch
///
/// # Safety
/// `t`'s pointers are valid for the implied regions. `c` does not alias `a` or `b`. `epi`'s
/// bias is valid for the problem's `m` and `n`. The API layer already validates this
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
        // A*B vanishes (k == 0 or alpha == 0): C <- beta*C + bias, element-wise in the user
        // frame. Conj is irrelevant since there is no product to conjugate
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
/// (bias-only for complex). Every dispatched complex type supplies one
#[cfg(all(feature = "complex", feature = "epilogue"))]
type CplxFusedFn<T> = unsafe fn(bool, bool, Task<T>, FusedEpi<T>, Parallelism, &mut Workspace);

#[cfg(feature = "complex")]
#[derive(Copy, Clone)]
struct CplxDispatched<T> {
    run: CplxFn<T>,
    /// Fused-bias entry, using the same tile as `run`. The epilogue is a tile-local post-pass
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
// SoA tiles size against real-lane geometry, where LANES = SimdOps<Real>::LANES and real lanes
// are complex rows. The kernel needs 2*MR_REG*NR accumulators (re and im banks) plus 2*MR_REG
// A-plane registers, twice a real kernel's budget for the same (MR_REG, NR)
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_fma(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 FMA: real LANES = 8, MR = 1*8 = 8 complex rows, NR = 5, for 10 accumulators plus 2
    // A and 2 B splats, 14 of 16 YMM registers. A full 16/16 tile (NR = 6) would spill
    // accumulators
    unsafe { run_complex::<C32, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_fma(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 FMA: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 5, same 14-YMM budget as c32
    unsafe { run_complex::<C64, Fma, 1, 5>(Fma, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c32_avx512f(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 AVX-512F: real LANES = 16, MR = 2*16 = 32, NR = 6 -> 24 acc + 4 A + 2 B splat = 30 ZMM
    unsafe { run_complex::<C32, Avx512F, 2, 6>(Avx512F, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_c64_avx512f(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 AVX-512F: real LANES = 8, MR = 2*8 = 16, NR = 6, same 30-ZMM budget as c32
    unsafe { run_complex::<C64, Avx512F, 2, 6>(Avx512F, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c32_neon(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 NEON: real LANES = 4, MR = 2*4 = 8 complex rows, NR = 5 -> 20 acc + 4 A + 2 B
    // splats live at once, 26 of the 32 v0-v31 registers, which leaves headroom for load
    // and lane temporaries
    unsafe { run_complex::<C32, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
#[cfg(all(feature = "complex", target_arch = "aarch64"))]
unsafe fn gemm_c64_neon(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 NEON: real LANES = 2, MR = 2*2 = 4 complex rows, NR = 5, same 26-vreg budget as
    // c32 above
    unsafe { run_complex::<C64, Neon, 2, 5>(Neon, ca, cb, t, par, ws) }
}
// wasm simd128 complex: real Reg = v128. The SoA kernel needs 2*MR_REG*NR accumulators
// (re and im banks) plus 2*MR_REG A regs plus 2 B splats, the same register math as the
// other ISAs above
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c32_simd128(ca: bool, cb: bool, t: Task<C32>, par: Parallelism, ws: &mut Workspace) {
    // c32 simd128: real LANES = 4, MR = 1*4 = 4 complex rows, NR = 4 -> 8 acc + 2 A + 2 B = 12 v128
    unsafe { run_complex::<C32, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}
#[cfg(all(
    feature = "complex",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn gemm_c64_simd128(ca: bool, cb: bool, t: Task<C64>, par: Parallelism, ws: &mut Workspace) {
    // c64 simd128: real LANES = 2, MR = 1*2 = 2 complex rows, NR = 4, same 12-v128 budget as c32
    unsafe { run_complex::<C64, Simd128, 1, 4>(Simd128, ca, cb, t, par, ws) }
}

// Each fused-bias entry point pairs 1 c32/c64 type with 1 ISA, reusing the plain wrapper's
// tile above. The bias epilogue is a tile-local post-pass, so the accumulator budget stays
// the same

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
unsafe fn gemm_c32_avx512f_fused(
    ca: bool,
    cb: bool,
    t: Task<C32>,
    epi: FusedEpi<C32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C32, Avx512F, 2, 6>(Avx512F, ca, cb, t, epi, par, ws) }
}
#[cfg(all(
    feature = "complex",
    feature = "epilogue",
    any(target_arch = "x86", target_arch = "x86_64")
))]
unsafe fn gemm_c64_avx512f_fused(
    ca: bool,
    cb: bool,
    t: Task<C64>,
    epi: FusedEpi<C64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_complex_fused::<C64, Avx512F, 2, 6>(Avx512F, ca, cb, t, epi, par, ws) }
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
const CDISP_C32_AVX512F: CplxDispatched<C32> = CplxDispatched {
    run: gemm_c32_avx512f,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c32_avx512f_fused,
};
#[cfg(all(feature = "complex", any(target_arch = "x86", target_arch = "x86_64")))]
const CDISP_C64_AVX512F: CplxDispatched<C64> = CplxDispatched {
    run: gemm_c64_avx512f,
    #[cfg(feature = "epilogue")]
    run_fused: gemm_c64_avx512f_fused,
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

/// c32 ISA selection: the complex kernel only needs plain AVX2/AVX-512F float ops (no
/// VNNI/BF16 dot path)
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
                "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
            );
            return CDISP_C32_AVX512F;
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
            return CDISP_C32_AVX512F;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return CDISP_C32_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        CDISP_C32_NEON
    }
    // wasm32: simd128 if enabled, else scalar
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

/// c64 ISA selection (mirrors [`select_c32`])
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
                "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
            );
            return CDISP_C64_AVX512F;
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
            return CDISP_C64_AVX512F;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return CDISP_C64_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        CDISP_C64_NEON
    }
    // wasm32: simd128 if enabled, else scalar
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
