//! `f32`/`f64` homogeneous-float dispatch: driver entries, per-ISA wrappers
//! (plain / prepacked / fused), descriptors, selection, and the `GemmScalar` +
//! `FusedScalar` impls.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::{GemmScalar, PackedConsume, Task, orient_transpose, scale_c_float, small_mn_eligible};
use crate::driver;
use crate::kernel::FloatGemm;
use crate::kernel::epilogue::{BiasSpec, Epilogue, FusedEpi};
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

/// gemv route + orientation normalization + the generic driver, for a concrete
/// `(type, ISA, tile)`. Concrete typing here gives us the `Float` bound the
/// fully generic driver intentionally lacks.
///
/// # Safety
/// As [`execute`].
#[inline]
unsafe fn run_typed<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        // gemv shape, unless the dedicated path has been disabled via tuning
        // (then it falls through to the general driver, which is also correct).
        if (t.n == 1 || t.m == 1) && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold() {
            gemv::run_typed::<T, S>(
                simd, t.m, t.k, t.n, par, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                t.c, t.rsc, t.csc,
            );
            return;
        }

        orient_transpose(&mut t);
        // Small `m,n` with a long contraction, and both operands streaming contiguously along
        // `k`: the driver would pad the tiny row/col tiles up to a full microtile and pack mostly
        // padding, whereas the horizontal path computes each output as a direct SIMD dot over `k`,
        // reading A/B in place. (At small `k` the small_k route below is already the right in-place
        // tool, so this only claims the long-`k` regime; a strided layout would force a scalar dot
        // that loses to the driver, so it stays on the driver.)
        if small_mn_eligible(&t) {
            small_mn::run::<T, S>(
                simd, t.m, t.k, t.n, par, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                t.c, t.rsc, t.csc,
            );
            return;
        }
        // Skinny / low-depth shape: the whole product is one depth panel, so the driver's
        // blocking + packing setup is pure overhead. Read A/B in place over the microkernel.
        if t.k <= tuning::small_k_threshold() {
            small_k::run::<FloatGemm<T>, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            );
            return;
        }
        driver::run::<FloatGemm<T>, S, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, par, ws,
        );
    }
}

/// **Fused-epilogue** driver entry for a concrete `(type, ISA, tile)`: a mirror of [`run_typed`]
/// with the epilogue fused into each route, so a fused shape takes the *same* kernel plain `gemm`
/// would (gemv / small_mn / small_k / general driver) rather than paying the driver's
/// pack/blocking overhead on a shape the special paths win. Each route stores exactly the bits
/// plain `gemm` would and applies the same scalar map exactly once per element, so the fused
/// result is bit-identical to `gemm()` followed by that map for *every* shape (the vector fast
/// path agrees bitwise with the scalar map by the [`Epilogue::apply_reg`] contract). gemv routes
/// before orientation normalization in the **user** frame (no bias flip); the other routes run
/// after the orientation swap, which flips the bias axis (a row-major-ish C makes the engine
/// compute `Cᵀ = Bᵀ·Aᵀ`, swapping `m↔n`, so a user per-row bias becomes per-col in the oriented
/// frame — a field write, not a new monomorphization).
///
/// # Safety
/// As [`run_typed`], plus `epi`'s interior pointers valid for the (pre-swap) problem's `m`/`n`.
#[inline]
unsafe fn run_typed_fused<T, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<T>,
    mut epi: FusedEpi<T>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    // The real-float epilogue arithmetic (`FusedEpi<T>: Epilogue<FloatGemm<T>>`) and the special
    // paths need `Float<Acc = T> + PartialOrd`; `FusedScalar` no longer implies them (it now also
    // covers the narrow `f16`/`bf16` types, which route through `run_typed_mixed_fused` instead).
    T: Float<Acc = T> + PartialOrd,
    S: SimdOps<T>,
{
    unsafe {
        // gemv shape (unless the dedicated path is disabled via tuning): fused via a final
        // in-place epilogue sweep. gemv dispatches BEFORE orientation normalization, so `epi`
        // stays in the user frame (no bias-axis flip) — `run_typed_epi` resolves the per-row /
        // per-col coordinate itself from the `n == 1` / `m == 1` branch.
        if (t.n == 1 || t.m == 1) && core::cmp::min(t.m, t.n) <= tuning::gemv_threshold() {
            gemv::run_typed_epi::<T, S, FusedEpi<T>>(
                simd, t.m, t.k, t.n, par, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                t.c, t.rsc, t.csc, &epi,
            );
            return;
        }

        // Orientation normalization flips the bias axis for the routes below (they all consume
        // the oriented `epi`): a row-major-ish C computes `Cᵀ = Bᵀ·Aᵀ` (swapping `m↔n`), so a
        // user per-row bias becomes per-col in the oriented frame.
        let swap = orient_transpose(&mut t);
        if swap {
            epi.bias = match epi.bias {
                BiasSpec::None => BiasSpec::None,
                BiasSpec::Row(p) => BiasSpec::Col(p),
                BiasSpec::Col(p) => BiasSpec::Row(p),
            };
        }

        // Small `m,n` with a long contraction and both operands streaming contiguously along
        // `k`: the horizontal path computes each output as a direct SIMD dot, applying the
        // epilogue at each cell's single store.
        if small_mn_eligible(&t) {
            small_mn::run_epi::<T, S, FusedEpi<T>>(
                simd, t.m, t.k, t.n, par, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta,
                t.c, t.rsc, t.csc, &epi,
            );
            return;
        }
        // Skinny / low-depth shape: the whole product is one depth panel over the microkernel,
        // the epilogue fused into the single per-tile store (`last_k` structurally true).
        if t.k <= tuning::small_k_threshold() {
            small_k::run_epi::<FloatGemm<T>, S, FusedEpi<T>, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, &epi, par, ws,
            );
            return;
        }
        driver::run_epilogue::<FloatGemm<T>, S, FusedEpi<T>, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, &epi, par, ws,
        );
    }
}

/// Prepacked-RHS driver entry for a concrete `(type, ISA, tile)`. No gemv
/// route and **no orientation swap** — the API guarantees column-major-ish C
/// (`|csc| >= |rsc|`), so the prepacked buffer is always the genuine RHS.
///
/// # Safety
/// As [`run_typed`], plus `req.packed` valid for the recorded geometry.
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
        // The driver reads panels with the buffer's own `(kc, nc)`, so nothing is
        // re-derived. `nr` is structural (the panel width is this kernel's `NR`);
        // one process's memoized ISA choice guarantees they agree.
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<FloatGemm<T>, S, MR_REG, NR>(
            simd, req.m, req.k, req.n, req.alpha, req.a, req.rsa, req.csa, req.packed, req.kc,
            req.nc, req.beta, req.c, req.rsc, req.csc, par, ws,
        );
    }
}

// ---- per-type, per-ISA monomorphized entry points (the dispatch slots) ----
//
// Tile geometry (MR_REG, NR) is the *only* per-(type, ISA) knob; everything else
// is shared generic code. MR = MR_REG * LANES.

unsafe fn gemm_f32_scalar(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed::<f32, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
unsafe fn gemm_f64_scalar(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed::<f64, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_fma(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 6 → 12 acc + 2 lhs + 1 rhs = 15 YMM.
    unsafe { run_typed::<f32, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_fma(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*4 = 8, NR = 6.
    unsafe { run_typed::<f64, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_avx512(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*16 = 32, NR = 12 → 24 acc + 2 lhs + 1 rhs = 27 ZMM.
    unsafe { run_typed::<f32, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_avx512(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*8 = 16, NR = 12.
    unsafe { run_typed::<f64, Avx512, 2, 12>(Avx512, t, par, ws) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f32_neon(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 4*4 = 16, NR = 4 → 16 acc + 4 lhs + 1 rhs = 21 of the 32 v0–v31 vector
    // registers (NR == LANES, so one loaded RHS vector feeds all four columns). The
    // ~11 spare registers are deliberate: they give the wide out-of-order window the
    // rename headroom to overlap the next step's loads with the current FMAs (the
    // same low-pressure regime gemm uses)
    unsafe { run_typed::<f32, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f64_neon(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 4*2 = 8, NR = 4 → 16 acc + 4 lhs + 2 rhs = 22 vregs (same low-pressure
    // tile as f32).
    unsafe { run_typed::<f64, Neon, 4, 4>(Neon, t, par, ws) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f32_simd128(t: Task<f32>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*4 = 8, NR = 4 → 8 acc + 2 lhs + 1 rhs = 11 live `v128`
    // LLVM's wasm backend spills past ~16 live vectors, and wasm has no hardware FMA
    // (no `LANE_FMA`), so the 4×4 NEON tile would over-subscribe
    unsafe { run_typed::<f32, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f64_simd128(t: Task<f64>, par: Parallelism, ws: &mut Workspace) {
    // MR = 2*2 = 4, NR = 4 → 8 acc + 2 lhs + 1 rhs = 11 live `v128`
    // (same tile shape as f32, f64 just packs 2 lanes per register)
    unsafe { run_typed::<f64, Simd128, 2, 4>(Simd128, t, par, ws) }
}

// ---- prepacked-RHS entry points: one per (type, ISA), same tiles ----

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

// ---- fused-epilogue entry points: one per (f32/f64, ISA), same tiles as the plain
// wrappers (the epilogue is tile-local, so the register budget is unchanged) ----

unsafe fn gemm_f32_scalar_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
unsafe fn gemm_f64_scalar_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, ScalarTok, 4, 4>(ScalarTok, t, epi, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_fma_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_fma_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Fma, 2, 6>(Fma, t, epi, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f32_avx512_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f64_avx512_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Avx512, 2, 12>(Avx512, t, epi, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f32_neon_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f64_neon_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Neon, 4, 4>(Neon, t, epi, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f32_simd128_fused(
    t: Task<f32>,
    epi: FusedEpi<f32>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f32, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f64_simd128_fused(
    t: Task<f64>,
    epi: FusedEpi<f64>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe { run_typed_fused::<f64, Simd128, 2, 4>(Simd128, t, epi, par, ws) }
}

/// The sealed element-type bound for the fused-epilogue public API: the real floats `f32`/`f64`
/// and, under the `half` feature, the narrow floats `f16`/`bf16`. It is a superset of
/// [`GemmScalar`] (for dispatch), sealed (a private supertrait) so downstream crates cannot widen
/// the fused surface. It no longer requires `Float<Acc = Self> + PartialOrd`: the real-float
/// [`FusedEpi`] arithmetic keeps those bounds on its own `Epilogue<FloatGemm<T>>` impl, and the
/// narrow types are not `Float` (they widen to `f32`). What every fused type must provide is the
/// finiteness test used to validate a `LeakyRelu` slope, and the degenerate `C <- act(β·C + bias)`
/// map (type-specific: real floats compute in `T`, narrow types in `f32`, narrowing once).
pub trait FusedScalar: GemmScalar + sealed::Sealed {
    /// `true` iff `self` is finite. `f32`/`f64` use the inherent test; `f16`/`bf16` widen exactly
    /// to `f32` first. `core`-only, so it is `no_std`-safe.
    #[doc(hidden)]
    fn finite(self) -> bool;

    /// The degenerate fused epilogue `C[i,j] <- apply(β·C[i,j], i, j)` in the user frame, run when
    /// the `A·B` term vanishes (`k == 0` or `alpha == 0`).
    ///
    /// # Safety
    /// `c` valid for the `m × n` region; `epi`'s bias valid for the problem's `m`/`n`.
    #[doc(hidden)]
    unsafe fn fused_degenerate(t: &Task<Self>, epi: &FusedEpi<Self>);
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    #[cfg(feature = "half")]
    impl Sealed for half::f16 {}
    #[cfg(feature = "half")]
    impl Sealed for half::bf16 {}
}

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

/// Top-level fused entry (called by the API layer): handle the degenerate cases in the
/// **user** frame (before orientation), then run the ISA-dispatched fused kernel.
///
/// # Safety
/// `task`'s pointers must be valid; `c` must not alias `a`/`b`, and `epi`'s bias slice must
/// not overlap `c` (the API validates this).
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
        // The A·B term vanishes (`k == 0` or `alpha == 0`): `C <- act(beta·C + bias)`,
        // element-wise in the user frame (bias axes as the caller specified). This is
        // type-specific — narrow types combine in `f32` and narrow once — so it is a
        // `FusedScalar` method (the real floats and the narrow types provide their own).
        if task.k == 0 || task.alpha == T::ZERO {
            T::fused_degenerate(&task, &epi);
            return;
        }
        T::dispatch_fused(task, epi, par, ws);
    }
}

/// The degenerate fused epilogue `C[i,j] <- apply(beta·C[i,j], i, j)` in the user frame, for the
/// **real floats** (`f32`/`f64`): all arithmetic is in `T`. The narrow (`f16`/`bf16`) sibling —
/// which combines in `f32` and narrows once — lives in [`crate::dispatch`]'s `mixed` module.
///
/// # Safety
/// `c` valid for the `m × n` region; `epi`'s bias valid for the problem's `m`/`n`.
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
/// Every [`FusedScalar`] type (`f32`/`f64` here, `f16`/`bf16` in the `mixed` module) supplies one,
/// so the slot is non-optional. `pub(super)` so `dispatch/mixed.rs` can name it (as with
/// [`Dispatched`]).
pub(super) type FusedFn<T> = unsafe fn(Task<T>, FusedEpi<T>, Parallelism, &mut Workspace);

/// The memoized dispatch slot for one element type: the standard kernel, the
/// prepacked-RHS kernel, the fused-epilogue kernel, and the microtile `(mr, nr)` they
/// share. Bundling them keeps adding an ISA a single `select_*` ladder arm. `mr`/`nr`
/// mirror the tile constants in the wrappers above and feed `prepack_rhs` (via `rhs_tile`)
/// so the buffer and the consume path agree on the blocking geometry.
#[derive(Copy, Clone)]
pub(super) struct Dispatched<T> {
    pub(super) run: GemmFn<T>,
    pub(super) run_packed: PackedFn<T>,
    /// Fused-epilogue entry (`bias`/activation). Every dispatched type supplies one (`f32`/`f64`
    /// and, under `half`, `f16`/`bf16`), so it is non-optional.
    pub(super) run_fused: FusedFn<T>,
    pub(super) mr: usize,
    pub(super) nr: usize,
    /// The dispatched kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`].
    /// `1` for every widen/homogeneous kernel; `2` for the bf16 `vdpbf16ps` dot kernel.
    /// The prepack constructor rounds the packed depth up to it (via [`GemmScalar`]).
    /// Read only by the `bf16` prepack path, so it is dead code without the `half` feature.
    #[cfg_attr(not(feature = "half"), allow(dead_code))]
    pub(super) depth_multiple: usize,
}

// One descriptor per (type, ISA). `mr = MR_REG·LANES`, `nr = NR` — mirrors the
// tile in each wrapper's comment (scalar 4×4; FMA 16×6 / f64 8×6; AVX-512 32×12 /
// f64 16×12; NEON 16×4 / f64 8×4).
const DISP_F32_SCALAR: Dispatched<f32> = Dispatched {
    run: gemm_f32_scalar,
    run_packed: gemm_f32_scalar_packed,
    run_fused: gemm_f32_scalar_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
const DISP_F64_SCALAR: Dispatched<f64> = Dispatched {
    run: gemm_f64_scalar,
    run_packed: gemm_f64_scalar_packed,
    run_fused: gemm_f64_scalar_fused,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_FMA: Dispatched<f32> = Dispatched {
    run: gemm_f32_fma,
    run_packed: gemm_f32_fma_packed,
    run_fused: gemm_f32_fma_fused,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_FMA: Dispatched<f64> = Dispatched {
    run: gemm_f64_fma,
    run_packed: gemm_f64_fma_packed,
    run_fused: gemm_f64_fma_fused,
    mr: 8,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_AVX512: Dispatched<f32> = Dispatched {
    run: gemm_f32_avx512,
    run_packed: gemm_f32_avx512_packed,
    run_fused: gemm_f32_avx512_fused,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_AVX512: Dispatched<f64> = Dispatched {
    run: gemm_f64_avx512,
    run_packed: gemm_f64_avx512_packed,
    run_fused: gemm_f64_avx512_fused,
    mr: 16,
    nr: 12,
    depth_multiple: 1,
};

#[cfg(target_arch = "aarch64")]
const DISP_F32_NEON: Dispatched<f32> = Dispatched {
    run: gemm_f32_neon,
    run_packed: gemm_f32_neon_packed,
    run_fused: gemm_f32_neon_fused,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(target_arch = "aarch64")]
const DISP_F64_NEON: Dispatched<f64> = Dispatched {
    run: gemm_f64_neon,
    run_packed: gemm_f64_neon_packed,
    run_fused: gemm_f64_neon_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F32_SIMD128: Dispatched<f32> = Dispatched {
    run: gemm_f32_simd128,
    run_packed: gemm_f32_simd128_packed,
    run_fused: gemm_f32_simd128_fused,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F64_SIMD128: Dispatched<f64> = Dispatched {
    run: gemm_f64_simd128,
    run_packed: gemm_f64_simd128_packed,
    run_fused: gemm_f64_simd128_fused,
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
        ForcedIsa::Neon => return DISP_F32_NEON, // NEON is baseline on aarch64
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
    // NEON is mandatory on aarch64
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F32_NEON
    }
    // `simd128` on wasm32, else scalar
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
        ForcedIsa::Neon => return DISP_F64_NEON, // NEON is baseline on aarch64
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
    // NEON is mandatory on aarch64
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F64_NEON
    }
    // `simd128` on wasm32, else scalar
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

/// Emit the `GemmScalar` impl for a **homogeneous float** type (`f32` / `f64`): `Out == Acc`
/// (`OUT_IS_ACC = true`), in-place `scale_c`, packing through `FloatGemm<$t>`, and the
/// always-present fused path. `$disp` is the memoized dispatch accessor; `$name` names the
/// type in the fused-kernel assert. The two float impls are pure type substitutions, so this
/// keeps them from drifting. (Narrow `f16`/`bf16` differ — narrow scale, `MixedGemm`, no
/// fused, bf16's depth-multiple pack switch — and stay manual below.)
macro_rules! float_gemm_scalar {
    ($t:ty, $disp:ident, $name:literal) => {
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
            unsafe fn pack_lhs_full(
                dst: *mut $t,
                a: *const $t,
                rsa: isize,
                csa: isize,
                m: usize,
                k: usize,
                kc: usize,
                nc: usize,
                nr: usize,
            ) {
                unsafe {
                    driver::pack_lhs_full::<FloatGemm<$t>>(dst, a, rsa, csa, m, k, kc, nc, nr)
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
            #[inline]
            unsafe fn dispatch_fused(
                t: Task<$t>,
                epi: FusedEpi<$t>,
                par: Parallelism,
                ws: &mut Workspace,
            ) {
                unsafe { ($disp().run_fused)(t, epi, par, ws) }
            }
            #[inline]
            fn rhs_tile() -> (usize, usize) {
                let d = $disp();
                (d.mr, d.nr)
            }
        }
    };
}

float_gemm_scalar!(f32, dispatched_f32, "f32");
float_gemm_scalar!(f64, dispatched_f64, "f64");
