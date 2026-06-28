//! Runtime ISA dispatch (layer L7).
//!
//! Each element type has one `OnceLock<fn>`: feature detection runs once, the
//! winning monomorphized entry point is cached, and later calls are a plain
//! indirect call. **No `transmute`, no `AtomicPtr<()>`** — the slot is a typed
//! function pointer. Adding an ISA is one line in the `select_*` ladder plus the
//! one-line `#[allow]`-free wrapper; adding a type is a new `OnceLock` + impl,
//! not a new crate.
//!
//! ## Pinning the kernel: `GEMMKIT_REQUIRE_ISA`
//!
//! By default the best available ISA is selected at runtime. Setting the
//! environment variable `GEMMKIT_REQUIRE_ISA` to `scalar`, `fma`, `avx512`, or
//! `neon` **forces** exactly that kernel; if the CPU (or an emulator such as
//! Intel SDE) does not report the required feature — or the requested ISA does
//! not exist on this target architecture — dispatch **panics** rather than
//! falling back, so a CI job that means to exercise a given kernel fails loudly
//! instead of silently testing a different one. (`neon` is only valid on
//! aarch64, where it is baseline; `fma`/`avx512` only on x86.) `auto`/unset is
//! the normal auto-selecting behavior. The value is read once (the choice is
//! memoized), so set it in the process environment before the first GEMM call.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use half::{bf16, f16};

use crate::driver;
use crate::kernel::{FloatGemm, IntGemm, MixedGemm};
use crate::parallel::Parallelism;
use crate::scalar::{Float, NarrowFloat, Scalar};
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::simd::{KernelSimd, ScalarTok, SimdOps};
use crate::special::gemv;
use crate::tuning;
use crate::workspace::Workspace;

/// A fully described GEMM problem (`C <- alpha·A·B + beta·C`) with raw pointers
/// and `isize` strides. This is the homogeneous-type dispatch boundary.
#[derive(Copy, Clone)]
pub struct Task<T> {
    /// Rows of A and C.
    pub m: usize,
    /// Shared dimension (cols of A, rows of B).
    pub k: usize,
    /// Cols of B and C.
    pub n: usize,
    /// Product scale.
    pub alpha: T,
    /// LHS base pointer (element `(0,0)`).
    pub a: *const T,
    /// LHS row / column strides.
    pub rsa: isize,
    pub csa: isize,
    /// RHS base pointer.
    pub b: *const T,
    /// RHS row / column strides.
    pub rsb: isize,
    pub csb: isize,
    /// Accumulator scale.
    pub beta: T,
    /// Output base pointer.
    pub c: *mut T,
    /// Output row / column strides.
    pub rsc: isize,
    pub csc: isize,
}

/// A GEMM whose RHS is already prepacked: `C <- alpha·A·(prepacked B) + beta·C`.
/// Carries the blocking geometry the buffer was packed for (`nr`, `kc`, `nc`),
/// which the driver reads back verbatim so a reused panel always matches its
/// tiling.
///
/// `pub` (like [`Task`]) only so it can appear in the doc-hidden [`GemmScalar`]
/// methods; the `dispatch` module is private, so it is not nameable externally.
pub struct PackedConsume<T> {
    /// Rows of A and C.
    pub m: usize,
    /// Shared dimension (cols of A == prepacked B's depth).
    pub k: usize,
    /// Cols of the prepacked B and of C.
    pub n: usize,
    /// Product scale.
    pub alpha: T,
    /// LHS base pointer + strides.
    pub a: *const T,
    pub rsa: isize,
    pub csa: isize,
    /// Prepacked RHS micropanel buffer base (see [`crate::driver::pack_rhs_full`]).
    pub packed: *const T,
    /// Blocking geometry baked into `packed` at pack time.
    pub nr: usize,
    pub kc: usize,
    pub nc: usize,
    /// Accumulator scale.
    pub beta: T,
    /// Output base pointer + strides.
    pub c: *mut T,
    pub rsc: isize,
    pub csc: isize,
}

/// A heterogeneous **integer** GEMM problem: `i8` inputs, `i32` accumulator/output
/// (`C <- alpha·A·B + beta·C`, all of `alpha`/`beta`/`C` in `i32`). The homogeneous
/// [`Task`] / [`GemmScalar`] machinery assumes `Lhs = Out`, which integer GEMM
/// breaks (`Out = i32 != Lhs = i8`), so it gets this dedicated task + dispatch.
#[derive(Copy, Clone)]
pub(crate) struct IntTask {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub alpha: i32,
    pub a: *const i8,
    pub rsa: isize,
    pub csa: isize,
    pub b: *const i8,
    pub rsb: isize,
    pub csb: isize,
    pub beta: i32,
    pub c: *mut i32,
    pub rsc: isize,
    pub csc: isize,
}

/// Element types gemmkit can dispatch. Sealed in practice: `f32`/`f64` (the
/// homogeneous float family) and `f16`/`bf16` (the mixed-precision family,
/// `Acc = f32`) have registered dispatch tables.
///
/// The bound is just [`Scalar`] — **not** `Float<Acc = Self>` — so the accumulator
/// may differ from the element type (the mixed-precision seam). Everything a type's
/// kernel family needs that is *not* expressible generically (its degenerate
/// `beta`-scale, which is `f32`-mediated for the narrow types; and which family to
/// pack/dispatch through) is supplied by the methods below, so the driver and the
/// public API stay entirely type-agnostic.
pub trait GemmScalar: Scalar {
    /// Mirror of [`crate::kernel::KernelFamily::OUT_IS_ACC`] for this type's family:
    /// `true` for `f32`/`f64` (homogeneous), `false` for `f16`/`bf16` (mixed). The
    /// prepack constructor reads it to compute the same `kc` the driver will use, so
    /// the prepacked and plain paths block identically.
    const OUT_IS_ACC: bool;

    /// `C <- beta·C` over the strided output — the degenerate path when the `A·B`
    /// term vanishes (`k == 0` or `alpha == 0`). For the narrow types this scales in
    /// `f32` and rounds back.
    ///
    /// # Safety
    /// `c` valid for the `m × n` region at `rsc`/`csc`.
    #[doc(hidden)]
    unsafe fn scale_c(beta: Self, c: *mut Self, m: usize, n: usize, rsc: isize, csc: isize);

    /// Pack a full RHS into the prepacked micropanel buffer through this type's
    /// kernel family. The layout is family-independent (plain micropanels), but the
    /// family *type* differs (`FloatGemm` vs `MixedGemm`), so the call is dispatched
    /// here rather than hard-wired in [`crate::prepack_rhs`].
    ///
    /// # Safety
    /// As [`crate::driver::pack_rhs_full`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_rhs_full(
        dst: *mut Self,
        b: *const Self,
        rsb: isize,
        csb: isize,
        k: usize,
        n: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    );

    /// Pack a full LHS (in the transposed-RHS layout) for the prepacked-LHS path,
    /// through this type's kernel family. Mirror of [`GemmScalar::pack_rhs_full`].
    ///
    /// # Safety
    /// As [`crate::driver::pack_lhs_full`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn pack_lhs_full(
        dst: *mut Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        m: usize,
        k: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    );

    /// Run the dispatched kernel for this type. Used by the API layer.
    ///
    /// # Safety
    /// `task`'s pointers must be valid and `c` must not alias `a`/`b`.
    #[doc(hidden)]
    unsafe fn dispatch(task: Task<Self>, par: Parallelism, ws: &mut Workspace);

    /// Run the dispatched prepacked-RHS kernel for this type.
    ///
    /// # Safety
    /// `req`'s pointers must be valid, `c` must not alias `a`/`packed`, and
    /// `packed` must have been produced by [`GemmScalar::pack_rhs_full`] for the
    /// geometry recorded in `req`.
    #[doc(hidden)]
    unsafe fn dispatch_packed(req: PackedConsume<Self>, par: Parallelism, ws: &mut Workspace);

    /// The selected kernel's microtile `(mr, nr)` = `(MR_REG·LANES, NR)`. Used by
    /// the prepack constructor to compute the buffer's blocking geometry through
    /// the *same* ISA choice the consuming call will make.
    #[doc(hidden)]
    fn rhs_tile() -> (usize, usize);
}

/// Top-level entry used by the API layer: handle the degenerate cases (here,
/// where the element type is concrete) and then run the ISA-dispatched kernel.
///
/// # Safety
/// `task`'s pointers must be valid for the implied regions and `c` must not
/// alias `a`/`b`.
pub(crate) unsafe fn execute<T: GemmScalar>(task: Task<T>, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if task.m == 0 || task.n == 0 {
            return;
        }
        // k == 0 or alpha == 0 ⇒ the A·B term vanishes: C <- beta·C only.
        if task.k == 0 || task.alpha == T::ZERO {
            T::scale_c(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
            return;
        }
        T::dispatch(task, par, ws);
    }
}

/// Top-level entry for the prepacked-RHS path: handle the degenerate cases
/// (the A·B term vanishes ⇒ `C <- beta·C`, never touching the packed buffer) and
/// then run the ISA-dispatched prepacked kernel.
///
/// # Safety
/// As [`execute`], plus `req.packed` valid for the recorded geometry and not
/// aliasing `c`.
pub(crate) unsafe fn execute_packed<T: GemmScalar>(
    req: PackedConsume<T>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if req.m == 0 || req.n == 0 {
            return;
        }
        if req.k == 0 || req.alpha == T::ZERO {
            T::scale_c(req.beta, req.c, req.m, req.n, req.rsc, req.csc);
            return;
        }
        T::dispatch_packed(req, par, ws);
    }
}

/// Orientation normalization shared by the float / mixed [`Task`] paths: if `C` is
/// row-major-ish (`|csc| < |rsc|`), compute `Cᵀ = Bᵀ·Aᵀ` instead so the kernel writes
/// columns contiguously (`rsc == 1`). Swaps `m↔n`, the `A`/`B` pointers and strides,
/// and `rsc↔csc`. (The integer [`IntTask`] is a distinct struct and inlines the same
/// swap.)
#[inline]
fn orient_transpose<T>(t: &mut Task<T>) {
    if t.csc.unsigned_abs() < t.rsc.unsigned_abs() {
        let (oa, orsa, ocsa) = (t.a, t.rsa, t.csa);
        let (ob, orsb, ocsb) = (t.b, t.rsb, t.csb);
        core::mem::swap(&mut t.m, &mut t.n);
        t.a = ob;
        t.rsa = ocsb;
        t.csa = orsb;
        t.b = oa;
        t.rsb = ocsa;
        t.csb = orsa;
        core::mem::swap(&mut t.rsc, &mut t.csc);
    }
}

/// `C <- beta·C` for a **homogeneous float** type (`f32`/`f64`): in-place scale,
/// with `beta == 0` overwriting to zero without reading C. The `GemmScalar::scale_c`
/// for the float types forwards here; the narrow types use [`scale_c_narrow`].
unsafe fn scale_c_float<T: Float>(beta: T, c: *mut T, m: usize, n: usize, rsc: isize, csc: isize) {
    unsafe {
        for j in 0..n {
            for i in 0..m {
                let p = c.offset(i as isize * rsc + j as isize * csc);
                if beta == T::ZERO {
                    *p = T::ZERO;
                } else if beta != T::ONE {
                    *p = beta * *p;
                }
            }
        }
    }
}

/// `C <- beta·C` for a **narrow** type (`f16`/`bf16`): widen each element to `f32`,
/// scale, and round back. Matches the mixed kernel's epilogue precision.
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
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc,
            );
            return;
        }

        orient_transpose(&mut t);
        driver::run::<FloatGemm<T>, S, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, par, ws,
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

/// Mixed-precision driver entry for a concrete `(narrow type, ISA, tile)`. Mirror
/// of [`run_typed`] but driving [`MixedGemm`]: no gemv special path (the general
/// driver is correct for those shapes; narrow gemv is a deferred optimization), the
/// same orientation swap, and `alpha`/`beta` **widened to the `f32` accumulator**
/// before the driver call.
///
/// # Safety
/// As [`run_typed`].
#[inline]
unsafe fn run_typed_mixed<N, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        orient_transpose(&mut t);
        driver::run::<MixedGemm<N>, S, MR_REG, NR>(
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

/// Prepacked-RHS mixed-precision entry (mirror of [`run_packed_typed`] for
/// [`MixedGemm`]); no swap, `alpha`/`beta` widened to `f32`.
///
/// # Safety
/// As [`run_packed_typed`].
#[inline]
unsafe fn run_packed_typed_mixed<N, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<MixedGemm<N>, S, MR_REG, NR>(
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
    // tile as f32; ~54 vs ~43 GFLOP/s for 3×8).
    unsafe { run_typed::<f64, Neon, 4, 4>(Neon, t, par, ws) }
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

// ---- mixed-precision (f16 / bf16) entry points: same tiles as f32 (the
// accumulator is f32, so the register budget matches) ----

unsafe fn gemm_f16_scalar(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
unsafe fn gemm_bf16_scalar(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
unsafe fn gemm_f16_scalar_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
unsafe fn gemm_bf16_scalar_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f16_fma(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator → MR = 2*8 = 16, NR = 6 (the f32 FMA tile).
    unsafe { run_typed_mixed::<f16, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_bf16_fma(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f16_fma_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_bf16_fma_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f16_avx512(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile).
    unsafe { run_typed_mixed::<f16, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_bf16_avx512(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_f16_avx512_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_bf16_avx512_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f16_neon(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_bf16_neon(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_f16_neon_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_bf16_neon_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, Neon, 4, 4>(Neon, r, par, ws) }
}

type GemmFn<T> = unsafe fn(Task<T>, Parallelism, &mut Workspace);
type PackedFn<T> = unsafe fn(PackedConsume<T>, Parallelism, &mut Workspace);

/// The memoized dispatch slot for one element type: the standard kernel, the
/// prepacked-RHS kernel, and the microtile `(mr, nr)` they share. Bundling them
/// keeps adding an ISA a single `select_*` ladder arm. `mr`/`nr` mirror the tile
/// constants in the wrappers above and feed `prepack_rhs` (via `rhs_tile`) so the
/// buffer and the consume path agree on the blocking geometry.
#[derive(Copy, Clone)]
struct Dispatched<T> {
    run: GemmFn<T>,
    run_packed: PackedFn<T>,
    mr: usize,
    nr: usize,
}

// One descriptor per (type, ISA). `mr = MR_REG·LANES`, `nr = NR` — mirrors the
// tile in each wrapper's comment (scalar 4×4; FMA 16×6 / f64 8×6; AVX-512 32×12 /
// f64 16×12; NEON 16×4 / f64 8×4).
const DISP_F32_SCALAR: Dispatched<f32> = Dispatched {
    run: gemm_f32_scalar,
    run_packed: gemm_f32_scalar_packed,
    mr: 4,
    nr: 4,
};
const DISP_F64_SCALAR: Dispatched<f64> = Dispatched {
    run: gemm_f64_scalar,
    run_packed: gemm_f64_scalar_packed,
    mr: 4,
    nr: 4,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_FMA: Dispatched<f32> = Dispatched {
    run: gemm_f32_fma,
    run_packed: gemm_f32_fma_packed,
    mr: 16,
    nr: 6,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_FMA: Dispatched<f64> = Dispatched {
    run: gemm_f64_fma,
    run_packed: gemm_f64_fma_packed,
    mr: 8,
    nr: 6,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_AVX512: Dispatched<f32> = Dispatched {
    run: gemm_f32_avx512,
    run_packed: gemm_f32_avx512_packed,
    mr: 32,
    nr: 12,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_AVX512: Dispatched<f64> = Dispatched {
    run: gemm_f64_avx512,
    run_packed: gemm_f64_avx512_packed,
    mr: 16,
    nr: 12,
};
#[cfg(target_arch = "aarch64")]
const DISP_F32_NEON: Dispatched<f32> = Dispatched {
    run: gemm_f32_neon,
    run_packed: gemm_f32_neon_packed,
    mr: 16,
    nr: 4,
};
#[cfg(target_arch = "aarch64")]
const DISP_F64_NEON: Dispatched<f64> = Dispatched {
    run: gemm_f64_neon,
    run_packed: gemm_f64_neon_packed,
    mr: 8,
    nr: 4,
};

// Mixed-precision descriptors. `mr = MR_REG · f32-LANES` (the accumulator width):
// scalar 4×4, FMA 16×6, AVX-512 32×12, NEON 16×4 — the same tiles as f32.
const DISP_F16_SCALAR: Dispatched<f16> = Dispatched {
    run: gemm_f16_scalar,
    run_packed: gemm_f16_scalar_packed,
    mr: 4,
    nr: 4,
};
const DISP_BF16_SCALAR: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_scalar,
    run_packed: gemm_bf16_scalar_packed,
    mr: 4,
    nr: 4,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F16_FMA: Dispatched<f16> = Dispatched {
    run: gemm_f16_fma,
    run_packed: gemm_f16_fma_packed,
    mr: 16,
    nr: 6,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_BF16_FMA: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_fma,
    run_packed: gemm_bf16_fma_packed,
    mr: 16,
    nr: 6,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F16_AVX512: Dispatched<f16> = Dispatched {
    run: gemm_f16_avx512,
    run_packed: gemm_f16_avx512_packed,
    mr: 32,
    nr: 12,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_BF16_AVX512: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512,
    run_packed: gemm_bf16_avx512_packed,
    mr: 32,
    nr: 12,
};
#[cfg(target_arch = "aarch64")]
const DISP_F16_NEON: Dispatched<f16> = Dispatched {
    run: gemm_f16_neon,
    run_packed: gemm_f16_neon_packed,
    mr: 16,
    nr: 4,
};
#[cfg(target_arch = "aarch64")]
const DISP_BF16_NEON: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_neon,
    run_packed: gemm_bf16_neon_packed,
    mr: 16,
    nr: 4,
};

/// An explicitly requested kernel, parsed from `GEMMKIT_REQUIRE_ISA`.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ForcedIsa {
    /// No override: auto-select the best available ISA (the default).
    Auto,
    Scalar,
    Fma,
    Avx512,
    Neon,
}

/// Parse the `GEMMKIT_REQUIRE_ISA` pin. Unset/empty ⇒ [`ForcedIsa::Auto`]; an
/// unrecognized value is a hard error (catches typos in CI config). Read once,
/// since the selection is memoized in the per-type `OnceLock`.
#[cfg(feature = "std")]
fn forced_isa() -> ForcedIsa {
    match std::env::var("GEMMKIT_REQUIRE_ISA") {
        Err(_) => ForcedIsa::Auto,
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("auto") {
                ForcedIsa::Auto
            } else if t.eq_ignore_ascii_case("scalar") {
                ForcedIsa::Scalar
            } else if t.eq_ignore_ascii_case("fma") || t.eq_ignore_ascii_case("avx2") {
                ForcedIsa::Fma
            } else if t.eq_ignore_ascii_case("avx512") || t.eq_ignore_ascii_case("avx512f") {
                ForcedIsa::Avx512
            } else if t.eq_ignore_ascii_case("neon") {
                ForcedIsa::Neon
            } else {
                panic!(
                    "GEMMKIT_REQUIRE_ISA: unknown value `{t}` (expected scalar|fma|avx512|neon|auto)"
                )
            }
        }
    }
}
#[cfg(not(feature = "std"))]
fn forced_isa() -> ForcedIsa {
    ForcedIsa::Auto
}

fn select_f32() -> Dispatched<f32> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F32_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_F32_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F32_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F32_NEON, // NEON is baseline on aarch64
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return DISP_F32_AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DISP_F32_FMA;
        }
    }
    // NEON is mandatory on aarch64: it is always the Auto choice there, so the
    // scalar fallback below is gated out (it would be unreachable) on aarch64.
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F32_NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
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
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_F64_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F64_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F64_NEON, // NEON is baseline on aarch64
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return DISP_F64_AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DISP_F64_FMA;
        }
    }
    // NEON is mandatory on aarch64: it is always the Auto choice there, so the
    // scalar fallback below is gated out (it would be unreachable) on aarch64.
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F64_NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        DISP_F64_SCALAR
    }
}

/// `f16` ISA selection. The mixed-precision FMA path additionally needs **F16C**
/// (`vcvtph2ps`/`vcvtps2ph`) — universal on AVX2+FMA hardware but checked here so a
/// forced or auto FMA selection on a (hypothetical) F16C-less part falls back rather
/// than faulting. AVX-512 covers `f16` within `avx512f`.
fn select_f16() -> Dispatched<f16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma")
                    && is_x86_feature_detected!("f16c"),
                "GEMMKIT_REQUIRE_ISA=fma for f16, but this CPU does not report avx2+fma+f16c"
            );
            return DISP_F16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F16_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return DISP_F16_AVX512;
        }
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return DISP_F16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F16_NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        DISP_F16_SCALAR
    }
}

/// `bf16` ISA selection. The FMA path uses only AVX2 integer ops (shift / pack), so
/// no F16C is required; AVX-512 covers `bf16` within `avx512f`.
fn select_bf16() -> Dispatched<bf16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_BF16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_BF16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_BF16_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_BF16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return DISP_BF16_AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DISP_BF16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_BF16_NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        DISP_BF16_SCALAR
    }
}

#[cfg(feature = "std")]
static GEMM_F32: OnceLock<Dispatched<f32>> = OnceLock::new();
#[cfg(feature = "std")]
static GEMM_F64: OnceLock<Dispatched<f64>> = OnceLock::new();
#[cfg(feature = "std")]
static GEMM_F16: OnceLock<Dispatched<f16>> = OnceLock::new();
#[cfg(feature = "std")]
static GEMM_BF16: OnceLock<Dispatched<bf16>> = OnceLock::new();

/// The memoized dispatch descriptor for `f32` (selection runs once).
#[inline]
fn dispatched_f32() -> Dispatched<f32> {
    #[cfg(feature = "std")]
    {
        *GEMM_F32.get_or_init(select_f32)
    }
    #[cfg(not(feature = "std"))]
    {
        select_f32()
    }
}

/// The memoized dispatch descriptor for `f64` (selection runs once).
#[inline]
fn dispatched_f64() -> Dispatched<f64> {
    #[cfg(feature = "std")]
    {
        *GEMM_F64.get_or_init(select_f64)
    }
    #[cfg(not(feature = "std"))]
    {
        select_f64()
    }
}

/// The memoized dispatch descriptor for `f16` (selection runs once).
#[inline]
fn dispatched_f16() -> Dispatched<f16> {
    #[cfg(feature = "std")]
    {
        *GEMM_F16.get_or_init(select_f16)
    }
    #[cfg(not(feature = "std"))]
    {
        select_f16()
    }
}

/// The memoized dispatch descriptor for `bf16` (selection runs once).
#[inline]
fn dispatched_bf16() -> Dispatched<bf16> {
    #[cfg(feature = "std")]
    {
        *GEMM_BF16.get_or_init(select_bf16)
    }
    #[cfg(not(feature = "std"))]
    {
        select_bf16()
    }
}

impl GemmScalar for f32 {
    const OUT_IS_ACC: bool = true;
    #[inline]
    unsafe fn scale_c(beta: f32, c: *mut f32, m: usize, n: usize, rsc: isize, csc: isize) {
        unsafe { scale_c_float(beta, c, m, n, rsc, csc) }
    }
    #[inline]
    unsafe fn pack_rhs_full(
        dst: *mut f32,
        b: *const f32,
        rsb: isize,
        csb: isize,
        k: usize,
        n: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_rhs_full::<FloatGemm<f32>>(dst, b, rsb, csb, k, n, kc, nc, nr) }
    }
    #[inline]
    unsafe fn pack_lhs_full(
        dst: *mut f32,
        a: *const f32,
        rsa: isize,
        csa: isize,
        m: usize,
        k: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_lhs_full::<FloatGemm<f32>>(dst, a, rsa, csa, m, k, kc, nc, nr) }
    }
    #[inline]
    unsafe fn dispatch(task: Task<f32>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f32().run)(task, par, ws) }
    }
    #[inline]
    unsafe fn dispatch_packed(req: PackedConsume<f32>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f32().run_packed)(req, par, ws) }
    }
    #[inline]
    fn rhs_tile() -> (usize, usize) {
        let d = dispatched_f32();
        (d.mr, d.nr)
    }
}

impl GemmScalar for f64 {
    const OUT_IS_ACC: bool = true;
    #[inline]
    unsafe fn scale_c(beta: f64, c: *mut f64, m: usize, n: usize, rsc: isize, csc: isize) {
        unsafe { scale_c_float(beta, c, m, n, rsc, csc) }
    }
    #[inline]
    unsafe fn pack_rhs_full(
        dst: *mut f64,
        b: *const f64,
        rsb: isize,
        csb: isize,
        k: usize,
        n: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_rhs_full::<FloatGemm<f64>>(dst, b, rsb, csb, k, n, kc, nc, nr) }
    }
    #[inline]
    unsafe fn pack_lhs_full(
        dst: *mut f64,
        a: *const f64,
        rsa: isize,
        csa: isize,
        m: usize,
        k: usize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { driver::pack_lhs_full::<FloatGemm<f64>>(dst, a, rsa, csa, m, k, kc, nc, nr) }
    }
    #[inline]
    unsafe fn dispatch(task: Task<f64>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f64().run)(task, par, ws) }
    }
    #[inline]
    unsafe fn dispatch_packed(req: PackedConsume<f64>, par: Parallelism, ws: &mut Workspace) {
        unsafe { (dispatched_f64().run_packed)(req, par, ws) }
    }
    #[inline]
    fn rhs_tile() -> (usize, usize) {
        let d = dispatched_f64();
        (d.mr, d.nr)
    }
}

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
        unsafe { driver::pack_rhs_full::<MixedGemm<bf16>>(dst, b, rsb, csb, k, n, kc, nc, nr) }
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
        unsafe { driver::pack_lhs_full::<MixedGemm<bf16>>(dst, a, rsa, csa, m, k, kc, nc, nr) }
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
}

// ===========================================================================
// Integer GEMM (i8 -> i32): a dedicated heterogeneous dispatch path, since the
// homogeneous `GemmScalar` cannot express `Out != Lhs`.
// ===========================================================================

/// Top-level integer entry: degenerate cases (`C <- beta·C` when the `A·B` term
/// vanishes) then the ISA-dispatched integer kernel.
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` must not alias `a`/`b`.
pub(crate) unsafe fn execute_int(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        if t.k == 0 || t.alpha == 0 {
            scale_c_int(t.beta, t.c, t.m, t.n, t.rsc, t.csc);
            return;
        }
        (dispatched_i8().run)(t, par, ws);
    }
}

/// `C <- beta·C` for the integer output (wrapping i32; `beta == 0` overwrites to 0).
unsafe fn scale_c_int(beta: i32, c: *mut i32, m: usize, n: usize, rsc: isize, csc: isize) {
    unsafe {
        for j in 0..n {
            for i in 0..m {
                let p = c.offset(i as isize * rsc + j as isize * csc);
                if beta == 0 {
                    *p = 0;
                } else if beta != 1 {
                    *p = beta.wrapping_mul(*p);
                }
            }
        }
    }
}

/// Integer driver entry for a concrete `(ISA, tile)`: gemv shapes fall through the
/// general driver (a dedicated integer gemv is deferred), then the orientation swap
/// (identical to the float path — only strides move) and `driver::run::<IntGemm>`.
///
/// # Safety
/// As [`execute_int`].
#[inline]
unsafe fn run_typed_int<S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: IntTask,
    par: Parallelism,
    ws: &mut Workspace,
) where
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        if t.csc.unsigned_abs() < t.rsc.unsigned_abs() {
            let (oa, orsa, ocsa) = (t.a, t.rsa, t.csa);
            let (ob, orsb, ocsb) = (t.b, t.rsb, t.csb);
            core::mem::swap(&mut t.m, &mut t.n);
            t.a = ob;
            t.rsa = ocsb;
            t.csa = orsb;
            t.b = oa;
            t.rsb = ocsa;
            t.csb = orsa;
            core::mem::swap(&mut t.rsc, &mut t.csc);
        }
        driver::run::<IntGemm, S, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, par, ws,
        );
    }
}

unsafe fn gemm_i8_scalar(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_i8_fma(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 accumulator → MR = 2*8 = 16, NR = 6 (the f32 FMA tile).
    unsafe { run_typed_int::<Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gemm_i8_avx512(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile).
    unsafe { run_typed_int::<Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_i8_neon(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<Neon, 4, 4>(Neon, t, par, ws) }
}

type IntFn = unsafe fn(IntTask, Parallelism, &mut Workspace);

/// Memoized integer dispatch slot (mirror of [`Dispatched`] but a single kernel —
/// integer prepack is not yet a public API).
#[derive(Copy, Clone)]
struct IntDispatched {
    run: IntFn,
}

const DISP_I8_SCALAR: IntDispatched = IntDispatched {
    run: gemm_i8_scalar,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_I8_FMA: IntDispatched = IntDispatched { run: gemm_i8_fma };
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_I8_AVX512: IntDispatched = IntDispatched {
    run: gemm_i8_avx512,
};
#[cfg(target_arch = "aarch64")]
const DISP_I8_NEON: IntDispatched = IntDispatched { run: gemm_i8_neon };

/// `i8` ISA selection. The widen-and-multiply integer kernel uses only AVX2/AVX-512
/// integer ops (no VNNI), so the gates mirror the `f32` ladder.
fn select_i8() -> IntDispatched {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_I8_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_I8_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_I8_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_I8_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return DISP_I8_AVX512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return DISP_I8_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_I8_NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        DISP_I8_SCALAR
    }
}

#[cfg(feature = "std")]
static GEMM_I8: OnceLock<IntDispatched> = OnceLock::new();

#[inline]
fn dispatched_i8() -> IntDispatched {
    #[cfg(feature = "std")]
    {
        *GEMM_I8.get_or_init(select_i8)
    }
    #[cfg(not(feature = "std"))]
    {
        select_i8()
    }
}
