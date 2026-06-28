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

use crate::driver;
use crate::kernel::FloatGemm;
use crate::parallel::Parallelism;
use crate::scalar::Float;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::simd::{ScalarTok, SimdOps};
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

/// A GEMM whose RHS is already prepacked (the reuse path, B2):
/// `C <- alpha·A·(prepacked B) + beta·C`. Carries the blocking geometry the
/// buffer was packed for (`nr`, `kc`, `nc`), which the consume path re-derives at
/// the real `m` and asserts matches — so a reused panel can never be read against
/// a different tiling.
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

/// Element types gemmkit can dispatch. Sealed in practice: only `f32` and `f64`
/// have a registered dispatch table in v1.
pub trait GemmScalar: Float<Acc = Self> {
    /// Run the dispatched kernel for this type. Used by the API layer.
    ///
    /// # Safety
    /// `task`'s pointers must be valid and `c` must not alias `a`/`b`.
    #[doc(hidden)]
    unsafe fn dispatch(task: Task<Self>, par: Parallelism, ws: &mut Workspace);

    /// Run the dispatched **prepacked-RHS** kernel for this type (B2).
    ///
    /// # Safety
    /// `req`'s pointers must be valid, `c` must not alias `a`/`packed`, and
    /// `packed` must have been produced by [`crate::driver::pack_rhs_full`] for
    /// the geometry recorded in `req`.
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
            scale_c::<T>(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
            return;
        }
        T::dispatch(task, par, ws);
    }
}

/// Top-level entry for the prepacked-RHS path (B2): handle the degenerate cases
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
            scale_c::<T>(req.beta, req.c, req.m, req.n, req.rsc, req.csc);
            return;
        }
        T::dispatch_packed(req, par, ws);
    }
}

/// `C <- beta·C` (with `beta == 0` overwriting to zero without reading C).
unsafe fn scale_c<T: Float>(beta: T, c: *mut T, m: usize, n: usize, rsc: isize, csc: isize) {
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

        // Orientation: if C is row-major-ish (|csc| < |rsc|), compute Cᵀ = Bᵀ·Aᵀ
        // so the kernel writes columns contiguously (rsc == 1).
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

        driver::run::<FloatGemm<T>, S, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, par, ws,
        );
    }
}

/// Prepacked-RHS driver entry for a concrete `(type, ISA, tile)` (B2). No gemv
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
        // The buffer carries the blocking geometry (`kc`, `nc`) it was packed for;
        // the driver reads panels with exactly that geometry, so nothing is
        // re-derived and the panel addresses always match the buffer regardless of
        // how this `m` would otherwise block. `nr` is structural (the panel width
        // is this kernel's `NR`); a single process's memoized ISA choice guarantees
        // agreement, asserted in debug builds.
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
    // tile as f32; ~54 vs ~43 GFLOP/s for 3×8).
    unsafe { run_typed::<f64, Neon, 4, 4>(Neon, t, par, ws) }
}

// ---- prepacked-RHS (B2) entry points: one per (type, ISA), same tiles ----

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

type GemmFn<T> = unsafe fn(Task<T>, Parallelism, &mut Workspace);
type PackedFn<T> = unsafe fn(PackedConsume<T>, Parallelism, &mut Workspace);

/// The memoized dispatch slot for one element type: the standard kernel, the
/// prepacked-RHS kernel, and the microtile `(mr, nr)` they share. Bundling them
/// keeps adding an ISA a single `select_*` ladder arm. `mr`/`nr` mirror the tile
/// constants in the wrappers above; the consume path re-derives `mr` from
/// `MR_REG·LANES` and asserts the resulting blocking matches, so a stale literal
/// fails loudly rather than silently.
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

#[cfg(feature = "std")]
static GEMM_F32: OnceLock<Dispatched<f32>> = OnceLock::new();
#[cfg(feature = "std")]
static GEMM_F64: OnceLock<Dispatched<f64>> = OnceLock::new();

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

impl GemmScalar for f32 {
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
