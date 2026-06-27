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

/// Element types gemmkit can dispatch. Sealed in practice: only `f32` and `f64`
/// have a registered dispatch table in v1.
pub trait GemmScalar: Float<Acc = Self> {
    /// Run the dispatched kernel for this type. Used by the API layer.
    ///
    /// # Safety
    /// `task`'s pointers must be valid and `c` must not alias `a`/`b`.
    #[doc(hidden)]
    unsafe fn dispatch(task: Task<Self>, par: Parallelism, ws: &mut Workspace);
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

type GemmFn<T> = unsafe fn(Task<T>, Parallelism, &mut Workspace);

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

fn select_f32() -> GemmFn<f32> {
    match forced_isa() {
        ForcedIsa::Scalar => return gemm_f32_scalar,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return gemm_f32_fma;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return gemm_f32_avx512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return gemm_f32_neon, // NEON is baseline on aarch64
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return gemm_f32_avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return gemm_f32_fma;
        }
    }
    // NEON is mandatory on aarch64: it is always the Auto choice there, so the
    // scalar fallback below is gated out (it would be unreachable) on aarch64.
    #[cfg(target_arch = "aarch64")]
    {
        gemm_f32_neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemm_f32_scalar
    }
}

fn select_f64() -> GemmFn<f64> {
    match forced_isa() {
        ForcedIsa::Scalar => return gemm_f64_scalar,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return gemm_f64_fma;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512 => {
            assert!(
                is_x86_feature_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return gemm_f64_avx512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return gemm_f64_neon, // NEON is baseline on aarch64
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => {
            panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
        }
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return gemm_f64_avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return gemm_f64_fma;
        }
    }
    // NEON is mandatory on aarch64: it is always the Auto choice there, so the
    // scalar fallback below is gated out (it would be unreachable) on aarch64.
    #[cfg(target_arch = "aarch64")]
    {
        gemm_f64_neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemm_f64_scalar
    }
}

#[cfg(feature = "std")]
static GEMM_F32: OnceLock<GemmFn<f32>> = OnceLock::new();
#[cfg(feature = "std")]
static GEMM_F64: OnceLock<GemmFn<f64>> = OnceLock::new();

impl GemmScalar for f32 {
    #[inline]
    unsafe fn dispatch(task: Task<f32>, par: Parallelism, ws: &mut Workspace) {
        #[cfg(feature = "std")]
        let f = *GEMM_F32.get_or_init(select_f32);
        #[cfg(not(feature = "std"))]
        let f = select_f32();
        unsafe { f(task, par, ws) }
    }
}

impl GemmScalar for f64 {
    #[inline]
    unsafe fn dispatch(task: Task<f64>, par: Parallelism, ws: &mut Workspace) {
        #[cfg(feature = "std")]
        let f = *GEMM_F64.get_or_init(select_f64);
        #[cfg(not(feature = "std"))]
        let f = select_f64();
        unsafe { f(task, par, ws) }
    }
}
