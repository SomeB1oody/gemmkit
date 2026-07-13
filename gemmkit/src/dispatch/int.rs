//! Integer GEMM dispatch: `i8 -> i32` (`IntTask`) and the fused `i8 -> i8`
//! requantizing path (`RequantTask`), with their per-ISA wrappers, descriptors,
//! and selection ladders.

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::orient_swap;
use crate::driver;
use crate::kernel::KernelFamily;
use crate::kernel::epilogue::{BiasDim, Epilogue, KRequantize};
use crate::kernel::{IntGemm, IntGemmQ};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::{IntGemmVnni, IntGemmVnniQ};
use crate::parallel::Parallelism;
use crate::parallel::Ptr;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::Avx512Vnni;
use crate::simd::KernelSimd;
#[cfg(target_arch = "aarch64")]
use crate::simd::Neon;
use crate::simd::ScalarTok;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::simd::{Avx512, Fma};
use crate::special::small_k;
use crate::tuning;
use crate::workspace::Workspace;

/// A heterogeneous **integer** GEMM problem: `i8` inputs, `i32` accumulator/output
/// (all of `alpha`/`beta`/`C` in `i32`). The homogeneous [`Task`] / [`GemmScalar`]
/// machinery assumes `Lhs = Out`, which `i8 -> i32` breaks, so integer GEMM gets
/// this dedicated task + dispatch.
#[cfg(feature = "int8")]
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

// ===========================================================================
// Integer GEMM (i8 -> i32): a dedicated heterogeneous dispatch path, since the
// homogeneous `GemmScalar` cannot express `Out != Lhs`.
// ===========================================================================

/// Pick the integer kernel fn for this problem, shared by the plain and requantizing
/// entries (`F` is `IntFn` / `RequantFn`, both `Copy` fn pointers). Auto VNNI hands *small
/// multi-threaded* problems to the widen fallback — the dot kernel's mandatory pack barrier
/// dominates there — while `Rayon(1)`/`Serial` keep VNNI at any size; `small_par_fallback`
/// is `None` for every non-VNNI kernel, so `run` is returned unchanged. Centralizing the
/// `I8_VNNI_MIN_PAR_MNK` gate keeps the two paths' calibration from drifting apart.
#[cfg(feature = "int8")]
#[inline]
fn pick_int_kernel<F: Copy>(
    par: Parallelism,
    mnk: usize,
    run: F,
    small_par_fallback: Option<F>,
) -> F {
    match small_par_fallback {
        Some(fallback)
            if matches!(par, Parallelism::Rayon(n) if n != 1)
                && mnk < tuning::i8_vnni_min_par_mnk() =>
        {
            fallback
        }
        _ => run,
    }
}

/// Top-level integer entry: degenerate cases (`C <- beta·C` when the `A·B` term
/// vanishes) then the ISA-dispatched integer kernel.
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` must not alias `a`/`b`.
#[cfg(feature = "int8")]
pub(crate) unsafe fn execute_int(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        if t.k == 0 || t.alpha == 0 {
            scale_c_int(t.beta, t.c, t.m, t.n, t.rsc, t.csc);
            return;
        }
        let d = dispatched_i8();
        let mnk = t.m.saturating_mul(t.n).saturating_mul(t.k);
        let run = pick_int_kernel(par, mnk, d.run, d.small_par_fallback);
        run(t, par, ws);
    }
}

/// `C <- beta·C` for the integer output (wrapping i32; `beta == 0` overwrites to 0).
#[cfg(feature = "int8")]
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
#[cfg(feature = "int8")]
#[inline]
unsafe fn run_typed_int<Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: IntTask,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily<Lhs = i8, Rhs = i8, Acc = i32, Out = i32>,
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        orient_swap(
            &mut t.m, &mut t.n, &mut t.a, &mut t.rsa, &mut t.csa, &mut t.b, &mut t.rsb, &mut t.csb,
            &mut t.rsc, &mut t.csc,
        );
        // Skinny / low-depth shape: route through the widen `IntGemm` (never `IntGemmVnni`) —
        // at tiny `k` VNNI's mandatory quad-pack barrier never amortizes. Stays bit-exact
        // (i32 modular), so it reproduces the widen and VNNI results alike.
        if t.k <= tuning::small_k_threshold() {
            small_k::run::<IntGemm, S, MR_REG, NR>(
                simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c,
                t.rsc, t.csc, par, ws,
            );
            return;
        }
        driver::run::<Fam, S, MR_REG, NR>(
            simd, t.m, t.k, t.n, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, t.beta, t.c, t.rsc,
            t.csc, par, ws,
        );
    }
}

#[cfg(feature = "int8")]
unsafe fn gemm_i8_scalar(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<IntGemm, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_fma(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 accumulator → MR = 2*8 = 16, NR = 6 (the f32 FMA tile).
    unsafe { run_typed_int::<IntGemm, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile).
    unsafe { run_typed_int::<IntGemm, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512vnni(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // VNNI dot kernel, same tile as AVX-512: MR = 2*16 = 32, NR = 12 → 24 acc + 2 vA
    // + 1 vB = 27 ZMM. `vpdpbusd` folds 4 depth steps × 16 lanes per instruction.
    unsafe { run_typed_int::<IntGemmVnni, Avx512Vnni, 2, 12>(Avx512Vnni, t, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
unsafe fn gemm_i8_neon(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<IntGemm, Neon, 4, 4>(Neon, t, par, ws) }
}
// wasm simd128 i8
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_i8_simd128(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<IntGemm, Simd128, 2, 4>(Simd128, t, par, ws) }
}

#[cfg(feature = "int8")]
type IntFn = unsafe fn(IntTask, Parallelism, &mut Workspace);

/// Memoized integer dispatch slot (mirror of [`Dispatched`] but a single kernel —
/// integer prepack is not yet a public API).
///
/// `small_par_fallback` replaces `run` for *auto-selected, multi-threaded, small*
/// problems. Only the VNNI auto path sets it: VNNI's mandatory RHS-pack barrier (the
/// quad layout can't be read in place) outweighs the compute saving on a small parallel
/// problem, so the in-place widen kernel wins; serial and large-parallel runs keep VNNI.
/// `None` for every other selection and when VNNI is *forced* (force must run exactly
/// that kernel). Bit-identical to VNNI (exact i32), so the swap never perturbs results.
#[cfg(feature = "int8")]
#[derive(Copy, Clone)]
struct IntDispatched {
    run: IntFn,
    small_par_fallback: Option<IntFn>,
}

#[cfg(feature = "int8")]
const DISP_I8_SCALAR: IntDispatched = IntDispatched {
    run: gemm_i8_scalar,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_FMA: IntDispatched = IntDispatched {
    run: gemm_i8_fma,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_AVX512: IntDispatched = IntDispatched {
    run: gemm_i8_avx512,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_AVX512VNNI: IntDispatched = IntDispatched {
    run: gemm_i8_avx512vnni,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
const DISP_I8_NEON: IntDispatched = IntDispatched {
    run: gemm_i8_neon,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_I8_SIMD128: IntDispatched = IntDispatched {
    run: gemm_i8_simd128,
    small_par_fallback: None,
};

/// `i8` ISA selection. The widen-and-multiply integer kernel uses only AVX2/AVX-512
/// integer ops (no VNNI), so the gates mirror the `f32` ladder.
#[cfg(feature = "int8")]
fn select_i8() -> IntDispatched {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_I8_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_I8_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_I8_AVX512;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512Vnni => {
            assert!(
                x86_isa_detected!("avx512vnni")
                    && x86_isa_detected!("avx512bw")
                    && x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512vnni, but this CPU/emulator does not report avx512f+bw+vnni"
            );
            return DISP_I8_AVX512VNNI;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_I8_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_I8_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // VNNI dot kernel first — `vpdpbusd` is a structural win over widen-and-multiply,
        // except for small *parallel* problems, where it hands off to the widen kernel
        // (`small_par_fallback`) so its mandatory pack barrier does not dominate.
        if x86_isa_detected!("avx512vnni")
            && x86_isa_detected!("avx512bw")
            && x86_isa_detected!("avx512f")
        {
            return IntDispatched {
                small_par_fallback: Some(gemm_i8_avx512),
                ..DISP_I8_AVX512VNNI
            };
        }
        if x86_isa_detected!("avx512f") {
            return DISP_I8_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_I8_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_I8_NEON
    }
    // `simd128` on wasm32, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_I8_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_I8_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_I8_SCALAR
    }
}

memoized_select!(
    GEMM_I8,
    dispatched_i8,
    IntDispatched,
    select_i8,
    "The memoized integer dispatch descriptor (selection runs once).",
    "int8"
);

// ===========================================================================
// Integer requantizing GEMM (i8 · i8 -> i8): the `IntGemmQ` / `IntGemmVnniQ` families
// fused with the `KRequantize` epilogue (per-tensor scale + zero-point + optional per-row
// i32 bias). A dedicated task/dispatch, like `IntTask`, because the output is `i8` (not i32)
// and it carries the quantization parameters.
// ===========================================================================

/// A fully described integer requantizing GEMM: `i8` inputs, `i32` accumulator, `i8` output.
/// No `alpha` (folds into `scale`) and no `beta` (accumulating into a quantized C is
/// ill-defined). `bias` is an optional per-row / per-col `i32` vector (`bias_dim` in the
/// user frame; the dispatch flips it on an orientation swap).
#[cfg(feature = "int8")]
#[derive(Copy, Clone)]
pub(crate) struct RequantTask {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a: *const i8,
    pub rsa: isize,
    pub csa: isize,
    pub b: *const i8,
    pub rsb: isize,
    pub csb: isize,
    pub c: *mut i8,
    pub rsc: isize,
    pub csc: isize,
    pub scale: f32,
    pub zp: i32,
    pub bias: *const i32,
    pub has_bias: bool,
    pub bias_dim: BiasDim,
}

/// Top-level requantizing entry: the degenerate `k == 0` case (fill `C` with the requantized
/// bias / zero-point) then the ISA-dispatched fused kernel.
///
/// # Safety
/// `t`'s pointers valid; `c` not aliasing `a`/`b`, and `bias` (if `has_bias`) valid for the
/// oriented axis and disjoint from `c` (the API validates this).
#[cfg(feature = "int8")]
pub(crate) unsafe fn execute_int_requant(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        // The A·B term vanishes (k == 0): C[i,j] = clamp(zp + round_ne(scale·bias[..])).
        if t.k == 0 {
            requant_degenerate(&t);
            return;
        }
        let d = dispatched_i8_requant();
        // Mirror `execute_int`: an auto-VNNI *small parallel* problem hands off to the widen
        // `IntGemmQ` fallback (bit-identical, VNNI's pack barrier dominates there).
        let mnk = t.m.saturating_mul(t.n).saturating_mul(t.k);
        let run = pick_int_kernel(par, mnk, d.run, d.small_par_fallback);
        run(t, par, ws);
    }
}

/// `k == 0` fill: `C[i,j] = clamp(zp + round_ne(scale·bias[i or j]), -128, 127)` (= `zp as i8`
/// without bias). Uses the same `KRequantize::apply` as the kernel, applied to a zero
/// accumulator, so it is bit-identical to a `k > 0` run whose products are all zero.
#[cfg(feature = "int8")]
unsafe fn requant_degenerate(t: &RequantTask) {
    let epi = KRequantize {
        scale: t.scale,
        zp: t.zp,
        bias: Ptr(t.bias as *mut i32),
        has_bias: t.has_bias,
        bias_dim: t.bias_dim,
    };
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                // UFCS: `KRequantize` implements `Epilogue` for every `Acc = i32, Out = i8`
                // family, so the bare `apply` would be ambiguous. Any of them gives the same
                // scalar map; `IntGemmQ` is the always-available one.
                let out = <KRequantize as Epilogue<IntGemmQ>>::apply(&epi, 0, i, j);
                *t.c.offset(i as isize * t.rsc + j as isize * t.csc) = out;
            }
        }
    }
}

/// Requantizing driver entry for a concrete `(family, ISA, tile)`: the inline orientation
/// swap (which **flips the bias axis**), build the `KRequantize` epilogue, then the general
/// driver. No gemv / small_k reroute (correct at any `k` since `kc = k`).
///
/// # Safety
/// As [`execute_int_requant`].
#[cfg(feature = "int8")]
#[inline]
unsafe fn run_typed_int_requant<Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: RequantTask,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily<Lhs = i8, Rhs = i8, Acc = i32, Out = i8>,
    S: KernelSimd<i8, i8, i32, i8>,
{
    unsafe {
        let swap = orient_swap(
            &mut t.m, &mut t.n, &mut t.a, &mut t.rsa, &mut t.csa, &mut t.b, &mut t.rsb, &mut t.csb,
            &mut t.rsc, &mut t.csc,
        );
        if swap {
            // Cᵀ = Bᵀ·Aᵀ makes a per-row bias per-col in the driver frame (and vice versa).
            t.bias_dim = match t.bias_dim {
                BiasDim::PerRow => BiasDim::PerCol,
                BiasDim::PerCol => BiasDim::PerRow,
            };
        }
        let epi = KRequantize {
            scale: t.scale,
            zp: t.zp,
            bias: Ptr(t.bias as *mut i32),
            has_bias: t.has_bias,
            bias_dim: t.bias_dim,
        };
        // alpha = 1 (folded into scale), beta = 0 (no accumulate) — the family debug-asserts
        // exactly these.
        driver::run_epilogue::<Fam, S, KRequantize, MR_REG, NR>(
            simd, t.m, t.k, t.n, 1, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, 0, t.c, t.rsc, t.csc,
            &epi, par, ws,
        );
    }
}

#[cfg(feature = "int8")]
unsafe fn gemm_i8_requant_scalar(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmQ, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_requant_fma(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmQ, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_requant_avx512(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmQ, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_requant_avx512vnni(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmVnniQ, Avx512Vnni, 2, 12>(Avx512Vnni, t, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
unsafe fn gemm_i8_requant_neon(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmQ, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_i8_requant_simd128(t: RequantTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int_requant::<IntGemmQ, Simd128, 2, 4>(Simd128, t, par, ws) }
}

#[cfg(feature = "int8")]
type RequantFn = unsafe fn(RequantTask, Parallelism, &mut Workspace);

/// Memoized requantizing dispatch slot (mirror of [`IntDispatched`]): the `small_par_fallback`
/// swaps auto-VNNI to widen `IntGemmQ` for small parallel problems (bit-identical).
#[cfg(feature = "int8")]
#[derive(Copy, Clone)]
struct IntRequantDispatched {
    run: RequantFn,
    small_par_fallback: Option<RequantFn>,
}

#[cfg(feature = "int8")]
const RDISP_I8_SCALAR: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_scalar,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const RDISP_I8_FMA: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_fma,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const RDISP_I8_AVX512: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_avx512,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const RDISP_I8_AVX512VNNI: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_avx512vnni,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
const RDISP_I8_NEON: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_neon,
    small_par_fallback: None,
};
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
const RDISP_I8_SIMD128: IntRequantDispatched = IntRequantDispatched {
    run: gemm_i8_requant_simd128,
    small_par_fallback: None,
};

/// `i8` requantize ISA selection (mirror of [`select_i8`]).
#[cfg(feature = "int8")]
fn select_i8_requant() -> IntRequantDispatched {
    match forced_isa() {
        ForcedIsa::Scalar => return RDISP_I8_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return RDISP_I8_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return RDISP_I8_AVX512;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512Vnni => {
            assert!(
                x86_isa_detected!("avx512vnni")
                    && x86_isa_detected!("avx512bw")
                    && x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512vnni, but this CPU/emulator does not report avx512f+bw+vnni"
            );
            return RDISP_I8_AVX512VNNI;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return RDISP_I8_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return RDISP_I8_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // VNNI dot kernel first, with the widen `IntGemmQ` as the small-parallel fallback.
        if x86_isa_detected!("avx512vnni")
            && x86_isa_detected!("avx512bw")
            && x86_isa_detected!("avx512f")
        {
            return IntRequantDispatched {
                small_par_fallback: Some(gemm_i8_requant_avx512),
                ..RDISP_I8_AVX512VNNI
            };
        }
        if x86_isa_detected!("avx512f") {
            return RDISP_I8_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return RDISP_I8_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        RDISP_I8_NEON
    }
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            RDISP_I8_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            RDISP_I8_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        RDISP_I8_SCALAR
    }
}

memoized_select!(
    GEMM_I8_REQUANT,
    dispatched_i8_requant,
    IntRequantDispatched,
    select_i8_requant,
    "The memoized requantizing-integer dispatch descriptor (selection runs once).",
    "int8"
);
