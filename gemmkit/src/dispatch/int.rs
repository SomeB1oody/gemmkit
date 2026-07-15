//! Integer GEMM dispatch: `i8 -> i32` (`IntTask`) and the fused `i8 -> i8`
//! requantizing path (`RequantTask`), with their per-ISA wrappers, descriptors,
//! and selection ladders

#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::isa::{ForcedIsa, forced_isa};
use super::orient_swap;
use crate::driver;
use crate::kernel::IntGemm;
#[cfg(feature = "epilogue")]
use crate::kernel::IntGemmQ;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::kernel::IntGemmVnni;
#[cfg(all(feature = "epilogue", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::IntGemmVnniQ;
use crate::kernel::KernelFamily;
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{BiasDim, BiasSpec, Epilogue, KRequantize, QuantOut};
use crate::parallel::Parallelism;
#[cfg(feature = "epilogue")]
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
/// this dedicated task + dispatch
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

// Integer GEMM (i8 -> i32): a dedicated heterogeneous dispatch path, since the
// homogeneous `GemmScalar` cannot express `Out != Lhs`

/// Pick the integer kernel fn for this problem, shared by the plain and requantizing
/// entries (`F` is `IntFn` / `RequantFn`, both `Copy` fn pointers). Auto VNNI hands *small
/// multi-threaded* problems to the widen fallback (the dot kernel's mandatory pack barrier
/// dominates there) while `Rayon(1)`/`Serial` keep VNNI at any size; `small_par_fallback`
/// is `None` for every non-VNNI kernel, so `run` is returned unchanged. Centralizing the
/// `I8_VNNI_MIN_PAR_MNK` gate keeps the 2 paths' calibration from drifting apart
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

/// Top-level integer entry: degenerate cases (`C <- beta*C` when the `A*B` term
/// vanishes) then the ISA-dispatched integer kernel
///
/// # Safety
/// `t`'s pointers valid for the implied regions; `c` must not alias `a`/`b`
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

/// `C <- beta*C` for the integer output (wrapping i32; `beta == 0` overwrites to 0)
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
/// (identical to the float path, only strides move) and `driver::run::<IntGemm>`
///
/// # Safety
/// As [`execute_int`]
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
        // Skinny / low-depth shape: route through the widen `IntGemm` (never `IntGemmVnni`):
        // at tiny `k` VNNI's mandatory quad-pack barrier never amortizes. Stays bit-exact
        // (i32 modular), so it reproduces the widen and VNNI results alike
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
    // i32 accumulator -> MR = 2*8 = 16, NR = 6 (the f32 FMA tile)
    unsafe { run_typed_int::<IntGemm, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 accumulator -> MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile)
    unsafe { run_typed_int::<IntGemm, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512vnni(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // VNNI dot kernel, same tile as AVX-512: MR = 2*16 = 32, NR = 12 -> 24 acc + 2 vA
    // + 1 vB = 27 ZMM. `vpdpbusd` folds 4 depth steps x 16 lanes per instruction
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

/// Memoized integer dispatch slot (mirror of [`Dispatched`] but a single kernel, integer
/// prepack is not yet a public API)
///
/// `small_par_fallback` replaces `run` for *auto-selected, multi-threaded, small*
/// problems. Only the VNNI auto path sets it: VNNI's mandatory RHS-pack barrier (the
/// quad layout can't be read in place) outweighs the compute saving on a small parallel
/// problem, so the in-place widen kernel wins; serial and large-parallel runs keep VNNI.
/// `None` for every other selection and when VNNI is *forced* (force must run exactly
/// that kernel). Bit-identical to VNNI (exact i32), so the swap never perturbs results
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
/// integer ops (no VNNI), so the gates mirror the `f32` ladder
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
        // VNNI dot kernel first: `vpdpbusd` is a structural win over widen-and-multiply,
        // except for small *parallel* problems, where it hands off to the widen kernel
        // (`small_par_fallback`) so its mandatory pack barrier does not dominate
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

// Integer requantizing GEMM (i8 * i8 -> O, O in {i8, u8}): the `IntGemmQ<O>` /
// `IntGemmVnniQ<O>` families fused with the `KRequantize` epilogue (per-tensor scale +
// zero-point + optional per-row i32 bias). A dedicated task/dispatch, like `IntTask`, because
// the output is a quantized byte (not i32) and it carries the quantization parameters. The
// task, dispatch descriptor, and every per-ISA wrapper are generic over the output byte `O`;
// `requant_dispatch!` stamps the wrappers / consts / select ladder / memoized slot once per
// `O`, and `RequantOut` maps `O` to its memoized descriptor at the top entry

/// A fully described integer requantizing GEMM: `i8` inputs, `i32` accumulator, `O` output
/// (`i8` signed `[-128, 127]` or `u8` `[0, 255]`). No `alpha` (folds into `scale`) and no
/// `beta` (accumulating into a quantized C is ill-defined). `bias` is an optional per-row /
/// per-col `i32` vector (`bias_dim` in the user frame; the dispatch flips it on an orientation
/// swap)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) struct RequantTask<O> {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a: *const i8,
    pub rsa: isize,
    pub csa: isize,
    pub b: *const i8,
    pub rsb: isize,
    pub csb: isize,
    pub c: *mut O,
    pub rsc: isize,
    pub csc: isize,
    pub scale: f32,
    pub zp: i32,
    pub bias: *const i32,
    pub has_bias: bool,
    pub bias_dim: BiasDim,
}

/// Top-level requantizing entry (generic over the output byte `O`): the degenerate `k == 0`
/// case (fill `C` with the requantized bias / zero-point) then the ISA-dispatched fused kernel.
/// `O::dispatched()` selects the per-`O` memoized descriptor
///
/// # Safety
/// `t`'s pointers valid; `c` not aliasing `a`/`b`, and `bias` (if `has_bias`) valid for the
/// oriented axis and disjoint from `c` (the API validates this)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) unsafe fn execute_int_requant<O: RequantOut>(
    t: RequantTask<O>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if t.m == 0 || t.n == 0 {
            return;
        }
        // The A*B term vanishes (k == 0): C[i,j] = clamp(zp + round_ne(scale*bias[..]))
        if t.k == 0 {
            requant_degenerate(&t);
            return;
        }
        let d = O::dispatched();
        // Mirror `execute_int`: an auto-VNNI *small parallel* problem hands off to the widen
        // `IntGemmQ<O>` fallback (bit-identical, VNNI's pack barrier dominates there)
        let mnk = t.m.saturating_mul(t.n).saturating_mul(t.k);
        let run = pick_int_kernel(par, mnk, d.run, d.small_par_fallback);
        run(t, par, ws);
    }
}

/// Build the `KRequantize` bias spec from a task's (already axis-flipped) bias fields:
/// `has_bias == false` maps to `BiasSpec::None`, else the `Row` / `Col` variant selected by
/// `bias_dim`. Shared by both construction sites so the encoding lives in one place
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[inline]
fn requant_bias_spec<O>(t: &RequantTask<O>) -> BiasSpec<i32> {
    if t.has_bias {
        let p = Ptr(t.bias as *mut i32);
        match t.bias_dim {
            BiasDim::PerRow => BiasSpec::Row(p),
            BiasDim::PerCol => BiasSpec::Col(p),
        }
    } else {
        BiasSpec::None
    }
}

/// `k == 0` fill: `C[i,j] = clamp(zp + round_ne(scale*bias[i or j]), O::LO, O::HI)` (= `zp`
/// clamped into the output band, without bias). Uses the same `KRequantize::apply` as the
/// kernel, applied to a zero accumulator, so it is bit-identical to a `k > 0` run whose
/// products are all zero
#[cfg(all(feature = "int8", feature = "epilogue"))]
unsafe fn requant_degenerate<O: QuantOut>(t: &RequantTask<O>) {
    let epi = KRequantize {
        scale: t.scale,
        zp: t.zp,
        bias: requant_bias_spec(t),
    };
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                // UFCS: `KRequantize` implements `Epilogue` for every `Acc = i32, Out = O`
                // family, so the bare `apply` would be ambiguous. Any of them gives the same
                // scalar map; `IntGemmQ<O>` is the always-available one for this output byte
                let out = <KRequantize as Epilogue<IntGemmQ<O>>>::apply(&epi, 0, i, j);
                *t.c.offset(i as isize * t.rsc + j as isize * t.csc) = out;
            }
        }
    }
}

/// Requantizing driver entry for a concrete `(family, ISA, tile, output byte)`: the inline
/// orientation swap (which **flips the bias axis**), build the `KRequantize` epilogue, then the
/// general driver. No gemv / small_k reroute (correct at any `k` since `kc = k`). The generic
/// param order is `<Fam, S, O, MR_REG, NR>` so the wrapper turbofish provides all 5
///
/// # Safety
/// As [`execute_int_requant`]
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[inline]
unsafe fn run_typed_int_requant<Fam, S, O, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: RequantTask<O>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    O: QuantOut,
    Fam: KernelFamily<Lhs = i8, Rhs = i8, Acc = i32, Out = O>,
    S: KernelSimd<i8, i8, i32, O>,
{
    unsafe {
        let swap = orient_swap(
            &mut t.m, &mut t.n, &mut t.a, &mut t.rsa, &mut t.csa, &mut t.b, &mut t.rsb, &mut t.csb,
            &mut t.rsc, &mut t.csc,
        );
        if swap {
            // C^T = B^T*A^T makes a per-row bias per-col in the driver frame (and vice versa)
            t.bias_dim = match t.bias_dim {
                BiasDim::PerRow => BiasDim::PerCol,
                BiasDim::PerCol => BiasDim::PerRow,
            };
        }
        let epi = KRequantize {
            scale: t.scale,
            zp: t.zp,
            bias: requant_bias_spec(&t),
        };
        // alpha = 1 (folded into scale), beta = 0 (no accumulate): the family debug-asserts
        // exactly these
        driver::run_epilogue::<Fam, S, KRequantize, MR_REG, NR>(
            simd, t.m, t.k, t.n, 1, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, 0, t.c, t.rsc, t.csc,
            &epi, par, ws,
        );
    }
}

/// A per-ISA requant kernel for a given output byte `O`. `Copy` (a fn pointer), so
/// [`pick_int_kernel`] can swap in the small-parallel fallback
#[cfg(all(feature = "int8", feature = "epilogue"))]
type RequantFn<O> = unsafe fn(RequantTask<O>, Parallelism, &mut Workspace);

/// Memoized requantizing dispatch slot (mirror of [`IntDispatched`]), parametrized by the output
/// byte `O`: the `small_par_fallback` swaps auto-VNNI to the widen `IntGemmQ<O>` kernel for small
/// parallel problems (bit-identical). One instantiation exists per output type (`i8` / `u8`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) struct IntRequantDispatched<O> {
    run: RequantFn<O>,
    small_par_fallback: Option<RequantFn<O>>,
}

/// Stamp the per-ISA wrapper fns, descriptor consts, ISA-selection ladder, and memoized slot for
/// one output byte `$O` (`i8` / `u8`). The 2 invocations are byte-identical apart from `$O` and
/// the item names: same tiles, same cfg gates, same VNNI-first auto ladder (with the widen kernel
/// as the small-parallel fallback) as the historic `i8`-only requant dispatch. Every wrapper is a
/// thin `run_typed_int_requant::<Family<$O>, Token, $O, MR, NR>` call
#[cfg(all(feature = "int8", feature = "epilogue"))]
macro_rules! requant_dispatch {
    (
        $O:ty,
        $w_scalar:ident, $w_fma:ident, $w_avx512:ident, $w_vnni:ident, $w_neon:ident,
        $w_simd128:ident,
        $d_scalar:ident, $d_fma:ident, $d_avx512:ident, $d_vnni:ident, $d_neon:ident,
        $d_simd128:ident,
        $select:ident, $slot:ident, $accessor:ident, $doc:literal
    ) => {
        // per-ISA wrapper fns (families `IntGemmQ<$O>` / `IntGemmVnniQ<$O>`)
        #[cfg(feature = "int8")]
        unsafe fn $w_scalar(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, ScalarTok, $O, 4, 4>(ScalarTok, t, par, ws) }
        }
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe fn $w_fma(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Fma, $O, 2, 6>(Fma, t, par, ws) }
        }
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe fn $w_avx512(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Avx512, $O, 2, 12>(Avx512, t, par, ws) }
        }
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe fn $w_vnni(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe {
                run_typed_int_requant::<IntGemmVnniQ<$O>, Avx512Vnni, $O, 2, 12>(
                    Avx512Vnni,
                    t,
                    par,
                    ws,
                )
            }
        }
        #[cfg(all(feature = "int8", target_arch = "aarch64"))]
        unsafe fn $w_neon(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Neon, $O, 4, 4>(Neon, t, par, ws) }
        }
        #[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
        unsafe fn $w_simd128(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Simd128, $O, 2, 4>(Simd128, t, par, ws) }
        }

        // descriptor consts (one per ISA)
        #[cfg(feature = "int8")]
        const $d_scalar: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_scalar,
            small_par_fallback: None,
        };
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        const $d_fma: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_fma,
            small_par_fallback: None,
        };
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        const $d_avx512: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_avx512,
            small_par_fallback: None,
        };
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        const $d_vnni: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_vnni,
            small_par_fallback: None,
        };
        #[cfg(all(feature = "int8", target_arch = "aarch64"))]
        const $d_neon: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_neon,
            small_par_fallback: None,
        };
        #[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
        const $d_simd128: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_simd128,
            small_par_fallback: None,
        };

        /// Requantize ISA selection for this output byte (mirror of [`select_i8`])
        #[cfg(feature = "int8")]
        fn $select() -> IntRequantDispatched<$O> {
            match forced_isa() {
                ForcedIsa::Scalar => return $d_scalar,
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                ForcedIsa::Fma => {
                    assert!(
                        x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                        "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
                    );
                    return $d_fma;
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                ForcedIsa::Avx512F | ForcedIsa::Avx512Bf16 => {
                    assert!(
                        x86_isa_detected!("avx512f"),
                        "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
                    );
                    return $d_avx512;
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                ForcedIsa::Avx512Vnni => {
                    assert!(
                        x86_isa_detected!("avx512vnni")
                            && x86_isa_detected!("avx512bw")
                            && x86_isa_detected!("avx512f"),
                        "GEMMKIT_REQUIRE_ISA=avx512vnni, but this CPU/emulator does not report avx512f+bw+vnni"
                    );
                    return $d_vnni;
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                ForcedIsa::Fma
                | ForcedIsa::Avx512F
                | ForcedIsa::Avx512Vnni
                | ForcedIsa::Avx512Bf16 => {
                    panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
                }
                #[cfg(target_arch = "aarch64")]
                ForcedIsa::Neon => return $d_neon,
                #[cfg(not(target_arch = "aarch64"))]
                ForcedIsa::Neon => {
                    panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64")
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                ForcedIsa::Simd128 => return $d_simd128,
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                ForcedIsa::Simd128 => panic!(
                    "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
                ),
                ForcedIsa::Auto => {}
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // VNNI dot kernel first, with the widen `IntGemmQ<$O>` as the small-parallel fallback
                if x86_isa_detected!("avx512vnni")
                    && x86_isa_detected!("avx512bw")
                    && x86_isa_detected!("avx512f")
                {
                    return IntRequantDispatched {
                        small_par_fallback: Some($w_avx512),
                        ..$d_vnni
                    };
                }
                if x86_isa_detected!("avx512f") {
                    return $d_avx512;
                }
                if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
                    return $d_fma;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                $d_neon
            }
            #[cfg(target_arch = "wasm32")]
            {
                #[cfg(target_feature = "simd128")]
                {
                    $d_simd128
                }
                #[cfg(not(target_feature = "simd128"))]
                {
                    $d_scalar
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
            {
                $d_scalar
            }
        }

        memoized_select!($slot, $accessor, IntRequantDispatched<$O>, $select, $doc, "int8");
    };
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
requant_dispatch!(
    i8,
    requant_i8_scalar,
    requant_i8_fma,
    requant_i8_avx512,
    requant_i8_vnni,
    requant_i8_neon,
    requant_i8_simd128,
    RDISP_I8_SCALAR,
    RDISP_I8_FMA,
    RDISP_I8_AVX512,
    RDISP_I8_VNNI,
    RDISP_I8_NEON,
    RDISP_I8_SIMD128,
    select_requant_i8,
    GEMM_REQUANT_I8,
    dispatched_requant_i8,
    "The memoized `i8`-output requantizing dispatch descriptor (selection runs once)."
);

#[cfg(all(feature = "int8", feature = "epilogue"))]
requant_dispatch!(
    u8,
    requant_u8_scalar,
    requant_u8_fma,
    requant_u8_avx512,
    requant_u8_vnni,
    requant_u8_neon,
    requant_u8_simd128,
    RDISP_U8_SCALAR,
    RDISP_U8_FMA,
    RDISP_U8_AVX512,
    RDISP_U8_VNNI,
    RDISP_U8_NEON,
    RDISP_U8_SIMD128,
    select_requant_u8,
    GEMM_REQUANT_U8,
    dispatched_requant_u8,
    "The memoized `u8`-output requantizing dispatch descriptor (selection runs once)."
);

/// Maps an output byte `O` to its memoized per-`O` requant dispatch descriptor. Implemented for
/// the 2 quantized outputs (`i8` / `u8`); [`execute_int_requant`] is generic over
/// `O: RequantOut` and calls `O::dispatched()` to pick the matching memoized slot without a
/// runtime branch. Sealed by `QuantOut` (only `i8` / `u8` implement it)
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) trait RequantOut: QuantOut {
    /// The memoized ISA descriptor for this output byte
    fn dispatched() -> IntRequantDispatched<Self>
    where
        Self: Sized;
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl RequantOut for i8 {
    #[inline]
    fn dispatched() -> IntRequantDispatched<i8> {
        dispatched_requant_i8()
    }
}

#[cfg(all(feature = "int8", feature = "epilogue"))]
impl RequantOut for u8 {
    #[inline]
    fn dispatched() -> IntRequantDispatched<u8> {
        dispatched_requant_u8()
    }
}
