//! Integer GEMM dispatch: the plain `i8 -> i32` path (`IntTask`, `execute_int`) and,
//! under `epilogue`, the fused requantizing path (`RequantTask`, output `i8`/`u8`).
//! Each has a plain and a prepacked-RHS entry, with its own per-ISA wrappers, memoized
//! descriptor, and selection ladder

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
use crate::kernel::epilogue::{BiasDim, BiasSpec, Epilogue, KRequantize, QuantOut, ScaleSpec};
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
use crate::simd::{Avx512F, Fma};
use crate::special::{small_k, small_mn};
use crate::tuning;
use crate::workspace::Workspace;

/// A heterogeneous **integer** GEMM problem: `i8` inputs, `i32` accumulator and output
/// (`alpha`, `beta`, `C` all `i32`). The homogeneous [`Task`] / [`GemmScalar`] machinery
/// assumes `Lhs == Out`, which `i8 -> i32` breaks, so integer GEMM gets its own task type
/// and dispatch path instead
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

/// An **integer** GEMM whose RHS is already prepacked into the selected kernel family's
/// micropanel layout: `C(i32) <- alpha*A(i8)*(prepacked B) + beta*C`. The heterogeneous
/// (`i8 -> i32`) twin of [`crate::dispatch::PackedConsume`]: it carries the blocking geometry
/// (`nr`, `kc`, `nc`) the buffer was packed for, so the consuming call reads panels with the
/// exact tiling the pack step used rather than re-deriving it
#[cfg(feature = "int8")]
pub(crate) struct IntPackedConsume {
    /// Row count of A and C
    pub m: usize,
    /// Contraction length: column count of A, which must match the prepacked B's depth
    pub k: usize,
    /// Column count of the prepacked B, and of C
    pub n: usize,
    /// Scale applied to the A*B product
    pub alpha: i32,
    /// LHS base pointer and element strides
    pub a: *const i8,
    pub rsa: isize,
    pub csa: isize,
    /// Base of the prepacked RHS micropanel buffer (see [`crate::driver::pack_rhs_full`])
    pub packed: *const i8,
    /// Blocking geometry `packed` was built with
    pub nr: usize,
    pub kc: usize,
    pub nc: usize,
    /// Scale applied to the incoming C before adding alpha*A*B
    pub beta: i32,
    /// Output base pointer and element strides
    pub c: *mut i32,
    pub rsc: isize,
    pub csc: isize,
}

// Plain i8 -> i32 GEMM: a dedicated heterogeneous dispatch path, since `GemmScalar` assumes Lhs == Out

/// Choose between the memoized kernel and its small-parallel fallback, shared by the plain and
/// requantizing entries (`F` is `IntFn` / `RequantFn`, both `Copy` fn pointers). Falls back only
/// when `small_par_fallback` is `Some` (only the auto-VNNI selection ever sets it), `par` is
/// genuinely multi-threaded (`Rayon(n)` with `n != 1`), and `mnk` is below
/// `tuning::i8_vnni_min_par_mnk()`; every other case returns `run` unchanged. Centralizing the
/// gate here keeps [`execute_int`] and [`execute_int_requant`] from calibrating it separately
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

/// Top-level plain-integer entry: the degenerate `C <- beta*C` cases (when the `A*B` term
/// vanishes) then the memoized ISA-dispatched kernel, through [`pick_int_kernel`]
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

/// `C <- beta*C`, `i32` wrapping on overflow; `beta == 0` overwrites with 0 rather than
/// multiplying (`beta == 1` leaves `C` unwritten)
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

/// Integer driver entry for a concrete `(family, ISA, tile)`: the orientation swap (as the
/// float path), then small_mn / small_k / the general driver. There is no dedicated gemv
/// route here (unlike the float/mixed entries), so a gemv-shaped integer problem falls
/// through to whichever of those 3 fits it
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
        // Small m,n with long k: a direct i8 -> i32 widening SIMD dot per output, instead of
        // packing the tiny tile up to a full MR x NR microtile (mostly padding). Contiguous-along-k
        // operands read in place; a strided one copies to k-contiguous scratch first. Shares the
        // dims-level small_mn_eligible_dims / small_mn_pack_eligible_dims gates with float/mixed so
        // the 3 never drift apart, and fires even under a forced ISA, so a forced-VNNI tiny shape
        // widens here rather than paying VNNI's pack barrier. Bit-exact vs every other route: i32
        // wraparound sums are associative regardless of accumulation order
        if super::small_mn_eligible_dims(t.m, t.n, t.k, t.csa, t.rsb)
            || super::small_mn_pack_eligible_dims(t.m, t.n, t.k, t.csa, t.rsb)
        {
            small_mn::run_int::<S>(
                simd, t.m, t.k, t.n, par, ws, t.alpha, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb,
                t.beta, t.c, t.rsc, t.csc,
            );
            return;
        }
        // Low-depth shape: always the widen IntGemm (never IntGemmVnni), since VNNI's mandatory
        // quad-pack barrier never amortizes at tiny k; bit-exact with every other route
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
    // i32 is the same width as f32 -> MR = 2*8 = 16, NR = 6, the f32 FMA tile
    unsafe { run_typed_int::<IntGemm, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512f(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // i32 is the same width as f32 -> MR = 2*16 = 32, NR = 12, the f32 AVX-512F tile
    unsafe { run_typed_int::<IntGemm, Avx512F, 2, 12>(Avx512F, t, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512vnni(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    // Same tile as plain AVX-512F: MR = 2*16 = 32, NR = 12 -> 24 acc + 2 A + 1 B = 27 of 32
    // ZMM. vpdpbusd dot-folds 4 depth steps x 16 lanes per instruction
    unsafe { run_typed_int::<IntGemmVnni, Avx512Vnni, 2, 12>(Avx512Vnni, t, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
unsafe fn gemm_i8_neon(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<IntGemm, Neon, 4, 4>(Neon, t, par, ws) }
}
// i8 on wasm simd128
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_i8_simd128(t: IntTask, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_int::<IntGemm, Simd128, 2, 4>(Simd128, t, par, ws) }
}

/// Prepacked-RHS integer driver entry for a concrete `(family, ISA, tile)`. No gemv route, no
/// small_mn / small_k reroute, and **no orientation swap**: the API guarantees column-major-ish
/// C, so the prepacked buffer is always the genuine RHS. The heterogeneous mirror of the float
/// `run_packed_typed`
///
/// # Safety
/// As [`run_typed_int`], plus `req.packed` valid for the geometry recorded in `req`
#[cfg(feature = "int8")]
#[inline]
unsafe fn run_packed_typed_int<Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: IntPackedConsume,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily<Lhs = i8, Rhs = i8, Acc = i32, Out = i32>,
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        // NR is structural (the panel width is this kernel's NR); one process's memoized ISA
        // choice guarantees it matches what packed the buffer
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<Fam, S, MR_REG, NR>(
            simd, req.m, req.k, req.n, req.alpha, req.a, req.rsa, req.csa, req.packed, req.kc,
            req.nc, req.beta, req.c, req.rsc, req.csc, par, ws,
        );
    }
}

// prepacked-RHS integer entry points: one per (ISA, family), same tiles as the plain wrappers
// The widen ISAs consume a plain-panel buffer (IntGemm); the VNNI entry consumes the
// k-quad-interleaved buffer (IntGemmVnni). Each is cfg-gated exactly like its plain sibling

#[cfg(feature = "int8")]
unsafe fn gemm_i8_scalar_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_int::<IntGemm, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_fma_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_int::<IntGemm, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512f_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_int::<IntGemm, Avx512F, 2, 12>(Avx512F, r, par, ws) }
}
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_i8_avx512vnni_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    // The buffer is k-quad-interleaved, so only the VNNI family can read it; the widen
    // small-parallel fallback would misread that layout, so the packed path never applies it
    unsafe { run_packed_typed_int::<IntGemmVnni, Avx512Vnni, 2, 12>(Avx512Vnni, r, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
unsafe fn gemm_i8_neon_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_int::<IntGemm, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_i8_simd128_packed(r: IntPackedConsume, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_int::<IntGemm, Simd128, 2, 4>(Simd128, r, par, ws) }
}

/// Pack a full `(k, n)` RHS into the widen (`IntGemm`) micropanel layout: plain panels,
/// `DEPTH_MULTIPLE = 1`, the same [`driver::pack_rhs_full`] the per-call widen driver runs
///
/// # Safety
/// As [`driver::pack_rhs_full`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_rhs_i8_widen(
    dst: *mut i8,
    b: *const i8,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    unsafe { driver::pack_rhs_full::<IntGemm>(dst, b, rsb, csb, k, n, kc, nc, nr) }
}

/// Pack a full `(k, n)` RHS into the VNNI (`IntGemmVnni`) k-quad-interleaved layout
/// (`DEPTH_MULTIPLE = 4`; identity transform, since the `+128` signedness bias lives only on
/// the LHS pack), the same [`driver::pack_rhs_full`] the per-call VNNI driver runs
///
/// # Safety
/// As [`driver::pack_rhs_full`]
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_rhs_i8_vnni(
    dst: *mut i8,
    b: *const i8,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    unsafe { driver::pack_rhs_full::<IntGemmVnni>(dst, b, rsb, csb, k, n, kc, nc, nr) }
}

/// A per-ISA plain-integer kernel entry (`Copy`, a fn pointer)
#[cfg(feature = "int8")]
type IntFn = unsafe fn(IntTask, Parallelism, &mut Workspace);
/// A per-ISA prepacked-RHS integer consume entry (`Copy`, a fn pointer)
#[cfg(feature = "int8")]
type IntPackedFn = unsafe fn(IntPackedConsume, Parallelism, &mut Workspace);
/// Packs a full RHS into the selected family's micropanel layout (`pack_rhs_full` bound to
/// `IntGemm` / `IntGemmVnni`); a `Copy` fn pointer carried by the descriptor
#[cfg(feature = "int8")]
type IntPackFn = unsafe fn(*mut i8, *const i8, isize, isize, usize, usize, usize, usize, usize);

/// Memoized integer dispatch slot (the heterogeneous mirror of `Dispatched`): the plain
/// kernel, the prepacked-RHS kernel, the RHS packer, and the microtile geometry they share
///
/// `small_par_fallback` replaces `run` for an *auto-selected, multi-threaded, small* problem.
/// Only the VNNI auto selection sets it: VNNI's mandatory RHS-pack barrier (the quad layout
/// cannot be read in place) outweighs the compute saving on a small parallel problem, so the
/// in-place widen kernel wins there; a serial or large-parallel run keeps VNNI. `None` for every
/// other selection, and for a *forced* VNNI pin (a forced ISA has to run exactly that kernel).
/// Bit-identical to VNNI regardless (exact `i32`), so the swap never perturbs results
///
/// `run_packed` / `pack_rhs` serve the fixed-weight prepacked-RHS path and are always the
/// memoized family's own (VNNI's when auto-VNNI is selected), never the widen
/// `small_par_fallback`: the prepacked buffer's micropanel layout is family-specific
/// (VNNI k-quad-interleave vs plain panels), so a buffer packed by `pack_rhs` is consumable only
/// by the matching `run_packed`. [`execute_int_packed`] therefore bypasses the dynamic
/// small-parallel gate [`execute_int`] applies (the pack barrier that gate hedges against is
/// already amortized once at prepack time). `mr`/`nr`/`depth_multiple` mirror the tile constants
/// and feed [`prepack_rhs_i8`](crate::prepack_rhs_i8), so the buffer and the consuming call agree
/// on the blocking geometry
#[cfg(feature = "int8")]
#[derive(Copy, Clone)]
struct IntDispatched {
    run: IntFn,
    small_par_fallback: Option<IntFn>,
    run_packed: IntPackedFn,
    pack_rhs: IntPackFn,
    mr: usize,
    nr: usize,
    depth_multiple: usize,
}

#[cfg(feature = "int8")]
const DISP_I8_SCALAR: IntDispatched = IntDispatched {
    run: gemm_i8_scalar,
    small_par_fallback: None,
    run_packed: gemm_i8_scalar_packed,
    pack_rhs: pack_rhs_i8_widen,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_FMA: IntDispatched = IntDispatched {
    run: gemm_i8_fma,
    small_par_fallback: None,
    run_packed: gemm_i8_fma_packed,
    pack_rhs: pack_rhs_i8_widen,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_AVX512F: IntDispatched = IntDispatched {
    run: gemm_i8_avx512f,
    small_par_fallback: None,
    run_packed: gemm_i8_avx512f_packed,
    pack_rhs: pack_rhs_i8_widen,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_I8_AVX512VNNI: IntDispatched = IntDispatched {
    run: gemm_i8_avx512vnni,
    small_par_fallback: None,
    run_packed: gemm_i8_avx512vnni_packed,
    pack_rhs: pack_rhs_i8_vnni,
    mr: 32,
    nr: 12,
    // k-quad-interleaved pack: the prepack buffer rounds its depth up to a multiple of 4
    depth_multiple: 4,
};
#[cfg(all(feature = "int8", target_arch = "aarch64"))]
const DISP_I8_NEON: IntDispatched = IntDispatched {
    run: gemm_i8_neon,
    small_par_fallback: None,
    run_packed: gemm_i8_neon_packed,
    pack_rhs: pack_rhs_i8_widen,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "int8", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_I8_SIMD128: IntDispatched = IntDispatched {
    run: gemm_i8_simd128,
    small_par_fallback: None,
    run_packed: gemm_i8_simd128_packed,
    pack_rhs: pack_rhs_i8_widen,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

/// `i8` ISA selection: auto-selects the VNNI dot kernel first (falling back to the widen
/// kernel on a small parallel problem), then the same widen-path feature gates `select_f32`
/// uses (`avx512f`, then `avx2 + fma`), since the widen-and-multiply kernel needs no feature
/// beyond what f32 needs
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
                "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
            );
            return DISP_I8_AVX512F;
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
        // VNNI dot kernel first: vpdpbusd is a structural win over widen-and-multiply, except on
        // a small parallel problem, where small_par_fallback hands off to the widen kernel so
        // VNNI's mandatory pack barrier does not dominate
        if x86_isa_detected!("avx512vnni")
            && x86_isa_detected!("avx512bw")
            && x86_isa_detected!("avx512f")
        {
            return IntDispatched {
                small_par_fallback: Some(gemm_i8_avx512f),
                ..DISP_I8_AVX512VNNI
            };
        }
        if x86_isa_detected!("avx512f") {
            return DISP_I8_AVX512F;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_I8_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_I8_NEON
    }
    // simd128 on wasm32 (when compiled in), scalar everywhere else
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

/// The memoized integer kernel's microtile `(mr, nr) = (MR_REG*LANES, NR)`, read by
/// [`prepack_rhs_i8`](crate::prepack_rhs_i8) to block a prepacked buffer against the *same*
/// ISA choice the consuming call will make (the heterogeneous mirror of
/// [`GemmScalar::rhs_tile`](crate::dispatch::GemmScalar))
#[cfg(feature = "int8")]
pub(crate) fn i8_rhs_tile() -> (usize, usize) {
    let d = dispatched_i8();
    (d.mr, d.nr)
}

/// The memoized integer kernel family's pack depth multiple: `4` for the VNNI `vpdpbusd` dot
/// kernel, `1` for every widen kernel. The prepack constructor rounds the packed depth up to it
#[cfg(feature = "int8")]
pub(crate) fn i8_rhs_depth_multiple() -> usize {
    dispatched_i8().depth_multiple
}

/// Pack a full `(k, n)` RHS into the memoized integer kernel's micropanel buffer, the same layout
/// [`execute_int_packed`] consumes. Delegates to the family-specific `pack_rhs_full` (widen plain
/// panels vs VNNI k-quad-interleave), so the bytes match the per-call pack exactly
///
/// # Safety
/// As [`driver::pack_rhs_full`]
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pack_rhs_full_i8(
    dst: *mut i8,
    b: *const i8,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    unsafe { (dispatched_i8().pack_rhs)(dst, b, rsb, csb, k, n, kc, nc, nr) }
}

/// Top-level prepacked-RHS integer entry: the degenerate cases (the `A*B` term vanishes, so
/// `C <- beta*C` without ever touching the packed buffer) then the memoized prepacked kernel.
/// Runs the buffer's own family unconditionally, deliberately bypassing the `small_par_fallback`
/// gate [`execute_int`] applies: a quad-interleaved VNNI buffer is not consumable by the widen
/// fallback, and the pack barrier that gate hedges against was already amortized at prepack time
///
/// # Safety
/// As [`execute_int`], plus `req.packed` valid for the recorded geometry and not aliasing `c`
#[cfg(feature = "int8")]
pub(crate) unsafe fn execute_int_packed(
    req: IntPackedConsume,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        if req.m == 0 || req.n == 0 {
            return;
        }
        if req.k == 0 || req.alpha == 0 {
            scale_c_int(req.beta, req.c, req.m, req.n, req.rsc, req.csc);
            return;
        }
        (dispatched_i8().run_packed)(req, par, ws);
    }
}

// Integer requantizing GEMM (i8 * i8 -> O, O in {i8, u8}): IntGemmQ<O> / IntGemmVnniQ<O> fused
// with the KRequantize epilogue (per-tensor or per-row/per-col scale, zero-point, optional i32
// bias). A dedicated task/dispatch, like IntTask, since the output is a quantized byte (not i32)
// and carries the quantization parameters. Task, descriptor, and every wrapper are generic over
// the output byte O; requant_dispatch! stamps the wrappers/consts/ladder/memoized slot once per
// O, and RequantOut maps O to its memoized descriptor at the top entry

/// A fully described integer requantizing GEMM: `i8` inputs, `i32` accumulator, `O` output
/// (`i8` signed `[-128, 127]` or `u8` `[0, 255]`). No `alpha` (folded into `scale`) and no
/// `beta` (accumulating into a quantized C is ill-defined). The output scale is per-tensor
/// (`scale`, used when `has_row_scales == false`) or per-row / per-col (`row_scales`, an `f32`
/// vector enabled by `has_row_scales`, the per-channel quantized-inference convention). `bias`
/// is an optional per-row / per-col `i32` vector. `bias_dim` selects the shared per-row / per-col
/// axis of BOTH `row_scales` and `bias` (length `m` for `PerRow`, `n` for `PerCol`); the dispatch
/// flips it on an orientation swap
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
    pub row_scales: *const f32,
    pub has_row_scales: bool,
    pub zp: i32,
    pub bias: *const i32,
    pub has_bias: bool,
    pub bias_dim: BiasDim,
}

/// Top-level requantizing entry (generic over the output byte `O`): the degenerate `k == 0`
/// case (fill `C` with the requantized bias / zero-point) then the memoized ISA-dispatched fused
/// kernel, through `O::dispatched()` and [`pick_int_kernel`]. Unlike [`execute_int`] there is no
/// `alpha == 0` degenerate case: `RequantTask` has no `alpha` field (it is folded into `scale`)
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
        // As execute_int: an auto-VNNI small parallel problem hands off to the widen
        // IntGemmQ<O> fallback (bit-identical, VNNI's pack barrier dominates there)
        let mnk = t.m.saturating_mul(t.n).saturating_mul(t.k);
        let run = pick_int_kernel(par, mnk, d.run, d.small_par_fallback);
        run(t, par, ws);
    }
}

/// Build the `KRequantize` bias spec from a task's (already axis-flipped) bias fields:
/// `has_bias == false` maps to `BiasSpec::None`, else the `Row` / `Col` variant `bias_dim`
/// selects. Shared by both construction sites so the encoding lives in exactly one place
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

/// Build the `KRequantize` scale spec from a task's (already axis-flipped) scale fields:
/// `has_row_scales == false` maps to the per-tensor `Tensor(scale)`, else the `Row` / `Col`
/// variant `bias_dim` selects (the scales share the bias's user axis, so the same flipped
/// `bias_dim` picks both). Shared by both construction sites so the encoding lives in one place
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[inline]
fn requant_scale_spec<O>(t: &RequantTask<O>) -> ScaleSpec {
    if t.has_row_scales {
        let p = Ptr(t.row_scales as *mut f32);
        match t.bias_dim {
            BiasDim::PerRow => ScaleSpec::Row(p),
            BiasDim::PerCol => ScaleSpec::Col(p),
        }
    } else {
        ScaleSpec::Tensor(t.scale)
    }
}

/// `k == 0` fill: `C[i,j] = clamp(zp + round_ne(scale*bias[i or j]), O::LO, O::HI)`, which
/// collapses to `zp` clamped into the output band when there is no bias. Uses the same
/// `KRequantize::apply` as the kernel, applied to a zero accumulator, so it is bit-identical to
/// a `k > 0` run whose products are all zero
#[cfg(all(feature = "int8", feature = "epilogue"))]
unsafe fn requant_degenerate<O: QuantOut>(t: &RequantTask<O>) {
    let epi = KRequantize {
        scale: requant_scale_spec(t),
        zp: t.zp,
        bias: requant_bias_spec(t),
    };
    unsafe {
        for j in 0..t.n {
            for i in 0..t.m {
                // UFCS: KRequantize implements Epilogue for every Acc = i32, Out = O family, so
                // the bare `apply` would be ambiguous. Any of them gives the same scalar map;
                // IntGemmQ<O> is the one always available for this output byte
                let out = <KRequantize as Epilogue<IntGemmQ<O>>>::apply(&epi, 0, i, j);
                *t.c.offset(i as isize * t.rsc + j as isize * t.csc) = out;
            }
        }
    }
}

/// Requantizing driver entry for a concrete `(family, ISA, tile, output byte)`: the orientation
/// swap (which **flips the bias axis**), build the `KRequantize` epilogue, then the general
/// driver. No gemv / small_mn / small_k reroute: `Fam::OUT_IS_ACC == false` already forces
/// `kc = k` in the driver, so there is no dedicated-path win left to chase at any `k`. The
/// generic param order is `<Fam, S, O, MR_REG, NR>`, so the wrapper turbofish provides all 5
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
            // The scales share the same user axis, so this one flip re-orients both
            t.bias_dim = match t.bias_dim {
                BiasDim::PerRow => BiasDim::PerCol,
                BiasDim::PerCol => BiasDim::PerRow,
            };
        }
        let epi = KRequantize {
            scale: requant_scale_spec(&t),
            zp: t.zp,
            bias: requant_bias_spec(&t),
        };
        // alpha = 1 (folded into scale), beta = 0 (no accumulate): the family debug_asserts
        // both statuses match
        driver::run_epilogue::<Fam, S, KRequantize, MR_REG, NR>(
            simd, t.m, t.k, t.n, 1, t.a, t.rsa, t.csa, t.b, t.rsb, t.csb, 0, t.c, t.rsc, t.csc,
            &epi, par, ws,
        );
    }
}

/// A per-ISA requant kernel entry for a given output byte `O`. `Copy` (a fn pointer), so
/// [`pick_int_kernel`] can swap in the small-parallel fallback
#[cfg(all(feature = "int8", feature = "epilogue"))]
type RequantFn<O> = unsafe fn(RequantTask<O>, Parallelism, &mut Workspace);

/// Memoized requantizing dispatch slot (mirror of [`IntDispatched`]), parametrized by the output
/// byte `O`: `small_par_fallback` swaps auto-VNNI to the widen `IntGemmQ<O>` kernel on a small
/// parallel problem (bit-identical). One instantiation exists per output type (`i8` / `u8`)
#[cfg(all(feature = "int8", feature = "epilogue"))]
#[derive(Copy, Clone)]
pub(crate) struct IntRequantDispatched<O> {
    run: RequantFn<O>,
    small_par_fallback: Option<RequantFn<O>>,
}

/// Stamps the per-ISA wrapper fns, descriptor consts, ISA-selection ladder, and memoized slot for
/// one output byte `$O` (`i8` / `u8`). The 2 invocations differ only in `$O` and the item names:
/// same tiles, same cfg gates, and the same VNNI-first auto ladder (with the widen kernel as the
/// small-parallel fallback) as [`select_i8`]. Every wrapper is a thin
/// `run_typed_int_requant::<Family<$O>, Token, $O, MR, NR>` call
#[cfg(all(feature = "int8", feature = "epilogue"))]
macro_rules! requant_dispatch {
    (
        $O:ty,
        $w_scalar:ident, $w_fma:ident, $w_avx512f:ident, $w_vnni:ident, $w_neon:ident,
        $w_simd128:ident,
        $d_scalar:ident, $d_fma:ident, $d_avx512f:ident, $d_vnni:ident, $d_neon:ident,
        $d_simd128:ident,
        $select:ident, $slot:ident, $accessor:ident, $doc:literal
    ) => {
        // per-ISA wrapper fns (families IntGemmQ<$O> / IntGemmVnniQ<$O>)
        #[cfg(feature = "int8")]
        unsafe fn $w_scalar(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, ScalarTok, $O, 4, 4>(ScalarTok, t, par, ws) }
        }
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe fn $w_fma(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Fma, $O, 2, 6>(Fma, t, par, ws) }
        }
        #[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe fn $w_avx512f(t: RequantTask<$O>, par: Parallelism, ws: &mut Workspace) {
            unsafe { run_typed_int_requant::<IntGemmQ<$O>, Avx512F, $O, 2, 12>(Avx512F, t, par, ws) }
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
        const $d_avx512f: IntRequantDispatched<$O> = IntRequantDispatched {
            run: $w_avx512f,
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

        /// Requantizing ISA selection for this output byte (mirror of [`select_i8`])
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
                        "GEMMKIT_REQUIRE_ISA=avx512f, but this CPU/emulator does not report avx512f"
                    );
                    return $d_avx512f;
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
                // VNNI dot kernel first, with the widen IntGemmQ<$O> as the small-parallel fallback
                if x86_isa_detected!("avx512vnni")
                    && x86_isa_detected!("avx512bw")
                    && x86_isa_detected!("avx512f")
                {
                    return IntRequantDispatched {
                        small_par_fallback: Some($w_avx512f),
                        ..$d_vnni
                    };
                }
                if x86_isa_detected!("avx512f") {
                    return $d_avx512f;
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
    requant_i8_avx512f,
    requant_i8_vnni,
    requant_i8_neon,
    requant_i8_simd128,
    RDISP_I8_SCALAR,
    RDISP_I8_FMA,
    RDISP_I8_AVX512F,
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
    requant_u8_avx512f,
    requant_u8_vnni,
    requant_u8_neon,
    requant_u8_simd128,
    RDISP_U8_SCALAR,
    RDISP_U8_FMA,
    RDISP_U8_AVX512F,
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
/// runtime branch. Restricted to `i8`/`u8` since those are `QuantOut`'s only implementors and
/// `QuantOut` itself is `pub(crate)`
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
