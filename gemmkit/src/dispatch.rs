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
//! environment variable `GEMMKIT_REQUIRE_ISA` to `scalar`, `fma`, `avx512`,
//! `avx512vnni`, `avx512bf16`, `neon`, or `simd128` **forces** exactly that kernel
//! (`avx512vnni` selects the `i8` `vpdpbusd` dot kernel, `avx512bf16` the `bf16`
//! `vdpbf16ps` dot kernel, and the plain AVX-512 path for every other type); if
//! the CPU (or an emulator such as
//! Intel SDE) does not report the required feature — or the requested ISA does
//! not exist on this target architecture — dispatch **panics** rather than
//! falling back, so a CI job that means to exercise a given kernel fails loudly
//! instead of silently testing a different one. (`neon` is only valid on
//! aarch64, where it is baseline; `fma`/`avx512*` only on x86; `simd128` only on a
//! `wasm32` build compiled with `-C target-feature=+simd128` — there it asserts the
//! SIMD path is live rather than silently degrading to the scalar fallback when the
//! flag was forgotten.) `auto`/unset is the normal auto-selecting behavior. The value
//! is read once (the choice is memoized), so set it in the process environment before
//! the first GEMM call.

#![cfg_attr(
    not(feature = "std"),
    allow(
        clippy::assertions_on_constants,
        clippy::nonminimal_bool,
        clippy::eq_op
    )
)]

#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(feature = "half")]
use half::{bf16, f16};

/// `c32` / `c64` element-type aliases (the complex-GEMM dispatch types).
#[cfg(feature = "complex")]
type C32 = num_complex::Complex<f32>;
#[cfg(feature = "complex")]
type C64 = num_complex::Complex<f64>;

use crate::driver;
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::Bf16DotGemm;
use crate::kernel::FloatGemm;
#[cfg(feature = "int8")]
use crate::kernel::IntGemm;
#[cfg(feature = "int8")]
use crate::kernel::IntGemmQ;
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::IntGemmVnni;
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::kernel::IntGemmVnniQ;
#[cfg(any(feature = "int8", feature = "half"))]
use crate::kernel::KernelFamily;
#[cfg(feature = "half")]
use crate::kernel::MixedGemm;
#[cfg(feature = "int8")]
use crate::kernel::epilogue::{BiasDim, KRequantize};
use crate::kernel::epilogue::{BiasSpec, Epilogue, FusedEpi};
use crate::parallel::Parallelism;
#[cfg(feature = "int8")]
use crate::parallel::Ptr;
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;
use crate::scalar::{Float, Scalar};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::simd::Avx512Bf16;
#[cfg(all(feature = "int8", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::simd::Avx512Vnni;
#[cfg(any(feature = "half", feature = "int8", feature = "complex"))]
use crate::simd::KernelSimd;
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

/// x86 ISA probe for the `select_*` ladders: the runtime `is_x86_feature_detected!`
/// with `std`, else a compile-time `cfg!(target_feature = …)` — off `std` there is no
/// runtime CPU detection (`raw-cpuid` is `std`-gated), so a no_std build runs whatever
/// its compile-time target-features guarantee.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! x86_isa_detected {
    ($feat:tt) => {
        is_x86_feature_detected!($feat)
    };
}
#[cfg(all(not(feature = "std"), any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! x86_isa_detected {
    ($feat:tt) => {
        cfg!(target_feature = $feat)
    };
}

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

/// One GEMM problem for the pointer-array batched API ([`crate::gemm_batched_ptr_unchecked`]):
/// `C <- alpha·A·B + beta·C` over raw pointers and `isize` strides, so each element of a batch can
/// have its own shape and live anywhere in memory (unlike the strided [`crate::gemm_batched`],
/// which shares one shape and steps by a fixed batch stride).
#[derive(Copy, Clone)]
pub struct GemmProblem<T> {
    /// Rows of A and C.
    pub m: usize,
    /// Shared dimension (cols of A, rows of B).
    pub k: usize,
    /// Cols of B and C.
    pub n: usize,
    /// Product scale.
    pub alpha: T,
    /// LHS base pointer.
    pub a: *const T,
    /// LHS row stride.
    pub rsa: isize,
    /// LHS column stride.
    pub csa: isize,
    /// RHS base pointer.
    pub b: *const T,
    /// RHS row stride.
    pub rsb: isize,
    /// RHS column stride.
    pub csb: isize,
    /// Accumulator scale.
    pub beta: T,
    /// Output base pointer.
    pub c: *mut T,
    /// Output row stride.
    pub rsc: isize,
    /// Output column stride.
    pub csc: isize,
}

impl<T: Copy> GemmProblem<T> {
    /// The equivalent internal [`Task`] (a field move; no allocation).
    #[inline]
    pub(crate) fn task(&self) -> Task<T> {
        Task {
            m: self.m,
            k: self.k,
            n: self.n,
            alpha: self.alpha,
            a: self.a,
            rsa: self.rsa,
            csa: self.csa,
            b: self.b,
            rsb: self.rsb,
            csb: self.csb,
            beta: self.beta,
            c: self.c,
            rsc: self.rsc,
            csc: self.csc,
        }
    }
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

/// Element types gemmkit can dispatch: `f32`/`f64` (homogeneous float) and
/// `f16`/`bf16` (mixed precision, `Acc = f32`).
///
/// The bound is [`Scalar`], **not** `Float<Acc = Self>`, so the accumulator may
/// differ from the element type (the mixed-precision seam). The methods below supply
/// what isn't expressible generically — the degenerate `beta`-scale and which kernel
/// family to pack/dispatch through — keeping the driver and public API type-agnostic.
pub trait GemmScalar: Scalar {
    /// Mirror of [`crate::kernel::KernelFamily::OUT_IS_ACC`]: `true` for `f32`/`f64`,
    /// `false` for `f16`/`bf16`. The prepack constructor reads it so the prepacked and
    /// plain paths block with the same `kc`.
    const OUT_IS_ACC: bool;

    /// `C <- beta·C` over the strided output — the degenerate path when the `A·B` term
    /// vanishes (`k == 0` or `alpha == 0`). Narrow types scale in `f32` and round back.
    ///
    /// # Safety
    /// `c` valid for the `m × n` region at `rsc`/`csc`.
    #[doc(hidden)]
    unsafe fn scale_c(beta: Self, c: *mut Self, m: usize, n: usize, rsc: isize, csc: isize);

    /// Pack a full RHS into the prepacked micropanel buffer through this type's kernel
    /// family. The layout is family-independent, but the family *type* differs
    /// (`FloatGemm` vs `MixedGemm`), so the call is dispatched here rather than
    /// hard-wired in [`crate::prepack_rhs`].
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

    /// The selected kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`]. The
    /// prepack constructor rounds the packed depth up to it so the prepacked buffer's
    /// layout matches the consuming kernel's. `1` for every family except the bf16
    /// `vdpbf16ps` dot kernel (`2`), so the default suits `f32`/`f64`/`f16`.
    #[doc(hidden)]
    fn rhs_depth_multiple() -> usize {
        1
    }

    /// Run the ISA-dispatched **fused-epilogue** kernel for this type. The default is
    /// unreachable — only the real floats (`f32`/`f64`, the sealed [`FusedScalar`] set) have
    /// a fused path; `f16`/`bf16` keep the default and never reach it (the public API bound
    /// forbids them).
    ///
    /// # Safety
    /// `task`'s pointers valid and `c` not aliasing `a`/`b`; `epi`'s bias valid and disjoint
    /// from `c` (validated by the API layer).
    #[doc(hidden)]
    unsafe fn dispatch_fused(
        _task: Task<Self>,
        _epi: FusedEpi<Self>,
        _par: Parallelism,
        _ws: &mut Workspace,
    ) {
        unreachable!("fused epilogue is dispatched only for f32/f64 (the FusedScalar bound)")
    }
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

/// Orientation normalization shared by every dispatch path (float / mixed [`Task`],
/// integer [`IntTask`], requantizing [`RequantTask`]): if `C` is row-major-ish
/// (`|csc| < |rsc|`), compute `Cᵀ = Bᵀ·Aᵀ` so the kernel writes columns contiguously
/// (`rsc == 1`), swapping `m↔n`, the `A`/`B` pointers/strides, and `rsc↔csc`. Returns
/// `true` if it swapped, so callers can flip any co-varying policy (bias axis, conj
/// flags). Generic over the element pointer type `L` (the three tasks differ only there).
#[inline]
#[allow(clippy::too_many_arguments)]
fn orient_swap<L>(
    m: &mut usize,
    n: &mut usize,
    a: &mut *const L,
    rsa: &mut isize,
    csa: &mut isize,
    b: &mut *const L,
    rsb: &mut isize,
    csb: &mut isize,
    rsc: &mut isize,
    csc: &mut isize,
) -> bool {
    if csc.unsigned_abs() < rsc.unsigned_abs() {
        core::mem::swap(m, n);
        core::mem::swap(a, b); // new A = old B (same element type L)
        core::mem::swap(rsa, csb); // new rsa = old csb, new csb = old rsa
        core::mem::swap(csa, rsb); // new csa = old rsb, new rsb = old csa
        core::mem::swap(rsc, csc);
        true
    } else {
        false
    }
}

/// Orientation swap for the homogeneous float / mixed [`Task`] path (see [`orient_swap`]).
#[inline]
fn orient_transpose<T>(t: &mut Task<T>) -> bool {
    orient_swap(
        &mut t.m, &mut t.n, &mut t.a, &mut t.rsa, &mut t.csa, &mut t.b, &mut t.rsb, &mut t.csb,
        &mut t.rsc, &mut t.csc,
    )
}

/// `true` when a post-swap [`Task`] should take the horizontal `small_mn` path: small `m,n`
/// with a long contraction and both operands streaming contiguously along `k` (A rows
/// unit-stride `csa == 1`, B columns unit-stride `rsb == 1`). Shared by the float / mixed /
/// bf16-dot entries — the gate has been re-tuned as one unit, so it lives in one place.
#[inline]
fn small_mn_eligible<T>(t: &Task<T>) -> bool {
    t.m <= tuning::small_mn_dim()
        && t.n <= tuning::small_mn_dim()
        && t.k > tuning::small_k_threshold()
        && t.csa == 1
        && t.rsb == 1
}

/// `C <- beta·C` for a **homogeneous float** type (`f32`/`f64`): in-place scale,
/// `beta == 0` overwriting to zero without reading C. The float `GemmScalar::scale_c`
/// forwards here; narrow types use [`scale_c_narrow`].
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
#[cfg(feature = "half")]
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

/// **Fused-epilogue** driver entry for a concrete `(type, ISA, tile)`: mirror of [`run_typed`]
/// but with the special-path routing **deleted**. gemv / small_mn / small_k have no epilogue
/// hook — routing there would silently drop the epilogue — so every fused shape goes through
/// the general driver (correct for any shape). The orientation swap flips the bias axis: a
/// row-major-ish C makes the engine compute `Cᵀ = Bᵀ·Aᵀ`, swapping `m↔n`, so a user per-row
/// bias becomes per-col in the driver frame (a field write, not a new monomorphization).
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
    T: FusedScalar,
    S: SimdOps<T>,
{
    unsafe {
        let swap = orient_transpose(&mut t);
        if swap {
            epi.bias = match epi.bias {
                BiasSpec::None => BiasSpec::None,
                BiasSpec::Row(p) => BiasSpec::Col(p),
                BiasSpec::Col(p) => BiasSpec::Row(p),
            };
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

/// Mixed-precision driver entry for a concrete `(narrow type, family, ISA, tile)`. Mirror
/// of [`run_typed`] driving a narrow-in / `f32`-accumulate family: no gemv special path (the
/// general driver handles those shapes), the same orientation swap, and `alpha`/`beta`
/// **widened to the `f32` accumulator** before the driver call. `Fam` selects the general-
/// driver kernel — `MixedGemm<N>` for the widen path, `Bf16DotGemm` for the `vdpbf16ps` dot
/// path — while the `small_mn` / small-`k` reroutes deliberately stay on `MixedGemm<N>`
/// (both special paths bypass any dot kernel: a tiny output folds nothing and the dot pack's
/// `DEPTH_MULTIPLE` is pure loss there).
///
/// # Safety
/// As [`run_typed`].
#[cfg(feature = "half")]
#[inline]
unsafe fn run_typed_mixed<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    mut t: Task<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        orient_transpose(&mut t);
        // Small `m,n` + long `k` + contiguous-along-`k` layout: the horizontal path, widening
        // `N → f32` on load and accumulating in `f32` (see [`run_typed`]'s float gate).
        if small_mn_eligible(&t) {
            small_mn::run_mixed::<N, S>(
                simd,
                t.m,
                t.k,
                t.n,
                par,
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
            );
            return;
        }
        // Skinny / low-depth shape through the widen microkernel (see [`run_typed`]).
        if t.k <= tuning::small_k_threshold() {
            small_k::run::<MixedGemm<N>, S, MR_REG, NR>(
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
            return;
        }
        driver::run::<Fam, S, MR_REG, NR>(
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

/// Prepacked-RHS mixed-precision entry (mirror of [`run_packed_typed`] for a narrow-in /
/// `f32`-accumulate family `Fam`); no swap, `alpha`/`beta` widened to `f32`.
///
/// # Safety
/// As [`run_packed_typed`].
#[cfg(feature = "half")]
#[inline]
unsafe fn run_packed_typed_mixed<N, Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    req: PackedConsume<N>,
    par: Parallelism,
    ws: &mut Workspace,
) where
    N: NarrowFloat,
    Fam: KernelFamily<Lhs = N, Rhs = N, Acc = f32, Out = N>,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        debug_assert_eq!(NR, req.nr, "prepacked RHS panel width != kernel NR");
        driver::run_packed_rhs::<Fam, S, MR_REG, NR>(
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

/// The sealed element-type bound for the fused-epilogue public API: exactly the real floats
/// `f32`/`f64`. It is a superset of [`GemmScalar`] (for dispatch) plus `Float<Acc = Self>`
/// and `PartialOrd` (for the [`FusedEpi`] arithmetic and the `ReLU` comparisons), and it is
/// sealed (a private supertrait) so downstream crates cannot widen the fused surface.
pub trait FusedScalar: GemmScalar + Float<Acc = Self> + PartialOrd + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

impl FusedScalar for f32 {}
impl FusedScalar for f64 {}

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
        // element-wise in the user frame (bias axes as the caller specified).
        if task.k == 0 || task.alpha == T::ZERO {
            fused_degenerate(&task, &epi);
            return;
        }
        T::dispatch_fused(task, epi, par, ws);
    }
}

/// The degenerate fused epilogue `C[i,j] <- apply(beta·C[i,j], i, j)` in the user frame.
///
/// # Safety
/// `c` valid for the `m × n` region; `epi`'s bias valid for the problem's `m`/`n`.
unsafe fn fused_degenerate<T: FusedScalar>(t: &Task<T>, epi: &FusedEpi<T>) {
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

// ---- mixed-precision (f16 / bf16) entry points: same tiles as f32 (the
// accumulator is f32, so the register budget matches) ----

#[cfg(feature = "half")]
unsafe fn gemm_f16_scalar(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_bf16_scalar(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(ScalarTok, t, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_f16_scalar_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, ScalarTok, 4, 4>(ScalarTok, r, par, ws) }
}
#[cfg(feature = "half")]
unsafe fn gemm_bf16_scalar_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        run_packed_typed_mixed::<bf16, MixedGemm<bf16>, ScalarTok, 4, 4>(ScalarTok, r, par, ws)
    }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_fma(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator → MR = 2*8 = 16, NR = 6 (the f32 FMA tile).
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_fma(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_fma_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_fma_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Fma, 2, 6>(Fma, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_avx512(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    // f32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile).
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(Avx512, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_f16_avx512_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Avx512, 2, 12>(Avx512, r, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_dot(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    // bf16 dot: f32 accumulator → MR = 2*16 = 32, NR = 12 (the f32 AVX-512 tile). The
    // `Bf16DotGemm` family swaps in the `vdpbf16ps` pack + inner loop; the shared
    // `run_typed_mixed` routes small_mn / small_k through `MixedGemm<bf16>` as before.
    unsafe { run_typed_mixed::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, t, par, ws) }
}
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn gemm_bf16_avx512_dot_packed(
    r: PackedConsume<bf16>,
    par: Parallelism,
    ws: &mut Workspace,
) {
    unsafe {
        run_packed_typed_mixed::<bf16, Bf16DotGemm, Avx512Bf16, 2, 12>(Avx512Bf16, r, par, ws)
    }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_f16_neon_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "aarch64"))]
unsafe fn gemm_bf16_neon_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Neon, 4, 4>(Neon, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f16_simd128(t: Task<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_bf16_simd128(t: Task<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_typed_mixed::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(Simd128, t, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_f16_simd128_packed(r: PackedConsume<f16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<f16, MixedGemm<f16>, Simd128, 2, 4>(Simd128, r, par, ws) }
}
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn gemm_bf16_simd128_packed(r: PackedConsume<bf16>, par: Parallelism, ws: &mut Workspace) {
    unsafe { run_packed_typed_mixed::<bf16, MixedGemm<bf16>, Simd128, 2, 4>(Simd128, r, par, ws) }
}

type GemmFn<T> = unsafe fn(Task<T>, Parallelism, &mut Workspace);
type PackedFn<T> = unsafe fn(PackedConsume<T>, Parallelism, &mut Workspace);
/// The fused-epilogue kernel entry: a plain [`Task`] plus the runtime-composed
/// [`FusedEpi`]. `Some` only for the real floats (`f32`/`f64`); the sealed
/// [`FusedScalar`] bound makes the `None` (`f16`/`bf16`) case unreachable.
type FusedFn<T> = unsafe fn(Task<T>, FusedEpi<T>, Parallelism, &mut Workspace);

/// The memoized dispatch slot for one element type: the standard kernel, the
/// prepacked-RHS kernel, the fused-epilogue kernel, and the microtile `(mr, nr)` they
/// share. Bundling them keeps adding an ISA a single `select_*` ladder arm. `mr`/`nr`
/// mirror the tile constants in the wrappers above and feed `prepack_rhs` (via `rhs_tile`)
/// so the buffer and the consume path agree on the blocking geometry.
#[derive(Copy, Clone)]
struct Dispatched<T> {
    run: GemmFn<T>,
    run_packed: PackedFn<T>,
    /// Fused-epilogue entry (`bias`/activation), or `None` for a type with no fused path.
    run_fused: Option<FusedFn<T>>,
    mr: usize,
    nr: usize,
    /// The dispatched kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`].
    /// `1` for every widen/homogeneous kernel; `2` for the bf16 `vdpbf16ps` dot kernel.
    /// The prepack constructor rounds the packed depth up to it (via [`GemmScalar`]).
    /// Read only by the `bf16` prepack path, so it is dead code without the `half` feature.
    #[cfg_attr(not(feature = "half"), allow(dead_code))]
    depth_multiple: usize,
}

// One descriptor per (type, ISA). `mr = MR_REG·LANES`, `nr = NR` — mirrors the
// tile in each wrapper's comment (scalar 4×4; FMA 16×6 / f64 8×6; AVX-512 32×12 /
// f64 16×12; NEON 16×4 / f64 8×4).
const DISP_F32_SCALAR: Dispatched<f32> = Dispatched {
    run: gemm_f32_scalar,
    run_packed: gemm_f32_scalar_packed,
    run_fused: Some(gemm_f32_scalar_fused),
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
const DISP_F64_SCALAR: Dispatched<f64> = Dispatched {
    run: gemm_f64_scalar,
    run_packed: gemm_f64_scalar_packed,
    run_fused: Some(gemm_f64_scalar_fused),
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_FMA: Dispatched<f32> = Dispatched {
    run: gemm_f32_fma,
    run_packed: gemm_f32_fma_packed,
    run_fused: Some(gemm_f32_fma_fused),
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_FMA: Dispatched<f64> = Dispatched {
    run: gemm_f64_fma,
    run_packed: gemm_f64_fma_packed,
    run_fused: Some(gemm_f64_fma_fused),
    mr: 8,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F32_AVX512: Dispatched<f32> = Dispatched {
    run: gemm_f32_avx512,
    run_packed: gemm_f32_avx512_packed,
    run_fused: Some(gemm_f32_avx512_fused),
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const DISP_F64_AVX512: Dispatched<f64> = Dispatched {
    run: gemm_f64_avx512,
    run_packed: gemm_f64_avx512_packed,
    run_fused: Some(gemm_f64_avx512_fused),
    mr: 16,
    nr: 12,
    depth_multiple: 1,
};

#[cfg(target_arch = "aarch64")]
const DISP_F32_NEON: Dispatched<f32> = Dispatched {
    run: gemm_f32_neon,
    run_packed: gemm_f32_neon_packed,
    run_fused: Some(gemm_f32_neon_fused),
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(target_arch = "aarch64")]
const DISP_F64_NEON: Dispatched<f64> = Dispatched {
    run: gemm_f64_neon,
    run_packed: gemm_f64_neon_packed,
    run_fused: Some(gemm_f64_neon_fused),
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F32_SIMD128: Dispatched<f32> = Dispatched {
    run: gemm_f32_simd128,
    run_packed: gemm_f32_simd128_packed,
    run_fused: Some(gemm_f32_simd128_fused),
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F64_SIMD128: Dispatched<f64> = Dispatched {
    run: gemm_f64_simd128,
    run_packed: gemm_f64_simd128_packed,
    run_fused: Some(gemm_f64_simd128_fused),
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(feature = "half")]
const DISP_F16_SCALAR: Dispatched<f16> = Dispatched {
    run: gemm_f16_scalar,
    run_packed: gemm_f16_scalar_packed,
    run_fused: None,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(feature = "half")]
const DISP_BF16_SCALAR: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_scalar,
    run_packed: gemm_bf16_scalar_packed,
    run_fused: None,
    mr: 4,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_FMA: Dispatched<f16> = Dispatched {
    run: gemm_f16_fma,
    run_packed: gemm_f16_fma_packed,
    run_fused: None,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_FMA: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_fma,
    run_packed: gemm_bf16_fma_packed,
    run_fused: None,
    mr: 16,
    nr: 6,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_F16_AVX512: Dispatched<f16> = Dispatched {
    run: gemm_f16_avx512,
    run_packed: gemm_f16_avx512_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512,
    run_packed: gemm_bf16_avx512_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
const DISP_BF16_AVX512_DOT: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_avx512_dot,
    run_packed: gemm_bf16_avx512_dot_packed,
    run_fused: None,
    mr: 32,
    nr: 12,
    // k-pair-interleaved pack → the prepack buffer rounds its depth up to 2.
    depth_multiple: 2,
};

#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_F16_NEON: Dispatched<f16> = Dispatched {
    run: gemm_f16_neon,
    run_packed: gemm_f16_neon_packed,
    run_fused: None,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "aarch64"))]
const DISP_BF16_NEON: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_neon,
    run_packed: gemm_bf16_neon_packed,
    run_fused: None,
    mr: 16,
    nr: 4,
    depth_multiple: 1,
};

#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_F16_SIMD128: Dispatched<f16> = Dispatched {
    run: gemm_f16_simd128,
    run_packed: gemm_f16_simd128_packed,
    run_fused: None,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};
#[cfg(all(feature = "half", target_arch = "wasm32", target_feature = "simd128"))]
const DISP_BF16_SIMD128: Dispatched<bf16> = Dispatched {
    run: gemm_bf16_simd128,
    run_packed: gemm_bf16_simd128_packed,
    run_fused: None,
    mr: 8,
    nr: 4,
    depth_multiple: 1,
};

/// An explicitly requested kernel, parsed from `GEMMKIT_REQUIRE_ISA`.
///
/// The non-`Auto` variants are constructed only by the `std` `forced_isa` (env-var
/// parsing); the no-`std` `forced_isa` always yields `Auto`. The `select_*` ladders
/// still match on every variant, so they must remain in the type — hence the
/// `dead_code` allowance for the no-`std` build rather than `#[cfg]`-ing them out.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
#[derive(Copy, Clone, PartialEq, Eq)]
enum ForcedIsa {
    /// No override: auto-select the best available ISA (the default).
    Auto,
    /// Scalar: the fallback path when no other ISA is available
    Scalar,
    /// FMA: the `fma`-based (AVX2) widen kernel
    Fma,
    /// AVX-512 foundation (`avx512f`): the widen kernel
    Avx512F,
    /// AVX-512 VNNI: the `i8` `vpdpbusd` dot kernel
    Avx512Vnni,
    /// AVX-512 BF16: the `bf16` `vdpbf16ps` dot kernel
    Avx512Bf16,
    /// NEON: the AArch64 kernel
    Neon,
    /// WebAssembly `simd128`. Baseline-by-cfg like `Neon`, but `simd128` is an easily-forgotten
    /// compile-time `-C target-feature=+simd128`; pinning it makes a build **assert** the SIMD
    /// path is live (panics if absent) instead of silently falling back to scalar.
    Simd128,
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
                ForcedIsa::Avx512F
            } else if t.eq_ignore_ascii_case("avx512vnni") || t.eq_ignore_ascii_case("vnni") {
                ForcedIsa::Avx512Vnni
            } else if t.eq_ignore_ascii_case("avx512bf16") || t.eq_ignore_ascii_case("bf16") {
                ForcedIsa::Avx512Bf16
            } else if t.eq_ignore_ascii_case("neon") {
                ForcedIsa::Neon
            } else if t.eq_ignore_ascii_case("simd128") || t.eq_ignore_ascii_case("wasm") {
                ForcedIsa::Simd128
            } else {
                panic!(
                    "GEMMKIT_REQUIRE_ISA: unknown value `{t}` (expected scalar|fma|avx512|avx512vnni|avx512bf16|neon|simd128|auto)"
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

/// `f16` ISA selection. The FMA path additionally needs **F16C**
/// (`vcvtph2ps`/`vcvtps2ph`) — checked here so an FMA selection on an F16C-less part
/// falls back rather than faulting. AVX-512 covers `f16` within `avx512f`.
#[cfg(feature = "half")]
fn select_f16() -> Dispatched<f16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_F16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma") && x86_isa_detected!("f16c"),
                "GEMMKIT_REQUIRE_ISA=fma for f16, but this CPU does not report avx2+fma+f16c"
            );
            return DISP_F16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_F16_AVX512;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_F16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_F16_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if x86_isa_detected!("avx512f") {
            return DISP_F16_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") && x86_isa_detected!("f16c") {
            return DISP_F16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_F16_NEON
    }
    // `simd128` on wasm32, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_F16_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_F16_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_F16_SCALAR
    }
}

/// `bf16` ISA selection. The FMA path uses only AVX2 integer ops (shift / pack), so
/// no F16C is required; AVX-512 covers `bf16` within `avx512f`.
#[cfg(feature = "half")]
fn select_bf16() -> Dispatched<bf16> {
    match forced_isa() {
        ForcedIsa::Scalar => return DISP_BF16_SCALAR,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Fma => {
            assert!(
                x86_isa_detected!("avx2") && x86_isa_detected!("fma"),
                "GEMMKIT_REQUIRE_ISA=fma, but this CPU/emulator does not report avx2+fma"
            );
            return DISP_BF16_FMA;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni => {
            assert!(
                x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512, but this CPU/emulator does not report avx512f"
            );
            return DISP_BF16_AVX512;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        ForcedIsa::Avx512Bf16 => {
            assert!(
                x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f"),
                "GEMMKIT_REQUIRE_ISA=avx512bf16, but this CPU/emulator does not report avx512f+bf16"
            );
            return DISP_BF16_AVX512_DOT;
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        ForcedIsa::Fma | ForcedIsa::Avx512F | ForcedIsa::Avx512Vnni | ForcedIsa::Avx512Bf16 => {
            panic!("GEMMKIT_REQUIRE_ISA: requested SIMD ISA is unavailable on this target")
        }
        #[cfg(target_arch = "aarch64")]
        ForcedIsa::Neon => return DISP_BF16_NEON,
        #[cfg(not(target_arch = "aarch64"))]
        ForcedIsa::Neon => panic!("GEMMKIT_REQUIRE_ISA=neon, but this target is not aarch64"),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        ForcedIsa::Simd128 => return DISP_BF16_SIMD128,
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        ForcedIsa::Simd128 => panic!(
            "GEMMKIT_REQUIRE_ISA=simd128, but this build is not wasm32 with -C target-feature=+simd128"
        ),
        ForcedIsa::Auto => {}
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // bf16 dot kernel first - `vdpbf16ps` ~doubles bf16
        if x86_isa_detected!("avx512bf16") && x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512_DOT;
        }
        if x86_isa_detected!("avx512f") {
            return DISP_BF16_AVX512;
        }
        if x86_isa_detected!("avx2") && x86_isa_detected!("fma") {
            return DISP_BF16_FMA;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        DISP_BF16_NEON
    }
    // `simd128` on wasm32, else scalar
    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        {
            DISP_BF16_SIMD128
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            DISP_BF16_SCALAR
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        DISP_BF16_SCALAR
    }
}

/// Emit the memoized dispatch accessor for one element type: a `#[cfg(std)]`
/// `OnceLock<$ty>` plus a `fn $accessor() -> $ty` that runs `$select` **once** under `std`
/// (feature detection is memoized) and directly on each call without `std`. The optional
/// trailing `$feat` additionally gates the accessor and the `OnceLock` on that feature (the
/// static is always further gated by `std`). Every `dispatched_*` slot shares this shape.
macro_rules! memoized_select {
    ($static:ident, $accessor:ident, $ty:ty, $select:ident, $doc:literal) => {
        #[cfg(feature = "std")]
        static $static: OnceLock<$ty> = OnceLock::new();
        #[doc = $doc]
        #[inline]
        fn $accessor() -> $ty {
            #[cfg(feature = "std")]
            {
                *$static.get_or_init($select)
            }
            #[cfg(not(feature = "std"))]
            {
                $select()
            }
        }
    };
    ($static:ident, $accessor:ident, $ty:ty, $select:ident, $doc:literal, $feat:literal) => {
        #[cfg(all(feature = "std", feature = $feat))]
        static $static: OnceLock<$ty> = OnceLock::new();
        #[doc = $doc]
        #[cfg(feature = $feat)]
        #[inline]
        fn $accessor() -> $ty {
            #[cfg(feature = "std")]
            {
                *$static.get_or_init($select)
            }
            #[cfg(not(feature = "std"))]
            {
                $select()
            }
        }
    };
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
memoized_select!(
    GEMM_F16,
    dispatched_f16,
    Dispatched<f16>,
    select_f16,
    "The memoized dispatch descriptor for `f16` (selection runs once).",
    "half"
);
memoized_select!(
    GEMM_BF16,
    dispatched_bf16,
    Dispatched<bf16>,
    select_bf16,
    "The memoized dispatch descriptor for `bf16` (selection runs once).",
    "half"
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
                unsafe {
                    ($disp()
                        .run_fused
                        .expect(concat!($name, " fused kernel is always present")))(
                        t, epi, par, ws
                    )
                }
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

#[cfg(feature = "half")]
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

#[cfg(feature = "half")]
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
        unsafe {
            // The dot kernel packs k-pair-interleaved; pack through *its* family so the
            // prepacked layout matches what the consuming call reads. Identified by the
            // depth multiple (> 1 only for the bf16 dot descriptor).
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if dispatched_bf16().depth_multiple > 1 {
                driver::pack_rhs_full::<Bf16DotGemm>(dst, b, rsb, csb, k, n, kc, nc, nr);
                return;
            }
            driver::pack_rhs_full::<MixedGemm<bf16>>(dst, b, rsb, csb, k, n, kc, nc, nr);
        }
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
        unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if dispatched_bf16().depth_multiple > 1 {
                driver::pack_lhs_full::<Bf16DotGemm>(dst, a, rsa, csa, m, k, kc, nc, nr);
                return;
            }
            driver::pack_lhs_full::<MixedGemm<bf16>>(dst, a, rsa, csa, m, k, kc, nc, nr);
        }
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
    #[inline]
    fn rhs_depth_multiple() -> usize {
        dispatched_bf16().depth_multiple
    }
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
/// Separate from [`GemmScalar`] because complex carries the conj op-family. The
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
