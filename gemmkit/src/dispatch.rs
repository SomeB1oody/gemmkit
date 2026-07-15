//! Runtime ISA dispatch (layer L7)
//!
//! Each element type has one `OnceLock<fn>`: feature detection runs once, the
//! winning monomorphized entry point is cached, and later calls are a plain
//! indirect call. **No `transmute`, no `AtomicPtr<()>`**: the slot is a typed
//! function pointer. Adding an ISA is 1 line in the `select_*` ladder plus the
//! 1-line `#[allow]`-free wrapper; adding a type is a new `OnceLock` + impl,
//! not a new crate
//!
//! ## Pinning the kernel: `GEMMKIT_REQUIRE_ISA`
//!
//! By default the best available ISA is selected at runtime. Setting the
//! environment variable `GEMMKIT_REQUIRE_ISA` to `scalar`, `fma`, `avx512`,
//! `avx512vnni`, `avx512bf16`, `neon`, or `simd128` **forces** exactly that kernel
//! (`avx512vnni` selects the `i8` `vpdpbusd` dot kernel, `avx512bf16` the `bf16`
//! `vdpbf16ps` dot kernel, and the plain AVX-512 path for every other type); if
//! the CPU (or an emulator such as Intel SDE) does not report the required
//! feature, or the requested ISA does not exist on this target architecture,
//! dispatch **panics** rather than falling back, so a CI job that means to
//! exercise a given kernel fails loudly instead of silently testing a different
//! one. (`neon` is only valid on aarch64, where it is baseline; `fma`/`avx512*`
//! only on x86; `simd128` only on a `wasm32` build compiled with
//! `-C target-feature=+simd128`: there it asserts the SIMD path is live rather
//! than silently degrading to the scalar fallback when the flag was forgotten.)
//! `auto`/unset is the normal auto-selecting behavior. The value is read once
//! (the choice is memoized), so set it in the process environment before the
//! first GEMM call

#![cfg_attr(
    not(feature = "std"),
    allow(
        clippy::assertions_on_constants,
        clippy::nonminimal_bool,
        clippy::eq_op
    )
)]

// GEMMKIT_REQUIRE_ISA parsing and the memoized-select machinery shared by every family
#[macro_use]
mod isa;

// c32/c64 complex GEMM dispatch (conjA/conjB, per-ISA wrappers, ComplexScalar impls)
#[cfg(feature = "complex")]
mod complex;
// f32/f64 homogeneous-float dispatch: driver entries, per-ISA wrappers, GemmScalar/FusedScalar impls
mod float;
// Integer (i8 -> i32) GEMM dispatch and the fused i8 requantizing path
#[cfg(feature = "int8")]
mod int;
// f16/bf16 mixed-precision dispatch (Acc = f32)
#[cfg(feature = "half")]
mod mixed;

#[cfg(feature = "complex")]
pub use complex::ComplexScalar;
#[cfg(feature = "complex")]
pub(crate) use complex::execute_complex;
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub(crate) use complex::execute_complex_fused;
#[cfg(feature = "epilogue")]
pub use float::FusedScalar;
#[cfg(feature = "epilogue")]
pub(crate) use float::execute_fused;
#[cfg(feature = "int8")]
pub(crate) use int::{IntTask, execute_int};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) use int::{RequantTask, execute_int_requant};

#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::FusedEpi;
use crate::parallel::Parallelism;
use crate::scalar::{Float, Scalar};
use crate::tuning;
use crate::workspace::Workspace;

/// A fully described GEMM problem (`C <- alpha*A*B + beta*C`) with raw pointers
/// and `isize` strides. This is the homogeneous-type dispatch boundary
#[derive(Copy, Clone)]
pub struct Task<T> {
    /// Rows of A and C
    pub m: usize,
    /// Shared dimension (cols of A, rows of B)
    pub k: usize,
    /// Cols of B and C
    pub n: usize,
    /// Product scale
    pub alpha: T,
    /// LHS base pointer (element `(0,0)`)
    pub a: *const T,
    /// LHS row / column strides
    pub rsa: isize,
    pub csa: isize,
    /// RHS base pointer
    pub b: *const T,
    /// RHS row / column strides
    pub rsb: isize,
    pub csb: isize,
    /// Accumulator scale
    pub beta: T,
    /// Output base pointer
    pub c: *mut T,
    /// Output row / column strides
    pub rsc: isize,
    pub csc: isize,
}

/// One GEMM problem for the pointer-array batched API ([`crate::gemm_batched_ptr_unchecked`]):
/// `C <- alpha*A*B + beta*C` over raw pointers and `isize` strides, so each element of a batch can
/// have its own shape and live anywhere in memory (unlike the strided [`crate::gemm_batched`],
/// which shares one shape and steps by a fixed batch stride)
#[derive(Copy, Clone)]
pub struct GemmProblem<T> {
    /// Rows of A and C
    pub m: usize,
    /// Shared dimension (cols of A, rows of B)
    pub k: usize,
    /// Cols of B and C
    pub n: usize,
    /// Product scale
    pub alpha: T,
    /// LHS base pointer
    pub a: *const T,
    /// LHS row stride
    pub rsa: isize,
    /// LHS column stride
    pub csa: isize,
    /// RHS base pointer
    pub b: *const T,
    /// RHS row stride
    pub rsb: isize,
    /// RHS column stride
    pub csb: isize,
    /// Accumulator scale
    pub beta: T,
    /// Output base pointer
    pub c: *mut T,
    /// Output row stride
    pub rsc: isize,
    /// Output column stride
    pub csc: isize,
}

impl<T: Copy> GemmProblem<T> {
    /// The equivalent internal [`Task`] (a field move; no allocation)
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

/// A GEMM whose RHS is already prepacked: `C <- alpha*A*(prepacked B) + beta*C`.
/// Carries the blocking geometry the buffer was packed for (`nr`, `kc`, `nc`),
/// which the driver reads back verbatim so a reused panel always matches its
/// tiling
///
/// `pub` (like [`Task`]) only so it can appear in the doc-hidden [`GemmScalar`]
/// methods; the `dispatch` module is private, so it is not nameable externally
pub struct PackedConsume<T> {
    /// Rows of A and C
    pub m: usize,
    /// Shared dimension (cols of A == prepacked B's depth)
    pub k: usize,
    /// Cols of the prepacked B and of C
    pub n: usize,
    /// Product scale
    pub alpha: T,
    /// LHS base pointer + strides
    pub a: *const T,
    pub rsa: isize,
    pub csa: isize,
    /// Prepacked RHS micropanel buffer base (see [`crate::driver::pack_rhs_full`])
    pub packed: *const T,
    /// Blocking geometry baked into `packed` at pack time
    pub nr: usize,
    pub kc: usize,
    pub nc: usize,
    /// Accumulator scale
    pub beta: T,
    /// Output base pointer + strides
    pub c: *mut T,
    pub rsc: isize,
    pub csc: isize,
}

/// Element types gemmkit can dispatch: `f32`/`f64` (homogeneous float) and
/// `f16`/`bf16` (mixed precision, `Acc = f32`)
///
/// The bound is [`Scalar`], **not** `Float<Acc = Self>`, so the accumulator may
/// differ from the element type (the mixed-precision seam). The methods below supply
/// what isn't expressible generically: the degenerate `beta`-scale and which kernel
/// family to pack/dispatch through, keeping the driver and public API type-agnostic
pub trait GemmScalar: Scalar {
    /// Mirror of [`crate::kernel::KernelFamily::OUT_IS_ACC`]: `true` for `f32`/`f64`,
    /// `false` for `f16`/`bf16`. The prepack constructor reads it so the prepacked and
    /// plain paths block with the same `kc`
    const OUT_IS_ACC: bool;

    /// `C <- beta*C` over the strided output: the degenerate path when the `A*B` term
    /// vanishes (`k == 0` or `alpha == 0`). Narrow types scale in `f32` and round back
    ///
    /// # Safety
    /// `c` valid for the `m x n` region at `rsc`/`csc`
    #[doc(hidden)]
    unsafe fn scale_c(beta: Self, c: *mut Self, m: usize, n: usize, rsc: isize, csc: isize);

    /// Pack a full RHS into the prepacked micropanel buffer through this type's kernel
    /// family. The layout is family-independent, but the family *type* differs
    /// (`FloatGemm` vs `MixedGemm`), so the call is dispatched here rather than
    /// hard-wired in [`crate::prepack_rhs`]
    ///
    /// # Safety
    /// As [`crate::driver::pack_rhs_full`]
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

    /// Run the dispatched kernel for this type. Used by the API layer
    ///
    /// # Safety
    /// `task`'s pointers must be valid and `c` must not alias `a`/`b`
    #[doc(hidden)]
    unsafe fn dispatch(task: Task<Self>, par: Parallelism, ws: &mut Workspace);

    /// Run the dispatched prepacked-RHS kernel for this type
    ///
    /// # Safety
    /// `req`'s pointers must be valid, `c` must not alias `a`/`packed`, and
    /// `packed` must have been produced by [`GemmScalar::pack_rhs_full`] for the
    /// geometry recorded in `req`
    #[doc(hidden)]
    unsafe fn dispatch_packed(req: PackedConsume<Self>, par: Parallelism, ws: &mut Workspace);

    /// The selected kernel's microtile `(mr, nr)` = `(MR_REG*LANES, NR)`. Used by
    /// the prepack constructor to compute the buffer's blocking geometry through
    /// the *same* ISA choice the consuming call will make
    #[doc(hidden)]
    fn rhs_tile() -> (usize, usize);

    /// The selected kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`]. The
    /// prepack constructor rounds the packed depth up to it so the prepacked buffer's
    /// layout matches the consuming kernel's. `1` for every family except the bf16
    /// `vdpbf16ps` dot kernel (`2`), so the default suits `f32`/`f64`/`f16`
    #[doc(hidden)]
    fn rhs_depth_multiple() -> usize {
        1
    }

    /// Run the ISA-dispatched **fused-epilogue** kernel for this type. Every fused element type
    /// provides one: the real floats (`f32`/`f64`) via [`crate::dispatch`]'s `float` module, and
    /// the narrow floats (`f16`/`bf16`, `Acc = f32`) via its `mixed` module. It is a required
    /// method: the [`FusedScalar`] bound on the public fused API admits exactly these 4 types
    ///
    /// # Safety
    /// `task`'s pointers valid and `c` not aliasing `a`/`b`; `epi`'s bias valid and disjoint
    /// from `c` (validated by the API layer)
    #[doc(hidden)]
    #[cfg(feature = "epilogue")]
    unsafe fn dispatch_fused(
        task: Task<Self>,
        epi: FusedEpi<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );
}

/// Top-level entry used by the API layer: handle the degenerate cases (here,
/// where the element type is concrete) and then run the ISA-dispatched kernel
///
/// # Safety
/// `task`'s pointers must be valid for the implied regions and `c` must not
/// alias `a`/`b`
pub(crate) unsafe fn execute<T: GemmScalar>(task: Task<T>, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if task.m == 0 || task.n == 0 {
            return;
        }
        // k == 0 or alpha == 0 => the A*B term vanishes: C <- beta*C only
        if task.k == 0 || task.alpha == T::ZERO {
            T::scale_c(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
            return;
        }
        T::dispatch(task, par, ws);
    }
}

/// Top-level entry for the prepacked-RHS path: handle the degenerate cases
/// (the A*B term vanishes => `C <- beta*C`, never touching the packed buffer) and
/// then run the ISA-dispatched prepacked kernel
///
/// # Safety
/// As [`execute`], plus `req.packed` valid for the recorded geometry and not
/// aliasing `c`
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
/// (`|csc| < |rsc|`), compute `C^T = B^T*A^T` so the kernel writes columns contiguously
/// (`rsc == 1`), swapping `m<->n`, the `A`/`B` pointers/strides, and `rsc<->csc`. Returns
/// `true` if it swapped, so callers can flip any co-varying policy (bias axis, conj
/// flags). Generic over the element pointer type `L` (the 3 tasks differ only there)
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

/// Orientation swap for the homogeneous float / mixed [`Task`] path (see [`orient_swap`])
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
/// bf16-dot entries: the gate has been re-tuned as one unit, so it lives in one place
#[inline]
fn small_mn_eligible<T>(t: &Task<T>) -> bool {
    t.m <= tuning::small_mn_dim()
        && t.n <= tuning::small_mn_dim()
        && t.k > tuning::small_k_threshold()
        && t.csa == 1
        && t.rsb == 1
}

/// `C <- beta*C` for a **homogeneous float** type (`f32`/`f64`): in-place scale,
/// `beta == 0` overwriting to zero without reading C. The float `GemmScalar::scale_c`
/// forwards here; narrow types use [`scale_c_narrow`]
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
