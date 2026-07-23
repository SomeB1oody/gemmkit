//! Runtime ISA dispatch (layer L7)
//!
//! Every element type gemmkit supports gets its own `OnceLock`-memoized descriptor: a small
//! struct of monomorphized function pointers (plain, prepacked-RHS, and, under `epilogue`,
//! fused variants) plus the microtile its ISA probe committed to. Feature detection runs once
//! per type, on that type's first dispatch; every call after that is a plain indirect call
//! through a typed function pointer (no `transmute`, no `AtomicPtr<()>`, nothing type-erased).
//! Adding an ISA is small and mechanical: a new arm in each affected type's `select_*` ladder
//! plus a thin wrapper (and its packed/fused siblings) delegating to the shared generic driver
//! entry, no new logic. Adding an element type means a new descriptor plus `GemmScalar` impl in
//! its own file under this module, not a new crate
//!
//! ## Pinning the kernel: `GEMMKIT_REQUIRE_ISA`
//!
//! By default the best available ISA is selected at runtime. Setting the environment variable
//! `GEMMKIT_REQUIRE_ISA` to `scalar`, `fma`, `avx512f`, `avx512vnni`, `avx512bf16`, `neon`, or
//! `simd128` **forces** exactly that kernel (`avx512vnni` selects the `i8` `vpdpbusd` dot
//! kernel, `avx512bf16` the `bf16` `vdpbf16ps` dot kernel, and the plain AVX-512F path for every
//! other type); if the CPU (or an emulator such as Intel SDE) does not report the required
//! feature, or the requested ISA does not exist on this target architecture, selection
//! **panics** rather than falling back, so a CI job that means to exercise a given kernel fails
//! loudly instead of silently testing a different one. (`neon` is only valid on aarch64, where
//! it is baseline; `fma`/`avx512*` only on x86; `simd128` only on a `wasm32` build compiled with
//! `-C target-feature=+simd128`: there it asserts the SIMD path is live rather than silently
//! degrading to the scalar fallback when the flag was forgotten.) `auto`, unset, or empty is the
//! normal auto-selecting behavior. Each type reads the variable at most once, the first time
//! that type's descriptor is built, since the choice its `select_*` makes is memoized in that
//! type's `OnceLock`: set the variable in the process environment before the first GEMM call

// no_std has no runtime CPU probe, so `x86_isa_detected!` collapses to a compile-time `cfg!(...)`
// constant: these clippy lints would otherwise fire on the resulting always-true/false checks
// across every select_* ladder in this module tree
#![cfg_attr(
    not(feature = "std"),
    allow(
        clippy::assertions_on_constants,
        clippy::nonminimal_bool,
        clippy::eq_op
    )
)]

// ForcedIsa (GEMMKIT_REQUIRE_ISA) parsing plus the OnceLock-memoized select-once plumbing
// shared by every family's select_* ladder
#[macro_use]
mod isa;

// c32/c64 complex GEMM dispatch: conj-aware orientation swap, per-ISA wrappers, ComplexScalar impls
#[cfg(feature = "complex")]
mod complex;
// f32/f64 homogeneous-float dispatch: driver entries, per-ISA wrappers, GemmScalar/FusedScalar impls
mod float;
// i8 -> i32 integer GEMM dispatch, plus the fused i8 requantizing path
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
pub use float::{FusedScalar, MapScalar};
#[cfg(feature = "epilogue")]
pub(crate) use float::{execute_fused, execute_map, execute_packed_fused};
#[cfg(feature = "int8")]
pub(crate) use int::{
    IntPackedConsume, IntTask, execute_int, execute_int_packed, i8_rhs_depth_multiple, i8_rhs_tile,
    pack_rhs_full_i8,
};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub(crate) use int::{RequantTask, execute_int_requant};

#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::FusedEpi;
use crate::parallel::Parallelism;
use crate::scalar::{Float, Scalar};
use crate::tuning;
use crate::workspace::Workspace;

/// A fully described GEMM problem, `C <- alpha*A*B + beta*C`, as raw pointers and per-axis
/// `isize` element strides. The homogeneous-type dispatch boundary: A, B, and C all share
/// element type `T` (the accumulator, per [`GemmScalar`], may still differ)
#[derive(Copy, Clone)]
pub struct Task<T> {
    /// Row count of A and C
    pub m: usize,
    /// Contraction length: column count of A, row count of B
    pub k: usize,
    /// Column count of B and C
    pub n: usize,
    /// Scale applied to the A*B product
    pub alpha: T,
    /// LHS base pointer, element `(0,0)`
    pub a: *const T,
    /// LHS element strides: row, column
    pub rsa: isize,
    pub csa: isize,
    /// RHS base pointer
    pub b: *const T,
    /// RHS element strides: row, column
    pub rsb: isize,
    pub csb: isize,
    /// Scale applied to the incoming C before adding alpha*A*B
    pub beta: T,
    /// Output base pointer
    pub c: *mut T,
    /// Output element strides: row, column
    pub rsc: isize,
    pub csc: isize,
}

/// One problem for the pointer-array batched API ([`crate::gemm_batched_ptr_unchecked`]): the
/// same `C <- alpha*A*B + beta*C` as `Task`, as raw pointers and `isize` element strides, so
/// each batch entry can have its own shape and live anywhere in memory (the strided
/// [`crate::gemm_batched`] instead shares one shape and steps every operand by a fixed batch
/// stride)
#[derive(Copy, Clone)]
pub struct GemmProblem<T> {
    /// Row count of A and C
    pub m: usize,
    /// Contraction length: column count of A, row count of B
    pub k: usize,
    /// Column count of B and C
    pub n: usize,
    /// Scale applied to the A*B product
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
    /// Scale applied to the incoming C before adding alpha*A*B
    pub beta: T,
    /// Output base pointer
    pub c: *mut T,
    /// Output row stride
    pub rsc: isize,
    /// Output column stride
    pub csc: isize,
}

impl<T: Copy> GemmProblem<T> {
    /// Copy the fields into the internal [`Task`] the dispatch layer consumes; no allocation
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

/// A GEMM whose RHS is already packed into micropanels: `C <- alpha*A*(prepacked B) + beta*C`.
/// Carries the blocking geometry the buffer was packed for (`nr`, `kc`, `nc`) so the consuming
/// driver call reads panels with the exact tiling the pack step used, rather than re-deriving it
///
/// `pub` (like [`Task`]) only so it can appear in the doc-hidden [`GemmScalar`] methods; the
/// `dispatch` module itself is private, so this type is not nameable from outside the crate
pub struct PackedConsume<T> {
    /// Row count of A and C
    pub m: usize,
    /// Contraction length: column count of A, which must match the prepacked B's depth
    pub k: usize,
    /// Column count of the prepacked B, and of C
    pub n: usize,
    /// Scale applied to the A*B product
    pub alpha: T,
    /// LHS base pointer and element strides
    pub a: *const T,
    pub rsa: isize,
    pub csa: isize,
    /// Base of the prepacked RHS micropanel buffer (see [`crate::driver::pack_rhs_full`])
    pub packed: *const T,
    /// Blocking geometry `packed` was built with
    pub nr: usize,
    pub kc: usize,
    pub nc: usize,
    /// Scale applied to the incoming C before adding alpha*A*B
    pub beta: T,
    /// Output base pointer and element strides
    pub c: *mut T,
    pub rsc: isize,
    pub csc: isize,
}

/// Element types the dispatch layer knows how to run: `f32`/`f64` (homogeneous float) and,
/// under `half`, `f16`/`bf16` (mixed precision, `Acc = f32`)
///
/// The bound is [`Scalar`], not `Float<Acc = Self>`, so the accumulator type may differ from the
/// element type (the mixed-precision seam). The methods below supply what a generic bound alone
/// cannot express: the degenerate `beta`-only scale and which kernel family to pack and dispatch
/// through, keeping the driver and public API type-agnostic
pub trait GemmScalar: Scalar {
    /// Mirrors [`crate::kernel::KernelFamily::OUT_IS_ACC`] for this type: `true` for `f32`/`f64`,
    /// `false` for `f16`/`bf16`. The prepack constructor reads it so a prepacked buffer blocks
    /// with the same `kc` the consuming kernel will use
    const OUT_IS_ACC: bool;

    /// `C <- beta*C` over the strided output: the degenerate path taken when the `A*B` term
    /// vanishes (`k == 0` or `alpha == 0`). Narrow types scale in `f32` and round back once
    ///
    /// # Safety
    /// `c` valid for the `m x n` region at strides `rsc`/`csc`
    #[doc(hidden)]
    unsafe fn scale_c(beta: Self, c: *mut Self, m: usize, n: usize, rsc: isize, csc: isize);

    /// Pack a full RHS into the prepacked micropanel buffer, through this type's kernel family.
    /// The panel layout does not depend on the type, but the family *type* does (`FloatGemm` vs
    /// `MixedGemm`), so the call is routed through here rather than hard-wired in
    /// [`crate::prepack_rhs`]
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

    /// Run the ISA-dispatched plain kernel for this type. Called by the API layer
    ///
    /// # Safety
    /// `task`'s pointers must be valid and `c` must not alias `a`/`b`
    #[doc(hidden)]
    unsafe fn dispatch(task: Task<Self>, par: Parallelism, ws: &mut Workspace);

    /// Run the ISA-dispatched prepacked-RHS kernel for this type
    ///
    /// # Safety
    /// `req`'s pointers must be valid, `c` must not alias `a`/`packed`, and `packed` must have
    /// been produced by [`GemmScalar::pack_rhs_full`] for the geometry recorded in `req`
    #[doc(hidden)]
    unsafe fn dispatch_packed(req: PackedConsume<Self>, par: Parallelism, ws: &mut Workspace);

    /// This type's dispatched kernel's microtile `(mr, nr)`, i.e. `(MR_REG*LANES, NR)`. The
    /// prepack constructor calls this to size the buffer's blocking geometry through the *same*
    /// ISA choice the consuming call will make
    #[doc(hidden)]
    fn rhs_tile() -> (usize, usize);

    /// This type's dispatched kernel family's [`crate::kernel::KernelFamily::DEPTH_MULTIPLE`].
    /// The prepack constructor rounds the packed depth up to it so the buffer's layout matches
    /// the consuming kernel's. `1` (the default) covers every family except the bf16
    /// `vdpbf16ps` dot kernel, which overrides it to `2`
    #[doc(hidden)]
    fn rhs_depth_multiple() -> usize {
        1
    }

    /// Run the ISA-dispatched **fused-epilogue** kernel for this type. Every type covered by
    /// [`FusedScalar`] provides one: the real floats (`f32`/`f64`) through the `float` module,
    /// the narrow floats (`f16`/`bf16`, `Acc = f32`) through `mixed`. Required rather than
    /// defaulted, since the [`FusedScalar`] bound on the public fused API admits exactly those
    /// 4 types
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

    /// Run the ISA-dispatched **prepacked-RHS fused-epilogue** kernel for this type: the fused
    /// twin of [`GemmScalar::dispatch_packed`], threading `epi` into the prepacked driver entry
    /// (`driver::run_packed_rhs_epilogue`). `epi` already lives in the prepacked buffer's
    /// (oriented) frame: the packed path never re-orients inside the driver, so the public
    /// `gemm_packed_a_fused` entry pre-flips the bias axis before building the transposed consume
    ///
    /// # Safety
    /// As [`GemmScalar::dispatch_packed`], plus `epi`'s bias valid for the problem's `m`/`n` and
    /// disjoint from `c` (validated by the API layer)
    #[doc(hidden)]
    #[cfg(feature = "epilogue")]
    unsafe fn dispatch_packed_fused(
        req: PackedConsume<Self>,
        epi: FusedEpi<Self>,
        par: Parallelism,
        ws: &mut Workspace,
    );
}

/// Top-level entry used by the API layer: handle the degenerate cases (here, where the element
/// type is concrete) and then run the ISA-dispatched kernel
///
/// # Safety
/// `task`'s pointers must be valid for the implied regions and `c` must not alias `a`/`b`
pub(crate) unsafe fn execute<T: GemmScalar>(task: Task<T>, par: Parallelism, ws: &mut Workspace) {
    unsafe {
        if task.m == 0 || task.n == 0 {
            return;
        }
        // k == 0 or alpha == 0: the A*B term vanishes, C <- beta*C only
        if task.k == 0 || task.alpha == T::ZERO {
            T::scale_c(task.beta, task.c, task.m, task.n, task.rsc, task.csc);
            return;
        }
        T::dispatch(task, par, ws);
    }
}

/// Top-level entry for the prepacked-RHS path: handle the degenerate cases (the `A*B` term
/// vanishes, so `C <- beta*C` without ever touching the packed buffer) and then run the
/// ISA-dispatched prepacked kernel
///
/// # Safety
/// As [`execute`], plus `req.packed` valid for the recorded geometry and not aliasing `c`
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

/// Orientation normalization shared by every dispatch path (the float/mixed [`Task`], the
/// integer `IntTask`, the requantizing `RequantTask`): when `C` is row-major-ish
/// (`|csc| < |rsc|`), compute `C^T = B^T*A^T` instead so the kernel still writes columns
/// contiguously (`rsc == 1` after the swap), by exchanging `m<->n`, the `A`/`B` pointers and
/// strides, and `rsc<->csc`. Returns `true` when it swapped, so callers can flip whatever policy
/// co-varies with orientation (bias axis, conj flags). Generic over the element pointer type `L`
/// since the 3 task shapes differ only there, all with A and B sharing one element type
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
        core::mem::swap(a, b); // new A pointer = old B pointer
        core::mem::swap(rsa, csb); // new rsa = old csb, new csb = old rsa
        core::mem::swap(csa, rsb); // new csa = old rsb, new rsb = old csa
        core::mem::swap(rsc, csc);
        true
    } else {
        false
    }
}

/// [`orient_swap`] specialized to the homogeneous float / mixed [`Task`] shape
#[inline]
fn orient_transpose<T>(t: &mut Task<T>) -> bool {
    orient_swap(
        &mut t.m, &mut t.n, &mut t.a, &mut t.rsa, &mut t.csa, &mut t.b, &mut t.rsb, &mut t.csb,
        &mut t.rsc, &mut t.csc,
    )
}

/// `true` when a post-swap problem should take the horizontal `small_mn` path, from the raw
/// oriented dimensions and strides: small `m`/`n` with a long contraction, and both operands
/// already streaming contiguously along `k` (A's columns unit-stride, `csa == 1`; B's rows
/// unit-stride, `rsb == 1`), so the horizontal kernel can read them in place. The field-level
/// core, so the float/mixed [`Task`] path and the heterogeneous integer `IntTask` path (which
/// has no `Task<T>` to wrap) can call the identical calibrated gate instead of 2 copies that
/// could drift apart
#[inline]
fn small_mn_eligible_dims(m: usize, n: usize, k: usize, csa: isize, rsb: isize) -> bool {
    m <= tuning::small_mn_dim()
        && n <= tuning::small_mn_dim()
        && k > tuning::small_k_threshold()
        && csa == 1
        && rsb == 1
}

/// [`small_mn_eligible_dims`] over a [`Task`]'s fields, for the float and mixed dispatch entries
#[inline]
fn small_mn_eligible<T>(t: &Task<T>) -> bool {
    small_mn_eligible_dims(t.m, t.n, t.k, t.csa, t.rsb)
}

/// `true` when a post-swap problem clears the `small_mn` dims/`k` gates but at least one operand
/// misses the unit-stride-along-`k` predicate (`csa != 1` or `rsb != 1`): the pack tier copies
/// only the failing operand into `k`-contiguous scratch and then runs the same horizontal kernel
/// (see [`crate::special::small_mn::prepack_operands`]). An all-row-major or all-col-major shape
/// hits this with exactly one operand needing the copy. Shares the `m`/`n` bound with
/// [`small_mn_eligible_dims`] (one calibration, no drift) but has its own `k` floor
/// ([`crate::tuning::small_mn_pack_min_k`], not `small_k_threshold`), and requires a failing
/// stride where the zero-copy gate forbids one: the 2 gates are mutually exclusive, so a
/// small_mn-shaped call takes at most one of them. The field-level core, shared by the [`Task`]
/// and `IntTask` paths exactly as [`small_mn_eligible_dims`] is
#[inline]
fn small_mn_pack_eligible_dims(m: usize, n: usize, k: usize, csa: isize, rsb: isize) -> bool {
    m <= tuning::small_mn_dim()
        && n <= tuning::small_mn_dim()
        && k > tuning::small_mn_pack_min_k()
        && !(csa == 1 && rsb == 1)
}

/// [`small_mn_pack_eligible_dims`] over a [`Task`]'s fields, for the float and mixed dispatch
/// entries
#[inline]
fn small_mn_pack_eligible<T>(t: &Task<T>) -> bool {
    small_mn_pack_eligible_dims(t.m, t.n, t.k, t.csa, t.rsb)
}

/// `C <- beta*C` for a **homogeneous float** type (`f32`/`f64`): scales in place, and
/// `beta == 0` overwrites with zero rather than multiplying (so a NaN/inf already in `C` does
/// not poison the result). The float impl of [`GemmScalar::scale_c`] forwards here; narrow
/// types use `scale_c_narrow` in the `mixed` module instead
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
