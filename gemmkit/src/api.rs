//! Public core API (layer L8a).
//!
//! Two layers of safety over the same engine:
//!
//! * [`gemm`] / [`gemm_with`] — safe `&[T]` + stride views. Shape mismatches,
//!   out-of-bounds strides, and C aliasing A/B all **panic** before any unsafe
//!   work runs.
//! * [`gemm_unchecked`] — the raw pointer + `isize` stride engine for advanced
//!   callers (e.g. the ndarray adapter), which validate their own inputs.
//!
//! Semantics are exactly `C <- alpha·A·B + beta·C`. Transposition is expressed
//! through strides (a transposed view swaps `rs`/`cs`, no copy). When `beta == 0`
//! the output C is **not read**, so it may be uninitialized.

use crate::dispatch::{self, GemmScalar, Task};
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{Act, BiasDim, BiasSpec, FusedEpi};
use crate::parallel::Parallelism;
#[cfg(feature = "epilogue")]
use crate::parallel::Ptr;
use crate::workspace::{self, Workspace};

mod batched;
#[cfg(feature = "complex")]
mod cplx;
#[cfg(feature = "epilogue")]
mod fused;
#[cfg(feature = "int8")]
mod int8;
mod packed;

pub use batched::{
    BatchProblem, gemm_batched, gemm_batched_ptr_unchecked, gemm_batched_slice,
    gemm_batched_unchecked, gemm_batched_unchecked_with, gemm_batched_with,
};
#[cfg(feature = "epilogue")]
pub use batched::{
    gemm_batched_fused, gemm_batched_fused_unchecked, gemm_batched_fused_unchecked_with,
    gemm_batched_fused_with,
};
#[cfg(feature = "complex")]
pub use cplx::{gemm_cplx, gemm_cplx_unchecked, gemm_cplx_unchecked_with, gemm_cplx_with};
#[cfg(all(feature = "complex", feature = "epilogue"))]
pub use cplx::{
    gemm_cplx_fused, gemm_cplx_fused_unchecked, gemm_cplx_fused_unchecked_with,
    gemm_cplx_fused_with,
};
#[cfg(feature = "epilogue")]
pub use fused::{
    Activation, Bias, gemm_fused, gemm_fused_unchecked, gemm_fused_unchecked_with, gemm_fused_with,
};
#[cfg(all(feature = "int8", feature = "epilogue"))]
pub use int8::{
    Requantize, gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_unchecked,
    gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_u8_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with, gemm_i8_requant_with,
};
#[cfg(feature = "int8")]
pub use int8::{gemm_i8, gemm_i8_unchecked, gemm_i8_unchecked_with, gemm_i8_with};
pub use packed::{
    PackedLhs, PackedRhs, gemm_packed_a, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_a_with, gemm_packed_b, gemm_packed_b_unchecked, gemm_packed_b_unchecked_with,
    gemm_packed_b_with, prepack_lhs, prepack_lhs_unchecked, prepack_rhs, prepack_rhs_unchecked,
};

/// An immutable strided matrix view over a slice.
///
/// Element `(i, j)` lives at slice offset `i*rs + j*cs`. Strides must be
/// non-negative for the safe API (use [`gemm_unchecked`] for negative strides /
/// pointers into the middle of a buffer).
#[derive(Copy, Clone)]
pub struct MatRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

/// A mutable strided matrix view over a slice.
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

impl<'a, T> MatRef<'a, T> {
    /// A view with explicit strides. Panics in [`gemm`] if the strides address
    /// outside `data`.
    pub fn new(data: &'a [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) `rows × cols` view.
    pub fn from_row_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, cols as isize, 1)
    }
    /// A column-major (Fortran-order) `rows × cols` view.
    pub fn from_col_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, 1, rows as isize)
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, T> MatMut<'a, T> {
    /// A mutable view with explicit strides.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) mutable view.
    pub fn from_row_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let cs = cols as isize;
        Self::new(data, rows, cols, cs, 1)
    }
    /// A column-major (Fortran-order) mutable view.
    pub fn from_col_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let rs = rows as isize;
        Self::new(data, rows, cols, 1, rs)
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Highest slice offset (exclusive) reached by a strided view, or `None` if the strides are
/// negative (unsupported by the safe API) or the view is too large to address
fn extent(rows: usize, cols: usize, rs: isize, cs: isize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let mut lo: isize = 0;
    let mut hi: isize = 0;
    for &(dim, s) in &[(rows, rs), (cols, cs)] {
        let e = isize::try_from(dim).ok()?.checked_sub(1)?.checked_mul(s)?;
        if e < 0 {
            lo = lo.checked_add(e)?;
        } else {
            hi = hi.checked_add(e)?;
        }
    }
    if lo < 0 {
        None // negative strides — not allowed in the safe API
    } else {
        (hi as usize).checked_add(1)
    }
}

fn check_view<T>(data: &[T], rows: usize, cols: usize, rs: isize, cs: isize, name: &str) {
    match extent(rows, cols, rs, cs) {
        Some(need) if need <= data.len() => {}
        Some(need) => panic!(
            "gemmkit: {name} view of {rows}x{cols} (strides {rs},{cs}) needs {need} elements but slice has {}",
            data.len()
        ),
        None => panic!(
            "gemmkit: {name} view has negative strides or is too large to address; use gemm_unchecked"
        ),
    }
}

/// `true` if a strided `rows×cols` view maps two *distinct* `(i,j)` to the same
/// offset. Such a view is fine to read from (a broadcast input) but invalid as an
/// output: the parallel driver assumes output tiles are disjoint, so writing
/// through it would be a data race. Strides are taken by magnitude (negative
/// strides are already rejected by [`extent`]). A dimension of length ≤ 1 spans
/// nothing, so its stride is irrelevant; for two real dimensions, no collision is
/// possible exactly when the larger stride clears the smaller dimension's whole
/// span (`big ≥ small_stride · small_dim`).
fn self_aliases(rows: usize, cols: usize, rs: isize, cs: isize) -> bool {
    if rows == 0 || cols == 0 {
        return false; // empty view: nothing is written, so nothing can race
    }
    let r = (rows > 1).then_some((rs.unsigned_abs(), rows));
    let c = (cols > 1).then_some((cs.unsigned_abs(), cols));
    match (r, c) {
        (None, None) => false,
        (Some((s, _)), None) | (None, Some((s, _))) => s == 0,
        (Some(a), Some(b)) => {
            let (sm, big) = if a.0 <= b.0 { (a, b.0) } else { (b, a.0) };
            sm.0 == 0 || big < sm.0.saturating_mul(sm.1)
        }
    }
}

/// `true` if the byte ranges of two `[T]`-typed views overlap (same element type;
/// the heterogeneous integer API uses [`overlaps_bytes`] directly).
fn overlaps<T>(pa: *const T, na: usize, pb: *const T, nb: usize) -> bool {
    let s = core::mem::size_of::<T>();
    overlaps_bytes(pa as *const u8, na, s, pb as *const u8, nb, s)
}

/// The shared checked-API validation prologue for the full `(A, B, C)` trio: matching
/// inner dimensions, every view in bounds, `C` addressing each element uniquely, and `C`
/// not overlapping `A`/`B`. Generic over the input element type `TI` and output element
/// type `TO` (so the homogeneous, `i8 -> i32`, and requantizing `i8 -> i8` entries all
/// share it), comparing byte ranges via [`overlaps_bytes`]. Panic messages are worded
/// identically to the historic per-entry blocks (tests match the substrings). Callers add
/// any entry-specific checks (fused bias / requant scale) after this returns.
fn validate_gemm_views<TI, TO>(a: &MatRef<'_, TI>, b: &MatRef<'_, TI>, c: &MatMut<'_, TO>) {
    assert_eq!(
        a.cols, b.rows,
        "gemmkit: A.cols ({}) != B.rows ({})",
        a.cols, b.rows
    );
    assert_eq!(
        a.rows, c.rows,
        "gemmkit: A.rows ({}) != C.rows ({})",
        a.rows, c.rows
    );
    assert_eq!(
        b.cols, c.cols,
        "gemmkit: B.cols ({}) != C.cols ({})",
        b.cols, c.cols
    );

    check_view(a.data, a.rows, a.cols, a.rs, a.cs, "A");
    check_view(b.data, b.rows, b.cols, b.rs, b.cs, "B");
    check_view(c.data, c.rows, c.cols, c.rs, c.cs, "C");

    // C is written, so its strides must address each (i,j) uniquely. A
    // self-aliasing output (e.g. `rsc == 0`) would otherwise become a data race
    // in parallel mode — reachable from entirely safe code. (A/B may alias
    // themselves: they are only read, so broadcast strides are allowed there.)
    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C must not alias A or B (it is written). In safe Rust the borrow checker
    // already forbids overlapping &mut/& slices; this is a defensive guarantee.
    // Byte ranges (not element counts) so heterogeneous element sizes are exact.
    let cp = c.data.as_ptr() as *const u8;
    let cl = c.data.len();
    let si = core::mem::size_of::<TI>();
    let so = core::mem::size_of::<TO>();
    if overlaps_bytes(cp, cl, so, a.data.as_ptr() as *const u8, a.data.len(), si)
        || overlaps_bytes(cp, cl, so, b.data.as_ptr() as *const u8, b.data.len(), si)
    {
        panic!("gemmkit: C aliases A or B");
    }
}

/// The shared fused-bias validation for the checked fused entries (`gemm_fused_with`,
/// `gemm_batched_fused_with`, and the complex `gemm_cplx_fused_with`): a `PerRow` bias must have
/// length `m` (the output rows) and a `PerCol` bias length `n` (the output cols), and the bias
/// slice must not overlap `C`'s storage (compared via [`overlaps`]). `None` is a no-op. Panic
/// messages are worded identically to the historic per-entry blocks (tests match the substrings),
/// so this helper is the single source of that byte-for-byte wording. The activation /
/// `LeakyRelu`-slope check is entry-local (complex has no activation) and stays at the call sites.
#[cfg(feature = "epilogue")]
fn validate_bias<T>(bias: &Option<Bias<'_, T>>, m: usize, n: usize, c: &MatMut<'_, T>) {
    if let Some(bd) = bias {
        let (bp, bl) = match bd {
            Bias::PerRow(s) => {
                assert_eq!(
                    s.len(),
                    m,
                    "gemmkit: PerRow bias length ({}) != A.rows ({})",
                    s.len(),
                    m
                );
                (s.as_ptr(), s.len())
            }
            Bias::PerCol(s) => {
                assert_eq!(
                    s.len(),
                    n,
                    "gemmkit: PerCol bias length ({}) != B.cols ({})",
                    s.len(),
                    n
                );
                (s.as_ptr(), s.len())
            }
        };
        if overlaps(c.data.as_ptr(), c.data.len(), bp, bl) {
            panic!("gemmkit: bias slice overlaps C");
        }
    }
}

/// Lower the public `Option<Bias>` / `Option<Activation>` epilogue selectors into the internal
/// [`FusedEpi`] the dispatch layer consumes: the bias pointer is erased to the `Send + Sync`
/// [`Ptr`] shim, and a `None` selector maps to the matching `None` variant. Shared by the two
/// full fused entries (`gemm_fused_with`, `gemm_batched_fused_with`); the complex entry builds its
/// own bias-only [`FusedEpi`] (no activation) and `gemm_fused_unchecked` lowers raw pointers
/// directly, so neither routes through here.
#[cfg(feature = "epilogue")]
fn to_fused_epi<T>(bias: Option<Bias<'_, T>>, act: Option<Activation<T>>) -> FusedEpi<T> {
    let bias = match bias {
        None => BiasSpec::None,
        Some(Bias::PerRow(s)) => BiasSpec::Row(Ptr(s.as_ptr() as *mut T)),
        Some(Bias::PerCol(s)) => BiasSpec::Col(Ptr(s.as_ptr() as *mut T)),
    };
    let act = match act {
        None => Act::None,
        Some(Activation::Relu) => Act::Relu,
        Some(Activation::LeakyRelu(s)) => Act::LeakyRelu(s),
    };
    FusedEpi { bias, act }
}

/// The raw-pointer analogue of [`to_fused_epi`]: lower a `(bias ptr, [`BiasDim`], `has_bias`)`
/// selector plus an optional [`Activation`] into the internal [`FusedEpi`] the dispatch layer
/// consumes. `has_bias == false` maps to [`BiasSpec::None`] (the `bias` pointer is then ignored);
/// otherwise the pointer is erased to the `Send + Sync` [`Ptr`] shim under the chosen axis. Shared by
/// the unchecked fused entries — `gemm_fused_unchecked_with`, `gemm_batched_fused_unchecked_with`,
/// and the complex `gemm_cplx_fused_unchecked_with` (which always passes `act == None`, since an
/// ordering activation is undefined on `ℂ`) — so the raw lowering lives in one place.
#[cfg(feature = "epilogue")]
fn to_fused_epi_raw<T>(
    bias: *const T,
    bias_dim: BiasDim,
    has_bias: bool,
    act: Option<Activation<T>>,
) -> FusedEpi<T> {
    let bias = if has_bias {
        match bias_dim {
            BiasDim::PerRow => BiasSpec::Row(Ptr(bias as *mut T)),
            BiasDim::PerCol => BiasSpec::Col(Ptr(bias as *mut T)),
        }
    } else {
        BiasSpec::None
    };
    let act = match act {
        None => Act::None,
        Some(Activation::Relu) => Act::Relu,
        Some(Activation::LeakyRelu(s)) => Act::LeakyRelu(s),
    };
    FusedEpi { bias, act }
}

/// `C <- alpha·A·B + beta·C` over safe slice views, using the thread-local
/// workspace pool.
///
/// # Panics
/// If the inner dimensions disagree (`A.cols != B.rows`, `A.rows != C.rows`,
/// `B.cols != C.cols`), if any view addresses outside its slice, if `C`'s
/// storage overlaps `A`'s or `B`'s, or if the problem is so large (broadcast
/// strides allow logical dimensions up to `isize::MAX`) that an internal pack
/// buffer size overflows `usize`.
pub fn gemm<T: GemmScalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    workspace::with_thread_pool(|ws| gemm_with(ws, alpha, a, b, beta, c, par));
}

/// Like [`gemm`] but reuses a caller-owned [`Workspace`] — zero heap allocation
/// after the first sufficiently large call.
///
/// # Panics
/// Same conditions as [`gemm`].
pub fn gemm_with<T: GemmScalar>(
    ws: &mut Workspace,
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Parallelism,
) {
    validate_gemm_views(&a, &b, &c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    // SAFETY: validated above — shapes agree, every stride stays in bounds, and
    // C does not alias A/B.
    unsafe {
        dispatch::execute(
            Task {
                m,
                k,
                n,
                alpha,
                a: a.data.as_ptr(),
                rsa: a.rs,
                csa: a.cs,
                b: b.data.as_ptr(),
                rsb: b.rs,
                csb: b.cs,
                beta,
                c: c.data.as_mut_ptr(),
                rsc: c.rs,
                csc: c.cs,
            },
            par,
            ws,
        );
    }
}

/// The raw engine: `C <- alpha·A·B + beta·C` over pointers and `isize` strides,
/// with **no** bounds/alias/shape checks. Uses the thread-local workspace pool.
///
/// # Safety
/// The caller guarantees: `a`/`b` are valid for reads and `c` for read+write
/// over every `(i,j)` implied by the dimensions and strides; `c` does not alias
/// `a`/`b`; and when `beta == 0`, `c` need not be initialized.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_unchecked<T: GemmScalar>(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    unsafe {
        workspace::with_thread_pool(|ws| {
            dispatch::execute(
                Task {
                    m,
                    k,
                    n,
                    alpha,
                    a,
                    rsa,
                    csa,
                    b,
                    rsb,
                    csb,
                    beta,
                    c,
                    rsc,
                    csc,
                },
                par,
                ws,
            );
        });
    }
}

/// `true` if two byte ranges (base pointer + element count + element size) overlap.
/// The heterogeneous analogue of [`overlaps`] for the integer API, where C (`i32`)
/// and A/B (`i8`) have different element sizes.
fn overlaps_bytes(
    pa: *const u8,
    na: usize,
    sa: usize,
    pb: *const u8,
    nb: usize,
    sb: usize,
) -> bool {
    let a0 = pa as usize;
    let a1 = a0 + na * sa;
    let b0 = pb as usize;
    let b1 = b0 + nb * sb;
    a0 < b1 && b0 < a1
}

/// As [`gemm_unchecked`] but with a caller-owned [`Workspace`].
///
/// # Safety
/// See [`gemm_unchecked`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_unchecked_with<T: GemmScalar>(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    par: Parallelism,
) {
    unsafe {
        dispatch::execute(
            Task {
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                beta,
                c,
                rsc,
                csc,
            },
            par,
            ws,
        );
    }
}
