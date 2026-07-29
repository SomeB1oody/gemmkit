//! Public core API (layer L8a)
//!
//! 2 tiers of safety sit over the same dispatch engine:
//!
//! - [`gemm`] / [`gemm_with`] check the [`MatRef`]/[`MatMut`] slice views before
//!   dispatch runs. A shape mismatch, an out-of-bounds stride, or C aliasing A or
//!   B panics before any unsafe code runs
//! - [`gemm_unchecked`] / [`gemm_unchecked_with`] take raw pointers and `isize`
//!   strides with no checks. Use these only when the caller, such as the ndarray
//!   adapter, validates its own inputs
//!
//! The semantics are `C <- alpha*A*B + beta*C`. A transposed view swaps `rs` and
//! `cs` with no copy. When `beta == 0`, gemm does not read C, so C may be
//! uninitialized
//!
//! The submodules below add batched, complex, fused-epilogue, integer,
//! map-epilogue, and prepacked-operand entries on top of the shape and alias
//! checks defined here

use crate::dispatch::{self, GemmScalar, Task};
#[cfg(feature = "epilogue")]
use crate::kernel::epilogue::{Act, BiasDim, BiasSpec, FusedEpi};
use crate::parallel::Parallelism;
#[cfg(feature = "epilogue")]
use crate::parallel::Ptr;
use crate::workspace::{self, Workspace};

// Strided- and pointer-array-batched GEMM entries
mod batched;
// Complex GEMM entries with optional conjugation
#[cfg(feature = "complex")]
mod cplx;
// Fused-epilogue (bias/activation) GEMM entries
#[cfg(feature = "epilogue")]
mod fused;
// Integer (i8 -> i32) and requantizing (i8 -> i8 or i8 -> u8) GEMM entries
#[cfg(feature = "int8")]
mod int8;
// User-defined per-element map-epilogue GEMM entries
#[cfg(feature = "epilogue")]
mod map;
// Prepacked-operand (PackedLhs/PackedRhs) entries
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
    RequantScale, Requantize, gemm_i8_requant, gemm_i8_requant_u8, gemm_i8_requant_u8_unchecked,
    gemm_i8_requant_u8_unchecked_with, gemm_i8_requant_u8_with, gemm_i8_requant_unchecked,
    gemm_i8_requant_unchecked_with, gemm_i8_requant_with,
};
#[cfg(feature = "int8")]
pub use int8::{gemm_i8, gemm_i8_unchecked, gemm_i8_unchecked_with, gemm_i8_with};
#[cfg(feature = "epilogue")]
pub use map::{gemm_map, gemm_map_unchecked, gemm_map_unchecked_with, gemm_map_with};
pub use packed::{
    PackedLhs, PackedRhs, gemm_packed_a, gemm_packed_a_unchecked, gemm_packed_a_unchecked_with,
    gemm_packed_a_with, gemm_packed_b, gemm_packed_b_unchecked, gemm_packed_b_unchecked_with,
    gemm_packed_b_with, prepack_lhs, prepack_lhs_unchecked, prepack_rhs, prepack_rhs_unchecked,
};
#[cfg(feature = "int8")]
pub use packed::{
    gemm_i8_packed_b, gemm_i8_packed_b_unchecked, gemm_i8_packed_b_unchecked_with,
    gemm_i8_packed_b_with, prepack_rhs_i8, prepack_rhs_i8_unchecked,
};
#[cfg(feature = "epilogue")]
pub use packed::{
    gemm_packed_a_fused, gemm_packed_a_fused_unchecked, gemm_packed_a_fused_unchecked_with,
    gemm_packed_a_fused_with, gemm_packed_b_fused, gemm_packed_b_fused_unchecked,
    gemm_packed_b_fused_unchecked_with, gemm_packed_b_fused_with,
};

/// An immutable strided matrix view over a slice
///
/// Element `(i, j)` lives at slice offset `i*rs + j*cs`. Negative strides are stored as
/// given. Every checked entry, such as [`gemm`]/[`gemm_with`], rejects them at call
/// time. Use [`gemm_unchecked`] for negative strides or a pointer into the middle of
/// a buffer
#[derive(Copy, Clone)]
pub struct MatRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

/// A mutable strided matrix view over a slice
///
/// Uses the same offset formula and stride rules as [`MatRef`]. Represents the
/// output `C`
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
    rs: isize,
    cs: isize,
}

impl<'a, T> MatRef<'a, T> {
    /// A view with explicit strides
    ///
    /// Construction never panics. An out-of-bounds or negative stride is only caught
    /// when the view reaches a checked entry, such as [`gemm`]/[`gemm_with`]
    pub fn new(data: &'a [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) `rows x cols` view: row stride `cols`, column stride 1
    pub fn from_row_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, cols as isize, 1)
    }
    /// A column-major (Fortran-order) `rows x cols` view: row stride 1, column stride `rows`
    pub fn from_col_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::new(data, rows, cols, 1, rows as isize)
    }
    /// Number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, T> MatMut<'a, T> {
    /// A mutable view with explicit strides
    pub fn new(data: &'a mut [T], rows: usize, cols: usize, rs: isize, cs: isize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs,
            cs,
        }
    }
    /// A row-major (C-order) mutable view: row stride `cols`, column stride 1
    pub fn from_row_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let cs = cols as isize;
        Self::new(data, rows, cols, cs, 1)
    }
    /// A column-major (Fortran-order) mutable view: row stride 1, column stride `rows`
    pub fn from_col_major(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let rs = rows as isize;
        Self::new(data, rows, cols, 1, rs)
    }
    /// Number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Highest slice offset (exclusive) reached by a `rows x cols` view at strides
/// `rs`/`cs`. Returns `None` if a stride paired with a dimension longer than 1 is
/// negative, because the safe API does not support that case. A dimension of
/// length 1 or less contributes nothing, regardless of its stride's sign. Also
/// returns `None` if the arithmetic overflows `usize`, because the view is too
/// large to address
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
        None // a negative-stride dimension: not allowed in the safe API
    } else {
        (hi as usize).checked_add(1)
    }
}

/// Panics if the view addressed by `rows`/`cols`/`rs`/`cs` does not fit in `data`
///
/// Delegates the reachable-extent math to [`extent`], then reports the mismatch
/// or invalid-stride case with a message naming the view
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

/// `true` if a strided `rows x cols` view maps 2 distinct `(i, j)` pairs to the same
/// slice offset
///
/// A view that aliases itself this way is fine to read, such as a broadcast input.
/// It is invalid as an output, because the parallel driver assumes output tiles are
/// disjoint, and 2 workers could then race on the same element. Strides are compared
/// by magnitude, since [`extent`] already rejects negative strides. A dimension of
/// length <= 1 spans nothing, so its stride cannot collide. With 2 real dimensions,
/// there is no collision when the larger stride clears the smaller dimension's whole
/// span, `big >= small_stride * small_dim`
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

/// `true` if the byte ranges of 2 `[T]`-typed views overlap
///
/// A same-type wrapper around [`overlaps_bytes`], the common primitive both this
/// function and [`validate_gemm_views`] build on. A caller with 2 different element
/// types, such as `i8` A/B against `i32` C, calls [`overlaps_bytes`] directly instead
fn overlaps<T>(pa: *const T, na: usize, pb: *const T, nb: usize) -> bool {
    let s = core::mem::size_of::<T>();
    overlaps_bytes(pa as *const u8, na, s, pb as *const u8, nb, s)
}

/// The shared validation prologue for the checked API's `(A, B, C)` trio
///
/// Checks that `A`'s columns match `B`'s rows and that `C`'s shape matches `A`'s
/// rows and `B`'s columns. Also checks that every view stays in bounds, `C`
/// addresses each element uniquely, and `C` does not overlap `A` or `B`. Generic
/// over the input type `TI` and the output type `TO`, so the same checks cover
/// every entry, even when the 2 types differ. Callers add any entry-specific
/// checks after this returns
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

    // C is written, so its strides must address each (i,j) uniquely. A self-aliasing
    // output, such as rsc == 0, lets 2 workers race in parallel mode, reachable from
    // safe code. A and B may alias themselves since a broadcast read is fine
    if self_aliases(c.rows, c.cols, c.rs, c.cs) {
        panic!(
            "gemmkit: C view aliases itself (strides {},{} map distinct elements to the same \
             memory); C must address each (i,j) uniquely",
            c.rs, c.cs
        );
    }

    // C must not alias A or B, since C is written. The borrow checker already forbids this
    // for a single call, so this check guards the raw buffers behind the views. It compares
    // byte ranges rather than element counts, so TI != TO, such as i8 A/B vs i32 C, stays exact
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

/// The shared bias validation for every checked fused entry: plain, batched,
/// packed-A, packed-B, and the complex `gemm_cplx_fused_with`
///
/// A `PerRow` bias must have length `m`, the output rows, and a `PerCol` bias must
/// have length `n`, the output cols. The bias slice must not overlap `C`'s storage.
/// `None` is a no-op. The length and overlap checks both delegate to
/// [`crate::adapter::lower_bias`], the single implementation every entry shares. The
/// activation and `LeakyRelu` slope check stay entry-local, since complex numbers have
/// no activation
#[cfg(feature = "epilogue")]
fn validate_bias<T: Copy>(bias: &Option<Bias<'_, T>>, m: usize, n: usize, c: &MatMut<'_, T>) {
    // This only validates the bias. Lowering to FusedEpi happens separately. Passing C's
    // full slice as a single unit-stride axis covers its complete byte range
    let _ = crate::adapter::lower_bias(*bias, m, n, c.data.as_ptr(), &[(c.data.len(), 1)]);
}

/// Lowers the public `Option<Bias>`/`Option<Activation>` selectors into the internal
/// [`FusedEpi`] the dispatch layer consumes
///
/// Erases the bias slice pointer to the `Send + Sync` [`Ptr`] shim, and maps a `None`
/// selector to the matching `None` variant. Every checked fused entry that takes
/// borrowed `Bias`/`Activation` values calls this. The `_unchecked` entries lower raw
/// pointers through [`to_fused_epi_raw`] instead
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

/// The raw-pointer analogue of [`to_fused_epi`]
///
/// Lowers a `(bias ptr, BiasDim, has_bias)` selector plus an optional [`Activation`]
/// into the internal [`FusedEpi`] the dispatch layer consumes. When `has_bias` is
/// `false`, this maps to [`BiasSpec::None`] and ignores the `bias` pointer. Otherwise
/// it erases the pointer to the `Send + Sync` [`Ptr`] shim under the chosen axis.
/// Every `_unchecked` fused entry uses this, including the complex
/// `gemm_cplx_fused_unchecked_with`, which always passes `act == None` because
/// complex numbers have no ordering activation
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

/// `C <- alpha*A*B + beta*C` over safe slice views, using the thread-local
/// workspace pool
///
/// # Panics
///
/// Panics if any of the following holds:
///
/// - `A.cols != B.rows`, `A.rows != C.rows`, or `B.cols != C.cols`
/// - a view's strides address outside its slice, are negative, or overflow while
///   computing the addressed extent
/// - `C`'s strides map 2 distinct elements to the same slot
/// - `C`'s storage overlaps `A`'s or `B`'s storage
/// - the strides let the logical dimensions run up to `isize::MAX` while the slice
///   stays small, and an internal pack buffer size overflows `usize`
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

/// Like [`gemm`] but reuses a caller-owned [`Workspace`]: zero heap allocation once
/// the workspace has grown to fit the 1st sufficiently large call
///
/// # Panics
///
/// Same conditions as [`gemm`]
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
    // SAFETY: validate_gemm_views checked the shapes agree, every stride stays in
    // bounds, C addresses each element uniquely, and C does not alias A or B
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

/// The raw engine: `C <- alpha*A*B + beta*C` over pointers and `isize` strides,
/// with no bounds, alias, or shape checks. Uses the thread-local workspace pool
///
/// # Safety
///
/// The caller guarantees all of the following:
///
/// - `a` and `b` are valid for reads, and `c` is valid for reads and writes, over
///   every `(i, j)` implied by the dimensions and strides
/// - `c` does not alias `a` or `b`
/// - when `beta == 0`, `c` need not be initialized
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

/// `true` if 2 byte ranges overlap, each given as a base pointer, an element count,
/// and an element size
///
/// The common primitive both [`overlaps`] and [`validate_gemm_views`] build on.
/// [`overlaps`] wraps it for the common case where both sides share an element
/// type. [`validate_gemm_views`] calls it directly, since its input and output
/// types can differ in size, such as `i8` A/B against `i32` C
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

/// Like [`gemm_unchecked`] but reuses a caller-owned [`Workspace`]
///
/// # Safety
///
/// See [`gemm_unchecked`]
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
