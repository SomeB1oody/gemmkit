//! Packing-buffer workspace (cross-cutting)
//!
//! Packing A and B into contiguous panels needs scratch memory, and this module owns
//! that scratch. By default, a thread-local pool supplies it transparently. See
//! [`with_thread_pool`]
//!
//! The common [`crate::gemm`] entry point allocates only when a call needs more scratch
//! than the pool already holds for that thread. Otherwise it reuses the existing buffer
//!
//! A caller creates a [`Workspace`] directly, instead of using the pool, in 3 cases:
//!
//! - a hot loop over many small products, to control the allocation pattern
//! - real-time code that cannot tolerate an allocator call
//! - a need for explicit control over a buffer's lifetime
//!
//! Thread the workspace through [`crate::gemm_with`] directly. Any call whose scratch
//! need fits inside what the buffer already holds performs no heap allocation at all

use core::alloc::Layout;

/// Byte alignment applied to every packed buffer. 64 bytes covers an AVX-512 ZMM register
const ALIGN: usize = 64;

/// Bytes needed for `elems` elements of `esize` bytes each, rounded up to [`ALIGN`]
///
/// Fails closed, panicking via [`workspace_too_large`], rather than wrapping, if the
/// `elems * esize` product or its round-up to `ALIGN` overflows `usize`. See the overflow
/// note in [`Workspace::regions`] for why that matters here
fn region_bytes(elems: usize, esize: usize) -> usize {
    elems
        .checked_mul(esize)
        .and_then(|b| b.checked_next_multiple_of(ALIGN))
        .unwrap_or_else(|| workspace_too_large())
}

/// Shared abort path for every workspace-sizing overflow
///
/// An oversized GEMM must fail closed here, instead of letting a wrapped, too-small byte
/// count reach the allocator. A wrapped count would under-size the buffer that a pack
/// then writes past
#[cold]
#[inline(never)]
fn workspace_too_large() -> ! {
    panic!("gemmkit: GEMM is too large; the pack workspace size overflows usize")
}

/// A growable, 64-byte-aligned scratch buffer for packing A and B
///
/// Reusing one across many `gemm_with` calls drives allocation toward zero. A call grows
/// the buffer only when it needs more scratch than the buffer already holds. Once a call
/// has grown the buffer to some size, every later call needing that much or less
/// allocates nothing
pub struct Workspace {
    ptr: *mut u8,
    cap: usize,
}

// SAFETY: `Workspace` owns a unique heap allocation and only ever hands out pointers scoped
// to a `&mut self` borrow. Moving the whole buffer to another thread therefore carries no
// aliasing risk
unsafe impl Send for Workspace {}

impl Workspace {
    /// Create an empty workspace. The backing buffer is allocated lazily on first use
    pub const fn new() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Create a workspace pre-sized to `bytes`, so the first real packing call does not
    /// pay an allocation
    ///
    /// # Parameters
    ///
    /// - `bytes` - minimum backing buffer size, in bytes
    pub fn with_capacity(bytes: usize) -> Self {
        let mut ws = Self::new();
        if bytes > 0 {
            ws.ensure(bytes);
        }
        ws
    }

    /// Grow the buffer to at least `bytes` if it is not already that large. A no-op
    /// otherwise
    fn ensure(&mut self, bytes: usize) {
        if bytes <= self.cap {
            return;
        }
        let new_cap = bytes.next_power_of_two().max(ALIGN);
        // SAFETY: `new_cap` is non-zero and `ALIGN`-aligned, satisfying `Layout`'s
        // requirements. When reallocating, `old` reconstructs the layout the live block
        // was allocated with, since `self.cap` is only ever set to a value produced the
        // same way
        unsafe {
            let layout = Layout::from_size_align(new_cap, ALIGN).expect("valid layout");
            let p = if self.ptr.is_null() {
                alloc::alloc::alloc(layout)
            } else {
                let old = Layout::from_size_align(self.cap, ALIGN).expect("valid layout");
                alloc::alloc::realloc(self.ptr, old, new_cap)
            };
            if p.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            self.ptr = p;
            self.cap = new_cap;
        }
    }

    /// Carve `a_regions` equal-size LHS regions plus 1 shared RHS region out of the
    /// buffer, growing it first if needed. Each region is 64-byte aligned and sized in
    /// elements of `T`
    ///
    /// Callers pick `a_regions` to match their own layout. The driver passes the worker
    /// count on its per-worker LHS-pack path, or the row-block count on its shared-LHS
    /// pre-pass. Either way, it wants 1 region per thing that packs its own copy of A.
    /// Other callers want a single scratch region and pass `a_regions = 1`, sometimes
    /// with `b_elems = 0` to skip the RHS carve entirely
    ///
    /// # Parameters
    ///
    /// - `a_elems_per_region` - element count of a single LHS region
    /// - `a_regions` - number of equal-size LHS regions to carve
    /// - `b_elems` - element count of the single shared RHS region
    ///
    /// # Returns
    ///
    /// - `Regions<T>` - the LHS base pointer, the per-region LHS element stride, and the
    ///   RHS base pointer
    ///
    /// # Safety
    ///
    /// The returned pointers are valid only while this `&mut self` borrow lives, and only
    /// for the requested element counts
    pub(crate) fn regions<T>(
        &mut self,
        a_elems_per_region: usize,
        a_regions: usize,
        b_elems: usize,
    ) -> Regions<T> {
        let esize = core::mem::size_of::<T>().max(1);
        // A broadcast, zero-stride, operand can present a logical dimension up to
        // `isize::MAX`. The element count reaching this point is not bounded to
        // something the byte conversion or the regions-times-bytes sum can hold, and
        // both can wrap `usize`. A wrapped total would under-allocate the buffer that a
        // pack then writes past, which is memory-unsafe. Every multiply and add below is
        // checked and fails closed, matching the driver's own pack-sizing before this
        // point
        let a_bytes_per_region = region_bytes(a_elems_per_region, esize);
        let a_total = a_bytes_per_region
            .checked_mul(a_regions.max(1))
            .unwrap_or_else(|| workspace_too_large());
        let b_bytes = region_bytes(b_elems, esize);
        self.ensure(
            a_total
                .checked_add(b_bytes)
                .unwrap_or_else(|| workspace_too_large()),
        );

        let base = self.ptr;
        // SAFETY: `base` is 64-byte aligned, and `ensure` just grew the buffer to at least
        // `a_total + b_bytes`. So `base + a_total` lands within the live allocation
        let b_base = unsafe { base.add(a_total) };
        Regions {
            a_base: base as *mut T,
            a_stride: a_bytes_per_region / esize,
            b_base: b_base as *mut T,
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: a non-null `ptr` was allocated with `Layout::from_size_align(cap, ALIGN)`
            // in `ensure`, and `cap` is never mutated except alongside `ptr` there. This
            // reconstructs the exact layout the allocator needs to free it
            unsafe {
                let layout = Layout::from_size_align(self.cap, ALIGN).expect("valid layout");
                alloc::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

/// Packing regions carved out of a [`Workspace`] by [`Workspace::regions`]
pub(crate) struct Regions<T> {
    /// Base pointer of the 1st LHS region. Later regions start at multiples of `a_stride`
    pub a_base: *mut T,
    /// Element stride between consecutive LHS regions
    pub a_stride: usize,
    /// Base pointer of the shared RHS region
    pub b_base: *mut T,
}

#[cfg(feature = "std")]
std::thread_local! {
    static POOL: core::cell::RefCell<Workspace> = const { core::cell::RefCell::new(Workspace::new()) };
}

/// Run `f` against the calling thread's pooled [`Workspace`], falling back to a fresh
/// one if the pool is already borrowed
///
/// The fallback exists for re-entrancy. Nested rayon can run a 2nd GEMM on a thread that
/// is already inside one
///
/// For instance, a worker might work-steal another GEMM while blocked inside its own
/// `for_each`. A batch-parallel worker might also run one element inline while the outer
/// call still holds the pool
///
/// In that case `POOL.try_borrow_mut` fails, and instead of panicking this hands out a
/// one-off `Workspace` for that single call. A packing buffer holds no result state
/// between calls, so a one-off substitute is fully correct. Only the pooling's
/// allocation reuse is skipped for that call
#[cfg(feature = "std")]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    POOL.with(|p| match p.try_borrow_mut() {
        Ok(mut ws) => f(&mut ws),
        Err(_) => f(&mut Workspace::new()),
    })
}

/// Run `f` against a freshly created [`Workspace`]
///
/// Without `std` there is no thread-local storage for a pool. Since the `parallel`
/// feature itself requires `std`, there are no worker threads to re-enter either, so a
/// per-call buffer is correct here
///
/// This trades away the pool's allocation reuse for `no_std` portability. A caller who
/// wants reuse on such a build can hold its own [`Workspace`] and thread it through
/// [`crate::gemm_with`]. That call is zero-allocation after its first sufficiently large
/// use
#[cfg(not(feature = "std"))]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    f(&mut Workspace::new())
}

// Unit tests for the region-sizing overflow guards in `region_bytes`
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{ALIGN, region_bytes};

    /// Run `f`, catching a panic and returning its message, or an empty string if it did
    /// not panic
    fn panic_msg(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        match std::panic::catch_unwind(f) {
            Ok(()) => String::new(),
            Err(e) => e
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_default(),
        }
    }

    #[test]
    fn region_bytes_normal() {
        assert_eq!(region_bytes(0, 4), 0);
        assert_eq!(
            region_bytes(1000, 4),
            (1000usize * 4).next_multiple_of(ALIGN)
        );
        assert_eq!(region_bytes(7, 1), ALIGN); // rounds up to the 64-byte ALIGN floor
    }

    /// The element count fits `usize`, but the element-to-byte product overflows
    ///
    /// A broadcast operand reaches exactly this band after the driver's element-count
    /// guard, so the byte conversion must fail closed instead of wrapping. A wrap would
    /// under-allocate the pack
    #[test]
    fn region_bytes_byte_product_overflow_fails_closed() {
        // Top bit of usize, so the shift is legal on 32-bit targets (wasm32) too
        let elems = 1usize << (usize::BITS - 1); // fits usize
        let msg = panic_msg(|| {
            region_bytes(elems, 2); // *2 == 1 << usize::BITS, overflows
        });
        assert!(
            msg.contains("too large"),
            "expected too-large panic, got {msg:?}"
        );
    }

    /// The product fits but rounding up to `ALIGN` overflows
    #[test]
    fn region_bytes_roundup_overflow_fails_closed() {
        let msg = panic_msg(|| {
            region_bytes(usize::MAX, 1); // usize::MAX, next_multiple_of(64) overflows
        });
        assert!(
            msg.contains("too large"),
            "expected too-large panic, got {msg:?}"
        );
    }
}
