//! Packing-buffer workspace (cross-cutting)
//!
//! Packing A and B into contiguous panels needs scratch memory; this module owns that scratch.
//! By default a thread-local pool supplies it transparently (see [`with_thread_pool`]), so the
//! common [`crate::gemm`] entry point allocates only when a call needs more scratch than the pool
//! already holds for that thread, and reuses the existing buffer otherwise. For hot loops over
//! many small products, real-time code that cannot tolerate an allocator call, or explicit
//! control over a buffer's lifetime, a [`Workspace`] can instead be created once and threaded
//! through [`crate::gemm_with`] directly: any call whose scratch need fits inside what the
//! buffer already holds performs no heap allocation at all

use core::alloc::Layout;

/// Byte alignment applied to every packed buffer; 64 bytes covers an AVX-512 ZMM register
const ALIGN: usize = 64;

/// Bytes needed for `elems` elements of `esize` bytes each, rounded up to [`ALIGN`]
///
/// Fails closed (panics via [`workspace_too_large`]) rather than wrapping if the `elems * esize`
/// product, or its round-up to `ALIGN`, overflows `usize`; see the overflow note in
/// [`Workspace::regions`] for why that matters here
fn region_bytes(elems: usize, esize: usize) -> usize {
    elems
        .checked_mul(esize)
        .and_then(|b| b.checked_next_multiple_of(ALIGN))
        .unwrap_or_else(|| workspace_too_large())
}

/// Shared abort path for every workspace-sizing overflow: an oversized GEMM must fail closed
/// here rather than let a wrapped (too-small) byte count reach the allocator, which would
/// under-size the buffer a pack then writes past
#[cold]
#[inline(never)]
fn workspace_too_large() -> ! {
    panic!("gemmkit: GEMM is too large; the pack workspace size overflows usize")
}

/// A growable, 64-byte-aligned scratch buffer for packing A and B
///
/// Reusing one across many `gemm_with` calls amortizes allocation toward zero: a call only grows
/// the buffer when it needs more scratch than the buffer already holds, so once a call has grown
/// it to some size, every later call needing that much or less allocates nothing
pub struct Workspace {
    ptr: *mut u8,
    cap: usize,
}

// SAFETY: `Workspace` owns a unique heap allocation and only ever hands out pointers scoped to a
// `&mut self` borrow, so moving the whole buffer to another thread carries no aliasing risk
unsafe impl Send for Workspace {}

impl Workspace {
    /// Create an empty workspace; the backing buffer is allocated lazily on first use
    pub const fn new() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Create a workspace pre-sized to `bytes`, so the first real packing call does not pay an
    /// allocation
    pub fn with_capacity(bytes: usize) -> Self {
        let mut ws = Self::new();
        if bytes > 0 {
            ws.ensure(bytes);
        }
        ws
    }

    /// Grow the buffer to at least `bytes` if it is not already that large; a no-op otherwise
    fn ensure(&mut self, bytes: usize) {
        if bytes <= self.cap {
            return;
        }
        let new_cap = bytes.next_power_of_two().max(ALIGN);
        // SAFETY: `new_cap` is non-zero and `ALIGN`-aligned, satisfying `Layout`'s requirements;
        // when reallocating, `old` reconstructs the layout the live block was actually allocated
        // with, since `self.cap` is only ever set to a value produced this same way
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

    /// Carve `a_regions` equal-size LHS regions plus 1 shared RHS region out of the buffer
    /// (growing it first if needed), each 64-byte aligned and sized in elements of `T`
    ///
    /// Callers pick `a_regions` to match their own layout: the driver passes the worker count on
    /// its per-worker LHS-pack path, or the row-block count on its shared-LHS pre-pass, since
    /// either way it wants 1 region per thing that packs its own copy of A. Other callers just
    /// want a single scratch region and pass `a_regions = 1`, sometimes with `b_elems = 0` to
    /// skip the RHS carve entirely
    ///
    /// # Parameters
    /// - `a_elems_per_region` - element count of a single LHS region
    /// - `a_regions` - number of equal-size LHS regions to carve
    /// - `b_elems` - element count of the single shared RHS region
    ///
    /// # Returns
    /// - `Regions<T>` - the LHS base pointer, the per-region LHS element stride, and the RHS base
    ///   pointer
    ///
    /// # Safety
    /// The returned pointers are valid only while this `&mut self` borrow lives and only for the
    /// requested element counts
    pub(crate) fn regions<T>(
        &mut self,
        a_elems_per_region: usize,
        a_regions: usize,
        b_elems: usize,
    ) -> Regions<T> {
        let esize = core::mem::size_of::<T>().max(1);
        // A broadcast (zero-stride) operand can present a logical dimension up to `isize::MAX`,
        // so the element count reaching this point is not itself bounded to something the byte
        // conversion or the regions*bytes sum can hold - both can wrap `usize`. A wrapped total
        // would under-allocate the buffer the pack then writes past, which is memory-unsafe, so
        // every multiply/add below is checked and fails closed exactly like the driver's own
        // pack-sizing does before it ever reaches here
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
        // SAFETY: `base` is 64-byte aligned and `ensure` just grew the buffer to at least
        // `a_total + b_bytes`, so `base + a_total` lands within the live allocation
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
            // in `ensure`, and `cap` is never mutated except alongside `ptr` there, so this
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
    /// Base pointer of the 1st LHS region; later regions start at multiples of `a_stride`
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

/// Run `f` against the calling thread's pooled [`Workspace`], falling back to a fresh one if the
/// pool is already borrowed
///
/// The fallback exists for re-entrancy: nested rayon can run a 2nd GEMM on a thread that is
/// already inside one, for instance a worker that work-steals another GEMM while blocked inside
/// its own `for_each`, or a batch-parallel worker running one element inline while the outer call
/// still holds the pool. In that case `POOL.try_borrow_mut` fails, and rather than panic this
/// hands out a one-off `Workspace` for that single call. A packing buffer holds no result state
/// between calls, so a one-off substitute is fully correct; only the pooling's allocation reuse
/// is skipped for that call
#[cfg(feature = "std")]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    POOL.with(|p| match p.try_borrow_mut() {
        Ok(mut ws) => f(&mut ws),
        Err(_) => f(&mut Workspace::new()),
    })
}

/// Run `f` against a freshly created [`Workspace`]
///
/// Without `std` there is no thread-local storage for a pool, and since the `parallel` feature
/// itself requires `std`, there are no worker threads to re-enter either - a per-call buffer is
/// simply correct here. This trades away the pool's allocation reuse for `no_std` portability;
/// callers who want reuse on such a build hold their own [`Workspace`] and thread it through
/// [`crate::gemm_with`], which is zero-allocation after its first sufficiently large call
#[cfg(not(feature = "std"))]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    f(&mut Workspace::new())
}

// Unit tests for the region-sizing overflow guards in `region_bytes`
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{ALIGN, region_bytes};

    /// Run `f`, catching a panic and returning its message (or an empty string if it did not panic)
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

    /// The element count fits `usize` but the element->byte product overflows: a broadcast
    /// operand reaches exactly this band after the driver's element-count guard, so the byte
    /// conversion must fail closed instead of wrapping (a wrap would under-allocate the pack)
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
