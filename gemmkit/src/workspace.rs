//! Packing-buffer workspace (cross-cutting).
//!
//! GEMM needs scratch memory to pack A and B. By default a transparent
//! thread-local pool supplies it, so the common [`crate::gemm`] call allocates
//! at most once per thread and reuses thereafter. For hot loops of small
//! products, real-time code, or explicit lifetime control, a [`Workspace`] can
//! be created and threaded through [`crate::gemm_with`] — its second and later
//! uses perform zero heap allocation.

use core::alloc::Layout;

/// All packed buffers are aligned to this many bytes (covers AVX-512).
const ALIGN: usize = 64;

/// `elems` elements of `esize` bytes each, rounded up to [`ALIGN`] — or a
/// fail-closed panic if that product/round-up overflows `usize` (see the
/// overflow note in [`Workspace::regions`]).
fn region_bytes(elems: usize, esize: usize) -> usize {
    elems
        .checked_mul(esize)
        .and_then(|b| b.checked_next_multiple_of(ALIGN))
        .unwrap_or_else(|| workspace_too_large())
}

#[cold]
#[inline(never)]
fn workspace_too_large() -> ! {
    panic!("gemmkit: GEMM is too large; the pack workspace size overflows usize")
}

/// A growable, 64-byte-aligned scratch buffer for packing A and B.
///
/// Reusing one across many `gemm_with` calls amortizes allocation to zero after
/// the first sufficiently large call.
pub struct Workspace {
    ptr: *mut u8,
    cap: usize,
}

// SAFETY: `Workspace` owns a unique heap allocation and hands out pointers only
// for the duration of a `&mut self` borrow; it is sound to move between threads.
unsafe impl Send for Workspace {}

impl Workspace {
    /// Create an empty workspace (allocates lazily on first use).
    pub const fn new() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Create a workspace pre-sized to `bytes`, avoiding a first-call allocation
    /// spike.
    pub fn with_capacity(bytes: usize) -> Self {
        let mut ws = Self::new();
        if bytes > 0 {
            ws.ensure(bytes);
        }
        ws
    }

    fn ensure(&mut self, bytes: usize) {
        if bytes <= self.cap {
            return;
        }
        let new_cap = bytes.next_power_of_two().max(ALIGN);
        // SAFETY: layout is non-zero and validly aligned; old block (if any) was
        // allocated with the same alignment and recorded `cap`.
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

    /// Carve out `a_regions` equal LHS regions plus one shared RHS region, each
    /// 64-byte aligned, sized in elements of `T`. Returns the LHS base, the
    /// per-region LHS element stride, and the RHS base. The LHS region count is the
    /// worker count on the per-worker pack path, or the row-block count on the
    /// shared-A path — the carving is identical either way.
    ///
    /// # Safety
    /// The returned pointers are valid only while this `&mut self` borrow lives
    /// and only for the requested element counts.
    pub(crate) fn regions<T>(
        &mut self,
        a_elems_per_region: usize,
        a_regions: usize,
        b_elems: usize,
    ) -> Regions<T> {
        let esize = core::mem::size_of::<T>().max(1);
        // Fail closed on overflow: a broadcast (zero-stride) operand can present a
        // logical dimension up to `isize::MAX`, so the element→byte conversion and
        // the region/total sums can wrap `usize`. A wrapped (too-small) size would
        // under-allocate the buffer the pack then writes past — memory-unsafe — so
        // this panics with the same "too large" contract as the checked driver
        // sizing that feeds it.
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
        // SAFETY: `base` is 64-aligned and the offsets stay within `cap`.
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
            // SAFETY: `ptr`/`cap` describe the live allocation.
            unsafe {
                let layout = Layout::from_size_align(self.cap, ALIGN).expect("valid layout");
                alloc::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

/// Carved packing regions (see [`Workspace::regions`]).
pub(crate) struct Regions<T> {
    pub a_base: *mut T,
    pub a_stride: usize,
    pub b_base: *mut T,
}

#[cfg(feature = "std")]
std::thread_local! {
    static POOL: core::cell::RefCell<Workspace> = const { core::cell::RefCell::new(Workspace::new()) };
}

/// Run `f` with the current thread's pooled workspace, **re-entrancy-safe**. Nested rayon can
/// re-enter a GEMM on a thread already inside one — a worker that, while blocked in its own
/// `for_each`, work-steals another GEMM, or a batch-parallel worker running an element inline while
/// the outer call still holds the pool. The pool is then already borrowed on this thread, so this
/// hands out a fresh scratch workspace *that one time* rather than panicking. Packing buffers hold
/// no result state, so the fallback is transparent — only the buffer reuse is skipped.
#[cfg(feature = "std")]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    POOL.with(|p| match p.try_borrow_mut() {
        Ok(mut ws) => f(&mut ws),
        Err(_) => f(&mut Workspace::new()),
    })
}

/// Run `f` on a fresh workspace. Without `std` there is no thread-local pool (and, since `parallel`
/// requires `std`, no threads to re-enter): a per-call `Workspace` is correct. This trades the
/// pool's allocation-amortization for portability — callers wanting reuse hold a [`Workspace`] and
/// thread it through [`crate::gemm_with`], the zero-alloc-after-first path.
#[cfg(not(feature = "std"))]
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    f(&mut Workspace::new())
}
