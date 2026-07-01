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

#[inline]
fn round_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
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
                std::alloc::alloc(layout)
            } else {
                let old = Layout::from_size_align(self.cap, ALIGN).expect("valid layout");
                std::alloc::realloc(self.ptr, old, new_cap)
            };
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
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
        let a_bytes_per_region = round_up(a_elems_per_region * esize, ALIGN);
        let a_total = a_bytes_per_region * a_regions.max(1);
        let b_bytes = round_up(b_elems * esize, ALIGN);
        self.ensure(a_total + b_bytes);

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
                std::alloc::dealloc(self.ptr, layout);
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

std::thread_local! {
    static POOL: core::cell::RefCell<Workspace> = const { core::cell::RefCell::new(Workspace::new()) };
    // A second per-thread pool used only by the batched path's per-worker packing. It is
    // *distinct* from `POOL` so a batch-parallel worker running inline under an outer
    // `with_thread_pool` (which is holding `POOL`) can borrow its own scratch without a re-borrow
    // panic — while still reusing the buffer across batched calls (unlike a fresh `Workspace`).
    static BATCH_POOL: core::cell::RefCell<Workspace> = const { core::cell::RefCell::new(Workspace::new()) };
}

/// Run `f` with the current thread's pooled workspace.
pub(crate) fn with_thread_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    POOL.with(|p| f(&mut p.borrow_mut()))
}

/// Run `f` with the current thread's *batched* pool (see [`BATCH_POOL`]). Used by the batched
/// path so each worker reuses persistent packing buffers across calls.
pub(crate) fn with_batch_pool<R>(f: impl FnOnce(&mut Workspace) -> R) -> R {
    BATCH_POOL.with(|p| f(&mut p.borrow_mut()))
}
