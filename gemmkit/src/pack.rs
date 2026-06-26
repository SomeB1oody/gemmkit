//! Packing primitives (layer L2).
//!
//! The *layout* of a packed buffer is chosen by each [`crate::kernel::KernelFamily`]
//! (so a future integer family can use VNNI-friendly interleaving), but the
//! mechanical micropanel copy is identical for LHS and RHS — they differ only in
//! which stride is "leading" vs "depth". That one reusable routine lives here.
//!
//! This file is part of the fixed packing framework: adding a kernel family does
//! not modify it.

use crate::scalar::Scalar;

/// Pack into micropanel-major layout: `ceil(n_lead / width)` panels, each
/// `width` long in the leading dimension and `depth_len` deep. Within a panel,
/// elements are stored depth-major with `width` contiguous leading elements per
/// depth step; the tail panel is zero-filled past `n_lead`.
///
/// For LHS packing the leading dimension is rows (`lead = rs`, `depth = cs`,
/// `width = mr`); for RHS it is columns (`lead = cs`, `depth = rs`, `width = nr`)
/// — the caller swaps the strides accordingly.
///
/// # Safety
/// `src` must cover the `n_lead × depth_len` region addressed by the strides;
/// `dst` must hold `ceil(n_lead/width) * width * depth_len` elements.
#[inline]
pub(crate) unsafe fn pack_panels<T: Scalar>(
    dst: *mut T,
    src: *const T,
    lead: isize,
    depth: isize,
    n_lead: usize,
    depth_len: usize,
    width: usize,
) {
    unsafe {
        let mut d = dst;
        let mut base = 0usize;
        while base < n_lead {
            let live = core::cmp::min(width, n_lead - base);
            if lead == 1 && live == width {
                // Contiguous leading dimension and a full panel → straight copy.
                for p in 0..depth_len {
                    let s = src.offset(base as isize + p as isize * depth);
                    core::ptr::copy_nonoverlapping(s, d, width);
                    d = d.add(width);
                }
            } else {
                for p in 0..depth_len {
                    let col = src.offset(p as isize * depth);
                    for i in 0..width {
                        *d = if i < live {
                            *col.offset((base + i) as isize * lead)
                        } else {
                            T::ZERO
                        };
                        d = d.add(1);
                    }
                }
            }
            base += width;
        }
    }
}
