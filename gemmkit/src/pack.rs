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
                // Cache-blocked transpose: walk the source along its contiguous
                // dimension (`depth` — stride 1 for a row-major LHS or column-major
                // RHS) in short strips and scatter each into the panel, rather than
                // gathering `width` strided elements per depth step (a cache miss
                // per element when `lead` is large). A pure reordered copy (packed
                // bytes identical), but far cheaper for a strided source.
                const TILE: usize = 16;
                let panel = d;
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + TILE, depth_len);
                    for i in 0..width {
                        // Address each slot directly (no running pointer past the
                        // panel end); LLVM strength-reduces the `p` loop. Every
                        // `p*width + i < depth_len*width`, so it stays in bounds.
                        if i < live {
                            let row = src.offset((base + i) as isize * lead);
                            for p in p0..pe {
                                *panel.add(p * width + i) = *row.offset(p as isize * depth);
                            }
                        } else {
                            for p in p0..pe {
                                *panel.add(p * width + i) = T::ZERO;
                            }
                        }
                    }
                    p0 = pe;
                }
                d = panel.add(depth_len * width);
            }
            base += width;
        }
    }
}
