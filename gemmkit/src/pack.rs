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
        let tile = crate::tuning::pack_transpose_tile();
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
                let panel = d;
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
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

/// Pack into **k-group-interleaved** micropanel-major layout for a dot-product kernel
/// (VNNI `vpdpbusd`, `vdpbf16ps`): like [`pack_panels`], `ceil(n_lead / width)` panels of
/// `width` leading elements, but within a panel the `depth` axis is grouped `Q` at a time
/// so the `Q` consecutive depth values of one leading element are *contiguous* (lane/column
/// `i`'s group `g` at panel offset `g*width*Q + i*Q + t`), ready to feed one dot
/// instruction. The depth is padded up to a multiple of `Q`; padded leading positions
/// (past `n_lead`) and padded depth (past `depth_len`) are filled with `xform(T::ZERO)`.
///
/// `xform` is the per-element transform applied on the way in — identity for a plain dot
/// (bf16), or the `u8` bias `v -> v + 128` for VNNI's signed→unsigned A. The pad is
/// `xform(0)` so it is consistent with the live elements (VNNI's A pad `= 128`, the bias of
/// `0`; every other case `= 0`). LHS sets `lead = rows` / `depth = cols`, RHS swaps them,
/// exactly as [`pack_panels`]. This is the single source of truth for the interleave index
/// math, shared by every dot family.
///
/// # Safety
/// `src` must cover the `n_lead × depth_len` region addressed by `lead`/`depth`; `dst`
/// must hold `ceil(n_lead/width) * width * depth_len.next_multiple_of(Q)` elements.
#[cfg(any(feature = "int8", feature = "half"))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) unsafe fn pack_kgroup_panels<T: Scalar, const Q: usize, F: Fn(T) -> T>(
    dst: *mut T,
    src: *const T,
    lead: isize,
    depth: isize,
    n_lead: usize,
    depth_len: usize,
    width: usize,
    xform: F,
) {
    unsafe {
        let kc_pad = depth_len.next_multiple_of(Q);
        let ngroups = kc_pad / Q;
        let pad = xform(T::ZERO);
        let mut d = dst;
        let mut base = 0usize;
        while base < n_lead {
            for g in 0..ngroups {
                for i in 0..width {
                    let lead_pos = base + i;
                    let live_lead = lead_pos < n_lead;
                    for t in 0..Q {
                        let dp = g * Q + t;
                        let v = if live_lead && dp < depth_len {
                            xform(*src.offset(lead_pos as isize * lead + dp as isize * depth))
                        } else {
                            pad
                        };
                        *d.add(g * width * Q + i * Q + t) = v;
                    }
                }
            }
            d = d.add(width * kc_pad);
            base += width;
        }
    }
}
