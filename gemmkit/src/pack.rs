//! Packing primitives (layer L2)
//!
//! The *layout* of a packed buffer is chosen by each [`crate::kernel::KernelFamily`]
//! (so a future integer family can use VNNI-friendly interleaving), but the
//! mechanical micropanel copy is identical for LHS and RHS - they differ only in
//! which stride is "leading" vs "depth". That one reusable routine lives here
//!
//! This file is part of the fixed packing framework: adding a kernel family does
//! not modify it

use crate::scalar::Scalar;

/// Pack into micropanel-major layout: `ceil(n_lead / width)` panels, each
/// `width` long in the leading dimension and `depth_len` deep. Within a panel,
/// elements are stored depth-major with `width` contiguous leading elements per
/// depth step; the tail panel is zero-filled past `n_lead`
///
/// For LHS packing the leading dimension is rows (`lead = rs`, `depth = cs`,
/// `width = mr`); for RHS it is columns (`lead = cs`, `depth = rs`, `width = nr`)
/// - the caller swaps the strides accordingly
///
/// # Safety
/// `src` must cover the `n_lead x depth_len` region addressed by the strides;
/// `dst` must hold `ceil(n_lead/width) * width * depth_len` elements
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
                // Contiguous leading dimension and a full panel: straight copy
                for p in 0..depth_len {
                    let s = src.offset(base as isize + p as isize * depth);
                    core::ptr::copy_nonoverlapping(s, d, width);
                    d = d.add(width);
                }
            } else {
                // Cache-blocked transpose: walk the source along its contiguous
                // dimension (`depth` - stride 1 for a row-major LHS or column-major
                // RHS) in short strips and scatter each into the panel, rather than
                // gathering `width` strided elements per depth step (a cache miss
                // per element when `lead` is large). A pure reordered copy (packed
                // bytes identical), but far cheaper for a strided source
                let panel = d;
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
                    for i in 0..width {
                        // Address each slot directly (no running pointer past the
                        // panel end); LLVM strength-reduces the `p` loop. Every
                        // `p*width + i < depth_len*width`, so it stays in bounds
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
/// (past `n_lead`) and padded depth (past `depth_len`) are filled with `xform(T::ZERO)`
///
/// `xform` is the per-element transform applied on the way in: identity for a plain dot
/// (bf16), or the `u8` bias `v -> v + 128` for VNNI's signed-to-unsigned A. The pad is
/// `xform(0)` so it is consistent with the live elements (VNNI's A pad `= 128`, the bias of
/// `0`; every other case `= 0`). LHS sets `lead = rows` / `depth = cols`, RHS swaps them,
/// exactly as [`pack_panels`]. This is the single source of truth for the interleave index
/// math, shared by every dot family
///
/// # Safety
/// `src` must cover the `n_lead x depth_len` region addressed by `lead`/`depth`; `dst`
/// must hold `ceil(n_lead/width) * width * depth_len.next_multiple_of(Q)` elements
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
        // Groups whose whole `Q`-run of depth positions is live (`dp < depth_len`). Only the
        // last group can straddle `depth_len`, and only when `depth_len` is not a multiple of
        // `Q`; splitting it off lets every interior group drop the per-element depth bound
        // check (its guard could only ever fail in that one tail group)
        let full_groups = depth_len / Q;
        let has_tail = !depth_len.is_multiple_of(Q);
        let pad = xform(T::ZERO);
        let mut d = dst;
        let mut base = 0usize;
        while base < n_lead {
            // Live leading positions in this panel; `base < n_lead` guarantees `live >= 1`
            // Hoisted out of the group/quad loops - the lane bound is the only guard the
            // full groups need
            let live = core::cmp::min(width, n_lead - base);
            for g in 0..full_groups {
                let gbase = g * width * Q;
                let dp0 = (g * Q) as isize;
                if depth == 1 {
                    // Contiguous source depth: lane `i`'s `Q` depth values are contiguous in
                    // both src and dst, so copy the `Q`-run straight. A const-`Q` loop, which
                    // LLVM lowers to a wide copy for the identity transform and to `Q`
                    // transformed stores for the `+128` bias
                    for i in 0..live {
                        let s = src.offset((base + i) as isize * lead + dp0);
                        let dd = d.add(gbase + i * Q);
                        for t in 0..Q {
                            *dd.add(t) = xform(*s.add(t));
                        }
                    }
                } else {
                    // Strided source depth: gather the `Q`-run one depth step at a time, still
                    // free of the depth bound check
                    for i in 0..live {
                        let row = src.offset((base + i) as isize * lead + dp0 * depth);
                        let dd = d.add(gbase + i * Q);
                        for t in 0..Q {
                            *dd.add(t) = xform(*row.offset(t as isize * depth));
                        }
                    }
                }
                // Leading positions past `n_lead`: the whole group is `xform(0)` pad
                for i in live..width {
                    let dd = d.add(gbase + i * Q);
                    for t in 0..Q {
                        *dd.add(t) = pad;
                    }
                }
            }
            if has_tail {
                // The single straddling group: keep the exact per-element guard so the padded
                // depth slots (`dp >= depth_len`) stay byte-for-byte identical to the pad
                let g = full_groups;
                let gbase = g * width * Q;
                for i in 0..width {
                    let live_lead = base + i < n_lead;
                    let dd = d.add(gbase + i * Q);
                    for t in 0..Q {
                        let dp = g * Q + t;
                        let v = if live_lead && dp < depth_len {
                            xform(*src.offset((base + i) as isize * lead + dp as isize * depth))
                        } else {
                            pad
                        };
                        *dd.add(t) = v;
                    }
                }
            }
            d = d.add(width * kc_pad);
            base += width;
        }
    }
}

#[cfg(all(test, any(feature = "int8", feature = "half")))]
mod tests {
    use super::*;

    // Independent byte-level oracle for [`pack_kgroup_panels`]: reimplements the current
    // semantics naively (the same `live_lead && dp < depth_len` guard, the same `xform(0)`
    // pad, the same `g*width*Q + i*Q + t` index math) with no restructuring. The routine
    // under test must reproduce this buffer bit-for-bit in every case
    #[allow(clippy::too_many_arguments)]
    fn reference<T: Scalar>(
        base: *const T,
        lead: isize,
        depth: isize,
        n_lead: usize,
        depth_len: usize,
        width: usize,
        q: usize,
        xform: impl Fn(T) -> T,
    ) -> Vec<T> {
        let kc_pad = depth_len.next_multiple_of(q);
        let ngroups = kc_pad / q;
        let pad = xform(T::ZERO);
        let panels = n_lead.div_ceil(width);
        let mut out = vec![T::ZERO; panels * width * kc_pad];
        let mut d = 0usize;
        let mut b = 0usize;
        while b < n_lead {
            for g in 0..ngroups {
                for i in 0..width {
                    let lead_pos = b + i;
                    let live_lead = lead_pos < n_lead;
                    for t in 0..q {
                        let dp = g * q + t;
                        let v = if live_lead && dp < depth_len {
                            unsafe {
                                xform(*base.offset(lead_pos as isize * lead + dp as isize * depth))
                            }
                        } else {
                            pad
                        };
                        out[d + g * width * q + i * q + t] = v;
                    }
                }
            }
            d += width * kc_pad;
            b += width;
        }
        out
    }

    // Compare two packed buffers by their raw bytes (not element equality): the contract is
    // bit-identity, and a byte compare is immune to `bf16` NaN payloads that would make a
    // float `PartialEq` reject even a bit-exact copy
    fn same_bytes<T>(a: &[T], b: &[T]) -> bool {
        let (pa, la) = (a.as_ptr() as *const u8, core::mem::size_of_val(a));
        let (pb, lb) = (b.as_ptr() as *const u8, core::mem::size_of_val(b));
        // SAFETY: both slices are live for the read and `size_of_val` is their byte extent
        unsafe { core::slice::from_raw_parts(pa, la) == core::slice::from_raw_parts(pb, lb) }
    }

    // Run one shape through the reference and the real routine and assert byte identity
    // `depth_stride` picks the contiguous fast path (1) or a strided source (> 1); the lead
    // stride steps past the whole depth extent so rows never alias
    fn check_case<T: Scalar, const Q: usize>(
        n_lead: usize,
        depth_len: usize,
        width: usize,
        depth_stride: isize,
        val: impl Fn(usize) -> T,
        xform: impl Fn(T) -> T + Copy,
    ) {
        let depth = depth_stride;
        let lead = depth_len as isize * depth + 1;
        let max_off = if n_lead == 0 || depth_len == 0 {
            0
        } else {
            ((n_lead - 1) as isize * lead + (depth_len - 1) as isize * depth) as usize
        };
        let backing: Vec<T> = (0..=max_off).map(&val).collect();
        let base = backing.as_ptr();

        let expected = reference::<T>(base, lead, depth, n_lead, depth_len, width, Q, xform);

        let kc_pad = depth_len.next_multiple_of(Q);
        let panels = n_lead.div_ceil(width);
        let mut actual = vec![T::ZERO; panels * width * kc_pad];
        // SAFETY: `backing` covers every `lead_pos*lead + dp*depth` offset addressed for
        // `lead_pos < n_lead`, `dp < depth_len`; `actual` holds the exact layout size
        unsafe {
            pack_kgroup_panels::<T, Q, _>(
                actual.as_mut_ptr(),
                base,
                lead,
                depth,
                n_lead,
                depth_len,
                width,
                xform,
            );
        }
        assert!(
            same_bytes(&actual, &expected),
            "Q={Q} n_lead={n_lead} depth_len={depth_len} width={width} depth={depth}"
        );
    }

    // Sweep width tails (live < width), depth tails (depth_len % Q != 0), multi-panel and
    // partial-last-panel `n_lead`, and both contiguous (depth == 1) and strided sources
    const N_LEADS: [usize; 7] = [1, 3, 7, 8, 9, 16, 17];
    const DEPTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 8, 11];
    const WIDTHS: [usize; 5] = [1, 3, 4, 5, 8];
    const STRIDES: [isize; 2] = [1, 5];

    // A spread-out bit pattern so the pad `xform(0)` differs from the live payload and a
    // permuted index shows up in the byte compare
    fn i8_val(i: usize) -> i8 {
        (i as u32).wrapping_mul(2_654_435_761) as u8 as i8
    }

    #[cfg(feature = "int8")]
    #[test]
    fn kgroup_bit_identical_i8() {
        // The VNNI LHS `+128` bias (identity's non-trivial sibling) and plain identity
        let plus128 = |v: i8| ((v as i32 + 128) as u8) as i8;
        let ident = |v: i8| v;
        for &n_lead in &N_LEADS {
            for &depth_len in &DEPTHS {
                for &width in &WIDTHS {
                    for &stride in &STRIDES {
                        // Q = 4 (vpdpbusd) with both transforms
                        check_case::<i8, 4>(n_lead, depth_len, width, stride, i8_val, plus128);
                        check_case::<i8, 4>(n_lead, depth_len, width, stride, i8_val, ident);
                        // Q = 2 (exercises the other group size with a byte element)
                        check_case::<i8, 2>(n_lead, depth_len, width, stride, i8_val, plus128);
                        check_case::<i8, 2>(n_lead, depth_len, width, stride, i8_val, ident);
                    }
                }
            }
        }
    }

    #[cfg(feature = "half")]
    #[test]
    fn kgroup_bit_identical_bf16() {
        use half::bf16;
        // Arbitrary bit patterns (including NaN payloads) - the byte compare is exact
        let val = |i: usize| bf16::from_bits((i as u32).wrapping_mul(40_503) as u16);
        let ident = |v: bf16| v;
        for &n_lead in &N_LEADS {
            for &depth_len in &DEPTHS {
                for &width in &WIDTHS {
                    for &stride in &STRIDES {
                        // bf16 dot kernel folds Q = 2 depth steps per vdpbf16ps
                        check_case::<bf16, 2>(n_lead, depth_len, width, stride, val, ident);
                    }
                }
            }
        }
    }
}
