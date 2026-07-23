//! Packing primitives (layer L1)
//!
//! Copies a strided A or B region into contiguous, microkernel-sized panels. Every
//! [`crate::kernel::KernelFamily`] chooses its own packed layout (interleaved for a
//! VNNI-style dot kernel, plain for a float FMA kernel), but the mechanical copy
//! that walks source strides into destination panels is the same for LHS and RHS -
//! they differ only in which stride plays "leading" and which plays "depth". That
//! one reusable copy lives here, so adding a kernel family never touches this file

use crate::scalar::Scalar;

/// Copy a strided `n_lead x depth_len` source region into micropanel-major layout:
/// `ceil(n_lead / width)` panels, each `width` elements wide in the leading
/// dimension and `depth_len` deep. Within a panel, storage is depth-major: `width`
/// contiguous leading elements per depth step. A tail panel (`n_lead` not a multiple
/// of `width`) is zero-padded past `n_lead`
///
/// LHS packing treats rows as the leading dimension (`lead = rs`, `depth = cs`,
/// `width = mr`); RHS packing treats columns as leading (`lead = cs`, `depth = rs`,
/// `width = nr`) - the caller picks which stride plays which role
///
/// # Safety
/// `src` must cover the `n_lead x depth_len` region addressed by `lead`/`depth`;
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
            if lead == 1 {
                // `lead == 1`: the `live` leading elements at each depth step are
                // contiguous in `src`, so copy them straight and zero-fill the
                // `[live, width)` tail. Produces the same bytes as the general
                // transpose branch below, just without a per-element stride walk
                for p in 0..depth_len {
                    let s = src.offset(base as isize + p as isize * depth);
                    core::ptr::copy_nonoverlapping(s, d, live);
                    for i in live..width {
                        *d.add(i) = T::ZERO;
                    }
                    d = d.add(width);
                }
            } else {
                // Strided leading dimension: walk short `tile`-long strips along the
                // contiguous `depth` axis and scatter each into the panel, instead of
                // gathering `width` strided elements per depth step (a cache miss per
                // element once `lead` is large). Same bytes as the naive transpose,
                // just cache-blocked
                let panel = d;
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
                    for i in 0..width {
                        // Index each slot directly rather than track a running
                        // pointer; `p*width + i < depth_len*width` always, so
                        // this stays in bounds
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

/// Pack into k-group-interleaved micropanel-major layout, for a dot-product kernel
/// (VNNI `vpdpbusd`, `vdpbf16ps`) that consumes `Q` depth steps in one instruction.
/// Like [`pack_panels`]: `ceil(n_lead / width)` panels of `width` leading elements.
/// Unlike it: within a panel the depth axis is grouped `Q` at a time so lane `i`'s
/// group `g` sits contiguous at panel offset `g*width*Q + i*Q + t`, ready to feed
/// straight into one dot instruction. Depth is padded up to a multiple of `Q`; both
/// the padded leading positions (past `n_lead`) and the padded depth (past
/// `depth_len`) are filled with `xform(T::ZERO)`
///
/// `xform` is applied to every element on the way in: identity for a plain dot
/// (bf16), or the `u8` bias `v -> v + 128` for VNNI's signed-to-unsigned A operand.
/// Padding with `xform(0)` keeps the pad consistent with the live data (128 for the
/// VNNI bias, 0 otherwise). LHS sets `lead = rows` / `depth = cols`, RHS swaps them,
/// same convention as [`pack_panels`]. The single place the k-group interleave index
/// math is written, shared by every dot family
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
        // Groups whose whole Q-run of depth positions is live. Only the last group can
        // straddle `depth_len` (and only when it is not a Q multiple); splitting that one
        // off lets every full group skip the per-element depth bound check
        let full_groups = depth_len / Q;
        let has_tail = !depth_len.is_multiple_of(Q);
        let pad = xform(T::ZERO);
        let mut d = dst;
        let mut base = 0usize;
        while base < n_lead {
            // Live leading positions in this panel (`base < n_lead` guarantees `live >= 1`);
            // hoisted above the group loop since it is the only guard a full group needs
            let live = core::cmp::min(width, n_lead - base);
            for g in 0..full_groups {
                let gbase = g * width * Q;
                let dp0 = (g * Q) as isize;
                if depth == 1 {
                    // Contiguous depth: lane `i`'s Q depth values are contiguous in both src
                    // and dst, so copy the run straight. The const-Q loop lets LLVM lower this
                    // to a wide copy (identity) or Q transformed stores (the +128 bias)
                    for i in 0..live {
                        let s = src.offset((base + i) as isize * lead + dp0);
                        let dd = d.add(gbase + i * Q);
                        for t in 0..Q {
                            *dd.add(t) = xform(*s.add(t));
                        }
                    }
                } else {
                    // Strided depth: gather the Q-run one depth step at a time; still free
                    // of the per-element depth bound check
                    for i in 0..live {
                        let row = src.offset((base + i) as isize * lead + dp0 * depth);
                        let dd = d.add(gbase + i * Q);
                        for t in 0..Q {
                            *dd.add(t) = xform(*row.offset(t as isize * depth));
                        }
                    }
                }
                // Leading positions past `n_lead` are pure `xform(0)` pad
                for i in live..width {
                    let dd = d.add(gbase + i * Q);
                    for t in 0..Q {
                        *dd.add(t) = pad;
                    }
                }
            }
            if has_tail {
                // The one straddling group needs the full per-element guard, so padded
                // depth slots (`dp >= depth_len`) come out byte-identical to the pad
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

// Byte-level oracle for `pack_panels`, compiled on every build (unlike the k-group
// oracle below, gated to the dot families). `check_case` always forces `lead != 1`,
// so this exercises only the strided-transpose branch bit-for-bit; the `lead == 1`
// straight-copy fast path is timed, not bit-checked, by `bench_tail_panel_pack`
#[cfg(test)]
mod panels_tests {
    use super::*;

    // Naive reference for the plain micropanel layout: `ceil(n_lead/width)` panels of
    // `depth_len` depth-major steps, `width` leading elements per step; a leading
    // position past `n_lead` reads as `T::ZERO`. `pack_panels` must reproduce this
    // buffer bit-for-bit
    fn reference<T: Scalar>(
        base: *const T,
        lead: isize,
        depth: isize,
        n_lead: usize,
        depth_len: usize,
        width: usize,
    ) -> Vec<T> {
        let panels = n_lead.div_ceil(width);
        let mut out = vec![T::ZERO; panels * width * depth_len];
        let mut d = 0usize;
        let mut b = 0usize;
        while b < n_lead {
            for p in 0..depth_len {
                for i in 0..width {
                    let lead_pos = b + i;
                    let v = if lead_pos < n_lead {
                        // SAFETY: `check_case` sizes `backing` to cover every lead_pos < n_lead
                        unsafe { *base.offset(lead_pos as isize * lead + p as isize * depth) }
                    } else {
                        T::ZERO
                    };
                    out[d + p * width + i] = v;
                }
            }
            d += width * depth_len;
            b += width;
        }
        out
    }

    // Compare raw bytes, not element equality: the packing contract is bit-identity
    fn same_bytes<T>(a: &[T], b: &[T]) -> bool {
        let (pa, la) = (a.as_ptr() as *const u8, core::mem::size_of_val(a));
        let (pb, lb) = (b.as_ptr() as *const u8, core::mem::size_of_val(b));
        // SAFETY: both slices are live for the read; `size_of_val` gives their exact byte extent
        unsafe { core::slice::from_raw_parts(pa, la) == core::slice::from_raw_parts(pb, lb) }
    }

    // `depth_stride` selects the contiguous (1) or strided (4) depth read; `lead` steps
    // past the whole depth extent so leading rows never alias, which also keeps
    // `lead != 1` and away from the straight-copy fast path
    fn check_case(n_lead: usize, depth_len: usize, width: usize, depth_stride: isize) {
        let depth = depth_stride;
        let lead = depth_len as isize * depth + 1;
        let max_off = if n_lead == 0 || depth_len == 0 {
            0
        } else {
            ((n_lead - 1) as isize * lead + (depth_len - 1) as isize * depth) as usize
        };
        // A spread-out bit pattern so a permuted or missed slot shows in the byte compare
        let backing: Vec<f32> = (0..=max_off)
            .map(|i| f32::from_bits((i as u32).wrapping_mul(2_654_435_761)))
            .collect();
        let base = backing.as_ptr();

        let expected = reference::<f32>(base, lead, depth, n_lead, depth_len, width);
        let panels = n_lead.div_ceil(width);
        let mut actual = vec![0.0f32; panels * width * depth_len];
        // SAFETY: `backing` covers every offset addressed for `lead_pos < n_lead`,
        // `p < depth_len`; `actual` is sized to the exact packed layout
        unsafe {
            pack_panels::<f32>(
                actual.as_mut_ptr(),
                base,
                lead,
                depth,
                n_lead,
                depth_len,
                width,
            );
        }
        assert!(
            same_bytes(&actual, &expected),
            "n_lead={n_lead} depth_len={depth_len} width={width} depth={depth}"
        );
    }

    // The old strided-transpose path for a `lead == 1`, `live < width` tail (before the
    // straight-copy fast path existed), kept as the A/B baseline for the microbench
    // below. Produces the same output bytes, just walks `src` at the `depth` stride
    fn transpose_tail<T: Scalar>(
        dst: *mut T,
        src: *const T,
        lead: isize,
        depth: isize,
        live: usize,
        depth_len: usize,
        width: usize,
    ) {
        let tile = crate::tuning::pack_transpose_tile();
        unsafe {
            let panel = dst;
            let mut p0 = 0;
            while p0 < depth_len {
                let pe = core::cmp::min(p0 + tile, depth_len);
                for i in 0..width {
                    if i < live {
                        let row = src.offset(i as isize * lead);
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
        }
    }

    /// Times the `lead == 1` straight-copy fast path against the old strided transpose
    /// on a column-major LHS tail (`lead = 1`, depth stride = the full row count).
    /// Reports median ns per pack
    #[test]
    #[ignore = "microbench; run with --release --ignored --nocapture"]
    fn bench_tail_panel_pack() {
        use std::time::Instant;
        let (width, live, depth_len, m) = (32usize, 8usize, 4096usize, 520isize);
        let (lead, depth) = (1isize, m); // column-major tail: rows contiguous, depth strides by m
        let max_off = (live as isize - 1) * lead + (depth_len as isize - 1) * depth;
        let backing: Vec<f32> = (0..=max_off as usize).map(|i| i as f32 * 0.5).collect();
        let base = backing.as_ptr();
        let mut dst = vec![0.0f32; width * depth_len];

        let bench = |reps: usize, mut f: Box<dyn FnMut()>| -> f64 {
            for _ in 0..20 {
                f();
            }
            let mut s: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                for _ in 0..200 {
                    f();
                }
                s.push(t.elapsed().as_secs_f64() * 1e9 / 200.0);
            }
            s.sort_by(f64::total_cmp);
            s[reps / 2]
        };

        let (p, b) = (dst.as_mut_ptr(), base);
        let t_new = bench(
            25,
            Box::new(move || unsafe {
                pack_panels::<f32>(p, b, lead, depth, live, depth_len, width);
                core::hint::black_box(p);
            }),
        );
        let (p, b) = (dst.as_mut_ptr(), base);
        let t_old = bench(
            25,
            Box::new(move || {
                transpose_tail::<f32>(p, b, lead, depth, live, depth_len, width);
                core::hint::black_box(p);
            }),
        );
        println!(
            "\ntail-panel pack (live={live}/{width}, depth={depth_len}, stride={m}): straight-copy {t_new:7.1} ns  transpose {t_old:7.1} ns  ({:.2}x)",
            t_old / t_new.max(1e-9)
        );
    }

    // Sweeps width tails (partial last panel), single- and multi-panel `n_lead`, varying
    // depth, and both contiguous (depth == 1) and strided sources
    #[test]
    fn panels_bit_identical() {
        const N_LEADS: [usize; 8] = [1, 3, 4, 5, 7, 8, 9, 17];
        const DEPTHS: [usize; 5] = [1, 2, 3, 5, 8];
        const WIDTHS: [usize; 5] = [1, 3, 4, 6, 8];
        const STRIDES: [isize; 2] = [1, 4];
        for &n_lead in &N_LEADS {
            for &depth_len in &DEPTHS {
                for &width in &WIDTHS {
                    for &stride in &STRIDES {
                        check_case(n_lead, depth_len, width, stride);
                    }
                }
            }
        }
    }
}

#[cfg(all(test, any(feature = "int8", feature = "half")))]
mod tests {
    use super::*;

    // Independent oracle for `pack_kgroup_panels`: the same `live_lead && dp < depth_len`
    // guard, the same `xform(0)` pad, the same `g*width*Q + i*Q + t` index math, written
    // separately so the routine under test must reproduce it bit-for-bit
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

    // Byte compare, not element `PartialEq`: the contract is bit-identity, and a bf16
    // NaN payload would otherwise make an exact copy fail equality
    fn same_bytes<T>(a: &[T], b: &[T]) -> bool {
        let (pa, la) = (a.as_ptr() as *const u8, core::mem::size_of_val(a));
        let (pb, lb) = (b.as_ptr() as *const u8, core::mem::size_of_val(b));
        // SAFETY: both slices are live for the read; `size_of_val` gives their exact byte extent
        unsafe { core::slice::from_raw_parts(pa, la) == core::slice::from_raw_parts(pb, lb) }
    }

    // Runs one shape through both the reference and the real routine and asserts byte
    // identity. `depth_stride` picks the contiguous path (1) or a strided source (> 1);
    // `lead` steps past the whole depth extent so rows never alias
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
        // `lead_pos < n_lead`, `dp < depth_len`; `actual` is sized to the exact packed layout
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

    // Sweeps width tails, depth tails (depth_len % Q != 0), multi- and partial-last-panel
    // `n_lead`, and both contiguous (depth == 1) and strided sources
    const N_LEADS: [usize; 7] = [1, 3, 7, 8, 9, 16, 17];
    const DEPTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 8, 11];
    const WIDTHS: [usize; 5] = [1, 3, 4, 5, 8];
    const STRIDES: [isize; 2] = [1, 5];

    // Spread-out bit pattern so the `xform(0)` pad differs from live payload bytes and a
    // permuted index shows up in the byte compare
    fn i8_val(i: usize) -> i8 {
        (i as u32).wrapping_mul(2_654_435_761) as u8 as i8
    }

    #[cfg(feature = "int8")]
    #[test]
    fn kgroup_bit_identical_i8() {
        // VNNI's LHS `+128` bias transform, and plain identity
        let plus128 = |v: i8| ((v as i32 + 128) as u8) as i8;
        let ident = |v: i8| v;
        for &n_lead in &N_LEADS {
            for &depth_len in &DEPTHS {
                for &width in &WIDTHS {
                    for &stride in &STRIDES {
                        // Q = 4: vpdpbusd's group size, both transforms
                        check_case::<i8, 4>(n_lead, depth_len, width, stride, i8_val, plus128);
                        check_case::<i8, 4>(n_lead, depth_len, width, stride, i8_val, ident);
                        // Q = 2: the other group size, same byte element
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
        // Arbitrary bit patterns, NaN payloads included: the byte compare is exact
        let val = |i: usize| bf16::from_bits((i as u32).wrapping_mul(40_503) as u16);
        let ident = |v: bf16| v;
        for &n_lead in &N_LEADS {
            for &depth_len in &DEPTHS {
                for &width in &WIDTHS {
                    for &stride in &STRIDES {
                        // The bf16 dot kernel folds Q = 2 depth steps into each vdpbf16ps
                        check_case::<bf16, 2>(n_lead, depth_len, width, stride, val, ident);
                    }
                }
            }
        }
    }
}
