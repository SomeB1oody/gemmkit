//! Packing primitives (layer L1)
//!
//! Copies a strided A or B region into contiguous, microkernel-sized panels. Every
//! [`crate::kernel::KernelFamily`] chooses its own packed layout: interleaved for a
//! VNNI-style dot kernel, plain for a float FMA kernel. The mechanical copy that walks
//! source strides into destination panels is the same for the LHS and the RHS. They
//! differ only in which stride plays "leading" and which plays "depth". That one
//! reusable copy lives here, so adding a kernel family never touches this file

use crate::scalar::Scalar;

/// Copy a strided `n_lead x depth_len` source region into micropanel-major layout
///
/// The output holds `ceil(n_lead / width)` panels, each `width` elements wide in the
/// leading dimension and `depth_len` deep. Within a panel, storage is depth-major:
/// `width` contiguous leading elements per depth step. A tail panel, where `n_lead`
/// is not a multiple of `width`, is zero-padded past `n_lead`
///
/// LHS packing treats rows as the leading dimension (`lead = rs`, `depth = cs`,
/// `width = mr`). RHS packing treats columns as leading (`lead = cs`, `depth = rs`,
/// `width = nr`). The caller picks which stride plays which role
///
/// # Safety
///
/// `src` must cover the `n_lead x depth_len` region addressed by `lead` and `depth`.
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
                // When `lead == 1`, the live leading elements are contiguous in `src` at
                // each depth step. This copies them directly and zero-fills the tail past
                // `live`, producing the same bytes as the general transpose branch below
                for p in 0..depth_len {
                    let s = src.offset(base as isize + p as isize * depth);
                    core::ptr::copy_nonoverlapping(s, d, live);
                    for i in live..width {
                        *d.add(i) = T::ZERO;
                    }
                    d = d.add(width);
                }
            } else {
                // When the leading dimension is strided, this walks short `tile`-long
                // strips along the depth axis, instead of gathering elements one at a
                // time. That avoids a cache miss per element once `lead` is large. This
                // produces the same bytes as a naive transpose, but is cache-blocked
                let panel = d;
                let mut p0 = 0;
                while p0 < depth_len {
                    let pe = core::cmp::min(p0 + tile, depth_len);
                    for i in 0..width {
                        // This indexes each slot directly instead of tracking a running
                        // pointer. `p*width + i` stays below `depth_len*width`, so the
                        // access stays in bounds
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

/// Pack into a k-group-interleaved micropanel-major layout, for a dot-product kernel
/// (VNNI `vpdpbusd`, `vdpbf16ps`) that consumes `Q` depth steps in 1 instruction
///
/// Like [`pack_panels`], the output holds `ceil(n_lead / width)` panels of `width`
/// leading elements. Unlike it, the depth axis inside a panel groups `Q` steps at a
/// time. Lane `i`'s group `g` then sits contiguous at panel offset
/// `g*width*Q + i*Q + t`, ready to feed 1 dot instruction directly
///
/// The function pads depth up to a multiple of `Q`. It then fills the padded leading
/// positions past `n_lead`, and the padded depth past `depth_len`, with
/// `xform(T::ZERO)`. This is the only place the k-group interleave index math
/// appears, shared by every dot kernel family
///
/// The packer applies `xform` to every element on the way in. For a plain dot
/// (bf16), `xform` is the identity. For VNNI's signed-to-unsigned A operand, `xform`
/// adds the `u8` bias `v -> v + 128`. The pad value is `xform(0)`, which matches the
/// live data: 128 under the VNNI bias, 0 otherwise
///
/// LHS packing sets `lead = rows` and `depth = cols`. RHS packing swaps them, the
/// same convention as [`pack_panels`]
///
/// # Safety
///
/// `src` must cover the `n_lead x depth_len` region addressed by `lead` and `depth`.
/// `dst` must hold `ceil(n_lead/width) * width * depth_len.next_multiple_of(Q)`
/// elements
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
        // `full_groups` counts the groups whose whole Q-run of depth positions is live
        // Only the last group can straddle `depth_len`, and only when depth_len is not a
        // multiple of Q. Splitting that group out lets every full group skip the
        // per-element depth bound check
        let full_groups = depth_len / Q;
        let has_tail = !depth_len.is_multiple_of(Q);
        let pad = xform(T::ZERO);
        let mut d = dst;
        let mut base = 0usize;
        while base < n_lead {
            // `live` counts the live leading positions in this panel. `base < n_lead`
            // guarantees `live >= 1`. This check sits above the group loop, since it is
            // the only guard a full group needs
            let live = core::cmp::min(width, n_lead - base);
            for g in 0..full_groups {
                let gbase = g * width * Q;
                let dp0 = (g * Q) as isize;
                if depth == 1 {
                    // Contiguous depth: lane `i`'s Q depth values are contiguous in both `src`
                    // and `dst`, so this copies the run directly. The const-generic `Q` loop
                    // lets LLVM lower it to a wide copy for identity, or Q transformed stores
                    // for the +128 bias
                    for i in 0..live {
                        let s = src.offset((base + i) as isize * lead + dp0);
                        let dd = d.add(gbase + i * Q);
                        for t in 0..Q {
                            *dd.add(t) = xform(*s.add(t));
                        }
                    }
                } else {
                    // Strided depth: gather the Q-run one depth step at a time. This still
                    // skips the per-element depth bound check
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

// Byte-level oracle for `pack_panels`, compiled on every build unlike the k-group
// oracle below. `check_case` forces `lead != 1` and checks only the strided-transpose branch
#[cfg(test)]
mod panels_tests {
    use super::*;

    // Naive reference for the plain micropanel layout. A leading position past `n_lead`
    // reads as `T::ZERO`. `pack_panels` must reproduce this buffer bit-for-bit
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
                        // SAFETY: `check_case` sizes `backing` to cover every `lead_pos < n_lead`
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

    // Compare raw bytes, not element equality. The packing contract is bit-identity
    fn same_bytes<T>(a: &[T], b: &[T]) -> bool {
        let (pa, la) = (a.as_ptr() as *const u8, core::mem::size_of_val(a));
        let (pb, lb) = (b.as_ptr() as *const u8, core::mem::size_of_val(b));
        // SAFETY: both slices are live for the read. `size_of_val` gives their exact byte extent
        unsafe { core::slice::from_raw_parts(pa, la) == core::slice::from_raw_parts(pb, lb) }
    }

    // `depth_stride` selects the contiguous (1) or strided (4) depth read. `lead` steps past
    // the whole depth extent so leading rows never alias, keeping `lead != 1` and away from
    // the straight-copy fast path
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
        // SAFETY: `backing` covers every offset addressed for `lead_pos < n_lead` and
        // `p < depth_len`. `actual` is sized to the exact packed layout
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

    // Strided-transpose path for a `lead == 1`, `live < width` tail, used as the baseline
    // for the microbenchmark below. It produces the same output bytes as `pack_panels`,
    // but walks `src` at the `depth` stride
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

    /// Times the `lead == 1` straight-copy fast path against the strided-transpose path.
    /// Uses a column-major LHS tail: `lead = 1`, depth stride equal to the full row count
    ///
    /// Reports the median nanoseconds per pack for each path
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

// Independent-oracle tests for `pack_kgroup_panels`, gated behind the `int8` and
// `half` features it packs for
#[cfg(all(test, any(feature = "int8", feature = "half")))]
mod tests {
    use super::*;

    // Independent oracle for `pack_kgroup_panels`. It uses the same `live_lead && dp <
    // depth_len` guard, the same `xform(0)` pad, and the same index math. The routine
    // under test must reproduce this separately written reference bit-for-bit
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

    // Byte compare, not element `PartialEq`. The contract is bit-identity, and a bf16
    // NaN payload would otherwise make an exact copy fail equality
    fn same_bytes<T>(a: &[T], b: &[T]) -> bool {
        let (pa, la) = (a.as_ptr() as *const u8, core::mem::size_of_val(a));
        let (pb, lb) = (b.as_ptr() as *const u8, core::mem::size_of_val(b));
        // SAFETY: both slices are live for the read. `size_of_val` gives their exact byte extent
        unsafe { core::slice::from_raw_parts(pa, la) == core::slice::from_raw_parts(pb, lb) }
    }

    // Runs one shape through the reference and the real routine, and asserts byte identity
    // `depth_stride` picks the contiguous path (1) or a strided source (> 1). `lead` steps
    // past the whole depth extent so rows never alias
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
        // `lead_pos < n_lead` and `dp < depth_len`. `actual` is sized to the exact packed
        // layout
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
        // Arbitrary bit patterns, NaN payloads included. The byte compare is exact
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
