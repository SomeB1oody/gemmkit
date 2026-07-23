//! Small-`m,n` horizontal route: `C[i,j] = alpha*sum_k(A[i,k]*B[k,j]) + beta*C[i,j]`,
//! computed as a grid of dot products rather than through the register-tiling driver
//!
//! The driver's microkernel is built around a fixed `MR x NR` microtile: when `m` and `n`
//! are both far smaller than that tile, the driver still packs full micropanels and
//! computes a whole padded microtile, so most of the work done is on padding that never
//! reaches the real output. This route instead treats each `C[i,j]` as a single horizontal
//! dot product over `k`, computed with a SIMD `mul_add` sweep plus an ascending scalar
//! tail (the same primitive [`gemv`](crate::special::gemv) uses per row, generalized here
//! to an `m x n` grid of rows/columns)
//!
//! The dot kernel needs both operands unit-stride along `k`: A's rows (`csa == 1`) and B's
//! columns (`rsb == 1`). Of the 2 common dense layouts, only one operand ever fails this at
//! a time (all-row-major fails `rsb`, all-column-major fails `csa`), so [`prepack_operands`]
//! copies just the failing operand into a flat `k`-contiguous scratch buffer and the same
//! kernel then runs over it with unit strides; when both operands already qualify, the
//! pre-pack is a no-op and every pointer passes through unchanged. The `m*k` (or `n*k`)
//! copy cost is small next to the `m*n*k` dot work it unlocks, so this still beats falling
//! back to the driver's padded microtile
//!
//! Output is tiled `MT x NT` at a time: a full tile keeps `MT*NT` accumulators live across
//! the whole `k`-sweep (each A-row and B-column loaded once per depth step, shared across
//! the tile's cells), with an edge tile (where `m`/`n` is not a multiple of the register
//! tile) falling back to one dot per cell
//!
//! Every output cell is one fixed-order reduction (SIMD `reduce_sum` in the token's lane
//! order, plus an ascending scalar tail) computed entirely by a single worker: splitting
//! output tiles across workers adds no cross-thread reduction, so results are bit-identical
//! to the serial run at any worker count. The tile grid itself does not depend on the
//! worker count

use crate::kernel::FloatGemm;
#[cfg(feature = "half")]
use crate::kernel::MixedGemm;
use crate::kernel::epilogue::Epilogue;
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;
#[cfg(any(feature = "half", feature = "int8"))]
use crate::simd::KernelSimd;
use crate::simd::SimdOps;
use crate::workspace::Workspace;

/// Output register-tile shape: `MT` rows by `NT` columns of live accumulators per pass over
/// `k`. `4x4` was chosen so the tile's accumulators plus its 1 live A-row-vector-per-row and
/// 1 B-column-vector fit comfortably inside the ISA's vector register file: on Zen5
/// (AVX-512F, 32 `zmm`) that is 16 + 4 + 1 = 21 registers, well under the 32 available.
/// Measured on M4 (NEON, also 32 vector registers): larger tiles (`4x6`/`6x4`/`5x5`, 29-31
/// live vectors) regress sharply, from ~120 GFLOP/s down to 44-62, because the larger
/// accumulator array no longer fits and LLVM starts spilling it. `4x4` stays in the same
/// low-register-pressure regime the production NEON microkernel targets
const MT: usize = 4;
const NT: usize = 4;

/// Common output-tile sweep shared by every small-`m,n` entry point ([`run_epi`],
/// [`run_mixed_epi`], [`run_int`]): builds the `MT x NT` tile grid, caps the worker count
/// with the bandwidth model from the caller-supplied byte count, and either runs `body`
/// serially over the whole grid or hands out flat-tile ranges to workers through a shared
/// [`JobCursor`]. `body(q_start, q_end)` computes tiles `[q_start, q_end)` using whichever
/// per-type tile kernels the caller closed over. Because every tile is a self-contained
/// reduction, this partition never changes the result: bit-identical for any worker count
fn tile_sweep(
    m: usize,
    n: usize,
    bytes: usize,
    par: Parallelism,
    body: impl Fn(usize, usize) + Copy + Send + Sync,
) {
    let n_row_tiles = m.div_ceil(MT);
    let n_col_tiles = n.div_ceil(NT);
    let n_tiles = n_row_tiles * n_col_tiles;
    let n_threads = par.resolve_bandwidth(bytes, n_tiles);

    if n_threads <= 1 {
        body(0, n_tiles);
        return;
    }

    // Each worker claims disjoint flat-tile ranges; every tile is a complete k-reduction
    // owned by one worker, so no barrier or cross-worker combine step is needed
    let cur = JobCursor::new(n_tiles, parallel::job_grain(n_tiles, n_threads));
    parallel::for_each_worker(n_threads, |_tid| {
        while let Some((s, e)) = cur.next_chunk() {
            body(s, e);
        }
    });
}

/// Line stride, in elements, for a packed `k`-contiguous scratch buffer: `k`
/// rounded up to an odd number of 64-byte cache lines
///
/// A stride of exactly `k` would make every packed line start at the same offset modulo
/// the cache-line size whenever `k*sizeof(T)` is a multiple of the L1 set span, so with up
/// to `small_mn_dim` (16) lines all landing on the same handful of L1 sets, the tile
/// kernel's repeated re-reads of those lines thrash the cache (measured: a `16x16`
/// all-column-major GEMM this route packs drops roughly 3x below the driver once `k >=
/// 1024`). Rounding the line count up to an odd number makes it coprime with the (typically
/// power-of-two) L1 set count, so consecutive lines land on distinct sets instead. Padding
/// past `k` is allocated but never read
#[inline]
fn packed_line_stride<T>(k: usize) -> usize {
    // Elements per 64-byte cache line; `.max(1)` keeps this a valid divisor even for a
    // (currently nonexistent) element type wider than one line
    let lane = (64 / core::mem::size_of::<T>().max(1)).max(1);
    let lines = k.div_ceil(lane).max(1);
    let odd_lines = if lines.is_multiple_of(2) {
        lines + 1
    } else {
        lines
    };
    odd_lines * lane
}

/// Copy one strided operand into a flat, `k`-contiguous scratch layout: for each of the
/// `lead` lines (A rows or B columns), `dst[l*dst_stride + t] = src[l*lead_stride +
/// t*depth_stride]` for `t in 0..k`. `dst_stride` is [`packed_line_stride`] so consecutive
/// lines never alias the same L1 set
///
/// The operand being packed here is exactly the one whose `k` axis is strided in memory
/// (`depth_stride != 1`), so its `lead` axis (rows for A, columns for B) is the one that is
/// contiguous in `src`. The loop walks depth `t` in [`crate::tuning::pack_transpose_tile`]
/// strips and `lead` inside each strip, so both the `src` reads (unit-stride along `lead`)
/// and the scattered `dst` writes to `lead` distinct lines stay within a small working set
/// instead of a full stride-`k` gather. This is a pure reordering copy: the values landing
/// in `dst` are the same values in the same per-line order, so the dot afterward reads
/// identical numbers in identical order
///
/// # Safety
/// `src` valid for the `lead x k` region at `lead`/`depth`; `dst` valid for `lead*dst_stride` writes
#[inline]
unsafe fn pack_k_contiguous<T: Copy>(
    dst: *mut T,
    src: *const T,
    lead: usize,
    k: usize,
    dst_stride: usize,
    lead_stride: isize,
    depth_stride: isize,
) {
    unsafe {
        let tile = crate::tuning::pack_transpose_tile();
        let mut t0 = 0;
        while t0 < k {
            let te = core::cmp::min(t0 + tile, k);
            for t in t0..te {
                // Depth line `t`; the inner sweep over `lead` then reads it unit-stride
                let col = src.offset(t as isize * depth_stride);
                for l in 0..lead {
                    *dst.add(l * dst_stride + t) = *col.offset(l as isize * lead_stride);
                }
            }
            t0 = te;
        }
    }
}

/// Pre-pack whichever of `A`/`B` fails the unit-stride-along-`k` predicate into `ws`,
/// returning the (possibly repointed) `(a, rsa, csa, b, rsb, csb)` for the caller to feed
/// the horizontal kernel, which then always sees `csa == 1 && rsb == 1`. One generic-over-`T`
/// helper shared by the float, mixed, and integer routes so the pack logic cannot drift
/// between them. When both operands already qualify, this returns the inputs untouched
/// without touching `ws` at all, so an already-eligible call is identical to the
/// pre-pack-free path
///
/// A packs to `m` rows, each `k` contiguous elements (new `rsa` = [`packed_line_stride`],
/// `csa = 1`); B packs to `n` columns, each `k` contiguous elements (`rsb = 1`, new `csb` =
/// [`packed_line_stride`]). `Workspace::regions` applies the same fail-closed
/// element-to-byte overflow guard the driver's own pack sizing uses
///
/// # Safety
/// `a`/`b` valid for the `m x k` / `k x n` regions at their strides; the returned pointers
/// are valid only while `ws`'s `&mut` borrow lives (the caller must consume them before it ends)
#[allow(clippy::too_many_arguments)]
unsafe fn prepack_operands<T: Copy>(
    ws: &mut Workspace,
    m: usize,
    k: usize,
    n: usize,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
) -> (*const T, isize, isize, *const T, isize, isize) {
    let pack_a = csa != 1;
    let pack_b = rsb != 1;
    // Both operands already stream unit-stride along k: nothing to do
    if !pack_a && !pack_b {
        return (a, rsa, csa, b, rsb, csb);
    }
    let stride = packed_line_stride::<T>(k);
    unsafe {
        // Carve out only the region(s) actually needed
        let a_elems = if pack_a { m.saturating_mul(stride) } else { 0 };
        let b_elems = if pack_b { n.saturating_mul(stride) } else { 0 };
        let r = ws.regions::<T>(a_elems, 1, b_elems);
        let (mut a, mut rsa, mut csa) = (a, rsa, csa);
        let (mut b, mut rsb, mut csb) = (b, rsb, csb);
        if pack_a {
            // A[i, :] -> dst[i*stride + t]: rows are the lead axis, k the (strided) depth
            pack_k_contiguous::<T>(r.a_base, a, m, k, stride, rsa, csa);
            a = r.a_base;
            rsa = stride as isize;
            csa = 1;
        }
        if pack_b {
            // B[:, j] -> dst[j*stride + t]: cols are the lead axis, k the (strided) depth
            pack_k_contiguous::<T>(r.b_base, b, n, k, stride, csb, rsb);
            b = r.b_base;
            rsb = 1;
            csb = stride as isize;
        }
        (a, rsa, csa, b, rsb, csb)
    }
}

/// Small-`m,n` horizontal GEMM with a fused [`Epilogue`] `E` applied at each cell's single
/// store. Each cell is one complete `k`-reduction, so the epilogue fires exactly once per
/// element; a non-identity `E` changes only that store, leaving the tiling and partition
/// identical to the `E = Identity` plain path the float dispatch ladder drives this generic
/// through directly, const-folding every hook away there (`row`/`col` passed to `epi` are
/// oriented-frame coordinates - dispatch already flipped the bias axis on an orientation swap
/// before calling in)
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must not alias
/// `a`/`b`; A rows must be unit-stride (`csa == 1`) and B columns unit-stride (`rsb == 1`) so both
/// stream contiguously along `k`; the CPU must support `S`'s features; and `epi`'s interior
/// pointers must be valid for the (oriented) problem's `m`/`n`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_epi<T, S, E>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
    ws: &mut Workspace,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
    E: Epilogue<FloatGemm<T>>,
{
    // Epilogue is Copy; move a value copy into the worker closure below
    let epi = *epi;
    unsafe {
        // No-op when both operands already stream unit-stride along k
        let (a, rsa, csa, b, rsb, csb) =
            prepack_operands::<T>(ws, m, k, n, a, rsa, csa, b, rsb, csb);
        debug_assert!(
            csa == 1 && rsb == 1,
            "small_mn kernel requires A rows / B cols unit-stride along k"
        );
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count: minimum traffic is A read once, B read once, C
        // written once
        let sizeof = core::mem::size_of::<T>();
        let bytes = m
            .saturating_mul(k)
            .saturating_add(k.saturating_mul(n))
            .saturating_add(m.saturating_mul(n))
            .saturating_mul(sizeof);

        let a = Ptr(a as *mut T);
        let b = Ptr(b as *mut T);
        let c = Ptr(c);

        // Column-tile-outer flat order: a worker's consecutive tiles share a C column
        // block, giving contiguous stores for a column-major C
        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c, epi) = (a, b, c, epi);
            let a = a.0 as *const T;
            let b = b.0 as *const T;
            let c = c.0;
            simd.vectorize(|| {
                for q in q_start..q_end {
                    let it = q % n_row_tiles;
                    let jt = q / n_row_tiles;
                    let i0 = it * MT;
                    let j0 = jt * NT;
                    let mi = core::cmp::min(MT, m - i0);
                    let nj = core::cmp::min(NT, n - j0);
                    if mi == MT && nj == NT {
                        full_tile::<T, S, E, MT, NT>(
                            simd, k, i0, j0, alpha, a, rsa, b, csb, beta, c, rsc, csc, &epi,
                        );
                    } else {
                        // Edge tile (m or n not a multiple of MT/NT): one dot per cell
                        for cc in 0..nj {
                            for ir in 0..mi {
                                cell_dot::<T, S, E>(
                                    simd,
                                    k,
                                    i0 + ir,
                                    j0 + cc,
                                    alpha,
                                    a,
                                    rsa,
                                    b,
                                    csb,
                                    beta,
                                    c,
                                    rsc,
                                    csc,
                                    &epi,
                                );
                            }
                        }
                    }
                }
            });
        };

        tile_sweep(m, n, bytes, par, body);
    }
}

/// Compute a full `MT x NT` output tile at origin `(i0, j0)`. Holds `MT*NT` accumulators
/// live across the entire `k`-sweep, loading each of the tile's `MT` A-rows and `NT`
/// B-columns once per depth step (both contiguous: `csa == 1`, `rsb == 1`) and feeding them
/// into every accumulator that needs them, then finishing each cell with `reduce_sum` plus
/// an ascending scalar tail and the `beta` combine. The fused [`Epilogue`] `E` is applied at
/// each cell's single store; `E::IS_IDENTITY` const-folds that branch away entirely
///
/// # Safety
/// `a`/`b`/`c` valid for the tile's reads/writes; `csa == 1` and `rsb == 1` (unit-stride
/// along `k`); the tile is fully in-bounds (`i0 + MT <= m`, `j0 + NT <= n`). Run inside `S::vectorize`
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn full_tile<T, S, E, const MT: usize, const NT: usize>(
    simd: S,
    k: usize,
    i0: usize,
    j0: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    b: *const T,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
    E: Epilogue<FloatGemm<T>>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let rows: [*const T; MT] = core::array::from_fn(|r| a.offset((i0 + r) as isize * rsa));
        let cols: [*const T; NT] = core::array::from_fn(|cc| b.offset((j0 + cc) as isize * csb));

        let mut acc = [[simd.zero(); MT]; NT];
        let mut kk = 0;
        while kk + lanes <= k {
            let av: [S::Reg; MT] = core::array::from_fn(|r| simd.loadu(rows[r].add(kk)));
            for cc in 0..NT {
                let bv = simd.loadu(cols[cc].add(kk));
                for r in 0..MT {
                    acc[cc][r] = simd.mul_add(av[r], bv, acc[cc][r]);
                }
            }
            kk += lanes;
        }
        for cc in 0..NT {
            for r in 0..MT {
                let mut dot = simd.reduce_sum(acc[cc][r]);
                let mut t = kk;
                while t < k {
                    dot = (*rows[r].add(t)).mul_add(*cols[cc].add(t), dot);
                    t += 1;
                }
                let cp = c.offset((i0 + r) as isize * rsc + (j0 + cc) as isize * csc);
                let ov = if beta == T::ZERO {
                    T::ZERO
                } else if beta == T::ONE {
                    *cp
                } else {
                    beta * *cp
                };
                let out = alpha.mul_add(dot, ov);
                // Applied once, at the single store for this cell
                *cp = if E::IS_IDENTITY {
                    out
                } else {
                    epi.apply(out, i0 + r, j0 + cc)
                };
            }
        }
    }
}

/// Compute one output cell `C[i,j] = alpha*sum_k(A[i,k]*B[k,j]) + beta*C[i,j]` as a
/// single-accumulator SIMD dot over the contiguous A-row / B-column, plus an ascending
/// scalar `k`-tail. Used for the edge tile, where `m`/`n` is not a multiple of `MT`/`NT`.
/// The fused [`Epilogue`] `E` is applied once, at the store (`E::IS_IDENTITY` const-folds
/// that branch away entirely)
///
/// # Safety
/// `a`/`b`/`c` valid for the element's reads/writes; `csa == 1` and `rsb == 1`. Run inside
/// `S::vectorize`
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn cell_dot<T, S, E>(
    simd: S,
    k: usize,
    i: usize,
    j: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    b: *const T,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
    E: Epilogue<FloatGemm<T>>,
{
    unsafe {
        let row = a.offset(i as isize * rsa); // A[i, :], contiguous since csa == 1
        let col = b.offset(j as isize * csb); // B[:, j], contiguous since rsb == 1
        let dot = super::dot_contiguous::<T, S>(simd, k, row, col);
        let cp = c.offset(i as isize * rsc + j as isize * csc);
        let ov = if beta == T::ZERO {
            T::ZERO
        } else if beta == T::ONE {
            *cp
        } else {
            beta * *cp
        };
        let out = alpha.mul_add(dot, ov);
        *cp = if E::IS_IDENTITY {
            out
        } else {
            epi.apply(out, i, j)
        };
    }
}

/// Mixed-precision small-`m,n` horizontal GEMM with a fused [`Epilogue`] `E` (over the
/// [`MixedGemm`] family) applied to each cell's `f32` accumulated value, right before that
/// value is narrowed to `N` at its single store. The mixed dispatch ladder drives this generic
/// directly; `E = Identity` on the plain path const-folds every hook away to the raw narrowing
/// store. Applying `E` before the narrowing (rather than narrowing first and mapping
/// after) matches the driver's mixed-precision epilogue semantics and is more precise than
/// narrowing first, since it avoids rounding to `N` before the epilogue's own math runs
/// (`row`/`col` are oriented-frame coordinates - dispatch already flipped the bias axis on
/// an orientation swap before calling in)
///
/// # Safety
/// As [`run_epi`], with `N` operands and an `f32` accumulator, plus `epi`'s interior pointers
/// must be valid for the (oriented) `m`/`n`
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_mixed_epi<N, S, E>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
    ws: &mut Workspace,
    alpha: f32,
    a: *const N,
    rsa: isize,
    csa: isize,
    b: *const N,
    rsb: isize,
    csb: isize,
    beta: f32,
    c: *mut N,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
    E: Epilogue<MixedGemm<N>>,
{
    // Epilogue is Copy; move a value copy into the worker closure below
    let epi = *epi;
    unsafe {
        // No-op when both operands already stream unit-stride along k; the narrow (N-byte)
        // operand is packed as-is, still widened on load same as before
        let (a, rsa, csa, b, rsb, csb) =
            prepack_operands::<N>(ws, m, k, n, a, rsa, csa, b, rsb, csb);
        debug_assert!(
            csa == 1 && rsb == 1,
            "small_mn kernel requires A rows / B cols unit-stride along k"
        );
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count, counted in narrow-type bytes
        let sizeof = core::mem::size_of::<N>();
        let bytes = m
            .saturating_mul(k)
            .saturating_add(k.saturating_mul(n))
            .saturating_add(m.saturating_mul(n))
            .saturating_mul(sizeof);

        let a = Ptr(a as *mut N);
        let b = Ptr(b as *mut N);
        let c = Ptr(c);

        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c, epi) = (a, b, c, epi);
            let a = a.0 as *const N;
            let b = b.0 as *const N;
            let c = c.0;
            simd.vectorize(|| {
                for q in q_start..q_end {
                    let it = q % n_row_tiles;
                    let jt = q / n_row_tiles;
                    let i0 = it * MT;
                    let j0 = jt * NT;
                    let mi = core::cmp::min(MT, m - i0);
                    let nj = core::cmp::min(NT, n - j0);
                    if mi == MT && nj == NT {
                        full_tile_mixed::<N, S, E, MT, NT>(
                            simd, k, i0, j0, alpha, a, rsa, b, csb, beta, c, rsc, csc, &epi,
                        );
                    } else {
                        for cc in 0..nj {
                            for ir in 0..mi {
                                cell_dot_mixed::<N, S, E>(
                                    simd,
                                    k,
                                    i0 + ir,
                                    j0 + cc,
                                    alpha,
                                    a,
                                    rsa,
                                    b,
                                    csb,
                                    beta,
                                    c,
                                    rsc,
                                    csc,
                                    &epi,
                                );
                            }
                        }
                    }
                }
            });
        };

        tile_sweep(m, n, bytes, par, body);
    }
}

/// Mixed-precision sibling of [`full_tile`] (see [`run_mixed_epi`]): accumulates in `f32`
/// via widened `N -> f32` loads, using plain (non-fused) `a*b + c` for the scalar tail and
/// combine so this route's rounding matches the reference scalar path exactly. The fused
/// [`Epilogue`] `E` runs on the `f32` cell value at its single narrowing store
/// (`E::IS_IDENTITY` const-folds that branch away to a raw narrowing store)
///
/// # Safety
/// As [`full_tile`], with `N`/`f32` operands
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn full_tile_mixed<N, S, E, const MT: usize, const NT: usize>(
    simd: S,
    k: usize,
    i0: usize,
    j0: usize,
    alpha: f32,
    a: *const N,
    rsa: isize,
    b: *const N,
    csb: isize,
    beta: f32,
    c: *mut N,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
    E: Epilogue<MixedGemm<N>>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let rows: [*const N; MT] = core::array::from_fn(|r| a.offset((i0 + r) as isize * rsa));
        let cols: [*const N; NT] = core::array::from_fn(|cc| b.offset((j0 + cc) as isize * csb));

        let mut acc: [[<S as SimdOps<f32>>::Reg; MT]; NT] = [[simd.zero(); MT]; NT];
        let mut kk = 0;
        while kk + lanes <= k {
            let av: [<S as SimdOps<f32>>::Reg; MT] =
                core::array::from_fn(|r| simd.load_lhs(rows[r].add(kk)));
            for cc in 0..NT {
                let bv = simd.load_lhs(cols[cc].add(kk));
                for r in 0..MT {
                    acc[cc][r] = simd.mul_add(av[r], bv, acc[cc][r]);
                }
            }
            kk += lanes;
        }
        for cc in 0..NT {
            for r in 0..MT {
                let mut dot = simd.reduce_sum(acc[cc][r]);
                let mut t = kk;
                while t < k {
                    dot += (*rows[r].add(t)).widen() * (*cols[cc].add(t)).widen();
                    t += 1;
                }
                let cp = c.offset((i0 + r) as isize * rsc + (j0 + cc) as isize * csc);
                let ov = if beta == 0.0 {
                    0.0
                } else if beta == 1.0 {
                    (*cp).widen()
                } else {
                    beta * (*cp).widen()
                };
                let out = alpha * dot + ov;
                // E runs on the f32 value, before the single narrowing to N
                *cp = if E::IS_IDENTITY {
                    N::narrow(out)
                } else {
                    epi.apply(out, i0 + r, j0 + cc)
                };
            }
        }
    }
}

/// Mixed-precision sibling of [`cell_dot`] (edge-tile path; see [`run_mixed_epi`]): an `f32`
/// widen-load dot, with the fused [`Epilogue`] `E` applied to the accumulated `f32` value
/// before it is narrowed to `N` once (`E::IS_IDENTITY` const-folds to a raw narrowing store)
///
/// # Safety
/// As [`cell_dot`], with `N`/`f32` operands
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn cell_dot_mixed<N, S, E>(
    simd: S,
    k: usize,
    i: usize,
    j: usize,
    alpha: f32,
    a: *const N,
    rsa: isize,
    b: *const N,
    csb: isize,
    beta: f32,
    c: *mut N,
    rsc: isize,
    csc: isize,
    epi: &E,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
    E: Epilogue<MixedGemm<N>>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let row = a.offset(i as isize * rsa);
        let col = b.offset(j as isize * csb);
        let mut acc = simd.zero();
        let mut kk = 0;
        while kk + lanes <= k {
            acc = simd.mul_add(simd.load_lhs(row.add(kk)), simd.load_lhs(col.add(kk)), acc);
            kk += lanes;
        }
        let mut dot = simd.reduce_sum(acc);
        while kk < k {
            dot += (*row.add(kk)).widen() * (*col.add(kk)).widen();
            kk += 1;
        }
        let cp = c.offset(i as isize * rsc + j as isize * csc);
        let ov = if beta == 0.0 {
            0.0
        } else if beta == 1.0 {
            (*cp).widen()
        } else {
            beta * (*cp).widen()
        };
        let out = alpha * dot + ov;
        *cp = if E::IS_IDENTITY {
            N::narrow(out)
        } else {
            epi.apply(out, i, j)
        };
    }
}

/// Integer (`i8` in, `i32` accumulate) sibling of [`run_epi`]: same `MT x NT` tiling and output
/// partition, but each A-row / B-column load widens `i8 -> i32` ([`KernelSimd::load_lhs`],
/// the same seam the `IntGemm` microkernel uses), and `alpha`/`beta`/`C` are all `i32`,
/// combined as `C <- alpha*dot + beta*C` in wrapping `i32` arithmetic
///
/// Because wrapping `i32` addition is associative and wrapping multiplication distributes
/// over it, this route's single fixed-order dot lands on the same result the driver's
/// panel-split accumulation produces, regardless of how either splits the sum: bit-identical
/// to both the `IntGemm` driver route and the `IntGemmVnni` dot kernel this route bypasses.
/// No epilogue variant exists here (the plain `i8 -> i32` path never fuses; requantizing
/// families keep their own dedicated route). Same reproducibility guarantee as [`run_epi`]
///
/// # Safety
/// As [`run_epi`] (A rows / B cols unit-stride along `k`, `c` not aliasing `a`/`b`, CPU supports `S`)
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_int<S>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
    ws: &mut Workspace,
    alpha: i32,
    a: *const i8,
    rsa: isize,
    csa: isize,
    b: *const i8,
    rsb: isize,
    csb: isize,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
) where
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        // No-op when both operands already stream unit-stride along k; i8 packs as-is
        // (byte copy). A pure reorder cannot change a wrapping-i32 result
        let (a, rsa, csa, b, rsb, csb) =
            prepack_operands::<i8>(ws, m, k, n, a, rsa, csa, b, rsb, csb);
        debug_assert!(
            csa == 1 && rsb == 1,
            "small_mn kernel requires A rows / B cols unit-stride along k"
        );
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count: A/B read once as i8, C written once as i32
        let bytes = m
            .saturating_mul(k)
            .saturating_add(k.saturating_mul(n))
            .saturating_mul(core::mem::size_of::<i8>())
            .saturating_add(
                m.saturating_mul(n)
                    .saturating_mul(core::mem::size_of::<i32>()),
            );

        let a = Ptr(a as *mut i8);
        let b = Ptr(b as *mut i8);
        let c = Ptr(c);

        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c) = (a, b, c);
            let a = a.0 as *const i8;
            let b = b.0 as *const i8;
            let c = c.0;
            simd.vectorize(|| {
                for q in q_start..q_end {
                    let it = q % n_row_tiles;
                    let jt = q / n_row_tiles;
                    let i0 = it * MT;
                    let j0 = jt * NT;
                    let mi = core::cmp::min(MT, m - i0);
                    let nj = core::cmp::min(NT, n - j0);
                    if mi == MT && nj == NT {
                        full_tile_int::<S, MT, NT>(
                            simd, k, i0, j0, alpha, a, rsa, b, csb, beta, c, rsc, csc,
                        );
                    } else {
                        for cc in 0..nj {
                            for ir in 0..mi {
                                cell_dot_int::<S>(
                                    simd,
                                    k,
                                    i0 + ir,
                                    j0 + cc,
                                    alpha,
                                    a,
                                    rsa,
                                    b,
                                    csb,
                                    beta,
                                    c,
                                    rsc,
                                    csc,
                                );
                            }
                        }
                    }
                }
            });
        };

        tile_sweep(m, n, bytes, par, body);
    }
}

/// Integer sibling of [`full_tile`] (see [`run_int`]): holds `MT*NT` `i32` accumulators live
/// across the `k`-sweep, widen-loading each A-row / B-column `i8 -> i32` once per depth
/// step, then finishing each cell with `reduce_sum`, an ascending scalar tail, and a
/// wrapping `alpha`/`beta` combine. Uses the same `load_lhs` / `mul_add` / `reduce_sum`
/// `i32`-accumulator seams the `IntGemm` driver kernel uses, so the 2 match bit-for-bit
///
/// # Safety
/// As [`full_tile`], with `i8` inputs / `i32` accumulator and output
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn full_tile_int<S, const MT: usize, const NT: usize>(
    simd: S,
    k: usize,
    i0: usize,
    j0: usize,
    alpha: i32,
    a: *const i8,
    rsa: isize,
    b: *const i8,
    csb: isize,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
) where
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        let lanes = <S as SimdOps<i32>>::LANES;
        let rows: [*const i8; MT] = core::array::from_fn(|r| a.offset((i0 + r) as isize * rsa));
        let cols: [*const i8; NT] = core::array::from_fn(|cc| b.offset((j0 + cc) as isize * csb));

        let mut acc: [[<S as SimdOps<i32>>::Reg; MT]; NT] = [[simd.zero(); MT]; NT];
        let mut kk = 0;
        while kk + lanes <= k {
            // Fully-qualified call: an i8 widen token also implements the requantizing
            // `KernelSimd<i8,i8,i32,{i8,u8}>` variants, so a bare `load_lhs` would be
            // ambiguous between them (see `kernel::int::i32_accumulate`)
            let av: [<S as SimdOps<i32>>::Reg; MT] = core::array::from_fn(|r| {
                <S as KernelSimd<i8, i8, i32, i32>>::load_lhs(simd, rows[r].add(kk))
            });
            for cc in 0..NT {
                let bv = <S as KernelSimd<i8, i8, i32, i32>>::load_lhs(simd, cols[cc].add(kk));
                for r in 0..MT {
                    acc[cc][r] = simd.mul_add(av[r], bv, acc[cc][r]);
                }
            }
            kk += lanes;
        }
        for cc in 0..NT {
            for r in 0..MT {
                let mut dot = simd.reduce_sum(acc[cc][r]);
                let mut t = kk;
                while t < k {
                    dot = dot.wrapping_add(
                        (*rows[r].add(t) as i32).wrapping_mul(*cols[cc].add(t) as i32),
                    );
                    t += 1;
                }
                let cp = c.offset((i0 + r) as isize * rsc + (j0 + cc) as isize * csc);
                let ov = if beta == 0 {
                    0
                } else if beta == 1 {
                    *cp
                } else {
                    beta.wrapping_mul(*cp)
                };
                *cp = alpha.wrapping_mul(dot).wrapping_add(ov);
            }
        }
    }
}

/// Integer sibling of [`cell_dot`] (edge-tile path; see [`run_int`]): a single-accumulator
/// `i8 -> i32` widen-load dot plus an ascending scalar tail, with `alpha`/`beta` folded in
/// using wrapping `i32` arithmetic
///
/// # Safety
/// As [`cell_dot`], with `i8` inputs / `i32` accumulator and output
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn cell_dot_int<S>(
    simd: S,
    k: usize,
    i: usize,
    j: usize,
    alpha: i32,
    a: *const i8,
    rsa: isize,
    b: *const i8,
    csb: isize,
    beta: i32,
    c: *mut i32,
    rsc: isize,
    csc: isize,
) where
    S: KernelSimd<i8, i8, i32, i32>,
{
    unsafe {
        let lanes = <S as SimdOps<i32>>::LANES;
        let row = a.offset(i as isize * rsa);
        let col = b.offset(j as isize * csb);
        let mut acc = simd.zero();
        let mut kk = 0;
        while kk + lanes <= k {
            acc = simd.mul_add(
                <S as KernelSimd<i8, i8, i32, i32>>::load_lhs(simd, row.add(kk)),
                <S as KernelSimd<i8, i8, i32, i32>>::load_lhs(simd, col.add(kk)),
                acc,
            );
            kk += lanes;
        }
        let mut dot = simd.reduce_sum(acc);
        while kk < k {
            dot = dot.wrapping_add((*row.add(kk) as i32).wrapping_mul(*col.add(kk) as i32));
            kk += 1;
        }
        let cp = c.offset(i as isize * rsc + j as isize * csc);
        let ov = if beta == 0 {
            0
        } else if beta == 1 {
            *cp
        } else {
            beta.wrapping_mul(*cp)
        };
        *cp = alpha.wrapping_mul(dot).wrapping_add(ov);
    }
}
