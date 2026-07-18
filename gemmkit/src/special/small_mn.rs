//! Small-matrix horizontal (inner-product) path: both output dimensions small, long `k`
//!
//! When `m` and `n` are both far below the microtile but the contraction `k` is long, the
//! register-tiling driver is the wrong tool: it packs A/B into micropanels and pads the tiny
//! row/col tiles up to a full `MR x NR` microtile, so it computes (and reads) mostly padding.
//! This route instead computes each output as a horizontal dot
//! `C[i,j] = alpha*<A[i,:], B[:,j]> + beta*C[i,j]`, streaming SIMD along the contraction and
//! reading A/B **in place**: no packing, blocking, workspace, or orientation machinery. It is
//! [`crate::special::gemv`]'s `dot_rows` generalized from a vector to a small `m x n` grid
//!
//! Both operands must stream contiguously along `k`: A rows unit-stride (`csa == 1`, row-major
//! A) and B columns unit-stride (`rsb == 1`, column-major B), which the dispatch gate enforces;
//! a strided layout would force a scalar dot that loses to the driver's packed microkernel, so it
//! stays on the driver. The output is register-blocked into `MT x NT` tiles: each cell holds one
//! accumulator live across the whole `k`-sweep, so a full tile keeps `MT*NT` independent FMA
//! chains in flight (hiding the reduction latency) while loading each A-row and B-column once per
//! tile
//!
//! Each output element is a single fixed-order reduction (a `reduce_sum` with fixed lane order
//! plus an ascending scalar `k`-tail) computed wholly by one worker: the output-tile partition
//! touches disjoint C tiles and adds no cross-thread reduction, so the result is **reproducible**
//! and bit-identical to the serial run for any worker count. The tile grid is decided once from
//! the shape, independent of the worker count

use crate::kernel::FloatGemm;
#[cfg(feature = "half")]
use crate::kernel::MixedGemm;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;
#[cfg(any(feature = "half", feature = "int8"))]
use crate::simd::KernelSimd;
use crate::simd::SimdOps;

/// Output register-block tile: `MT` rows x `NT` columns of accumulators. A full tile keeps
/// `MT*NT` independent FMA chains live across the `k`-sweep (enough to saturate the FMA pipes)
/// while its A-rows and B-columns are each loaded once per depth step: an arithmetic intensity
/// of `MT*NT / (MT+NT)` MACs per vector load, which keeps the operand stream off the critical
/// path when A/B spill L1 into L2. Calibrated on Zen5 (AVX-512): `4x4` = 16 accumulators + 4
/// A-vectors + 1 B-vector = 21 of 32 `zmm`. A 16-vector ISA (x86 FMA / wasm) would spill this
/// tile and wants a smaller one; NEON's 32 vectors do not
///
/// Confirmed on M4 (NEON): `4x4` is optimal there too. Enlarging the accumulator block
/// regresses sharply: `4x6`/`6x4`/`5x5` (29-31 `MT*NT+MT+1` live vectors) all peak at
/// 44-62 GFLOP/s vs `4x4`'s ~120, because LLVM spills the larger accumulator array on NEON.
/// `4x4` is the same low-register-pressure regime (21 of 32, ~11 spare for the wide OoO
/// window's rename headroom) the production NEON microkernel uses
const MT: usize = 4;
const NT: usize = 4;

/// Shared output-tile sweep for the small-`m,n` horizontal routes ([`run_epi`] and its mixed
/// sibling [`run_mixed_epi`]): compute the `MT x NT` tile grid, cap the worker count with the
/// bandwidth model (`bytes`, the per-type traffic, passed in so each caller keeps its own
/// `sizeof`), take the `n_threads <= 1` serial fast path, else sweep the flat tiles through a
/// shared [`JobCursor`]. `body(q_start, q_end)` computes the flat-tile range `[q_start, q_end)`
/// with the caller's own tile kernels. The grid and the output partition are identical for every
/// caller and add no cross-worker reduction, so the result is bit-identical to the serial run for
/// any worker count
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

    // Output-partitioned parallel sweep: workers pull disjoint flat-tile ranges from a
    // shared cursor. No cross-worker reduction, so no barrier and no perturbation of the
    // per-element summation order
    let cur = JobCursor::new(n_tiles, parallel::job_grain(n_tiles, n_threads));
    parallel::for_each_worker(n_threads, |_tid| {
        while let Some((s, e)) = cur.next_chunk() {
            body(s, e);
        }
    });
}

/// Dispatch a small-`m,n` horizontal GEMM, partitioning the `MT x NT` output tiles across up to
/// `par`-many workers. Like the other special paths this shape is memory-bound (the operands
/// stream from cache), so the worker count comes from the bandwidth model, not the compute ramp.
/// Each output tile is computed wholly by one worker over the full `k`, so the split adds no
/// cross-thread reduction and the result is bit-identical to the serial run
///
/// The zero-cost [`Identity`] forwarder over [`run_epi`]: with `E = Identity` the epilogue guard
/// const-folds to the raw store, so this route is byte-for-byte unchanged from the pre-epilogue
/// path and the public signature is preserved for every existing caller
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must not alias
/// `a`/`b`; A rows must be unit-stride (`csa == 1`) and B columns unit-stride (`rsb == 1`) so
/// both stream contiguously along `k`. Must be called only when the CPU supports `S`'s features
#[allow(clippy::too_many_arguments)]
pub unsafe fn run<T, S>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
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
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    // SAFETY: forwarded to `run_epi` with the zero-cost `Identity` epilogue: the raw store this
    // route always did (`E::IS_IDENTITY` folds the per-cell hook away)
    unsafe {
        run_epi::<T, S, Identity>(
            simd, m, k, n, par, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity,
        )
    }
}

/// Small-`m,n` horizontal GEMM applying the fused [`Epilogue`] `E` to each output element as its
/// single store happens, instead of materializing the raw product and mapping it afterward.
/// [`run`] is exactly this with `E = Identity`; a non-identity `E` changes only the per-cell
/// store, so the tiling / partition / read pattern is identical and the fused result equals this
/// route's plain output followed by the same scalar map. Each cell is one complete `k`-reduction,
/// so the epilogue fires exactly once per element (`row`/`col` are oriented-frame coordinates;
/// dispatch flips the bias axis on an orientation swap before calling)
///
/// # Safety
/// As [`run`], plus `epi`'s interior pointers must be valid for the (oriented) problem's `m`/`n`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_epi<T, S, E>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
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
    debug_assert!(
        csa == 1 && rsb == 1,
        "small_mn requires A rows / B cols unit-stride along k"
    );
    // `E: Copy` (an `Epilogue` supertrait): copy it out of the borrow so the `move` worker
    // closure captures it by value
    let epi = *epi;
    unsafe {
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count: min traffic is A read once, B once, C written once
        let sizeof = core::mem::size_of::<T>();
        let bytes = m
            .saturating_mul(k)
            .saturating_add(k.saturating_mul(n))
            .saturating_add(m.saturating_mul(n))
            .saturating_mul(sizeof);

        let a = Ptr(a as *mut T);
        let b = Ptr(b as *mut T);
        let c = Ptr(c);

        // Column-tile-outer flat iteration (`jt` outer): a worker's consecutive tiles share a C
        // column block, so a column-major C is stored down contiguous columns
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
                        // Edge tile (`m`/`n` not a multiple of the tile): one SIMD dot per cell
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

/// Compute a full `MT x NT` tile at output origin `(i0, j0)` from the contiguous-along-`k`
/// layout (`csa == 1`, `rsb == 1`): hold `MT*NT` accumulators in registers across the whole
/// `k`-sweep, loading each A-row and B-column once per depth step, then one `reduce_sum` +
/// ascending scalar tail + `beta` epilogue per cell. The fused [`Epilogue`] `E` is applied to the
/// final value at each cell's single store, exactly once (`E::IS_IDENTITY` const-folds it away)
///
/// # Safety
/// `a`/`b`/`c` valid for the tile's reads/writes; `csa == 1` and `rsb == 1` (unit-stride along
/// `k`); the tile is fully in-bounds (`i0 + MT <= m`, `j0 + NT <= n`). Run inside `S::vectorize`
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
        // A[i0+r, :] and B[:, j0+cc] are each contiguous over `k` (unit-stride)
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
                // Fused transform at the oriented-frame coordinate, applied once at the store
                *cp = if E::IS_IDENTITY {
                    out
                } else {
                    epi.apply(out, i0 + r, j0 + cc)
                };
            }
        }
    }
}

/// Compute one output element `C[i,j] = alpha*<A[i,:], B[:,j]> + beta*C[i,j]` as a
/// single-accumulator SIMD dot over the contiguous A-row / B-column (`csa == 1`, `rsb == 1`) plus
/// an ascending scalar `k`-tail. The edge-tile path (`m`/`n` not a multiple of the register
/// tile); `beta` folded into the epilogue, then the fused [`Epilogue`] `E` applied once at the
/// store (`E::IS_IDENTITY` const-folds it away)
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
        let row = a.offset(i as isize * rsa); // A[i, :] contiguous (csa == 1)
        let col = b.offset(j as isize * csb); // B[:, j] contiguous (rsb == 1)
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
        // Fused transform at the oriented-frame coordinate, applied once at the store
        *cp = if E::IS_IDENTITY {
            out
        } else {
            epi.apply(out, i, j)
        };
    }
}

/// Mixed-precision (`f16`/`bf16` inputs, `f32` accumulate) sibling of [`run`]: same `MT x NT`
/// tiling and output partition, but each A-row / B-col is **widen-loaded** `N -> f32`
/// ([`KernelSimd::load_lhs`]), accumulated in `f32`, and rounded to the narrow output once in the
/// per-cell epilogue. `alpha`/`beta` arrive already widened to `f32`. Same reproducibility as [`run`]
///
/// The zero-cost [`Identity`] forwarder over [`run_mixed_epi`]: with `E = Identity` the per-cell
/// epilogue guard const-folds to the raw narrowing store, so this route is byte-for-byte unchanged
/// from the pre-epilogue path and the public signature is preserved for every existing caller
///
/// # Safety
/// As [`run`] (A rows / B cols unit-stride along `k`, `c` not aliasing `a`/`b`, CPU supports `S`)
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_mixed<N, S>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
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
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    // SAFETY: forwarded to `run_mixed_epi` with the zero-cost `Identity` epilogue: the raw
    // narrowing store this route always did (`E::IS_IDENTITY` folds the per-cell hook away)
    unsafe {
        run_mixed_epi::<N, S, Identity>(
            simd, m, k, n, par, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity,
        )
    }
}

/// Mixed-precision small-`m,n` horizontal GEMM applying the fused [`Epilogue`] `E` (over the
/// [`MixedGemm`] family) to each output element at its single narrowing store, instead of
/// materializing the raw product and mapping it afterward. [`run_mixed`] is exactly this with
/// `E = Identity`; a non-identity `E` changes only the per-cell store, so the tiling / partition /
/// read pattern is identical. The epilogue applies to the `f32` cell value **before** the single
/// narrowing (matching the driver-path mixed semantics), so it is more precise than a separate map
/// (`row`/`col` are oriented-frame coordinates; dispatch flips the bias axis on an orientation
/// swap before calling)
///
/// # Safety
/// As [`run_mixed`], plus `epi`'s interior pointers must be valid for the (oriented) `m`/`n`
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_mixed_epi<N, S, E>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
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
    debug_assert!(
        csa == 1 && rsb == 1,
        "small_mn requires A rows / B cols unit-stride along k"
    );
    // `E: Copy` (an `Epilogue` supertrait): copy it out of the borrow so the `move` worker
    // closure captures it by value
    let epi = *epi;
    unsafe {
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count: A read once, B once, C written once (narrow bytes)
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

/// Mixed sibling of [`full_tile`] (see [`run_mixed_epi`]). The `f32` scalar tail and epilogue use
/// plain `a*b + c` (not the fused intrinsic) so the route stays reproducible; the fused
/// [`Epilogue`] `E` is applied to the `f32` cell value at each cell's single narrowing store,
/// exactly once (`E::IS_IDENTITY` const-folds it away to a raw narrowing store)
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
                // Fused transform on the `f32` cell value, before the single narrowing
                *cp = if E::IS_IDENTITY {
                    N::narrow(out)
                } else {
                    epi.apply(out, i0 + r, j0 + cc)
                };
            }
        }
    }
}

/// Mixed sibling of [`cell_dot`] (edge-tile path; see [`run_mixed_epi`]): an `f32` widen-load dot,
/// with the fused [`Epilogue`] `E` applied to the `f32` cell value before it is rounded to `N`
/// once (`E::IS_IDENTITY` const-folds to a raw narrowing store)
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
        // Fused transform on the `f32` cell value, before the single narrowing
        *cp = if E::IS_IDENTITY {
            N::narrow(out)
        } else {
            epi.apply(out, i, j)
        };
    }
}

/// Integer (`i8` inputs, `i32` accumulate) sibling of [`run`]: same `MT x NT` tiling and output
/// partition, but each A-row / B-col is **widen-loaded** `i8 -> i32` ([`KernelSimd::load_lhs`],
/// the same seam the `IntGemm` microkernel uses) and accumulated in `i32`. `alpha`/`beta`/`C` are
/// all `i32`, combined `C <- alpha*<A[i,:], B[:,j]> + beta*C[i,j]` in wrapping `i32`
///
/// Bit-identical to the register-tiling driver route (`IntGemm`, or the `IntGemmVnni` dot kernel
/// this route bypasses): `i32` is a ring, so wrapping add is fully associative and wrapping mul
/// distributes over it, so the driver's panel-split accumulation and this single fixed-order dot
/// land on the same `i32`. No epilogue variant exists (the `i8 -> i32` `IntTask` path never fuses;
/// the fused requantizing families keep their own dedicated route). Same reproducibility as [`run`]
///
/// # Safety
/// As [`run`] (A rows / B cols unit-stride along `k`, `c` not aliasing `a`/`b`, CPU supports `S`)
#[cfg(feature = "int8")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_int<S>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    par: Parallelism,
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
    debug_assert!(
        csa == 1 && rsb == 1,
        "small_mn requires A rows / B cols unit-stride along k"
    );
    unsafe {
        let n_row_tiles = m.div_ceil(MT);

        // Bandwidth-capped worker count: A / B read once as `i8`, C written once as `i32`
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

/// Integer sibling of [`full_tile`] (see [`run_int`]): hold `MT*NT` `i32` accumulators live across
/// the `k`-sweep, widen-loading each A-row and B-column `i8 -> i32` once per depth step, then one
/// `reduce_sum` + ascending scalar `k`-tail + wrapping `alpha`/`beta` combine per cell. The
/// `load_lhs` / `mul_add` / `reduce_sum` are the `i32`-accumulator seams (wrapping arithmetic), so
/// the per-cell result matches the `IntGemm` driver's exactly
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
            // UFCS: a widen token also carries the requant `KernelSimd<i8,i8,i32,{i8,u8}>` impls,
            // so the bare `load_lhs` would be ambiguous (see `kernel::int::i32_accumulate`)
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

/// Integer sibling of [`cell_dot`] (edge-tile path; see [`run_int`]): an `i8 -> i32` widen-load
/// single-accumulator dot plus ascending scalar tail, `beta`/`alpha` folded in wrapping `i32`
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
