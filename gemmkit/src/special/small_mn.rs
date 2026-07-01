//! Small-matrix horizontal (inner-product) path: both output dimensions small, long `k`.
//!
//! When `m` and `n` are both far below the microtile but the contraction `k` is long, the
//! register-tiling driver is the wrong tool: it packs A/B into micropanels and pads the tiny
//! row/col tiles up to a full `MR × NR` microtile, so it computes (and reads) mostly padding.
//! This route instead computes each output as a horizontal dot
//! `C[i,j] = α·⟨A[i,:], B[:,j]⟩ + β·C[i,j]`, streaming SIMD along the contraction and reading
//! A/B **in place** — no packing, blocking, workspace, or orientation machinery. It is
//! [`crate::special::gemv`]'s `dot_rows` generalized from a vector to a small `m×n` grid.
//!
//! Both operands must stream contiguously along `k` — A rows unit-stride (`csa == 1`, row-major
//! A) and B columns unit-stride (`rsb == 1`, column-major B) — which the dispatch gate enforces;
//! a strided layout would force a scalar dot that loses to the driver's packed microkernel, so it
//! stays on the driver. The output is register-blocked into `MT × NT` tiles: each cell holds one
//! accumulator live across the whole `k`-sweep, so a full tile keeps `MT·NT` independent FMA
//! chains in flight (hiding the reduction latency) while loading each A-row and B-column once per
//! tile.
//!
//! Each output element is a single fixed-order reduction (a `reduce_sum` with fixed lane order
//! plus an ascending scalar `k`-tail) computed wholly by one worker: the output-tile partition
//! touches disjoint C tiles and adds no cross-thread reduction, so the result is **reproducible**
//! and bit-identical to the serial run for any worker count. The tile grid is decided once from
//! the shape, independent of the worker count.

use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
use crate::simd::SimdOps;

/// Output register-block tile: `MT` rows × `NT` columns of accumulators. A full tile keeps
/// `MT·NT` independent FMA chains live across the `k`-sweep (enough to saturate the FMA pipes)
/// while its A-rows and B-columns are each loaded once per depth step — an arithmetic intensity
/// of `MT·NT / (MT+NT)` MACs per vector load, which keeps the operand stream off the critical
/// path when A/B spill L1 into L2. Calibrated on Zen5 (AVX-512): `4×4` = 16 accumulators + 4
/// A-vectors + 1 B-vector = 21 of 32 `zmm`. A 16-vector ISA (x86 FMA / wasm) would spill this
/// tile and wants a smaller one; NEON's 32 vectors do not.
const MT: usize = 4;
const NT: usize = 4;

/// Dispatch a small-`m,n` horizontal GEMM, partitioning the `MT×NT` output tiles across up to
/// `par`-many workers. Like the other special paths this shape is memory-bound (the operands
/// stream from cache), so the worker count comes from the bandwidth model, not the compute ramp.
/// Each output tile is computed wholly by one worker over the full `k`, so the split adds no
/// cross-thread reduction and the result is bit-identical to the serial run.
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must not alias
/// `a`/`b`; A rows must be unit-stride (`csa == 1`) and B columns unit-stride (`rsb == 1`) so
/// both stream contiguously along `k`. Must be called only when the CPU supports `S`'s features.
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
    debug_assert!(
        csa == 1 && rsb == 1,
        "small_mn requires A rows / B cols unit-stride along k"
    );
    unsafe {
        let n_row_tiles = m.div_ceil(MT);
        let n_col_tiles = n.div_ceil(NT);
        let n_tiles = n_row_tiles * n_col_tiles;

        // Bandwidth-capped worker count: min traffic is A read once, B once, C written once.
        let sizeof = core::mem::size_of::<T>();
        let bytes = m
            .saturating_mul(k)
            .saturating_add(k.saturating_mul(n))
            .saturating_add(m.saturating_mul(n))
            .saturating_mul(sizeof);
        let n_threads = par.resolve_bandwidth(bytes, n_tiles);

        let a = Ptr(a as *mut T);
        let b = Ptr(b as *mut T);
        let c = Ptr(c);

        // Column-tile-outer flat iteration (`jt` outer): a worker's consecutive tiles share a C
        // column block, so a column-major C is stored down contiguous columns.
        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c) = (a, b, c);
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
                        full_tile::<T, S, MT, NT>(
                            simd, k, i0, j0, alpha, a, rsa, b, csb, beta, c, rsc, csc,
                        );
                    } else {
                        // Edge tile (`m`/`n` not a multiple of the tile): one SIMD dot per cell.
                        for cc in 0..nj {
                            for ir in 0..mi {
                                cell_dot::<T, S>(
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

        if n_threads <= 1 {
            body(0, n_tiles);
            return;
        }

        // Output-partitioned parallel sweep: workers pull disjoint flat-tile ranges from a
        // shared cursor. No cross-worker reduction, so no barrier and no perturbation of the
        // per-element summation order.
        let cur = JobCursor::new(n_tiles, parallel::job_grain(n_tiles, n_threads));
        parallel::for_each_worker(n_threads, |_tid| {
            while let Some((s, e)) = cur.next_chunk() {
                body(s, e);
            }
        });
    }
}

/// Compute a full `MT × NT` tile at output origin `(i0, j0)` from the contiguous-along-`k`
/// layout (`csa == 1`, `rsb == 1`): hold `MT·NT` accumulators in registers across the whole
/// `k`-sweep, loading each A-row and B-column once per depth step, then one `reduce_sum` +
/// ascending scalar tail + `β` epilogue per cell.
///
/// # Safety
/// `a`/`b`/`c` valid for the tile's reads/writes; `csa == 1` and `rsb == 1` (unit-stride along
/// `k`); the tile is fully in-bounds (`i0 + MT <= m`, `j0 + NT <= n`). Run inside `S::vectorize`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn full_tile<T, S, const MT: usize, const NT: usize>(
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
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        // A[i0+r, :] and B[:, j0+cc] are each contiguous over `k` (unit-stride).
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
                *cp = alpha.mul_add(dot, ov);
            }
        }
    }
}

/// Compute one output element `C[i,j] = α·⟨A[i,:], B[:,j]⟩ + β·C[i,j]` as a single-accumulator
/// SIMD dot over the contiguous A-row / B-column (`csa == 1`, `rsb == 1`) plus an ascending
/// scalar `k`-tail. The edge-tile path (`m`/`n` not a multiple of the register tile); `β` folded
/// into the epilogue.
///
/// # Safety
/// `a`/`b`/`c` valid for the element's reads/writes; `csa == 1` and `rsb == 1`. Run inside
/// `S::vectorize`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn cell_dot<T, S>(
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
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
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
        *cp = alpha.mul_add(dot, ov);
    }
}
