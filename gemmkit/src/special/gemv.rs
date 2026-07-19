//! gemv: matrix*vector product, dispatched for `n == 1` or `m == 1` shapes
//!
//! A gemv touches `O(m*k)` bytes to do `O(m*k)` flops, so it is memory-bound: the arithmetic is
//! trivial and the whole design question is minimizing DRAM traffic. Both `n == 1` and `m == 1`
//! reduce to the same core routine by treating the matrix (transposed for `m == 1`) as a
//! `rows x k` block times a `k`-vector; every stride combination is handled correctly, and the
//! contiguous ones get a vectorized path
//!
//! For a column-major matrix (the axpy shape) there are 2 strategies that produce the same
//! result and differ only in memory traffic, since both fuse the accumulation for a given output
//! element in the same ascending-`k` order:
//!
//! * Register-blocked output: hold a few-SIMD-register-wide panel of output rows in registers
//!   across the whole `k`-sweep, so the matrix and the output panel are each read exactly once.
//!   Used once the output is too large to stay resident in the last-level cache across that
//!   sweep, where the alternative's per-column output re-read would otherwise reach DRAM
//! * Plain axpy: column-outer, re-reading and re-writing the output panel every few columns
//!   instead of holding it in registers. Cheaper when the output stays cache-resident, and its
//!   single contiguous matrix stream is what a large `k` wants
//!
//! Every output element is reduced over the full `k` by exactly 1 worker, so splitting the output
//! rows across workers changes nothing about how any single element is computed: the result is
//! reproducible, and in fact bit-identical, at any worker count
//!
//! The mixed-precision twin (`f16`/`bf16` in, `f32` accumulate), [`run_mixed`], sits in the lower
//! half of this file: same row partition and same reproducibility guarantee, but every load
//! widens through the `KernelSimd` seam to `f32` and the narrow result is rounded back exactly
//! once, at the store

use crate::kernel::FloatGemm;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
#[cfg(feature = "half")]
use crate::scalar::NarrowFloat;
#[cfg(feature = "half")]
use crate::simd::KernelSimd;
use crate::simd::SimdOps;

/// Row-panel width, in SIMD registers, for the axpy register-blocking strategy: `MB_REG`
/// accumulator registers plus 1 broadcast register for the scaled RHS element stay well inside
/// the vector file on every ISA, while a `MB_REG`-wide panel gives the matrix read a wide
/// contiguous burst per column. Doubles as the row-partitioning grain, so worker boundaries fall
/// on panel edges and every row lands in the same wide-panel/single-register/scalar tier no
/// matter how the rows are split, keeping serial and parallel bit-for-bit identical
const MB_REG: usize = 8;

/// Output-row partition shared by both gemv cores ([`core_epi`] and [`core_mixed`]): with
/// `n_threads <= 1` this just calls `body(0, rows)` on the caller; otherwise workers draw disjoint
/// panel ranges from a [`JobCursor`] and each calls `body(row_start, row_end)` on its own range
/// only. `block` should be a multiple of `lanes` (the axpy/dot tier boundaries), so a row's
/// SIMD-vs-scalar treatment never depends on where the partition happened to cut, and no worker
/// ever reduces another worker's rows, so there is nothing to synchronize once every `body` call
/// returns
#[inline]
fn row_sweep(
    rows: usize,
    block: usize,
    n_threads: usize,
    body: impl Fn(usize, usize) + Copy + Send + Sync,
) {
    if n_threads <= 1 {
        body(0, rows);
        return;
    }
    let n_blocks = rows.div_ceil(block);
    let cur = JobCursor::new(n_blocks, parallel::job_grain(n_blocks, n_threads));
    parallel::for_each_worker(n_threads, |_tid| {
        while let Some((bs, be)) = cur.next_chunk() {
            let row_start = bs * block;
            let row_end = core::cmp::min(be * block, rows);
            body(row_start, row_end);
        }
    });
}

/// Entry point for a gemv shape: dispatches to the core routine, partitioning the output rows
/// across up to `par` workers. Since gemv is bandwidth-bound rather than compute-bound, the
/// worker count comes from a bandwidth model ([`Parallelism::resolve_bandwidth`]) instead of the
/// usual compute ramp: past however many cores saturate DRAM, adding workers stops helping.
/// Every output row is reduced over the full `k` by 1 worker, so the split never crosses a
/// reduction boundary and the result is unchanged from the serial run
///
/// Calls [`run_typed_epi`] with the zero-cost [`Identity`] epilogue: `E::IS_IDENTITY` folds the
/// fused-epilogue pass away entirely, so this stays the plain, epilogue-free route while sharing
/// its implementation with the fused entry point
///
/// # Safety
/// Pointers must be valid for the regions their strides and sizes imply; `c` must not alias
/// `a`/`b`. The CPU must support `S`'s target features
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_typed<T, S>(
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
    // SAFETY: `Identity` is the zero-cost epilogue, so this reproduces exactly what the route
    // stored before `run_typed_epi` existed (`E::IS_IDENTITY` folds the epilogue sweep away)
    unsafe {
        run_typed_epi::<T, S, Identity>(
            simd, m, k, n, par, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity,
        )
    }
}

/// gemv entry point with a fused [`Epilogue`] `E` applied to the output. gemv is dispatched
/// before orientation normalization runs, so `epi` still speaks the caller's original, unflipped
/// coordinate frame. In the `n == 1` branch output element `i` is `C[i, 0]` (`swap_rc = false`);
/// the `m == 1` branch instead views `C^T`, where element `i` is `C[0, i]` (`swap_rc = true`)
///
/// # Safety
/// As [`run_typed`], plus `epi`'s interior pointers must be valid for the problem's `m`/`n`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_typed_epi<T, S, E>(
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
    unsafe {
        if n == 1 {
            // C (m x 1) = beta*C + alpha*A*b: A is m x k, b a k-vector, output element `i` is
            // `C[i, 0]`, so the epilogue reads coordinate `(i, 0)` (`swap_rc = false`)
            core_epi::<T, S, E>(
                simd, m, k, par, alpha, a, rsa, csa, b, rsb, beta, c, rsc, false, epi,
            );
        } else {
            // C (1 x n) = beta*C + alpha*a*B: view B^T (n x k) times the k-vector a, so
            // B^T[j,k] = B[k,j] gives row stride csb, column stride rsb, output stride csc. Under
            // that transpose output element `i` is `C[0, i]`, so `swap_rc = true`
            core_epi::<T, S, E>(
                simd, n, k, par, alpha, b, csb, rsb, a, csa, beta, c, csc, true, epi,
            );
        }
    }
}

/// `out[i] = beta*out[i] + alpha * sum_k(mat[i,k]*vec[k])` for `i in 0..rows`: partitions the
/// rows across bandwidth-capped workers, picks a layout strategy once for the whole call, then,
/// if `E` is not [`Identity`], sweeps each worker's own row range once more to apply the fused
/// epilogue in place. `swap_rc` selects the epilogue coordinate for output element `i`: `(i, 0)`
/// when `false` (the `n == 1` shape), `(0, i)` when `true` (the transposed `m == 1` view)
///
/// # Safety
/// `mat` valid for the `rows x k` region at `mat_rs`/`mat_cs`; `vec` valid for `k` reads at
/// `vec_s`; `out` valid for `rows` writes at `out_s`, and for `rows` reads too when `beta != 0`;
/// `epi`'s interior pointers valid for the problem's `m`/`n`. The CPU must support `S`'s target
/// features
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn core_epi<T, S, E>(
    simd: S,
    rows: usize,
    k: usize,
    par: Parallelism,
    alpha: T,
    mat: *const T,
    mat_rs: isize,
    mat_cs: isize,
    vec: *const T,
    vec_s: isize,
    beta: T,
    out: *mut T,
    out_s: isize,
    swap_rc: bool,
    epi: &E,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
    E: Epilogue<FloatGemm<T>>,
{
    // `Epilogue: Copy`, so dereferencing hands each `move` worker closure below its own value
    let epi = *epi;
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let sizeof = core::mem::size_of::<T>();

        // Classify the layout once, up front, so every worker below runs the identical branch
        // and no worker's tier choice can depend on which rows it happened to draw
        let axpy = mat_rs == 1 && out_s == 1;
        let output_block = axpy && output_register_block(rows, sizeof, k);
        let dot = !axpy && mat_cs == 1 && vec_s == 1;

        // The minimum traffic this call must move: the matrix once, the vector once, the output
        // once. `rows` caps the worker count so no worker can end up with 0 rows
        let bytes_touched = (rows.saturating_mul(k) + k + rows).saturating_mul(sizeof);
        let n_threads = par.resolve_bandwidth(bytes_touched, rows);

        // Grain the partition on register-blocked panels (`MB_REG*lanes`) for the axpy path, so
        // worker boundaries land on panel edges; plain SIMD rows otherwise. Both are multiples of
        // `lanes`, so a row's SIMD-vs-scalar tier never shifts with how the rows are cut
        let block = if output_block { MB_REG * lanes } else { lanes }.max(1);

        let mat = Ptr(mat as *mut T);
        let vec = Ptr(vec as *mut T);
        let out = Ptr(out);

        let body = move |row_start: usize, row_end: usize| {
            let (mat, vec, out, epi) = (mat, vec, out, epi);
            let mat = mat.0 as *const T;
            let vec = vec.0 as *const T;
            let out = out.0;
            // Stay inside the ISA's `#[target_feature]` token so every SIMD call below compiles
            // to feature-enabled codegen, the same discipline the driver uses per tile
            simd.vectorize(|| {
                if output_block {
                    axpy_regblocked::<T, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_cs, vec, vec_s, beta, out,
                    );
                } else if axpy {
                    axpy_plain::<T, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_cs, vec, vec_s, beta, out,
                    );
                } else if dot {
                    dot_rows::<T, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_rs, vec, beta, out, out_s,
                    );
                } else {
                    strided_rows::<T, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_rs, mat_cs, vec, vec_s, beta,
                        out, out_s,
                    );
                }
                // A 2nd, separate pass over `[row_start, row_end)` applies the epilogue in place,
                // rather than threading `epi` into the 4 strategy kernels above: the output is 1
                // vector, tiny next to the matrix read that dominates this memory-bound path, so
                // the extra pass costs nothing and it keeps the 4 kernels themselves identical to
                // the non-fused build. Since the strategy above already stored the exact bits
                // plain gemv would, re-reading and mapping here matches gemm-then-map bit for
                // bit. Each output element belongs to exactly 1 worker's range, so it is mapped
                // exactly once, after its full `k`-reduction; `E::IS_IDENTITY` folds this whole
                // block away for the non-fused instantiation. Staying inside `vectorize` keeps
                // the epilogue in the same target-feature token, even though `apply` itself is
                // scalar
                if !E::IS_IDENTITY {
                    for i in row_start..row_end {
                        let op = out.offset(i as isize * out_s);
                        let (r, c) = if swap_rc { (0, i) } else { (i, 0) };
                        *op = epi.apply(*op, r, c);
                    }
                }
            });
        };

        // Disjoint row ranges, no cross-worker reduction (see [`row_sweep`]), so this matches the
        // serial result regardless of `n_threads`
        row_sweep(rows, block, n_threads, body);
    }
}

/// Whether an axpy-shape gemv should use the register-blocked output strategy instead of the
/// plain one: both conditions must hold. 1st, the output (`rows*sizeof` bytes) must be large
/// enough that the plain form's per-column output re-read would spill out of the last-level cache
/// (the byte gate, a fraction of L3, lives in [`crate::cache::gemv_regblock_engage_bytes`]).
/// 2nd, `k` must be small enough (`<= k_stream_max`) that register-blocking's `k` concurrent
/// matrix column-streams still fit the hardware prefetcher's window. Below the byte gate the
/// output stays cache-resident, so the plain form's cheap re-reads and single contiguous matrix
/// stream win outright; above the `k` gate, register-blocking's many streams start thrashing the
/// prefetcher instead of helping
#[inline]
fn output_register_block(rows: usize, sizeof: usize, k: usize) -> bool {
    k <= crate::tuning::k_stream_max()
        && rows.saturating_mul(sizeof) > crate::cache::gemv_regblock_engage_bytes()
}

/// Register-blocked axpy over output rows `[s, e)`: an output panel stays in SIMD registers
/// across the whole `k`-sweep, so the column-major matrix and the output are each read once (the
/// output written once too). Folding `beta` into the accumulator's initial value means `beta ==
/// 0` never touches the existing output. Row for row this computes the same ascending-`k` fused
/// accumulation, and applies the same wide-panel/single-register/scalar-remainder tiering, as
/// [`axpy_plain`], so the 2 kernels are interchangeable and produce identical output
///
/// # Safety
/// `mat`/`vec` valid for the region the strides imply; `out` valid for `[s, e)` writes, and for
/// `[s, e)` reads too when `beta != 0`. Must run inside `S`'s `vectorize` context
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn axpy_regblocked<T, S>(
    simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: T,
    mat: *const T,
    mat_cs: isize,
    vec: *const T,
    vec_s: isize,
    beta: T,
    out: *mut T,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let mb = MB_REG * lanes;
        let mut i = s;

        // A wide panel of `MB_REG` accumulators, held in registers across the whole k-sweep
        while i + mb <= e {
            let mut acc = [simd.zero(); MB_REG];
            // acc <- beta*out; beta == 0 skips the load, leaving the zero init untouched
            if beta == T::ONE {
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.loadu(out.add(i + r * lanes));
                }
            } else if beta != T::ZERO {
                let bv = simd.splat(beta);
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.mul(simd.loadu(out.add(i + r * lanes)), bv);
                }
            }
            for kk in 0..k {
                let sv = simd.splat(alpha * *vec.offset(kk as isize * vec_s));
                let col = mat.offset(kk as isize * mat_cs).add(i);
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.mul_add(simd.loadu(col.add(r * lanes)), sv, *a);
                }
            }
            for (r, a) in acc.iter().enumerate() {
                simd.storeu(out.add(i + r * lanes), *a);
            }
            i += mb;
        }

        // Then single-SIMD-register rows, then a sub-lane scalar remainder: the same 2 tiers
        // [`axpy_plain`] uses, so a row rounds the same way regardless of which path took it
        while i + lanes <= e {
            let mut acc = if beta == T::ONE {
                simd.loadu(out.add(i))
            } else if beta == T::ZERO {
                simd.zero()
            } else {
                simd.mul(simd.loadu(out.add(i)), simd.splat(beta))
            };
            for kk in 0..k {
                let sv = simd.splat(alpha * *vec.offset(kk as isize * vec_s));
                acc = simd.mul_add(simd.loadu(mat.offset(kk as isize * mat_cs).add(i)), sv, acc);
            }
            simd.storeu(out.add(i), acc);
            i += lanes;
        }
        while i < e {
            let op = out.add(i);
            let mut acc = if beta == T::ZERO {
                T::ZERO
            } else if beta == T::ONE {
                *op
            } else {
                beta * *op
            };
            for kk in 0..k {
                let s = alpha * *vec.offset(kk as isize * vec_s);
                acc = s.mul_add(*mat.offset(kk as isize * mat_cs).add(i), acc);
            }
            *op = acc;
            i += 1;
        }
    }
}

/// Plain column-outer axpy over output rows `[s, e)`: `out[i] = beta*out[i] + sum_k((alpha*vec[k])*
/// mat[i,k])`, re-reading and re-writing the output panel every `KB` columns instead of holding it
/// in registers for the whole `k`-sweep (the strategy for the cache-resident regime, where that
/// periodic re-touch is cheap). `beta` is applied once, up front, as a pre-scale over the whole
/// range, so each worker touches only its own rows. Produces the same ascending-`k` fused
/// accumulation per row, and the same SIMD-vs-scalar row split, as [`axpy_regblocked`]
///
/// # Safety
/// As [`axpy_regblocked`]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn axpy_plain<T, S>(
    simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: T,
    mat: *const T,
    mat_cs: isize,
    vec: *const T,
    vec_s: isize,
    beta: T,
    out: *mut T,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        // Apply beta once, up front, over the whole range (beta == 0 overwrites without a read)
        for i in s..e {
            let op = out.add(i);
            if beta == T::ZERO {
                *op = T::ZERO;
            } else if beta != T::ONE {
                *op = beta * *op;
            }
        }
        // Group KB columns per output load/store instead of 1: the output panel is then touched
        // once every KB columns rather than every column, which is this form's main cache cost
        // once the matrix read itself is DRAM-bound, while keeping only KB matrix column-streams
        // active at once. The KB steps still fuse in ascending-k order, so the result matches the
        // 1-column-at-a-time form below bit for bit
        const KB: usize = 4;
        let mut kk = 0;
        while kk + KB <= k {
            let scal: [T; KB] =
                core::array::from_fn(|j| alpha * *vec.offset((kk + j) as isize * vec_s));
            let sv: [S::Reg; KB] = core::array::from_fn(|j| simd.splat(scal[j]));
            let col: [*const T; KB] =
                core::array::from_fn(|j| mat.offset((kk + j) as isize * mat_cs));
            let mut i = s;
            while i + lanes <= e {
                let mut ov = simd.loadu(out.add(i));
                for j in 0..KB {
                    ov = simd.mul_add(simd.loadu(col[j].add(i)), sv[j], ov);
                }
                simd.storeu(out.add(i), ov);
                i += lanes;
            }
            while i < e {
                let op = out.add(i);
                let mut o = *op;
                for j in 0..KB {
                    o = scal[j].mul_add(*col[j].add(i), o);
                }
                *op = o;
                i += 1;
            }
            kk += KB;
        }
        // The remaining `k % KB` columns, 1 at a time
        while kk < k {
            let scal = alpha * *vec.offset(kk as isize * vec_s);
            let sv = simd.splat(scal);
            let col = mat.offset(kk as isize * mat_cs);
            let mut i = s;
            while i + lanes <= e {
                let mv = simd.loadu(col.add(i));
                let ov = simd.loadu(out.add(i));
                simd.storeu(out.add(i), simd.mul_add(mv, sv, ov));
                i += lanes;
            }
            while i < e {
                let op = out.add(i);
                *op = scal.mul_add(*col.add(i), *op);
                i += 1;
            }
            kk += 1;
        }
    }
}

/// Row-group width for the dot path's register blocking: `DOT_RB` output rows are reduced side by
/// side, each keeping its own accumulator, so `DOT_RB` independent FMA chains overlap across the
/// shared `k`-sweep, and `vec` is loaded once per depth step and shared by the whole group. A
/// single row's reduction is 1 dependent `mul_add` chain and so is latency-bound (an FMA takes
/// about 4 cycles on Zen5, only 1 in flight) well short of the 2 FMAs/cycle the hardware can
/// retire; running several rows' chains together fills that latency gap. Set to 4 by
/// measurement, not by the naive latency*throughput target of about 8: 4 chains already recover
/// most of the stall, and every extra row opens another concurrent matrix read-stream, so 8
/// over-subscribes the prefetcher and grows the per-group working set instead of helping
/// further. 4 wins on both cache-resident shapes and under the DRAM-bound bandwidth cap (where
/// the extra memory-level parallelism from a few streams even beats a single core's naive
/// single-stream rate); only the rare few-rows/long-`k` shape favors 8. Not a partition grain:
/// [`row_sweep`] still cuts the output into `lanes`-wide granules for this path, so, unlike
/// [`MB_REG`], this value carries no serial-vs-parallel reproducibility constraint
const DOT_RB: usize = 4;

/// Dot-form sweep over output rows `[s, e)`, for a row-major matrix: `out[i] = alpha*<mat[i,:],
/// vec> + beta*out[i]` in 1 pass (each matrix row read once, the output touched once, `vec` reused
/// from L1 across every row). Rows are processed [`DOT_RB`] at a time so their FMA chains overlap;
/// the trailing `< DOT_RB` rows fall back to the plain 1-row-at-a-time form
///
/// # Safety
/// As [`axpy_regblocked`], with `mat`'s rows contiguous over `k` and `vec` unit-stride
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn dot_rows<T, S>(
    simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: T,
    mat: *const T,
    mat_rs: isize,
    vec: *const T,
    beta: T,
    out: *mut T,
    out_s: isize,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let mut i = s;

        // DOT_RB rows at a time, each with its own SIMD accumulator (and its own scalar-tail
        // accumulator once k % lanes != 0), so DOT_RB FMA chains run concurrently over the
        // shared k-sweep. Every row still follows dot_contiguous's exact order: 1 accumulator,
        // ascending k in lanes-sized steps via mul_add, then reduce_sum, then an ascending
        // scalar tail, so interleaving the rows' chains leaves each row's result bit-identical to
        // the per-row dot_contiguous used by the tail below (and by small_mn's edge cell)
        while i + DOT_RB <= e {
            let rows: [*const T; DOT_RB] =
                core::array::from_fn(|r| mat.offset((i + r) as isize * mat_rs));
            let mut acc = [simd.zero(); DOT_RB];
            let mut kk = 0;
            while kk + lanes <= k {
                // vec is shared: load it once per step and feed every row's chain
                let v = simd.loadu(vec.add(kk));
                for r in 0..DOT_RB {
                    acc[r] = simd.mul_add(simd.loadu(rows[r].add(kk)), v, acc[r]);
                }
                kk += lanes;
            }
            let mut dots: [T; DOT_RB] = core::array::from_fn(|r| simd.reduce_sum(acc[r]));
            while kk < k {
                let y = *vec.add(kk);
                for r in 0..DOT_RB {
                    dots[r] = (*rows[r].add(kk)).mul_add(y, dots[r]);
                }
                kk += 1;
            }
            for (r, dot) in dots.into_iter().enumerate() {
                let op = out.offset((i + r) as isize * out_s);
                let ov = if beta == T::ZERO {
                    T::ZERO
                } else if beta == T::ONE {
                    *op
                } else {
                    beta * *op
                };
                *op = alpha.mul_add(dot, ov);
            }
            i += DOT_RB;
        }

        // The < DOT_RB tail: the plain per-row form the blocked groups above reproduce exactly
        while i < e {
            let row = mat.offset(i as isize * mat_rs);
            let dot = super::dot_contiguous::<T, S>(simd, k, row, vec);
            let op = out.offset(i as isize * out_s);
            let ov = if beta == T::ZERO {
                T::ZERO
            } else if beta == T::ONE {
                *op
            } else {
                beta * *op
            };
            *op = alpha.mul_add(dot, ov);
            i += 1;
        }
    }
}

/// Fully strided fallback over output rows `[s, e)`, used when neither the matrix rows nor `vec`
/// are contiguous: a scalar dot per row, with `beta` applied in the per-row epilogue
///
/// # Safety
/// As [`axpy_regblocked`], for arbitrary strides
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn strided_rows<T, S>(
    _simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: T,
    mat: *const T,
    mat_rs: isize,
    mat_cs: isize,
    vec: *const T,
    vec_s: isize,
    beta: T,
    out: *mut T,
    out_s: isize,
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        for i in s..e {
            let mut dot = T::ZERO;
            for kk in 0..k {
                dot = (*mat.offset(i as isize * mat_rs + kk as isize * mat_cs))
                    .mul_add(*vec.offset(kk as isize * vec_s), dot);
            }
            let op = out.offset(i as isize * out_s);
            let ov = if beta == T::ZERO {
                T::ZERO
            } else if beta == T::ONE {
                *op
            } else {
                beta * *op
            };
            *op = alpha.mul_add(dot, ov);
        }
    }
}

// Mixed-precision gemv (f16/bf16 operands, f32 accumulate): the narrow twin of the float routines
// above. Same output-row partition and same reproducibility guarantee, but every N load widens to
// f32 through the KernelSimd<N, N, f32, N> seam, the reduction runs in f32, and the result rounds
// back to N exactly once at the store (the same single-rounding discipline small_mn::run_mixed
// follows). Kept as its own family instead of generalizing the float code over a widen seam, so
// the float instantiation stays byte-identical: f32 is not a NarrowFloat, so it has no
// widen/narrow scalar ops for such a generalization to fold to. i8 and complex gemv are out of
// scope here

/// Entry point for a mixed-precision gemv shape (`f16`/`bf16` operands, `f32` accumulate): the
/// sibling of [`run_typed`], viewing the `m == 1` problem as a transposed `n x k` matrix times a
/// `k`-vector exactly as [`run_typed_epi`] does. `alpha`/`beta` arrive already widened to `f32`.
/// There is no fused-epilogue sibling here: the mixed fused path deliberately keeps gemv on the
/// general driver instead (see `dispatch/mixed.rs`'s `run_typed_mixed_fused`), so this route stays
/// plain-only
///
/// # Safety
/// As [`run_typed`], with `N` operands and an `f32` accumulator; `c` must not alias `a`/`b`, and
/// the CPU must support `S`'s target features
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
    unsafe {
        if n == 1 {
            // C (m x 1) = beta*C + alpha*A*b: A is m x k, b a k-vector, output element `i` is
            // `C[i, 0]`
            core_mixed::<N, S>(simd, m, k, par, alpha, a, rsa, csa, b, rsb, beta, c, rsc);
        } else {
            // C (1 x n) = beta*C + alpha*a*B: view B^T (n x k) times a (k-vector), so
            // B^T[j,k] = B[k,j] gives row stride csb, column stride rsb, output stride csc;
            // output element `i` is `C[0, i]`
            core_mixed::<N, S>(simd, n, k, par, alpha, b, csb, rsb, a, csa, beta, c, csc);
        }
    }
}

/// Mixed-precision sibling of [`core_epi`], without a fused epilogue: `out[i] =
/// narrow(beta*out[i] + alpha * sum_k(mat[i,k]*vec[k]))`, the reduction run in `f32` and rounded
/// to `N` exactly once at the store. Splits the output rows across bandwidth-capped workers over
/// disjoint panels ([`row_sweep`]), picking from 3 layout strategies mirroring the float core's
/// dot/axpy/strided split: the dot form for a contiguous-`k` matrix row ([`dot_rows_mixed`]), the
/// register-blocked axpy for a column-major matrix ([`axpy_mixed`]), and the fully strided
/// fallback ([`strided_rows_mixed`]). Unlike the float axpy, the mixed axpy has no plain
/// column-outer variant: that form re-reads and re-writes the output panel every depth group,
/// which would round the narrow output more than once, so the mixed path always keeps the panel
/// in `f32` registers for the whole `k`-sweep instead (matrix read once, output written and
/// rounded once)
///
/// # Safety
/// `mat` valid for the `rows x k` region at `mat_rs`/`mat_cs`; `vec` valid for `k` reads at
/// `vec_s`; `out` valid for `rows` writes (and reads when `beta != 0`) at `out_s`. The CPU must
/// support `S`'s target features
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn core_mixed<N, S>(
    simd: S,
    rows: usize,
    k: usize,
    par: Parallelism,
    alpha: f32,
    mat: *const N,
    mat_rs: isize,
    mat_cs: isize,
    vec: *const N,
    vec_s: isize,
    beta: f32,
    out: *mut N,
    out_s: isize,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let sizeof = core::mem::size_of::<N>();

        // Classify the layout once, up front, so every worker runs the identical branch. The
        // axpy form always register-blocks (see its own doc), so there is no output-size gate
        // here the way there is in the float core
        let axpy = mat_rs == 1 && out_s == 1;
        let dot = !axpy && mat_cs == 1 && vec_s == 1;

        // The minimum narrow-element traffic: the matrix once, the vector once, the output once
        // `rows` caps the worker count so no worker can end up with 0 rows
        let bytes_touched = (rows.saturating_mul(k) + k + rows).saturating_mul(sizeof);
        let n_threads = par.resolve_bandwidth(bytes_touched, rows);

        // Grain the partition on register-blocked panels (`MB_REG*lanes`) for the axpy path, so
        // worker boundaries land on panel edges; plain SIMD rows otherwise. Both are multiples of
        // `lanes`, so a row's SIMD-vs-scalar tier never shifts with how the rows are cut
        let block = if axpy { MB_REG * lanes } else { lanes }.max(1);

        let mat = Ptr(mat as *mut N);
        let vec = Ptr(vec as *mut N);
        let out = Ptr(out);

        let body = move |row_start: usize, row_end: usize| {
            let (mat, vec, out) = (mat, vec, out);
            let mat = mat.0 as *const N;
            let vec = vec.0 as *const N;
            let out = out.0;
            // Stay inside the ISA's `#[target_feature]` token, as the float core does
            simd.vectorize(|| {
                if axpy {
                    axpy_mixed::<N, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_cs, vec, vec_s, beta, out,
                    );
                } else if dot {
                    dot_rows_mixed::<N, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_rs, vec, beta, out, out_s,
                    );
                } else {
                    strided_rows_mixed::<N, S>(
                        simd, row_start, row_end, k, alpha, mat, mat_rs, mat_cs, vec, vec_s, beta,
                        out, out_s,
                    );
                }
            });
        };

        row_sweep(rows, block, n_threads, body);
    }
}

/// Register-blocked mixed axpy over output rows `[s, e)`, for a column-major matrix (depth stride
/// `mat_cs`): an `f32` accumulator panel stays in registers across the whole `k`-sweep, so the
/// narrow matrix and output are each read once and the output is rounded to `N` exactly once, at
/// the store. Folding `beta` into the accumulator's initial value means `beta == 0` never touches
/// the existing output. Every `N` load widens to `f32` ([`KernelSimd::load_lhs`] /
/// [`KernelSimd::load_out`]) and the result narrows back on store ([`KernelSimd::store_out`]);
/// the wide-panel/single-register/scalar-remainder row tiering is the mixed twin of
/// [`axpy_regblocked`], so a row's tier never depends on the partition. The SIMD tiers use a
/// fused `f32` `mul_add`; the sub-lane scalar remainder instead uses plain `f32` `a*b + c`
/// (matching [`crate::special::small_mn`]'s mixed tail), a choice made per row, so it cannot
/// differ between the serial and parallel sweeps
///
/// # Safety
/// `mat`/`vec` valid for the region the strides imply; `out` valid for `[s, e)` writes, and for
/// `[s, e)` reads too when `beta != 0`. Must run inside `S`'s `vectorize` context
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn axpy_mixed<N, S>(
    simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: f32,
    mat: *const N,
    mat_cs: isize,
    vec: *const N,
    vec_s: isize,
    beta: f32,
    out: *mut N,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mb = MB_REG * lanes;
        let mut i = s;

        // A wide panel of `MB_REG` f32 accumulators, held in registers across the whole k-sweep
        while i + mb <= e {
            let mut acc: [<S as SimdOps<f32>>::Reg; MB_REG] = [simd.zero(); MB_REG];
            // acc <- beta*out; beta == 0 skips the load, leaving the zero init untouched
            if beta == 1.0 {
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.load_out(out.add(i + r * lanes));
                }
            } else if beta != 0.0 {
                let bv = simd.splat(beta);
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.mul(simd.load_out(out.add(i + r * lanes)), bv);
                }
            }
            for kk in 0..k {
                let sv = simd.splat(alpha * (*vec.offset(kk as isize * vec_s)).widen());
                let col = mat.offset(kk as isize * mat_cs).add(i);
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = simd.mul_add(simd.load_lhs(col.add(r * lanes)), sv, *a);
                }
            }
            for (r, a) in acc.iter().enumerate() {
                simd.store_out(out.add(i + r * lanes), *a);
            }
            i += mb;
        }

        // Then single-SIMD-register rows, then a sub-lane scalar remainder: the same tiering the
        // float path uses, so a row's tier never depends on the partition
        while i + lanes <= e {
            let mut acc = if beta == 1.0 {
                simd.load_out(out.add(i))
            } else if beta == 0.0 {
                simd.zero()
            } else {
                simd.mul(simd.load_out(out.add(i)), simd.splat(beta))
            };
            for kk in 0..k {
                let sv = simd.splat(alpha * (*vec.offset(kk as isize * vec_s)).widen());
                acc = simd.mul_add(
                    simd.load_lhs(mat.offset(kk as isize * mat_cs).add(i)),
                    sv,
                    acc,
                );
            }
            simd.store_out(out.add(i), acc);
            i += lanes;
        }
        while i < e {
            let op = out.add(i);
            let mut acc: f32 = if beta == 0.0 {
                0.0
            } else if beta == 1.0 {
                (*op).widen()
            } else {
                beta * (*op).widen()
            };
            for kk in 0..k {
                let sv = alpha * (*vec.offset(kk as isize * vec_s)).widen();
                acc += sv * (*mat.offset(kk as isize * mat_cs).add(i)).widen();
            }
            *op = N::narrow(acc);
            i += 1;
        }
    }
}

/// Dot-form mixed gemv over output rows `[s, e)`, for a row-major matrix (rows contiguous over
/// `k`): `out[i] = narrow(alpha*<mat[i,:], vec> + beta*out[i])`, the reduction run in `f32` and
/// rounded to `N` once. Rows are register-blocked in groups of [`DOT_RB`] to keep several
/// independent `f32` FMA chains in flight, the mixed twin of [`dot_rows`]; `vec` is widen-loaded
/// once per depth step and shared by the whole group. Each row is still its own independent
/// `f32`-accumulator reduction, bit-identical to [`dot_contiguous_mixed`] (the form the
/// `< DOT_RB` tail uses), so grouping the rows does not change any row's result, and the split
/// matches the serial sweep
///
/// # Safety
/// As [`axpy_mixed`], with `mat`'s rows contiguous over `k` (`mat_cs == 1`) and `vec` unit-stride
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn dot_rows_mixed<N, S>(
    simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: f32,
    mat: *const N,
    mat_rs: isize,
    vec: *const N,
    beta: f32,
    out: *mut N,
    out_s: isize,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mut i = s;

        // DOT_RB rows at a time, each with its own f32 accumulator, so DOT_RB FMA chains run
        // concurrently over the shared k-sweep. Every row still follows dot_contiguous_mixed's
        // exact order, so interleaving the rows' chains leaves each row's result bit-identical to
        // the per-row tail below
        while i + DOT_RB <= e {
            let rows: [*const N; DOT_RB] =
                core::array::from_fn(|r| mat.offset((i + r) as isize * mat_rs));
            let mut acc: [<S as SimdOps<f32>>::Reg; DOT_RB] = [simd.zero(); DOT_RB];
            let mut kk = 0;
            while kk + lanes <= k {
                // vec is shared: widen-load it once per step and feed every row's chain
                let v = simd.load_lhs(vec.add(kk));
                for r in 0..DOT_RB {
                    acc[r] = simd.mul_add(simd.load_lhs(rows[r].add(kk)), v, acc[r]);
                }
                kk += lanes;
            }
            let mut dots: [f32; DOT_RB] = core::array::from_fn(|r| simd.reduce_sum(acc[r]));
            while kk < k {
                let y = (*vec.add(kk)).widen();
                for r in 0..DOT_RB {
                    dots[r] += (*rows[r].add(kk)).widen() * y;
                }
                kk += 1;
            }
            for (r, dot) in dots.into_iter().enumerate() {
                let op = out.offset((i + r) as isize * out_s);
                let ov = if beta == 0.0 {
                    0.0
                } else if beta == 1.0 {
                    (*op).widen()
                } else {
                    beta * (*op).widen()
                };
                *op = N::narrow(alpha * dot + ov);
            }
            i += DOT_RB;
        }

        // The < DOT_RB tail: the plain per-row form the blocked groups above reproduce exactly
        while i < e {
            let row = mat.offset(i as isize * mat_rs);
            let dot = dot_contiguous_mixed::<N, S>(simd, k, row, vec);
            let op = out.offset(i as isize * out_s);
            let ov = if beta == 0.0 {
                0.0
            } else if beta == 1.0 {
                (*op).widen()
            } else {
                beta * (*op).widen()
            };
            *op = N::narrow(alpha * dot + ov);
            i += 1;
        }
    }
}

/// Horizontal dot of 2 unit-stride length-`k` narrow vectors, widen-loaded and accumulated in
/// `f32`: a SIMD widen `mul_add` sweep, reduced by `reduce_sum` in its fixed lane order, then an
/// ascending scalar `f32` widen tail for the remainder. This is the 1 fixed-order reduction that
/// [`dot_rows_mixed`]'s register-blocked groups and its `< DOT_RB` tail both go through, which is
/// what lets a row round the same way no matter which of the 2 forms computed it. The mixed twin
/// of [`crate::special::dot_contiguous`]
///
/// # Safety
/// `x`/`y` valid for `k` contiguous reads; must run inside `S`'s `vectorize` context
#[cfg(feature = "half")]
#[inline(always)]
unsafe fn dot_contiguous_mixed<N, S>(simd: S, k: usize, x: *const N, y: *const N) -> f32
where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        let lanes = <S as SimdOps<f32>>::LANES;
        let mut acc = simd.zero();
        let mut kk = 0;
        while kk + lanes <= k {
            acc = simd.mul_add(simd.load_lhs(x.add(kk)), simd.load_lhs(y.add(kk)), acc);
            kk += lanes;
        }
        let mut dot = simd.reduce_sum(acc);
        while kk < k {
            dot += (*x.add(kk)).widen() * (*y.add(kk)).widen();
            kk += 1;
        }
        dot
    }
}

/// Fully strided mixed fallback over output rows `[s, e)`, used when neither operand is
/// contiguous: a scalar widen dot accumulated in `f32`, with `beta` applied before the single
/// narrowing round to `N`
///
/// # Safety
/// As [`axpy_mixed`], for arbitrary strides
#[cfg(feature = "half")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn strided_rows_mixed<N, S>(
    _simd: S,
    s: usize,
    e: usize,
    k: usize,
    alpha: f32,
    mat: *const N,
    mat_rs: isize,
    mat_cs: isize,
    vec: *const N,
    vec_s: isize,
    beta: f32,
    out: *mut N,
    out_s: isize,
) where
    N: NarrowFloat,
    S: KernelSimd<N, N, f32, N>,
{
    unsafe {
        for i in s..e {
            let mut dot: f32 = 0.0;
            for kk in 0..k {
                dot += (*mat.offset(i as isize * mat_rs + kk as isize * mat_cs)).widen()
                    * (*vec.offset(kk as isize * vec_s)).widen();
            }
            let op = out.offset(i as isize * out_s);
            let ov = if beta == 0.0 {
                0.0
            } else if beta == 1.0 {
                (*op).widen()
            } else {
                beta * (*op).widen()
            };
            *op = N::narrow(alpha * dot + ov);
        }
    }
}

// Correctness checks for the axpy and dot gemv kernels above
#[cfg(test)]
mod tests {
    use super::{DOT_RB, MB_REG};
    use crate::simd::{ScalarTok, SimdOps};

    /// Builds a per-float-type checker for [`super::axpy_regblocked`]: a row count chosen to hit
    /// all 3 tiers (1 wide `MB_REG*lanes` panel, 1 single-register `lanes`-wide row group, and a
    /// sub-lane scalar remainder), swept over `beta` in `{0, 1, other}` so every accumulator-init
    /// branch runs. Verified against a plain column-major axpy reference within a per-type
    /// tolerance, not bitwise, since the kernel's `a*b + c` is a fused multiply-add
    macro_rules! axpy_regblock_check {
        ($fn:ident, $t:ty, $tol:expr) => {
            fn $fn<S: SimdOps<$t>>(simd: S, label: &str) {
                let lanes = <S as SimdOps<$t>>::LANES;
                // 1 wide panel, 1 single-register row-group, and a sub-lane scalar remainder
                let rows = MB_REG * lanes + lanes + lanes.saturating_sub(1);
                let k = 37usize;

                // u64 index arithmetic so the multipliers can't overflow a 32-bit usize
                let mat: Vec<$t> = (0..rows * k)
                    .map(|i| (((i as u64 * 1103515245 + 12345) % 251) as $t) * 0.008 - 1.0)
                    .collect();
                let vec: Vec<$t> = (0..k)
                    .map(|i| (((i as u64 * 2654435761) % 193) as $t) * 0.01 - 0.9)
                    .collect();
                let out0: Vec<$t> = (0..rows)
                    .map(|i| (((i as u64 * 40503) % 131) as $t) * 0.05 - 3.0)
                    .collect();

                for &(alpha, beta) in &[
                    (1.3 as $t, 0.0 as $t),
                    (0.7 as $t, 1.0 as $t),
                    (1.1 as $t, 2.5 as $t),
                ] {
                    let mut out = out0.clone();
                    // Column-major matrix (mat_cs == rows), unit-stride vector and output
                    unsafe {
                        simd.vectorize(|| {
                            super::axpy_regblocked::<$t, S>(
                                simd,
                                0,
                                rows,
                                k,
                                alpha,
                                mat.as_ptr(),
                                rows as isize,
                                vec.as_ptr(),
                                1,
                                beta,
                                out.as_mut_ptr(),
                            );
                        });
                    }
                    for i in 0..rows {
                        let mut acc = if beta == 0.0 {
                            0.0 as $t
                        } else {
                            beta * out0[i]
                        };
                        for kk in 0..k {
                            acc += mat[kk * rows + i] * (alpha * vec[kk]);
                        }
                        let tol = $tol * (1.0 as $t + acc.abs());
                        assert!(
                            (out[i] - acc).abs() <= tol,
                            "{label} lanes={lanes} beta={beta} row {i}: got {} want {}",
                            out[i],
                            acc
                        );
                    }
                }
            }
        };
    }

    axpy_regblock_check!(check_f32, f32, 1e-4);
    axpy_regblock_check!(check_f64, f64, 1e-10);

    /// The scalar token (`LANES == 1`) runs unconditionally, covering the wide-panel and
    /// single-register tiers on every platform; the runtime-detected SIMD tokens additionally
    /// exercise the sub-lane scalar remainder
    #[test]
    fn axpy_regblocked_spans_all_regimes() {
        check_f32(ScalarTok, "scalar/f32");
        check_f64(ScalarTok, "scalar/f64");

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use crate::simd::{Avx512, Fma};
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                check_f32(Fma, "fma/f32");
                check_f64(Fma, "fma/f64");
            }
            if is_x86_feature_detected!("avx512f") {
                check_f32(Avx512, "avx512/f32");
                check_f64(Avx512, "avx512/f64");
            }
        }

        // Neon is baseline on aarch64, the platform whose gemv dispatch actually uses it, so no
        // runtime probe is needed; its `LANES > 1` also exercises the sub-lane remainder
        #[cfg(target_arch = "aarch64")]
        {
            check_f32(crate::simd::Neon, "neon/f32");
            check_f64(crate::simd::Neon, "neon/f64");
        }
    }

    /// Builds a per-float-type bit-identity checker for [`super::dot_rows`]: its register-blocked
    /// path must match a reference that reduces each row with [`crate::special::dot_contiguous`]
    /// bit for bit, since interleaving independent rows' chains does not change any single row's
    /// accumulator order. Row count spans 2 full `DOT_RB` groups plus a `< DOT_RB` remainder, `k`
    /// spans the SIMD loop plus a sub-lane scalar tail, and `beta` sweeps `{0, 1, other}` so every
    /// epilogue branch runs. Compared with `to_bits`, not a tolerance
    macro_rules! dot_rows_bit_identity_check {
        ($fn:ident, $t:ty) => {
            fn $fn<S: SimdOps<$t>>(simd: S, label: &str) {
                let lanes = <S as SimdOps<$t>>::LANES;
                // 2 full DOT_RB groups plus a sub-group remainder
                let rows = DOT_RB * 2 + 3;
                // A full SIMD vector loop plus a sub-lane scalar tail
                let k = lanes * 5 + 3;

                // Row-major matrix (mat_rs == k, rows contiguous over k), unit-stride vector
                // and unit-stride output: the dot-path layout
                let mat: Vec<$t> = (0..rows * k)
                    .map(|i| (((i as u64 * 1103515245 + 12345) % 251) as $t) * 0.008 - 1.0)
                    .collect();
                let vec: Vec<$t> = (0..k)
                    .map(|i| (((i as u64 * 2654435761) % 193) as $t) * 0.01 - 0.9)
                    .collect();
                let out0: Vec<$t> = (0..rows)
                    .map(|i| (((i as u64 * 40503) % 131) as $t) * 0.05 - 3.0)
                    .collect();

                for &(alpha, beta) in &[
                    (1.3 as $t, 0.0 as $t),
                    (0.7 as $t, 1.0 as $t),
                    (1.1 as $t, 2.5 as $t),
                ] {
                    let mut out = out0.clone();
                    let mut refr = out0.clone();
                    unsafe {
                        simd.vectorize(|| {
                            super::dot_rows::<$t, S>(
                                simd,
                                0,
                                rows,
                                k,
                                alpha,
                                mat.as_ptr(),
                                k as isize,
                                vec.as_ptr(),
                                beta,
                                out.as_mut_ptr(),
                                1,
                            );
                            // Reference: the plain per-row dot_contiguous form the blocked
                            // groups must reproduce exactly. `alpha*dot + ov` here matches
                            // `Float::mul_add` (a plain multiply-add, not a hardware FMA), so any
                            // reordering in the blocked path would flip a bit against this
                            for i in 0..rows {
                                let row = mat.as_ptr().add(i * k);
                                let dot = crate::special::dot_contiguous::<$t, S>(
                                    simd,
                                    k,
                                    row,
                                    vec.as_ptr(),
                                );
                                let ov = if beta == 0.0 as $t {
                                    0.0 as $t
                                } else if beta == 1.0 as $t {
                                    refr[i]
                                } else {
                                    beta * refr[i]
                                };
                                refr[i] = alpha * dot + ov;
                            }
                        });
                    }
                    for i in 0..rows {
                        assert_eq!(
                            out[i].to_bits(),
                            refr[i].to_bits(),
                            "{label} lanes={lanes} beta={beta} row {i}: blocked {} vs ref {}",
                            out[i],
                            refr[i]
                        );
                    }
                }
            }
        };
    }

    dot_rows_bit_identity_check!(dot_check_f32, f32);
    dot_rows_bit_identity_check!(dot_check_f64, f64);

    /// The scalar token (`LANES == 1`) runs unconditionally, covering the register-blocked
    /// groups and the remainder on every platform; the runtime-detected SIMD tokens additionally
    /// exercise the shared SIMD `mul_add` sweep and the sub-lane scalar tail
    #[test]
    fn dot_rows_bit_identical() {
        dot_check_f32(ScalarTok, "scalar/f32");
        dot_check_f64(ScalarTok, "scalar/f64");

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use crate::simd::{Avx512, Fma};
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                dot_check_f32(Fma, "fma/f32");
                dot_check_f64(Fma, "fma/f64");
            }
            if is_x86_feature_detected!("avx512f") {
                dot_check_f32(Avx512, "avx512/f32");
                dot_check_f64(Avx512, "avx512/f64");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            dot_check_f32(crate::simd::Neon, "neon/f32");
            dot_check_f64(crate::simd::Neon, "neon/f64");
        }
    }
}
