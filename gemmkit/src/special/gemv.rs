//! gemv: matrix*vector (`n == 1` or `m == 1`)
//!
//! gemv is memory-bound, so the arithmetic is trivial; the whole game is minimizing
//! DRAM traffic. Both `m == 1` and `n == 1` reduce to one core routine by viewing the
//! matrix (transposed for `m == 1`) as `rows x k` times a `k`-vector. Correct for every
//! layout; vectorized for the contiguous ones
//!
//! The column-major (axpy) shape has 2 bit-identical strategies that differ only in
//! memory traffic (both do the same ascending-`k` fused accumulation per output element):
//!
//! * **Register-blocked output**: block the rows into panels a few SIMD registers wide
//!   and sweep all `k` columns per panel, holding the output panel in registers. The
//!   matrix and the output are each read exactly once. Chosen when the output can't stay
//!   resident in the last-level cache across the `k`-sweep, so the plain form's per-column
//!   re-reads of the output would otherwise hit DRAM
//! * **Plain axpy**: column-outer, re-reading the output each column. Fine when the
//!   output stays cache-resident, and its perfectly contiguous single-stream matrix read
//!   is what large `k` prefers
//!
//! Each output element is computed in a single pass over `k` by one worker, so the result
//! is **reproducible** and bit-identical regardless of the worker count: the output-row
//! partitioning ([`run_typed`]) never splits an element's reduction

use crate::kernel::FloatGemm;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
use crate::simd::SimdOps;

/// Output row panels register-blocked at once, in SIMD registers: `MB_REG` accumulator
/// registers plus one broadcast RHS register stay well within the vector file on every
/// ISA, while giving the matrix read a wide contiguous burst per column. Also the row
/// partitioning grain, so worker boundaries land on panel edges and the SIMD/scalar
/// split of every row is partition-independent (so serial == parallel bit-for-bit)
const MB_REG: usize = 8;

/// Dispatch a gemv shape to the core routine, partitioning the output rows across up to
/// `par`-many workers. gemv is memory-bandwidth-bound, so the worker count is capped by a
/// bandwidth model ([`Parallelism::resolve_bandwidth`]), not the compute ramp: past the
/// few cores that saturate DRAM, more workers stop helping. Each output row is computed by
/// one worker over the full `k`, so the split adds no cross-thread reduction and the result
/// stays bit-identical to the serial run
///
/// The zero-cost [`Identity`] forwarder over [`run_typed_epi`]: with `E = Identity` the fused
/// pass const-folds away, so this route is byte-for-byte unchanged and the public signature is
/// preserved for every existing caller
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must
/// not alias `a`/`b`. Must be called only when the CPU supports `S`'s features
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
    // SAFETY: forwarded to `run_typed_epi` with the zero-cost `Identity` epilogue: the exact
    // bits this route always stored (`E::IS_IDENTITY` folds the final sweep away)
    unsafe {
        run_typed_epi::<T, S, Identity>(
            simd, m, k, n, par, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity,
        )
    }
}

/// gemv applying the fused [`Epilogue`] `E` to its output. gemv is dispatched **before**
/// orientation normalization, so `epi`'s coordinates are the **user** frame (the caller passes
/// the unflipped epilogue). The `n == 1` branch's output element `i` is `C[i, 0]` (`swap_rc =
/// false`); the `m == 1` branch views `C^T`, so its element `i` is `C[0, i]` (`swap_rc = true`)
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
            // C (mx1) = beta*C + alpha*A*b, A = mxk, b = k-vector. Output element `i` is
            // `C[i, 0]`, so the epilogue reads coordinate `(i, 0)` (`swap_rc = false`)
            core_epi::<T, S, E>(
                simd, m, k, par, alpha, a, rsa, csa, b, rsb, beta, c, rsc, false, epi,
            );
        } else {
            // C (1xn) = beta*C + alpha*a*B. View B^T (nxk) times a (k-vector):
            // B^T[j,k] = B[k,j] -> row stride csb, col stride rsb; out stride csc. The transposed
            // view makes output element `i` map to `C[0, i]`, so the epilogue reads coordinate
            // `(0, i)` (`swap_rc = true`)
            core_epi::<T, S, E>(
                simd, n, k, par, alpha, b, csb, rsb, a, csa, beta, c, csc, true, epi,
            );
        }
    }
}

/// `out[i] = beta*out[i] + alpha * sum_k(mat[i,k]*vec[k])` for `i in 0..rows`, split across
/// bandwidth-capped workers over disjoint output-row panels, then a fused [`Epilogue`] `E`
/// applied as a final in-place sweep over each worker's own range (see the pass below).
/// `swap_rc` picks the epilogue coordinate for output element `i`: `(i, 0)` when `false`
/// (the `n == 1` shape), `(0, i)` when `true` (the transposed `m == 1` view)
///
/// # Safety
/// `mat` valid for the `rows x k` region at `mat_rs`/`mat_cs`; `vec` valid for `k` reads at
/// `vec_s`; `out` valid for `rows` writes at `out_s`, and for `rows` reads too when `beta !=
/// 0`; `epi`'s interior pointers valid for the problem's `m`/`n`. Must be called only when the
/// CPU supports `S`'s features
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
    // `E: Copy` (an `Epilogue` supertrait): copy it out of the borrow so each `move` worker
    // closure captures it by value
    let epi = *epi;
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let sizeof = core::mem::size_of::<T>();

        // Which shape, decided once (same for every worker, so every worker runs the same
        // per-row arithmetic and the row partition is bit-identical to the serial sweep)
        let axpy = mat_rs == 1 && out_s == 1;
        let output_block = axpy && output_register_block(rows, sizeof, k);
        let dot = !axpy && mat_cs == 1 && vec_s == 1;

        // Bandwidth-capped worker count. `bytes_touched` is the minimum traffic (matrix
        // read once, vector once, output written once); `rows` bounds the partition
        let bytes_touched = (rows.saturating_mul(k) + k + rows).saturating_mul(sizeof);
        let n_threads = par.resolve_bandwidth(bytes_touched, rows);

        // Partition grain: register-blocked panels (`MB_REG*lanes`) for the axpy path so
        // worker boundaries fall on panel edges; single SIMD rows otherwise. Either way a
        // multiple of `lanes`, so each row's SIMD/scalar classification is the same in
        // every partition and the split reproduces the serial result bit-for-bit
        let block = if output_block { MB_REG * lanes } else { lanes }.max(1);
        let n_blocks = rows.div_ceil(block);

        let mat = Ptr(mat as *mut T);
        let vec = Ptr(vec as *mut T);
        let out = Ptr(out);

        let body = move |row_start: usize, row_end: usize| {
            let (mat, vec, out, epi) = (mat, vec, out, epi);
            let mat = mat.0 as *const T;
            let vec = vec.0 as *const T;
            let out = out.0;
            // Run the sweep inside the ISA's `#[target_feature]` context so the SIMD
            // primitives fold into feature-enabled codegen (as the driver does per tile)
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
                // Fused epilogue: a final in-place sweep over this worker's own output range
                // gemv fuses this way, not by threading `epi` into the 4 strategy kernels,
                // because the output is one vector, tiny next to the matrix read that dominates
                // this memory-bound path: one extra pass over `[row_start, row_end)` is
                // negligible and keeps all 4 strategies byte-identical to the non-fused build
                // And because the strategy first stores exactly the bits plain gemv would store,
                // mapping the re-read value is bitwise-identical to gemm-then-map by construction
                // Fires exactly once per element (each range is owned by one worker, swept after
                // its full k-reduction). `E::IS_IDENTITY` const-folds the whole sweep away, so the
                // non-fused instantiation is zero-cost. Kept inside `vectorize` so the epilogue
                // runs in the token context (its `apply` uses scalar ops only)
                if !E::IS_IDENTITY {
                    for i in row_start..row_end {
                        let op = out.offset(i as isize * out_s);
                        let (r, c) = if swap_rc { (0, i) } else { (i, 0) };
                        *op = epi.apply(*op, r, c);
                    }
                }
            });
        };

        if n_threads <= 1 {
            body(0, rows);
            return;
        }

        // Output-partitioned parallel sweep: workers pull disjoint panel ranges from a
        // shared cursor and run the per-row body on their rows only. No reduction across
        // workers, so no barrier and no perturbation of the summation order
        let cur = JobCursor::new(n_blocks, parallel::job_grain(n_blocks, n_threads));
        parallel::for_each_worker(n_threads, |_tid| {
            while let Some((bs, be)) = cur.next_chunk() {
                let row_start = bs * block;
                let row_end = core::cmp::min(be * block, rows);
                body(row_start, row_end);
            }
        });
    }
}

/// Register-block the output for an axpy-shape gemv when *both* hold: the output
/// (`rows*sizeof` bytes) is large enough that the plain column-outer form's per-column re-read of
/// the output spills toward DRAM (the size gate is a fraction of L3, see
/// [`crate::cache::gemv_regblock_engage_bytes`]); and `k` is small enough (`<= k_stream_max`) that
/// the register-blocked form's `k` in-place matrix column-streams (one per depth step) stay within
/// the hardware prefetcher's window. When the output stays cache-resident the plain form's re-reads
/// are cheap and its single contiguous matrix stream wins; when `k` is large its many streams thrash
/// the prefetcher
#[inline]
fn output_register_block(rows: usize, sizeof: usize, k: usize) -> bool {
    k <= crate::tuning::k_stream_max()
        && rows.saturating_mul(sizeof) > crate::cache::gemv_regblock_engage_bytes()
}

/// Register-blocked axpy over output rows `[s, e)`: the output panel is held in SIMD
/// registers across all `k` columns, so the output is read/written once and the
/// column-major matrix is read once. `beta` is folded into the accumulator init (`beta == 0`
/// never reads the output). Bit-identical to [`axpy_plain`]: same ascending-`k` fused
/// accumulation per element, same SIMD/scalar split by row (panels of `MB_REG*lanes`, then
/// single SIMD rows, then a scalar remainder), so the 2 strategies are interchangeable
///
/// # Safety
/// `mat`/`vec` valid for the region implied by the strides; `out` valid for `[s, e)` writes
/// and, when `beta != 0`, reads. Run inside `S`'s `vectorize` context
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

        // Wide panels: `MB_REG` accumulator registers, live across the whole k-sweep
        while i + mb <= e {
            let mut acc = [simd.zero(); MB_REG];
            // acc <- beta*out (beta == 0 leaves the zero init and never reads out)
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

        // Single-register SIMD rows, then the sub-lane scalar remainder: the same
        // per-row classification the plain path uses, so both round identically
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
/// mat[i,k])`, re-reading the output panel each column (cache-resident regime). `beta` is folded
/// via a per-range pre-scale so the range is self-contained (workers scale only their own
/// rows). Same fused accumulation and per-row SIMD/scalar split as [`axpy_regblocked`]
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
        // Pre-scale this range's output by beta (beta == 0 overwrites without reading)
        for i in s..e {
            let op = out.add(i);
            if beta == T::ZERO {
                *op = T::ZERO;
            } else if beta != T::ONE {
                *op = beta * *op;
            }
        }
        // Fold KB columns per output load/store: the output panel is touched once per KB-group
        // instead of once per column, cutting its cache traffic (the axpy form's main cost once
        // the matrix read is DRAM-bound) while keeping only KB concurrent matrix column-streams
        // The fused steps run in ascending `k`, so this is bit-identical to the one-column form
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
        // Remainder columns (`k % KB`): one column at a time
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

/// Output rows register-blocked in the dot path: `DOT_RB` rows are reduced side by side,
/// each keeping its own accumulator, so `DOT_RB` independent FMA chains overlap across the
/// shared k-sweep. A single row's reduction is one dependent `mul_add` chain, latency-bound
/// (~4-cycle FMA latency on Zen5, so ~1 FMA per 4 cycles) rather than throughput-bound (2
/// FMAs/cycle); running several rows' chains at once fills that latency shadow, and the
/// vector `vec` is loaded once per depth step and shared by the whole group. Chosen at 4 by
/// measurement, not by the latency*throughput product (which wants ~8): 4 chains already
/// recover most of the stall, and each row adds a concurrent matrix read-stream, so 8
/// over-subscribes the hardware prefetcher and grows the per-group load footprint. 4 wins on
/// both cache-resident shapes and the DRAM-bound guard (where the extra memory-level
/// parallelism from a few independent streams even pushes a single core past its naive
/// single-stream rate); 8 pulls ahead only on the few-long-rows shape, where the stream count
/// is small. Not a partition grain (the dot path partitions on single SIMD rows), so its
/// value is free of the serial-vs-parallel reproducibility constraint that pins [`MB_REG`]
const DOT_RB: usize = 4;

/// Dot-form over output rows `[s, e)` (row-major matrix): `out[i] = alpha*<mat[i,:], vec> +
/// beta*out[i]`, one pass (matrix row read once, output once, vector kept in L1). `beta` folded
/// into the per-row epilogue. Rows are register-blocked in groups of [`DOT_RB`] to keep several
/// independent FMA chains in flight (the per-row reduction is latency-bound otherwise); the
/// `< DOT_RB` tail falls through to the plain per-row form
///
/// # Safety
/// As [`axpy_regblocked`], with `mat` rows contiguous over `k` and `vec` unit-stride
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

        // Register-blocked groups of DOT_RB rows: DOT_RB independent SIMD accumulators plus
        // DOT_RB independent scalar-tail accumulators, so DOT_RB FMA chains overlap across the
        // shared k-sweep. Every row runs exactly `dot_contiguous`'s order (single accumulator,
        // ascending k in `lanes` steps with `mul_add(row, vec, acc)`, `reduce_sum`, then the
        // ascending scalar tail), so interleaving the chains leaves each row bit-identical to
        // the per-row `dot_contiguous` used by the tail below and by the small_mn edge cell
        while i + DOT_RB <= e {
            let rows: [*const T; DOT_RB] =
                core::array::from_fn(|r| mat.offset((i + r) as isize * mat_rs));
            let mut acc = [simd.zero(); DOT_RB];
            let mut kk = 0;
            while kk + lanes <= k {
                // Load the shared vector once, feed every row's chain
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

        // Remaining `< DOT_RB` rows: the plain single-accumulator per-row form (unchanged),
        // which is exactly the per-row arithmetic the blocked groups above reproduce
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

/// Fully general strided fallback over output rows `[s, e)` (neither operand contiguous):
/// scalar dot per row, `beta` folded into the epilogue
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

#[cfg(test)]
mod tests {
    use super::{DOT_RB, MB_REG};
    use crate::simd::{ScalarTok, SimdOps};

    /// Generate a per-float-type checker for [`super::axpy_regblocked`]. It picks a row count
    /// that spans all 3 regimes: a wide `MB_REG*lanes` register-blocked panel, one
    /// single-register `lanes`-wide tail, and a sub-lane scalar remainder, and sweeps
    /// `beta` in `{0, 1, other}` so every accumulator-init branch runs. The result is compared
    /// against a straightforward column-major axpy reference at a per-type tolerance (the
    /// kernel fuses `a*b + c`, so the match is not bitwise)
    macro_rules! axpy_regblock_check {
        ($fn:ident, $t:ty, $tol:expr) => {
            fn $fn<S: SimdOps<$t>>(simd: S, label: &str) {
                let lanes = <S as SimdOps<$t>>::LANES;
                // wide panel + one single-register row-group + a sub-lane scalar remainder
                let rows = MB_REG * lanes + lanes + lanes.saturating_sub(1);
                let k = 37usize;

                // Index arithmetic in u64 so the multipliers don't overflow a 32-bit usize
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
                    // Column-major matrix (`mat_cs == rows`), unit-stride vector and output
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

    /// The scalar token (`LANES == 1`) always runs, covering the wide-panel and
    /// single-register regimes platform-independently; the SIMD tokens (guarded by runtime
    /// detection) additionally exercise the sub-lane scalar remainder
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

        // Neon is baseline on aarch64 (the platform whose gemv dispatch uses it), so it
        // needs no runtime probe; its `LANES > 1` also exercises the sub-lane remainder
        #[cfg(target_arch = "aarch64")]
        {
            check_f32(crate::simd::Neon, "neon/f32");
            check_f64(crate::simd::Neon, "neon/f64");
        }
    }

    /// Generate a per-float-type bit-identity checker for [`super::dot_rows`]. The
    /// register-blocked dot path must produce output *bitwise identical* to a reference loop
    /// that reduces each row with [`crate::special::dot_contiguous`] (the shared fixed-order
    /// reduction gemv and small_mn depend on): interleaving independent rows' chains keeps
    /// each row's single-accumulator ascending-`k` order intact, so the bits must match. The
    /// row count spans 2 full `DOT_RB` groups plus a `< DOT_RB` remainder, `k` spans the SIMD
    /// vector loop plus a sub-lane scalar tail, and `beta` sweeps `{0, 1, other}` so every
    /// epilogue branch runs. Compared on raw bits (`to_bits`), not a tolerance
    macro_rules! dot_rows_bit_identity_check {
        ($fn:ident, $t:ty) => {
            fn $fn<S: SimdOps<$t>>(simd: S, label: &str) {
                let lanes = <S as SimdOps<$t>>::LANES;
                // 2 full DOT_RB groups + a sub-group remainder
                let rows = DOT_RB * 2 + 3;
                // full SIMD vector loop + a sub-lane scalar tail
                let k = lanes * 5 + 3;

                // Row-major matrix (`mat_rs == k`, rows contiguous over `k`), unit-stride
                // vector and unit-stride output (the dot-path layout)
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
                            // Reference: the plain per-row `dot_contiguous` form the blocked
                            // groups must reproduce bit-for-bit. The epilogue uses `alpha*dot +
                            // ov`, matching `Float::mul_add` (plain mul-add, not a hardware
                            // FMA), so any reordering in the blocked path would flip a bit here
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

    /// The scalar token (`LANES == 1`) always runs, covering the register-blocked groups and
    /// remainder platform-independently; the SIMD tokens (guarded by runtime detection)
    /// additionally exercise the shared SIMD `mul_add` sweep and the sub-lane scalar tail
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
