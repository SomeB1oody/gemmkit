//! gemv: matrix·vector (`n == 1` or `m == 1`).
//!
//! gemv is memory-bound, so the arithmetic is trivial; the whole game is minimizing
//! DRAM traffic. Both `m == 1` and `n == 1` reduce to one core routine by viewing the
//! matrix (transposed for `m == 1`) as `rows × k` times a `k`-vector. Correct for every
//! layout; vectorized for the contiguous ones.
//!
//! The column-major (axpy) shape has two bit-identical strategies that differ only in
//! memory traffic (both do the same ascending-`k` fused accumulation per output element):
//!
//! * **Register-blocked output** — block the rows into panels a few SIMD registers wide
//!   and sweep all `k` columns per panel, holding the output panel in registers. The
//!   matrix and the output are each read exactly once. Chosen when the output can't stay
//!   resident in the last-level cache across the `k`-sweep, so the plain form's per-column
//!   re-reads of the output would otherwise hit DRAM.
//! * **Plain axpy** — column-outer, re-reading the output each column. Fine when the
//!   output stays cache-resident, and its perfectly contiguous single-stream matrix read
//!   is what large `k` prefers.
//!
//! Each output element is computed in a single pass over `k` by one worker, so the result
//! is **reproducible** and bit-identical regardless of the worker count: the output-row
//! partitioning ([`run_typed`]) never splits an element's reduction.

use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::scalar::Float;
use crate::simd::SimdOps;

/// Output row panels register-blocked at once, in SIMD registers: `MB_REG` accumulator
/// registers plus one broadcast RHS register stay well within the vector file on every
/// ISA, while giving the matrix read a wide contiguous burst per column. Also the row
/// partitioning grain, so worker boundaries land on panel edges and the SIMD/scalar
/// split of every row is partition-independent (⇒ serial == parallel bit-for-bit).
const MB_REG: usize = 8;

/// Dispatch a gemv shape to the core routine, partitioning the output rows across up to
/// `par`-many workers. gemv is memory-bandwidth-bound, so the worker count is capped by a
/// bandwidth model ([`Parallelism::resolve_bandwidth`]), not the compute ramp: past the
/// few cores that saturate DRAM, more workers stop helping. Each output row is computed by
/// one worker over the full `k`, so the split adds no cross-thread reduction and the result
/// stays bit-identical to the serial run.
///
/// # Safety
/// Pointers must be valid for the regions implied by the strides/sizes; `c` must
/// not alias `a`/`b`. Must be called only when the CPU supports `S`'s features.
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
    unsafe {
        if n == 1 {
            // C (m×1) = beta·C + alpha·A·b, A = m×k, b = k-vector.
            core::<T, S>(simd, m, k, par, alpha, a, rsa, csa, b, rsb, beta, c, rsc);
        } else {
            // C (1×n) = beta·C + alpha·a·B. View Bᵀ (n×k) times a (k-vector):
            // Bᵀ[j,k] = B[k,j] → row stride csb, col stride rsb; out stride csc.
            core::<T, S>(simd, n, k, par, alpha, b, csb, rsb, a, csa, beta, c, csc);
        }
    }
}

/// `out[i] = beta·out[i] + alpha · Σ_k mat[i,k]·vec[k]` for `i in 0..rows`, split across
/// bandwidth-capped workers over disjoint output-row panels.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn core<T, S>(
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
) where
    T: Float<Acc = T>,
    S: SimdOps<T>,
{
    unsafe {
        let lanes = <S as SimdOps<T>>::LANES;
        let sizeof = core::mem::size_of::<T>();

        // Which shape, decided once (same for every worker ⇒ every worker runs the same
        // per-row arithmetic, so the row partition is bit-identical to the serial sweep).
        let axpy = mat_rs == 1 && out_s == 1;
        let output_block = axpy && output_register_block(rows, sizeof, k);
        let dot = !axpy && mat_cs == 1 && vec_s == 1;

        // Bandwidth-capped worker count. `bytes_touched` is the minimum traffic (matrix
        // read once, vector once, output written once); `rows` bounds the partition.
        let bytes_touched = (rows.saturating_mul(k) + k + rows).saturating_mul(sizeof);
        let n_threads = par.resolve_bandwidth(bytes_touched, rows);

        // Partition grain: register-blocked panels (`MB_REG·lanes`) for the axpy path so
        // worker boundaries fall on panel edges; single SIMD rows otherwise. Either way a
        // multiple of `lanes`, so each row's SIMD/scalar classification is the same in
        // every partition and the split reproduces the serial result bit-for-bit.
        let block = if output_block { MB_REG * lanes } else { lanes }.max(1);
        let n_blocks = rows.div_ceil(block);

        let mat = Ptr(mat as *mut T);
        let vec = Ptr(vec as *mut T);
        let out = Ptr(out);

        let body = move |row_start: usize, row_end: usize| {
            let (mat, vec, out) = (mat, vec, out);
            let mat = mat.0 as *const T;
            let vec = vec.0 as *const T;
            let out = out.0;
            // Run the sweep inside the ISA's `#[target_feature]` context so the SIMD
            // primitives fold into feature-enabled codegen (as the driver does per tile).
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
            });
        };

        if n_threads <= 1 {
            body(0, rows);
            return;
        }

        // Output-partitioned parallel sweep: workers pull disjoint panel ranges from a
        // shared cursor and run the per-row body on their rows only. No reduction across
        // workers, so no barrier and no perturbation of the summation order.
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
/// (`rows·sizeof` bytes) spills the L2, so the plain column-outer form's per-column re-read
/// of the output leaves the core's private cache; and `k` is small enough that the
/// register-blocked form's `k` concurrent matrix column-streams (one per depth step it reads
/// in place) stay within the hardware prefetcher's tracking window. When the output fits L2
/// the plain form's re-reads are cheap and its single contiguous matrix stream wins; when `k`
/// is large the register-blocked form's many streams thrash the prefetcher and the plain form
/// wins. The `K_STREAM_MAX = 32` bound is calibrated on Zen5: at `k ≤ 16` register-blocking
/// runs ~20% faster than the plain form on a DRAM-resident output, is a wash near `k = 32`,
/// and regresses by `k ≈ 48` as the streams exceed the prefetcher.
const K_STREAM_MAX: usize = 32;

#[inline]
fn output_register_block(rows: usize, sizeof: usize, k: usize) -> bool {
    k <= K_STREAM_MAX
        && rows.saturating_mul(sizeof) > crate::cache::topology().l2.effective_bytes().max(1)
}

/// Register-blocked axpy over output rows `[s, e)`: the output panel is held in SIMD
/// registers across all `k` columns, so the output is read/written once and the
/// column-major matrix is read once. `β` is folded into the accumulator init (`β == 0`
/// never reads the output). Bit-identical to [`axpy_plain`] — same ascending-`k` fused
/// accumulation per element, same SIMD/scalar split by row (panels of `MB_REG·lanes`, then
/// single SIMD rows, then a scalar remainder), so the two strategies are interchangeable.
///
/// # Safety
/// `mat`/`vec` valid for the region implied by the strides; `out` valid for `[s, e)` writes
/// and, when `β != 0`, reads. Run inside `S`'s `vectorize` context.
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

        // Wide panels: `MB_REG` accumulator registers, live across the whole k-sweep.
        while i + mb <= e {
            let mut acc = [simd.zero(); MB_REG];
            // acc <- β·out (β == 0 leaves the zero init and never reads out).
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

        // Single-register SIMD rows, then the sub-lane scalar remainder — the same
        // per-row classification the plain path uses, so both round identically.
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

/// Plain column-outer axpy over output rows `[s, e)`: `out[i] = β·out[i] + Σ_k (α·vec[k])·
/// mat[i,k]`, re-reading the output panel each column (cache-resident regime). `β` is folded
/// via a per-range pre-scale so the range is self-contained (workers scale only their own
/// rows). Same fused accumulation and per-row SIMD/scalar split as [`axpy_regblocked`].
///
/// # Safety
/// As [`axpy_regblocked`].
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
        // Pre-scale this range's output by β (β == 0 overwrites without reading).
        for i in s..e {
            let op = out.add(i);
            if beta == T::ZERO {
                *op = T::ZERO;
            } else if beta != T::ONE {
                *op = beta * *op;
            }
        }
        for kk in 0..k {
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
        }
    }
}

/// Dot-form over output rows `[s, e)` (row-major matrix): `out[i] = α·⟨mat[i,:], vec⟩ +
/// β·out[i]`, one pass (matrix row read once, output once, vector kept in L1). `β` folded
/// into the per-row epilogue.
///
/// # Safety
/// As [`axpy_regblocked`], with `mat` rows contiguous over `k` and `vec` unit-stride.
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
        for i in s..e {
            let row = mat.offset(i as isize * mat_rs);
            let mut acc = simd.zero();
            let mut kk = 0;
            while kk + lanes <= k {
                acc = simd.mul_add(simd.loadu(row.add(kk)), simd.loadu(vec.add(kk)), acc);
                kk += lanes;
            }
            let mut dot = simd.reduce_sum(acc);
            while kk < k {
                dot = (*row.add(kk)).mul_add(*vec.add(kk), dot);
                kk += 1;
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

/// Fully general strided fallback over output rows `[s, e)` (neither operand contiguous):
/// scalar dot per row, `β` folded into the epilogue.
///
/// # Safety
/// As [`axpy_regblocked`], for arbitrary strides.
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
