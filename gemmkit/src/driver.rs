//! The generic GEMM driver (layer L5): one 5-loop nest, fully generic over the
//! [`KernelFamily`] and the ISA token. It never names a concrete element type, a
//! concrete ISA, or a macro, so adding a family or an ISA never touches this file.
//! This is the open/closed property the architecture promises. A test proves it by
//! declaring a 2nd family and driving it through this same `run`
//!
//! Loop structure (BLIS order): `jc` (columns, L3-sized blocks), then `pc` (depth,
//! not parallelized, the panels always run in the same sequential order). Inside
//! each `pc` panel, a flat 1-D job list covers every `(ic row-block x jt
//! column-tile)` pair. Workers pull contiguous chunks from a shared cursor on
//! demand, so a faster worker drains more chunks than a slower one. `beta` applies
//! only on the 1st depth panel. Later panels accumulate (`beta = 1`) into the same
//! `C`
//!
//! Reproducibility: `kc` and `nc` come from the problem size and the cache
//! topology alone, never the thread count. The `pc` panels run in the same fixed
//! order regardless of the split, so the partial-sum sequence into `C` does not
//! depend on it. `mc` may shrink with the worker count, but `mc` always stays an
//! `mr` multiple. The microtile set, every `mr`-aligned row offset plus the one
//! `m`-tail tile, is the same under any split. Each tile's accumulation order then
//! depends only on `kc` and the fixed `pc` order. A fixed input and config
//! therefore always produce the same output under any [`Parallelism`]

use core::mem::MaybeUninit;

use crate::cache;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::kernel::{AlphaStatus, BetaStatus, KernelFamily, SCRATCH_LEN};
use crate::parallel::{self, Parallelism, Ptr};
use crate::scalar::Scalar;
use crate::simd::{KernelSimd, SimdOps};
use crate::tuning;
use crate::workspace::{Regions, Workspace};

/// Precomputed `alpha` state so a microkernel call branches instead of comparing
/// floats. `alpha == 0` is intercepted upstream as a beta-only scale of `C`, so
/// only `One` or `Other` reach here. [`crate::special::small_k`] and the deep-k
/// narrowing sweep in `dispatch::mixed` also call this directly
#[inline]
pub(crate) fn alpha_status<T: crate::scalar::Scalar>(a: T) -> AlphaStatus {
    if a == T::ONE {
        AlphaStatus::One
    } else {
        AlphaStatus::Other
    }
}

/// Precomputed `beta` state for a microkernel call. See [`alpha_status`].
/// [`crate::special::small_k`] and `dispatch::mixed` also call this directly
#[inline]
pub(crate) fn beta_status<T: crate::scalar::Scalar>(b: T) -> BetaStatus {
    if b == T::ZERO {
        BetaStatus::Zero
    } else if b == T::ONE {
        BetaStatus::One
    } else {
        BetaStatus::Other
    }
}

/// Whether to pack the LHS macro-block for one row-block/column-tile call
///
/// A non-unit row stride (`rsa != 1`) means A is not column-major, so its rows
/// are not contiguous, and packing is forced. A short tail row-block also forces
/// packing, since the microkernel always reads a contiguous `mr`-row vector.
/// Otherwise packing runs only when the caller has already set `want_pack`
#[inline]
fn do_pack_lhs(rsa: isize, mc_eff: usize, mr: usize, want_pack: bool) -> bool {
    rsa != 1 || !mc_eff.is_multiple_of(mr) || want_pack
}

/// Prefetch one output microtile (`mr_eff x nr_eff` at element strides `rsc`/`csc`) into L1
/// with a T0 hint. Issue it just ahead of the microkernel call that reads and writes the tile
///
/// The prefetch fires only when the call's working set exceeds the LLC (see
/// `cache::prefetch_ws_bytes`), where the tile would otherwise stream from DRAM
///
/// It walks whole cache lines along the tile's unit-stride dimension. A tile strided in both
/// dimensions has no contiguous lines to fetch, so it is skipped. A line that only straddles
/// the tile's tail may be left unfetched. This is only a hint, never a correctness concern,
/// and prefetch never faults, so running past the matrix edge inside the last line is safe
///
/// x86_64 only (`prefetcht0` is baseline SSE, no feature gate needed). A no-op elsewhere, so
/// aarch64 and wasm behavior is unchanged
#[inline(always)]
fn prefetch_c_tile<T>(c: *const T, rsc: isize, csc: isize, mr_eff: usize, nr_eff: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let esz = core::mem::size_of::<T>();
        if rsc == 1 {
            // Column-major tile: each of the nr_eff columns is mr_eff * esz contiguous bytes
            let col_bytes = mr_eff * esz;
            for j in 0..nr_eff {
                let col = c.offset(j as isize * csc) as *const i8;
                let mut off = 0;
                while off < col_bytes {
                    _mm_prefetch::<_MM_HINT_T0>(col.add(off));
                    off += 64;
                }
            }
        } else if csc == 1 {
            // Row-major tile: each of the mr_eff rows is nr_eff * esz contiguous bytes
            let row_bytes = nr_eff * esz;
            for i in 0..mr_eff {
                let row = c.offset(i as isize * rsc) as *const i8;
                let mut off = 0;
                while off < row_bytes {
                    _mm_prefetch::<_MM_HINT_T0>(row.add(off));
                    off += 64;
                }
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (c, rsc, csc, mr_eff, nr_eff);
    }
}

/// Run a GEMM with the given family, ISA token, and microtile geometry
///
/// The dispatch layer establishes these preconditions before this function runs:
/// `m, n, k > 0`, `alpha != 0`, and the problem is already orientation-normalized.
/// The driver itself has no shape restriction. An `m == 1` or `n == 1` gemv shape
/// still computes correctly through the ordinary edge-tile handling. The dispatch
/// layer normally routes those shapes to the faster, dedicated gemv path instead
///
/// # Safety
///
/// All pointers must be valid for the regions implied by the strides and sizes.
/// `C` must not alias `A` or `B`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run<Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: Fam::Acc,
    a: *const Fam::Lhs,
    rsa: isize,
    csa: isize,
    b: *const Fam::Rhs,
    rsb: isize,
    csb: isize,
    beta: Fam::Acc,
    c: *mut Fam::Out,
    rsc: isize,
    csc: isize,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
{
    // SAFETY: forwards to `run_inner` with no prepacked RHS and the zero-cost `Identity`
    // epilogue. The caller's preconditions above cover the rest
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, Identity>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, par, ws, None,
            &Identity,
        )
    }
}

/// Run a GEMM that applies the fused [`Epilogue`] `E` to each output element as it is stored
///
/// This applies the epilogue during the store instead of computing the raw product and
/// mapping it in a 2nd pass. [`run`] is this same call with `E = Identity`. The engine
/// underneath (blocking, packing, scheduling) never depends on `E`, so a fused call computes
/// the same pre-epilogue value that a plain GEMM would. The result equals a plain GEMM
/// followed by a scalar map
///
/// # Safety
///
/// As [`run`], plus `epi`'s interior pointers must be valid for the problem's `m`
/// and `n`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_epilogue<Fam, S, E, const MR_REG: usize, const NR: usize>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: Fam::Acc,
    a: *const Fam::Lhs,
    rsa: isize,
    csa: isize,
    b: *const Fam::Rhs,
    rsb: isize,
    csb: isize,
    beta: Fam::Acc,
    c: *mut Fam::Out,
    rsc: isize,
    csc: isize,
    epi: &E,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    E: Epilogue<Fam>,
{
    // SAFETY: forwards to `run_inner` with no prepacked RHS and the caller's epilogue
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, E>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, par, ws, None, epi,
        )
    }
}

/// Run a GEMM whose RHS was already packed by [`pack_rhs_full`]
///
/// `packed_b` is the buffer base. It carries no RHS strides, because the panel layout is
/// already baked in by the pack. The buffer is read-only and shared immutably across every
/// worker, so unlike the driver's own per-call B-pack it needs no write-before-read barrier
///
/// `kc` and `nc` are the blocking sizes the buffer was packed for. Both are used verbatim, so
/// the panel addresses this call computes always line up with the buffer contents. `mc`, the A
/// row-block size, is still derived fresh from the real `m`
///
/// # Safety
///
/// As [`run`], plus `packed_b` must come from [`pack_rhs_full`] for this same
/// `(k, n, kc, nc, nr = NR)`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_packed_rhs<Fam, S, const MR_REG: usize, const NR: usize>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: Fam::Acc,
    a: *const Fam::Lhs,
    rsa: isize,
    csa: isize,
    packed_b: *const Fam::Rhs,
    kc: usize,
    nc: usize,
    beta: Fam::Acc,
    c: *mut Fam::Out,
    rsc: isize,
    csc: isize,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
{
    // SAFETY: forwards the prepacked buffer with its packed (kc, nc). `rsb` and `csb` are
    // unused here, since the panel layout is already baked into the buffer
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, Identity>(
            simd,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            packed_b,
            0,
            0,
            beta,
            c,
            rsc,
            csc,
            par,
            ws,
            Some((packed_b, kc, nc)),
            &Identity,
        )
    }
}

/// Run a GEMM whose RHS was already packed by [`pack_rhs_full`], applying the fused
/// [`Epilogue`] `E` to each output element as it is stored
///
/// This is the prepacked twin of [`run_epilogue`]. It rides the same `run_inner` engine as
/// [`run_packed_rhs`] (whole buffer supplied, no per-call B-pack, `kc` and `nc` used
/// verbatim), just with a non-identity `E`. Since the engine never depends on `E`, the
/// pre-epilogue value stored is identical to plain [`run_packed_rhs`]. The result equals a
/// plain prepacked GEMM followed by a scalar map. The epilogue fires once per element, on the
/// final depth panel (`run_inner`'s `last` flag)
///
/// # Safety
///
/// As [`run_packed_rhs`], plus `epi`'s interior pointers must be valid for the problem's `m`
/// and `n`
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_packed_rhs_epilogue<Fam, S, E, const MR_REG: usize, const NR: usize>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: Fam::Acc,
    a: *const Fam::Lhs,
    rsa: isize,
    csa: isize,
    packed_b: *const Fam::Rhs,
    kc: usize,
    nc: usize,
    beta: Fam::Acc,
    c: *mut Fam::Out,
    rsc: isize,
    csc: isize,
    epi: &E,
    par: Parallelism,
    ws: &mut Workspace,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    E: Epilogue<Fam>,
{
    // SAFETY: forwards the prepacked buffer with its packed (kc, nc) and the caller's
    // epilogue. `rsb` and `csb` are unused on the prepacked path
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, E>(
            simd,
            m,
            k,
            n,
            alpha,
            a,
            rsa,
            csa,
            packed_b,
            0,
            0,
            beta,
            c,
            rsc,
            csc,
            par,
            ws,
            Some((packed_b, kc, nc)),
            epi,
        )
    }
}

/// Pack the entire RHS of a fixed `(k, n)` problem into one micropanel-major buffer
///
/// The layout matches the exact order [`run_packed_rhs`] reads it: `jc` column blocks
/// outermost, then depth slices (`pc`), then the `nr`-wide panels of each slice. Each panel's
/// depth is rounded up to `Fam::DEPTH_MULTIPLE` before the cursor advances (the identity for
/// every family except the dot-product ones). This matches the driver's own per-slice packing
/// bit-for-bit, so a prepacked GEMM reproduces a plain one run under the same config
///
/// `dst` must hold `ceil(n/nr) * nr * k` elements for a `DEPTH_MULTIPLE == 1` family, which
/// covers every shipped family except the dot-product ones. A `DEPTH_MULTIPLE > 1` family may
/// only be prepacked with a single depth slice (`kc >= k`). [`run_packed_rhs`] and `run_inner`
/// enforce this with a hard `assert!`. In that case the slice's own padding changes the
/// requirement to `ceil(n/nr) * nr * k.next_multiple_of(DEPTH_MULTIPLE)` instead
///
/// # Safety
///
/// `b` must be valid for the `k x n` region at `rsb` and `csb`. `dst` must be valid for the
/// element count above, for the caller's chosen `kc`
#[allow(clippy::too_many_arguments)]
pub unsafe fn pack_rhs_full<Fam: KernelFamily>(
    dst: *mut Fam::Rhs,
    b: *const Fam::Rhs,
    rsb: isize,
    csb: isize,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    unsafe {
        let mut d = dst;
        let mut jc = 0;
        while jc < n {
            let nc_eff = core::cmp::min(nc, n - jc);
            let n_nt = nc_eff.div_ceil(nr);
            let mut pc = 0;
            while pc < k {
                let kc_eff = core::cmp::min(kc, k - pc);
                // Depth-padded panel size (identity when `DEPTH_MULTIPLE == 1`)
                // `Fam::pack_rhs` zero-fills the padding tail itself
                let kc_eff_pad = kc_eff.next_multiple_of(Fam::DEPTH_MULTIPLE);
                for jt in 0..n_nt {
                    let col = jc + jt * nr;
                    let nr_eff = core::cmp::min(nr, nc_eff - jt * nr);
                    let src = b.offset(pc as isize * rsb + col as isize * csb);
                    Fam::pack_rhs(d, src, rsb, csb, kc_eff, nr_eff, nr);
                    d = d.add(kc_eff_pad * nr);
                }
                pc += kc;
            }
            jc += nc;
        }
    }
}

/// The shared engine behind every public entry point above: plain, fused, and
/// prepacked-RHS, with or without an epilogue. `packed` is `Some((buffer, kc, nc))`
/// on the prepacked path, which skips the per-call B-pack region and reads panels
/// straight from the caller's buffer instead
#[allow(clippy::too_many_arguments)]
unsafe fn run_inner<Fam, S, const MR_REG: usize, const NR: usize, E>(
    simd: S,
    m: usize,
    k: usize,
    n: usize,
    alpha: Fam::Acc,
    a: *const Fam::Lhs,
    rsa: isize,
    csa: isize,
    b: *const Fam::Rhs,
    rsb: isize,
    csb: isize,
    beta: Fam::Acc,
    c: *mut Fam::Out,
    rsc: isize,
    csc: isize,
    par: Parallelism,
    ws: &mut Workspace,
    // `Some((buffer, kc, nc))` on the prepacked-RHS path. Carries the buffer base and the
    // blocking sizes it was packed for, used verbatim so panel addresses line up
    packed: Option<(*const Fam::Rhs, usize, usize)>,
    // The fused epilogue (`Identity`, zero-cost, on every non-fused path). Taken by value
    // here and copied into each worker closure, the same discipline as `Ptr`
    epi: &E,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    E: Epilogue<Fam>,
{
    let epi = *epi;
    unsafe {
        // `mr` counts accumulator lanes, not input lanes, since narrow inputs widen to
        // `Acc` on load. A packed panel row therefore maps 1:1 to an `Acc` lane. This is
        // unchanged for a homogeneous float family (`Lhs == Acc`), but mixed precision
        // (`f16`/`bf16` in, `f32` accumulator) uses the `f32` lane count instead
        let lanes = <S as SimdOps<Fam::Acc>>::LANES;
        let mr = MR_REG * lanes;
        let nr = NR;
        debug_assert!(mr * nr <= SCRATCH_LEN, "microtile exceeds scratch capacity");
        // Every shipped family has `Lhs` and `Rhs` the same size. This lets the single
        // packing allocation below be sized in `Lhs` units and shared between the A and
        // B pack regions
        debug_assert_eq!(
            core::mem::size_of::<Fam::Lhs>(),
            core::mem::size_of::<Fam::Rhs>()
        );

        // `blocking()` sizes the A/B panels by the bytes resident in the packed buffers,
        // the `Lhs`/`Rhs` element size, not `Acc`. This is unchanged for a homogeneous
        // family (`Acc == Lhs`). For a narrow-input family the element size is smaller
        // than `Acc` (i8: 1 vs 4 bytes, f16/bf16: 2 vs 4). The model then fits more
        // elements per cache level than an `Acc`-sized estimate would
        let sizeof_lhs = core::mem::size_of::<Fam::Lhs>().max(1);
        let blk = cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k);
        let mc = blk.mc.next_multiple_of(mr).max(mr);
        // Depth/column panel sizes come from the cache model normally. On the prepacked
        // path they come verbatim from the prepacked buffer's own recorded geometry
        // instead, so panel addressing always lines up with what was packed. `mc`, the A
        // row-block size, is still derived fresh from the real `m` either way
        let (kc, nc) = match packed {
            Some((_, pkc, pnc)) => (pkc.max(1), pnc.next_multiple_of(nr).max(nr)),
            None => {
                // A family whose `Out` is narrower than `Acc` (`OUT_IS_ACC = false`) must not
                // split K across panels. Splitting would round the running sum to `Out` at
                // every panel boundary instead of once at the end. Use the whole contraction
                // as a single panel instead. An `OUT_IS_ACC = true` family keeps the ordinary
                // cache-model `kc`
                //
                // A multi-slice dot family (`DEPTH_MULTIPLE > 1` with `OUT_IS_ACC = true`)
                // also rounds the cache-model `kc` up to the group multiple. This way a group
                // of `DEPTH_MULTIPLE` interleaved depth steps never straddles an interior
                // slice boundary. Only the final, already-padded tail may be short
                // `next_multiple_of(1)` is the identity for every other family, so this
                // changes nothing for them
                let kc = if Fam::OUT_IS_ACC {
                    blk.kc.next_multiple_of(Fam::DEPTH_MULTIPLE)
                } else {
                    k
                };
                (kc.max(1), blk.nc.next_multiple_of(nr).max(nr))
            }
        };

        // Every packed micropanel's depth is rounded up to `Fam::DEPTH_MULTIPLE`. The
        // family's own pack fills the padding tail. This is `1` for every non-dot family,
        // so the `next_multiple_of` calls below are identities
        let q_depth = Fam::DEPTH_MULTIPLE;
        let kc_pad_block = kc.next_multiple_of(q_depth);

        // The prepacked-RHS branch reads panel offsets assuming a single depth slice
        // (`n_nt * pc` below carries no per-slice padding term). Handed a multi-slice buffer
        // from a `DEPTH_MULTIPLE > 1` family, it would read RHS micropanels at the wrong byte
        // offset and silently miscompute. Every `DEPTH_MULTIPLE > 1` family that reaches here
        // already forces `kc = k` via `OUT_IS_ACC = false` above, so this check normally never
        // fires. It is a hard `assert!` rather than `debug_assert!`, since the failure mode
        // is silent wrong output, not a crash. The `q_depth == 1` short-circuit skips it at
        // no cost for every other family
        assert!(
            q_depth == 1 || packed.is_none() || kc >= k,
            "prepacked-RHS with DEPTH_MULTIPLE > 1 requires a single depth slice (kc >= k)"
        );

        let n_nt_max = nc.div_ceil(nr);
        let n_jobs_max = m.div_ceil(mc) * n_nt_max;
        let mnk = m.saturating_mul(n).saturating_mul(k);
        let n_threads = par.resolve(mnk, n_jobs_max);

        // Parallel job-depth floor: the flat job list must be several chunks deep per
        // worker. Otherwise the run's tail degenerates into idle workers waiting on
        // whoever drew the last chunks. When the list is too shallow, shrink `mc` instead,
        // since jobs are `(row-block x column tile)` pairs, so more row-blocks means
        // proportionally more jobs
        //
        // Shrinking `mc` changes no result bit. `mc` stays an `mr` multiple, so the
        // microtile set (every `mr`-aligned row offset plus the one `m`-tail) is identical
        // under any split. `kc`, the only blocking dimension that shapes the per-tile
        // accumulation order, stays untouched. Only the traversal grouping moves
        const PAR_JOBS_PER_WORKER: usize = 8;
        let mc = if n_threads > 1 && n_jobs_max < n_threads * PAR_JOBS_PER_WORKER {
            let n_mc_target = (n_threads * PAR_JOBS_PER_WORKER)
                .div_ceil(n_nt_max)
                .min(m.div_ceil(mr));
            mc.min(m.div_ceil(n_mc_target).next_multiple_of(mr).max(mr))
        } else {
            mc
        };
        let n_mc = m.div_ceil(mc); // count of A/C row macro-blocks, fixed for the whole call
        let n_jobs_max = n_mc * n_nt_max;

        // Rough reuse estimate for the pack/no-pack decision below. If jobs split evenly,
        // each worker handles about `jobs_per_worker` column-tile jobs, capped at
        // `n_nt_max` (one full row-block's worth). This estimates the number of `nr`-wide
        // columns that would share one packed A panel before the pack cost is worth paying
        let jobs_per_worker = n_jobs_max.div_ceil(n_threads.max(1));
        let reuse_cols = jobs_per_worker.min(n_nt_max) * nr;
        // A column-major A (`rsa == 1`) is read in place by walking K with stride `csa`
        // That walk turns TLB/cache-hostile only when both hold. The per-step stride is
        // page-scale (the stride gate), and the whole depth-slice walk spans more address
        // range than stays resident under it (the span gate). A page-scale stride over a
        // slice span that fits cache re-walks warm lines and is faster in place than the
        // packed copy would be. When the gates do fire, the pack cost is shared once per
        // row-block wherever the `shared_a` pre-pass below is open
        //
        // The stride and span gates price the pack's cost. The reuse floor prices its
        // benefit. A pack pays off in proportion to the `n_nt_max` column tiles that
        // re-read each packed panel. A tall/skinny shape (a huge span with few column
        // tiles) amortizes an expensive pack over too few tiles instead, and loses. The
        // reuse floor (`tuning::lhs_pack_reuse`) marks the point where this trade turns,
        // and its default varies by architecture. A knob of `0` makes the conjunct
        // vacuously true, dropping the floor
        let pack_stride = cache::lhs_pack_stride_bytes();
        let lhs_step_bytes = csa
            .unsigned_abs()
            .saturating_mul(core::mem::size_of::<Fam::Lhs>());
        let strided_lhs = rsa == 1
            && lhs_step_bytes >= pack_stride
            && lhs_step_bytes.saturating_mul(kc.min(k)) >= cache::lhs_pack_span_bytes()
            && n_nt_max >= tuning::lhs_pack_reuse();
        let want_pack_lhs =
            reuse_cols > tuning::lhs_pack_threshold() || strided_lhs || Fam::FORCE_PACK_LHS;

        // Shared-LHS pre-pack: instead of every worker re-packing the same row-block's A
        // panel, pack each row-block once up front into a region every worker then reads
        //
        // This only pays when A is actually going to be packed (`rsa != 1 || want_pack_lhs`)
        // It also needs real parallelism to deduplicate across (`n_threads > 1`). Serial
        // keeps the simpler per-worker path instead
        //
        // Beyond that, the pre-pass also needs one of 2 conditions. Either the problem
        // clears a size gate (see `tuning::shared_lhs_mnk`), since the pre-pass adds a
        // fork-join per depth slice that only pays off above that size. Or the worker count
        // reaches `SHARED_A_MIN_THREADS`, where per-worker redundancy becomes the bigger cost
        // regardless of size. Each extra worker is another redundant copy of every A panel it
        // touches
        const SHARED_A_MIN_THREADS: usize = 16;
        let shared_a = n_threads > 1
            && (rsa != 1 || want_pack_lhs)
            && (mnk >= tuning::shared_lhs_mnk() || n_threads >= SHARED_A_MIN_THREADS);

        // A prepacked RHS arrives whole, in micropanel-major layout, so the per-call
        // B-pack region is skipped and compute reads straight from the caller's read-only
        // buffer instead
        let prepacked = packed.is_some();

        // Adaptive RHS packing: every element of B is only ever read via a broadcast, so an
        // unpacked read works under any layout. Packing it once pays off only when that one
        // packed copy gets reused across enough of the `n_mc` row blocks, meaning `m` is
        // large. This never packs when the RHS is already prepacked
        let pack_b = !prepacked && (m > tuning::rhs_pack_threshold() || Fam::FORCE_PACK_RHS);

        // C-tile prefetch gate, decided once per call. Past this working set the output
        // tiles stream from beyond the LLC. A T0 prefetch just ahead of each microkernel
        // call then hides part of the tile's read-modify-write latency (see
        // `prefetch_c_tile` and `cache::prefetch_ws_bytes`). A prefetch only moves cache
        // lines, never arithmetic, so results are bit-identical whether the gate is on,
        // off, or forced
        //
        // The size sum below saturates, since a broadcast (zero-stride) operand's enormous
        // logical extent should gate the prefetch on rather than wrap the sum around
        let ws_bytes = m
            .saturating_mul(k)
            .saturating_mul(core::mem::size_of::<Fam::Lhs>())
            .saturating_add(
                k.saturating_mul(n)
                    .saturating_mul(core::mem::size_of::<Fam::Rhs>()),
            )
            .saturating_add(
                m.saturating_mul(n)
                    .saturating_mul(core::mem::size_of::<Fam::Out>()),
            );
        let prefetch_c = ws_bytes > cache::prefetch_ws_bytes();

        // Whether any row block ever packs its A panel, via the shared pre-pass or the
        // per-worker path. The region-reservation code below uses this to decide whether
        // to reserve A-pack scratch at all
        //
        // `do_pack_lhs` decides per tile from that tile's own `mc_eff`, but `mc` (not
        // `mc_eff`) is always an `mr` multiple. Every row block except possibly the last
        // therefore has `mc_eff == mc` and packs under exactly `rsa != 1 || want_pack_lhs`
        // Only the final, short row block can carry a non-`mr`-multiple `mc_eff`, and it
        // does exactly when `m` itself is not an `mr` multiple. Combining that case here
        // makes this one predicate true whenever any per-tile `do_pack_lhs` call would be
        // This also covers the shared pre-pass, since it packs under
        // `rsa != 1 || want_pack_lhs`, already a subset of this
        let need_a_pack = rsa != 1 || !m.is_multiple_of(mr) || want_pack_lhs;
        // These bounds are computed unconditionally, before either pack decision is
        // consulted. A broadcast (zero-stride) operand can pass shape validation with a
        // logically enormous `k`
        //
        // On a route where neither side ends up packing (`need_a_pack` and `pack_b` both
        // false), skipping this sizing would also skip its "too large" abort. An absurd `k`
        // would then reach the in-place loops below. It would run for a length of time
        // indistinguishable from forever, instead of failing closed with a clear panic
        //
        // Each bound is exactly what packing that side would allocate. It is therefore the
        // correct cap whether or not that side actually packs on this call
        let a_full_region = mc
            .next_multiple_of(mr)
            .checked_mul(kc_pad_block)
            .unwrap_or_else(|| {
                panic!("gemmkit: GEMM {m}x{k}x{n} is too large; the LHS pack region size overflows usize")
            });
        let b_full_elems = nc
            .next_multiple_of(nr)
            .checked_mul(kc_pad_block)
            .unwrap_or_else(|| {
                panic!("gemmkit: GEMM {m}x{k}x{n} is too large; the RHS pack region size overflows usize")
            });
        // Slot count for the A-pack region: one slot per row-block (`n_mc`) when the shared
        // pre-pass owns packing. Each block is written exactly once before any worker reads
        // it. Otherwise one private slot per worker (`n_threads`) on the per-worker path. For
        // a typical square parallel problem `n_mc < n_threads`, so shared-A here uses fewer
        // regions, not more
        let (a_per_region, a_regions) = if need_a_pack {
            (a_full_region, if shared_a { n_mc } else { n_threads })
        } else {
            (0, 0)
        };
        let b_elems = if pack_b { b_full_elems } else { 0 };
        // Reserve workspace scratch only when some side actually packs on this call
        // Otherwise hand back null bases instead. They are never dereferenced, since every
        // tile reads A in place, and B either in place or from the prepacked buffer. An
        // all-in-place workload therefore never grows the pool
        let regions: Regions<Fam::Lhs> = if need_a_pack || pack_b {
            ws.regions::<Fam::Lhs>(a_per_region, a_regions, b_elems)
        } else {
            Regions {
                a_base: core::ptr::null_mut(),
                a_stride: 0,
                b_base: core::ptr::null_mut(),
            }
        };
        let a_base = Ptr(regions.a_base);
        let a_stride = regions.a_stride;
        // `b_base` points at the caller's prepacked buffer when there is one. Otherwise it
        // points at this call's own scratch region, which is never read when B is not
        // packed
        let b_base = match packed {
            Some((pb, _, _)) => Ptr(pb as *mut Fam::Rhs),
            None => Ptr(regions.b_base as *mut Fam::Rhs),
        };

        let a = Ptr(a as *mut Fam::Lhs);
        let b = Ptr(b as *mut Fam::Rhs);
        let c = Ptr(c);
        let ash = alpha_status(alpha);

        // Running element offset of the current `jc` block inside a prepacked RHS buffer
        // Each block holds `n_nt * nr * k.next_multiple_of(DEPTH_MULTIPLE)` elements (the
        // identity `k` for every family except a dot-product one). Unused when not prepacked
        let mut jc_off = 0usize;
        let mut jc = 0;
        while jc < n {
            let nc_eff = core::cmp::min(nc, n - jc);
            let n_nt = nc_eff.div_ceil(nr);

            // `n_jobs` and `grain` depend only on this jc block's `n_nt` (via `nc_eff`) and
            // the worker count, both fixed across every `pc` iteration below. Compute them
            // once per jc block instead of once per depth panel
            let n_jobs = n_mc * n_nt;
            let grain = if shared_a {
                // A is already pre-packed once per row-block below, so a worker no longer
                // gains extra pack reuse from a whole-block chunk. Use the finer general
                // grain for the best load balance instead
                parallel::job_grain(n_jobs, n_threads)
            } else if (rsa != 1 || want_pack_lhs) && n_mc >= n_threads {
                parallel::packed_block_grain(n_nt, n_mc, n_threads)
            } else {
                parallel::job_grain(n_jobs, n_threads)
            };

            let mut pc = 0;
            while pc < k {
                let kc_eff = core::cmp::min(kc, k - pc);
                // Depth-padded panel size the packed buffers are strided by (the identity
                // when `q_depth == 1`)
                let kc_eff_pad = kc_eff.next_multiple_of(q_depth);
                let first = pc == 0;
                // The final depth slice is the only one the fused epilogue may apply to
                // Earlier slices store raw `Acc` partials, per `OUT_IS_ACC`'s contract. An
                // `OUT_IS_ACC = false` family (`kc = k`) has only ever the one slice, so
                // this is always `true`
                let last = pc + kc_eff >= k;
                let beta_eff = if first { beta } else { Fam::Acc::ONE };
                let bst = if first {
                    beta_status(beta)
                } else {
                    BetaStatus::One
                };

                // Pack this depth slice's RHS macro-panel in parallel. Workers pull
                // `nr`-wide column panels from a shared cursor. The `for_each_worker` join
                // is the write-before-read barrier the compute region below relies on. The
                // packed-B buffer is one region shared, not disjoint, across every compute
                // worker, so every panel must finish writing here before any worker reads
                // it. Interleaving this with the compute loop, or moving the pack inside it,
                // would reintroduce that race
                if pack_b {
                    let bcur = parallel::JobCursor::new(n_nt, parallel::job_grain(n_nt, n_threads));
                    parallel::for_each_worker(n_threads, |_tid| {
                        let (b, b_base) = (b, b_base);
                        while let Some((s, e)) = bcur.next_chunk() {
                            for jt in s..e {
                                let col = jc + jt * nr;
                                let nr_eff = core::cmp::min(nr, nc_eff - jt * nr);
                                let dst = b_base.0.add(jt * kc_eff_pad * nr);
                                let src = b.0.offset(pc as isize * rsb + col as isize * csb)
                                    as *const Fam::Rhs;
                                Fam::pack_rhs(dst, src, rsb, csb, kc_eff, nr_eff, nr);
                            }
                        }
                    });
                }

                // Shared-LHS pre-pack: pack each row-block's A panel exactly once into
                // `a_base[ic_idx]`, ahead of the compute loop that reads it. This is the
                // same write-before-read discipline as the packed-B region above, enforced
                // by the same `for_each_worker` join. Workers here pull disjoint `ic` ranges
                // and write disjoint, `ALIGN`-rounded slots, so there is no race even though
                // it is the same region
                if shared_a {
                    let acur = parallel::JobCursor::new(n_mc, parallel::job_grain(n_mc, n_threads));
                    parallel::for_each_worker(n_threads, |_tid| {
                        let (a, a_base) = (a, a_base);
                        while let Some((s, e)) = acur.next_chunk() {
                            for ic_idx in s..e {
                                let ic = ic_idx * mc;
                                let mc_eff = core::cmp::min(mc, m - ic);
                                let dst = a_base.0.add(ic_idx * a_stride);
                                let src = a.0.offset(ic as isize * rsa + pc as isize * csa)
                                    as *const Fam::Lhs;
                                Fam::pack_lhs(dst, src, rsa, csa, mc_eff, kc_eff, mr);
                            }
                        }
                    });
                }

                let cur = parallel::JobCursor::new(n_jobs, grain);

                parallel::for_each_worker(n_threads, |tid| {
                    // Capture the `Send + Sync` pointer shims as whole structs. Edition
                    // 2024's disjoint closure capture (RFC 2229) would otherwise capture each
                    // one's inner `*mut` field on its own, and that field is not `Sync`. The
                    // closure would then fail the bound rayon's worker spawn needs
                    let (a, b, c, a_base, b_base, epi) = (a, b, c, a_base, b_base, epi);
                    // This worker's private A-pack scratch slot, per-worker path only. On
                    // the shared-A path `tid` can run past `n_mc` (there are only `n_mc`
                    // slots there). There `a_base + tid*a_stride` would compute an
                    // out-of-bounds pointer, even though nothing ever reads through it
                    // Use a null base instead to stay clear of that
                    let a_buf = if shared_a {
                        core::ptr::null_mut::<Fam::Lhs>()
                    } else {
                        a_base.0.add(tid * a_stride)
                    };
                    let mut scratch = [const { MaybeUninit::<Fam::Acc>::uninit() }; SCRATCH_LEN];
                    let scratch_ptr = scratch.as_mut_ptr() as *mut Fam::Acc;

                    // Cached A-panel pack state for whichever row block the worker is
                    // currently on. `packed_a` is tracked explicitly rather than inferred
                    // from `a_cs == mr`. An unpacked column-major A could legitimately have
                    // `csa == mr` too, which would then be misread as the packed address
                    // formula
                    let mut cur_ic = usize::MAX;
                    let mut a_panel_base: *const Fam::Lhs = core::ptr::null();
                    let mut a_cs: isize = 0;
                    let mut packed_a = false;

                    while let Some((start, end)) = cur.next_chunk() {
                        for q in start..end {
                            let ic_idx = q / n_nt;
                            let jt = q % n_nt;
                            let ic = ic_idx * mc;
                            let mc_eff = core::cmp::min(mc, m - ic);

                            if ic_idx != cur_ic {
                                cur_ic = ic_idx;
                                if shared_a {
                                    // This block's A panel was already packed once by the
                                    // pre-pass above. That used the same byte address and
                                    // the same pack call the per-worker branch below would
                                    // make. Reading it here reproduces that packed input
                                    // bit-for-bit
                                    a_panel_base =
                                        a_base.0.add(ic_idx * a_stride) as *const Fam::Lhs;
                                    a_cs = mr as isize;
                                    packed_a = true;
                                } else {
                                    let src = a.0.offset(ic as isize * rsa + pc as isize * csa)
                                        as *const Fam::Lhs;
                                    if do_pack_lhs(rsa, mc_eff, mr, want_pack_lhs) {
                                        Fam::pack_lhs(a_buf, src, rsa, csa, mc_eff, kc_eff, mr);
                                        a_panel_base = a_buf as *const Fam::Lhs;
                                        a_cs = mr as isize;
                                        packed_a = true;
                                    } else {
                                        a_panel_base = src;
                                        a_cs = csa;
                                        packed_a = false;
                                    }
                                }
                            }

                            let col = jc + jt * nr;
                            let nr_eff = core::cmp::min(nr, nc_eff - jt * nr);
                            // Where this call reads its RHS panel from, and at what strides:
                            //
                            // - prepacked: the global panel at
                            //   `jc_off + nr*(n_nt*pc + jt*kc_eff_pad)`, matching the layout
                            //   `pack_rhs_full` wrote for this exact (jc, pc, jt)
                            // - packed (this call's own B-pack): the per-slice contiguous
                            //   panel at `jt*kc_eff_pad*nr`, with strides (nr, 1)
                            // - unpacked: read straight from `b` at the original strides
                            let (bpan, b_rs_k, b_cs_k) = if prepacked {
                                // `n_nt * pc` stands in for the padded depth already consumed
                                // by earlier slices. This is exact when `q_depth == 1`, since
                                // every prior slice then has `kc_eff_pad == kc_eff == kc`. It
                                // is trivially exact otherwise, because the assert above
                                // forces `pc == 0` (a single slice) for any
                                // `DEPTH_MULTIPLE > 1` family reaching this branch
                                (
                                    b_base.0.add(jc_off + nr * (n_nt * pc + jt * kc_eff_pad))
                                        as *const Fam::Rhs,
                                    nr as isize,
                                    1,
                                )
                            } else if pack_b {
                                (
                                    b_base.0.add(jt * kc_eff_pad * nr) as *const Fam::Rhs,
                                    nr as isize,
                                    1,
                                )
                            } else {
                                (
                                    b.0.offset(pc as isize * rsb + col as isize * csb)
                                        as *const Fam::Rhs,
                                    rsb,
                                    csb,
                                )
                            };

                            // Everything in this column strip (every `mr`-row sub-tile of the
                            // current row block) runs inside the ISA's target-feature
                            // context
                            simd.vectorize(|| {
                                let mut ir = 0;
                                while ir < mc_eff {
                                    let mr_eff = core::cmp::min(mr, mc_eff - ir);
                                    let apan = if packed_a {
                                        a_panel_base.add((ir / mr) * mr * kc_eff_pad)
                                    } else {
                                        a_panel_base.offset(ir as isize * rsa)
                                    };
                                    let cptr =
                                        c.0.offset((ic + ir) as isize * rsc + col as isize * csc);
                                    if prefetch_c {
                                        prefetch_c_tile(cptr, rsc, csc, mr_eff, nr_eff);
                                    }
                                    // `ic + ir` and `col` are this sub-tile's origin in the
                                    // oriented problem frame, letting a per-row or per-col
                                    // epilogue bias resolve its absolute base. `last` gates
                                    // the epilogue to fire at most once per output element
                                    Fam::microkernel_epi::<S, E, MR_REG, NR>(
                                        simd,
                                        kc_eff,
                                        alpha,
                                        beta_eff,
                                        ash,
                                        bst,
                                        apan,
                                        a_cs,
                                        bpan,
                                        b_rs_k,
                                        b_cs_k,
                                        cptr,
                                        rsc,
                                        csc,
                                        mr_eff,
                                        nr_eff,
                                        ic + ir,
                                        col,
                                        last,
                                        &epi,
                                        scratch_ptr,
                                    );
                                    ir += mr;
                                }
                            });
                        }
                    }
                });

                pc += kc;
            }
            // Advance past this jc block's region in the prepacked buffer: `n_nt` panels of
            // `nr` columns by `k.next_multiple_of(q_depth)` depth. This is the single padded
            // slice for any `q_depth > 1` family, by the guard above, or plain `k` when
            // `q_depth == 1`
            jc_off += n_nt * nr * k.next_multiple_of(q_depth);
            jc += nc;
        }
    }
}
