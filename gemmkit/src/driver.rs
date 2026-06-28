//! The generic GEMM driver (layer L4): one five-loop nest, fully generic over
//! the [`KernelFamily`] and the ISA token. It never mentions a concrete element
//! type, a concrete ISA, or a macro. Adding a family or an ISA leaves this file
//! untouched — the open/closed property the architecture promises.
//!
//! Loop structure (BLIS order): `jc` (N / L3) → `pc` (K, *not* parallel) → a
//! flat 1-D job list over `(ic row-block × jt column-tile)` that workers drain by
//! pulling chunks from a shared cursor on demand (a single work-gate; faster cores
//! take more). `beta` applies only on the first depth slice; later slices
//! accumulate. Each output tile is computed start-to-finish by one worker over
//! the full K, and the blocking is thread-count independent, so the result is
//! bit-identical for any [`Parallelism`] regardless of how the chunks land.

use core::mem::MaybeUninit;

use crate::cache;
use crate::kernel::{AlphaStatus, BetaStatus, KernelFamily, SCRATCH_LEN};
use crate::parallel::{self, Parallelism};
use crate::scalar::Scalar;
use crate::simd::SimdOps;
use crate::tuning;
use crate::workspace::Workspace;

/// `Send + Sync` raw-pointer shim so worker closures can capture the shared
/// matrices. Soundness rests on the driver's invariants: workers write disjoint
/// C tiles / A buffers and only read shared inputs, and `C` is validated not to
/// alias `A`/`B` by the safe API layer.
#[derive(Copy, Clone)]
struct Ptr<T>(*mut T);
// SAFETY: see the type comment — access is disjoint by construction.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

#[inline]
fn alpha_status<T: crate::scalar::Scalar>(a: T) -> AlphaStatus {
    // `alpha == 0` is handled upstream, so only One / Other reach here.
    if a == T::ONE {
        AlphaStatus::One
    } else {
        AlphaStatus::Other
    }
}

#[inline]
fn beta_status<T: crate::scalar::Scalar>(b: T) -> BetaStatus {
    if b == T::ZERO {
        BetaStatus::Zero
    } else if b == T::ONE {
        BetaStatus::One
    } else {
        BetaStatus::Other
    }
}

/// Whether to pack the LHS macro-block. A non-unit row stride or a partial row
/// panel *forces* packing (the microkernel always reads full `mr`-row vectors).
/// Otherwise pack only when `want_pack` — i.e. each worker reuses the packed
/// block across enough column tiles to amortize the copy. Because every worker
/// packs its own block, redundant packing across workers makes column-major
/// inputs cheaper left *unpacked* unless the per-worker reuse is high.
#[inline]
fn do_pack_lhs(rsa: isize, mc_eff: usize, mr: usize, want_pack: bool) -> bool {
    rsa != 1 || !mc_eff.is_multiple_of(mr) || want_pack
}

/// Run a GEMM with the given family, ISA token, and microtile geometry.
///
/// Preconditions (established by the dispatch layer): `m, n, k > 0`, `alpha != 0`,
/// and the problem has been orientation-normalized. The driver is correct for
/// any shape (gemv shapes fall through the partial-tile path), but the dispatch
/// layer routes `m == 1 || n == 1` to the dedicated gemv path for speed.
///
/// # Safety
/// All pointers must be valid for the regions implied by the strides/sizes, and
/// `C` must not alias `A` or `B`.
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
    S: SimdOps<Fam::Lhs> + SimdOps<Fam::Acc>,
{
    // SAFETY: forwarded to `run_inner` with no prepacked RHS (the standard path).
    unsafe {
        run_inner::<Fam, S, MR_REG, NR>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, par, ws, None,
        )
    }
}

/// Run a GEMM whose RHS is already prepacked by [`pack_rhs_full`]. `packed_b` is
/// the buffer base (no RHS strides — the layout is baked in). It is read-only and
/// shared immutably across workers, so unlike the per-call B-pack it needs no
/// barrier. `kc`/`nc` are the sizes the buffer was packed for; the driver uses
/// them verbatim so panel addresses always match the buffer (`mc` is still derived
/// at the real `m`).
///
/// # Safety
/// As [`run`], plus `packed_b` must come from [`pack_rhs_full`] for the same
/// `(k, n, kc, nc, nr = NR)`.
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
    S: SimdOps<Fam::Lhs> + SimdOps<Fam::Acc>,
{
    // SAFETY: forwarded with the prepacked buffer and its packed (kc, nc); `rsb`/
    // `csb` are unused on the prepacked path (the panel layout is baked in).
    unsafe {
        run_inner::<Fam, S, MR_REG, NR>(
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
        )
    }
}

/// Pack the entire RHS of a fixed `(k, n)` problem into one micropanel-major
/// buffer, in the exact order [`run_packed_rhs`] reads panels: `jc` blocks
/// outermost, then depth slices, then the panels of each slice (cursor advancing
/// `kc_eff * nr` per panel). The bytes match the driver's own per-slice packing,
/// so a prepacked GEMM is bit-identical to a plain one — the single source of
/// truth for the layout. `dst` must hold `ceil(n/nr) * nr * k` elements.
///
/// # Safety
/// `b` valid for the `k × n` region at `rsb`/`csb`; `dst` valid for the count above.
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
                for jt in 0..n_nt {
                    let col = jc + jt * nr;
                    let nr_eff = core::cmp::min(nr, nc_eff - jt * nr);
                    let src = b.offset(pc as isize * rsb + col as isize * csb);
                    Fam::pack_rhs(d, src, rsb, csb, kc_eff, nr_eff, nr);
                    d = d.add(kc_eff * nr);
                }
                pc += kc;
            }
            jc += nc;
        }
    }
}

/// Pack the entire LHS of a fixed `(m, k)` problem into one micropanel-major
/// buffer, for the prepacked-LHS reuse path ([`crate::gemm_packed_a`]).
///
/// A prepacked **LHS** is, by the engine's A/B symmetry, the prepacked **RHS** of
/// the *transposed* product `Cᵀ = Bᵀ·Aᵀ` — the orientation the dispatch layer
/// already takes for a row-major-ish `C`. So the genuine `m × k` LHS is packed in
/// exactly the same micropanel-major order [`pack_rhs_full`] would lay down for a
/// `k × m` RHS: the depth is `k` and the leading (column) dimension is the LHS's
/// `m` rows, so the LHS row stride plays the RHS *column* stride and the LHS column
/// stride plays the RHS *depth* stride. Delegating to [`pack_rhs_full`] keeps that
/// single layout the one source of truth — the consuming [`run_packed_rhs`] (driven
/// transposed) reads back the very bytes written here, so a prepacked-A GEMM is
/// bit-identical to a plain one (modulo the documented small-matrix / gemv carve-outs).
/// `dst` must hold `ceil(m/nr) * nr * k` elements; `(kc, nc)` are the transposed
/// problem's blocking (depth `kc`, leading `nc` over the `m` rows). The pointers are
/// `Fam::Rhs`-typed because the LHS is laid down *as the transposed product's RHS*
/// and read back as such — on the size-homogeneous float family `Lhs == Rhs`, so the
/// concrete call site passes the LHS buffer with no cast.
///
/// # Safety
/// `a` valid for the `m × k` region at `rsa`/`csa`; `dst` valid for the count above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pack_lhs_full<Fam: KernelFamily>(
    dst: *mut Fam::Rhs,
    a: *const Fam::Rhs,
    rsa: isize,
    csa: isize,
    m: usize,
    k: usize,
    kc: usize,
    nc: usize,
    nr: usize,
) {
    // The transposed RHS is `k × m`: depth = k (LHS columns, stride `csa`), leading =
    // m (LHS rows, stride `rsa`). `pack_rhs_full` takes `(rsb = depth stride, csb =
    // leading stride)`, so feed it `(rsb = csa, csb = rsa)` and `(k, n = m)`.
    unsafe { pack_rhs_full::<Fam>(dst, a, csa, rsa, k, m, kc, nc, nr) }
}

/// The shared GEMM engine behind [`run`] (no prepacked RHS) and [`run_packed_rhs`]
/// (`packed_b = Some(..)`). When prepacked, the per-call B-pack region is skipped
/// and the compute region reads panels from the prepacked buffer instead.
#[allow(clippy::too_many_arguments)]
unsafe fn run_inner<Fam, S, const MR_REG: usize, const NR: usize>(
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
    // `Some((buffer, kc, nc))` on the prepacked-RHS path: the buffer base plus the
    // blocking sizes it was packed for (used verbatim so panel addresses match).
    packed: Option<(*const Fam::Rhs, usize, usize)>,
) where
    Fam: KernelFamily,
    S: SimdOps<Fam::Lhs> + SimdOps<Fam::Acc>,
{
    unsafe {
        let lanes = <S as SimdOps<Fam::Lhs>>::LANES;
        let mr = MR_REG * lanes;
        let nr = NR;
        debug_assert!(mr * nr <= SCRATCH_LEN, "microtile exceeds scratch capacity");
        // v1 families are size-homogeneous; the packing buffer is sized in Lhs
        // units and shared with Rhs.
        debug_assert_eq!(
            core::mem::size_of::<Fam::Lhs>(),
            core::mem::size_of::<Fam::Rhs>()
        );

        let sizeof_acc = core::mem::size_of::<Fam::Acc>().max(1);
        let blk = cache::topology().blocking(mr, nr, sizeof_acc, m, n, k);
        let mc = blk.mc.next_multiple_of(mr).max(mr);
        // Depth/column panel sizes: from the cache model normally, but taken
        // verbatim from the prepacked buffer's recorded geometry on the prepacked
        // path so the global panel addressing always matches what was packed (the
        // A row-block size `mc` is still model-derived at the real `m`).
        let (kc, nc) = match packed {
            Some((_, pkc, pnc)) => (pkc.max(1), pnc.next_multiple_of(nr).max(nr)),
            None => (blk.kc.max(1), blk.nc.next_multiple_of(nr).max(nr)),
        };

        let n_mc = m.div_ceil(mc); // row macro-blocks (constant across panels)
        let n_nt_max = nc.div_ceil(nr);
        let n_jobs_max = n_mc * n_nt_max;
        let mnk = m.saturating_mul(n).saturating_mul(k);
        let n_threads = par.resolve(mnk, n_jobs_max);

        // Reuse-aware LHS pack decision: each worker handles roughly
        // `jobs_per_worker` column strips, all within one row block
        let jobs_per_worker = n_jobs_max.div_ceil(n_threads.max(1));
        let reuse_cols = jobs_per_worker.min(n_nt_max) * nr;
        // A column-major A (`rsa == 1`) is read in place by walking K with stride `csa`,
        // so once `csa * sizeof(Lhs)` reaches ~a memory page the strided read thrashes
        // the TLB and packing A into a contiguous panel wins regardless of reuse
        // and it is redundancy-free here
        let pack_stride = cache::lhs_pack_stride_bytes();
        let strided_lhs = rsa == 1
            && csa
                .unsigned_abs()
                .saturating_mul(core::mem::size_of::<Fam::Lhs>())
                >= pack_stride;
        let want_pack_lhs = reuse_cols > tuning::lhs_pack_threshold() || strided_lhs;

        // Shared-LHS pre-pack: on the parallel packed-A path, pack each row-block's
        // A panel once into a shared region (below) instead of every worker that
        // touches it re-packing. Gated to the packed path (`rsa != 1 ||
        // want_pack_lhs`), to real parallelism (serial keeps the unchanged
        // per-worker path), and to a workload threshold (the pre-pass adds a
        // fork-join per depth slice that only pays at large sizes; see
        // `tuning::shared_lhs_mnk`).
        let shared_a =
            n_threads > 1 && (rsa != 1 || want_pack_lhs) && mnk >= tuning::shared_lhs_mnk();

        // A prepacked RHS is supplied whole in micropanel-major layout, so the
        // per-call B-pack is disabled and the compute region reads from the caller's
        // read-only buffer instead.
        let prepacked = packed.is_some();

        // Adaptive RHS packing: B (read only via broadcast, so any layout works
        // unpacked) is packed once and reused across all `n_mc` row blocks. The
        // copy amortizes only when that reuse is high (large `m`); otherwise B is
        // read in place. Never packed here when the RHS is already prepacked.
        let pack_b = !prepacked && m > tuning::rhs_pack_threshold();

        // One packing allocation. The LHS region count is `n_mc` when shared (one
        // slot per row-block, written once by the pre-pass) or `n_threads` when
        // per-worker (each worker owns a private scratch slot). For square parallel
        // problems `n_mc < n_threads`, so shared-A uses *fewer* slots, not more.
        let a_per_region = mc.next_multiple_of(mr) * kc;
        let a_regions = if shared_a { n_mc } else { n_threads };
        let b_elems = if pack_b {
            nc.next_multiple_of(nr) * kc
        } else {
            0
        };
        let regions = ws.regions::<Fam::Lhs>(a_per_region, a_regions, b_elems);
        let a_base = Ptr(regions.a_base);
        let a_stride = regions.a_stride;
        // Prepacked: read from the caller buffer; else from the per-call scratch.
        let b_base = match packed {
            Some((pb, _, _)) => Ptr(pb as *mut Fam::Rhs),
            None => Ptr(regions.b_base as *mut Fam::Rhs),
        };

        let a = Ptr(a as *mut Fam::Lhs);
        let b = Ptr(b as *mut Fam::Rhs);
        let c = Ptr(c);
        let ash = alpha_status(alpha);

        // Running element offset of the current `jc` block inside a prepacked RHS
        // buffer: each block holds `n_nt * nr * k` elements (every padded column
        // appears once per depth row). Unused when not prepacked.
        let mut jc_off = 0usize;
        let mut jc = 0;
        while jc < n {
            let nc_eff = core::cmp::min(nc, n - jc);
            let n_nt = nc_eff.div_ceil(nr);

            // Job count and cursor grain depend only on this column panel and the worker
            // count — invariant across the depth (`pc`) loop, so compute them once here.
            // The packed-LHS path (whole-row-block chunks) is split for load balance; the
            // general path uses the shared `job_grain` oversample. See `packed_block_grain`.
            let n_jobs = n_mc * n_nt;
            let grain = if shared_a {
                // A is pre-packed once per block below, so whole-block chunking no
                // longer buys pack reuse — use the fine grain for the best balance.
                parallel::job_grain(n_jobs, n_threads)
            } else if (rsa != 1 || want_pack_lhs) && n_mc >= n_threads {
                parallel::packed_block_grain(n_nt, n_mc, n_threads)
            } else {
                parallel::job_grain(n_jobs, n_threads)
            };

            let mut pc = 0;
            while pc < k {
                let kc_eff = core::cmp::min(kc, k - pc);
                let first = pc == 0;
                let beta_eff = if first { beta } else { Fam::Acc::ONE };
                let bst = if first {
                    beta_status(beta)
                } else {
                    BetaStatus::One
                };

                // Pack the RHS macro-panel in parallel (when packing): workers pull
                // NR-wide column panels from a shared cursor. The `for_each_worker`
                // join below is the write-before-read barrier the compute region
                // depends on — packed B is the *one* buffer shared (non-disjoint)
                // across all compute workers, so every panel must be written here
                // before any worker reads it. Fusing this region into the compute
                // loop, or moving packing inside it, would reintroduce a data race.
                if pack_b {
                    let bcur = parallel::JobCursor::new(n_nt, parallel::job_grain(n_nt, n_threads));
                    parallel::for_each_worker(n_threads, |_tid| {
                        let (b, b_base) = (b, b_base);
                        while let Some((s, e)) = bcur.next_chunk() {
                            for jt in s..e {
                                let col = jc + jt * nr;
                                let nr_eff = core::cmp::min(nr, nc_eff - jt * nr);
                                let dst = b_base.0.add(jt * kc_eff * nr);
                                let src = b.0.offset(pc as isize * rsb + col as isize * csb)
                                    as *const Fam::Rhs;
                                Fam::pack_rhs(dst, src, rsb, csb, kc_eff, nr_eff, nr);
                            }
                        }
                    });
                }

                // Shared-LHS pre-pack: pack each row-block's A panel once into
                // `a_base[ic_idx]`. The `for_each_worker` join is the write-before-
                // read barrier the compute region depends on — same discipline as the
                // packed-B region above. Workers pull disjoint `ic` ranges and write
                // disjoint, ALIGN-rounded slots, so there is no intra-region race.
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
                    // Force whole-struct capture of the `Send + Sync` pointer shims;
                    // edition-2024 (RFC 2229) closures otherwise capture the inner
                    // `*mut` fields disjointly, which are not `Sync`, so the closure
                    // would fail the bound needed to move into the rayon workers.
                    let (a, b, c, a_base, b_base) = (a, b, c, a_base, b_base);
                    // Per-worker scratch slot (per-worker path only). On the shared-A
                    // path `tid` may exceed `n_mc`, so `a_base + tid*a_stride` would be
                    // out-of-bounds pointer arithmetic even though it is never read —
                    // use a null base there instead.
                    let a_buf = if shared_a {
                        core::ptr::null_mut::<Fam::Lhs>()
                    } else {
                        a_base.0.add(tid * a_stride)
                    };
                    let mut scratch = [const { MaybeUninit::<Fam::Acc>::uninit() }; SCRATCH_LEN];
                    let scratch_ptr = scratch.as_mut_ptr() as *mut Fam::Acc;

                    // Cached LHS pack state for the current row block. `packed_a` is
                    // carried explicitly (not re-derived as `a_cs == mr`, which could
                    // collide for a self-overlapping column-major A whose `csa` equals
                    // `mr` and then read the in-place A with the packed address formula).
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
                                    // Read the block's A panel that the pre-pass
                                    // packed once (same bytes, same packed read
                                    // formula as the per-worker case ⇒ bit-identical).
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
                            // Prepacked B -> global panel at the layout-identity
                            //   offset jc_off + nr*(n_nt*pc + jt*kc_eff)
                            //   (pc == sum of prior kc_eff, so this matches the
                            //   per-(jc,pc) layout pack_rhs_full wrote)
                            // packed B -> per-slice contiguous panel with (NR, 1)
                            // unpacked B -> read in place with the original strides
                            let (bpan, b_rs_k, b_cs_k) = if prepacked {
                                (
                                    b_base.0.add(jc_off + nr * (n_nt * pc + jt * kc_eff))
                                        as *const Fam::Rhs,
                                    nr as isize,
                                    1,
                                )
                            } else if pack_b {
                                (
                                    b_base.0.add(jt * kc_eff * nr) as *const Fam::Rhs,
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

                            // Process the whole column strip in the ISA's target-feature context
                            simd.vectorize(|| {
                                let mut ir = 0;
                                while ir < mc_eff {
                                    let mr_eff = core::cmp::min(mr, mc_eff - ir);
                                    let apan = if packed_a {
                                        a_panel_base.add((ir / mr) * mr * kc_eff)
                                    } else {
                                        a_panel_base.offset(ir as isize * rsa)
                                    };
                                    let cptr =
                                        c.0.offset((ic + ir) as isize * rsc + col as isize * csc);
                                    Fam::microkernel::<S, MR_REG, NR>(
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
            jc_off += n_nt * nr * k;
            jc += nc;
        }
    }
}
