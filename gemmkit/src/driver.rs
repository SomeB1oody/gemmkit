//! The generic GEMM driver (layer L4): one 5-loop nest, fully generic over
//! the [`KernelFamily`] and the ISA token. It never mentions a concrete element
//! type, a concrete ISA, or a macro. Adding a family or an ISA leaves this file
//! untouched - the open/closed property the architecture promises
//!
//! Loop structure (BLIS order): `jc` (N / L3) -> `pc` (K, *not* parallel) -> a
//! flat 1-D job list over `(ic row-block x jt column-tile)` that workers drain
//! by pulling chunks from a shared cursor on demand (a single work-gate; faster
//! cores take more). `beta` applies only on the first depth slice; later slices
//! accumulate. Each output tile is computed start-to-finish by one worker over
//! the full K, and the blocking is thread-count independent, so the result is
//! **reproducible** for any [`Parallelism`] regardless of how the chunks land -
//! a fixed input/config gives the same output, independent of the worker
//! count. (Bitwise serial-vs-parallel identity holds today because both paths
//! run the same kernel, but the contract is reproducibility under a fixed
//! config.)

use core::mem::MaybeUninit;

use crate::cache;
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::kernel::{AlphaStatus, BetaStatus, KernelFamily, SCRATCH_LEN};
use crate::parallel::{self, Parallelism, Ptr};
use crate::scalar::Scalar;
use crate::simd::{KernelSimd, SimdOps};
use crate::tuning;
use crate::workspace::{Regions, Workspace};

/// Precomputed `alpha` state so the microkernel never compares floats.
/// `alpha == 0` is handled upstream (routed to the scale-only path), so only
/// One / Other reach here. Shared with the [`crate::special`] paths
#[inline]
pub(crate) fn alpha_status<T: crate::scalar::Scalar>(a: T) -> AlphaStatus {
    if a == T::ONE {
        AlphaStatus::One
    } else {
        AlphaStatus::Other
    }
}

/// Precomputed `beta` state for the microkernel. Shared with the [`crate::special`] paths
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

/// Whether to pack the LHS macro-block. A non-unit row stride or a partial row
/// panel *forces* packing (the microkernel always reads full `mr`-row vectors).
/// Otherwise pack only when `want_pack` - i.e. each worker reuses the packed
/// block across enough column tiles to amortize the copy. Because every worker
/// packs its own block, redundant packing across workers makes column-major
/// inputs cheaper left *unpacked* unless the per-worker reuse is high
#[inline]
fn do_pack_lhs(rsa: isize, mc_eff: usize, mr: usize, want_pack: bool) -> bool {
    rsa != 1 || !mc_eff.is_multiple_of(mr) || want_pack
}

/// Run a GEMM with the given family, ISA token, and microtile geometry
///
/// Preconditions (established by the dispatch layer): `m, n, k > 0`, `alpha != 0`,
/// and the problem has been orientation-normalized. The driver is correct for
/// any shape (gemv shapes fall through the partial-tile path), but the dispatch
/// layer routes `m == 1 || n == 1` to the dedicated gemv path for speed
///
/// # Safety
/// All pointers must be valid for the regions implied by the strides/sizes, and
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
    // SAFETY: forwarded to `run_inner` with no prepacked RHS (the standard path) and
    // the zero-cost `Identity` epilogue - the exact code path this driver has always run
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, Identity>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, par, ws, None,
            &Identity,
        )
    }
}

/// Run a GEMM applying the fused [`Epilogue`] `E` to each output element as it is stored,
/// instead of materializing the raw product and mapping it afterward. The plain-GEMM
/// forwarder ([`run`]) is exactly this with `E = Identity`; with a non-identity `E` the
/// engine (blocking, packing, scheduling) is unchanged, so the pre-epilogue bits are
/// identical to plain `gemm` and the fused result equals `gemm()` then a scalar map
///
/// # Safety
/// As [`run`], plus `epi`'s interior pointers must be valid for the problem's `m`/`n`
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
    // SAFETY: forwarded to `run_inner` with no prepacked RHS and the caller's epilogue
    unsafe {
        run_inner::<Fam, S, MR_REG, NR, E>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, par, ws, None, epi,
        )
    }
}

/// Run a GEMM whose RHS is already prepacked by [`pack_rhs_full`]. `packed_b` is
/// the buffer base (no RHS strides - the layout is baked in). It is read-only and
/// shared immutably across workers, so unlike the per-call B-pack it needs no
/// barrier. `kc`/`nc` are the sizes the buffer was packed for; the driver uses
/// them verbatim so panel addresses always match the buffer (`mc` is still
/// derived at the real `m`)
///
/// # Safety
/// As [`run`], plus `packed_b` must come from [`pack_rhs_full`] for the same
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
    // SAFETY: forwarded with the prepacked buffer and its packed (kc, nc); `rsb`/
    // `csb` are unused on the prepacked path (the panel layout is baked in)
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

/// Run a GEMM whose RHS is already prepacked by [`pack_rhs_full`], applying the
/// fused [`Epilogue`] `E` to each output element as it is stored. The prepacked twin
/// of [`run_epilogue`]: it is exactly [`run_packed_rhs`] with a non-identity `E`
/// instead of `Identity`, riding the same `run_inner` engine (the buffer is
/// supplied whole, the per-call B-pack is skipped, `kc`/`nc` are used verbatim). The
/// engine (blocking, scheduling, the panel bytes read) is epilogue-independent, so
/// the pre-epilogue store is identical to plain [`run_packed_rhs`] and the fused
/// result equals a plain prepacked GEMM then a scalar map. The epilogue fires on the
/// final depth panel (the `last_k` gate in `run_inner`)
///
/// # Safety
/// As [`run_packed_rhs`], plus `epi`'s interior pointers must be valid for the
/// problem's `m`/`n`
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
    // SAFETY: forwarded with the prepacked buffer and its packed (kc, nc) plus the
    // caller's epilogue; `rsb`/`csb` are unused on the prepacked path
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

/// Pack the entire RHS of a fixed `(k, n)` problem into one micropanel-major
/// buffer, in the exact order [`run_packed_rhs`] reads panels: `jc` blocks
/// outermost, then depth slices, then the panels of each slice (cursor
/// advancing `kc_eff * nr` per panel). The packed bytes match the driver's own
/// per-slice packing, so a prepacked GEMM reproduces a plain one under the same
/// config - one single source of truth for the layout. `dst` must hold
/// `ceil(n/nr) * nr * k` elements
///
/// # Safety
/// `b` valid for the `k x n` region at `rsb`/`csb`; `dst` valid for the count above
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
                // Dot families depth-pad each panel (`Fam::pack_rhs` fills the tail), so
                // the cursor advances by the padded depth. Identity for `DEPTH_MULTIPLE == 1`
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

/// The shared GEMM engine behind [`run`] (no prepacked RHS) and [`run_packed_rhs`]
/// (`packed_b = Some(..)`). When prepacked, the per-call B-pack region is skipped
/// and the compute region reads panels from the prepacked buffer instead
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
    // `Some((buffer, kc, nc))` on the prepacked-RHS path: the buffer base plus
    // the blocking sizes it was packed for (used verbatim so panel addresses match)
    packed: Option<(*const Fam::Rhs, usize, usize)>,
    // The fused epilogue (zero-cost `Identity` on the plain/prepacked paths)
    // Copied into each worker closure (`E: Copy + Send + Sync`), the same
    // capture discipline as `Ptr`
    epi: &E,
) where
    Fam: KernelFamily,
    S: KernelSimd<Fam::Lhs, Fam::Rhs, Fam::Acc, Fam::Out>,
    E: Epilogue<Fam>,
{
    let epi = *epi;
    unsafe {
        // `mr` is in *accumulator* lanes: the microkernel widens narrow inputs
        // into `Acc` registers, so a panel row maps to an `Acc` lane. Homogeneous
        // float families have `Lhs == Acc`; for mixed precision (`f16` in, `f32`
        // acc) this is the `f32` lane count
        let lanes = <S as SimdOps<Fam::Acc>>::LANES;
        let mr = MR_REG * lanes;
        let nr = NR;
        debug_assert!(mr * nr <= SCRATCH_LEN, "microtile exceeds scratch capacity");
        // v1 families are size-homogeneous; the packing buffer is sized in Lhs
        // units and shared with Rhs
        debug_assert_eq!(
            core::mem::size_of::<Fam::Lhs>(),
            core::mem::size_of::<Fam::Rhs>()
        );

        // Block on the *packed input* element size: `blocking()` sizes the A/B
        // panels, which are stored in `Lhs` (== `Rhs`) units, not the accumulator
        // For homogeneous families (`Acc == Lhs`) this is unchanged; it only
        // affects narrow packed types (i8: 1 vs 4 bytes; f16/bf16: 2 vs 4)
        let sizeof_lhs = core::mem::size_of::<Fam::Lhs>().max(1);
        let blk = cache::topology().blocking(mr, nr, sizeof_lhs, m, n, k);
        let mc = blk.mc.next_multiple_of(mr).max(mr);
        // Depth/column panel sizes: from the cache model normally, but taken
        // verbatim from the prepacked buffer's recorded geometry on the prepacked
        // path so the global panel addressing always matches what was packed (the
        // A row-block size `mc` is still model-derived at the real `m`)
        let (kc, nc) = match packed {
            Some((_, pkc, pnc)) => (pkc.max(1), pnc.next_multiple_of(nr).max(nr)),
            None => {
                // A family whose `Out` is narrower than `Acc` (mixed precision) must
                // not split K, or the running sum would round to `Out` between panels
                // Use the whole contraction as one panel so it accumulates in `Acc`
                // and rounds once. Homogeneous families keep the cache-model `kc`
                // A multi-slice DOT family (`DEPTH_MULTIPLE > 1` with `OUT_IS_ACC =
                // true`, i.e. the f32-output narrow twin) rounds the slice depth up to
                // the group multiple so an interior boundary never splits a depth-group:
                // a split group would depth-pad its tail (a zero pad-pair) mid-
                // contraction, regrouping the fused dot and rounding differently from
                // one panel. Only the final short tail is then padded, exactly as the
                // single-panel case. `next_multiple_of(1)` is the identity for every
                // other family, so nothing else moves
                let kc = if Fam::OUT_IS_ACC {
                    blk.kc.next_multiple_of(Fam::DEPTH_MULTIPLE)
                } else {
                    k
                };
                (kc.max(1), blk.nc.next_multiple_of(nr).max(nr))
            }
        };

        // Dot-product families (VNNI, vdpbf16ps) fold `Fam::DEPTH_MULTIPLE`
        // consecutive depth steps into one instruction, so every packed
        // micropanel's depth is rounded up to that multiple (the family's pack
        // depth-pads the tail). `1` for every other family => the
        // `next_multiple_of` calls below are identities and nothing changes
        let q_depth = Fam::DEPTH_MULTIPLE;
        let kc_pad_block = kc.next_multiple_of(q_depth);

        // The prepacked-RHS branch pads each panel's depth like the per-call
        // pack, but its `jc`-block offset assumes a single depth slice (its
        // `n_nt*pc` term would need a padded cumulative depth otherwise). A
        // `DEPTH_MULTIPLE > 1` family therefore may use prepack only when the
        // whole contraction is one slice (`kc >= k`), which is exactly the
        // mixed/dot families' `OUT_IS_ACC = false => kc = k`. This is a hard
        // `assert!` (not `debug_assert!`) because violating it makes the
        // consume read RHS micropanels at wrong byte offsets - a silent
        // miscompute. It is near-free: the `q_depth == 1` short-circuit means
        // every existing family skips it immediately
        assert!(
            q_depth == 1 || packed.is_none() || kc >= k,
            "prepacked-RHS with DEPTH_MULTIPLE > 1 requires a single depth slice (kc >= k)"
        );

        let n_mc = m.div_ceil(mc); // row macro-blocks (constant across panels)
        let n_nt_max = nc.div_ceil(nr);
        let n_jobs_max = n_mc * n_nt_max;
        let mnk = m.saturating_mul(n).saturating_mul(k);
        let n_threads = par.resolve(mnk, n_jobs_max);

        // Reuse-aware LHS pack decision: each worker handles roughly
        // `jobs_per_worker` column strips, all within one row block
        let jobs_per_worker = n_jobs_max.div_ceil(n_threads.max(1));
        let reuse_cols = jobs_per_worker.min(n_nt_max) * nr;
        // A column-major A (`rsa == 1`) is read in place by walking K with
        // stride `csa`; once `csa * sizeof(Lhs)` reaches about a memory page,
        // the strided read thrashes the TLB, so packing A into a contiguous
        // panel wins regardless of reuse and is redundancy-free here
        let pack_stride = cache::lhs_pack_stride_bytes();
        let strided_lhs = rsa == 1
            && csa
                .unsigned_abs()
                .saturating_mul(core::mem::size_of::<Fam::Lhs>())
                >= pack_stride;
        let want_pack_lhs =
            reuse_cols > tuning::lhs_pack_threshold() || strided_lhs || Fam::FORCE_PACK_LHS;

        // Shared-LHS pre-pack: on the parallel packed-A path, pack each
        // row-block's A panel once into a shared region (below) instead of
        // every worker that touches it re-packing. Gated to the packed path
        // (`rsa != 1 || want_pack_lhs`), to real parallelism (serial keeps the
        // unchanged per-worker path), and to a workload threshold (the
        // pre-pass adds a fork-join per depth slice that only pays at large
        // sizes; see `tuning::shared_lhs_mnk`)
        let shared_a =
            n_threads > 1 && (rsa != 1 || want_pack_lhs) && mnk >= tuning::shared_lhs_mnk();

        // A prepacked RHS is supplied whole in micropanel-major layout, so the
        // per-call B-pack is disabled and the compute region reads from the
        // caller's read-only buffer instead
        let prepacked = packed.is_some();

        // Adaptive RHS packing: B (read only via broadcast, so any layout
        // works unpacked) is packed once and reused across all `n_mc` row
        // blocks. The copy amortizes only when that reuse is high (large
        // `m`); otherwise B is read in place. Never packed here when the RHS
        // is already prepacked
        let pack_b = !prepacked && (m > tuning::rhs_pack_threshold() || Fam::FORCE_PACK_RHS);

        // One packing allocation. The LHS region count is `n_mc` when shared
        // (one slot per row-block, written once by the pre-pass) or
        // `n_threads` when per-worker (each worker owns a private scratch
        // slot). For square parallel problems `n_mc < n_threads`, so shared-A
        // uses *fewer* slots, not more. Checked: on the mixed-precision path
        // `kc == k`, and a broadcast (zero-stride) operand passes validation
        // with a logically huge `k`, so these products can overflow; a
        // wrapped size would under-allocate the workspace the pack then
        // writes past
        // Whether the A macro-panel is ever packed. The per-worker path packs a
        // tile only when `do_pack_lhs` holds; taken over every tile that reduces
        // to this predicate, because only the last row block can carry an
        // `mc_eff` that is not an `mr` multiple (`mc` is), and it does exactly
        // when `m` is not. The shared pre-pass runs only under
        // `rsa != 1 || want_pack_lhs`, already implied here, so this one
        // predicate covers every A-pack site. When it is false the A region is
        // reserved but never written, so skip reserving it entirely
        let need_a_pack = rsa != 1 || !m.is_multiple_of(mr) || want_pack_lhs;
        // The overflow guards run UNCONDITIONALLY, before the pack gating: a broadcast
        // operand passes validation with a logically huge `k` (the mixed single panel makes
        // `kc_pad_block == k`), and on a no-pack route (`need_a_pack` and `pack_b` both
        // false - reachable when nothing forces packing and reuse is low) skipping the
        // sizing would also skip the "too large" abort, sending an absurd `k` into the
        // in-place loops to spin ~forever instead of failing closed. The products are what
        // packing WOULD allocate, so the bound is identical on every route
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
        let (a_per_region, a_regions) = if need_a_pack {
            (a_full_region, if shared_a { n_mc } else { n_threads })
        } else {
            (0, 0)
        };
        let b_elems = if pack_b { b_full_elems } else { 0 };
        // Reserve pack scratch only when a side actually packs. When neither
        // does, hand out null bases (never dereferenced: every tile reads A in
        // place and B is read in place or from the prepacked buffer), so an
        // all-in-place workload never grows the pool
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
        // Prepacked: read from the caller buffer; else from the per-call scratch
        let b_base = match packed {
            Some((pb, _, _)) => Ptr(pb as *mut Fam::Rhs),
            None => Ptr(regions.b_base as *mut Fam::Rhs),
        };

        let a = Ptr(a as *mut Fam::Lhs);
        let b = Ptr(b as *mut Fam::Rhs);
        let c = Ptr(c);
        let ash = alpha_status(alpha);

        // Running element offset of the current `jc` block inside a prepacked
        // RHS buffer: each block holds `n_nt * nr * k` elements (every padded
        // column appears once per depth row). Unused when not prepacked
        let mut jc_off = 0usize;
        let mut jc = 0;
        while jc < n {
            let nc_eff = core::cmp::min(nc, n - jc);
            let n_nt = nc_eff.div_ceil(nr);

            // Job count and cursor grain depend only on this column panel and
            // the worker count - invariant across the depth (`pc`) loop, so
            // compute them once here. The packed-LHS path (whole-row-block
            // chunks) is split for load balance; the general path uses the
            // shared `job_grain` oversample. See `packed_block_grain`
            let n_jobs = n_mc * n_nt;
            let grain = if shared_a {
                // A is pre-packed once per block below, so whole-block
                // chunking no longer buys pack reuse - use the fine grain for
                // the best balance
                parallel::job_grain(n_jobs, n_threads)
            } else if (rsa != 1 || want_pack_lhs) && n_mc >= n_threads {
                parallel::packed_block_grain(n_nt, n_mc, n_threads)
            } else {
                parallel::job_grain(n_jobs, n_threads)
            };

            let mut pc = 0;
            while pc < k {
                let kc_eff = core::cmp::min(kc, k - pc);
                // Depth-padded panel stride for dot families (identity when `q_depth == 1`)
                let kc_eff_pad = kc_eff.next_multiple_of(q_depth);
                let first = pc == 0;
                // Whether this is the final depth slice: the fused epilogue
                // applies only on the last panel (raw `Acc` partials store on
                // the earlier ones). For an `OUT_IS_ACC = false` family
                // (`kc = k`) this is always the single slice
                let last = pc + kc_eff >= k;
                let beta_eff = if first { beta } else { Fam::Acc::ONE };
                let bst = if first {
                    beta_status(beta)
                } else {
                    BetaStatus::One
                };

                // Pack the RHS macro-panel in parallel (when packing): workers
                // pull NR-wide column panels from a shared cursor. The
                // `for_each_worker` join below is the write-before-read
                // barrier the compute region depends on - packed B is the
                // *one* buffer shared (non-disjoint) across all compute
                // workers, so every panel must be written here before any
                // worker reads it. Fusing this region into the compute loop,
                // or moving packing inside it, would reintroduce a data race
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

                // Shared-LHS pre-pack: pack each row-block's A panel once
                // into `a_base[ic_idx]`. The `for_each_worker` join is the
                // write-before-read barrier the compute region depends on -
                // same discipline as the packed-B region above. Workers pull
                // disjoint `ic` ranges and write disjoint, ALIGN-rounded
                // slots, so there is no intra-region race
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
                    // Force whole-struct capture of the `Send + Sync` pointer
                    // shims; edition-2024 (RFC 2229) closures otherwise
                    // capture the inner `*mut` fields disjointly, which are
                    // not `Sync`, so the closure would fail the bound needed
                    // to move into the rayon workers
                    let (a, b, c, a_base, b_base, epi) = (a, b, c, a_base, b_base, epi);
                    // Per-worker scratch slot (per-worker path only). On the
                    // shared-A path `tid` may exceed `n_mc`, so
                    // `a_base + tid*a_stride` would be out-of-bounds pointer
                    // arithmetic even though it is never read - use a null
                    // base there instead
                    let a_buf = if shared_a {
                        core::ptr::null_mut::<Fam::Lhs>()
                    } else {
                        a_base.0.add(tid * a_stride)
                    };
                    let mut scratch = [const { MaybeUninit::<Fam::Acc>::uninit() }; SCRATCH_LEN];
                    let scratch_ptr = scratch.as_mut_ptr() as *mut Fam::Acc;

                    // Cached LHS pack state for the current row block
                    // `packed_a` is carried explicitly (not re-derived as
                    // `a_cs == mr`, which could collide for a self-overlapping
                    // column-major A whose `csa` equals `mr` and then read the
                    // in-place A with the packed address formula)
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
                                    // Read the block's A panel that the
                                    // pre-pass packed once (same bytes, same
                                    // packed read formula as the per-worker
                                    // case => identical packed input, so the
                                    // result reproduces it)
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
                                // Depth-padded panel stride (identity for
                                // `q_depth == 1`). The single-slice guard
                                // above keeps `pc == 0` for any `q_depth > 1`
                                // family, so `n_nt * pc` needs no padding
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

                            // Process the whole column strip in the ISA's target-feature context
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
                                    // `ic + ir` and `col` are the tile's
                                    // origin in the oriented frame, so a
                                    // per-row/per-col bias resolves its
                                    // absolute base; `last` gates the
                                    // once-per-element apply
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
            // Each prepacked `jc` block holds `n_nt` panels of `nr x kpad(k)`
            // (single slice for any depth-padded family - see the guard
            // above). `kpad == k` when q_depth == 1
            jc_off += n_nt * nr * k.next_multiple_of(q_depth);
            jc += nc;
        }
    }
}
