//! Generic small-`k` path: skinny / low-depth GEMM (gevv, rank-`k`, tall-skinny).
//!
//! At tiny `k` the whole product is one depth panel, so the register-tiling driver's
//! machinery — the cache blocking model, the workspace allocation, and above all the
//! A/B *packing* — is pure setup with nothing to amortize: every packed element is read
//! once. This route computes `C <- α·A·B + β·C` directly over the family's
//! [`microkernel`](crate::kernel::KernelFamily::microkernel), reading A/B **in place**
//! (unpacked) with `kc = k` in a single pass, so it inherits the family's encapsulated
//! widen / bias / conj / rounding for free while skipping all of that setup.
//!
//! Because the whole contraction is one panel, each output element is a single fixed-order
//! reduction: the output-tile partitioning across workers touches disjoint C tiles and adds
//! no cross-thread reduction, so the result is **reproducible** and bit-identical to the
//! serial run for any worker count.
//!
//! The microkernel requires LHS rows to be unit-stride ([`crate::kernel`]), i.e. a
//! column-major A (`rsa == 1`). When that does not hold, packing rarely amortizes at tiny
//! `k`, so this route defers to the general [`driver::run`] (still correct) rather than
//! packing A itself.

use core::mem::MaybeUninit;

use crate::driver::{self, alpha_status, beta_status};
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::kernel::{KernelFamily, MAX_MR, SCRATCH_LEN};
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::simd::{KernelSimd, SimdOps};
use crate::workspace::Workspace;

/// Largest `k` the in-place route handles: it bounds the stack panel that zero-pads the
/// bottom partial row-tile (`MAX_MR × SMALL_K_MAX` LHS elements). Comfortably above the
/// calibrated `small_k_threshold` (~16); if the threshold is set past it, the excess `k`
/// falls back to the driver — which is the faster path there anyway.
const SMALL_K_MAX: usize = 32;

/// Run a small-`k` GEMM with the given family, ISA token, and microtile geometry. The
/// dispatch layer routes here (after orientation normalization) only when `k` is small;
/// preconditions match [`driver::run`] (`m, n, k > 0`, `alpha != 0`, orientation-normalized).
///
/// The zero-cost [`Identity`] forwarder over [`run_epi`]: with `E = Identity` every epilogue
/// hook const-folds away (`microkernel_epi` becomes `microkernel`), so the public signature —
/// and every byte this route stores — is unchanged for all existing callers.
///
/// # Safety
/// All pointers must be valid for the regions implied by the strides/sizes, and `C` must
/// not alias `A` or `B`.
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
    // SAFETY: forwarded to `run_epi` with the zero-cost `Identity` epilogue — the exact code
    // path (and byte stream) this route stored before the epilogue seam existed.
    unsafe {
        run_epi::<Fam, S, Identity, MR_REG, NR>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity, par, ws,
        )
    }
}

/// Run a small-`k` GEMM applying the fused [`Epilogue`] `E` to each output element as the tile
/// is stored, instead of materializing the raw product and mapping it afterward. [`run`] is
/// exactly this with `E = Identity`; a non-identity `E` changes only the per-tile store, so the
/// blocking / partition / read pattern is identical and the fused result equals this route's
/// plain output followed by the same scalar map. The whole product is one depth panel
/// (`kc = k`), so `last_k` is structurally true and the epilogue fires exactly once per element.
///
/// # Safety
/// As [`run`], plus `epi`'s interior pointers must be valid for the (oriented) problem's `m`/`n`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn run_epi<Fam, S, E, const MR_REG: usize, const NR: usize>(
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
    unsafe {
        // Defer to the general driver (still correct) when the in-place route does not apply
        // — packing never amortizes at tiny `k`, so there is nothing to gain by handling these
        // here. A `FORCE_PACK_*` family transforms operands while packing (the complex conj
        // planar layout) and so cannot be read in place; the microkernel reads the LHS panel
        // with unit-stride rows, so an in-place A needs `rsa == 1` (column-major); and `k`
        // past `SMALL_K_MAX` both overflows the bottom-tile pad panel below and is where the
        // driver already wins. `FORCE_PACK_*` is a compile-time const, so it folds away for
        // the in-place families. The driver applies the *same* epilogue, so the fused contract
        // holds identically on the deferred shapes.
        if Fam::FORCE_PACK_LHS || Fam::FORCE_PACK_RHS || rsa != 1 || k > SMALL_K_MAX {
            driver::run_epilogue::<Fam, S, E, MR_REG, NR>(
                simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, epi, par, ws,
            );
            return;
        }

        // `E: Copy` (an `Epilogue` supertrait): copy it out of the borrow so each `move` worker
        // closure captures it by value (the same capture discipline the driver uses for `Ptr`).
        let epi = *epi;

        let lanes = <S as SimdOps<Fam::Acc>>::LANES;
        let mr = MR_REG * lanes;
        let nr = NR;
        debug_assert!(mr * nr <= SCRATCH_LEN, "microtile exceeds scratch capacity");
        debug_assert!(mr <= MAX_MR, "microtile rows exceed MAX_MR");

        let n_row_tiles = m.div_ceil(mr);
        let n_col_tiles = n.div_ceil(nr);
        let n_tiles = n_row_tiles * n_col_tiles;
        // The bottom row-tile is partial when `m` is not a multiple of `mr`. The microkernel
        // always loads the full `mr` rows, so reading that tile's A in place would run past
        // A's end; it must be zero-padded into a packed panel instead (see the body).
        let last_ic = (n_row_tiles - 1) * mr;
        let bottom_eff = m - last_ic; // in (0, mr]
        let bottom_partial = bottom_eff < mr;

        let ash = alpha_status(alpha);
        let bst = beta_status(beta);

        // Bandwidth-capped worker count: this shape is memory-bound (the `m·n` C write
        // dominates at small `k`), so use the bandwidth model, not the compute ramp. Bytes
        // are the exact operand traffic; the output tiles are the partitionable units.
        let bytes = m
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
        let n_threads = par.resolve_bandwidth(bytes, n_tiles);

        let a = Ptr(a as *mut Fam::Lhs);
        let b = Ptr(b as *mut Fam::Rhs);
        let c = Ptr(c);

        // Compute the flat tile range `[q_start, q_end)` of the `n_row_tiles × n_col_tiles`
        // grid, iterated column-tile-outer so a worker's consecutive tiles share a C column
        // block (contiguous stores for a column-major C). One `kc = k` panel per tile.
        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c, epi) = (a, b, c, epi);
            let a = a.0 as *const Fam::Lhs;
            let b = b.0 as *const Fam::Rhs;
            let c = c.0;
            simd.vectorize(|| {
                let mut scratch = [const { MaybeUninit::<Fam::Acc>::uninit() }; SCRATCH_LEN];
                let scratch_ptr = scratch.as_mut_ptr() as *mut Fam::Acc;

                // Zero-padded pad panel for the bottom partial row-tile: the microkernel loads
                // the full `mr` rows every depth column, so that tile (fewer than `mr` real
                // rows) must be packed instead of read in place. Its A block is the same for
                // every column tile, so pack it once and reuse. `bottom_eff < mr` is only true
                // when packing is needed; `mr·k <= MAX_MR·SMALL_K_MAX` bounds the buffer.
                let mut pad = [const { MaybeUninit::<Fam::Lhs>::uninit() }; MAX_MR * SMALL_K_MAX];
                let pad_base = if bottom_partial {
                    Fam::pack_lhs(
                        pad.as_mut_ptr() as *mut Fam::Lhs,
                        a.offset(last_ic as isize * rsa),
                        rsa,
                        csa,
                        bottom_eff,
                        k,
                        mr,
                    );
                    pad.as_ptr() as *const Fam::Lhs
                } else {
                    core::ptr::null()
                };

                for q in q_start..q_end {
                    let jt = q / n_row_tiles;
                    let it = q % n_row_tiles;
                    let ic = it * mr;
                    let jc = jt * nr;
                    let mr_eff = core::cmp::min(mr, m - ic);
                    let nr_eff = core::cmp::min(nr, n - jc);
                    // Full row tiles read A in place (rows unit-stride, depth stride `csa`);
                    // the bottom partial tile reads the zero-padded pad panel (packed, depth
                    // stride `mr`) so no read runs past A. RHS/C are always read in place.
                    let (apan, a_cs) = if mr_eff < mr {
                        (pad_base, mr as isize)
                    } else {
                        (a.offset(ic as isize * rsa), csa)
                    };
                    let bpan = b.offset(jc as isize * csb);
                    let cptr = c.offset(ic as isize * rsc + jc as isize * csc);
                    // `ic`/`jc` are the tile origin in the oriented frame (a per-row/per-col
                    // bias resolves its absolute base); `kc = k` is the single depth panel, so
                    // `last_k = true` — the epilogue fires exactly once per element here.
                    Fam::microkernel_epi::<S, E, MR_REG, NR>(
                        simd,
                        k,
                        alpha,
                        beta,
                        ash,
                        bst,
                        apan,
                        a_cs,
                        bpan,
                        rsb,
                        csb,
                        cptr,
                        rsc,
                        csc,
                        mr_eff,
                        nr_eff,
                        ic,
                        jc,
                        true,
                        &epi,
                        scratch_ptr,
                    );
                }
            });
        };

        if n_threads <= 1 {
            body(0, n_tiles);
            return;
        }

        // Output-partitioned parallel sweep: workers pull disjoint flat-tile ranges from a
        // shared cursor. No cross-worker reduction (each tile is one full `k`-pass), so no
        // barrier and no perturbation of the per-element summation order.
        let cur = JobCursor::new(n_tiles, parallel::job_grain(n_tiles, n_threads));
        parallel::for_each_worker(n_threads, |_tid| {
            while let Some((s, e)) = cur.next_chunk() {
                body(s, e);
            }
        });
    }
}
