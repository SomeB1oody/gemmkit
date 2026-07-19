//! Small-`k` route: `C <- alpha*A*B + beta*C` for a shape whose contraction depth is too
//! shallow for packing to pay off
//!
//! The register-tiling driver packs A/B into micropanels so the microkernel walks a
//! contiguous depth axis; that pack is a fixed cost per element, amortized over the reuse
//! the driver's blocking gets out of it. At small `k` there is only one depth panel and
//! every packed element is read exactly once, so the pack buys nothing and is pure
//! overhead. This route instead calls the family's
//! [`microkernel`](crate::kernel::KernelFamily::microkernel) directly over unpacked A/B,
//! `kc = k` in a single pass, inheriting whatever widen/bias/conj/rounding behavior the
//! family's microkernel already has
//!
//! Dispatch only reaches this route below the (arch-tuned) `small_k_threshold`, and only
//! for a family whose LHS the microkernel can read in place: `rsa == 1` (column-major A)
//! and no forced packing transform (complex conj-planar, mixed narrow repacking). When
//! either fails, [`run_epi`] falls back to [`driver::run_epilogue`] instead, which is
//! still correct there and is the faster route once `k` grows past the in-place regime
//!
//! Every output tile is one full-depth reduction computed by a single worker, so splitting
//! tiles across workers adds no cross-thread reduction step: the result is bit-identical to
//! the serial run at any worker count

use core::mem::MaybeUninit;

use crate::driver::{self, alpha_status, beta_status};
use crate::kernel::epilogue::{Epilogue, Identity};
use crate::kernel::{KernelFamily, MAX_MR, SCRATCH_LEN};
use crate::parallel::{self, JobCursor, Parallelism, Ptr};
use crate::simd::{KernelSimd, SimdOps};
use crate::workspace::Workspace;

/// Largest `k` this route accepts in place. Bounds the stack pad buffer used for a partial
/// bottom row-tile (`MAX_MR * SMALL_K_MAX` elements); `k` past this falls back to the
/// driver, which is already the better choice there
const SMALL_K_MAX: usize = 32;

/// Small-`k` GEMM, plain (non-fused) output. Thin [`Identity`] wrapper over [`run_epi`]:
/// with `E = Identity` every epilogue hook const-folds away, so this is exactly the code
/// this route ran before the epilogue mechanism existed
///
/// # Safety
/// All pointers must be valid for the regions implied by the strides/sizes, and `C` must
/// not alias `A` or `B`
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
    // SAFETY: forwards to `run_epi` with `Identity`, which const-folds to a no-op hook
    unsafe {
        run_epi::<Fam, S, Identity, MR_REG, NR>(
            simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, &Identity, par, ws,
        )
    }
}

/// Small-`k` GEMM with a fused [`Epilogue`] `E` applied at each element's store. Since the
/// whole contraction is a single `kc = k` panel, every element is written exactly once, so
/// `last_k` is unconditionally true and the epilogue fires exactly once per element - the
/// fused output equals plain [`run`] followed by the same scalar map
///
/// # Safety
/// As [`run`], plus `epi`'s interior pointers must be valid for the (oriented) problem's `m`/`n`
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
        // A family with FORCE_PACK_LHS/RHS transforms operands while packing (complex's
        // conj-planar layout, mixed's narrow repack), so reading A unpacked would be wrong,
        // not just slower. The microkernel also needs LHS rows unit-stride (rsa == 1) to
        // read A in place at all. Past SMALL_K_MAX the pad buffer below would overflow, and
        // the driver already wins there anyway. FORCE_PACK_* is a compile-time const, so it
        // folds away entirely for families that never take this branch
        if Fam::FORCE_PACK_LHS || Fam::FORCE_PACK_RHS || rsa != 1 || k > SMALL_K_MAX {
            driver::run_epilogue::<Fam, S, E, MR_REG, NR>(
                simd, m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, epi, par, ws,
            );
            return;
        }

        // Epilogue is Copy; move a value copy into the worker closures below
        let epi = *epi;

        let lanes = <S as SimdOps<Fam::Acc>>::LANES;
        let mr = MR_REG * lanes;
        let nr = NR;
        debug_assert!(mr * nr <= SCRATCH_LEN, "microtile exceeds scratch capacity");
        debug_assert!(mr <= MAX_MR, "microtile rows exceed MAX_MR");

        let n_row_tiles = m.div_ceil(mr);
        let n_col_tiles = n.div_ceil(nr);
        let n_tiles = n_row_tiles * n_col_tiles;
        // Row-tile origin and height of the bottom (possibly partial) row-tile
        let last_ic = (n_row_tiles - 1) * mr;
        let bottom_eff = m - last_ic; // in (0, mr]
        let bottom_partial = bottom_eff < mr;

        let ash = alpha_status(alpha);
        let bst = beta_status(beta);

        // This shape is memory-bound (m*n C writes dominate at small k), so size the worker
        // count off measured byte traffic rather than the compute-ramp heuristic
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

        // One kc = k panel per tile; column-tile-outer flat order so a worker's consecutive
        // tiles share a C column block (contiguous stores for column-major C)
        let body = move |q_start: usize, q_end: usize| {
            let (a, b, c, epi) = (a, b, c, epi);
            let a = a.0 as *const Fam::Lhs;
            let b = b.0 as *const Fam::Rhs;
            let c = c.0;
            simd.vectorize(|| {
                let mut scratch = [const { MaybeUninit::<Fam::Acc>::uninit() }; SCRATCH_LEN];
                let scratch_ptr = scratch.as_mut_ptr() as *mut Fam::Acc;

                // The microkernel always loads a full mr rows regardless of the tile's real
                // height, so reading the bottom partial tile's A in place would run past A's
                // end. Pack it once, zero-padded to mr rows, and reuse it for every column
                // tile (its A block does not depend on the column)
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
                    // A full row-tile reads A in place (rows unit-stride, depth stride csa);
                    // the bottom partial tile reads the padded panel instead (depth stride
                    // mr, since it is packed micropanel-major)
                    let (apan, a_cs) = if mr_eff < mr {
                        (pad_base, mr as isize)
                    } else {
                        (a.offset(ic as isize * rsa), csa)
                    };
                    let bpan = b.offset(jc as isize * csb);
                    let cptr = c.offset(ic as isize * rsc + jc as isize * csc);
                    // ic/jc: tile origin in the oriented frame, for a per-row/per-col bias
                    // last_k is always true here since kc = k is the only depth panel
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

        // Workers pull disjoint flat-tile ranges from a shared cursor; each tile is a
        // complete k-reduction owned by one worker, so no cross-worker reduction and no
        // barrier is needed
        let cur = JobCursor::new(n_tiles, parallel::job_grain(n_tiles, n_threads));
        parallel::for_each_worker(n_threads, |_tid| {
            while let Some((s, e)) = cur.next_chunk() {
                body(s, e);
            }
        });
    }
}
