//! SIMD abstraction (layer L0): the load-bearing wall of the library.
//!
//! This module is **self-contained**: it depends only on [`crate::scalar`] and
//! `core`, never on the kernel/driver/cache layers above it. That zero
//! reverse-dependency property is deliberate so the whole module could later be
//! split into its own crate unchanged.
//!
//! # The two traits
//!
//! * [`Simd`] — an ISA *token* (a zero-sized type like [`Fma`]). It is not
//!   parameterized by element type. Its sole job is [`Simd::vectorize`], the
//!   `#[target_feature]` boundary (see below).
//! * [`SimdOps<T>`] — the *thick* per-element-type vocabulary of a token: the
//!   register type, lane count, and every primitive the microkernel needs
//!   (load/store/broadcast/mul/add/fma/reduce). Because the token and the
//!   element type are decoupled, `LANES` varies with the `(ISA, T)` pair
//!   (`f32`@FMA = 8, `f32`@AVX-512 = 16, `f64` halved).
//!
//! This is the answer to matrixmultiply's thin-trait trap: *every* primitive the
//! kernel needs is here, so the kernel is **one** generic function over all ISAs.
//! Adding an ISA = a new token + its `SimdOps` impls + one dispatch line.
//!
//! # `#[target_feature]` correctness
//!
//! AVX/AVX-512 intrinsics must be code-generated in a context where the feature
//! is enabled. CPU support is decided at *runtime* (by the dispatch layer), so
//! we cannot put a fixed `#[target_feature]` on the generic kernel. Instead each
//! token's [`Simd::vectorize`] runs a closure inside a tiny
//! `#[target_feature]`-annotated function; the closure (and the `#[inline]`
//! primitives it calls) inline into that function, so all intrinsics land in a
//! feature-enabled codegen context. This is the proven pulp/faer pattern, and it
//! works for both the serial path and rayon worker closures.

use crate::scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod fma;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::avx512::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::fma::Fma;
#[cfg(target_arch = "aarch64")]
pub use self::neon::Neon;
pub use self::scalar::ScalarTok;

/// The SIMD capability an ISA token must provide to drive a [`crate::kernel::KernelFamily`]
/// with input types `L`/`R`, accumulator `A`, and output `O`: accumulate in `A`
/// (the [`SimdOps<A>`] supertrait) and move family inputs/outputs into and out of the
/// `A`-typed registers, **widening on load and narrowing on store** when the element
/// types are narrower than `A`.
///
/// This seam makes **mixed precision** (`A != L`) work without a per-type branch in
/// the driver. The homogeneous case (`L = R = A = O`) is covered by a blanket impl
/// forwarding to plain [`SimdOps`] load/splat/store; a narrow family (`f16`/`bf16`
/// inputs, `f32` accumulator) adds an ISA impl whose `load_*` widens and `store_out`
/// narrows. The all-equal blanket and a mixed impl (`L != A`) can never overlap.
pub trait KernelSimd<L: Scalar, R: Scalar, A: Scalar, O: Scalar>: SimdOps<A> {
    /// Load `LANES` LHS values and widen to one `A` register (plain load if `L == A`).
    ///
    /// # Safety
    /// `p` valid for `LANES` reads; run inside this token's [`Simd::vectorize`].
    unsafe fn load_lhs(self, p: *const L) -> <Self as SimdOps<A>>::Reg;
    /// Widen one RHS scalar and broadcast to all `A` lanes (plain splat if `R == A`).
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn splat_rhs(self, v: R) -> <Self as SimdOps<A>>::Reg;
    /// Load `LANES` output values and widen to one `A` register, for the `beta != 0`
    /// read of `C` (plain load if `O == A`).
    ///
    /// # Safety
    /// `p` valid for `LANES` reads; run inside [`Simd::vectorize`].
    unsafe fn load_out(self, p: *const O) -> <Self as SimdOps<A>>::Reg;
    /// Narrow one `A` register to `LANES` output values and store (plain store if
    /// `O == A`; rounds to nearest-even when narrowing).
    ///
    /// # Safety
    /// `p` valid for `LANES` writes; run inside [`Simd::vectorize`].
    unsafe fn store_out(self, p: *mut O, v: <Self as SimdOps<A>>::Reg);
}

/// Homogeneous blanket: when every family type is the accumulator type, the
/// widen/narrow ops are plain [`SimdOps`] load/splat/store, so any homogeneous
/// family (e.g. `FloatGemm<f32>`/`FloatGemm<f64>`) needs zero per-ISA code.
impl<A: Scalar, S: SimdOps<A>> KernelSimd<A, A, A, A> for S {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.splat(v) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const A) -> <S as SimdOps<A>>::Reg {
        unsafe { self.loadu(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut A, v: <S as SimdOps<A>>::Reg) {
        unsafe { self.storeu(p, v) }
    }
}

/// An ISA token: a zero-sized marker carrying a set of target features.
///
/// The only behaviour is [`Simd::vectorize`], which establishes the
/// `#[target_feature]` codegen context for everything it runs.
pub trait Simd: Copy + Send + Sync + 'static {
    /// Run `f` with this ISA's target features enabled.
    ///
    /// # Safety
    /// The caller must guarantee the current CPU actually supports this token's
    /// features (the runtime dispatcher does this once).
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R;
}

/// The thick SIMD vocabulary for element type `T` under ISA token `Self`.
///
/// All methods are `unsafe`: they assume (a) the target feature is enabled in
/// the current codegen context (guaranteed by running inside
/// [`Simd::vectorize`]) and (b) any pointers are valid for the access. They are
/// all `#[inline(always)]` in the impls so the intrinsics fold into the kernel.
pub trait SimdOps<T: Scalar>: Simd {
    /// The SIMD register type holding [`Self::LANES`] values of `T`.
    type Reg: Copy;
    /// Number of `T` lanes per register.
    const LANES: usize;
    /// Natural buffer alignment for this ISA in bytes (e.g. 32 for AVX2, 64 for
    /// AVX-512). Packed buffers are aligned to this.
    const ALIGN: usize;
    /// Whether this ISA has a hardware **lane-indexed FMA** — broadcasting a
    /// multiplier straight from a vector lane in one fused instruction (NEON
    /// `vfmaq_laneq`). When `true`, the microkernel takes the lane path via
    /// [`Self::fma_bvec`] for packed RHS, loading a block of `LANES` B columns
    /// as one vector instead of issuing a `splat` load per column. The default
    /// is `false`: per-column `splat` + FMA, which on x86 the assembler already
    /// folds into a broadcast-from-memory operand, so the lane path is no win
    /// there.
    const LANE_FMA: bool = false;

    /// A register of all zeros.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn zero(self) -> Self::Reg;
    /// Broadcast a scalar into every lane.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn splat(self, v: T) -> Self::Reg;
    /// Aligned load of [`Self::LANES`] contiguous values.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` reads and aligned to [`Self::ALIGN`].
    unsafe fn load(self, p: *const T) -> Self::Reg;
    /// Unaligned load of [`Self::LANES`] contiguous values.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` reads.
    unsafe fn loadu(self, p: *const T) -> Self::Reg;
    /// Aligned store.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` writes and aligned to [`Self::ALIGN`].
    unsafe fn store(self, p: *mut T, v: Self::Reg);
    /// Unaligned store.
    ///
    /// # Safety
    /// `p` must be valid for `LANES` writes.
    unsafe fn storeu(self, p: *mut T, v: Self::Reg);
    /// Lane-wise multiply.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise add.
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    /// Lane-wise fused multiply-add `a * b + c` (true FMA where available).
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg;
    /// Horizontal sum of all lanes (used by gemv / dot epilogues).
    ///
    /// # Safety
    /// See the trait-level note.
    unsafe fn reduce_sum(self, v: Self::Reg) -> T;

    /// Accumulate one contiguous block of `B` columns (loaded as the single
    /// register `bvec`) against the `MR_REG` already-loaded `A` registers,
    /// broadcasting each B lane: for `l in 0..acc.len()` and `i in 0..MR_REG`,
    /// `acc[l][i] = a_regs[i] * bvec[l] + acc[l][i]`. `acc.len()` must be
    /// `<= LANES`.
    ///
    /// This is the fused inner step of the lane-indexed kernel path, taken only
    /// when [`Self::LANE_FMA`] is set. The default implementation broadcasts
    /// each lane via a store + [`Self::splat`] (correct for any ISA, but no
    /// faster than the plain `splat` path); lane-capable ISAs override it with a
    /// single hardware lane-indexed FMA. The result is bit-identical to the
    /// `splat` path — same fused `a*b + c`.
    ///
    /// # Safety
    /// See the trait-level note; `acc.len()` must be `<= LANES` and `a_regs`
    /// valid for `MR_REG` reads.
    #[inline(always)]
    unsafe fn fma_bvec<const MR_REG: usize>(
        self,
        a_regs: &[Self::Reg; MR_REG],
        bvec: Self::Reg,
        acc: &mut [[Self::Reg; MR_REG]],
    ) {
        debug_assert!(acc.len() <= Self::LANES);
        unsafe {
            // Spill the B-vector to the stack, then broadcast each lane. 16 is
            // the widest LANES of any ISA (AVX-512 f32), so it always fits.
            let mut buf = [T::ZERO; 16];
            self.storeu(buf.as_mut_ptr(), bvec);
            for l in 0..acc.len() {
                let bl = self.splat(buf[l]);
                for i in 0..MR_REG {
                    acc[l][i] = self.mul_add(a_regs[i], bl, acc[l][i]);
                }
            }
        }
    }

    /// Accumulate one **full** `MR_REG × NR` microtile over `kc` depth steps into
    /// the register-resident `acc` (pre-zeroed by the caller):
    /// `acc[j][i] += A[p][i] · B[p][j]` for every `p in 0..kc`, in **ascending
    /// `p`** with a fused multiply-add. This is the GEMM inner loop and the single
    /// hottest piece of the library.
    ///
    /// `a` points at the LHS micropanel (`a_cs` = depth stride; rows are unit
    /// stride, `MR_REG` vectors of `LANES`); `b` at the RHS panel (`b_rs` depth
    /// stride, `b_cs` column stride — `(nr, 1)` packed or `(rsb, csb)` unpacked).
    ///
    /// The **default** is the portable per-step schedule: one broadcast (`splat`)
    /// per RHS column, or the lane-indexed fast path ([`Self::fma_bvec`]) when
    /// [`Self::LANE_FMA`] is set, the RHS block is contiguous (`b_cs == 1`), and `NR`
    /// is a multiple of `LANES` (so each `LANES`-wide column block is whole);
    /// otherwise the broadcast path runs.
    ///
    /// **Keep the default on any out-of-order core.** On a wide OoO core LLVM already
    /// lowers it to the canonical register-blocked kernel that saturates the FMA
    /// pipes — it schedules the next step's loads in among the FMAs and unrolls the
    /// `kc` loop on its own.
    ///
    /// **Override only for a target whose generated schedule genuinely stalls** in a
    /// way LLVM will not fix on its own — e.g. an **in-order / narrow-OoO** core, where
    /// explicitly hoisting the next step's loads (the textbook software pipeline) pays
    /// because the hardware cannot reorder, or a **scalable-vector** ISA (SVE/SME, RVV)
    /// whose length is not a compile-time `LANES`, so the fixed-width loop must be
    /// rewritten. Both still do a per-element fused `a·b + c` in ascending `p`, so they
    /// can satisfy the bit-identity contract below. Instructions that *reshape* the
    /// accumulation rounding itself (matrix / dot — `bfmmla`, `sdot`) cannot, so they
    /// are out of scope for this seam: adopting them means revisiting the driver's
    /// determinism contract, not just overriding this method. Before keeping any
    /// override, *prove it pays*: check the disassembly for spills, confirm it stays
    /// bit-identical to the default, and benchmark it — do not assume a hand schedule
    /// helps.
    ///
    /// Any override **must** preserve the ascending-`p`, fused `a·b + c` order so
    /// results stay **bit-identical** to the default (the driver's determinism
    /// contract: full and edge tiles of the same matrix must round the same way).
    /// Software pipelining reorders *loads*, never the arithmetic, so it is legal.
    /// Called only for full tiles (`nr_eff == NR`); partial column tiles stay on the
    /// microkernel's edge path.
    ///
    /// # Safety
    ///
    /// `a` valid for `MR_REG·LANES` rows × `kc` depth at stride `a_cs`; `b` valid
    /// for `NR` cols × `kc` depth at strides `b_rs`/`b_cs`; `acc` pre-initialized.
    /// Must run inside this token's [`Simd::vectorize`] context.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn accumulate_tile<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const T,
        a_cs: isize,
        b: *const T,
        b_rs: isize,
        b_cs: isize,
        acc: &mut [[Self::Reg; MR_REG]; NR],
    ) {
        let lanes = Self::LANES;
        unsafe {
            if Self::LANE_FMA && b_cs == 1 && NR.is_multiple_of(lanes) {
                // Lane-indexed fast path (NEON): load each contiguous `lanes`-wide
                // RHS block as one vector and broadcast its lanes via a fused
                // lane-indexed FMA, replacing `NR` per-column splats.
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [Self::Reg; MR_REG] =
                        core::array::from_fn(|i| self.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for jb in (0..NR).step_by(lanes) {
                        let bvec = self.loadu(pb.add(jb));
                        self.fma_bvec(&a_regs, bvec, &mut acc[jb..jb + lanes]);
                    }
                }
            } else {
                // Splat path: one broadcast per RHS column. Correct for any `b_cs`
                // (packed or unpacked) and the only full-tile path for ISAs without
                // a lane FMA. The const-bounded `j` loop fully unrolls.
                for p in 0..kc {
                    let pa = a.offset(p as isize * a_cs);
                    let a_regs: [Self::Reg; MR_REG] =
                        core::array::from_fn(|i| self.loadu(pa.add(i * lanes)));
                    let pb = b.offset(p as isize * b_rs);
                    for j in 0..NR {
                        let bj = self.splat(*pb.offset(j as isize * b_cs));
                        for i in 0..MR_REG {
                            acc[j][i] = self.mul_add(a_regs[i], bj, acc[j][i]);
                        }
                    }
                }
            }
        }
    }
}
