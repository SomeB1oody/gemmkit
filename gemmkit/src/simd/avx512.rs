//! AVX-512 ISA token (x86 / x86-64).
//!
//! `f32` → 512-bit registers (16 lanes), `f64` → 512-bit (8 lanes). Requires the
//! stable AVX-512 intrinsics (Rust 1.89+).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(any(feature = "half", feature = "int8"))]
use super::KernelSimd;
use super::{Simd, SimdOps};

/// AVX-512 (foundation) ISA token.
#[derive(Copy, Clone, Default)]
pub struct Avx512;

impl Simd for Avx512 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: dispatcher guarantees avx512f support; `inner` sets the
        // codegen context and `f` inlines into it.
        unsafe { inner(f) }
    }
}

impl SimdOps<f32> for Avx512 {
    type Reg = __m512;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> Self::Reg {
        unsafe { _mm512_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f32) -> Self::Reg {
        unsafe { _mm512_load_ps(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> Self::Reg {
        unsafe { _mm512_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm512_store_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: Self::Reg) {
        unsafe { _mm512_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `_mm512_fnmadd_ps(a, b, c)` == `c - a*b`.
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // x86 MAXPS returns the second operand on an unordered compare (NaN `a` -> `b`).
        unsafe { _mm512_max_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_min_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f32 {
        unsafe { _mm512_reduce_add_ps(v) }
    }
}

impl SimdOps<f64> for Avx512 {
    type Reg = __m512d;
    const LANES: usize = 8;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> Self::Reg {
        unsafe { _mm512_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f64) -> Self::Reg {
        unsafe { _mm512_set1_pd(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f64) -> Self::Reg {
        unsafe { _mm512_load_pd(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f64) -> Self::Reg {
        unsafe { _mm512_loadu_pd(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm512_store_pd(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f64, v: Self::Reg) {
        unsafe { _mm512_storeu_pd(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_mul_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_add_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: Self::Reg, b: Self::Reg, c: Self::Reg) -> Self::Reg {
        // `_mm512_fnmadd_pd(a, b, c)` == `c - a*b`.
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }
    #[inline(always)]
    unsafe fn max(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        // x86 MAXPD returns the second operand when unordered (NaN `a` -> `b`).
        unsafe { _mm512_max_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn min(self, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm512_min_pd(a, b) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: Self::Reg) -> f64 {
        unsafe { _mm512_reduce_add_pd(v) }
    }
}

// ---- mixed precision: f16 / bf16 inputs, f32 accumulator (16-wide __m512) ----

/// `f16` via AVX-512 `vcvtph2ps` / `vcvtps2ph` (round-to-nearest-even on store,
/// matching `half::f16::from_f32`).
#[cfg(feature = "half")]
impl KernelSimd<f16, f16, f32, f16> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const f16) -> __m512 {
        unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: f16) -> __m512 {
        // Broadcast the f16 bits to all 16 lanes, then widen via `vcvtph2ps`
        // (zmm) — pure AVX-512F, so no separate F16C feature is needed.
        unsafe { _mm512_cvtph_ps(_mm256_set1_epi16(v.to_bits() as i16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const f16) -> __m512 {
        // C (== Out == Lhs == f16) widens exactly like an A panel.
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut f16, v: __m512) {
        unsafe {
            let h = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
            _mm256_storeu_si256(p as *mut __m256i, h);
        }
    }
}

/// `bf16` via integer ops (all AVX-512F): widen = 16-bit left shift into `f32`;
/// narrow = round-to-nearest-even bias trick then truncate. The **narrowing** is
/// bit-identical to `half::bf16::from_f32`, including NaN (forced to `(bits>>16) | 0x0040`)
/// — so the bf16 *conversion* matches the scalar path. (This widen-and-FMA kernel's MAC
/// also matches scalar; the `vdpbf16ps` dot kernel's fused 2-term MAC does not — only its
/// conversion does. Conversion bit-identity keeps full and edge tiles of one run consistent.)
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> __m512 {
        unsafe {
            let w = _mm256_loadu_si256(p as *const __m256i); // 16 × u16
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w)))
        }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> __m512 {
        unsafe { _mm512_set1_ps(f32::from_bits((v.to_bits() as u32) << 16)) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> __m512 {
        unsafe { self.load_lhs(p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: __m512) {
        unsafe {
            let bits = _mm512_castps_si512(v);
            // RNE round-and-truncate for finite values.
            let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(bits), _mm512_set1_epi32(1));
            let bias = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF));
            let rounded = _mm512_srli_epi32::<16>(_mm512_add_epi32(bits, bias));
            // NaN lanes (|bits| > 0x7F80_0000): half forces `(bits>>16) | 0x0040`.
            let abs = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
            let nan = _mm512_cmpgt_epi32_mask(abs, _mm512_set1_epi32(0x7F80_0000));
            let nan_out = _mm512_or_si512(_mm512_srli_epi32::<16>(bits), _mm512_set1_epi32(0x0040));
            let out = _mm512_mask_blend_epi32(nan, rounded, nan_out);
            // Truncate each 32-bit lane to its low 16 bits → 16 contiguous u16.
            _mm256_storeu_si256(p as *mut __m256i, _mm512_cvtepi32_epi16(out));
        }
    }
}

// ---- integer: i8 inputs, i32 accumulator (16-wide __m512i, AVX-512F integer) ----

#[cfg(feature = "int8")]
impl SimdOps<i32> for Avx512 {
    type Reg = __m512i;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> __m512i {
        unsafe { _mm512_load_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_store_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(_mm512_mullo_epi32(a, b), c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        // `c - a*b` (wrapping i32). Present only to satisfy the trait; the integer
        // kernel never calls it.
        unsafe { _mm512_sub_epi32(c, _mm512_mullo_epi32(a, b)) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512i) -> i32 {
        unsafe { _mm512_reduce_add_epi32(v) }
    }
}

/// `i8 -> i32`: sign-extend 16 LHS bytes on load, broadcast a sign-extended RHS
/// byte; `Out == Acc == i32`, so `load_out`/`store_out` are plain `i32` load/store.
#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Avx512 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> __m512i {
        unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> __m512i {
        unsafe { _mm512_set1_epi32(v as i32) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }
}

// ---- AVX-512 VNNI: i8 -> i32 dot kernel via `vpdpbusd` (4 depth steps / instr) ----

/// AVX-512 VNNI ISA token: the integer dot kernel. A **distinct token** from [`Avx512`]
/// because `#[target_feature]` is per-token — `_mm512_dpbusd_epi32` needs an
/// `avx512vnni` codegen context that [`Avx512::vectorize`] (only `avx512f`) lacks. Its
/// [`SimdOps<i32>`] and `i8 -> i32` seam mirror [`Avx512`] (same `__m512i`, 16 lanes);
/// the one addition is the [`KernelSimd::dot_accumulate`] override folding 4 depth steps
/// × 16 lanes per `vpdpbusd`.
#[cfg(feature = "int8")]
#[derive(Copy, Clone, Default)]
pub struct Avx512Vnni;

#[cfg(feature = "int8")]
impl Simd for Avx512Vnni {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        // `vpdpbusd` needs `avx512vnni`; `avx512bw` rides along for the byte ops. The
        // dispatcher verifies all three before selecting this token.
        #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: dispatcher guarantees avx512f+bw+vnni; `inner` sets the codegen
        // context and `f` inlines into it.
        unsafe { inner(f) }
    }
}

#[cfg(feature = "int8")]
impl SimdOps<i32> for Avx512Vnni {
    type Reg = __m512i;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: i32) -> __m512i {
        unsafe { _mm512_set1_epi32(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const i32) -> __m512i {
        unsafe { _mm512_load_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_store_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_mullo_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(_mm512_mullo_epi32(a, b), c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi32(c, _mm512_mullo_epi32(a, b)) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512i) -> i32 {
        unsafe { _mm512_reduce_add_epi32(v) }
    }
}

/// `i8 -> i32` via VNNI. The load/store seam matches [`Avx512`] (plain `i32` epilogue;
/// `load_lhs`/`splat_rhs` are required by the trait but unused — the hot loop runs
/// through [`Self::dot_accumulate`], which reads the family's k-quad-interleaved panels
/// directly).
#[cfg(feature = "int8")]
impl KernelSimd<i8, i8, i32, i32> for Avx512Vnni {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const i8) -> __m512i {
        unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(p as *const __m128i)) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: i8) -> __m512i {
        unsafe { _mm512_set1_epi32(v as i32) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const i32) -> __m512i {
        unsafe { _mm512_loadu_si512(p as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut i32, v: __m512i) {
        unsafe { _mm512_storeu_si512(p as *mut __m512i, v) }
    }

    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const i8,
        b: *const i8,
        acc: &mut [[__m512i; MR_REG]; NR],
    ) {
        unsafe {
            let mr = MR_REG * 16;
            let nquads = kc.div_ceil(4);

            // Column sums of the *signed* B panel, for the `+128` bias correction. The
            // packed A holds `A + 128` (unsigned, as `vpdpbusd` requires), so per lane
            // `Σ_k (A+128)·B = Σ_k A·B + 128·Σ_k B`. B's depth/column pad is 0, so summing
            // the padded panel equals summing real B. B is broadcast, so every lane shares
            // the same `colsum[j]`; a scalar sum (mod 2³², matching the i32 accumulation)
            // suffices. (`s` over one quad is bounded by `4·127`, no overflow; the running
            // `colsum` wraps.) This recomputes the sum once per A row-panel rather than once
            // per B panel; the scalar pass is a small fraction of the `vpdpbusd` work and the
            // B strip is cache-warm, so it is kept here instead of widening the packed-panel
            // layout (and the driver's buffer sizing) to carry a precomputed column sum.
            let mut colsum = [0i32; NR];
            for q in 0..nquads {
                for j in 0..NR {
                    let base = q * NR * 4 + j * 4;
                    let mut s = 0i32;
                    for t in 0..4 {
                        s += *b.add(base + t) as i32;
                    }
                    colsum[j] = colsum[j].wrapping_add(s);
                }
            }

            // Dot accumulation: each `vpdpbusd` folds 4 depth steps × 16 lanes. A register
            // `i` is 16 rows × 4 contiguous k-bytes (64 B); a B quad broadcasts 4
            // contiguous k-bytes of one column as an i32.
            for q in 0..nquads {
                let a_regs: [__m512i; MR_REG] = core::array::from_fn(|i| {
                    _mm512_loadu_si512(a.add(q * mr * 4 + i * 64) as *const __m512i)
                });
                for j in 0..NR {
                    let bj = _mm512_set1_epi32(
                        (b.add(q * NR * 4 + j * 4) as *const i32).read_unaligned(),
                    );
                    for i in 0..MR_REG {
                        acc[j][i] = _mm512_dpbusd_epi32(acc[j][i], a_regs[i], bj);
                    }
                }
            }

            // Subtract the per-column bias `128·colsum[j]` (identical in every lane) to
            // recover the true signed `Σ_k A·B`.
            for j in 0..NR {
                let corr = _mm512_set1_epi32(128i32.wrapping_mul(colsum[j]));
                for i in 0..MR_REG {
                    acc[j][i] = _mm512_sub_epi32(acc[j][i], corr);
                }
            }
        }
    }
}

// ---- AVX-512 BF16: bf16 -> f32 dot kernel via `vdpbf16ps` (2 depth steps / instr) ----

/// AVX-512 BF16 ISA token: the bf16 dot kernel. A **distinct token** from [`Avx512`]
/// because `#[target_feature]` is per-token — `_mm512_dpbf16_ps` needs an `avx512bf16`
/// codegen context that [`Avx512::vectorize`] (only `avx512f`) lacks. Its
/// [`SimdOps<f32>`] and `bf16 -> f32` seam mirror [`Avx512`] (same `__m512`, 16 lanes,
/// identical RNE/NaN narrowing); the one addition is the [`KernelSimd::dot_accumulate`]
/// override folding 2 depth steps per `vdpbf16ps`.
#[cfg(feature = "half")]
#[derive(Copy, Clone, Default)]
pub struct Avx512Bf16;

#[cfg(feature = "half")]
impl Simd for Avx512Bf16 {
    #[inline(always)]
    unsafe fn vectorize<R>(self, f: impl FnOnce() -> R) -> R {
        #[target_feature(enable = "avx512f,avx512bf16")]
        unsafe fn inner<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        // SAFETY: dispatcher guarantees avx512f+bf16; `inner` sets the codegen context.
        unsafe { inner(f) }
    }
}

#[cfg(feature = "half")]
impl SimdOps<f32> for Avx512Bf16 {
    type Reg = __m512;
    const LANES: usize = 16;
    const ALIGN: usize = 64;

    #[inline(always)]
    unsafe fn zero(self) -> __m512 {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn splat(self, v: f32) -> __m512 {
        unsafe { _mm512_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(self, p: *const f32) -> __m512 {
        unsafe { _mm512_load_ps(p) }
    }
    #[inline(always)]
    unsafe fn loadu(self, p: *const f32) -> __m512 {
        unsafe { _mm512_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32, v: __m512) {
        unsafe { _mm512_store_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn storeu(self, p: *mut f32, v: __m512) {
        unsafe { _mm512_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn mul(self, a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn add(self, a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn fnma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }
    #[inline(always)]
    unsafe fn reduce_sum(self, v: __m512) -> f32 {
        unsafe { _mm512_reduce_add_ps(v) }
    }
}

/// `bf16 -> f32` via `vdpbf16ps`. The widen-load / narrow-store seam **delegates to
/// [`Avx512`]'s `bf16` impl** — one source of truth for the RNE-bias + `half`-NaN
/// narrowing (must stay bit-identical to `half::bf16::from_f32` and the scalar edge
/// path); `vectorize` enables a superset of `avx512f`, so the delegated conversions land
/// in a valid context. `splat_rhs` is trait-required but unused (the hot loop runs
/// [`Self::dot_accumulate`]); `load_out` *is* used, by the `beta != 0` C-read.
#[cfg(feature = "half")]
impl KernelSimd<bf16, bf16, f32, bf16> for Avx512Bf16 {
    #[inline(always)]
    unsafe fn load_lhs(self, p: *const bf16) -> __m512 {
        unsafe { <Avx512 as KernelSimd<bf16, bf16, f32, bf16>>::load_lhs(Avx512, p) }
    }
    #[inline(always)]
    unsafe fn splat_rhs(self, v: bf16) -> __m512 {
        unsafe { <Avx512 as KernelSimd<bf16, bf16, f32, bf16>>::splat_rhs(Avx512, v) }
    }
    #[inline(always)]
    unsafe fn load_out(self, p: *const bf16) -> __m512 {
        unsafe { <Avx512 as KernelSimd<bf16, bf16, f32, bf16>>::load_out(Avx512, p) }
    }
    #[inline(always)]
    unsafe fn store_out(self, p: *mut bf16, v: __m512) {
        unsafe { <Avx512 as KernelSimd<bf16, bf16, f32, bf16>>::store_out(Avx512, p, v) }
    }

    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn dot_accumulate<const MR_REG: usize, const NR: usize>(
        self,
        kc: usize,
        a: *const bf16,
        b: *const bf16,
        acc: &mut [[__m512; MR_REG]; NR],
    ) {
        unsafe {
            let mr = MR_REG * 16;
            let npairs = kc.div_ceil(2);

            // Each `vdpbf16ps` folds 2 depth steps: per f32 lane, the 2-term bf16 dot
            // `f32(a0)·f32(b0) + f32(a1)·f32(b1)`. A register `i` holds 16 rows × 2
            // contiguous bf16 (64 B → `__m512bh`); a B pair broadcasts 2 contiguous bf16
            // of one column. Odd-`k` tails were zero-padded in both panels at pack time
            // (0·0 = 0), so the loop reads whole pairs. No bias/signedness fixup (plain dot).
            for p2 in 0..npairs {
                let a_regs: [__m512bh; MR_REG] = core::array::from_fn(|i| {
                    core::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(
                        a.add(p2 * mr * 2 + i * 32) as *const __m512i,
                    ))
                });
                for j in 0..NR {
                    let bj = core::mem::transmute::<__m512i, __m512bh>(_mm512_set1_epi32(
                        (b.add(p2 * NR * 2 + j * 2) as *const i32).read_unaligned(),
                    ));
                    for i in 0..MR_REG {
                        acc[j][i] = _mm512_dpbf16_ps(acc[j][i], a_regs[i], bj);
                    }
                }
            }
        }
    }
}

// Complex (AVX-512): real `Reg`; `LANES` is the real lane count (16 / 8). Complex GEMM
// routes through the shared SoA `soa_microkernel`.
#[cfg(feature = "complex")]
impl_complex_simd!(Avx512, f32, __m512, 16, 64);
#[cfg(feature = "complex")]
impl_complex_simd!(Avx512, f64, __m512d, 8, 64);
