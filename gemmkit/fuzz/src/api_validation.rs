//! Adversarial-geometry plan and driver for fuzz_api_validation

use arbitrary::{Arbitrary, Result, Unstructured};

use crate::common::{AB_TABLE, CplxElem, I8_AB_TABLE, ab_index, arb_par};
use gemmkit::{
    MatMut, MatRef, Parallelism, c32, gemm, gemm_batched, gemm_cplx, gemm_i8, prepack_lhs,
    prepack_rhs,
};

// fuzz_api_validation

/// Edge-case dimension classes for validation fuzzing: `2^33` targets the extent
/// isize-mul overflow; `usize::MAX/2` / `usize::MAX` the "too large to address" reject
#[derive(Debug, Clone, Copy)]
pub enum DimClass {
    Zero,
    One,
    Small(usize),
    P31,
    P32p1,
    P33,
    HalfMax,
    Max,
}
impl DimClass {
    pub fn get(self) -> usize {
        match self {
            DimClass::Zero => 0,
            DimClass::One => 1,
            DimClass::Small(s) => s,
            DimClass::P31 => 1usize << 31,
            DimClass::P32p1 => (1usize << 32) + 1,
            DimClass::P33 => 1usize << 33,
            DimClass::HalfMax => usize::MAX / 2,
            DimClass::Max => usize::MAX,
        }
    }
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        Ok(match u.int_in_range(0u8..=7)? {
            0 => DimClass::Zero,
            1 => DimClass::One,
            2 => DimClass::Small(u.int_in_range(2usize..=17)?),
            3 => DimClass::P31,
            4 => DimClass::P32p1,
            5 => DimClass::P33,
            6 => DimClass::HalfMax,
            _ => DimClass::Max,
        })
    }
}

/// Adversarial isize stride table; `isize::MIN/MAX` and `+/-2^33` drive the checked-mul
/// overflow inside `extent()`
#[derive(Debug, Clone, Copy)]
pub enum StrideClass {
    Zero,
    P1,
    N1,
    PSmall(isize),
    NSmall(isize),
    P31,
    N31,
    P33,
    N33,
    IMin,
    IMax,
}
impl StrideClass {
    pub fn get(self) -> isize {
        match self {
            StrideClass::Zero => 0,
            StrideClass::P1 => 1,
            StrideClass::N1 => -1,
            StrideClass::PSmall(s) => s,
            StrideClass::NSmall(s) => -s,
            StrideClass::P31 => 1isize << 31,
            StrideClass::N31 => -(1isize << 31),
            StrideClass::P33 => 1isize << 33,
            StrideClass::N33 => -(1isize << 33),
            StrideClass::IMin => isize::MIN,
            StrideClass::IMax => isize::MAX,
        }
    }
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        Ok(match u.int_in_range(0u8..=10)? {
            0 => StrideClass::Zero,
            1 => StrideClass::P1,
            2 => StrideClass::N1,
            3 => StrideClass::PSmall(u.int_in_range(2isize..=17)?),
            4 => StrideClass::NSmall(u.int_in_range(2isize..=17)?),
            5 => StrideClass::P31,
            6 => StrideClass::N31,
            7 => StrideClass::P33,
            8 => StrideClass::N33,
            9 => StrideClass::IMin,
            _ => StrideClass::IMax,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EntryKind {
    Gemm,
    GemmI8,
    GemmCplx,
    Batched,
    PrepackB,
    PrepackA,
}

#[derive(Debug)]
pub struct ValidationPlan {
    pub entry: EntryKind,
    pub len_a: usize,
    pub len_b: usize,
    pub len_c: usize,
    pub m: DimClass,
    pub k: DimClass,
    pub n: DimClass,
    pub mc: DimClass, // C rows (independent, to exercise the shape assert)
    pub nc: DimClass, // C cols
    pub rsa: StrideClass,
    pub csa: StrideClass,
    pub rsb: StrideClass,
    pub csb: StrideClass,
    pub rsc: StrideClass,
    pub csc: StrideClass,
    pub batch: DimClass,
    pub a_bs: StrideClass,
    pub b_bs: StrideClass,
    pub c_bs: StrideClass,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub conj_a: bool,
    pub conj_b: bool,
    pub par: Parallelism,
}

impl<'a> Arbitrary<'a> for ValidationPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let entry = match u.int_in_range(0u8..=5)? {
            0 => EntryKind::Gemm,
            1 => EntryKind::GemmI8,
            2 => EntryKind::GemmCplx,
            3 => EntryKind::Batched,
            4 => EntryKind::PrepackB,
            _ => EntryKind::PrepackA,
        };
        Ok(ValidationPlan {
            entry,
            len_a: u.int_in_range(0usize..=8192)?,
            len_b: u.int_in_range(0usize..=8192)?,
            len_c: u.int_in_range(0usize..=8192)?,
            m: DimClass::arbitrary(u)?,
            k: DimClass::arbitrary(u)?,
            n: DimClass::arbitrary(u)?,
            mc: DimClass::arbitrary(u)?,
            nc: DimClass::arbitrary(u)?,
            rsa: StrideClass::arbitrary(u)?,
            csa: StrideClass::arbitrary(u)?,
            rsb: StrideClass::arbitrary(u)?,
            csb: StrideClass::arbitrary(u)?,
            rsc: StrideClass::arbitrary(u)?,
            csc: StrideClass::arbitrary(u)?,
            batch: DimClass::arbitrary(u)?,
            a_bs: StrideClass::arbitrary(u)?,
            b_bs: StrideClass::arbitrary(u)?,
            c_bs: StrideClass::arbitrary(u)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            conj_a: bool::arbitrary(u)?,
            conj_b: bool::arbitrary(u)?,
            par: arb_par(u)?,
        })
    }
}

/// Skip-cap on accepted-and-expensive plans: a bounded, valid geometry must not turn
/// into billions of MACs or a k-proportional gigabyte alloc under ASan (those would be
/// false-positive timeouts/OOMs, not memory-unsafety). 2^24 elements/MACs
const WORK_CAP: usize = 1 << 24;

/// Mirror of `api.rs::extent`: highest slice offset (exclusive), or `None` for a
/// negative-stride or too-large-to-address view (both of which validation rejects)
fn mirror_extent(rows: usize, cols: usize, rs: isize, cs: isize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let mut lo: isize = 0;
    let mut hi: isize = 0;
    for &(dim, s) in &[(rows, rs), (cols, cs)] {
        let e = isize::try_from(dim).ok()?.checked_sub(1)?.checked_mul(s)?;
        if e < 0 {
            lo = lo.checked_add(e)?;
        } else {
            hi = hi.checked_add(e)?;
        }
    }
    if lo < 0 {
        None
    } else {
        (hi as usize).checked_add(1)
    }
}

/// Mirror of `api.rs::self_aliases`
fn mirror_self_aliases(rows: usize, cols: usize, rs: isize, cs: isize) -> bool {
    if rows == 0 || cols == 0 {
        return false;
    }
    let r = (rows > 1).then_some((rs.unsigned_abs(), rows));
    let c = (cols > 1).then_some((cs.unsigned_abs(), cols));
    match (r, c) {
        (None, None) => false,
        (Some((s, _)), None) | (None, Some((s, _))) => s == 0,
        (Some(a), Some(b)) => {
            let (sm, big) = if a.0 <= b.0 { (a, b.0) } else { (b, a.0) };
            sm.0 == 0 || big < sm.0.saturating_mul(sm.1)
        }
    }
}

fn in_bounds(rows: usize, cols: usize, rs: isize, cs: isize, len: usize) -> bool {
    matches!(mirror_extent(rows, cols, rs, cs), Some(need) if need <= len)
}

fn sat3(a: usize, b: usize, c: usize) -> usize {
    a.saturating_mul(b).saturating_mul(c)
}

/// The raw driver behind `fuzz_api_validation`. May panic with a documented
/// `gemmkit:` message (accepted by the target's `catch_unwind`) or run cleanly; the
/// WORK_CAP guard only skips plans that WOULD fully validate and then do unbounded work
pub fn drive_validation(p: &ValidationPlan) {
    let (m, k, n) = (p.m.get(), p.k.get(), p.n.get());
    let (mc, nc) = (p.mc.get(), p.nc.get());
    let (rsa, csa) = (p.rsa.get(), p.csa.get());
    let (rsb, csb) = (p.rsb.get(), p.csb.get());
    let (rsc, csc) = (p.rsc.get(), p.csc.get());
    let alpha = AB_TABLE[p.alpha_i] as f32;
    let beta = AB_TABLE[p.beta_i] as f32;

    match p.entry {
        EntryKind::Gemm | EntryKind::GemmI8 | EntryKind::GemmCplx => {
            // Would this geometry fully pass validation? If so, cap the compute
            let would_pass = in_bounds(m, k, rsa, csa, p.len_a)
                && in_bounds(k, n, rsb, csb, p.len_b)
                && in_bounds(mc, nc, rsc, csc, p.len_c)
                && mc == m
                && nc == n
                && !mirror_self_aliases(mc, nc, rsc, csc);
            if would_pass && sat3(m, n, k) > WORK_CAP {
                return;
            }
            match p.entry {
                EntryKind::Gemm => {
                    let a = vec![0.0f32; p.len_a];
                    let b = vec![0.0f32; p.len_b];
                    let mut c = vec![0.0f32; p.len_c];
                    gemm(
                        alpha,
                        MatRef::new(&a, m, k, rsa, csa),
                        MatRef::new(&b, k, n, rsb, csb),
                        beta,
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
                EntryKind::GemmI8 => {
                    let a = vec![0i8; p.len_a];
                    let b = vec![0i8; p.len_b];
                    let mut c = vec![0i32; p.len_c];
                    gemm_i8(
                        I8_AB_TABLE[p.alpha_i],
                        MatRef::new(&a, m, k, rsa, csa),
                        MatRef::new(&b, k, n, rsb, csb),
                        I8_AB_TABLE[p.beta_i],
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
                _ => {
                    let a = vec![c32::ZERO; p.len_a];
                    let b = vec![c32::ZERO; p.len_b];
                    let mut c = vec![c32::ZERO; p.len_c];
                    gemm_cplx(
                        c32::make(alpha as f64, 0.0),
                        MatRef::new(&a, m, k, rsa, csa),
                        p.conj_a,
                        MatRef::new(&b, k, n, rsb, csb),
                        p.conj_b,
                        c32::make(beta as f64, 0.0),
                        MatMut::new(&mut c, mc, nc, rsc, csc),
                        p.par,
                    );
                }
            }
        }
        EntryKind::Batched => {
            let batch = p.batch.get();
            let a_bs = p.a_bs.get();
            let b_bs = p.b_bs.get();
            let c_bs = p.c_bs.get();
            // Batched bounds mirror (extent + (batch-1)*bs). A batch stride < 0 with
            // batch>1 is a documented reject, so it is not "would_pass"
            let batched_ok = |rows, cols, rs, cs, bs: isize, len: usize| -> bool {
                let Some(e) = mirror_extent(rows, cols, rs, cs) else {
                    return false;
                };
                if batch <= 1 {
                    return e <= len;
                }
                if bs < 0 {
                    return false;
                }
                let last = (batch - 1).saturating_mul(bs as usize);
                last.saturating_add(e) <= len
            };
            let ec = mirror_extent(mc, nc, rsc, csc);
            let would_pass = batch != 0
                && batched_ok(m, k, rsa, csa, a_bs, p.len_a)
                && batched_ok(k, n, rsb, csb, b_bs, p.len_b)
                && batched_ok(mc, nc, rsc, csc, c_bs, p.len_c)
                && mc == m
                && nc == n
                && !mirror_self_aliases(mc, nc, rsc, csc)
                && (batch <= 1 || (c_bs >= 0 && ec.map(|e| (c_bs as usize) >= e).unwrap_or(false)));
            // The batch LOOP count is itself unbounded work even when each element is
            // empty (m*n == 0 zeroes the product), so cap the raw batch too, else
            // gemm_batched spins over `batch` no-op elements and libFuzzer times out
            if would_pass && (batch > WORK_CAP || batch.saturating_mul(sat3(m, n, k)) > WORK_CAP) {
                return;
            }
            let a = vec![0.0f32; p.len_a];
            let b = vec![0.0f32; p.len_b];
            let mut c = vec![0.0f32; p.len_c];
            gemm_batched(
                batch,
                alpha,
                MatRef::new(&a, m, k, rsa, csa),
                a_bs,
                MatRef::new(&b, k, n, rsb, csb),
                b_bs,
                beta,
                MatMut::new(&mut c, mc, nc, rsc, csc),
                c_bs,
                p.par,
            );
        }
        EntryKind::PrepackB => {
            // Skip only the "representable but huge" middle band: a would-pass pack
            // whose ~n*k element count fits usize yet exceeds the work cap would OOM
            // on correct behavior. Everything else is fast to run: empty operands
            // short-circuit in prepack, and a pack size that overflows usize is a
            // documented "too large" reject - both stay fuzzed
            let would_pass = in_bounds(k, n, rsb, csb, p.len_b);
            let expensive = n != 0
                && k != 0
                && n.checked_mul(k).is_some()
                && (n > WORK_CAP || k > WORK_CAP || n * k > WORK_CAP);
            if would_pass && expensive {
                return;
            }
            let b = vec![0.0f32; p.len_b];
            let _ = prepack_rhs(MatRef::new(&b, k, n, rsb, csb));
        }
        EntryKind::PrepackA => {
            let would_pass = in_bounds(m, k, rsa, csa, p.len_a);
            let expensive = m != 0
                && k != 0
                && m.checked_mul(k).is_some()
                && (m > WORK_CAP || k > WORK_CAP || m * k > WORK_CAP);
            if would_pass && expensive {
                return;
            }
            let a = vec![0.0f32; p.len_a];
            let _ = prepack_lhs(MatRef::new(&a, m, k, rsa, csa));
        }
    }
}
