//! Valid-by-construction plans and entries for fuzz_gemm / fuzz_knobs / fuzz_batched / fuzz_prepack

use arbitrary::{Arbitrary, Result, Unstructured};

use crate::common::{AB_TABLE, I8_AB_TABLE, LayoutPlan, ab_index, arb_par, arb_par_knobs};
use crate::differential::{
    differential_batched_real, differential_gemm_cplx, differential_gemm_i8,
    differential_gemm_real, differential_packed_a_real, differential_packed_b_real,
};
use gemmkit::{Parallelism, bf16, c32, c64, f16, tuning};

// fuzz_gemm

#[derive(Debug, Clone, Copy)]
pub enum TypeTag {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    C32,
    C64,
}

#[derive(Debug)]
pub struct GemmPlan {
    pub ty: TypeTag,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub lc: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub alpha_im_i: usize,
    pub beta_im_i: usize,
    pub nan_c: bool,
    pub conj_a: bool,
    pub conj_b: bool,
    pub ws_reuse: bool,
    pub par: Parallelism,
    pub a_seed: u64,
    pub b_seed: u64,
    pub c_seed: u64,
}

impl<'a> Arbitrary<'a> for GemmPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ty = match u.int_in_range(0u8..=6)? {
            0 => TypeTag::F32,
            1 => TypeTag::F64,
            2 => TypeTag::F16,
            3 => TypeTag::Bf16,
            4 => TypeTag::I8,
            5 => TypeTag::C32,
            _ => TypeTag::C64,
        };
        Ok(GemmPlan {
            ty,
            // m,n cross the AVX-512 f32 tile edges (MR=32, NR=12); k crosses the
            // bf16/VNNI DEPTH_MULTIPLE padding and partial-depth panels
            m: u.int_in_range(0usize..=48)?,
            k: u.int_in_range(0usize..=130)?,
            n: u.int_in_range(0usize..=48)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            lc: LayoutPlan::arbitrary_general(u, false)?, // self-aliasing C is a documented panic
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            alpha_im_i: ab_index(u)?,
            beta_im_i: ab_index(u)?,
            nan_c: bool::arbitrary(u)?,
            conj_a: bool::arbitrary(u)?,
            conj_b: bool::arbitrary(u)?,
            ws_reuse: bool::arbitrary(u)?,
            par: arb_par(u)?,
            a_seed: u64::arbitrary(u)?,
            b_seed: u64::arbitrary(u)?,
            c_seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_gemm(p: GemmPlan) {
    let ctx = "fuzz_gemm";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    match p.ty {
        TypeTag::F32 => differential_gemm_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed,
            p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::F64 => differential_gemm_real::<f64>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed,
            p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::F16 => differential_gemm_real::<f16>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed,
            p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::Bf16 => differential_gemm_real::<bf16>(
            p.m, p.k, p.n, p.la, p.lb, p.lc, af, bf, p.nan_c, p.par, p.ws_reuse, p.a_seed,
            p.b_seed, p.c_seed, ctx,
        ),
        TypeTag::I8 => differential_gemm_i8(
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            p.lc,
            I8_AB_TABLE[p.alpha_i],
            I8_AB_TABLE[p.beta_i],
            p.par,
            p.a_seed,
            p.b_seed,
            p.c_seed,
            ctx,
        ),
        TypeTag::C32 => differential_gemm_cplx::<c32>(
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            p.lc,
            (af, AB_TABLE[p.alpha_im_i]),
            (bf, AB_TABLE[p.beta_im_i]),
            p.conj_a,
            p.conj_b,
            p.nan_c,
            p.par,
            p.a_seed,
            p.b_seed,
            p.c_seed,
            ctx,
        ),
        TypeTag::C64 => differential_gemm_cplx::<c64>(
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            p.lc,
            (af, AB_TABLE[p.alpha_im_i]),
            (bf, AB_TABLE[p.beta_im_i]),
            p.conj_a,
            p.conj_b,
            p.nan_c,
            p.par,
            p.a_seed,
            p.b_seed,
            p.c_seed,
            ctx,
        ),
    }
}

// fuzz_knobs

/// Every `set_*` compiled on x86_64 + `std,parallel,complex,half,int8`
/// (`gemmkit/src/tuning.rs`); `set_wasm_threads` is wasm-gated and excluded
pub(crate) const KNOB_SETTERS: &[(&str, fn(usize))] = &[
    ("parallel_threshold", tuning::set_parallel_threshold),
    ("rhs_pack_threshold", tuning::set_rhs_pack_threshold),
    ("lhs_pack_threshold", tuning::set_lhs_pack_threshold),
    ("lhs_pack_stride", tuning::set_lhs_pack_stride),
    ("gemv_threshold", tuning::set_gemv_threshold),
    ("small_k_threshold", tuning::set_small_k_threshold),
    ("small_mn_dim", tuning::set_small_mn_dim),
    ("gemv_parallel_bytes", tuning::set_gemv_parallel_bytes),
    ("gemv_thread_cap", tuning::set_gemv_thread_cap),
    ("parallel_oversample", tuning::set_parallel_oversample),
    ("thread_dim_stride", tuning::set_thread_dim_stride),
    ("shared_lhs_mnk", tuning::set_shared_lhs_mnk),
    ("k_stream_max", tuning::set_k_stream_max),
    (
        "seq_internal_bytes_per_worker",
        tuning::set_seq_internal_bytes_per_worker,
    ),
    ("packed_oversample", tuning::set_packed_oversample),
    ("mc_reg_panels", tuning::set_mc_reg_panels),
    ("nc_no_l3_panels", tuning::set_nc_no_l3_panels),
    ("tiny_block_dim", tuning::set_tiny_block_dim),
    ("kc", tuning::set_kc),
    ("kc_min", tuning::set_kc_min),
    ("pack_transpose_tile", tuning::set_pack_transpose_tile),
    ("deep_kc_bytes", tuning::set_deep_kc_bytes),
    ("i8_vnni_min_par_mnk", tuning::set_i8_vnni_min_par_mnk),
];

pub(crate) const N_KNOBS: usize = KNOB_SETTERS.len();

/// Knob-value classes exercising the setters' boundary behavior. Setters store
/// unconditionally and clamp `usize::MAX` to `MAX-1` (the UNSET sentinel), so `MAX`
/// exercises the clamp too
pub(crate) fn knob_value(u: &mut Unstructured) -> Result<usize> {
    Ok(match u.int_in_range(0u8..=8)? {
        0 => 0, // 0 = auto convention on several knobs
        1 => 1,
        2 => u.int_in_range(2usize..=17)?,  // small
        3 => u.int_in_range(31usize..=65)?, // dim-boundary (tile/tiny edges)
        4 => 4096,                          // page-ish (lhs_pack_stride is in bytes)
        5 => 1usize << 33,                  // > i32/f32-index range
        6 => 1usize << 48,                  // huge
        7 => usize::MAX - 1,
        _ => usize::MAX, // clamps to UNSET-1 in the setter
    })
}

#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    PlainF32,
    Gemv,
    SmallMn,
    PrepackB,
    PrepackA,
    I8,
    Batched,
}

#[derive(Debug)]
pub struct KnobsPlan {
    pub values: [usize; N_KNOBS],
    pub scenario: Scenario,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for KnobsPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let mut values = [0usize; N_KNOBS];
        for v in values.iter_mut() {
            *v = knob_value(u)?;
        }
        let scenario = match u.int_in_range(0u8..=6)? {
            0 => Scenario::PlainF32,
            1 => Scenario::Gemv,
            2 => Scenario::SmallMn,
            3 => Scenario::PrepackB,
            4 => Scenario::PrepackA,
            5 => Scenario::I8,
            _ => Scenario::Batched,
        };
        Ok(KnobsPlan {
            values,
            scenario,
            m: u.int_in_range(1usize..=24)?,
            k: u.int_in_range(1usize..=24)?,
            n: u.int_in_range(1usize..=24)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par_knobs(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_knobs(p: KnobsPlan) {
    // Set every knob on every input: setters store unconditionally, so each exec fully
    // overwrites the knob set - no state leaks between libFuzzer execs, making every
    // crash artifact self-contained/replayable
    for (i, (_, setter)) in KNOB_SETTERS.iter().enumerate() {
        setter(p.values[i]);
    }
    let ctx = "fuzz_knobs";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    let (s1, s2, s3) = (p.seed ^ 0x11, p.seed ^ 0x22, p.seed ^ 0x33);
    let lc_row = LayoutPlan::RowIsh { il: 1, pad: 1 };
    match p.scenario {
        Scenario::PlainF32 => differential_gemm_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, lc_row, af, bf, false, p.par, false, s1, s2, s3, ctx,
        ),
        Scenario::Gemv => differential_gemm_real::<f32>(
            p.m, p.k, 1, p.la, p.lb, lc_row, af, bf, false, p.par, false, s1, s2, s3, ctx,
        ),
        Scenario::SmallMn => differential_gemm_real::<f32>(
            p.m.min(8),
            p.k.max(32),
            p.n.min(8),
            p.la,
            p.lb,
            lc_row,
            af,
            bf,
            false,
            p.par,
            false,
            s1,
            s2,
            s3,
            ctx,
        ),
        Scenario::PrepackB => differential_packed_b_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
        ),
        Scenario::PrepackA => differential_packed_a_real::<f32>(
            p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
        ),
        Scenario::I8 => differential_gemm_i8(
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            lc_row,
            I8_AB_TABLE[p.alpha_i],
            I8_AB_TABLE[p.beta_i],
            p.par,
            s1,
            s2,
            s3,
            ctx,
        ),
        Scenario::Batched => differential_batched_real::<f32>(
            3, p.m, p.k, p.n, p.la, p.lb, lc_row, false, false, 0, 0, 0, af, bf, p.par, p.seed, ctx,
        ),
    }
}

// fuzz_batched

#[derive(Debug)]
pub struct BatchedPlan {
    pub ty64: bool, // false => f32, true => f64
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub lc: LayoutPlan,
    pub a_broadcast: bool,
    pub b_broadcast: bool,
    pub a_bs_pad: usize,
    pub b_bs_pad: usize,
    pub c_bs_pad: usize,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for BatchedPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(BatchedPlan {
            ty64: bool::arbitrary(u)?,
            batch: u.int_in_range(0usize..=4)?, // 0 is the documented no-op
            m: u.int_in_range(1usize..=24)?,
            k: u.int_in_range(1usize..=24)?,
            n: u.int_in_range(1usize..=24)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            lc: LayoutPlan::arbitrary_general(u, false)?,
            a_broadcast: bool::arbitrary(u)?,
            b_broadcast: bool::arbitrary(u)?,
            a_bs_pad: u.int_in_range(0usize..=8)?,
            b_bs_pad: u.int_in_range(0usize..=8)?,
            c_bs_pad: u.int_in_range(0usize..=8)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_batched(p: BatchedPlan) {
    let ctx = "fuzz_batched";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    if p.ty64 {
        differential_batched_real::<f64>(
            p.batch,
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            p.lc,
            p.a_broadcast,
            p.b_broadcast,
            p.a_bs_pad,
            p.b_bs_pad,
            p.c_bs_pad,
            af,
            bf,
            p.par,
            p.seed,
            ctx,
        );
    } else {
        differential_batched_real::<f32>(
            p.batch,
            p.m,
            p.k,
            p.n,
            p.la,
            p.lb,
            p.lc,
            p.a_broadcast,
            p.b_broadcast,
            p.a_bs_pad,
            p.b_bs_pad,
            p.c_bs_pad,
            af,
            bf,
            p.par,
            p.seed,
            ctx,
        );
    }
}

// fuzz_prepack

#[derive(Debug)]
pub struct PrepackPlan {
    pub ty: TypeTag, // F32, F64, or Bf16
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub la: LayoutPlan,
    pub lb: LayoutPlan,
    pub alpha_i: usize,
    pub beta_i: usize,
    pub par: Parallelism,
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for PrepackPlan {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let ty = match u.int_in_range(0u8..=2)? {
            0 => TypeTag::F32,
            1 => TypeTag::F64,
            _ => TypeTag::Bf16, // exercises the DEPTH_MULTIPLE single-slice depth-pad
        };
        Ok(PrepackPlan {
            ty,
            // dims from 1..=48 (crossing tile edges); 0 excluded so the orientation
            // invariant of the packed C holds for empty views too
            m: u.int_in_range(1usize..=48)?,
            k: u.int_in_range(1usize..=48)?,
            n: u.int_in_range(1usize..=48)?,
            la: LayoutPlan::arbitrary_general(u, true)?,
            lb: LayoutPlan::arbitrary_general(u, true)?,
            alpha_i: ab_index(u)?,
            beta_i: ab_index(u)?,
            par: arb_par(u)?,
            seed: u64::arbitrary(u)?,
        })
    }
}

pub fn run_prepack(p: PrepackPlan) {
    let ctx = "fuzz_prepack";
    let af = AB_TABLE[p.alpha_i];
    let bf = AB_TABLE[p.beta_i];
    let (s1, s2, s3) = (p.seed ^ 0x11, p.seed ^ 0x22, p.seed ^ 0x33);
    macro_rules! both {
        ($t:ty) => {{
            differential_packed_b_real::<$t>(
                p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
            );
            differential_packed_a_real::<$t>(
                p.m, p.k, p.n, p.la, p.lb, af, bf, p.par, s1, s2, s3, ctx,
            );
        }};
    }
    match p.ty {
        TypeTag::F32 => both!(f32),
        TypeTag::F64 => both!(f64),
        _ => both!(bf16),
    }
}
