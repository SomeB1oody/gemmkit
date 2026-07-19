//! Per-ISA kernel checks driven straight through the generic driver (bypassing the
//! dispatch layer's runtime ISA selection and orientation swap), plus the Miri
//! scalar-only suite covering every element family and the packing/epilogue/gemv
//! paths a real Miri run can reach

use crate::common::*;
use gemmkit::driver;
use gemmkit::kernel::FloatGemm;
#[cfg(target_arch = "aarch64")]
use gemmkit::simd::Neon;
use gemmkit::simd::ScalarTok;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use gemmkit::simd::Simd128;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use gemmkit::simd::{Avx512, Fma};
use gemmkit::{MatMut, MatRef, Parallelism, Workspace, gemm};

// Per-ISA kernels, driven directly through the generic driver, column-major throughout

/// Drive `FloatGemm` through `driver::run` with an explicit ISA token and tile shape,
/// all operands column-major, and check the result against the f64 reference: exercises
/// exactly the kernel named by `S`/`MR_REG`/`NR`, not whatever the runtime would pick
fn driver_case<T, S, const MR_REG: usize, const NR: usize>(simd: S, m: usize, k: usize, n: usize)
where
    T: Elem + gemmkit::Float<Acc = T>,
    S: gemmkit::simd::SimdOps<T>,
{
    let a = Mat::<T>::rand(m, k, 0x55 + (m + k + n) as u64);
    let b = Mat::<T>::rand(k, n, 0x66 + (m * 2 + n) as u64);
    let c0 = Mat::<T>::rand(m, n, 0x77 + (k * 3) as u64);
    let (abuf, rsa, csa) = build_view(&a, Layout::Col);
    let (bbuf, rsb, csb) = build_view(&b, Layout::Col);
    let (mut cbuf, rsc, csc) = build_view(&c0, Layout::Col);
    let alpha = T::from_f64(1.3);
    let beta = T::from_f64(-0.7);
    let cref = reference(&a, &b, &c0, alpha.to_f64(), beta.to_f64());

    let mut ws = Workspace::new();
    unsafe {
        driver::run::<FloatGemm<T>, S, MR_REG, NR>(
            simd,
            m,
            k,
            n,
            alpha,
            abuf.as_ptr(),
            rsa,
            csa,
            bbuf.as_ptr(),
            rsb,
            csb,
            beta,
            cbuf.as_mut_ptr(),
            rsc,
            csc,
            Parallelism::Serial,
            &mut ws,
        );
    }
    assert_accurate(&cbuf, rsc, csc, m, n, &cref, &a, &b, k, "driver per-ISA");
}

fn isa_shapes() -> [(usize, usize, usize); 6] {
    [
        (1, 1, 1),
        (3, 4, 5),
        (32, 32, 32),
        (33, 17, 19),
        (64, 80, 48),
        (128, 96, 112),
    ]
}

/// The portable scalar kernel, f32 and f64, always available regardless of target
#[test]
fn isa_scalar() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, ScalarTok, 4, 4>(ScalarTok, m, k, n);
        driver_case::<f64, ScalarTok, 4, 4>(ScalarTok, m, k, n);
    }
}

/// Runs the scalar `SimdOps` path (the only one Miri can execute, since it has no
/// SIMD intrinsics) through packing, scratch, epilogue, gemv, and the driver, small
/// shapes throughout to keep Miri fast. Under Miri, runtime CPU feature detection
/// reports nothing available, so the dispatched `gemm` calls below also land on the
/// scalar kernel, not just the direct `driver_case` calls
#[test]
fn miri_scalar_path() {
    for (m, k, n) in [
        (5, 4, 6),
        (8, 8, 8),
        (3, 7, 2),
        (1, 5, 4),
        (6, 2, 1),
        (4, 4, 4),
    ] {
        driver_case::<f32, ScalarTok, 4, 4>(ScalarTok, m, k, n);
        driver_case::<f64, ScalarTok, 4, 4>(ScalarTok, m, k, n);
    }
    // Safe-API round trip: partial tiles, beta != 0, general strides
    run_case::<f32>(
        7,
        9,
        5,
        Layout::Row,
        Layout::Col,
        Layout::Row,
        1.3,
        0.5,
        Parallelism::Serial,
    );
    run_case::<f64>(
        6,
        6,
        6,
        Layout::GeneralPad,
        Layout::Row,
        Layout::Col,
        0.0,
        2.0,
        Parallelism::Serial,
    );
    // Mixed-precision (f16/bf16) scalar path: widen-load, f32 accumulate, narrow store,
    // plus the strided copy-back epilogue and a beta != 0 read of narrow C
    #[cfg(feature = "half")]
    {
        run_case::<gemmkit::f16>(
            7,
            9,
            5,
            Layout::Row,
            Layout::Col,
            Layout::Row,
            // `Elem::from_f64` routes to the `*_const` software conversion under Miri
            // (see the `Elem` impls in oracle_common), keeping alpha/beta construction safe too
            Elem::from_f64(1.0),
            Elem::from_f64(0.5),
            Parallelism::Serial,
        );
        run_case::<gemmkit::bf16>(
            6,
            6,
            6,
            Layout::Col,
            Layout::Row,
            Layout::Col,
            Elem::from_f64(0.75),
            Elem::from_f64(-0.5),
            Parallelism::Serial,
        );
    }
    // Complex (c32) scalar path with conj_a: conjugate-on-pack plus the scalar
    // complex multiply and epilogue
    #[cfg(feature = "complex")]
    {
        let (m, k, n) = (5usize, 4, 6);
        let a = rand_cplx::<gemmkit::c32>(m * k, 11);
        let b = rand_cplx::<gemmkit::c32>(k * n, 12);
        let c0 = rand_cplx::<gemmkit::c32>(m * n, 13);
        let (alpha, beta) = (
            gemmkit::Complex::new(1.0f32, 0.0),
            gemmkit::Complex::new(0.5f32, -0.25),
        );
        let cref = ref_cplx(&a, &b, &c0, m, k, n, alpha, beta, true, false);
        let mut c = c0.clone();
        gemmkit::gemm_cplx(
            alpha,
            MatRef::from_col_major(&a, m, k),
            true,
            MatRef::from_col_major(&b, k, n),
            false,
            beta,
            MatMut::from_col_major(&mut c, m, n),
            Parallelism::Serial,
        );
        assert_cplx_accurate(&c, m, n, &cref, k, "miri complex conjA");
    }
    // Integer (i8 -> i32) scalar path: widen-load, i32 accumulate, partial-tile
    // copy-back, and a beta != 0 i32 read of C
    #[cfg(feature = "int8")]
    {
        let (m, k, n) = (7usize, 9, 5);
        let a = rand_i8(m * k, 1);
        let b = rand_i8(k * n, 2);
        let c0: Vec<i32> = (0..m * n).map(|x| x as i32 % 4 - 2).collect();
        let cref = ref_i8(&a, &b, &c0, m, k, n, 3, -2);
        let mut c = c0.clone();
        gemmkit::gemm_i8(
            3,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_row_major(&b, k, n),
            -2,
            MatMut::from_row_major(&mut c, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c, cref, "miri: i8 mismatch");
    }
    // gemv shapes (n == 1 and m == 1)
    run_case::<f32>(
        8,
        5,
        1,
        Layout::Col,
        Layout::Col,
        Layout::Col,
        1.0,
        0.0,
        Parallelism::Serial,
    );
    run_case::<f64>(
        1,
        5,
        8,
        Layout::Row,
        Layout::Row,
        Layout::Row,
        1.0,
        0.0,
        Parallelism::Serial,
    );
    // Prepacked-RHS on the scalar engine: m=66 clears the both-tiny gate (m,n <= 64), so
    // the packed micropanel layout matches plain gemm's and the result must be bit-identical
    {
        let (m, k, n) = (66usize, 4, 6);
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        let c0 = rand_vec::<f32>(m * n, 3);
        let mut c_ref = c0.clone();
        gemm(
            1.3,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            0.5,
            MatMut::from_col_major(&mut c_ref, m, n),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_rhs(MatRef::from_col_major(&b, k, n));
        let mut c_pk = c0.clone();
        gemmkit::gemm_packed_b(
            1.3,
            MatRef::from_col_major(&a, m, k),
            &packed,
            0.5,
            MatMut::from_col_major(&mut c_pk, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c_ref, c_pk, "miri: prepack != gemm");
    }
    // Prepacked-LHS on the scalar engine: m=66 clears the both-tiny gate and C is
    // row-major (the only orientation gemm_packed_a accepts), so this must be bit-identical
    {
        let (m, k, n) = (66usize, 4, 6);
        let a = rand_vec::<f32>(m * k, 1);
        let b = rand_vec::<f32>(k * n, 2);
        let c0 = rand_vec::<f32>(m * n, 3);
        let mut c_ref = c0.clone();
        gemm(
            1.3,
            MatRef::from_row_major(&a, m, k),
            MatRef::from_row_major(&b, k, n),
            0.5,
            MatMut::from_row_major(&mut c_ref, m, n),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_lhs(MatRef::from_row_major(&a, m, k));
        let mut c_pk = c0.clone();
        gemmkit::gemm_packed_a(
            1.3,
            &packed,
            MatRef::from_row_major(&b, k, n),
            0.5,
            MatMut::from_row_major(&mut c_pk, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c_ref, c_pk, "miri: prepack_lhs != gemm");
    }
}

/// Extra Miri coverage on the scalar engine, run alongside [`miri_scalar_path`] (the CI
/// filter matches both by the shared `miri_scalar_path` substring). Every case reaches its
/// target path through the ordinary dispatch gates at Miri-sized shapes, with no tuning-knob
/// override, so this stays safe to run alongside the rest of the correctness binary:
///
/// * `special::small_mn::pack_k_contiguous`, the small-`m,n` horizontal route's `k`-contiguous
///   scratch pack, engaged when a small-`m,n`, long-`k` shape has an operand strided along `k`:
///   an all-col-major shape packs A, an all-row-major shape packs B (`k = 20` clears
///   `small_mn_pack_min_k`, `m = n = 8` stays within `small_mn_dim`, C col-major so no
///   orientation swap redirects which operand packs). The `f32` engine never initializes
///   scratch up front, so a mis-sized or partial pack here is Miri-detectable UB
/// * `prepack_rhs_i8` + `gemm_i8_packed_b`: a small packed-B round trip, which must land on
///   the same exact `i32` bits as plain `gemm_i8`
///
/// Not duplicated here because [`miri_scalar_path`] already reaches them: `pack_panels`'s
/// `lead == 1` tail straight-copy (its `m = 5` `driver_case` packs a col-major A tail panel),
/// the driver's null-base `Regions` when nothing packs (its `m = 8` `driver_case` takes the
/// all-in-place branch), and the prepack path's uninitialized `Vec::with_capacity` + `set_len`
/// (its `f32` prepack round trips already exercise that type-generic allocation)
#[test]
fn miri_scalar_path_campaign() {
    // small_mn pack tier: all col-major packs A, then all row-major (col-major C) packs B
    run_case::<f32>(
        8,
        20,
        8,
        Layout::Col,
        Layout::Col,
        Layout::Col,
        1.3,
        0.5,
        Parallelism::Serial,
    );
    run_case::<f32>(
        8,
        20,
        8,
        Layout::Row,
        Layout::Row,
        Layout::Col,
        1.0,
        0.0,
        Parallelism::Serial,
    );
    // i8 prepacked-RHS round trip on the scalar widen engine: must equal plain gemm_i8 exactly
    #[cfg(feature = "int8")]
    {
        let (m, k, n) = (8usize, 6, 6);
        let a = rand_i8(m * k, 5);
        let b = rand_i8(k * n, 6);
        let c0: Vec<i32> = (0..m * n).map(|x| x as i32 % 3 - 1).collect();
        let mut c_ref = c0.clone();
        gemmkit::gemm_i8(
            2,
            MatRef::from_col_major(&a, m, k),
            MatRef::from_col_major(&b, k, n),
            -1,
            MatMut::from_col_major(&mut c_ref, m, n),
            Parallelism::Serial,
        );
        let packed = gemmkit::prepack_rhs_i8(MatRef::from_col_major(&b, k, n));
        let mut c_pk = c0.clone();
        gemmkit::gemm_i8_packed_b(
            2,
            MatRef::from_col_major(&a, m, k),
            &packed,
            -1,
            MatMut::from_col_major(&mut c_pk, m, n),
            Parallelism::Serial,
        );
        assert_eq!(c_ref, c_pk, "miri: i8 prepack != gemm_i8");
    }
}

/// AVX2+FMA (skipped when the running CPU lacks either), tile `MR_REG=2, NR=6`,
/// matching production dispatch for both f32 and f64
#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn isa_fma() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("skipping FMA test: not supported");
        return;
    }
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Fma, 2, 6>(Fma, m, k, n);
        driver_case::<f64, Fma, 2, 6>(Fma, m, k, n);
    }
}

/// AVX-512 (skipped when the running CPU lacks `avx512f`), tile `MR_REG=2, NR=12`,
/// matching production dispatch for both f32 and f64
#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn isa_avx512() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping AVX-512 test: not supported");
        return;
    }
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Avx512, 2, 12>(Avx512, m, k, n);
        driver_case::<f64, Avx512, 2, 12>(Avx512, m, k, n);
    }
}

/// NEON is baseline on aarch64, so it needs no feature-detection guard; tile `4x4`
/// matches production dispatch
#[test]
#[cfg(target_arch = "aarch64")]
fn isa_neon() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Neon, 4, 4>(Neon, m, k, n);
        driver_case::<f64, Neon, 4, 4>(Neon, m, k, n);
    }
}

/// wasm `simd128` is selected at compile time, not by runtime detection, so, like
/// the NEON baseline, this test needs no guard: whenever `simd128` is enabled it
/// compiles and runs. Tile `MR_REG=2, NR=4` matches production dispatch. `simd128`
/// has no hardware FMA, so its `mul_add` rounds twice rather than once; that stays
/// within `assert_accurate`'s relative-Frobenius tolerance, so no bitwise agreement
/// with other ISAs is implied
#[test]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn isa_simd128() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Simd128, 2, 4>(Simd128, m, k, n);
        driver_case::<f64, Simd128, 2, 4>(Simd128, m, k, n);
    }
}

/// `KernelSimd::store_out` (the vector narrowing store) must round every f32 lane to
/// the same bits as the scalar `NarrowFloat::narrow` (`half::from_f32`), across normals,
/// subnormals, +/-0, +/-Inf, and NaN payloads, so a full SIMD tile and a partial
/// scalar-drained tile never disagree at the boundary. AVX-512, 16 lanes wide
#[test]
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn simd_narrow_store_matches_half_avx512() {
    use gemmkit::simd::{Avx512, KernelSimd, Simd, SimdOps};
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: no avx512f");
        return;
    }
    let vals: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.0 / 3.0,
        65504.0,   // f16 max normal
        70000.0,   // rounds to +Inf in f16
        1.0e-5,    // an f16 subnormal
        1.0e-8,    // underflows to 0 in f16
        1.2340001, // exercises rounding
        2.5001,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F800001), // exponent all 1s, minimal mantissa
        f32::from_bits(0x7FFFFFFF), // exponent all 1s, mantissa all 1s
        f32::from_bits(0xFFC00000), // negative, quiet-NaN mantissa pattern
        123.456,
    ];
    // Round the count up to a whole number of 16-lane AVX-512 chunks
    let mut padded = vals.clone();
    while !padded.len().is_multiple_of(16) {
        padded.push(0.0);
    }

    unsafe {
        Avx512.vectorize(|| {
            for chunk in padded.chunks(16) {
                let reg = <Avx512 as SimdOps<f32>>::loadu(Avx512, chunk.as_ptr());
                let mut out = [gemmkit::f16::from_f32(0.0); 16];
                <Avx512 as KernelSimd<gemmkit::f16, gemmkit::f16, f32, gemmkit::f16>>::store_out(
                    Avx512,
                    out.as_mut_ptr(),
                    reg,
                );
                for (i, &v) in chunk.iter().enumerate() {
                    let scalar = gemmkit::f16::from_f32(v);
                    assert_eq!(
                        out[i].to_bits(),
                        scalar.to_bits(),
                        "f16 narrow mismatch for {v:?} (bits {:#010x})",
                        v.to_bits()
                    );
                }
            }
        });
    }
    unsafe {
        Avx512.vectorize(|| {
            for chunk in padded.chunks(16) {
                let reg = <Avx512 as SimdOps<f32>>::loadu(Avx512, chunk.as_ptr());
                let mut out = [gemmkit::bf16::from_f32(0.0); 16];
                <Avx512 as KernelSimd<gemmkit::bf16, gemmkit::bf16, f32, gemmkit::bf16>>::store_out(
                    Avx512,
                    out.as_mut_ptr(),
                    reg,
                );
                for (i, &v) in chunk.iter().enumerate() {
                    let scalar = gemmkit::bf16::from_f32(v);
                    assert_eq!(
                        out[i].to_bits(),
                        scalar.to_bits(),
                        "bf16 narrow mismatch for {v:?} (bits {:#010x})",
                        v.to_bits()
                    );
                }
            }
        });
    }
}
