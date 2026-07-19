//! Per-ISA kernels via the generic driver (column-major, no orientation needed),
//! plus the Miri scalar-path suite covering every element family

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

// per-ISA kernels via the generic driver (column-major, no orientation needed)

/// Run `FloatGemm` through the driver with an explicit ISA token + tile, all
/// column-major (rsc==1), and check accuracy. Exercises each kernel directly
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

#[test]
fn isa_scalar() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, ScalarTok, 4, 4>(ScalarTok, m, k, n);
        driver_case::<f64, ScalarTok, 4, 4>(ScalarTok, m, k, n);
    }
}

/// Miri (no SIMD intrinsics) over the scalar `SimdOps` + packing + scratch +
/// epilogue + gemv + driver. Small shapes keep Miri fast. Under Miri the runtime
/// feature detection reports nothing, so the dispatched `gemm` also takes the
/// scalar path
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
    // Safe API end-to-end (partial tiles, beta != 0, general strides)
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
    // Mixed-precision (f16 / bf16) scalar path: widen-load, f32 accumulate, narrow
    // store, plus the strided copy-back epilogue and beta != 0 read of narrow C
    #[cfg(feature = "half")]
    {
        run_case::<gemmkit::f16>(
            7,
            9,
            5,
            Layout::Row,
            Layout::Col,
            Layout::Row,
            // via `Elem::from_f64` so the alpha/beta build is Miri-safe too (routes to the
            // `*_const` software conversion under `cfg(miri)`; see the `Elem` impls above)
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
    // Complex (c32) scalar path with conj-A: the conjugate-on-pack variant + the
    // scalar complex multiply and epilogue
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
    // copy-back, and the beta != 0 i32 read of C
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
    // gemv shapes
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
    // Prepacked-RHS path on the scalar engine: bit-identical to plain gemm
    // Shape is not both-tiny (m > 64), so the prepacked geometry matches
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
    // Prepacked-LHS path on the scalar engine: bit-identical to plain gemm
    // Shape is not both-tiny (m > 64), and C is row-major (the supported
    // orientation), so the prepacked geometry matches plain gemm exactly
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

/// Miri-reachable coverage of the optimization campaign's new unsafe, on the scalar engine. The CI
/// filter matches this by the `miri_scalar_path` substring, so it runs under `cargo miri test` next
/// to [`miri_scalar_path`]. Every case reaches its target path through the **natural** gates at
/// Miri-sized shapes (no tuning-knob mutation, so this test stays safe to run concurrently with the
/// rest of the correctness binary):
///
/// * the small-`m,n` horizontal PACK tier's `k`-contiguous scratch pack
///   (`special::small_mn::pack_k_contiguous`), engaged when a small-`m,n`, long-`k` shape has an
///   operand strided along `k`: an all-col-major shape packs A, an all-row-major shape packs B
///   (`k = 20 > small_mn_pack_min_k`, `m = n = 8 <= small_mn_dim`, C col-major so no orientation
///   swap). The uninit-free `f32` engine drains it, so a mis-sized or partial pack is a Miri UB
/// * the `i8` prepack path (`prepack_rhs_i8` + `gemm_i8_packed_b`), a small packed-B round trip that
///   must stay bit-identical to plain `gemm_i8` (exact `i32`)
///
/// The remaining campaign paths are already reached by [`miri_scalar_path`] and are not duplicated:
/// `pack_panels`'s `lead == 1` tail straight-copy (its `driver_case` with `m = 5` packs a
/// col-major-A tail panel), the driver's null-base `Regions` when nothing packs (its `driver_case`
/// with `m = 8` reaches the all-in-place branch), and the prepack uninit `Vec::with_capacity` +
/// `set_len` (its `f32` prepack round trips exercise the identical type-generic alloc)
#[test]
fn miri_scalar_path_campaign() {
    // small_mn PACK tier: pack-A (all col-major) then pack-B (row-major A/B, col-major C). k above
    // the default small_mn_pack_min_k, m/n within small_mn_dim, so both hit the pack tier
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
    // i8 prepacked-RHS round trip on the scalar (widen) engine: bit-identical to plain gemm_i8
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

/// NEON is baseline on aarch64, so no feature-detection guard is needed: the
/// kernel always runs here. Tile matches the production dispatch choice (4x4)
#[test]
#[cfg(target_arch = "aarch64")]
fn isa_neon() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Neon, 4, 4>(Neon, m, k, n);
        driver_case::<f64, Neon, 4, 4>(Neon, m, k, n);
    }
}

/// wasm `simd128` is a compile-time feature (no runtime detection), so, like the
/// NEON baseline, no guard is needed: when the build enables `simd128` this test
/// is compiled and the kernel always runs. Tile matches the production dispatch
/// choice (`MR_REG=2, NR=4`). The 2-rounding `mul_add` rounds within the
/// `assert_accurate` relative-Frobenius tolerance, so no bitwise compare is implied
#[test]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn isa_simd128() {
    for (m, k, n) in isa_shapes() {
        driver_case::<f32, Simd128, 2, 4>(Simd128, m, k, n);
        driver_case::<f64, Simd128, 2, 4>(Simd128, m, k, n);
    }
}

/// The SIMD narrow-store (`KernelSimd::store_out`) must be **bit-identical** to the
/// scalar `NarrowFloat::narrow` (= `half::from_f32`) across edge values (normals,
/// subnormals, +/-0, +/-Inf, and NaN) so the full-tile vector path and the partial-tile
/// scalar path never disagree. AVX-512, 16-wide
#[test]
#[cfg(all(feature = "half", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg_attr(miri, ignore = "Miri cannot execute AVX intrinsics")]
fn simd_narrow_store_matches_half_avx512() {
    use gemmkit::simd::{Avx512, KernelSimd, Simd, SimdOps};
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("skipping: no avx512f");
        return;
    }
    // A spread of representative f32 bit patterns
    let vals: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1.0 / 3.0,
        65504.0,   // f16 max
        70000.0,   // overflows f16 -> Inf
        1.0e-5,    // f16 subnormal-ish
        1.0e-8,    // underflows to 0 in f16
        1.2340001, // rounding
        2.5001,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F800001), // sNaN-ish (exp all 1, low mantissa)
        f32::from_bits(0x7FFFFFFF), // NaN, all mantissa bits set
        f32::from_bits(0xFFC00000), // negative qNaN
        123.456,
    ];
    // Pad to a multiple of 16 lanes
    let mut padded = vals.clone();
    while !padded.len().is_multiple_of(16) {
        padded.push(0.0);
    }

    // f16
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
    // bf16
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
