//! Shared tiny-GEMM entry points, one per element-type dispatch ladder, for the
//! `GEMMKIT_REQUIRE_ISA` cross-arch pin binaries (`env_isa_neon`, `env_isa_wasm`). Each runs a
//! 2×2 product through the public entry for its dtype; the pin binaries wrap them in
//! `catch_unwind` to observe whether the pinned ISA's `select_*` ladder accepts or rejects the
//! current target. Feature-gated dtypes are simply absent from `dtype_cases` when their feature
//! is off. Lives in a subdirectory so cargo does not compile it as its own test binary.

use gemmkit::{MatMut, MatRef, Parallelism};

fn gemm_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];
    gemmkit::gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

fn gemm_f64() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [5.0f64, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];
    gemmkit::gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[cfg(feature = "half")]
fn gemm_f16() {
    use gemmkit::f16;
    let a = [f16::from_f32(1.0); 4];
    let b = [f16::from_f32(2.0); 4];
    let mut c = [f16::from_f32(0.0); 4];
    gemmkit::gemm(
        f16::from_f32(1.0),
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        f16::from_f32(0.0),
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[cfg(feature = "half")]
fn gemm_bf16() {
    use gemmkit::bf16;
    let a = [bf16::from_f32(1.0); 4];
    let b = [bf16::from_f32(2.0); 4];
    let mut c = [bf16::from_f32(0.0); 4];
    gemmkit::gemm(
        bf16::from_f32(1.0),
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        bf16::from_f32(0.0),
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[cfg(feature = "int8")]
fn gemm_i8() {
    let a = [1i8, 2, 3, 4];
    let b = [5i8, 6, 7, 8];
    let mut c = [0i32; 4];
    gemmkit::gemm_i8(
        1,
        MatRef::from_row_major(&a, 2, 2),
        MatRef::from_row_major(&b, 2, 2),
        0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[cfg(feature = "complex")]
fn gemm_c32() {
    use gemmkit::Complex;
    let a = [Complex::new(1.0f32, 0.5); 4];
    let b = [Complex::new(2.0f32, -0.5); 4];
    let mut c = [Complex::new(0.0f32, 0.0); 4];
    gemmkit::gemm_cplx(
        Complex::new(1.0, 0.0),
        MatRef::from_row_major(&a, 2, 2),
        false,
        MatRef::from_row_major(&b, 2, 2),
        false,
        Complex::new(0.0, 0.0),
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

#[cfg(feature = "complex")]
fn gemm_c64() {
    use gemmkit::Complex;
    let a = [Complex::new(1.0f64, 0.5); 4];
    let b = [Complex::new(2.0f64, -0.5); 4];
    let mut c = [Complex::new(0.0f64, 0.0); 4];
    gemmkit::gemm_cplx(
        Complex::new(1.0, 0.0),
        MatRef::from_row_major(&a, 2, 2),
        false,
        MatRef::from_row_major(&b, 2, 2),
        false,
        Complex::new(0.0, 0.0),
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

/// Every element-type dispatch ladder available under the active feature set, as
/// `(name, entry)`. `f32`/`f64` are always present; the rest track their features.
pub fn dtype_cases() -> Vec<(&'static str, fn())> {
    vec![
        ("f32", gemm_f32 as fn()),
        ("f64", gemm_f64 as fn()),
        #[cfg(feature = "half")]
        ("f16", gemm_f16 as fn()),
        #[cfg(feature = "half")]
        ("bf16", gemm_bf16 as fn()),
        #[cfg(feature = "int8")]
        ("i8", gemm_i8 as fn()),
        #[cfg(feature = "complex")]
        ("c32", gemm_c32 as fn()),
        #[cfg(feature = "complex")]
        ("c64", gemm_c64 as fn()),
    ]
}
