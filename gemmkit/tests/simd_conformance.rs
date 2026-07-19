//! L0 SIMD-vocabulary conformance: every ISA token's [`SimdOps`] primitives, the homogeneous
//! [`KernelSimd`] `L == A` blanket impl, and the portable default `fma_bvec` must all agree
//! with a plain scalar computation, for every element type the token supports. A product
//! kernel only ever calls a subset of these primitives per token (loadu/storeu + mul_add +
//! zero/splat, plus the dot seam), so this is the only place the integer `reduce_sum`/`fnma`,
//! the mixed-precision widen seam, and the lane-FMA fallback are exercised at all
//!
//! Each token is constructed directly here, bypassing dispatch, and guarded behind its own
//! runtime feature probe, so the suite silently runs whatever subset the host actually
//! supports; it does not read `GEMMKIT_REQUIRE_ISA`. Every check compares against a scalar
//! reference computed in this file, never a platform-specific constant, so the same test runs
//! on wasm too (no proptest dependency there), which is how the compile-time `simd128` token
//! gets conformance-tested; the scalar token itself covers any host
#![cfg(not(miri))]
// The lane loops index parallel scratch buffers by lane and print the lane on failure; an
// explicit index reads clearer here than an enumerate
#![allow(clippy::needless_range_loop)]

use gemmkit::simd::{KernelSimd, ScalarTok, SimdOps};

/// Generates a conformance check for one floating element type: every `SimdOps<$t>`
/// primitive plus the `KernelSimd<$t,$t,$t,$t>` blanket, validated lane-by-lane against a
/// scalar reference at a per-type tolerance (mul_add/fnma fuse a single rounding step, so
/// the comparison cannot be bitwise)
macro_rules! float_conformance {
    ($name:ident, $t:ty, $tol:expr) => {
        fn $name<S>(simd: S, label: &str)
        where
            S: SimdOps<$t> + KernelSimd<$t, $t, $t, $t>,
        {
            let lanes = <S as SimdOps<$t>>::LANES;
            assert!(
                (1..=16).contains(&lanes),
                "{label}: implausible LANES {lanes}"
            );

            let xs: [$t; 16] = core::array::from_fn(|i| (i as $t) * 0.5 - 2.0);
            let ys: [$t; 16] = core::array::from_fn(|i| (i as $t) * -0.25 + 1.0);
            let cs: [$t; 16] = core::array::from_fn(|i| (i as $t) * 0.125 - 0.5);
            let bvals: [$t; 16] = core::array::from_fn(|i| (i as $t) * 0.3 + 0.7);

            let close = |got: $t, want: $t, op: &str, l: usize| {
                let tol = $tol * (1.0 as $t + want.abs());
                assert!(
                    (got - want).abs() <= tol,
                    "{label} {op} lane {l}: got {got} want {want}"
                );
            };

            // SAFETY: guarded by the caller's feature probe; every buffer access below stays
            // within the 16-element arrays, and runs inside this token's `vectorize` codegen
            // context so any intrinsic in the primitives below is actually feature-enabled
            unsafe {
                simd.vectorize(|| {
                    let mut out = [0.0 as $t; 16];

                    // zero / splat
                    simd.storeu(out.as_mut_ptr(), simd.zero());
                    for l in 0..lanes {
                        assert_eq!(out[l], 0.0 as $t, "{label} zero lane {l}");
                    }
                    simd.storeu(out.as_mut_ptr(), simd.splat(3.5 as $t));
                    for l in 0..lanes {
                        assert_eq!(out[l], 3.5 as $t, "{label} splat lane {l}");
                    }

                    // unaligned load round-trip
                    let vx = simd.loadu(xs.as_ptr());
                    let vy = simd.loadu(ys.as_ptr());
                    let vc = simd.loadu(cs.as_ptr());
                    simd.storeu(out.as_mut_ptr(), vx);
                    for l in 0..lanes {
                        assert_eq!(out[l], xs[l], "{label} loadu/storeu lane {l}");
                    }

                    // mul / add / mul_add / fnma
                    simd.storeu(out.as_mut_ptr(), simd.mul(vx, vy));
                    for l in 0..lanes {
                        close(out[l], xs[l] * ys[l], "mul", l);
                    }
                    simd.storeu(out.as_mut_ptr(), simd.add(vx, vy));
                    for l in 0..lanes {
                        close(out[l], xs[l] + ys[l], "add", l);
                    }
                    simd.storeu(out.as_mut_ptr(), simd.mul_add(vx, vy, vc));
                    for l in 0..lanes {
                        close(out[l], xs[l] * ys[l] + cs[l], "mul_add", l);
                    }
                    simd.storeu(out.as_mut_ptr(), simd.fnma(vx, vy, vc));
                    for l in 0..lanes {
                        close(out[l], cs[l] - xs[l] * ys[l], "fnma", l);
                    }

                    // horizontal reduce
                    let got = simd.reduce_sum(vx);
                    let want: $t = xs[..lanes].iter().copied().sum();
                    assert!(
                        (got - want).abs() <= $tol * (lanes as $t) * (1.0 as $t + want.abs()),
                        "{label} reduce_sum: got {got} want {want}"
                    );

                    // KernelSimd<A,A,A,A> blanket: the widen seam collapses to plain
                    // loadu/splat/storeu when every type parameter equals A
                    simd.storeu(
                        out.as_mut_ptr(),
                        <S as KernelSimd<$t, $t, $t, $t>>::load_lhs(simd, xs.as_ptr()),
                    );
                    for l in 0..lanes {
                        assert_eq!(out[l], xs[l], "{label} load_lhs lane {l}");
                    }
                    simd.storeu(
                        out.as_mut_ptr(),
                        <S as KernelSimd<$t, $t, $t, $t>>::splat_rhs(simd, 2.0 as $t),
                    );
                    for l in 0..lanes {
                        assert_eq!(out[l], 2.0 as $t, "{label} splat_rhs lane {l}");
                    }
                    simd.storeu(
                        out.as_mut_ptr(),
                        <S as KernelSimd<$t, $t, $t, $t>>::load_out(simd, ys.as_ptr()),
                    );
                    for l in 0..lanes {
                        assert_eq!(out[l], ys[l], "{label} load_out lane {l}");
                    }
                    <S as KernelSimd<$t, $t, $t, $t>>::store_out(simd, out.as_mut_ptr(), vc);
                    for l in 0..lanes {
                        assert_eq!(out[l], cs[l], "{label} store_out lane {l}");
                    }

                    let nrows = lanes;
                    let a_regs = [vx, vy];
                    let bvec = simd.loadu(bvals.as_ptr());
                    let mut acc: Vec<[<S as SimdOps<$t>>::Reg; 2]> = vec![[vc, vc]; nrows];
                    simd.fma_bvec::<2>(&a_regs, bvec, &mut acc);
                    for row in 0..nrows {
                        for (i, ai) in [xs, ys].iter().enumerate() {
                            simd.storeu(out.as_mut_ptr(), acc[row][i]);
                            for l in 0..lanes {
                                let want = ai[l] * bvals[row] + cs[l];
                                close(out[l], want, "fma_bvec", l);
                            }
                        }
                    }
                });
            }
        }
    };
}

float_conformance!(conform_f32, f32, 1e-4);
float_conformance!(conform_f64, f64, 1e-12);

/// Integer (`i32`) conformance: exact (wrapping) equality against a scalar reference for
/// every `SimdOps<i32>` primitive plus the `KernelSimd<i32,i32,i32,i32>` blanket. The i8
/// kernel only ever uses zero/splat/add/mul_add plus the dot seam, so load/loadu/store/
/// fnma/reduce_sum are exercised only here
#[cfg(feature = "int8")]
fn conform_i32<S>(simd: S, label: &str)
where
    S: SimdOps<i32> + KernelSimd<i32, i32, i32, i32>,
{
    let lanes = <S as SimdOps<i32>>::LANES;
    assert!(
        (1..=16).contains(&lanes),
        "{label}: implausible LANES {lanes}"
    );

    let xs: [i32; 16] = core::array::from_fn(|i| i as i32 * 7 - 40);
    let ys: [i32; 16] = core::array::from_fn(|i| i as i32 * -3 + 11);
    let cs: [i32; 16] = core::array::from_fn(|i| i as i32 * 5 - 12);
    let bvals: [i32; 16] = core::array::from_fn(|i| i as i32 * 2 + 1);

    // SAFETY: guarded by the caller's feature probe; accesses stay within the 16-element
    // arrays and run inside this token's `vectorize` codegen context
    unsafe {
        simd.vectorize(|| {
            let mut out = [0i32; 16];

            simd.storeu(out.as_mut_ptr(), simd.zero());
            for l in 0..lanes {
                assert_eq!(out[l], 0, "{label} zero lane {l}");
            }
            simd.storeu(out.as_mut_ptr(), simd.splat(1234));
            for l in 0..lanes {
                assert_eq!(out[l], 1234, "{label} splat lane {l}");
            }

            let vx = simd.loadu(xs.as_ptr());
            let vy = simd.loadu(ys.as_ptr());
            let vc = simd.loadu(cs.as_ptr());
            simd.storeu(out.as_mut_ptr(), vx);
            for l in 0..lanes {
                assert_eq!(out[l], xs[l], "{label} loadu/storeu lane {l}");
            }

            simd.storeu(out.as_mut_ptr(), simd.mul(vx, vy));
            for l in 0..lanes {
                assert_eq!(out[l], xs[l].wrapping_mul(ys[l]), "{label} mul lane {l}");
            }
            simd.storeu(out.as_mut_ptr(), simd.add(vx, vy));
            for l in 0..lanes {
                assert_eq!(out[l], xs[l].wrapping_add(ys[l]), "{label} add lane {l}");
            }
            simd.storeu(out.as_mut_ptr(), simd.mul_add(vx, vy, vc));
            for l in 0..lanes {
                assert_eq!(
                    out[l],
                    xs[l].wrapping_mul(ys[l]).wrapping_add(cs[l]),
                    "{label} mul_add lane {l}"
                );
            }
            simd.storeu(out.as_mut_ptr(), simd.fnma(vx, vy, vc));
            for l in 0..lanes {
                assert_eq!(
                    out[l],
                    cs[l].wrapping_sub(xs[l].wrapping_mul(ys[l])),
                    "{label} fnma lane {l}"
                );
            }

            let got = simd.reduce_sum(vx);
            let want = xs[..lanes].iter().copied().fold(0i32, i32::wrapping_add);
            assert_eq!(got, want, "{label} reduce_sum");

            simd.storeu(
                out.as_mut_ptr(),
                <S as KernelSimd<i32, i32, i32, i32>>::load_lhs(simd, xs.as_ptr()),
            );
            for l in 0..lanes {
                assert_eq!(out[l], xs[l], "{label} load_lhs lane {l}");
            }
            simd.storeu(
                out.as_mut_ptr(),
                <S as KernelSimd<i32, i32, i32, i32>>::splat_rhs(simd, 99),
            );
            for l in 0..lanes {
                assert_eq!(out[l], 99, "{label} splat_rhs lane {l}");
            }
            simd.storeu(
                out.as_mut_ptr(),
                <S as KernelSimd<i32, i32, i32, i32>>::load_out(simd, ys.as_ptr()),
            );
            for l in 0..lanes {
                assert_eq!(out[l], ys[l], "{label} load_out lane {l}");
            }
            <S as KernelSimd<i32, i32, i32, i32>>::store_out(simd, out.as_mut_ptr(), vc);
            for l in 0..lanes {
                assert_eq!(out[l], cs[l], "{label} store_out lane {l}");
            }

            let nrows = lanes;
            let a_regs = [vx, vy];
            let bvec = simd.loadu(bvals.as_ptr());
            let mut acc: Vec<[<S as SimdOps<i32>>::Reg; 2]> = vec![[vc, vc]; nrows];
            simd.fma_bvec::<2>(&a_regs, bvec, &mut acc);
            for row in 0..nrows {
                for (i, ai) in [xs, ys].iter().enumerate() {
                    simd.storeu(out.as_mut_ptr(), acc[row][i]);
                    for l in 0..lanes {
                        let want = ai[l].wrapping_mul(bvals[row]).wrapping_add(cs[l]);
                        assert_eq!(out[l], want, "{label} fma_bvec lane {l}");
                    }
                }
            }
        });
    }
}

/// Conformance for the deep-k twin seam `KernelSimd<N, N, f32, f32>`, the f32-output twin
/// the mixed and bf16-dot deep-k families switch to once their narrow single-panel micropanel
/// outgrows the deep-k byte gate, for a narrow type `N` (`f16`/`bf16`). Checks 2 things
/// lane-by-lane against a scalar model:
///
/// 1 `load_out`/`store_out`: with `Out == Acc == f32` the twin seam collapses to a plain f32
///    `loadu`/`storeu`, a bit-preserving identity (the scalar model is just the input bits),
///    swept over edge patterns (-0, NaN, +/-Inf) a mangled load/store would corrupt
/// 2 `load_lhs`/`splat_rhs`: the twin forwards these verbatim to the narrow seam
///    `KernelSimd<N, N, f32, N>` (which widens `N -> f32`), so each must be **bit-identical**
///    to the narrow seam's own widen - the "identical bits regardless of Out" contract the
///    driver's twin-seed / mixed-accumulate logic depends on
///
/// Requires both seams (`Out = f32` and `Out = N`), which every token that ships the narrow
/// family provides. Platform-independent: the oracle is the scalar identity or the narrow
/// seam itself, never a machine-specific number
#[cfg(feature = "half")]
fn twin_seam_conformance<N, S>(simd: S, label: &str)
where
    N: gemmkit::NarrowFloat,
    S: SimdOps<f32> + KernelSimd<N, N, f32, f32> + KernelSimd<N, N, f32, N>,
{
    let lanes = <S as SimdOps<f32>>::LANES;
    assert!(
        (1..=16).contains(&lanes),
        "{label}: implausible LANES {lanes}"
    );

    // f32 payload the plain twin load/store must reproduce bit-for-bit, including edge patterns
    let mut fs: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.5 - 2.0);
    fs[0] = -0.0;
    fs[1] = f32::NAN;
    fs[2] = f32::INFINITY;
    fs[3] = f32::NEG_INFINITY;
    // Narrow inputs for the widen-forward equivalence check
    let ns: [N; 16] = core::array::from_fn(|i| N::narrow((i as f32) * 0.25 - 1.0));

    // SAFETY: guarded by the caller's feature probe; every access stays within the 16-element
    // buffers and inside this token's `vectorize` codegen context
    unsafe {
        simd.vectorize(|| {
            let mut out = [0.0f32; 16];

            // load_out through the twin seam == a plain f32 loadu (bit-preserving)
            simd.storeu(
                out.as_mut_ptr(),
                <S as KernelSimd<N, N, f32, f32>>::load_out(simd, fs.as_ptr()),
            );
            for l in 0..lanes {
                assert_eq!(
                    out[l].to_bits(),
                    fs[l].to_bits(),
                    "{label} twin load_out lane {l}"
                );
            }

            // store_out through the twin seam == a plain f32 storeu (bit-preserving)
            let v = simd.loadu(fs.as_ptr());
            let mut out2 = [0.0f32; 16];
            <S as KernelSimd<N, N, f32, f32>>::store_out(simd, out2.as_mut_ptr(), v);
            for l in 0..lanes {
                assert_eq!(
                    out2[l].to_bits(),
                    fs[l].to_bits(),
                    "{label} twin store_out lane {l}"
                );
            }

            // load_lhs / splat_rhs: the twin forwards to the narrow seam, so the widen result
            // must be identical
            let mut tw = [0.0f32; 16];
            let mut nr = [0.0f32; 16];
            simd.storeu(
                tw.as_mut_ptr(),
                <S as KernelSimd<N, N, f32, f32>>::load_lhs(simd, ns.as_ptr()),
            );
            simd.storeu(
                nr.as_mut_ptr(),
                <S as KernelSimd<N, N, f32, N>>::load_lhs(simd, ns.as_ptr()),
            );
            for l in 0..lanes {
                assert_eq!(
                    tw[l].to_bits(),
                    nr[l].to_bits(),
                    "{label} twin load_lhs != narrow lane {l}"
                );
            }
            simd.storeu(
                tw.as_mut_ptr(),
                <S as KernelSimd<N, N, f32, f32>>::splat_rhs(simd, ns[1]),
            );
            simd.storeu(
                nr.as_mut_ptr(),
                <S as KernelSimd<N, N, f32, N>>::splat_rhs(simd, ns[1]),
            );
            for l in 0..lanes {
                assert_eq!(
                    tw[l].to_bits(),
                    nr[l].to_bits(),
                    "{label} twin splat_rhs != narrow lane {l}"
                );
            }
        });
    }
}

#[test]
fn scalar_token_conformance() {
    conform_f32(ScalarTok, "scalar/f32");
    conform_f64(ScalarTok, "scalar/f64");
    #[cfg(feature = "int8")]
    conform_i32(ScalarTok, "scalar/i32");
    #[cfg(feature = "half")]
    {
        twin_seam_conformance::<gemmkit::f16, _>(ScalarTok, "scalar/f16");
        twin_seam_conformance::<gemmkit::bf16, _>(ScalarTok, "scalar/bf16");
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_token_conformance() {
    use gemmkit::simd::{Avx512, Fma};

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        conform_f32(Fma, "fma/f32");
        conform_f64(Fma, "fma/f64");
        #[cfg(feature = "int8")]
        conform_i32(Fma, "fma/i32");
    } else {
        eprintln!("skipping Fma conformance: avx2+fma not detected");
    }

    // The Fma narrow seam widens f16 through F16C (`vcvtph2ps`), so the twin's widen-forward
    // check additionally needs f16c on top of avx2+fma; the plain-f32 load_out/store_out
    // checks do not need it, but they ride the same probe for simplicity
    #[cfg(feature = "half")]
    if is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
    {
        twin_seam_conformance::<gemmkit::f16, _>(Fma, "fma/f16");
        twin_seam_conformance::<gemmkit::bf16, _>(Fma, "fma/bf16");
    }

    if is_x86_feature_detected!("avx512f") {
        conform_f32(Avx512, "avx512/f32");
        conform_f64(Avx512, "avx512/f64");
        #[cfg(feature = "int8")]
        conform_i32(Avx512, "avx512/i32");
        #[cfg(feature = "half")]
        {
            twin_seam_conformance::<gemmkit::f16, _>(Avx512, "avx512/f16");
            twin_seam_conformance::<gemmkit::bf16, _>(Avx512, "avx512/bf16");
        }
    } else {
        eprintln!("skipping Avx512 conformance: avx512f not detected");
    }

    #[cfg(feature = "int8")]
    {
        use gemmkit::simd::Avx512Vnni;
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            conform_i32(Avx512Vnni, "avx512vnni/i32");
        } else {
            eprintln!("skipping Avx512Vnni conformance: avx512vnni not detected");
        }
    }

    #[cfg(feature = "half")]
    {
        use gemmkit::simd::Avx512Bf16;
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512f") {
            conform_f32(Avx512Bf16, "avx512bf16/f32");
            // Avx512Bf16 provides only the bf16 narrow seam (built on its `vdpbf16ps` dot),
            // so it has no f16 twin to check here
            twin_seam_conformance::<gemmkit::bf16, _>(Avx512Bf16, "avx512bf16/bf16");
        } else {
            eprintln!("skipping Avx512Bf16 conformance: avx512bf16 not detected");
        }
    }
}

/// Neon is baseline on aarch64 (no runtime probe needed) and is the only token that
/// overrides `fma_bvec` and sets `LANE_FMA`, so this is the sole conformance check of
/// that seam
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_token_conformance() {
    use gemmkit::simd::Neon;
    conform_f32(Neon, "neon/f32");
    conform_f64(Neon, "neon/f64");
    #[cfg(feature = "int8")]
    conform_i32(Neon, "neon/i32");
    #[cfg(feature = "half")]
    {
        twin_seam_conformance::<gemmkit::f16, _>(Neon, "neon/f16");
        twin_seam_conformance::<gemmkit::bf16, _>(Neon, "neon/bf16");
    }
}

/// Simd128 is a compile-time token (`+simd128`, no runtime probe) that deviates from the
/// FMA tokens by emulating `mul_add` as `mul` + `add`, so it needs its own conformance arm
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn simd128_token_conformance() {
    use gemmkit::simd::Simd128;
    conform_f32(Simd128, "simd128/f32");
    conform_f64(Simd128, "simd128/f64");
    #[cfg(feature = "int8")]
    conform_i32(Simd128, "simd128/i32");
    #[cfg(feature = "half")]
    {
        twin_seam_conformance::<gemmkit::f16, _>(Simd128, "simd128/f16");
        twin_seam_conformance::<gemmkit::bf16, _>(Simd128, "simd128/bf16");
    }
}
