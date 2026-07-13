//! L0 SIMD-vocabulary conformance: every ISA token's [`SimdOps`] primitives, the homogeneous
//! [`KernelSimd`] `L == A` blanket, and the portable default `fma_bvec` must agree with a scalar
//! reference for each supported element type. The product kernels only ever call a subset of these
//! per token (loadu/storeu + mul_add + zero/splat, plus the dot seam), so the integer
//! `reduce_sum`/`fnma`, the blanket widen seam, and the lane-FMA fallback are otherwise untouched by
//! any driver run — this sweep exercises them directly.
//!
//! Tokens are constructed directly (bypassing dispatch) and each is guarded behind the matching
//! runtime feature probe, so the suite runs whatever the host supports and silently skips the rest;
//! it is independent of `GEMMKIT_REQUIRE_ISA`. All arithmetic is compared against a plain scalar
//! reference computed here — no platform-specific constants. Runs on wasm too (no proptest dep) so
//! the compile-time `simd128` token is conformance-tested; the scalar token covers any host.
#![cfg(not(miri))]
// The lane loops index parallel scratch buffers by lane and print the lane in failure messages;
// an explicit index is clearer than an enumerate here.
#![allow(clippy::needless_range_loop)]

use gemmkit::simd::{KernelSimd, ScalarTok, SimdOps};

/// Generate a conformance check for one floating element type: every `SimdOps<$t>` primitive plus
/// the `KernelSimd<$t,$t,$t,$t>` blanket, validated lane-by-lane against a scalar reference at a
/// per-type tolerance (mul_add/fnma fuse a single rounding, so the match is not bitwise).
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

            // SAFETY: guarded by the caller's feature probe; every access below stays within the
            // 16-element buffers and inside this token's `vectorize` codegen context.
            unsafe {
                simd.vectorize(|| {
                    let mut out = [0.0 as $t; 16];

                    // zero / splat.
                    simd.storeu(out.as_mut_ptr(), simd.zero());
                    for l in 0..lanes {
                        assert_eq!(out[l], 0.0 as $t, "{label} zero lane {l}");
                    }
                    simd.storeu(out.as_mut_ptr(), simd.splat(3.5 as $t));
                    for l in 0..lanes {
                        assert_eq!(out[l], 3.5 as $t, "{label} splat lane {l}");
                    }

                    // unaligned load round-trip.
                    let vx = simd.loadu(xs.as_ptr());
                    let vy = simd.loadu(ys.as_ptr());
                    let vc = simd.loadu(cs.as_ptr());
                    simd.storeu(out.as_mut_ptr(), vx);
                    for l in 0..lanes {
                        assert_eq!(out[l], xs[l], "{label} loadu/storeu lane {l}");
                    }

                    // mul / add / mul_add / fnma.
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

                    // horizontal reduce.
                    let got = simd.reduce_sum(vx);
                    let want: $t = xs[..lanes].iter().copied().sum();
                    assert!(
                        (got - want).abs() <= $tol * (lanes as $t) * (1.0 as $t + want.abs()),
                        "{label} reduce_sum: got {got} want {want}"
                    );

                    // KernelSimd<A,A,A,A> blanket: widen seam collapses to plain loadu/splat/storeu.
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

                    // Default `fma_bvec`: acc[l][i] = a_regs[i]·bvec[l] + acc[l][i] (splat + mul_add).
                    // Only NEON overrides it; on x86/scalar this is the portable fallback body.
                    let nrows = core::cmp::min(lanes, 3);
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

/// Integer (`i32`) conformance: exact (wrapping) equality against a scalar reference for every
/// `SimdOps<i32>` primitive plus the `KernelSimd<i32,i32,i32,i32>` blanket. The int kernel only
/// uses zero/splat/add/mul_add + the dot seam, so load/loadu/store/fnma/reduce_sum are covered
/// here for the first time.
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

    // SAFETY: guarded by the caller's feature probe; accesses stay within the 16-element buffers
    // and inside this token's `vectorize` context.
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

            let nrows = core::cmp::min(lanes, 3);
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

#[test]
fn scalar_token_conformance() {
    conform_f32(ScalarTok, "scalar/f32");
    conform_f64(ScalarTok, "scalar/f64");
    #[cfg(feature = "int8")]
    conform_i32(ScalarTok, "scalar/i32");
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

    if is_x86_feature_detected!("avx512f") {
        conform_f32(Avx512, "avx512/f32");
        conform_f64(Avx512, "avx512/f64");
        #[cfg(feature = "int8")]
        conform_i32(Avx512, "avx512/i32");
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
        } else {
            eprintln!("skipping Avx512Bf16 conformance: avx512bf16 not detected");
        }
    }
}

/// Neon is baseline on aarch64 (no runtime probe needed) and is the only token that
/// overrides `fma_bvec` / sets `LANE_FMA`, so this arm is the sole conformance check of
/// that seam.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_token_conformance() {
    use gemmkit::simd::Neon;
    conform_f32(Neon, "neon/f32");
    conform_f64(Neon, "neon/f64");
    #[cfg(feature = "int8")]
    conform_i32(Neon, "neon/i32");
}

/// Simd128 is a compile-time token (`+simd128`, no runtime probe) and deviates from the
/// FMA tokens (`mul_add` emulated as `mul + add`), so it needs its own conformance arm.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn simd128_token_conformance() {
    use gemmkit::simd::Simd128;
    conform_f32(Simd128, "simd128/f32");
    conform_f64(Simd128, "simd128/f64");
    #[cfg(feature = "int8")]
    conform_i32(Simd128, "simd128/i32");
}
