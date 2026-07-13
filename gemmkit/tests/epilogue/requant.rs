//! Requantize (i8 -> i8) fused-epilogue tests: bitwise vs `gemm_i8`-then-map, round-half-to-even
//! ties, saturation, bias, and checked/unchecked twin equivalence. The `i32` accumulation is exact
//! and ISA-independent, so the oracle holds bitwise under every `GEMMKIT_REQUIRE_ISA` pin.

use crate::common::Rng;
use gemmkit::{MatMut, MatRef, Parallelism, Requantize, gemm_i8, gemm_i8_requant};

/// The reference requantize map. The rounding uses the std `round_ties_even` — an
/// *independent* implementation of the contract, NOT a copy of the kernel's `2^52`
/// `round_ne_f64` — so a regression in the kernel's rounding is caught here rather than
/// mirrored. Applied to the exact `i32` accumulator from `gemm_i8`.
fn ref_requant(acc: i32, bias: i32, scale: f32, zp: i32) -> i8 {
    let scaled = (f64::from(acc.wrapping_add(bias)) * f64::from(scale)).round_ties_even();
    let q = (scaled as i64).saturating_add(i64::from(zp));
    q.clamp(-128, 127) as i8
}

fn make_i8(rng: &mut Rng, n: usize) -> Vec<i8> {
    (0..n)
        .map(|_| ((rng.next_u64() % 255) as i64 - 127) as i8)
        .collect()
}

/// Bitwise: `gemm_i8_requant` == `gemm_i8` (into i32) then the scalar requant map. Since
/// the `i32` accumulation is exact and ISA-independent, this holds under any ISA pin.
fn check_requant(
    rng: &mut Rng,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    zp: i32,
    has_bias: bool,
    row_major_c: bool,
    par: Parallelism,
    tag: &str,
) {
    let a = make_i8(rng, m * k); // col-major m×k
    let b = make_i8(rng, k * n); // col-major k×n
    let bias: Vec<i32> = if has_bias {
        (0..m)
            .map(|_| (rng.next_u64() % 2001) as i64 as i32 - 1000)
            .collect()
    } else {
        Vec::new()
    };
    let (rsc, csc) = if row_major_c {
        (n as isize, 1isize)
    } else {
        (1isize, m as isize)
    };

    // exact i32 accumulator via gemm_i8
    let mut acc = vec![0i32; m * n];
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut acc, m, n, rsc, csc);
        gemm_i8(1, ar, br, 0, cm, par);
    }

    // fused requantize
    let mut c = vec![0i8; m * n];
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c, m, n, rsc, csc);
        let req = Requantize {
            scale,
            zero_point: zp,
            bias: if has_bias { Some(&bias) } else { None },
        };
        gemm_i8_requant(ar, br, req, cm, par);
    }

    for j in 0..n {
        for i in 0..m {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let bterm = if has_bias { bias[i] } else { 0 };
            let want = ref_requant(acc[idx], bterm, scale, zp);
            assert_eq!(
                c[idx], want,
                "{tag}: requant mismatch at ({i},{j}) acc={} [m={m} k={k} n={n}]",
                acc[idx],
            );
        }
    }
}

#[test]
fn requant_bitwise_matrix() {
    let mut rng = Rng::new(0x9111);
    for &(m, k, n) in &[(17usize, 20usize, 19usize), (32, 40, 24), (48, 300, 33)] {
        for &scale in &[0.003f32, 0.5, 1.0, 7.25] {
            for &zp in &[-128i32, -13, 0, 27, 127] {
                for has_bias in [false, true] {
                    for row_major in [false, true] {
                        for par in [Parallelism::Serial, Parallelism::Rayon(8)] {
                            check_requant(
                                &mut rng, m, k, n, scale, zp, has_bias, row_major, par, "matrix",
                            );
                        }
                    }
                }
            }
        }
    }
}

/// `gemm_i8_requant` and `gemm_i8_requant_unchecked` are **parallel** entry points (the
/// checked twin does not delegate to the unchecked one), so exercise the unchecked fn
/// against the checked twin bit-for-bit on a driver-shaped case (m,n,k > 16, with bias).
#[test]
fn requant_unchecked_matches_checked() {
    use gemmkit::gemm_i8_requant_unchecked;

    let mut rng = Rng::new(0x5EED_1234);
    let (m, k, n) = (32usize, 40usize, 24usize);
    let a = make_i8(&mut rng, m * k);
    let b = make_i8(&mut rng, k * n);
    let bias: Vec<i32> = (0..m)
        .map(|_| (rng.next_u64() % 2001) as i64 as i32 - 1000)
        .collect();
    let (scale, zp) = (0.5f32, 13i32);
    let (rsc, csc) = (1isize, m as isize);
    let par = Parallelism::Serial;

    let mut c_checked = vec![0i8; m * n];
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c_checked, m, n, rsc, csc);
        let req = Requantize {
            scale,
            zero_point: zp,
            bias: Some(&bias),
        };
        gemm_i8_requant(ar, br, req, cm, par);
    }

    let mut c_unchecked = vec![0i8; m * n];
    // SAFETY: valid in-bounds col-major layouts; C aliases neither A/B nor the (per-row,
    // length-m) bias.
    unsafe {
        gemm_i8_requant_unchecked(
            m,
            k,
            n,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            1,
            k as isize,
            scale,
            zp,
            bias.as_ptr(),
            true,
            c_unchecked.as_mut_ptr(),
            rsc,
            csc,
            par,
        );
    }

    assert_eq!(
        c_checked, c_unchecked,
        "requant unchecked != checked [m={m} k={k} n={n}]"
    );
}

/// Hardcoded round-half-to-even ties, independent of any reference function: each row is a
/// `1×1` product giving an exact `acc`, and `scale = 0.5` lands `scale·acc` on a half-integer.
/// A round-half-up/away regression would flip 0.5→1, 2.5→3, etc.
#[test]
fn requant_ties_even_exact() {
    let a: [i8; 6] = [1, 3, 5, 7, -1, -3];
    let b: [i8; 1] = [1];
    // scale=0.5: 0.5→0, 1.5→2, 2.5→2, 3.5→4, -0.5→0, -1.5→-2 (ties to even).
    let expect: [i8; 6] = [0, 2, 2, 4, 0, -2];
    let mut c = [0i8; 6];
    gemm_i8_requant(
        MatRef::from_col_major(&a, 6, 1),
        MatRef::from_col_major(&b, 1, 1),
        Requantize {
            scale: 0.5,
            zero_point: 0,
            bias: None,
        },
        MatMut::from_col_major(&mut c, 6, 1),
        Parallelism::Serial,
    );
    assert_eq!(c, expect, "round-half-to-even ties");
}

/// Round-half-to-even ties (incl. odd zero-point) and saturation both ends.
#[test]
fn requant_ties_and_saturation() {
    let mut rng = Rng::new(0x7135);
    // A large scale drives many outputs to the ±clamp; a range of k exercises exact-tie
    // half-integers as scale*acc lands on x.5.
    for &k in &[1usize, 8, 300, 1000, 5000] {
        check_requant(
            &mut rng,
            20,
            k,
            18,
            0.5,
            3,
            true,
            false,
            Parallelism::Serial,
            "ties",
        );
        check_requant(
            &mut rng,
            20,
            k,
            18,
            100.0,
            120,
            false,
            false,
            Parallelism::Serial,
            "sat+",
        );
        check_requant(
            &mut rng,
            20,
            k,
            18,
            100.0,
            -120,
            true,
            true,
            Parallelism::Rayon(8),
            "sat-",
        );
    }
}

/// Small mnk under Rayon(8): exercises the auto-VNNI small-parallel fallback to the widen
/// `IntGemmQ` (bit-exact-equal), so the fused output still matches the oracle.
#[test]
fn requant_small_parallel_fallback() {
    let mut rng = Rng::new(0xFA11);
    check_requant(
        &mut rng,
        24,
        24,
        24,
        0.01,
        5,
        true,
        false,
        Parallelism::Rayon(8),
        "small-par",
    );
}

/// Run `gemm_i8_requant` (col-major `m×k` A, `k×n` B) into a C with the given strides and
/// return the (possibly padded) `i8` buffer of length `buflen`.
fn run_requant(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
    zp: i32,
    bias: Option<&[i32]>,
    rsc: isize,
    csc: isize,
    buflen: usize,
    par: Parallelism,
) -> Vec<i8> {
    let mut c = vec![0i8; buflen];
    let ar = MatRef::new(a, m, k, 1, m as isize);
    let br = MatRef::new(b, k, n, 1, k as isize);
    let cm = MatMut::new(&mut c, m, n, rsc, csc);
    let req = Requantize {
        scale,
        zero_point: zp,
        bias,
    };
    gemm_i8_requant(ar, br, req, cm, par);
    c
}

/// Phase 4: the vectorized requant map (unit-stride C, full lane-runs) must agree **bit-for-bit**
/// with the scalar map (a strided C forces the scalar path for every element). `m = 64` spans
/// several full lane-runs on every vector ISA; a `PerRow` bias including `i32::MAX`/`i32::MIN`
/// exercises the wrapping integer bias-add on both paths. Independent of any platform constant —
/// the two layouts are each other's oracle.
#[test]
fn requant_vector_scalar_bitwise() {
    let mut rng = Rng::new(0xB17E_5CA1);
    let (m, k, n) = (64usize, 37usize, 9usize);
    let a = make_i8(&mut rng, m * k);
    let b = make_i8(&mut rng, k * n);
    // Per-row bias including the wrapping-add extremes.
    let bias: Vec<i32> = (0..m)
        .map(|i| match i % 5 {
            0 => i32::MAX,
            1 => i32::MIN,
            2 => 1000,
            3 => -1000,
            _ => (rng.next_u64() % 4001) as i64 as i32 - 2000,
        })
        .collect();

    for has_bias in [false, true] {
        let bias_opt = if has_bias {
            Some(bias.as_slice())
        } else {
            None
        };
        for &scale in &[1.0f32, 0.0078125, 0.1, 1e30, 1e-30] {
            for &zp in &[0i32, -128, 127] {
                // (1) unit-stride col-major C: full lane-runs take the vector store path.
                let unit = run_requant(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    scale,
                    zp,
                    bias_opt,
                    1,
                    m as isize,
                    m * n,
                    Parallelism::Serial,
                );
                // (2) strided C (rsc = 2): forces the scalar map for every element.
                let strided = run_requant(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    scale,
                    zp,
                    bias_opt,
                    2,
                    (2 * m) as isize,
                    2 * m * n,
                    Parallelism::Serial,
                );
                for j in 0..n {
                    for i in 0..m {
                        assert_eq!(
                            unit[i + j * m],
                            strided[2 * i + j * (2 * m)],
                            "vector != scalar has_bias={has_bias} scale={scale} zp={zp} at ({i},{j})",
                        );
                    }
                }
            }
        }
    }
}

/// Phase 4: round-half-to-even ties through the **vector** path. `m = 37 (>= 32)` with unit-stride
/// C means full lane-runs hit the vector store on every vector ISA (plus a sub-lane tail); `1×1`
/// products with `scale = 0.5` land `scale·acc` on exact half-integers for odd `acc`. Asserted
/// against `ref_requant` (std `round_ties_even`, independent of the kernel), so a round-half-up/
/// away regression in the vector path flips the tie.
#[test]
fn requant_vector_ties() {
    let m = 37usize;
    let (k, n) = (1usize, 1usize);
    // `acc[i] = a[i]·1`; spread across both parities so odd `acc` gives x.5 ties.
    let a: Vec<i8> = (0..m).map(|i| (i as i32 - 18) as i8).collect();
    let b: [i8; 1] = [1];
    let scale = 0.5f32;
    let zp = 3i32; // odd zero-point joins in integer after the round
    let mut c = vec![0i8; m * n];
    {
        let ar = MatRef::new(&a, m, k, 1, m as isize);
        let br = MatRef::new(&b, k, n, 1, k as isize);
        let cm = MatMut::new(&mut c, m, n, 1, m as isize);
        gemm_i8_requant(
            ar,
            br,
            Requantize {
                scale,
                zero_point: zp,
                bias: None,
            },
            cm,
            Parallelism::Serial,
        );
    }
    for i in 0..m {
        let want = ref_requant(a[i] as i32, 0, scale, zp);
        assert_eq!(c[i], want, "ties vector path at row {i} acc={}", a[i]);
    }
}

/// Phase 4: accumulator + bias driven into the `i32` wrapping / f64-saturation corners of the
/// map (`a`/`b` in `{-128, 127}`, a per-row bias at the `i32` extremes) must requantize
/// identically through the vector (unit-stride) and scalar (strided) paths. Covers the
/// `t >= 2^52 -> hi` / `t <= -2^52 -> lo` clamp branches (via `scale = 1e30`).
#[test]
fn requant_vector_extreme_acc() {
    let (m, k, n) = (37usize, 1usize, 7usize);
    let a: Vec<i8> = (0..m * k)
        .map(|i| if i % 2 == 0 { -128 } else { 127 })
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|j| if j % 2 == 0 { 127 } else { -128 })
        .collect();
    let bias: Vec<i32> = (0..m)
        .map(|i| match i % 4 {
            0 => i32::MAX,
            1 => i32::MIN,
            2 => i32::MAX - 200,
            _ => i32::MIN + 200,
        })
        .collect();
    for &scale in &[1.0f32, 1e-9, 0.5, 1e30] {
        for &zp in &[0i32, -128, 127] {
            let unit = run_requant(
                &a,
                &b,
                m,
                k,
                n,
                scale,
                zp,
                Some(&bias),
                1,
                m as isize,
                m * n,
                Parallelism::Serial,
            );
            let strided = run_requant(
                &a,
                &b,
                m,
                k,
                n,
                scale,
                zp,
                Some(&bias),
                2,
                (2 * m) as isize,
                2 * m * n,
                Parallelism::Serial,
            );
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        unit[i + j * m],
                        strided[2 * i + j * (2 * m)],
                        "extreme vector != scalar scale={scale} zp={zp} at ({i},{j})",
                    );
                }
            }
        }
    }
}

/// Degenerate `k == 0`: C fills with `clamp(zp + round_ne(scale*bias))` (= `zp` without
/// bias).
#[test]
fn requant_degenerate_k0() {
    let m = 12usize;
    let n = 10usize;
    let bias: Vec<i32> = (0..m).map(|i| i as i32 * 40 - 200).collect();
    let a: Vec<i8> = Vec::new();
    let b: Vec<i8> = Vec::new();
    let scale = 0.5f32;
    let zp = 7i32;
    let mut c = vec![99i8; m * n];
    {
        let ar = MatRef::new(&a, m, 0, 1, m as isize);
        let br = MatRef::new(&b, 0, n, 1, 0);
        let cm = MatMut::new(&mut c, m, n, 1, m as isize);
        let req = Requantize {
            scale,
            zero_point: zp,
            bias: Some(&bias),
        };
        gemm_i8_requant(ar, br, req, cm, Parallelism::Serial);
    }
    for j in 0..n {
        for i in 0..m {
            let want = ref_requant(0, bias[i], scale, zp);
            assert_eq!(c[i + j * m], want, "degenerate requant ({i},{j})");
        }
    }
}

#[test]
#[should_panic(expected = "scale")]
fn requant_bad_scale() {
    let a = vec![0i8; 16];
    let b = vec![0i8; 16];
    let mut c = vec![0i8; 16];
    let req = Requantize {
        scale: 0.0,
        zero_point: 0,
        bias: None,
    };
    gemm_i8_requant(
        MatRef::from_col_major(&a, 4, 4),
        MatRef::from_col_major(&b, 4, 4),
        req,
        MatMut::from_col_major(&mut c, 4, 4),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "zero_point")]
fn requant_bad_zp() {
    let a = vec![0i8; 16];
    let b = vec![0i8; 16];
    let mut c = vec![0i8; 16];
    let req = Requantize {
        scale: 1.0,
        zero_point: 200,
        bias: None,
    };
    gemm_i8_requant(
        MatRef::from_col_major(&a, 4, 4),
        MatRef::from_col_major(&b, 4, 4),
        req,
        MatMut::from_col_major(&mut c, 4, 4),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "bias length")]
fn requant_bad_bias_len() {
    let a = vec![0i8; 16];
    let b = vec![0i8; 16];
    let mut c = vec![0i8; 16];
    let bias = vec![0i32; 3];
    let req = Requantize {
        scale: 1.0,
        zero_point: 0,
        bias: Some(&bias),
    };
    gemm_i8_requant(
        MatRef::from_col_major(&a, 4, 4),
        MatRef::from_col_major(&b, 4, 4),
        req,
        MatMut::from_col_major(&mut c, 4, 4),
        Parallelism::Serial,
    );
}
