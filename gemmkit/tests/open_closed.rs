//! Architecture acceptance (section 7.3, strongest claim): a *2nd* `KernelFamily`
//! can be declared using only gemmkit's **public** API and driven by the same
//! generic `driver::run`, with **no change** to `driver.rs` or `pack.rs`. The
//! mere fact that this external file compiles and produces correct results is
//! the proof that the operation-family seam is open for extension

use gemmkit::kernel::{AlphaStatus, BetaStatus, KernelFamily};
use gemmkit::scalar::Scalar;
use gemmkit::simd::{ScalarTok, SimdOps};
use gemmkit::{Parallelism, Workspace, driver};

/// A deliberately naive, independently-implemented float GEMM family. It shares
/// nothing with the built-in `FloatGemm` except the public trait it satisfies:
/// its own micropanel packing and a plain scalar microkernel
#[derive(Copy, Clone)]
struct NaiveFloat;

/// Re-implement micropanel-major packing using only public items (the crate's
/// internal `pack` helper is not visible here, exactly the third-party case)
unsafe fn pack_panels(
    mut d: *mut f32,
    src: *const f32,
    lead: isize,
    depth: isize,
    n_lead: usize,
    depth_len: usize,
    width: usize,
) {
    unsafe {
        let mut base = 0;
        while base < n_lead {
            let live = core::cmp::min(width, n_lead - base);
            for p in 0..depth_len {
                let col = src.offset(p as isize * depth);
                for i in 0..width {
                    *d = if i < live {
                        *col.offset((base + i) as isize * lead)
                    } else {
                        0.0
                    };
                    d = d.add(1);
                }
            }
            base += width;
        }
    }
}

impl KernelFamily for NaiveFloat {
    type Lhs = f32;
    type Rhs = f32;
    type Acc = f32;
    type Out = f32;

    unsafe fn pack_lhs(
        dst: *mut f32,
        src: *const f32,
        rs: isize,
        cs: isize,
        mc: usize,
        kc: usize,
        mr: usize,
    ) {
        unsafe { pack_panels(dst, src, rs, cs, mc, kc, mr) }
    }

    unsafe fn pack_rhs(
        dst: *mut f32,
        src: *const f32,
        rs: isize,
        cs: isize,
        kc: usize,
        nc: usize,
        nr: usize,
    ) {
        unsafe { pack_panels(dst, src, cs, rs, nc, kc, nr) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn microkernel<S, const MR_REG: usize, const NR: usize>(
        _simd: S,
        kc: usize,
        alpha: f32,
        beta: f32,
        ast: AlphaStatus,
        bst: BetaStatus,
        a: *const f32,
        a_cs: isize,
        b: *const f32,
        b_rs: isize,
        b_cs: isize,
        c: *mut f32,
        rsc: isize,
        csc: isize,
        mr_eff: usize,
        nr_eff: usize,
        _scratch: *mut f32,
    ) where
        S: SimdOps<f32>,
    {
        unsafe {
            for j in 0..nr_eff {
                for i in 0..mr_eff {
                    let mut acc = 0.0f32;
                    for p in 0..kc {
                        acc += *a.offset(p as isize * a_cs + i as isize)
                            * *b.offset(p as isize * b_rs + j as isize * b_cs);
                    }
                    let prod = match ast {
                        AlphaStatus::One => acc,
                        AlphaStatus::Other => alpha * acc,
                    };
                    let cp = c.offset(i as isize * rsc + j as isize * csc);
                    *cp = match bst {
                        BetaStatus::Zero => prod,
                        BetaStatus::One => *cp + prod,
                        BetaStatus::Other => beta * *cp + prod,
                    };
                }
            }
        }
    }
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = s;
        }
    }
    c
}

#[test]
fn second_kernel_family_drives_unchanged() {
    let (m, k, n) = (40usize, 33, 28);
    // Row-major logical inputs
    let a: Vec<f32> = (0..m * k).map(|x| (x % 17) as f32 * 0.1 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x % 13) as f32 * 0.2 - 0.7).collect();
    let cref = reference(&a, &b, m, k, n);

    // Present everything column-major so the driver needs no orientation
    let to_col = |v: &[f32], r: usize, c: usize| {
        let mut o = vec![0.0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = v[i * c + j];
            }
        }
        o
    };
    let acol = to_col(&a, m, k);
    let bcol = to_col(&b, k, n);
    let mut ccol = vec![0.0f32; m * n];

    let mut ws = Workspace::new();
    unsafe {
        driver::run::<NaiveFloat, ScalarTok, 4, 4>(
            ScalarTok,
            m,
            k,
            n,
            f32::ONE,
            acol.as_ptr(),
            1,
            m as isize,
            bcol.as_ptr(),
            1,
            k as isize,
            f32::ZERO,
            ccol.as_mut_ptr(),
            1,
            m as isize,
            Parallelism::Serial,
            &mut ws,
        );
    }

    for i in 0..m {
        for j in 0..n {
            let got = ccol[j * m + i] as f64;
            let exp = cref[i * n + j];
            assert!(
                (got - exp).abs() <= 1e-4 * (1.0 + exp.abs()),
                "mismatch at ({i},{j})"
            );
        }
    }
}
