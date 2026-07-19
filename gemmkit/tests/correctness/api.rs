//! Safe-API panic guarantees and cache-topology sanity

use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// cache-topology detection

/// The detected `L1d`/`L2` sizes fall in a plausible range for real hardware, and
/// `blocking()` returns a sane `(mc, kc, nc)` for a 4096-cubed problem with a 32x12,
/// 4-byte-element tile (mc a multiple of the 32-wide MR)
#[test]
fn cache_topology_is_plausible() {
    let t = gemmkit::topology();
    assert!(
        t.l1d.bytes >= 8 * 1024 && t.l1d.bytes <= 256 * 1024,
        "L1d={}",
        t.l1d.bytes
    );
    assert!(t.l1d.line >= 32 && t.l1d.line <= 256);
    assert!(t.l2.bytes >= 128 * 1024, "L2={}", t.l2.bytes);
    let blk = t.blocking(32, 12, 4, 4096, 4096, 4096);
    assert!(blk.mc >= 32 && blk.kc >= 1 && blk.nc >= 12);
    assert!(blk.mc.is_multiple_of(32), "mc should be a multiple of MR");
    eprintln!(
        "topology: L1d={}K L2={}K L3={:?}K  blocking(4096³): mc={} kc={} nc={}",
        t.l1d.bytes / 1024,
        t.l2.bytes / 1024,
        t.l3.map(|l| l.bytes / 1024),
        blk.mc,
        blk.kc,
        blk.nc
    );
}

// safe-API panic guarantees

/// `B.rows (2) != A.cols (3)`: the shape-compatibility check must reject it and
/// name `A.cols` in the panic message
#[test]
#[should_panic(expected = "A.cols")]
fn panic_shape_mismatch() {
    let a = vec![0.0f32; 6];
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 2, 3),
        0.0,
        MatMut::from_row_major(&mut c, 2, 3),
        Parallelism::Serial,
    );
}

/// `a` backs a 2x3 view with `rs=3` but holds only 3 elements, 1 row short of what
/// the view needs: the bounds check must catch it before any out-of-bounds read
#[test]
#[should_panic(expected = "needs")]
fn panic_out_of_bounds_view() {
    let a = vec![0.0f32; 3];
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 4];
    gemm(
        1.0,
        MatRef::new(&a, 2, 3, 3, 1),
        MatRef::from_row_major(&b, 3, 2),
        0.0,
        MatMut::from_row_major(&mut c, 2, 2),
        Parallelism::Serial,
    );
}

/// `(rows-1)*rs` is built to overflow `isize` on the target, so the view's extent
/// cannot be addressed: must panic rather than wrap or read out of bounds
#[test]
#[should_panic(expected = "too large to address")]
fn panic_extent_overflow_view() {
    let a = vec![0.0f32; 1];
    let b = vec![0.0f32; 1];
    let mut c = vec![0.0f32; 1];
    let half = isize::BITS / 2;
    let rows = (1usize << (half + 1)) + 1;
    let rs = 1isize << half;
    gemm(
        1.0,
        MatRef::new(&a, rows, 1, rs, 1),
        MatRef::from_row_major(&b, 1, 1),
        0.0,
        MatMut::new(&mut c, rows, 1, rs, 1),
        Parallelism::Serial,
    );
}

/// `rsc == 0` maps every row of C onto the same memory: the extent check alone
/// would accept it (nothing falls outside the buffer), so a dedicated
/// self-aliasing check must catch it, since `Rayon` workers would otherwise race on
/// that shared row
#[test]
#[should_panic(expected = "aliases itself")]
fn panic_self_aliasing_c() {
    let a = vec![0.0f32; 8];
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 3];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 4, 2),
        MatRef::from_row_major(&b, 2, 3),
        0.0,
        MatMut::new(&mut c, 4, 3, 0, 1),
        Parallelism::Rayon(0),
    );
}

/// A and C share one buffer, which the safe `MatRef`/`MatMut` borrow rules can
/// never let through, so the alias is built with raw pointers instead: must still
/// be caught, proving the check inspects the resulting views, not the borrows used
/// to build them
#[test]
#[should_panic(expected = "aliases")]
fn panic_c_aliases_a() {
    let mut buf = vec![1.0f32; 16];
    // A no-op split: split_at_mut can only yield disjoint slices, so this cannot
    // itself produce an alias; the raw pointers below are what actually do
    let (a_part, c_part) = buf.split_at_mut(0);
    let _ = (a_part, c_part);
    let ptr = buf.as_mut_ptr();
    let len = buf.len();
    unsafe {
        let a_slice = std::slice::from_raw_parts(ptr, len);
        let c_slice = std::slice::from_raw_parts_mut(ptr, len);
        gemm(
            1.0,
            MatRef::from_row_major(a_slice, 2, 2),
            MatRef::from_row_major(a_slice, 2, 2),
            0.0,
            MatMut::from_row_major(c_slice, 2, 2),
            Parallelism::Serial,
        );
    }
}
