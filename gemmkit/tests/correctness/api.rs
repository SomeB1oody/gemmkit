//! Safe-API panic guarantees and cache-topology sanity.

use gemmkit::{MatMut, MatRef, Parallelism, gemm};

// ---------------------------------------------------------------------------
// cache detection (§7.5)
// ---------------------------------------------------------------------------

#[test]
fn cache_topology_is_plausible() {
    let t = gemmkit::topology();
    // Sane bounds (true on any real CPU).
    assert!(
        t.l1d.bytes >= 8 * 1024 && t.l1d.bytes <= 256 * 1024,
        "L1d={}",
        t.l1d.bytes
    );
    assert!(t.l1d.line >= 32 && t.l1d.line <= 256);
    assert!(t.l2.bytes >= 128 * 1024, "L2={}", t.l2.bytes);
    // Blocking parameters are sane for a big problem.
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

// ---------------------------------------------------------------------------
// safe-API panic guarantees
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "A.cols")]
fn panic_shape_mismatch() {
    let a = vec![0.0f32; 6];
    let b = vec![0.0f32; 6];
    let mut c = vec![0.0f32; 4];
    gemm(
        1.0,
        MatRef::from_row_major(&a, 2, 3),
        MatRef::from_row_major(&b, 2, 3), // B.rows=2 != A.cols=3
        0.0,
        MatMut::from_row_major(&mut c, 2, 3),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "needs")]
fn panic_out_of_bounds_view() {
    let a = vec![0.0f32; 3]; // too small for 2x3
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

#[test]
#[should_panic(expected = "too large to address")]
fn panic_extent_overflow_view() {
    let a = vec![0.0f32; 1];
    let b = vec![0.0f32; 1];
    let mut c = vec![0.0f32; 1];
    let half = isize::BITS / 2;
    let rows = (1usize << (half + 1)) + 1;
    let rs = 1isize << half; // (rows-1)*rs = 2^(2·half+1), overflows isize on the target
    gemm(
        1.0,
        MatRef::new(&a, rows, 1, rs, 1),
        MatRef::from_row_major(&b, 1, 1),
        0.0,
        MatMut::new(&mut c, rows, 1, rs, 1),
        Parallelism::Serial,
    );
}

#[test]
#[should_panic(expected = "aliases itself")]
fn panic_self_aliasing_c() {
    // rsc == 0 collapses all rows of C onto the same memory — accepted by the
    // bounds check (extent fits) but a data race in parallel. Must panic.
    let a = vec![0.0f32; 8]; // 4x2
    let b = vec![0.0f32; 6]; // 2x3
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

#[test]
#[should_panic(expected = "aliases")]
fn panic_c_aliases_a() {
    // Force an alias via raw slices over the same buffer through MatRef/MatMut.
    let mut buf = vec![1.0f32; 16];
    let (a_part, c_part) = buf.split_at_mut(0); // a_part empty? need overlap
    let _ = (a_part, c_part);
    // Build overlapping views by unsafe transmute of lifetimes is messy; instead
    // use the same slice for A and C via raw pointers is not possible in safe
    // API. We simulate by pointing both at `buf` through separate borrows is
    // disallowed; so construct via std::slice::from_raw_parts.
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
