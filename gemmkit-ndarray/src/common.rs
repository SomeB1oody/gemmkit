//! Shared dims/strides extraction for the entry modules. The bias/requant validation the
//! epilogue-gated entries need lives once in gemmkit's `adapter` module (a raw-pointer-level
//! surface shared with gemmkit's own checked entries), which those modules import and reuse
//! rather than keeping a local copy
use super::*;

#[inline]
pub(crate) fn dims_strides<T, S: Data<Elem = T>>(
    a: &ArrayBase<S, Ix2>,
) -> (usize, usize, isize, isize) {
    let (r, c) = a.dim();
    let s = a.strides();
    (r, c, s[0], s[1])
}
