//! Special-case paths (layer L6).
//!
//! These bypass the register-tiling driver for shapes where it is the wrong
//! tool. v1 ships [`gemv`] (matrix·vector); gevv (rank ≤ 2), the small-matrix
//! horizontal kernel, and batched GEMM are deferred.

pub mod gemv;
