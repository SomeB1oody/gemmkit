//! Special-case paths (layer L6).
//!
//! These bypass the register-tiling driver for shapes where it is the wrong tool: [`gemv`]
//! (matrix·vector) and [`small_k`] (skinny / low-depth GEMM — gevv, rank-`k`, tall-skinny).
//! The small-matrix horizontal kernel and batched GEMM are deferred.

pub mod gemv;
pub mod small_k;
