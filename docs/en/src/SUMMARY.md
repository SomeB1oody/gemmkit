# Summary

[Introduction](Introduction.md)

# gemmkit User Guide

- [Getting Started](gemmkit-guide/Getting_Started.md)
- [Matrix Views and Layouts](gemmkit-guide/Matrix_Views_and_Layouts.md)
- [Element Types](gemmkit-guide/Element_Types.md)
- [Parallelism in Practice](gemmkit-guide/Parallelism_in_Practice.md)
- [Prepacked Operands](gemmkit-guide/Prepacked_Operands.md)
- [Fused Epilogues](gemmkit-guide/Fused_Epilogues.md)
- [Batched GEMM](gemmkit-guide/Batched_GEMM.md)
- [Small Shapes and GEMV](gemmkit-guide/Small_Shapes_and_GEMV.md)
- [Runtime ISA Dispatch](gemmkit-guide/Runtime_ISA_Dispatch.md)
- [Tuning Knobs](gemmkit-guide/Tuning_Knobs.md)
- [no_std and WebAssembly](gemmkit-guide/no_std_and_WebAssembly.md)
- [The Unchecked Tier](gemmkit-guide/The_Unchecked_Tier.md)

# gemmkit-ndarray

- [Using gemmkit with ndarray](gemmkit-ndarray/Using_gemmkit_with_ndarray.md)
- [ndarray Adapter Advanced Usage](gemmkit-ndarray/ndarray_Adapter_Advanced_Usage.md)

# gemmkit-nalgebra

- [Using gemmkit with nalgebra](gemmkit-nalgebra/Using_gemmkit_with_nalgebra.md)
- [nalgebra Adapter Advanced Usage](gemmkit-nalgebra/nalgebra_Adapter_Advanced_Usage.md)

# gemmkit-faer

- [Using gemmkit with faer](gemmkit-faer/Using_gemmkit_with_faer.md)
- [faer Adapter Advanced Usage](gemmkit-faer/faer_Adapter_Advanced_Usage.md)

# gemmkit-tune

- [Tuning with gemmkit-tune](gemmkit-tune/Tuning_with_gemmkit-tune.md)
- [Inside the Sweep](gemmkit-tune/Inside_the_Sweep.md)

# gemmkit Architecture

- [Design Goals and the Big Picture](architecture/Design_Goals_and_the_Big_Picture.md)
- [The Layer Stack](architecture/The_Layer_Stack.md)
- [Life of a GEMM Call](architecture/Life_of_a_GEMM_Call.md)
- [SIMD Tokens and ISA Dispatch](architecture/SIMD_Tokens_and_ISA_Dispatch.md)
- [Scalars and Kernel Families](architecture/Scalars_and_Kernel_Families.md)
- [Dot Kernels and the Deep-K Twin](architecture/Dot_Kernels_and_the_Deep-K_Twin.md)
- [The Complex Split Kernel](architecture/The_Complex_Split_Kernel.md)
- [Blocking and the Cache Model](architecture/Blocking_and_the_Cache_Model.md)
- [Packing and Workspaces](architecture/Packing_and_Workspaces.md)
- [Parallel Execution](architecture/Parallel_Execution.md)
- [Special Paths](architecture/Special_Paths.md)
- [Epilogue Fusion](architecture/Epilogue_Fusion.md)
- [Extension Points](architecture/Extension_Points.md)
- [Testing and Verification](architecture/Testing_and_Verification.md)
