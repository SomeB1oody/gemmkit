# Summary

[简介](简介.md)

# gemmkit 使用指南

- [快速上手](gemmkit-guide/快速上手.md)
- [矩阵视图与内存布局](gemmkit-guide/矩阵视图与内存布局.md)
- [元素类型](gemmkit-guide/元素类型.md)
- [并行实践](gemmkit-guide/并行实践.md)
- [预打包操作数](gemmkit-guide/预打包操作数.md)
- [融合Epilogue](gemmkit-guide/融合Epilogue.md)
- [批量GEMM](gemmkit-guide/批量GEMM.md)
- [小形状与GEMV](gemmkit-guide/小形状与GEMV.md)
- [运行时ISA分发](gemmkit-guide/运行时ISA分发.md)
- [调优旋钮](gemmkit-guide/调优旋钮.md)
- [no_std与WebAssembly](gemmkit-guide/no_std与WebAssembly.md)
- [Unchecked层](gemmkit-guide/Unchecked层.md)

# gemmkit-ndarray

- [在ndarray中使用gemmkit](gemmkit-ndarray/在ndarray中使用gemmkit.md)
- [ndarray适配器进阶用法](gemmkit-ndarray/ndarray适配器进阶用法.md)

# gemmkit-nalgebra

- [在nalgebra中使用gemmkit](gemmkit-nalgebra/在nalgebra中使用gemmkit.md)
- [nalgebra适配器进阶用法](gemmkit-nalgebra/nalgebra适配器进阶用法.md)

# gemmkit-faer

- [在faer中使用gemmkit](gemmkit-faer/在faer中使用gemmkit.md)
- [faer适配器进阶用法](gemmkit-faer/faer适配器进阶用法.md)

# gemmkit-tune

- [使用gemmkit-tune调优](gemmkit-tune/使用gemmkit-tune调优.md)
- [深入扫描过程](gemmkit-tune/深入扫描过程.md)

# gemmkit 架构说明

- [设计目标与总体图景](architecture/设计目标与总体图景.md)
- [分层结构](architecture/分层结构.md)
- [一次GEMM调用的生命周期](architecture/一次GEMM调用的生命周期.md)
- [SIMD令牌与ISA分发](architecture/SIMD令牌与ISA分发.md)
- [标量与内核家族](architecture/标量与内核家族.md)
- [点积内核与深K孪生](architecture/点积内核与深K孪生.md)
- [复数拆分内核](architecture/复数拆分内核.md)
- [分块与缓存模型](architecture/分块与缓存模型.md)
- [打包与工作区](architecture/打包与工作区.md)
- [并行执行](architecture/并行执行.md)
- [特殊路径](architecture/特殊路径.md)
- [Epilogue融合](architecture/Epilogue融合.md)
- [扩展点](architecture/扩展点.md)
- [测试与验证](architecture/测试与验证.md)
