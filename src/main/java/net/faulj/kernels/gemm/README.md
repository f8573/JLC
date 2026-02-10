# GEMM Kernel Nucleus

This package is the canonical entry point for dense matrix-matrix multiply.

- Primary API: `net.faulj.kernels.gemm.Gemm`
- Dispatch heuristics: `net.faulj.kernels.gemm.dispatch`
- Microkernel path: `net.faulj.kernels.gemm.microkernel`
- Packing/workspace: `net.faulj.kernels.gemm.packing`
- SIMD strided variants: `net.faulj.kernels.gemm.simd`
- Naive/blocked fallbacks: `net.faulj.kernels.gemm.naive`, `net.faulj.kernels.gemm.blocked`

Design intent:

- Keep historical compute implementations intact.
- Make GEMM immediately discoverable and reviewable as a first-class kernel.
- Route high-level blocked algorithms through this package instead of calling legacy compute classes directly.
