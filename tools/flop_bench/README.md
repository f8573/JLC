# FLOPs Benchmark Suite

This small suite measures attainable FLOPs for CPU (float/double), with options for OpenMP parallelization and vectorization flags, and for CUDA using cuBLAS.

Prerequisites
- `g++` (or `clang++`) with OpenMP support
- `nvcc` and CUDA Toolkit with cuBLAS for GPU tests (optional)
- Python 3 for the runner script

Quick usage

From the workspace root:

```bash
python tools/flop_bench/run_bench.py --help
```

The script will compile CPU binaries with/without vectorization and run tests across precisions and thread counts. If `nvcc` is present, it will compile and run the CUDA/cuBLAS benchmark.

Output
- Prints a JSON summary with measured GFLOPS per test case and a small human-readable table.
