package net.faulj.compute;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import net.faulj.matrix.Matrix;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;

/**
 * CUDA-backed GEMM implementation using JCublas.
 */
final class CudaGemm {
    // Lazy enablement of JCuda/JCublas exceptions is performed inside
    // the gemm method to avoid triggering native library load during
    // class initialization (which causes UnsatisfiedLinkError on
    // systems without CUDA drivers). The gemm call is already guarded
    // by CudaSupport and all native failures are caught and result in
    // a graceful fallback to CPU paths.

    private CudaGemm() {
    }

    /**
     * Multiply two matrices on GPU when available.
     *
     * @param a left matrix
     * @param b right matrix
     * @return product matrix or null if unavailable
     */
    static Matrix multiply(Matrix a, Matrix b) {
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int n = b.getColumnCount();
        Matrix c = Matrix.zero(m, n);
        if (gemm(a, b, c, 1.0, 0.0)) {
            return c;
        }
        return null;
    }

    /**
     * Compute $C = \alpha AB + \beta C$ on GPU when available.
     *
     * @param a left matrix
     * @param b right matrix
     * @param c output matrix
     * @param alpha scaling for AB
     * @param beta scaling for C
     * @return true if computation succeeded
     */
    static boolean gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta) {
        if (!CudaSupport.isCudaAvailable()) {
            return false;
        }
        if (a == null || b == null || c == null) {
            throw new IllegalArgumentException("Matrices must not be null");
        }
        if (a.getRawImagData() != null || b.getRawImagData() != null || c.getRawImagData() != null) {
            return false;
        }
        int m = a.getRowCount();
        int k = a.getColumnCount();
        int k2 = b.getRowCount();
        int n = b.getColumnCount();
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree for multiplication: " + k + " != " + k2);
        }
        if (c.getRowCount() != m || c.getColumnCount() != n) {
            throw new IllegalArgumentException("Output matrix dimensions must be " + m + "x" + n);
        }

        double[] hostA = a.getRawData();
        double[] hostB = b.getRawData();
        double[] hostC = c.getRawData();

        long sizeA = (long) m * k * Sizeof.DOUBLE;
        long sizeB = (long) k * n * Sizeof.DOUBLE;
        long sizeC = (long) m * n * Sizeof.DOUBLE;

        Pointer deviceA = new Pointer();
        Pointer deviceB = new Pointer();
        Pointer deviceC = new Pointer();
        cublasHandle handle = new cublasHandle();

        try {
            // Enable JCuda/JCublas exceptions lazily to avoid classloading
            // native libraries during static initialization. Any failures
            // to initialize native components will be caught and the
            // caller will fall back to CPU implementations.
            try {
                JCuda.setExceptionsEnabled(true);
                JCublas2.setExceptionsEnabled(true);
            } catch (Throwable ignored) {
                // If enabling exceptions triggers a native load failure,
                // propagate to outer catch which returns false.
                throw ignored;
            }

            JCublas2.cublasCreate(handle);
            JCuda.cudaMalloc(deviceA, sizeA);
            JCuda.cudaMalloc(deviceB, sizeB);
            JCuda.cudaMalloc(deviceC, sizeC);

            JCuda.cudaMemcpy(deviceA, Pointer.to(hostA), sizeA, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaMemcpy(deviceB, Pointer.to(hostB), sizeB, cudaMemcpyKind.cudaMemcpyHostToDevice);
            if (beta != 0.0) {
                JCuda.cudaMemcpy(deviceC, Pointer.to(hostC), sizeC, cudaMemcpyKind.cudaMemcpyHostToDevice);
            }

            double[] alphaArr = {alpha};
            double[] betaArr = {beta};

            // Row-major arrays map to column-major transposes, so compute C^T = B^T * A^T.
            JCublas2.cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                Pointer.to(alphaArr),
                deviceB, n,
                deviceA, k,
                Pointer.to(betaArr),
                deviceC, n);

            JCuda.cudaMemcpy(Pointer.to(hostC), deviceC, sizeC, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            return true;
        } catch (Throwable ex) {
            return false;
        } finally {
            try {
                JCuda.cudaFree(deviceA);
            } catch (Throwable ignored) {
            }
            try {
                JCuda.cudaFree(deviceB);
            } catch (Throwable ignored) {
            }
            try {
                JCuda.cudaFree(deviceC);
            } catch (Throwable ignored) {
            }
            try {
                JCublas2.cublasDestroy(handle);
            } catch (Throwable ignored) {
            }
        }
    }
}
