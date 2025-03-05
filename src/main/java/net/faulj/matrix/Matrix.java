// File: src/main/java/net/faulj/matrix/Matrix.java
package net.faulj.matrix;

import jcuda.*;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import static jcuda.driver.JCudaDriver.*;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.util.Arrays;

public class Matrix {
    private int rows;
    private int cols;
    // For real matrices: data.length == rows*cols,
    // For complex matrices: data.length == 2*rows*cols.
    private float[] data;
    private CUdeviceptr dData;
    private boolean isComplex;

    // Kernel source code for real matrix multiplication.
    private static final String matrixMulKernelSource =
            "extern \"C\"\n" +
                    "__global__ void matrixMultiply(const float *A, const float *B, float *C, int A_rows, int A_cols, int B_cols) {\n" +
                    "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n" +
                    "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (row < A_rows && col < B_cols) {\n" +
                    "        float sum = 0;\n" +
                    "        for (int k = 0; k < A_cols; k++) {\n" +
                    "            sum += A[row * A_cols + k] * B[k * B_cols + col];\n" +
                    "        }\n" +
                    "        C[row * B_cols + col] = sum;\n" +
                    "    }\n" +
                    "}\n";

    private static CUmodule matrixModule;
    private static CUfunction matrixMulFunction;
    private static boolean matrixInitialized = false;
    private static CUcontext cudaContext;

    // Static block to initialize JCuda and compile the kernel
    static {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cudaContext = new CUcontext();
        cuCtxCreate(cudaContext, 0, device);
        compileMatrixMulKernel();
    }

    // Compile the matrix multiplication kernel
    private static void compileMatrixMulKernel() {
        if (matrixInitialized) return;
        nvrtcProgram program = new nvrtcProgram();
        JNvrtc.nvrtcCreateProgram(program, matrixMulKernelSource, null, 0, null, null);
        JNvrtc.nvrtcCompileProgram(program, 0, null);
        String[] ptx = new String[1];
        JNvrtc.nvrtcGetPTX(program, ptx);
        matrixModule = new CUmodule();
        cuModuleLoadDataEx(matrixModule, ptx[0], 0, new int[0], Pointer.to(new int[0]));
        matrixMulFunction = new CUfunction();
        cuModuleGetFunction(matrixMulFunction, matrixModule, "matrixMultiply");
        matrixInitialized = true;
    }

    // Constructors
    public Matrix(int rows, int cols) {
        this(rows, cols, false);
    }

    public Matrix(int rows, int cols, boolean isComplex) {
        this.rows = rows;
        this.cols = cols;
        this.isComplex = isComplex;
        int length = isComplex ? 2 * rows * cols : rows * cols;
        this.data = new float[length];
        if (!isComplex) {
            dData = new CUdeviceptr();
            cuMemAlloc(dData, length * Sizeof.FLOAT);
        }
    }

    // Factory method from 2D array (assumes real matrix)
    public static Matrix of(float[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        Matrix m = new Matrix(rows, cols, false);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i * cols + j] = array[i][j];
            }
        }
        cuMemcpyHtoD(m.dData, Pointer.to(m.data), m.data.length * Sizeof.FLOAT);
        return m;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return cols;
    }

    public boolean isComplex() {
        return isComplex;
    }

    public float get(int i, int j) {
        return data[i * cols + j];
    }

    public void set(int i, int j, float value) {
        data[i * cols + j] = value;
        if (!isComplex) {
            cuMemcpyHtoD(dData, Pointer.to(data), data.length * Sizeof.FLOAT);
        }
    }

    public float[] getData() {
        return data;
    }

    // Element-wise addition
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols || this.isComplex != other.isComplex) {
            throw new IllegalArgumentException("Dimension or type mismatch");
        }
        Matrix result = new Matrix(rows, cols, isComplex);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        if (!isComplex) {
            cuMemcpyHtoD(result.dData, Pointer.to(result.data), result.data.length * Sizeof.FLOAT);
        }
        return result;
    }

    // Element-wise subtraction
    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols || this.isComplex != other.isComplex) {
            throw new IllegalArgumentException("Dimension or type mismatch");
        }
        Matrix result = new Matrix(rows, cols, isComplex);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        if (!isComplex) {
            cuMemcpyHtoD(result.dData, Pointer.to(result.data), result.data.length * Sizeof.FLOAT);
        }
        return result;
    }

    // Scalar multiplication
    public Matrix scalarMultiply(float scalar) {
        Matrix result = new Matrix(rows, cols, isComplex);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        if (!isComplex) {
            cuMemcpyHtoD(result.dData, Pointer.to(result.data), result.data.length * Sizeof.FLOAT);
        }
        return result;
    }

    // Matrix multiplication: kernel-accelerated for real; CPU-based for complex.
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows || this.isComplex != other.isComplex) {
            throw new IllegalArgumentException("Dimension or type mismatch");
        }
        Matrix result = new Matrix(this.rows, other.cols, this.isComplex);
        if (!isComplex) {
            cuMemcpyHtoD(this.dData, Pointer.to(this.data), this.data.length * Sizeof.FLOAT);
            cuMemcpyHtoD(other.dData, Pointer.to(other.data), other.data.length * Sizeof.FLOAT);
            int blockSize = 16;
            int gridX = (other.cols + blockSize - 1) / blockSize;
            int gridY = (this.rows + blockSize - 1) / blockSize;
            Pointer kernelParams = Pointer.to(
                    Pointer.to(this.dData),
                    Pointer.to(other.dData),
                    Pointer.to(result.dData),
                    Pointer.to(new int[]{this.rows}),
                    Pointer.to(new int[]{this.cols}),
                    Pointer.to(new int[]{other.cols})
            );
            cuLaunchKernel(matrixMulFunction,
                    gridX, gridY, 1,
                    blockSize, blockSize, 1,
                    0, null, kernelParams, null);
            cuCtxSynchronize();
            cuMemcpyDtoH(Pointer.to(result.data), result.dData, result.data.length * Sizeof.FLOAT);
            return result;
        }
        return multiplyCPU(other);
    }

    // CPU-based multiplication (used for complex matrices)
    public Matrix multiplyCPU(Matrix other) {
        if (this.cols != other.rows || this.isComplex != other.isComplex) {
            throw new IllegalArgumentException("Dimension or type mismatch");
        }
        Matrix result = new Matrix(this.rows, other.cols, this.isComplex);
        if (!isComplex) {
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    float sum = 0;
                    for (int k = 0; k < this.cols; k++) {
                        sum += this.get(i, k) * other.get(k, j);
                    }
                    result.data[i * other.cols + j] = sum;
                }
            }
        } else {
            // For complex matrices, data layout: [real, imag, real, imag, ...]
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < other.cols; j++) {
                    float real = 0, imag = 0;
                    for (int k = 0; k < this.cols; k++) {
                        int indexA = 2 * (i * this.cols + k);
                        int indexB = 2 * (k * other.cols + j);
                        float a = this.data[indexA];
                        float b = this.data[indexA + 1];
                        float c = other.data[indexB];
                        float d = other.data[indexB + 1];
                        real += a * c - b * d;
                        imag += a * d + b * c;
                    }
                    int indexRes = 2 * (i * other.cols + j);
                    result.data[indexRes] = real;
                    result.data[indexRes + 1] = imag;
                }
            }
        }
        return result;
    }

    // Transpose (non-conjugate for complex matrices)
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows, isComplex);
        if (!isComplex) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[j * rows + i] = this.get(i, j);
                }
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int srcIdx = 2 * (i * cols + j);
                    int dstIdx = 2 * (j * rows + i);
                    result.data[dstIdx] = this.data[srcIdx];
                    result.data[dstIdx + 1] = this.data[srcIdx + 1];
                }
            }
        }
        return result;
    }

    // Inverse via Gauss-Jordan elimination (only for small, real matrices)
    public Matrix inverse() {
        if (isComplex) {
            throw new UnsupportedOperationException("Inverse not implemented for complex matrices.");
        }
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square.");
        }
        int n = rows;
        Matrix result = identity(n, false);
        Matrix copy = this.copy();
        for (int i = 0; i < n; i++) {
            float pivot = copy.get(i, i);
            if (Math.abs(pivot) < 1e-8f) {
                throw new ArithmeticException("Matrix is singular.");
            }
            for (int j = 0; j < n; j++) {
                copy.data[i * n + j] /= pivot;
                result.data[i * n + j] /= pivot;
            }
            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                float factor = copy.get(k, i);
                for (int j = 0; j < n; j++) {
                    copy.data[k * n + j] -= factor * copy.get(i, j);
                    result.data[k * n + j] -= factor * result.get(i, j);
                }
            }
        }
        return result;
    }

    // Determinant (only for small, real matrices)
    public float determinant() {
        if (isComplex) {
            throw new UnsupportedOperationException("Determinant not implemented for complex matrices.");
        }
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square.");
        }
        return determinantRecursive(this.data, rows);
    }

    private float determinantRecursive(float[] m, int n) {
        if (n == 1) return m[0];
        float det = 0;
        for (int c = 0; c < n; c++) {
            float[] sub = new float[(n - 1) * (n - 1)];
            for (int i = 1; i < n; i++) {
                int subCol = 0;
                for (int j = 0; j < n; j++) {
                    if (j == c) continue;
                    sub[(i - 1) * (n - 1) + subCol] = m[i * n + j];
                    subCol++;
                }
            }
            float subDet = determinantRecursive(sub, n - 1);
            det += ((c % 2 == 0) ? 1 : -1) * m[c] * subDet;
        }
        return det;
    }

    // Adjugate (only for small, real matrices)
    public Matrix adjugate() {
        if (isComplex) {
            throw new UnsupportedOperationException("Adjugate not implemented for complex matrices.");
        }
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square.");
        }
        int n = rows;
        Matrix adj = new Matrix(n, n, false);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float[] sub = new float[(n - 1) * (n - 1)];
                int idx = 0;
                for (int r = 0; r < n; r++) {
                    if (r == i) continue;
                    for (int c = 0; c < n; c++) {
                        if (c == j) continue;
                        sub[idx++] = this.get(r, c);
                    }
                }
                float subDet = determinantRecursive(sub, n - 1);
                adj.data[j * n + i] = (((i + j) % 2 == 0) ? 1 : -1) * subDet;
            }
        }
        return adj;
    }

    // Copy of this matrix.
    public Matrix copy() {
        Matrix m = new Matrix(rows, cols, isComplex);
        System.arraycopy(this.data, 0, m.data, 0, data.length);
        if (!isComplex) {
            cuMemcpyHtoD(m.dData, Pointer.to(m.data), m.data.length * Sizeof.FLOAT);
        }
        return m;
    }

    // Identity matrix factory.
    public static Matrix identity(int size, boolean isComplex) {
        Matrix m = new Matrix(size, size, isComplex);
        if (!isComplex) {
            for (int i = 0; i < size; i++) {
                m.data[i * size + i] = 1.0f;
            }
        } else {
            for (int i = 0; i < size; i++) {
                m.data[2 * (i * size + i)] = 1.0f;
                m.data[2 * (i * size + i) + 1] = 0.0f;
            }
        }
        if (!isComplex) {
            cuMemcpyHtoD(m.dData, Pointer.to(m.data), m.data.length * Sizeof.FLOAT);
        }
        return m;
    }

    // LU Decomposition for real matrices using Doolittle's method.
    public LUResult lu() {
        if (isComplex) {
            throw new UnsupportedOperationException("LU decomposition not implemented for complex matrices.");
        }
        int n = rows;
        Matrix L = identity(n, false);
        Matrix U = this.copy();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float factor = U.get(j, i) / U.get(i, i);
                L.data[j * n + i] = factor;
                for (int k = i; k < n; k++) {
                    U.data[j * n + k] -= factor * U.get(i, k);
                }
            }
        }
        return new LUResult(L, U);
    }

    // QR Decomposition for real matrices using Gram-Schmidt.
    public QRResult qr() {
        if (isComplex) {
            throw new UnsupportedOperationException("QR decomposition not implemented for complex matrices.");
        }
        int m = rows;
        int n = cols;
        Matrix Q = new Matrix(m, n, false);
        Matrix R = new Matrix(n, n, false);
        Matrix[] a = new Matrix[n];
        for (int j = 0; j < n; j++) {
            float[] col = new float[m];
            for (int i = 0; i < m; i++) {
                col[i] = this.get(i, j);
            }
            a[j] = new Matrix(m, 1, false);
            for (int i = 0; i < m; i++) {
                a[j].data[i] = col[i];
            }
            for (int i = 0; i < j; i++) {
                float dot = 0;
                for (int k = 0; k < m; k++) {
                    dot += a[i].data[k] * this.get(k, j);
                }
                R.data[i * n + j] = dot;
                for (int k = 0; k < m; k++) {
                    a[j].data[k] -= dot * Q.data[i * m + k];
                }
            }
            float norm = 0;
            for (int k = 0; k < m; k++) {
                norm += a[j].data[k] * a[j].data[k];
            }
            norm = (float) Math.sqrt(norm);
            R.data[j * n + j] = norm;
            for (int k = 0; k < m; k++) {
                Q.data[j * m + k] = a[j].data[k] / norm;
            }
        }
        // Transpose Q to return an m x n matrix.
        Matrix Qfinal = new Matrix(m, n, false);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Qfinal.data[i * n + j] = Q.data[j * m + i];
            }
        }
        return new QRResult(Qfinal, R);
    }

    // Eigen decomposition for 2x2 real matrices.
    public EigenResult eigen() {
        throw new UnsupportedOperationException("Eigen decomposition is under construction.");
//        if (isComplex) {
//            throw new UnsupportedOperationException("Eigen decomposition not implemented for complex matrices.");
//        }
//        if (rows != 2 || cols != 2) {
//            throw new UnsupportedOperationException("Eigen decomposition implemented only for 2x2 matrices.");
//        }
//
//        float a = this.get(0, 0);
//        float b = this.get(0, 1);
//        float c = this.get(1, 0);
//        float d = this.get(1, 1);
//        float trace = a + d;
//        float det = a * d - b * c;
//        float disc = (float) Math.sqrt(trace * trace - 4 * det);
//        float lambda1 = (trace + disc) / 2f;
//        float lambda2 = (trace - disc) / 2f;
//        Matrix eigenValues = new Matrix(2, 2, false);
//        eigenValues.data[0] = lambda1;
//        eigenValues.data[3] = lambda2;
//        // Compute eigenvectors (simple approach)
//        Matrix eigenVectors = new Matrix(2, 2, false);
//        if (b != 0) {
//            eigenVectors.data[0] = lambda1 - d;
//            eigenVectors.data[1] = b;
//            eigenVectors.data[2] = lambda2 - d;
//            eigenVectors.data[3] = b;
//        } else if (c != 0) {
//            eigenVectors.data[0] = c;
//            eigenVectors.data[1] = lambda1 - a;
//            eigenVectors.data[2] = c;
//            eigenVectors.data[3] = lambda2 - a;
//        } else {
//            eigenVectors = identity(2, false);
//        }
//        return new EigenResult(eigenVectors, eigenValues);
    }

    // SVD for real matrices – placeholder which returns U as identity, S as a copy of A, and Vt as identity.
    public SVDResult svd() {
        if (isComplex) {
            throw new UnsupportedOperationException("SVD not implemented for complex matrices.");
        }
        Matrix U = identity(rows, false);
        Matrix S = this.copy();
        Matrix Vt = identity(cols, false);
        return new SVDResult(U, S, Vt);
    }

    // Inner classes for decomposition results
    public static class LUResult {
        private Matrix L, U;

        public LUResult(Matrix L, Matrix U) {
            this.L = L;
            this.U = U;
        }

        public Matrix getL() {
            return L;
        }

        public Matrix getU() {
            return U;
        }
    }

    public static class QRResult {
        private Matrix Q, R;

        public QRResult(Matrix Q, Matrix R) {
            this.Q = Q;
            this.R = R;
        }

        public Matrix getQ() {
            return Q;
        }

        public Matrix getR() {
            return R;
        }
    }

    public static class EigenResult {
        private Matrix eigenVectors, eigenValues;

        public EigenResult(Matrix eigenVectors, Matrix eigenValues) {
            this.eigenVectors = eigenVectors;
            this.eigenValues = eigenValues;
        }

        public Matrix getEigenVectors() {
            return eigenVectors;
        }

        public Matrix getEigenValues() {
            return eigenValues;
        }
    }

    public static class SVDResult {
        private Matrix U, S, Vt;

        public SVDResult(Matrix U, Matrix S, Matrix Vt) {
            this.U = U;
            this.S = S;
            this.Vt = Vt;
        }

        public Matrix getU() {
            return U;
        }

        public Matrix getS() {
            return S;
        }

        public Matrix getVt() {
            return Vt;
        }
    }
}