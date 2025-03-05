package net.faulj.vector;

import jcuda.*;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import static jcuda.driver.JCudaDriver.*;

/**
 * A JCuda-accelerated vector class that supports both real and complex numbers.
 * <p>
 * For real vectors, each element is held in one float. For complex vectors,
 * each element is represented in an interleaved fashion with 2 floats: [real, imag].
 * The vector type (real or complex) is set at construction.
 * </p>
 */
public class Vector {
    private int size;
    private boolean isComplex;
    // For real vectors, data.length == size; for complex vectors, data.length == 2*size.
    private float[] data;
    private CUdeviceptr dData;

    // --- JCuda kernels for real numbers ---
    private static final String vectorAddKernelSource =
            "extern \"C\"\n" +
                    "__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        C[i] = A[i] + B[i];\n" +
                    "    }\n" +
                    "}\n";

    private static final String vectorSubtractKernelSource =
            "extern \"C\"\n" +
                    "__global__ void vectorSubtract(const float *A, const float *B, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        C[i] = A[i] - B[i];\n" +
                    "    }\n" +
                    "}\n";

    private static final String scalarMultiplyKernelSource =
            "extern \"C\"\n" +
                    "__global__ void scalarMultiply(const float *A, float scalar, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        C[i] = A[i] * scalar;\n" +
                    "    }\n" +
                    "}\n";

    private static final String dotProductKernelSource =
            "extern \"C\"\n" +
                    "__global__ void dotProduct(const float *A, const float *B, float *partialSums, int N) {\n" +
                    "    extern __shared__ float cache[];\n" +
                    "    int tid = threadIdx.x;\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    float temp = 0.0f;\n" +
                    "    while (i < N) {\n" +
                    "        temp += A[i] * B[i];\n" +
                    "        i += blockDim.x * gridDim.x;\n" +
                    "    }\n" +
                    "    cache[tid] = temp;\n" +
                    "    __syncthreads();\n" +
                    "    int stride = blockDim.x / 2;\n" +
                    "    while (stride > 0) {\n" +
                    "        if (tid < stride) {\n" +
                    "            cache[tid] += cache[tid + stride];\n" +
                    "        }\n" +
                    "        __syncthreads();\n" +
                    "        stride /= 2;\n" +
                    "    }\n" +
                    "    if (tid == 0) {\n" +
                    "        partialSums[blockIdx.x] = cache[0];\n" +
                    "    }\n" +
                    "}\n";

    // --- JCuda kernels for complex numbers ---
    // The operations for complex numbers assume interleaved storage (real, imag).
    private static final String complexAddKernelSource =
            "extern \"C\"\n" +
                    "__global__ void complexAdd(const float *A, const float *B, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        int idx = 2 * i;\n" +
                    "        C[idx] = A[idx] + B[idx];\n" +
                    "        C[idx+1] = A[idx+1] + B[idx+1];\n" +
                    "    }\n" +
                    "}\n";

    private static final String complexSubtractKernelSource =
            "extern \"C\"\n" +
                    "__global__ void complexSubtract(const float *A, const float *B, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        int idx = 2 * i;\n" +
                    "        C[idx] = A[idx] - B[idx];\n" +
                    "        C[idx+1] = A[idx+1] - B[idx+1];\n" +
                    "    }\n" +
                    "}\n";

    private static final String complexScalarMultiplyKernelSource =
            "extern \"C\"\n" +
                    "__global__ void complexScalarMultiply(const float *A, float scalarReal, float scalarImag, float *C, int N) {\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    if (i < N) {\n" +
                    "        int idx = 2 * i;\n" +
                    "        float a = A[idx];\n" +
                    "        float b = A[idx+1];\n" +
                    "        // Perform complex multiplication: (a+bi) * (scalarReal+scalarImag*i)\n" +
                    "        C[idx] = a * scalarReal - b * scalarImag;\n" +
                    "        C[idx+1] = a * scalarImag + b * scalarReal;\n" +
                    "    }\n" +
                    "}\n";

    private static final String complexDotProductKernelSource =
            "extern \"C\"\n" +
                    "__global__ void complexDotProduct(const float *A, const float *B, float *partialSums, int N) {\n" +
                    "    extern __shared__ float cache[]; // cache size is 2*blockDim.x floats\n" +
                    "    int tid = threadIdx.x;\n" +
                    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n" +
                    "    float sumReal = 0.0f;\n" +
                    "    float sumImag = 0.0f;\n" +
                    "    while(i < N) {\n" +
                    "        int idx = 2 * i;\n" +
                    "        float a = A[idx];\n" +
                    "        float b = A[idx+1];\n" +
                    "        float c = B[idx];\n" +
                    "        float d = B[idx+1];\n" +
                    "        // Compute (a+bi)*(c+di) as (ac - bd) + i(ad + bc)\n" +
                    "        sumReal += a * c - b * d;\n" +
                    "        sumImag += a * d + b * c;\n" +
                    "        i += blockDim.x * gridDim.x;\n" +
                    "    }\n" +
                    "    cache[2*tid] = sumReal;\n" +
                    "    cache[2*tid+1] = sumImag;\n" +
                    "    __syncthreads();\n" +
                    "    int stride = blockDim.x / 2;\n" +
                    "    while(stride > 0) {\n" +
                    "        if(tid < stride) {\n" +
                    "            cache[2*tid] += cache[2*(tid+stride)];\n" +
                    "            cache[2*tid+1] += cache[2*(tid+stride)+1];\n" +
                    "        }\n" +
                    "        __syncthreads();\n" +
                    "        stride /= 2;\n" +
                    "    }\n" +
                    "    if(tid == 0) {\n" +
                    "        partialSums[2*blockIdx.x] = cache[0];\n" +
                    "        partialSums[2*blockIdx.x+1] = cache[1];\n" +
                    "    }\n" +
                    "}\n";

    // --- CUDA module and function handles ---
    private static CUmodule module;
    private static CUfunction vectorAddFunction;
    private static CUfunction vectorSubtractFunction;
    private static CUfunction scalarMultiplyFunction;
    private static CUfunction dotProductFunction;

    private static CUfunction complexAddFunction;
    private static CUfunction complexSubtractFunction;
    private static CUfunction complexScalarMultiplyFunction;
    private static CUfunction complexDotProductFunction;

    private static boolean initialized = false;

    /**
     * Initializes the JCuda context and compiles all the CUDA kernels if not already done.
     */
    public static void initializeJCuda() {
        if (initialized) {
            return;
        }
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Combine all kernel sources.
        String combinedKernels = vectorAddKernelSource +
                vectorSubtractKernelSource +
                scalarMultiplyKernelSource +
                dotProductKernelSource +
                complexAddKernelSource +
                complexSubtractKernelSource +
                complexScalarMultiplyKernelSource +
                complexDotProductKernelSource;

        nvrtcProgram prog = new nvrtcProgram();
        JNvrtc.nvrtcCreateProgram(prog, combinedKernels, null, 0, null, null);
        JNvrtc.nvrtcCompileProgram(prog, 0, null);
        String[] ptx = new String[1];
        JNvrtc.nvrtcGetPTX(prog, ptx);

        module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Retrieve handles for real operations.
        vectorAddFunction = new CUfunction();
        cuModuleGetFunction(vectorAddFunction, module, "vectorAdd");

        vectorSubtractFunction = new CUfunction();
        cuModuleGetFunction(vectorSubtractFunction, module, "vectorSubtract");

        scalarMultiplyFunction = new CUfunction();
        cuModuleGetFunction(scalarMultiplyFunction, module, "scalarMultiply");

        dotProductFunction = new CUfunction();
        cuModuleGetFunction(dotProductFunction, module, "dotProduct");

        // Retrieve handles for complex operations.
        complexAddFunction = new CUfunction();
        cuModuleGetFunction(complexAddFunction, module, "complexAdd");

        complexSubtractFunction = new CUfunction();
        cuModuleGetFunction(complexSubtractFunction, module, "complexSubtract");

        complexScalarMultiplyFunction = new CUfunction();
        cuModuleGetFunction(complexScalarMultiplyFunction, module, "complexScalarMultiply");

        complexDotProductFunction = new CUfunction();
        cuModuleGetFunction(complexDotProductFunction, module, "complexDotProduct");

        initialized = true;
    }

    /**
     * Constructs a real vector with the given size.
     *
     * @param size the number of elements in the vector
     */
    public Vector(int size) {
        this(size, false);
    }

    /**
     * Constructs a vector with the given size.
     *
     * @param size      the number of elements (or complex elements) in the vector
     * @param isComplex if true, the vector represents complex numbers (interleaved storage)
     */
    public Vector(int size, boolean isComplex) {
        this.size = size;
        this.isComplex = isComplex;
        this.data = new float[isComplex ? 2 * size : size];
        initializeJCuda();
        dData = new CUdeviceptr();
        int byteSize = (isComplex ? 2 * size : size) * Sizeof.FLOAT;
        cuMemAlloc(dData, byteSize);
    }

    /**
     * Sets the data for the vector.
     *
     * @param data the host data; its length must be size for real vectors or 2*size for complex vectors
     */
    public void setData(float[] data) {
        if (isComplex) {
            if (data.length != 2 * size) {
                throw new IllegalArgumentException("Data length must be twice the vector size for complex vectors.");
            }
        } else {
            if (data.length != size) {
                throw new IllegalArgumentException("Data length must match vector size.");
            }
        }
        this.data = data.clone();
        int byteSize = (isComplex ? 2 * size : size) * Sizeof.FLOAT;
        cuMemcpyHtoD(dData, Pointer.to(this.data), byteSize);
    }

    /**
     * Returns the current vector data from the device.
     *
     * @return the data array (size for real vectors or 2*size for complex vectors)
     */
    public float[] getData() {
        int byteSize = (isComplex ? 2 * size : size) * Sizeof.FLOAT;
        cuMemcpyDtoH(Pointer.to(data), dData, byteSize);
        return data;
    }

    /**
     * Adds this vector to another vector.
     *
     * @param other the vector to add to this vector (must be of the same type and size)
     * @return a new vector representing the sum
     */
    public Vector add(Vector other) {
        if (other.size != this.size || other.isComplex != this.isComplex) {
            throw new IllegalArgumentException("Vectors must be of the same size and type (real or complex).");
        }
        Vector result = new Vector(size, isComplex);
        CUdeviceptr dResult = new CUdeviceptr();
        int byteSize = (isComplex ? 2 * size : size) * Sizeof.FLOAT;
        cuMemAlloc(dResult, byteSize);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(other.dData),
                Pointer.to(dResult),
                Pointer.to(new int[]{size})
        );

        if (isComplex) {
            cuLaunchKernel(complexAddFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        } else {
            cuLaunchKernel(vectorAddFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        }
        cuCtxSynchronize();

        float[] resultData = new float[isComplex ? 2 * size : size];
        cuMemcpyDtoH(Pointer.to(resultData), dResult, byteSize);
        result.setData(resultData);
        cuMemFree(dResult);
        return result;
    }

    /**
     * Subtracts another vector from this vector.
     *
     * @param other the vector to subtract (must be of the same type and size)
     * @return a new vector representing the difference
     */
    public Vector subtract(Vector other) {
        if (other.size != this.size || other.isComplex != this.isComplex) {
            throw new IllegalArgumentException("Vectors must be of the same size and type (real or complex).");
        }
        Vector result = new Vector(size, isComplex);
        CUdeviceptr dResult = new CUdeviceptr();
        int byteSize = (isComplex ? 2 * size : size) * Sizeof.FLOAT;
        cuMemAlloc(dResult, byteSize);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(other.dData),
                Pointer.to(dResult),
                Pointer.to(new int[]{size})
        );

        if (isComplex) {
            cuLaunchKernel(complexSubtractFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        } else {
            cuLaunchKernel(vectorSubtractFunction,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, null,
                    kernelParameters, null
            );
        }
        cuCtxSynchronize();

        float[] resultData = new float[isComplex ? 2 * size : size];
        cuMemcpyDtoH(Pointer.to(resultData), dResult, byteSize);
        result.setData(resultData);
        cuMemFree(dResult);
        return result;
    }

    /**
     * Multiplies this vector by a real scalar.
     * <p>
     * For real vectors, the scalar is applied directly.
     * For complex vectors, the scalar is assumed to be real (imaginary part is 0).
     * </p>
     *
     * @param scalar the scalar value
     * @return a new vector representing the product
     */
    public Vector multiply(float scalar) {
        if (isComplex) {
            return multiply(scalar, 0.0f);
        }
        Vector result = new Vector(size, false);
        CUdeviceptr dResult = new CUdeviceptr();
        int byteSize = size * Sizeof.FLOAT;
        cuMemAlloc(dResult, byteSize);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(new float[]{scalar}),
                Pointer.to(dResult),
                Pointer.to(new int[]{size})
        );
        cuLaunchKernel(scalarMultiplyFunction,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        float[] resultData = new float[size];
        cuMemcpyDtoH(Pointer.to(resultData), dResult, byteSize);
        result.setData(resultData);
        cuMemFree(dResult);
        return result;
    }

    /**
     * Multiplies this complex vector by a complex scalar.
     *
     * @param scalarReal the real part of the scalar
     * @param scalarImag the imaginary part of the scalar
     * @return a new complex vector representing the product
     * @throws IllegalArgumentException if the vector is not complex
     */
    public Vector multiply(float scalarReal, float scalarImag) {
        if (!isComplex) {
            throw new IllegalArgumentException("This vector is not complex.");
        }
        Vector result = new Vector(size, true);
        CUdeviceptr dResult = new CUdeviceptr();
        int byteSize = 2 * size * Sizeof.FLOAT;
        cuMemAlloc(dResult, byteSize);

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(new float[]{scalarReal}),
                Pointer.to(new float[]{scalarImag}),
                Pointer.to(dResult),
                Pointer.to(new int[]{size})
        );
        cuLaunchKernel(complexScalarMultiplyFunction,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        float[] resultData = new float[2 * size];
        cuMemcpyDtoH(Pointer.to(resultData), dResult, byteSize);
        result.setData(resultData);
        cuMemFree(dResult);
        return result;
    }

    /**
     * Computes the dot product of two real vectors.
     *
     * @param other the other vector (must be real and of the same size)
     * @return the dot product as a float
     * @throws IllegalArgumentException if either vector is complex or sizes do not match
     */
    public float dot(Vector other) {
        if (other.size != this.size || other.isComplex != this.isComplex) {
            throw new IllegalArgumentException("Vectors must be of the same size and type (real or complex).");
        }
        if (isComplex) {
            throw new IllegalArgumentException("Use dotComplex for complex vectors.");
        }
        int blockSize = 256;
        int gridSize = Math.min(256, (size + blockSize - 1) / blockSize);

        CUdeviceptr dPartialSums = new CUdeviceptr();
        cuMemAlloc(dPartialSums, gridSize * Sizeof.FLOAT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(other.dData),
                Pointer.to(dPartialSums),
                Pointer.to(new int[]{size})
        );
        cuLaunchKernel(dotProductFunction,
                gridSize, 1, 1,
                blockSize, 1, 1,
                blockSize * Sizeof.FLOAT, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        float[] partialSums = new float[gridSize];
        cuMemcpyDtoH(Pointer.to(partialSums), dPartialSums, gridSize * Sizeof.FLOAT);
        cuMemFree(dPartialSums);

        float sum = 0;
        for (int i = 0; i < gridSize; i++) {
            sum += partialSums[i];
        }
        return sum;
    }

    /**
     * Computes the dot product of two complex vectors.
     *
     * @param other the other complex vector (must be of the same size)
     * @return a two-element float array [real, imag] representing the dot product
     * @throws IllegalArgumentException if either vector is not complex or sizes do not match
     */
    public float[] dotComplex(Vector other) {
        if (!isComplex || !other.isComplex || other.size != this.size) {
            throw new IllegalArgumentException("Both vectors must be complex and of the same size.");
        }
        int blockSize = 256;
        int gridSize = Math.min(256, (size + blockSize - 1) / blockSize);

        // Allocate space for partial sums (2 floats per block)
        CUdeviceptr dPartialSums = new CUdeviceptr();
        cuMemAlloc(dPartialSums, gridSize * 2 * Sizeof.FLOAT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(this.dData),
                Pointer.to(other.dData),
                Pointer.to(dPartialSums),
                Pointer.to(new int[]{size})
        );
        cuLaunchKernel(complexDotProductFunction,
                gridSize, 1, 1,
                blockSize, 1, 1,
                blockSize * 2 * Sizeof.FLOAT, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        float[] partialSums = new float[gridSize * 2];
        cuMemcpyDtoH(Pointer.to(partialSums), dPartialSums, gridSize * 2 * Sizeof.FLOAT);
        cuMemFree(dPartialSums);

        float sumReal = 0;
        float sumImag = 0;
        for (int i = 0; i < gridSize; i++) {
            sumReal += partialSums[2 * i];
            sumImag += partialSums[2 * i + 1];
        }
        return new float[]{sumReal, sumImag};
    }

    /**
     * Frees allocated device memory for this vector.
     */
    public void free() {
        if (dData != null) {
            cuMemFree(dData);
        }
    }
}