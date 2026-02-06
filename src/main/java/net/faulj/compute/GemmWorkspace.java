package net.faulj.compute;

/**
 * Reusable workspace for GEMM operations to eliminate allocations.
 * Thread-safe via ThreadLocal storage.
 */
public final class GemmWorkspace {
    private static final ThreadLocal<GemmWorkspace> INSTANCE = ThreadLocal.withInitial(GemmWorkspace::new);

    // Packing buffers
    private double[] packA;
    private double[] packB;

    // Temporary buffers for transpose/rearrange
    private double[] tempBuffer;

    // CUDA persistent resources (if available)
    private CudaContext cudaContext;

    private GemmWorkspace() {
        // Private constructor
    }

    public static GemmWorkspace get() {
        return INSTANCE.get();
    }

    /**
     * Get or grow pack A buffer to required size.
     */
    public double[] getPackA(int requiredSize) {
        if (packA == null || packA.length < requiredSize) {
            packA = new double[requiredSize];
        }
        return packA;
    }

    /**
     * Get or grow pack B buffer to required size.
     */
    public double[] getPackB(int requiredSize) {
        if (packB == null || packB.length < requiredSize) {
            packB = new double[requiredSize];
        }
        return packB;
    }

    /**
     * Get or grow temporary buffer to required size.
     */
    public double[] getTempBuffer(int requiredSize) {
        if (tempBuffer == null || tempBuffer.length < requiredSize) {
            tempBuffer = new double[requiredSize];
        }
        return tempBuffer;
    }

    /**
     * Get or initialize CUDA context (lazy).
     */
    public CudaContext getCudaContext() {
        if (cudaContext == null) {
            cudaContext = CudaContext.tryCreate();
        }
        return cudaContext;
    }

    /**
     * Release CUDA resources (call on thread shutdown).
     */
    public void releaseCuda() {
        if (cudaContext != null) {
            cudaContext.release();
            cudaContext = null;
        }
    }
}
