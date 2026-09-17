package net.faulj.nativeblas;

/** Optional JNI bridge for one whole generated R4 region invocation. */
public final class NativeGeneratedKernelSupport {
    private static final NativeBackend BACKEND = new NativeBackend(new JavaBackend());

    private NativeGeneratedKernelSupport() {
    }

    public static boolean isAvailable() {
        return BACKEND.probe(true).isAvailable();
    }

    public static boolean avx2Supported() {
        return isAvailable() && NativeBindings.nativeGeneratedAvx2Supported();
    }

    /**
     * Execute a registered generated function once for the complete region.
     * A false result means the caller must use its trusted lower-level fallback.
     */
    public static boolean execute(String signature,
                                  double[][] inputs,
                                  double[] output,
                                  int rows,
                                  int cols) {
        if (signature == null || inputs == null || output == null || rows < 0 || cols < 0) {
            return false;
        }
        if (!isAvailable()) {
            return false;
        }
        return NativeBindings.nativeGeneratedKernelExecute(signature, inputs, output, rows, cols);
    }
}
