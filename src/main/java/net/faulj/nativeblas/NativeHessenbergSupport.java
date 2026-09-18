package net.faulj.nativeblas;

/**
 * Native Hessenberg routing support.
 */
public final class NativeHessenbergSupport {
    private static final int DEFAULT_NATIVE_HESSENBERG_MAX_SIZE = 128;

    private NativeHessenbergSupport() {
    }

    public static boolean tryReduce(double[] h, int n) {
        if (!BackendRegistry.shouldUseCppForAlgorithm("hessenberg", "reduce", n, n, defaultThreadCount())
            || !NativeValidationGuards.allowSquare("hessenberg", n, DEFAULT_NATIVE_HESSENBERG_MAX_SIZE)) {
            return false;
        }
        try {
            NativeBindings.nativeHessenbergReduce(h, n);
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean tryDecompose(double[] h, int n, double[] q) {
        if (!BackendRegistry.shouldUseCppForAlgorithm("hessenberg", "decompose", n, n, defaultThreadCount())
            || !NativeValidationGuards.allowSquare("hessenberg", n, DEFAULT_NATIVE_HESSENBERG_MAX_SIZE)) {
            return false;
        }
        try {
            NativeBindings.nativeHessenbergDecompose(h, n, q);
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    private static int defaultThreadCount() {
        net.faulj.compute.DispatchPolicy policy = net.faulj.compute.DispatchPolicy.defaultPolicy();
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }
}
