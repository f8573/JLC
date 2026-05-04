package net.faulj.nativeblas;

/**
 * Placeholder for a future self-owned native Hessenberg implementation.
 *
 * <p>Public LAPACK/provider selection has been removed from algorithm dispatch.
 * Hessenberg remains on the Java path until a calibrated C++ implementation is
 * wired through the shared algorithm dispatcher.</p>
 */
public final class NativeHessenbergSupport {
    private NativeHessenbergSupport() {
    }

    public static boolean tryReduce(double[] h, int n) {
        return false;
    }

    public static boolean tryDecompose(double[] h, int n, double[] q) {
        return false;
    }
}
