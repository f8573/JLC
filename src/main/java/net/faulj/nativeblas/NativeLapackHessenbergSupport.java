package net.faulj.nativeblas;

/**
 * Optional vendor-LAPACK Hessenberg path for the canonical decomposition API.
 */
public final class NativeLapackHessenbergSupport {
    private static final int DEFAULT_MIN_SIZE = 128;

    private NativeLapackHessenbergSupport() {
    }

    public static boolean tryReduce(double[] h, int n) {
        if (!NativeLapackSupport.shouldUseVendorLapack("hessenberg", n, DEFAULT_MIN_SIZE)) {
            return false;
        }
        try {
            NativeBindings.nativeHessenbergReduceVendor(h, n);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean tryDecompose(double[] h, int n, double[] q) {
        if (!NativeLapackSupport.shouldUseVendorLapack("hessenberg", n, DEFAULT_MIN_SIZE)) {
            return false;
        }
        try {
            NativeBindings.nativeHessenbergDecomposeVendor(h, n, q);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

}
