package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;

public final class NativeFactorizationSupport {
    private static final int DEFAULT_NATIVE_LU_MAX_SIZE = 128;
    private static final int DEFAULT_NATIVE_CHOLESKY_MAX_SIZE = 128;

    private NativeFactorizationSupport() {
    }

    public static boolean tryLu(double[] packedLu, int n, int[] pivots) {
        if (!shouldUseCpp("lu", "factor", n, n)
            || !NativeValidationGuards.allowSquare("lu", n, DEFAULT_NATIVE_LU_MAX_SIZE)) {
            return false;
        }
        try {
            NativeBindings.nativeLuFactor(packedLu, n, pivots);
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean tryQr(double[] aWork, int m, int n, boolean thin, double[] q, double[] r) {
        String mode = thin ? "decompose_thin" : "decompose_full";
        if (!shouldUseCpp("qr", mode, m, n)) {
            return false;
        }
        int qCols = thin ? Math.min(m, n) : m;
        try {
            NativeBindings.nativeQrDecompose(aWork, m, n, qCols, q, r);
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean tryQrFactorizeOnly(double[] aWork, int m, int n) {
        if (!shouldUseCpp("qr", "factorize_only", m, n)) {
            return false;
        }
        try {
            NativeBindings.nativeQrFactorizeOnly(aWork, m, n);
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean tryCholesky(double[] packedL, int n) {
        if (!shouldUseCpp("cholesky", "decompose", n, n)
            || !NativeValidationGuards.allowSquare("cholesky", n, DEFAULT_NATIVE_CHOLESKY_MAX_SIZE)) {
            return false;
        }
        try {
            int info = NativeBindings.nativeCholeskyDecompose(packedL, n);
            if (info > 0) {
                throw new ArithmeticException("Matrix is not positive definite (non-positive pivot at " + (info - 1) + ")");
            }
            return true;
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    static String qrModeForTests(int rows, int cols, boolean factorizeOnly) {
        String mode = factorizeOnly ? "factorize_only" : "decompose_full";
        return shouldUseCpp("qr", mode, rows, cols) ? "CPP" : "JAVA";
    }

    static String qrShapeFamilyForTests(int rows, int cols) {
        return AlgorithmDispatch.shapeFamilyForTests(rows, cols).name();
    }

    static String qrSizeBandForTests(int rows, int cols) {
        return AlgorithmDispatch.sizeBandForTests(rows, cols).name();
    }

    static void resetCalibrationForTests() {
        AlgorithmDispatch.resetForTests();
    }

    private static boolean shouldUseCpp(String algorithm, String mode, int rows, int cols) {
        return BackendRegistry.shouldUseCppForAlgorithm(algorithm, mode, rows, cols, defaultThreadCount());
    }

    private static int defaultThreadCount() {
        DispatchPolicy policy = DispatchPolicy.defaultPolicy();
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }
}
