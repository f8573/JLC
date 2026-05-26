package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.decomposition.result.BidiagonalizationResult;
import net.faulj.matrix.Matrix;

public final class NativeBidiagonalSupport {
    private static final int DEFAULT_NATIVE_BIDIAGONAL_MAX_SIZE = 128;

    private NativeBidiagonalSupport() {
    }

    public static BidiagonalizationResult tryDecompose(Matrix original, Matrix working) {
        if (original == null || working == null) {
            return null;
        }
        int rows = working.getRowCount();
        int cols = working.getColumnCount();
        if (!shouldUseCpp("bidiagonal", "decompose", rows, cols)
            || !NativeValidationGuards.allowRectangular("bidiagonal", rows, cols, DEFAULT_NATIVE_BIDIAGONAL_MAX_SIZE)) {
            return null;
        }

        double[] u = new double[rows * rows];
        double[] b = new double[rows * cols];
        double[] v = new double[cols * cols];
        try {
            NativeBindings.nativeBidiagonalDecompose(working.getRawData(), rows, cols, u, b, v);
            return new BidiagonalizationResult(
                original,
                Matrix.wrap(u, rows, rows),
                Matrix.wrap(b, rows, cols),
                Matrix.wrap(v, cols, cols)
            );
        } catch (RuntimeException | UnsatisfiedLinkError ignored) {
            return null;
        }
    }

    private static boolean shouldUseCpp(String algorithm, String mode, int rows, int cols) {
        return BackendRegistry.shouldUseCppForAlgorithm(algorithm, mode, rows, cols, defaultThreadCount());
    }

    private static int defaultThreadCount() {
        DispatchPolicy policy = DispatchPolicy.defaultPolicy();
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }
}
