package net.faulj.svd;

import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.AlgorithmBackend;
import net.faulj.nativeblas.NativeAlgorithmScope;

import java.util.Map;

/**
 * Computes the Singular Value Decomposition (SVD) of a matrix.
 * <p>
 * The default implementation uses a divide-and-conquer solver on the bidiagonal
 * reduction. You can select Golub-Kahan QR iteration via {@link SVDAlgorithm}.
 * </p>
 */
public class SVDecomposition {
    private final SVDAlgorithm algorithm;

    /**
     * Create an SVD decomposer using the default algorithm.
     */
    public SVDecomposition() {
        // Use Golub-Kahan QR by default for improved robustness on small
        // and rank-deficient matrices (avoids issues in the divide-and-conquer
        // path uncovered by tests).
        this(SVDAlgorithm.GOLUB_KAHAN_QR);
    }

    /**
     * Create an SVD decomposer with a specific algorithm.
     *
     * @param algorithm algorithm selection
     */
    public SVDecomposition(SVDAlgorithm algorithm) {
        if (algorithm == null) {
            throw new IllegalArgumentException("Algorithm must not be null");
        }
        this.algorithm = algorithm;
    }

    /**
     * Compute the SVD of a matrix.
     *
     * @param A matrix to decompose
     * @return SVD result
     */
    public SVDResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        Map<String, AlgorithmBackend> stageBackends = nativeStageBackends(A, "full");
        SVDResult res = NativeAlgorithmScope.withOverrides(stageBackends, () -> switch (algorithm) {
            case GOLUB_KAHAN_QR -> new GolubKahanSVD().decompose(A);
            case DIVIDE_AND_CONQUER -> new DivideAndConquerSVD().decompose(A);
        });

        return res;
    }

    static Map<String, AlgorithmBackend> nativeStageBackends(Matrix A, String mode) {
        AlgorithmBackend backend = nativeAlgorithmBackend(A, mode);
        return Map.of(
            "bidiagonal", backend,
            "schur", backend
        );
    }

    static AlgorithmBackend nativeAlgorithmBackend(Matrix A, String mode) {
        int rows = A.getRowCount();
        int cols = A.getColumnCount();
        int threads = defaultThreadCount();
        boolean useCpp = net.faulj.nativeblas.BackendRegistry.shouldUseCppForAlgorithm("svd", mode, rows, cols, threads);
        return useCpp ? AlgorithmBackend.CPP : AlgorithmBackend.JAVA;
    }

    private static int defaultThreadCount() {
        net.faulj.compute.DispatchPolicy policy = net.faulj.compute.DispatchPolicy.defaultPolicy();
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }
}
