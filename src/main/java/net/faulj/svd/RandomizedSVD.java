package net.faulj.svd;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;

import java.util.Random;

/**
 * Computes an approximate SVD using randomized range finding.
 * <p>
 * This is useful for large, approximately low-rank matrices where full SVD is expensive.
 * </p>
 */
public class RandomizedSVD {
    private final int rank;
    private final int oversample;
    private final int powerIterations;
    private final Random rng;

    public RandomizedSVD(int rank) {
        this(rank, 5, 2, new Random());
    }

    public RandomizedSVD(int rank, int oversample, int powerIterations) {
        this(rank, oversample, powerIterations, new Random());
    }

    public RandomizedSVD(int rank, int oversample, int powerIterations, Random rng) {
        if (rank <= 0) {
            throw new IllegalArgumentException("Rank must be positive");
        }
        if (oversample < 0) {
            throw new IllegalArgumentException("Oversample must be non-negative");
        }
        if (powerIterations < 0) {
            throw new IllegalArgumentException("Power iterations must be non-negative");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random generator must not be null");
        }
        this.rank = rank;
        this.oversample = oversample;
        this.powerIterations = powerIterations;
        this.rng = rng;
    }

    /**
     * Computes a randomized low-rank SVD approximation.
     *
     * @param A input matrix
     * @return approximate SVD result with rank <= requested rank
     */
    public SVDResult decompose(Matrix A) {
        if (A == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        if (!A.isReal()) {
            throw new UnsupportedOperationException("Randomized SVD requires a real-valued matrix");
        }
        int m = A.getRowCount();
        int n = A.getColumnCount();
        int r = Math.min(rank, Math.min(m, n));
        int l = Math.min(Math.min(m, n), r + oversample);

        Matrix omega = gaussianMatrix(n, l, rng);
        Matrix Y = A.multiply(omega);
        Matrix Q = orthonormalize(Y);

        for (int i = 0; i < powerIterations; i++) {
            Matrix Z = A.transpose().multiply(Q);
            Matrix Yp = A.multiply(Z);
            Q = orthonormalize(Yp);
        }

        Matrix B = Q.transpose().multiply(A);
        DivideAndConquerSVD smallSvd = new DivideAndConquerSVD();
        SVDResult small = smallSvd.decomposeThin(B);

        Matrix U = Q.multiply(small.getU());
        Matrix V = small.getV();
        double[] sigma = small.getSingularValues();

        U = U.getColumnCount() == r ? U : U.crop(0, m - 1, 0, r - 1);
        V = V.getColumnCount() == r ? V : V.crop(0, n - 1, 0, r - 1);
        sigma = java.util.Arrays.copyOf(sigma, r);

        return new SVDResult(A, U, sigma, V);
    }

    private static Matrix orthonormalize(Matrix A) {
        QRResult qr = HouseholderQR.decomposeThin(A);
        return qr.getQ();
    }

    private static Matrix gaussianMatrix(int rows, int cols, Random rng) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.set(i, j, rng.nextGaussian());
            }
        }
        return m;
    }
}
