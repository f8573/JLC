import net.faulj.matrix.Matrix;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;

public class Decomp512Benchmark {

    private static long measureMillis(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000;
    }

    private static Matrix copy(Matrix m) {
        return new Matrix(m.getRowCount(), m.getColumnCount(), m.getRawData().clone());
    }

    public static void main(String[] args) {
        final int n = 512;
        System.out.println("Benchmark: single random " + n + "x" + n + " matrix decompositions\n");

        System.out.println("Generating matrix...");
        Matrix A = Matrix.randomMatrix(n, n);

        // Build a symmetric positive definite matrix for Cholesky
        Matrix P = A.multiply(A.transpose());
        // Stabilize diagonal
        for (int i = 0; i < n; i++) P.addTo(i, i, n);

        // Warmup (one quick call each) to mitigate JIT effects
        try { HouseholderQR.decompose(copy(A)); } catch (Exception ignored) {}
        try { HessenbergReduction.decompose(copy(A)); } catch (Exception ignored) {}
        try { new Bidiagonalization().decompose(copy(A)); } catch (Exception ignored) {}
        try { new LUDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new SVDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new PolarDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new CholeskyDecomposition().decompose(copy(P)); } catch (Exception ignored) {}
        try { new RealSchurDecomposition().decompose(copy(A)); } catch (Exception ignored) {}

        System.out.println("\nTiming (milliseconds):");

        long tQR = measureMillis(() -> HouseholderQR.decompose(copy(A)));
        System.out.printf("Householder QR: %d ms\n", tQR);

        long tHess = measureMillis(() -> HessenbergReduction.decompose(copy(A)));
        System.out.printf("Hessenberg: %d ms\n", tHess);

        long tBidiag = measureMillis(() -> new Bidiagonalization().decompose(copy(A)));
        System.out.printf("Bidiagonal: %d ms\n", tBidiag);

        long tLU = measureMillis(() -> new LUDecomposition().decompose(copy(A)));
        System.out.printf("LU: %d ms\n", tLU);

        long tPolar = measureMillis(() -> new PolarDecomposition().decompose(copy(A)));
        System.out.printf("Polar: %d ms\n", tPolar);

        long tChol = measureMillis(() -> new CholeskyDecomposition().decompose(copy(P)));
        System.out.printf("Cholesky: %d ms\n", tChol);

        long tSchur = 0;
        try {
            tSchur = measureMillis(() -> new RealSchurDecomposition().decompose(copy(A)));
            System.out.printf("Schur (Implicit QR): %d ms\n", tSchur);
        } catch (Exception e) {
            System.out.printf("Schur (Implicit QR): failed (%s)\n", e.getMessage());
        }

        long tSVD = measureMillis(() -> new SVDecomposition().decompose(copy(A)));
        System.out.printf("SVD: %d ms\n", tSVD);

        System.out.println("\nDone.");
    }
}
