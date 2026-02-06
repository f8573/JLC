package net.faulj.benchmark;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.RealSchurDecomposition;
import org.junit.Test;

public class Decomp512BenchmarkTest {

    private static long measureMillis(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000;
    }

    private static Matrix copy(Matrix m) {
        return Matrix.wrap(m.getRawData().clone(), m.getRowCount(), m.getColumnCount());
    }

    @Test
    public void runDecomp512Benchmark() {
        final int n = 512;
        System.out.println("Benchmark: single random " + n + "x" + n + " matrix decompositions\n");

        System.out.println("Generating matrix...");
        Matrix A = Matrix.randomMatrix(n, n);

        // Build a symmetric positive definite matrix for Cholesky
        Matrix P = A.multiply(A.transpose());
        for (int i = 0; i < n; i++) P.set(i, i, P.get(i, i) + n);

        // Pre-compute Hessenberg form for Schur benchmark
        HessenbergResult hessForSchur = HessenbergReduction.decompose(copy(A));

        // Warmup
        try { HouseholderQR.decompose(copy(A)); } catch (Exception ignored) {}
        try { HessenbergReduction.decompose(copy(A)); } catch (Exception ignored) {}
        try { new Bidiagonalization().decompose(copy(A)); } catch (Exception ignored) {}
        try { new LUDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new SVDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new PolarDecomposition().decompose(copy(A)); } catch (Exception ignored) {}
        try { new CholeskyDecomposition().decompose(copy(P)); } catch (Exception ignored) {}
        try { ImplicitQRFrancis.decomposeFromHessenberg(hessForSchur); } catch (Exception ignored) {}

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

        // Schur from pre-computed Hessenberg (QR iteration only)
        try {
            // Create fresh copies for each benchmark run
            HessenbergResult freshHess = new HessenbergResult(
                hessForSchur.getOriginal(),
                copy(hessForSchur.getH()),
                copy(hessForSchur.getQ())
            );
            long tSchurFromHess = measureMillis(() -> ImplicitQRFrancis.decomposeFromHessenberg(freshHess));
            System.out.printf("Schur (from Hessenberg, with U): %d ms\n", tSchurFromHess);
        } catch (Exception e) {
            System.out.printf("Schur (from Hessenberg, with U): failed (%s)\n", e.getMessage());
        }

        // Schur T only (without U accumulation) - faster for eigenvalues only
        try {
            HessenbergResult freshHess = new HessenbergResult(
                hessForSchur.getOriginal(),
                copy(hessForSchur.getH()),
                copy(hessForSchur.getQ())
            );
            long tSchurTOnly = measureMillis(() -> ImplicitQRFrancis.decomposeSchurFormOnly(freshHess));
            System.out.printf("Schur (from Hessenberg, T only): %d ms\n", tSchurTOnly);
        } catch (Exception e) {
            System.out.printf("Schur (from Hessenberg, T only): failed (%s)\n", e.getMessage());
        }

        // Full Schur for comparison
        try {
            long tSchur = measureMillis(() -> RealSchurDecomposition.schurT(copy(A)));
            System.out.printf("Schur (full, including Hessenberg): %d ms\n", tSchur);
        } catch (Exception e) {
            System.out.printf("Schur (full): failed (%s)\n", e.getMessage());
        }

        long tSVD = measureMillis(() -> new SVDecomposition().decompose(copy(A)));
        System.out.printf("SVD: %d ms\n", tSVD);

        System.out.println("\nDone.");
    }
}
