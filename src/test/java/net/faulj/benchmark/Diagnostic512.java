package net.faulj.benchmark;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;

import java.util.Set;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import net.faulj.vector.Vector;

public class Diagnostic512 {

    private static long measureMillis(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000;
    }

    private static double estimateFlops(String op, int n) {
        double N3 = (double) n * n * n;
        switch (op) {
            case "LU": return 2.0/3.0 * N3;
            case "QR": return 2.0/3.0 * N3;
            case "HESSENBERG": return 10.0/3.0 * N3;
            case "BIDIAG": return 4.0/3.0 * N3;
            case "SVD": return 4.0 * N3;
            case "POLAR": return 4.0 * N3;
            case "CHOLESKY": return 1.0/3.0 * N3;
            case "SCHUR": return 4.0 * N3;
            case "INVERSE": return 2.0 * N3;
            case "DETERMINANT": return 2.0/3.0 * N3;
            case "ROWSPACE": return 2.0/3.0 * N3;
            case "COLSPACE": return 2.0/3.0 * N3;
            case "NULLSPACE": return 2.0/3.0 * N3;
            default: return 0.0;
        }
    }

    private static BufferedWriter csvWriter = null;

    private static void report(String op, int n, long ms, double flops, String note) {
        double flopsPerSec = ms > 0 ? (flops * 1000.0 / ms) : Double.POSITIVE_INFINITY;
        String line = String.format("%s,%d,%d,%.0f,%.2f,%s", op, n, ms, flops, flopsPerSec, note == null ? "" : note);
        System.out.println(line);
        if (csvWriter != null) {
            try {
                csvWriter.write(line);
                csvWriter.newLine();
                csvWriter.flush();
            } catch (IOException ignored) {
            }
        }
    }

    public static void main(String[] args) {
        long init = System.nanoTime();
        final int n = 512;
        final int iterations = 5;
        System.out.println("operation,n,ms,flops,flop/s,info");
        try {
            csvWriter = new BufferedWriter(new FileWriter("diagnostic_512_results.csv"));
            csvWriter.write("operation,n,ms,flops,flop/s,info");
            csvWriter.newLine();
            csvWriter.flush();
        } catch (IOException ignored) {
        }

        System.out.println("Generating random matrix...");
        Matrix A = Matrix.randomMatrix(n, n);

        // Build SPD for Cholesky
        Matrix P = A.multiply(A.transpose());
        for (int i = 0; i < n; i++) P.set(i, i, P.get(i, i) + n);

        // Warmup to reduce JIT noise
        System.out.println("Warming up...");
        try { HouseholderQR.decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("QR done.");
        try { HessenbergReduction.decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("Hessenberg done.");
        try { new Bidiagonalization().decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("Bidiagonal done.");
        try { new LUDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("LU done.");
        try { new SVDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("SVD done.");
        try { new PolarDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("Polar done.");
        try { new CholeskyDecomposition().decompose(P.copy()); } catch (Exception ignored) {}
        System.out.println("Cholesky done.");
        try { new RealSchurDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        System.out.println("Schur done.");

        for (int iter = 1; iter <= iterations; iter++) {
            // QR
            long t = measureMillis(() -> HouseholderQR.decompose(A.copy()));
            report("QR", n, t, estimateFlops("QR", n), "iter=" + iter);

            // Hessenberg
            t = measureMillis(() -> HessenbergReduction.decompose(A.copy()));
            report("HESSENBERG", n, t, estimateFlops("HESSENBERG", n), "iter=" + iter);

            // Bidiagonal
            t = measureMillis(() -> new Bidiagonalization().decompose(A.copy()));
            report("BIDIAG", n, t, estimateFlops("BIDIAG", n), "iter=" + iter);

            // LU
            t = measureMillis(() -> new LUDecomposition().decompose(A.copy()));
            report("LU", n, t, estimateFlops("LU", n), "iter=" + iter);

            // Polar
            t = measureMillis(() -> new PolarDecomposition().decompose(A.copy()));
            report("POLAR", n, t, estimateFlops("POLAR", n), "iter=" + iter);

            // Cholesky
            t = measureMillis(() -> new CholeskyDecomposition().decompose(P.copy()));
            report("CHOLESKY", n, t, estimateFlops("CHOLESKY", n), "iter=" + iter);

            // Schur
            try {
                t = measureMillis(() -> new RealSchurDecomposition().decompose(A.copy()));
                report("SCHUR", n, t, estimateFlops("SCHUR", n), "iter=" + iter);
            } catch (Exception e) {
                report("SCHUR", n, 0, estimateFlops("SCHUR", n), "iter=" + iter + ",failed:" + e.getMessage());
            }

            // SVD
            t = measureMillis(() -> new SVDecomposition().decompose(A.copy()));
            report("SVD", n, t, estimateFlops("SVD", n), "iter=" + iter);

            // Determinant
            t = measureMillis(() -> { A.copy().determinant(); });
            report("DETERMINANT", n, t, estimateFlops("DETERMINANT", n), "iter=" + iter);

            // Inverse
            t = measureMillis(() -> { A.copy().inverse(); });
            report("INVERSE", n, t, estimateFlops("INVERSE", n), "iter=" + iter);

            // Row space
            t = measureMillis(() -> { Set<Vector> s = A.rowSpaceBasis(); });
            report("ROWSPACE", n, t, estimateFlops("ROWSPACE", n), "iter=" + iter + ",size=" + A.rowSpaceBasis().size());

            // Column space
            t = measureMillis(() -> { Set<Vector> s = A.columnSpaceBasis(); });
            report("COLSPACE", n, t, estimateFlops("COLSPACE", n), "iter=" + iter + ",size=" + A.columnSpaceBasis().size());

            // Null space
            t = measureMillis(() -> { Set<Vector> s = A.nullSpaceBasis(); });
            report("NULLSPACE", n, t, estimateFlops("NULLSPACE", n), "iter=" + iter + ",size=" + A.nullSpaceBasis().size());
        }

        System.out.println("# End diagnostic");
        if (csvWriter != null) {
            try {
                csvWriter.close();
            } catch (IOException ignored) {
            }
        }

        long end = System.nanoTime();
        System.out.println("Total time (ms): " + (end - init) / 1_000_000);
    }
}
