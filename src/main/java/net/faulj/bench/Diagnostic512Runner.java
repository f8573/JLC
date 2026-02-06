package net.faulj.bench;

import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Set;

/**
 * Standalone Diagnostic512 runner used by backend via isolated JVM process.
 */
public final class Diagnostic512Runner {
    private Diagnostic512Runner() {}

    private static BufferedWriter csvWriter;

    private static long measureMillis(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000;
    }

    private static double estimateFlops(String op, int n) {
        double N3 = (double) n * n * n;
        return switch (op) {
            case "LU" -> 2.0 / 3.0 * N3;
            case "QR" -> 2.0 / 3.0 * N3;
            case "HESSENBERG" -> 10.0 / 3.0 * N3;
            case "BIDIAG" -> 4.0 / 3.0 * N3;
            case "SVD" -> 4.0 * N3;
            case "POLAR" -> 4.0 * N3;
            case "CHOLESKY" -> 1.0 / 3.0 * N3;
            case "SCHUR" -> 4.0 * N3;
            case "INVERSE" -> 2.0 * N3;
            case "DETERMINANT" -> 2.0 / 3.0 * N3;
            case "ROWSPACE" -> 2.0 / 3.0 * N3;
            case "COLSPACE" -> 2.0 / 3.0 * N3;
            case "NULLSPACE" -> 2.0 / 3.0 * N3;
            default -> 0.0;
        };
    }

    private static void report(String op, int n, int iter, long ms, double flops, String note) {
        double flopsPerSec = ms > 0 ? (flops * 1000.0 / ms) : 0.0;
        String info = note == null ? "" : note.replace(',', '|');
        String line = String.format("%s,%d,%d,%d,%.0f,%.2f,%s", op, n, iter, ms, flops, flopsPerSec, info);
        System.out.println(line);
        if (csvWriter != null) {
            try {
                csvWriter.write(line);
                csvWriter.newLine();
                csvWriter.flush();
            } catch (IOException ignored) {}
        }
    }

    public static void main(String[] args) {
        final int n = 512;
        int iterations = 5;
        String output = "diagnostic_512_results.csv";

        for (String arg : args) {
            if (arg != null && arg.startsWith("--iterations=")) {
                try { iterations = Math.max(1, Integer.parseInt(arg.substring("--iterations=".length()))); }
                catch (Exception ignored) {}
            } else if (arg != null && arg.startsWith("--output=")) {
                output = arg.substring("--output=".length());
            }
        }

        System.out.println("operation,n,iteration,ms,flops,flop/s,info");
        try {
            csvWriter = new BufferedWriter(new FileWriter(output));
            csvWriter.write("operation,n,iteration,ms,flops,flop/s,info");
            csvWriter.newLine();
            csvWriter.flush();
        } catch (IOException ignored) {}

        long totalStart = System.nanoTime();

        Matrix A = Matrix.randomMatrix(n, n);
        Matrix P = A.multiply(A.transpose());
        for (int i = 0; i < n; i++) P.set(i, i, P.get(i, i) + n);

        try { HouseholderQR.decompose(A.copy()); } catch (Exception ignored) {}
        try { HessenbergReduction.decompose(A.copy()); } catch (Exception ignored) {}
        try { new Bidiagonalization().decompose(A.copy()); } catch (Exception ignored) {}
        try { new LUDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        try { new SVDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        try { new PolarDecomposition().decompose(A.copy()); } catch (Exception ignored) {}
        try { new CholeskyDecomposition().decompose(P.copy()); } catch (Exception ignored) {}
        try { new RealSchurDecomposition().decompose(A.copy()); } catch (Exception ignored) {}

        for (int iter = 1; iter <= iterations; iter++) {
            long t;
            t = measureMillis(() -> HouseholderQR.decompose(A.copy()));
            report("QR", n, iter, t, estimateFlops("QR", n), null);

            t = measureMillis(() -> HessenbergReduction.decompose(A.copy()));
            report("HESSENBERG", n, iter, t, estimateFlops("HESSENBERG", n), null);

            t = measureMillis(() -> new Bidiagonalization().decompose(A.copy()));
            report("BIDIAG", n, iter, t, estimateFlops("BIDIAG", n), null);

            t = measureMillis(() -> new LUDecomposition().decompose(A.copy()));
            report("LU", n, iter, t, estimateFlops("LU", n), null);

            t = measureMillis(() -> new PolarDecomposition().decompose(A.copy()));
            report("POLAR", n, iter, t, estimateFlops("POLAR", n), null);

            t = measureMillis(() -> new CholeskyDecomposition().decompose(P.copy()));
            report("CHOLESKY", n, iter, t, estimateFlops("CHOLESKY", n), null);

            try {
                t = measureMillis(() -> new RealSchurDecomposition().decompose(A.copy()));
                report("SCHUR", n, iter, t, estimateFlops("SCHUR", n), null);
            } catch (Exception ex) {
                report("SCHUR", n, iter, 0, estimateFlops("SCHUR", n), "failed:" + ex.getMessage());
            }

            t = measureMillis(() -> new SVDecomposition().decompose(A.copy()));
            report("SVD", n, iter, t, estimateFlops("SVD", n), null);

            t = measureMillis(() -> A.copy().determinant());
            report("DETERMINANT", n, iter, t, estimateFlops("DETERMINANT", n), null);

            t = measureMillis(() -> A.copy().inverse());
            report("INVERSE", n, iter, t, estimateFlops("INVERSE", n), null);

            t = measureMillis(A::rowSpaceBasis);
            Set<Vector> rs = A.rowSpaceBasis();
            report("ROWSPACE", n, iter, t, estimateFlops("ROWSPACE", n), "size=" + (rs == null ? 0 : rs.size()));

            t = measureMillis(A::columnSpaceBasis);
            Set<Vector> cs = A.columnSpaceBasis();
            report("COLSPACE", n, iter, t, estimateFlops("COLSPACE", n), "size=" + (cs == null ? 0 : cs.size()));

            t = measureMillis(A::nullSpaceBasis);
            Set<Vector> ns = A.nullSpaceBasis();
            report("NULLSPACE", n, iter, t, estimateFlops("NULLSPACE", n), "size=" + (ns == null ? 0 : ns.size()));
        }

        long totalMs = (System.nanoTime() - totalStart) / 1_000_000;
        System.out.println("TOTAL_MS=" + totalMs);
        if (csvWriter != null) {
            try { csvWriter.close(); } catch (IOException ignored) {}
        }
    }
}
