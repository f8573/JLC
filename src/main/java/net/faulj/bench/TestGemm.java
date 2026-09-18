package net.faulj.bench;

import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.util.PerfTimers;

import java.io.IOException;

public class TestGemm {
    public static void main(String[] args) throws IOException {
        int m = 256, k = 256, n = 256;
        Matrix A = randomMatrix(m, k);
        Matrix B = randomMatrix(k, n);
        Matrix C = Matrix.zero(m, n);

        // Warmup
        for (int i = 0; i < 2; i++) {
            Gemm.gemm(A, B, C, 1.0, 0.0, null);
        }

        long t0 = System.nanoTime();
        Gemm.gemm(A, B, C, 1.0, 0.0, null);
        long dt = System.nanoTime() - t0;
        System.out.println("GEMM dt(s): " + (dt / 1e9));

        PerfTimers.dump(new java.io.File("build/reports/caqr_timers_gemm.csv"));
        System.out.println("Wrote gemm timers");
    }

    private static Matrix randomMatrix(int r, int c) {
        double[][] a = new double[r][c];
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) a[i][j] = rnd.nextDouble();
        return new Matrix(a);
    }
}
