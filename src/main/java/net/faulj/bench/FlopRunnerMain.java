package net.faulj.bench;

import net.faulj.compute.RuntimeProfile;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.matrix.Matrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class FlopRunnerMain {
    public static void main(String[] args) throws Exception {
        RuntimeProfile.applyConfiguredProfile();
        int max = 2048;
        String env = System.getenv("BENCH_MAX_SIZE");
        if (env != null) {
            try { max = Integer.parseInt(env); } catch (NumberFormatException ignored) { }
        }

        int[] sizes = new int[] {128, 256, 512, 1024, max};

        double maxQr = 0.0;
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 42);
            long t0 = System.nanoTime();
            HouseholderQR.factorize(A);
            long t1 = System.nanoTime();
            double elapsed = (t1 - t0) / 1e9;
            double flops = (4.0/3.0) * n * (double)n * n;
            double gflops = flops / elapsed / 1e9;
            if (gflops > maxQr) maxQr = gflops;
            System.out.printf("QR n=%d GFLOPs=%.3f\n", n, gflops);
        }

        double maxH = 0.0;
        for (int n : sizes) {
            Matrix A = randomMatrix(n, n, 123);
            long t0 = System.nanoTime();
            HessenbergReduction.reduceToHessenberg(A);
            long t1 = System.nanoTime();
            double elapsed = (t1 - t0) / 1e9;
            double flops = (10.0/3.0) * n * (double)n * n;
            double gflops = flops / elapsed / 1e9;
            if (gflops > maxH) maxH = gflops;
            System.out.printf("HESS n=%d GFLOPs=%.3f\n", n, gflops);
        }

        try (FileWriter fw = new FileWriter("Algorithm_FLOP_results")) {
            fw.write(String.format("HouseholderQR_max_GFLOPs=%.6f\nHessenberg_max_GFLOPs=%.6f\n", maxQr, maxH));
        }
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.length; i++) a[i] = rnd.nextDouble() - 0.5;
        return Matrix.wrap(a, rows, cols);
    }
}
