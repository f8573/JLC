package net.faulj.bench;

import net.faulj.decomposition.qr.QRFactory;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import net.faulj.util.PerfTimers;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class RunCAQRPerf {
    public static void main(String[] args) throws IOException {
        System.setProperty("la.qr.strategy", "CAQR");
        System.setProperty("la.qr.caqr.p", "32");
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "16");

        int m = 4096;
        int n = 256;

        Matrix A = randomMatrix(m, n, 1234567L);

        // Warmup
        for (int i = 0; i < 2; i++) {
            QRFactory.decompose(A.copy(), true);
        }

        long t0 = System.nanoTime();
        QRResult res = QRFactory.decompose(A.copy(), true);
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("CAQR run finished in " + secs + " s");

        // Dump timers
        try {
            PerfTimers.dump(new File("build/reports/caqr_timers.csv"));
            System.out.println("Wrote build/reports/caqr_timers.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Touch result to avoid GC elimination
        System.out.println("R[0,0]=" + res.getR().get(0, 0));
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rnd.nextGaussian();
            }
        }
        return new Matrix(data);
    }
}
