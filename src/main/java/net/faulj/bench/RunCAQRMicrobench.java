package net.faulj.bench;

import net.faulj.decomposition.qr.QRFactory;
import net.faulj.matrix.Matrix;
import net.faulj.util.PerfTimers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Random;

public class RunCAQRMicrobench {
    public static void main(String[] args) throws IOException {
        String strategy = System.getProperty("la.qr.strategy", "CAQR");
        System.setProperty("la.qr.strategy", strategy);

        int[] ms = {1024, 2048, 4096};
        int[] ns = {64, 128, 256};

        Path OUT = Paths.get("build", "reports", "caqr_bench.csv");
        Files.createDirectories(OUT.getParent());
        StringBuilder sb = new StringBuilder();
        sb.append("strategy,m,n,warmup,measured_seconds,gflops,flops_count\n");

        for (int m : ms) {
            for (int n : ns) {
                if (n > m) continue;
                Matrix A = randomMatrix(m, n, 1234L + m + n);
                double flops = 2.0 * m * n * n - (2.0 / 3.0) * n * n * n;

                for (int w = 0; w < 3; w++) {
                    QRFactory.decompose(A.copy(), true);
                }

                double[] runs = new double[5];
                for (int r = 0; r < runs.length; r++) {
                    Matrix Ac = A.copy();
                    long t0 = System.nanoTime();
                    var res = QRFactory.decompose(Ac, true);
                    long dt = System.nanoTime() - t0;
                    runs[r] = dt / 1e9;
                    double check = res.getR().get(0, 0) + res.getQ().get(0, 0);
                    if (!Double.isFinite(check)) throw new RuntimeException("Non-finite result");
                }
                java.util.Arrays.sort(runs);
                double median = runs[runs.length / 2];
                double gflops = flops / (median * 1e9);

                sb.append(String.format(Locale.ROOT, "%s,%d,%d,%d,%.6f,%.6f,%.0f\n",
                    strategy, m, n, 3, median, gflops, flops));
            }
        }

        Files.writeString(OUT, sb.toString());
        System.out.println("Wrote microbench CSV to " + OUT.toString());

        try {
            PerfTimers.dump(new java.io.File("build/reports/caqr_timers.csv"));
            System.out.println("Wrote perf timers to build/reports/caqr_timers.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
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
