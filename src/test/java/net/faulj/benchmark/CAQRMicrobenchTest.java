package net.faulj.benchmark;

import net.faulj.decomposition.qr.QRFactory;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Random;

/**
 * Microbenchmark to compare CAQR vs Householder QR on tall matrices.
 * Run explicitly via Gradle: `./gradlew test --tests "net.faulj.benchmark.CAQRMicrobenchTest" -Dla.qr.strategy=CAQR`
 */
public class CAQRMicrobenchTest {

    private static final Path OUT = Paths.get("build", "reports", "caqr_bench.csv");

    @Test
    public void runMicrobench() throws IOException {
        int[] ms = {1024, 2048, 4096};
        int[] ns = {64, 128, 256};

        Files.createDirectories(OUT.getParent());
        StringBuilder sb = new StringBuilder();
        sb.append("strategy,m,n,warmup,measured_seconds,gflops,flops_count\n");

        String strategy = System.getProperty("la.qr.strategy", "HOUSEHOLDER");
        for (int m : ms) {
            for (int n : ns) {
                if (n > m) continue;
                Matrix A = randomMatrix(m, n, 1234L + m + n);
                // flop count for QR (m>=n): approx 2*m*n*n - 2/3*n^3
                double flops = 2.0 * m * n * n - (2.0 / 3.0) * n * n * n;

                // Warmup runs
                for (int w = 0; w < 3; w++) {
                    QRFactory.decompose(A.copy(), true);
                }

                // Measured runs (median of 5)
                double[] runs = new double[5];
                for (int r = 0; r < runs.length; r++) {
                    Matrix Ac = A.copy();
                    long t0 = System.nanoTime();
                    QRResult res = QRFactory.decompose(Ac, true);
                    long dt = System.nanoTime() - t0;
                    runs[r] = dt / 1e9;
                    // touch result
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
