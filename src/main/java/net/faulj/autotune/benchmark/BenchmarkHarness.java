package net.faulj.autotune.benchmark;

import net.faulj.compute.MicroKernel;
import net.faulj.compute.PackingUtils;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Runs a single GEMM benchmark with explicit blocking parameters.
 *
 * <p>Executes a BLIS-style 5-loop nest using the existing
 * {@link MicroKernel#compute} and {@link PackingUtils} with
 * caller-specified (MC, KC, NC, MR, NR) block sizes.</p>
 *
 * <p>Does not modify any GEMM kernel code. Calls existing infrastructure
 * with custom parameters for timing purposes only.</p>
 */
public final class BenchmarkHarness {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final int WARMUP_RUNS = 3;
    private static final int MEASURED_RUNS = 7;

    private static volatile double sink;

    private BenchmarkHarness() {}

    /**
     * Benchmark one GEMM configuration.
     *
     * @param n      square matrix dimension (C = A*B where A,B,C are n×n)
     * @param mc     M blocking parameter
     * @param kc     K blocking parameter
     * @param nc     N blocking parameter
     * @param mr     microkernel row count
     * @param nr     microkernel column count
     * @return benchmark result with GFLOP/s and stability metrics
     */
    public static Result run(int n, int mc, int kc, int nc, int mr, int nr) {
        int vecLen = SPECIES.length();

        // ── Allocate matrices (row-major) ───────────────────────────────
        double[] a = new double[n * n];
        double[] b = new double[n * n];
        double[] c = new double[n * n];
        fillRandom(a);
        fillRandom(b);

        // ── Allocate packing buffers ────────────────────────────────────
        int packedN = PackingUtils.roundUp(nc, vecLen);
        double[] aPack = new double[mr * kc];
        double[] bPack = new double[kc * packedN];

        // ── Warmup ──────────────────────────────────────────────────────
        for (int w = 0; w < WARMUP_RUNS; w++) {
            Arrays.fill(c, 0.0);
            gemmBlocked(a, b, c, n, mc, kc, nc, mr, nr, vecLen, packedN, aPack, bPack);
            sink += c[0];
        }

        // ── Measured runs ───────────────────────────────────────────────
        double[] times = new double[MEASURED_RUNS];
        for (int r = 0; r < MEASURED_RUNS; r++) {
            Arrays.fill(c, 0.0);
            long start = System.nanoTime();
            gemmBlocked(a, b, c, n, mc, kc, nc, mr, nr, vecLen, packedN, aPack, bPack);
            long elapsed = System.nanoTime() - start;
            times[r] = elapsed / 1e9;
            sink += c[0];
        }

        // ── Statistics ──────────────────────────────────────────────────
        double flops = 2.0 * n * n * n;
        Arrays.sort(times);
        double bestSec = times[0];
        double medianSec = times[MEASURED_RUNS / 2];

        double mean = 0.0;
        for (double t : times) mean += t;
        mean /= times.length;

        double variance = 0.0;
        for (double t : times) variance += (t - mean) * (t - mean);
        variance /= times.length;
        double stddev = Math.sqrt(variance);
        double cv = mean > 0 ? stddev / mean : 0.0;

        double bestGflops = flops / bestSec / 1e9;
        double medianGflops = flops / medianSec / 1e9;

        return new Result(mr, nr, kc, mc, nc, bestGflops, medianGflops, cv);
    }

    /**
     * BLIS-style 5-loop nest with explicit block sizes.
     * Calls existing PackingUtils and MicroKernel.compute().
     */
    private static void gemmBlocked(double[] a, double[] b, double[] c, int n,
                                     int mc, int kc, int nc, int mr, int nr,
                                     int vecLen, int packedN,
                                     double[] aPack, double[] bPack) {
        // Loop 1: NC blocks over N
        for (int jj = 0; jj < n; jj += nc) {
            int nBlock = Math.min(nc, n - jj);
            int packedNBlock = PackingUtils.roundUp(nBlock, vecLen);

            // Reallocate bPack if block size differs
            double[] bBuf = (packedNBlock * kc <= bPack.length)
                    ? bPack : new double[kc * packedNBlock];

            // Loop 2: KC blocks over K
            for (int kk = 0; kk < n; kk += kc) {
                int kBlock = Math.min(kc, n - kk);

                // Pack B panel [kBlock × nBlock]
                PackingUtils.packB(b, n, kk, kBlock, jj, nBlock, packedNBlock, bBuf);

                // Loop 3: MC blocks over M
                for (int ii = 0; ii < n; ii += mc) {
                    int mBlock = Math.min(mc, n - ii);

                    // Loop 4: MR strips within MC block
                    for (int i = ii; i < ii + mBlock; i += mr) {
                        int mStrip = Math.min(mr, ii + mBlock - i);

                        // Pack A strip [mStrip × kBlock]
                        double[] aBuf = (mStrip * kBlock <= aPack.length)
                                ? aPack : new double[mStrip * kBlock];
                        PackingUtils.packA(a, n, i, mStrip, kk, kBlock, 1.0, aBuf);

                        // Loop 5: Microkernel
                        MicroKernel.compute(mStrip, kBlock, packedNBlock, nBlock,
                                aBuf, bBuf, c, i * n + jj, n);
                    }
                }
            }
        }
    }

    private static void fillRandom(double[] arr) {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < arr.length; i++) {
            arr[i] = rng.nextDouble(-1.0, 1.0);
        }
    }

    /**
     * Result of a single benchmark configuration.
     */
    public static final class Result {
        public final int mr;
        public final int nr;
        public final int kc;
        public final int mc;
        public final int nc;
        public final double bestGflops;
        public final double medianGflops;
        public final double cv; // coefficient of variation (stability)

        public Result(int mr, int nr, int kc, int mc, int nc,
                      double bestGflops, double medianGflops, double cv) {
            this.mr = mr;
            this.nr = nr;
            this.kc = kc;
            this.mc = mc;
            this.nc = nc;
            this.bestGflops = bestGflops;
            this.medianGflops = medianGflops;
            this.cv = cv;
        }
    }
}
