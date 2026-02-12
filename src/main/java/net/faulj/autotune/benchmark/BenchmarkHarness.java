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
    private static final int MEASURED_RUNS = MeasurementTier.BASE_REPS;

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
        return run(n, mc, kc, nc, mr, nr, WARMUP_RUNS, MEASURED_RUNS);
    }

    /**
     * Run benchmark with explicit warmup and measured counts.
     */
    public static Result run(int n, int mc, int kc, int nc, int mr, int nr,
                             int warmupRuns, int measuredRuns) {
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
        for (int w = 0; w < warmupRuns; w++) {
            Arrays.fill(c, 0.0);
            gemmBlocked(a, b, c, n, mc, kc, nc, mr, nr, vecLen, packedN, aPack, bPack);
            sink += c[0];
        }

        // ── Measured runs ───────────────────────────────────────────────
        double[] times = new double[measuredRuns];
        for (int r = 0; r < measuredRuns; r++) {
            Arrays.fill(c, 0.0);
            long start = System.nanoTime();
            gemmBlocked(a, b, c, n, mc, kc, nc, mr, nr, vecLen, packedN, aPack, bPack);
            long elapsed = System.nanoTime() - start;
            times[r] = elapsed / 1e9;
            sink += c[0];
        }

        return computeResult(mr, nr, kc, mc, nc, n, times);
    }

    /**
     * Append measured runs to an existing Result (no warmup), returning a new Result
     * with accumulated measurements. Previous result's times are preserved.
     */
    public static Result appendMeasurements(Result previous, int extraMeasured) {
        if (previous == null) throw new IllegalArgumentException("previous result required");
        int n = previous.n;
        int mc = previous.mc;
        int kc = previous.kc;
        int nc = previous.nc;
        int mr = previous.mr;
        int nr = previous.nr;

        int vecLen = SPECIES.length();

        // Allocate matrices for extra runs
        double[] a = new double[n * n];
        double[] b = new double[n * n];
        double[] c = new double[n * n];
        fillRandom(a);
        fillRandom(b);

        int packedN = PackingUtils.roundUp(nc, vecLen);
        double[] aPack = new double[mr * kc];
        double[] bPack = new double[kc * packedN];

        double[] newTimes = new double[previous.times.length + extraMeasured];
        System.arraycopy(previous.times, 0, newTimes, 0, previous.times.length);

        for (int r = 0; r < extraMeasured; r++) {
            Arrays.fill(c, 0.0);
            long start = System.nanoTime();
            gemmBlocked(a, b, c, n, mc, kc, nc, mr, nr, vecLen, packedN, aPack, bPack);
            long elapsed = System.nanoTime() - start;
            newTimes[previous.times.length + r] = elapsed / 1e9;
            sink += c[0];
        }

        return computeResult(previous.mr, previous.nr, previous.kc, previous.mc, previous.nc, n, newTimes);
    }

    private static Result computeResult(int mr, int nr, int kc, int mc, int nc, int n, double[] times) {
        double flops = 2.0 * n * n * n;

        double[] timesCopy = times.clone();
        Arrays.sort(timesCopy);
        double bestSec = timesCopy[0];
        double medianSec = timesCopy[timesCopy.length / 2];

        // Convert to GFLOP/s values per-run
        double[] gflops = new double[times.length];
        for (int i = 0; i < times.length; i++) gflops[i] = flops / times[i] / 1e9;

        // Median in GFLOP/s
        double[] gflopsCopy = gflops.clone();
        Arrays.sort(gflopsCopy);
        double medianGflops = gflopsCopy[gflopsCopy.length / 2];

        // Unbiased sample stddev over GFLOP/s
        double mean = 0.0;
        for (double g : gflops) mean += g;
        mean /= gflops.length;

        double variance = 0.0;
        for (double g : gflops) variance += (g - mean) * (g - mean);
        double stddev = gflops.length > 1 ? Math.sqrt(variance / (gflops.length - 1)) : 0.0;

        double cv = medianGflops > 0 ? stddev / medianGflops : 0.0;

        double bestGflops = flops / bestSec / 1e9;

        return new Result(mr, nr, kc, mc, nc, n, bestGflops, medianGflops, cv, stddev, times);
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
        public final int n;
        public final double bestGflops;
        public final double medianGflops;
        public final double cv; // coefficient of variation (stability)
        public final double stddev; // unbiased sample stddev over GFLOP/s
        public final double[] times; // measured run times in seconds

        public Result(int mr, int nr, int kc, int mc, int nc, int n,
                      double bestGflops, double medianGflops, double cv,
                      double stddev, double[] times) {
            this.mr = mr;
            this.nr = nr;
            this.kc = kc;
            this.mc = mc;
            this.nc = nc;
            this.n = n;
            this.bestGflops = bestGflops;
            this.medianGflops = medianGflops;
            this.cv = cv;
            this.stddev = stddev;
            this.times = times != null ? times.clone() : new double[0];
        }

        public int getN() { return (int)Math.round(Math.cbrt((bestGflops>0? bestGflops:1.0) * 1e9)); }
    }
}
