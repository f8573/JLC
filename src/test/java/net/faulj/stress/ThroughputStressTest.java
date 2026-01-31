package net.faulj.stress;

import org.junit.Test;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.Assert.assertTrue;

/**
 * Throughput-focused stress tests to measure maximum FLOPS with minimal memory movement.
 * Benchmarks three methodologies for both float and double: scalar (single-threaded),
 * parallel (multi-threaded, each thread keeps state in registers), and register-vectorized
 * (keeps small vector in registers/local variables to simulate SIMD).
 */
public class ThroughputStressTest {

    // Target measurement durations (ms)
    private static final long WARMUP_MS = 500;
    private static final long MEASURE_MS = 2000;

    @Test
    public void throughputDouble() throws Exception {
        System.out.printf("[throughput-double] cores=%d%n", Runtime.getRuntime().availableProcessors());

        double scalar = runAdaptive(() -> runScalarDouble(1_000_000L), WARMUP_MS, MEASURE_MS, 8L);
        System.out.printf("[double] scalar: %.2f MFlops/s%n", scalar);

        double parallel = runAdaptive(() -> runParallelDouble(1_000_000L), WARMUP_MS, MEASURE_MS, 8L * Runtime.getRuntime().availableProcessors());
        System.out.printf("[double] parallel: %.2f MFlops/s%n", parallel);

        int vecLen = 8; // keep small vector in registers/local vars
        double vectorized = runAdaptive(() -> runVectorizedDouble(500_000L, vecLen), WARMUP_MS, MEASURE_MS, 8L * vecLen);
        System.out.printf("[double] register-vectorized(len=%d): %.2f MFlops/s%n", vecLen, vectorized);

        assertTrue(scalar > 0.0);
    }

    @Test
    public void throughputFloat() throws Exception {
        System.out.printf("[throughput-float] cores=%d%n", Runtime.getRuntime().availableProcessors());

        double scalar = runAdaptive(() -> runScalarFloat(2_000_000L), WARMUP_MS, MEASURE_MS, 8L);
        System.out.printf("[float] scalar: %.2f MFlops/s%n", scalar);

        double parallel = runAdaptive(() -> runParallelFloat(2_000_000L), WARMUP_MS, MEASURE_MS, 8L * Runtime.getRuntime().availableProcessors());
        System.out.printf("[float] parallel: %.2f MFlops/s%n", parallel);

        int vecLen = 16; // more lanes for float
        double vectorized = runAdaptive(() -> runVectorizedFloat(1_000_000L, vecLen), WARMUP_MS, MEASURE_MS, 8L * vecLen);
        System.out.printf("[float] register-vectorized(len=%d): %.2f MFlops/s%n", vecLen, vectorized);

        assertTrue(scalar > 0.0);
    }

    /**
     * Adaptively grows the iteration count to produce a stable measurement close to MEASURE_MS.
     * The callable should accept an iteration count baked in; here we invoke the callable repeatedly
     * to warm up and measure. The returned value is MFlops/s measured in the final run.
     *
     * @param work supplier that runs a workload and returns MFlops/s for the work it performed
     */
    private static double runAdaptive(Callable<Double> work, long warmupMs, long measureMs, long flopsPerIteration) throws Exception {
        // Warmup
        long startWarm = System.nanoTime();
        while ((System.nanoTime() - startWarm) < warmupMs * 1_000_000L) {
            work.call();
        }

        // Measurement runs: run repeatedly until we've measured at least measureMs total
        long start = System.nanoTime();
        double lastThroughput = 0.0;
        long measuredNanos = 0L;
        while (measuredNanos < measureMs * 1_000_000L) {
            lastThroughput = work.call();
            measuredNanos = System.nanoTime() - start;
        }

        return lastThroughput;
    }

    // --------------------- Double implementations ---------------------

    /**
     * Tight scalar loop using Math.fma on doubles. Each iteration performs 4 FMAs (8 flops).
     */
    private static double runScalarDouble(long iterations) {
        double a = 1.23456789;
        double b = 1.0000001;
        double c = 0.9999999;
        double d = 0.3333333;

        long iters = iterations;
        long t0 = System.nanoTime();
        for (long i = 0; i < iters; i++) {
            a = Math.fma(a, b, c);
            b = Math.fma(b, c, d);
            c = Math.fma(c, d, a);
            d = Math.fma(d, a, b);
        }
        long t1 = System.nanoTime();

        // Prevent the optimizer from removing the loop
        if (Double.isNaN(a + b + c + d)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) iters * 4.0 * 2.0; // 4 FMAs * 2 flops each
        return (flops / 1e6) / seconds;
    }

    /**
     * Parallel version: spawn N threads, each runs a scalar loop on local registers.
     */
    private static double runParallelDouble(long iterations) throws Exception {
        int threads = Runtime.getRuntime().availableProcessors();
        ExecutorService exec = Executors.newFixedThreadPool(threads);
        long itersPer = Math.max(1L, iterations / threads);

        long t0 = System.nanoTime();
        Future<Double>[] futures = new Future[threads];
        for (int t = 0; t < threads; t++) {
            futures[t] = exec.submit(() -> {
                double a = 1.1, b = 1.000001, c = 0.999999, d = 0.333333;
                for (long i = 0; i < itersPer; i++) {
                    a = Math.fma(a, b, c);
                    b = Math.fma(b, c, d);
                    c = Math.fma(c, d, a);
                    d = Math.fma(d, a, b);
                }
                // return local sum to avoid optimization
                return a + b + c + d;
            });
        }

        double acc = 0.0;
        for (int t = 0; t < threads; t++) acc += futures[t].get();
        long t1 = System.nanoTime();
        exec.shutdownNow();

        if (Double.isNaN(acc)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) itersPer * threads * 4.0 * 2.0;
        return (flops / 1e6) / seconds;
    }

    /**
     * Register-vectorized: keep a small vector of doubles in locals and operate on them.
     * The JIT can map these to registers and SIMD lanes.
     */
    private static double runVectorizedDouble(long iterations, int vecLen) {
        // allocate locals
        double[] v = new double[vecLen];
        double[] b = new double[vecLen];
        double[] c = new double[vecLen];
        for (int i = 0; i < vecLen; i++) {
            v[i] = 1.2345 + i * 0.1;
            b[i] = 1.000001 + i * 1e-6;
            c[i] = 0.999999 - i * 1e-6;
        }

        long iters = iterations;
        long t0 = System.nanoTime();
        for (long it = 0; it < iters; it++) {
            // operate on all lanes; kept in local array to allow registerization
            for (int k = 0; k < vecLen; k++) {
                v[k] = Math.fma(v[k], b[k], c[k]);
                v[k] = Math.fma(v[k], b[k], c[k]);
            }
        }
        long t1 = System.nanoTime();

        double s = 0.0;
        for (int i = 0; i < vecLen; i++) s += v[i];
        if (Double.isNaN(s)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) iters * vecLen * 2.0 * 2.0; // 2 FMAs per lane per iter * 2 flops
        return (flops / 1e6) / seconds;
    }

    // --------------------- Float implementations ---------------------

    private static double runScalarFloat(long iterations) {
        float a = 1.2345f;
        float b = 1.0000001f;
        float c = 0.9999999f;
        float d = 0.3333333f;

        long iters = iterations;
        long t0 = System.nanoTime();
        for (long i = 0; i < iters; i++) {
            a = Math.fma(a, b, c);
            b = Math.fma(b, c, d);
            c = Math.fma(c, d, a);
            d = Math.fma(d, a, b);
        }
        long t1 = System.nanoTime();

        if (Float.isNaN(a + b + c + d)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) iters * 4.0 * 2.0;
        return (flops / 1e6) / seconds;
    }

    private static double runParallelFloat(long iterations) throws Exception {
        int threads = Runtime.getRuntime().availableProcessors();
        ExecutorService exec = Executors.newFixedThreadPool(threads);
        long itersPer = Math.max(1L, iterations / threads);

        long t0 = System.nanoTime();
        Future<Float>[] futures = new Future[threads];
        for (int t = 0; t < threads; t++) {
            futures[t] = exec.submit(() -> {
                float a = 1.1f, b = 1.000001f, c = 0.999999f, d = 0.333333f;
                for (long i = 0; i < itersPer; i++) {
                    a = Math.fma(a, b, c);
                    b = Math.fma(b, c, d);
                    c = Math.fma(c, d, a);
                    d = Math.fma(d, a, b);
                }
                return a + b + c + d;
            });
        }

        double acc = 0.0;
        for (int t = 0; t < threads; t++) acc += futures[t].get();
        long t1 = System.nanoTime();
        exec.shutdownNow();

        if (Double.isNaN(acc)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) itersPer * threads * 4.0 * 2.0;
        return (flops / 1e6) / seconds;
    }

    private static double runVectorizedFloat(long iterations, int vecLen) {
        float[] v = new float[vecLen];
        float[] b = new float[vecLen];
        float[] c = new float[vecLen];
        for (int i = 0; i < vecLen; i++) {
            v[i] = 1.2345f + i * 0.1f;
            b[i] = 1.000001f + i * 1e-6f;
            c[i] = 0.999999f - i * 1e-6f;
        }

        long iters = iterations;
        long t0 = System.nanoTime();
        for (long it = 0; it < iters; it++) {
            for (int k = 0; k < vecLen; k++) {
                v[k] = Math.fma(v[k], b[k], c[k]);
                v[k] = Math.fma(v[k], b[k], c[k]);
            }
        }
        long t1 = System.nanoTime();

        float s = 0f;
        for (int i = 0; i < vecLen; i++) s += v[i];
        if (Float.isNaN(s)) throw new RuntimeException("NaN");

        double seconds = (t1 - t0) / 1e9;
        double flops = (double) iters * vecLen * 2.0 * 2.0;
        return (flops / 1e6) / seconds;
    }
}
