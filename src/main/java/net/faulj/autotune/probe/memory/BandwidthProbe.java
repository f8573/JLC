package net.faulj.autotune.probe.memory;

import net.faulj.autotune.probe.ProbeConfidence;

import java.util.concurrent.CountDownLatch;

/**
 * Measures sustainable memory bandwidth at each cache level using a
 * STREAM-triad microbenchmark ({@code a[i] = b[i] + scalar * c[i]}).
 */
public final class BandwidthProbe {

    private static final int WARMUP = 3;
    private static final int RUNS = 6;

    private static final int L1_ELEMENTS = 1_024;        // ~8 KB (elements × 8 bytes)
    private static final int L2_ELEMENTS = 8_192;         // ~64 KB
    private static final int L3_ELEMENTS = 262_144;       // ~2 MB
    private static final int DRAM_ELEMENTS = 4_194_304;   // ~32 MB

    private static volatile double sink;

    private BandwidthProbe() {}

    public static BandwidthResult run() {
        return run(Math.max(1, Runtime.getRuntime().availableProcessors()));
    }

    public static BandwidthResult run(int threads) {
        threads = Math.max(1, threads);
        try {
            double l1 = measureSingle(L1_ELEMENTS);
            double l2 = measureSingle(L2_ELEMENTS);
            double l3 = threads > 1 ? measureParallel(L3_ELEMENTS, threads) : measureSingle(L3_ELEMENTS);
            double dram = threads > 1 ? measureParallel(DRAM_ELEMENTS, threads) : measureSingle(DRAM_ELEMENTS);
            if (l1 <= 0 || l2 <= 0 || l3 <= 0 || dram <= 0) return new BandwidthResult(0,0,0,0, ProbeConfidence.FAILED);
            return new BandwidthResult(l1, l2, l3, dram, ProbeConfidence.MEASURED);
        } catch (Exception e) {
            return new BandwidthResult(0,0,0,0, ProbeConfidence.FAILED);
        }
    }

    private static double measureSingle(int elements) {
        int loops = Math.max(4, 1_048_576 / Math.max(1, elements));
        double[] a = new double[elements];
        double[] b = new double[elements];
        double[] c = new double[elements];
        initArrays(b, c);
        for (int w = 0; w < WARMUP; w++) triad(a, b, c, loops);
        double best = 0.0;
        for (int r = 0; r < RUNS; r++) best = Math.max(best, triad(a, b, c, loops));
        return best;
    }

    private static double measureParallel(int elementsPerThread, int threads) {
        int loops = Math.max(4, 1_048_576 / Math.max(1, elementsPerThread));
        double[][] aArr = new double[threads][];
        double[][] bArr = new double[threads][];
        double[][] cArr = new double[threads][];
        for (int t = 0; t < threads; t++) {
            aArr[t] = new double[elementsPerThread];
            bArr[t] = new double[elementsPerThread];
            cArr[t] = new double[elementsPerThread];
            initArrays(bArr[t], cArr[t]);
        }
        for (int w = 0; w < WARMUP; w++) runParallelTriad(aArr, bArr, cArr, loops, threads);
        double best = 0.0;
        for (int r = 0; r < RUNS; r++) best = Math.max(best, runParallelTriad(aArr, bArr, cArr, loops, threads));
        return best;
    }

    private static double runParallelTriad(double[][] aArr, double[][] bArr,
                                           double[][] cArr, int loops, int threads) {
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch endGate = new CountDownLatch(threads);
        for (int t = 0; t < threads; t++) {
            final int tid = t;
            Thread worker = new Thread(() -> {
                try { startGate.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); return; }
                triad(aArr[tid], bArr[tid], cArr[tid], loops);
                endGate.countDown();
            });
            worker.setDaemon(true);
            worker.start();
        }
        long start = System.nanoTime();
        startGate.countDown();
        try { endGate.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        long elapsed = System.nanoTime() - start;
        double seconds = Math.max(1e-9, elapsed / 1e9);
        int n = aArr[0].length;
        double bytesPerThread = (double) loops * n * BYTES_PER_ELEMENT;
        return (bytesPerThread * threads) / seconds;
    }

    private static final double BYTES_PER_ELEMENT = 4.0 * Double.BYTES;

    static double triad(double[] a, double[] b, double[] c, int loops) {
        final int n = a.length;
        final double scalar = 1.00000011921;
        long start = System.nanoTime();
        for (int l = 0; l < loops; l++) {
            for (int i = 0; i < n; i++) {
                a[i] = b[i] + scalar * c[i];
            }
            double[] tmp = b; b = a; a = c; c = tmp;
        }
        long elapsed = System.nanoTime() - start;
        sink += b[(loops * 131) % n];
        double seconds = Math.max(1e-9, elapsed / 1e9);
        double bytesMoved = (double) loops * n * BYTES_PER_ELEMENT;
        return bytesMoved / seconds;
    }

    private static void initArrays(double[] b, double[] c) {
        for (int i = 0; i < b.length; i++) {
            b[i] = 1.0 + (i % 7) * 0.1;
            c[i] = 2.0 + (i % 5) * 0.2;
        }
    }

    /**
     * Measures bandwidth for a specific size (in bytes) with configurable warmup/runs.
     * Used by CacheSizeProbe for sweep measurements.
     */
    public static double measureTriadBytesPerSec(int sizeBytes, int warmup, int runs) {
        int elements = Math.max(8, sizeBytes / Double.BYTES);
        int loops = Math.max(4, 1_048_576 / elements);
        double[] a = new double[elements];
        double[] b = new double[elements];
        double[] c = new double[elements];
        initArrays(b, c);
        for (int w = 0; w < warmup; w++) triad(a, b, c, loops);
        double best = 0.0;
        for (int r = 0; r < runs; r++) best = Math.max(best, triad(a, b, c, loops));
        return best;
    }

    public static final class BandwidthResult {
        public final double l1BytesPerSec;
        public final double l2BytesPerSec;
        public final double l3BytesPerSec;
        public final double dramBytesPerSec;
        public final ProbeConfidence confidence;

        public BandwidthResult(double l1, double l2, double l3, double dram,
                               ProbeConfidence confidence) {
            this.l1BytesPerSec = l1;
            this.l2BytesPerSec = l2;
            this.l3BytesPerSec = l3;
            this.dramBytesPerSec = dram;
            this.confidence = confidence;
        }
    }
}
