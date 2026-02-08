package net.faulj.benchmark.roofline;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.matrix.Matrix;

import java.util.Locale;
import java.util.Random;

final class ComputeRoofProbe {
    private static final String COMPUTE_GFLPS_PROPERTY = "jlc.roofline.compute_gflops";
    private static final String COMPUTE_GFLPS_ENV = "JLC_ROOFLINE_COMPUTE_GFLOPS";
    private static final String THEORETICAL_UTIL_PROPERTY = "jlc.roofline.theoretical_util";
    private static final String GEMM_ANCHOR_SCALE_PROPERTY = "jlc.roofline.gemm_anchor_scale";
    private static final String GEMM_ANCHOR_N_PROPERTY = "jlc.roofline.gemm_anchor_n";
    private static final String GEMM_ANCHOR_WARMUP_PROPERTY = "jlc.roofline.gemm_anchor_warmup";
    private static final String GEMM_ANCHOR_RUNS_PROPERTY = "jlc.roofline.gemm_anchor_runs";

    private static final double DEFAULT_THEORETICAL_UTIL = 0.60;
    private static final double DEFAULT_GEMM_ANCHOR_SCALE = 1.50;
    private static final int DEFAULT_GEMM_ANCHOR_N = 512;
    private static final int DEFAULT_GEMM_ANCHOR_WARMUP = 2;
    private static final int DEFAULT_GEMM_ANCHOR_RUNS = 4;

    private static volatile double sink;

    private ComputeRoofProbe() {
    }

    static ComputeEstimate probe(HardwareInfo hardware) {
        Double explicit = parsePositiveDouble(System.getProperty(COMPUTE_GFLPS_PROPERTY));
        if (explicit != null) {
            return new ComputeEstimate(
                explicit * 1e9,
                "property:" + COMPUTE_GFLPS_PROPERTY,
                hardware.peakFlopsPerSecond,
                Double.NaN
            );
        }
        explicit = parsePositiveDouble(System.getenv(COMPUTE_GFLPS_ENV));
        if (explicit != null) {
            return new ComputeEstimate(
                explicit * 1e9,
                "env:" + COMPUTE_GFLPS_ENV,
                hardware.peakFlopsPerSecond,
                Double.NaN
            );
        }

        double theoreticalUtil = parsePositiveDouble(System.getProperty(THEORETICAL_UTIL_PROPERTY), DEFAULT_THEORETICAL_UTIL);
        double theoreticalCap = hardware.peakFlopsPerSecond * theoreticalUtil;

        double anchorScale = parsePositiveDouble(System.getProperty(GEMM_ANCHOR_SCALE_PROPERTY), DEFAULT_GEMM_ANCHOR_SCALE);
        double anchoredPeak = Double.POSITIVE_INFINITY;
        double measuredGemmGflops = Double.NaN;
        try {
            measuredGemmGflops = measureGemmAnchor();
            if (Double.isFinite(measuredGemmGflops) && measuredGemmGflops > 0.0) {
                anchoredPeak = measuredGemmGflops * 1e9 * anchorScale;
            }
        } catch (Exception ignored) {
            measuredGemmGflops = Double.NaN;
        }

        double roof = Math.min(theoreticalCap, anchoredPeak);
        if (!Double.isFinite(roof) || roof <= 0.0) {
            roof = Math.max(1e9, theoreticalCap);
        }

        String source = String.format(
            Locale.ROOT,
            "min(theoretical*%.2f, gemm_anchor*%.2f)",
            theoreticalUtil,
            anchorScale
        );
        return new ComputeEstimate(roof, source, hardware.peakFlopsPerSecond, measuredGemmGflops);
    }

    private static double measureGemmAnchor() {
        int n = parsePositiveInt(System.getProperty(GEMM_ANCHOR_N_PROPERTY), DEFAULT_GEMM_ANCHOR_N);
        int warmup = parsePositiveInt(System.getProperty(GEMM_ANCHOR_WARMUP_PROPERTY), DEFAULT_GEMM_ANCHOR_WARMUP);
        int runs = parsePositiveInt(System.getProperty(GEMM_ANCHOR_RUNS_PROPERTY), DEFAULT_GEMM_ANCHOR_RUNS);

        Matrix a = randomMatrix(n, n, 7101L);
        Matrix b = randomMatrix(n, n, 7102L);
        Matrix c = new Matrix(n, n);
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(true)
            .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
            .build();

        for (int i = 0; i < warmup; i++) {
            OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, policy);
        }

        double bestSeconds = Double.POSITIVE_INFINITY;
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, policy);
            long elapsed = System.nanoTime() - start;
            double seconds = elapsed / 1e9;
            if (seconds < bestSeconds) {
                bestSeconds = seconds;
            }
            sink += c.get(0, 0);
        }

        double flops = 2.0 * n * (double) n * n;
        return flops / Math.max(1e-9, bestSeconds) / 1e9;
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static int parsePositiveInt(String value, int fallback) {
        if (value == null) {
            return fallback;
        }
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static double parsePositiveDouble(String value, double fallback) {
        Double parsed = parsePositiveDouble(value);
        return parsed == null ? fallback : parsed;
    }

    private static Double parsePositiveDouble(String value) {
        if (value == null) {
            return null;
        }
        try {
            double parsed = Double.parseDouble(value.trim());
            return parsed > 0.0 ? parsed : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    static final class ComputeEstimate {
        final double bytesFlopsPerSecond;
        final String source;
        final double rawTheoreticalPeak;
        final double measuredGemmAnchorGflops;

        ComputeEstimate(double bytesFlopsPerSecond,
                        String source,
                        double rawTheoreticalPeak,
                        double measuredGemmAnchorGflops) {
            this.bytesFlopsPerSecond = bytesFlopsPerSecond;
            this.source = source;
            this.rawTheoreticalPeak = rawTheoreticalPeak;
            this.measuredGemmAnchorGflops = measuredGemmAnchorGflops;
        }
    }
}
