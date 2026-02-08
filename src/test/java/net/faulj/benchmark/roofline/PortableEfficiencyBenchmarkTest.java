package net.faulj.benchmark.roofline;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.OptimizedBLAS3;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Collectors;

public class PortableEfficiencyBenchmarkTest {
    private static final Path OUTPUT_DIR = Paths.get("build", "reports", "roofline");
    private static final Path CSV_OUTPUT = OUTPUT_DIR.resolve("portable_efficiency_results.csv");
    private static final Path MAX_CSV_OUTPUT = OUTPUT_DIR.resolve("portable_efficiency_maxima.csv");
    private static final Path JSON_OUTPUT = OUTPUT_DIR.resolve("portable_efficiency_results.json");
    private static final int[] SWEEP_SIZES = {64, 128, 256, 512, 1024, 2048, 4096};

    private static volatile double sink;

    @Test
    public void runPortableEfficiencyBenchmarks() throws IOException {
//        Assume.assumeTrue(
//            "Benchmark mode is disabled. Enable with -D" + BenchmarkMode.MODE_PROPERTY + "=true",
//            BenchmarkMode.isEnabled()
//        );

        RooflineSession baseRoofline = RooflineSession.get();
        List<PesResult> results = new ArrayList<>();
        List<KernelSweepSummary> maxima = new ArrayList<>();

        List<PesResult> gemmRaw = new ArrayList<>();
        List<Integer> gemmSizes = new ArrayList<>();
        sweepKernel("GEMM", SWEEP_SIZES, 4096, baseRoofline, PortableEfficiencyBenchmarkTest::runGemm, gemmRaw, gemmSizes);

        double effectiveComputeRoof = deriveEffectiveComputeRoof(gemmRaw);
        RooflineSession roofline = baseRoofline.withComputeRoof(
            effectiveComputeRoof,
            "gemm_sweep_max*1.05"
        );

        List<PesResult> gemmRescored = rescoreResults(gemmRaw, roofline);
        results.addAll(gemmRescored);
        maxima.add(summarizeKernel("GEMM", gemmRescored, gemmSizes));

        maxima.add(sweepKernel("QR", SWEEP_SIZES, 4096, roofline, PortableEfficiencyBenchmarkTest::runQr, results, new ArrayList<>()));
        maxima.add(sweepKernel("LU", SWEEP_SIZES, 4096, roofline, PortableEfficiencyBenchmarkTest::runLu, results, new ArrayList<>()));
        maxima.add(sweepKernel("Hessenberg", SWEEP_SIZES, 2048, roofline, PortableEfficiencyBenchmarkTest::runHessenberg, results, new ArrayList<>()));
        maxima.add(sweepKernel("Schur", SWEEP_SIZES, 512, roofline, PortableEfficiencyBenchmarkTest::runSchur, results, new ArrayList<>()));
        maxima.add(sweepKernel("SVD", SWEEP_SIZES, 512, roofline, PortableEfficiencyBenchmarkTest::runSvd, results, new ArrayList<>()));

        Files.createDirectories(OUTPUT_DIR);
        writeCsv(results);
        writeMaxCsv(maxima);
        writeJson(roofline, results, maxima);
        printSummary(roofline, results, maxima);
    }

    private static PesResult runGemm(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 1001L);
        Matrix b = randomSquareMatrix(n, 1002L);
        Matrix c = new Matrix(n, n);
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(true)
            .parallelism(Math.max(1, Runtime.getRuntime().availableProcessors()))
            .build();

        double bestSeconds = bestOf(warmupForSize(n), measuredRunsForSize(n), () -> {
            OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, policy);
            sink += c.get(0, 0);
        });

        return PesScorer.score(KernelModel.gemm(n), bestSeconds, roofline);
    }

    private static PesResult runQr(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 2001L);
        double bestSeconds = bestOf(warmupForSize(n), measuredRunsForSize(n), () -> {
            HouseholderQR.factorize(a);
            sink += a.get(0, 0);
        });
        return PesScorer.score(KernelModel.qr(n), bestSeconds, roofline);
    }

    private static PesResult runLu(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 3001L);
        LUDecomposition lu = new LUDecomposition();
        double bestSeconds = bestOf(warmupForSize(n), measuredRunsForSize(n), () -> {
            sink += lu.decompose(a).getU().get(0, 0);
        });
        return PesScorer.score(KernelModel.lu(n), bestSeconds, roofline);
    }

    private static PesResult runHessenberg(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 4001L);
        double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
            sink += HessenbergReduction.decompose(a).getH().get(0, 0);
        });
        return PesScorer.score(KernelModel.hessenberg(n), bestSeconds, roofline);
    }

    private static PesResult runSchur(int n, RooflineSession roofline) {
        RuntimeException last = null;
        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                Matrix a = randomSquareMatrix(n, 5001L + attempt);
                double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
                    sink += RealSchurDecomposition.decompose(a).getT().get(0, 0);
                });
                return PesScorer.score(KernelModel.schur(n), bestSeconds, roofline);
            } catch (RuntimeException ex) {
                last = ex;
            }
        }
        System.out.println("Schur benchmark skipped due convergence failure: " + (last == null ? "unknown" : last.getMessage()));
        return PesScorer.score(KernelModel.schur(n), Double.POSITIVE_INFINITY, roofline);
    }

    private static PesResult runSvd(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 6001L);
        SVDecomposition svd = new SVDecomposition();
        double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
            double[] singularValues = svd.decompose(a).getSingularValues();
            sink += singularValues.length == 0 ? 0.0 : singularValues[0];
        });
        return PesScorer.score(KernelModel.svd(n), bestSeconds, roofline);
    }

    private static Matrix randomSquareMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[n * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, n, n);
    }

    private static double bestOf(int warmupRuns, int measuredRuns, Runnable task) {
        for (int i = 0; i < warmupRuns; i++) {
            task.run();
        }
        long bestNanos = Long.MAX_VALUE;
        for (int i = 0; i < measuredRuns; i++) {
            long start = System.nanoTime();
            task.run();
            long elapsed = System.nanoTime() - start;
            if (elapsed < bestNanos) {
                bestNanos = elapsed;
            }
        }
        return bestNanos / 1e9;
    }

    private static int warmupForSize(int n) {
        if (n >= 4096) {
            return 0;
        }
        if (n >= 2048) {
            return 1;
        }
        return 2;
    }

    private static int measuredRunsForSize(int n) {
        if (n >= 4096) {
            return 1;
        }
        if (n >= 2048) {
            return 2;
        }
        return 4;
    }

    private static KernelSweepSummary sweepKernel(String kernel,
                                                  int[] sizes,
                                                  int maxEnabledSize,
                                                  RooflineSession roofline,
                                                  KernelRunner runner,
                                                  List<PesResult> sinkResults,
                                                  List<Integer> tested) {
        double bestPes = Double.NEGATIVE_INFINITY;
        int bestN = -1;

        for (int n : sizes) {
            if (n > maxEnabledSize) {
                continue;
            }
            try {
                PesResult result = runner.run(n, roofline);
                sinkResults.add(result);
                tested.add(n);
                if (result.portableEfficiencyScore > bestPes) {
                    bestPes = result.portableEfficiencyScore;
                    bestN = n;
                }
            } catch (Throwable ex) {
                System.out.println("Skipping " + kernel + " n=" + n + " due to: " + ex.getMessage());
            }
        }

        if (bestN < 0) {
            return new KernelSweepSummary(kernel, 0.0, -1, tested);
        }
        return new KernelSweepSummary(kernel, bestPes, bestN, tested);
    }

    private static double deriveEffectiveComputeRoof(List<PesResult> gemmResults) {
        double maxMeasuredGflops = 0.0;
        for (PesResult result : gemmResults) {
            if (result.measuredGflops > maxMeasuredGflops) {
                maxMeasuredGflops = result.measuredGflops;
            }
        }
        return Math.max(1e9, maxMeasuredGflops * 1.05 * 1e9);
    }

    private static List<PesResult> rescoreResults(List<PesResult> results, RooflineSession roofline) {
        List<PesResult> rescored = new ArrayList<>(results.size());
        for (PesResult result : results) {
            KernelProfile profile = profileFor(result.kernel, result.n);
            rescored.add(PesScorer.score(profile, result.elapsedSeconds, roofline));
        }
        return rescored;
    }

    private static KernelProfile profileFor(String kernel, int n) {
        return switch (kernel) {
            case "GEMM" -> KernelModel.gemm(n);
            case "QR" -> KernelModel.qr(n);
            case "LU" -> KernelModel.lu(n);
            case "Hessenberg" -> KernelModel.hessenberg(n);
            case "Schur" -> KernelModel.schur(n);
            case "SVD" -> KernelModel.svd(n);
            default -> throw new IllegalArgumentException("Unknown kernel: " + kernel);
        };
    }

    private static KernelSweepSummary summarizeKernel(String kernel, List<PesResult> results, List<Integer> testedSizes) {
        double bestPes = Double.NEGATIVE_INFINITY;
        int bestN = -1;
        for (PesResult result : results) {
            if (!kernel.equals(result.kernel)) {
                continue;
            }
            if (result.portableEfficiencyScore > bestPes) {
                bestPes = result.portableEfficiencyScore;
                bestN = result.n;
            }
        }
        if (bestN < 0) {
            return new KernelSweepSummary(kernel, 0.0, -1, testedSizes);
        }
        return new KernelSweepSummary(kernel, bestPes, bestN, testedSizes);
    }

    private static void writeCsv(List<PesResult> results) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("kernel,n,arithmetic_intensity,bound_type,memory_level,compute_utilization,memory_utilization,algorithmic_efficiency,portable_efficiency_score,measured_gflops,roof_gflops,memory_roof_gbps,elapsed_seconds");
        sb.append('\n');
        for (PesResult r : results) {
            sb.append(r.kernel).append(',')
                .append(r.n).append(',')
                .append(format(r.arithmeticIntensity)).append(',')
                .append(r.boundType).append(',')
                .append(r.memoryLevel).append(',')
                .append(format(r.computeUtilization)).append(',')
                .append(format(r.memoryUtilization)).append(',')
                .append(format(r.algorithmicEfficiency)).append(',')
                .append(format(r.portableEfficiencyScore)).append(',')
                .append(format(r.measuredGflops)).append(',')
                .append(format(r.roofGflops)).append(',')
                .append(format(r.selectedMemoryRoofGbps)).append(',')
                .append(format(r.elapsedSeconds))
                .append('\n');
        }
        Files.writeString(CSV_OUTPUT, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeMaxCsv(List<KernelSweepSummary> maxima) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("kernel,pes_max,n_at_pes_max,tested_sizes");
        sb.append('\n');
        for (KernelSweepSummary summary : maxima) {
            String tested = summary.testedSizes.stream().map(String::valueOf).collect(Collectors.joining("|"));
            sb.append(summary.kernel).append(',')
                .append(format(summary.pesMax)).append(',')
                .append(summary.nAtPesMax).append(',')
                .append(tested)
                .append('\n');
        }
        Files.writeString(MAX_CSV_OUTPUT, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeJson(RooflineSession roofline, List<PesResult> results, List<KernelSweepSummary> maxima) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"compute_roof_gflops\": ").append(format(roofline.peakFlopsPerSecond / 1e9)).append(",\n");
        sb.append("  \"compute_theoretical_peak_gflops\": ").append(format(roofline.rawTheoreticalPeakFlopsPerSecond / 1e9)).append(",\n");
        sb.append("  \"compute_gemm_anchor_gflops\": ").append(format(roofline.measuredGemmAnchorGflops)).append(",\n");
        sb.append("  \"memory_roof_l1_gbps\": ").append(format(roofline.memoryBandwidths.l1BytesPerSecond / 1e9)).append(",\n");
        sb.append("  \"memory_roof_l2_gbps\": ").append(format(roofline.memoryBandwidths.l2BytesPerSecond / 1e9)).append(",\n");
        sb.append("  \"memory_roof_l3_gbps\": ").append(format(roofline.memoryBandwidths.l3BytesPerSecond / 1e9)).append(",\n");
        sb.append("  \"memory_roof_dram_gbps\": ").append(format(roofline.memoryBandwidths.dramBytesPerSecond / 1e9)).append(",\n");
        sb.append("  \"compute_roof_source\": \"")
            .append(escape(roofline.computeRoofSource))
            .append("\",\n");
        sb.append("  \"memory_roof_source\": \"")
            .append(escape(roofline.memoryRoofSource))
            .append("\",\n");
        sb.append("  \"results\": [\n");
        for (int i = 0; i < results.size(); i++) {
            PesResult r = results.get(i);
            sb.append("    {\n");
            sb.append("      \"kernel\": \"").append(escape(r.kernel)).append("\",\n");
            sb.append("      \"n\": ").append(r.n).append(",\n");
            sb.append("      \"arithmetic_intensity\": ").append(format(r.arithmeticIntensity)).append(",\n");
            sb.append("      \"bound_type\": \"").append(r.boundType).append("\",\n");
            sb.append("      \"memory_level\": \"").append(r.memoryLevel).append("\",\n");
            sb.append("      \"compute_utilization\": ").append(format(r.computeUtilization)).append(",\n");
            sb.append("      \"memory_utilization\": ").append(format(r.memoryUtilization)).append(",\n");
            sb.append("      \"algorithmic_efficiency\": ").append(format(r.algorithmicEfficiency)).append(",\n");
            sb.append("      \"portable_efficiency_score\": ").append(format(r.portableEfficiencyScore)).append(",\n");
            sb.append("      \"memory_roof_gbps\": ").append(format(r.selectedMemoryRoofGbps)).append('\n');
            sb.append("    }");
            if (i < results.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ],\n");
        sb.append("  \"maxima\": [\n");
        for (int i = 0; i < maxima.size(); i++) {
            KernelSweepSummary summary = maxima.get(i);
            sb.append("    {\n");
            sb.append("      \"kernel\": \"").append(escape(summary.kernel)).append("\",\n");
            sb.append("      \"pes_max\": ").append(format(summary.pesMax)).append(",\n");
            sb.append("      \"n_at_pes_max\": ").append(summary.nAtPesMax).append('\n');
            sb.append("    }");
            if (i < maxima.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ]\n");
        sb.append("}\n");
        Files.writeString(JSON_OUTPUT, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void printSummary(RooflineSession roofline, List<PesResult> results, List<KernelSweepSummary> maxima) {
        System.out.println("=== Portable Efficiency Score (PES) ===");
        System.out.printf(Locale.ROOT, "compute_roof_gflops=%.3f%n", roofline.peakFlopsPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "compute_theoretical_peak_gflops=%.3f%n", roofline.rawTheoreticalPeakFlopsPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "compute_gemm_anchor_gflops=%.3f%n", roofline.measuredGemmAnchorGflops);
        System.out.printf(Locale.ROOT, "memory_roof_l1_gbps=%.3f%n", roofline.memoryBandwidths.l1BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_l2_gbps=%.3f%n", roofline.memoryBandwidths.l2BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_l3_gbps=%.3f%n", roofline.memoryBandwidths.l3BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_dram_gbps=%.3f%n", roofline.memoryBandwidths.dramBytesPerSecond / 1e9);
        System.out.println("compute_roof_source=" + roofline.computeRoofSource);
        System.out.println("memory_roof_source=" + roofline.memoryRoofSource);
        for (PesResult r : results) {
            System.out.printf(
                Locale.ROOT,
                "%s n=%d ai=%.4f bound=%s level=%s pes=%.4f compute_util=%.4f memory_util=%.4f mem_roof=%.2fGB/s%n",
                r.kernel, r.n, r.arithmeticIntensity, r.boundType,
                r.memoryLevel, r.portableEfficiencyScore, r.computeUtilization, r.memoryUtilization,
                r.selectedMemoryRoofGbps
            );
        }
        System.out.println("=== PES Maxima ===");
        for (KernelSweepSummary summary : maxima) {
            if (summary.nAtPesMax > 0) {
                System.out.printf(Locale.ROOT, "%s PES_max=%.4f at n=%d%n",
                    summary.kernel, summary.pesMax, summary.nAtPesMax);
            } else {
                System.out.println(summary.kernel + " PES_max unavailable (no successful runs)");
            }
        }
        System.out.println("CSV=" + CSV_OUTPUT);
        System.out.println("MAX_CSV=" + MAX_CSV_OUTPUT);
        System.out.println("JSON=" + JSON_OUTPUT);
    }

    private static String format(double value) {
        if (!Double.isFinite(value)) {
            return "0.000000";
        }
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    @FunctionalInterface
    private interface KernelRunner {
        PesResult run(int n, RooflineSession roofline);
    }

    private static final class KernelSweepSummary {
        final String kernel;
        final double pesMax;
        final int nAtPesMax;
        final List<Integer> testedSizes;

        KernelSweepSummary(String kernel, double pesMax, int nAtPesMax, List<Integer> testedSizes) {
            this.kernel = kernel;
            this.pesMax = pesMax;
            this.nAtPesMax = nAtPesMax;
            this.testedSizes = testedSizes;
        }
    }
}
