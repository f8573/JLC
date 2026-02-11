package net.faulj.benchmark.roofline;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.GemmDispatch;
import net.faulj.kernels.gemm.Gemm;
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
    // Logarithmic sweep: 2^i and (2^i + 2^{i+1})/2 for coverage of cache transitions
    private static final int[] SWEEP_SIZES = {
        2, 3, 4, 6, 8, 12, 16, 24, 32, 48,
        64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
    };

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
        sweepKernel("GEMM", SWEEP_SIZES, 1024, baseRoofline, PortableEfficiencyBenchmarkTest::runGemm, gemmRaw, gemmSizes);

        double effectiveComputeRoof = deriveEffectiveComputeRoof(gemmRaw);
        RooflineSession roofline = baseRoofline.withComputeRoof(
            effectiveComputeRoof,
            "gemm_sweep_max*1.05"
        );

        List<PesResult> gemmRescored = rescoreResults(gemmRaw, roofline);
        results.addAll(gemmRescored);
        maxima.add(summarizeKernel("GEMM", gemmRescored, gemmSizes));

        maxima.add(sweepKernel("QR", SWEEP_SIZES, 2048, roofline, PortableEfficiencyBenchmarkTest::runQr, results, new ArrayList<>()));
        maxima.add(sweepKernel("LU", SWEEP_SIZES, 2048, roofline, PortableEfficiencyBenchmarkTest::runLu, results, new ArrayList<>()));
        maxima.add(sweepKernel("Hessenberg", SWEEP_SIZES, 1024, roofline, PortableEfficiencyBenchmarkTest::runHessenberg, results, new ArrayList<>()));
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
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            sink += c.get(0, 0);
        });

        // Use BLIS blocking-aware traffic model when the kernel is large enough to block.
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        KernelProfile profile = KernelModel.gemm(n, n, n, blocks.mc, blocks.nc, blocks.kc);
        return PesScorer.score(profile, bestSeconds, roofline);
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
            case "GEMM" -> {
                GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
                yield KernelModel.gemm(n, n, n, blocks.mc, blocks.nc, blocks.kc);
            }
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
        sb.append("kernel,m,n,k,arithmetic_intensity,traffic_model,bound_type,memory_level,")
            .append("compute_utilization,memory_utilization,algorithmic_efficiency,")
            .append("portable_efficiency_score,pes_l1,pes_l2,pes_l3,pes_dram,")
            .append("measured_gflops,compute_roof_gflops,roof_gflops,memory_roof_gbps,")
            .append("elapsed_seconds,confidence,flag");
        sb.append('\n');
        for (PesResult r : results) {
            sb.append(r.kernel).append(',')
                .append(r.m).append(',')
                .append(r.n).append(',')
                .append(r.k).append(',')
                .append(format(r.arithmeticIntensity)).append(',')
                .append(r.trafficModel).append(',')
                .append(r.boundType).append(',')
                .append(r.memoryLevel).append(',')
                .append(format(r.computeUtilization)).append(',')
                .append(format(r.memoryUtilization)).append(',')
                .append(format(r.algorithmicEfficiency)).append(',')
                .append(format(r.portableEfficiencyScore)).append(',')
                .append(format(r.pesL1)).append(',')
                .append(format(r.pesL2)).append(',')
                .append(format(r.pesL3)).append(',')
                .append(format(r.pesDram)).append(',')
                .append(format(r.measuredGflops)).append(',')
                .append(format(r.computeRoofGflops)).append(',')
                .append(format(r.roofGflops)).append(',')
                .append(format(r.selectedMemoryRoofGbps)).append(',')
                .append(format(r.elapsedSeconds)).append(',')
                .append(r.confidence).append(',')
                .append(r.flag)
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
        sb.append("  \"capability_tier\": \"").append(escape(roofline.hardware.tier.name())).append("\",\n");
        sb.append("  \"capability_tier_description\": \"").append(escape(roofline.hardware.tier.description)).append("\",\n");
        sb.append("  \"compute_roof_gflops\": ").append(format(roofline.peakFlopsPerSecond / 1e9)).append(",\n");
        sb.append("  \"compute_theoretical_peak_gflops\": ").append(format(roofline.rawTheoreticalPeakFlopsPerSecond / 1e9)).append(",\n");
        sb.append("  \"compute_gemm_anchor_gflops\": ").append(format(roofline.measuredGemmAnchorGflops)).append(",\n");
        sb.append("  \"hardware\": {\n");
        sb.append("    \"cores\": ").append(roofline.hardware.cores).append(",\n");
        sb.append("    \"clock_ghz\": ").append(format(roofline.hardware.clockGhz)).append(",\n");
        sb.append("    \"simd_lanes_double\": ").append(roofline.hardware.simdLanesDouble).append(",\n");
        sb.append("    \"fma_enabled\": ").append(roofline.hardware.fmaEnabled).append(",\n");
        sb.append("    \"vector_issue_width\": ").append(roofline.hardware.vectorIssueWidth).append(",\n");
        sb.append("    \"clock_source\": \"").append(escape(roofline.hardware.clockSource)).append("\"\n");
        sb.append("  },\n");
        sb.append("  \"assumptions\": [");
        for (int i = 0; i < roofline.hardware.assumptions.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append("\n    \"").append(escape(roofline.hardware.assumptions.get(i))).append('"');
        }
        if (!roofline.hardware.assumptions.isEmpty()) sb.append('\n');
        sb.append("  ],\n");
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
            sb.append("      \"m\": ").append(r.m).append(",\n");
            sb.append("      \"n\": ").append(r.n).append(",\n");
            sb.append("      \"k\": ").append(r.k).append(",\n");
            sb.append("      \"arithmetic_intensity\": ").append(format(r.arithmeticIntensity)).append(",\n");
            sb.append("      \"traffic_model\": \"").append(r.trafficModel).append("\",\n");
            sb.append("      \"bound_type\": \"").append(r.boundType).append("\",\n");
            sb.append("      \"memory_level\": \"").append(r.memoryLevel).append("\",\n");
            sb.append("      \"compute_utilization\": ").append(format(r.computeUtilization)).append(",\n");
            sb.append("      \"memory_utilization\": ").append(format(r.memoryUtilization)).append(",\n");
            sb.append("      \"algorithmic_efficiency\": ").append(format(r.algorithmicEfficiency)).append(",\n");
            sb.append("      \"portable_efficiency_score\": ").append(format(r.portableEfficiencyScore)).append(",\n");
            sb.append("      \"pes_l1\": ").append(format(r.pesL1)).append(",\n");
            sb.append("      \"pes_l2\": ").append(format(r.pesL2)).append(",\n");
            sb.append("      \"pes_l3\": ").append(format(r.pesL3)).append(",\n");
            sb.append("      \"pes_dram\": ").append(format(r.pesDram)).append(",\n");
            sb.append("      \"measured_gflops\": ").append(format(r.measuredGflops)).append(",\n");
            sb.append("      \"compute_roof_gflops\": ").append(format(r.computeRoofGflops)).append(",\n");
            sb.append("      \"roof_gflops\": ").append(format(r.roofGflops)).append(",\n");
            sb.append("      \"memory_roof_gbps\": ").append(format(r.selectedMemoryRoofGbps)).append(",\n");
            sb.append("      \"confidence\": \"").append(r.confidence).append("\",\n");
            sb.append("      \"flag\": \"").append(r.flag).append("\"\n");
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
        System.out.printf(Locale.ROOT, "capability_tier=%s%n", roofline.hardware.tier.description);
        System.out.printf(Locale.ROOT, "hardware=%d cores, %.2f GHz, %d SIMD lanes, FMA=%s, issue_width=%d%n",
            roofline.hardware.cores, roofline.hardware.clockGhz, roofline.hardware.simdLanesDouble,
            roofline.hardware.fmaEnabled, roofline.hardware.vectorIssueWidth);
        System.out.printf(Locale.ROOT, "compute_roof_gflops=%.3f%n", roofline.peakFlopsPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "compute_theoretical_peak_gflops=%.3f%n", roofline.rawTheoreticalPeakFlopsPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "compute_gemm_anchor_gflops=%.3f%n", roofline.measuredGemmAnchorGflops);
        System.out.printf(Locale.ROOT, "memory_roof_l1_gbps=%.3f%n", roofline.memoryBandwidths.l1BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_l2_gbps=%.3f%n", roofline.memoryBandwidths.l2BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_l3_gbps=%.3f%n", roofline.memoryBandwidths.l3BytesPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_dram_gbps=%.3f%n", roofline.memoryBandwidths.dramBytesPerSecond / 1e9);
        System.out.println("compute_roof_source=" + roofline.computeRoofSource);
        System.out.println("memory_roof_source=" + roofline.memoryRoofSource);
        if (!roofline.hardware.assumptions.isEmpty()) {
            System.out.println("assumptions:");
            for (String assumption : roofline.hardware.assumptions) {
                System.out.println("  - " + assumption);
            }
        }
        System.out.println("--- Per-kernel results ---");
        for (PesResult r : results) {
            System.out.printf(
                Locale.ROOT,
                "%s m=%d n=%d k=%d ai=%.4f model=%s bound=%s level=%s pes=%.4f " +
                    "compute_util=%.4f memory_util=%.4f mem_roof=%.2fGB/s conf=%s flag=%s%n",
                r.kernel, r.m, r.n, r.k, r.arithmeticIntensity, r.trafficModel,
                r.boundType, r.memoryLevel, r.portableEfficiencyScore,
                r.computeUtilization, r.memoryUtilization,
                r.selectedMemoryRoofGbps, r.confidence, r.flag
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
