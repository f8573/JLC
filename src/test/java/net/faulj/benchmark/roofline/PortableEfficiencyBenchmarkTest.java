package net.faulj.benchmark.roofline;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.GemmDispatch;
import net.faulj.decomposition.bidiagonal.Bidiagonalization;
import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.polar.PolarDecomposition;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.svd.SVDecomposition;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeGemmProfile;
import net.faulj.nativeblas.NativeProfiling;
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
import java.util.TreeSet;
import java.util.stream.Collectors;

public class PortableEfficiencyBenchmarkTest {
    private static final Path DEFAULT_OUTPUT_DIR = Paths.get("build", "reports", "roofline");
    private static final String OUTPUT_DIR_PROPERTY = "jlc.roofline.outputDir";
    private static final String OUTPUT_DIR_PROPERTY_ALT = "jlc.roofline.output_dir";
    private static final String OUTPUT_DIR_ENV = "JLC_ROOFLINE_OUTPUT_DIR";

    private static final String RESULTS_CSV = "portable_efficiency_results.csv";
    private static final String MAXIMA_CSV = "portable_efficiency_maxima.csv";
    private static final String RESULTS_JSON = "portable_efficiency_results.json";
    private static final String GEMM_RUNS_CSV = "portable_efficiency_gemm_runs.csv";
    private static final String GEMM_BEST_BY_SIZE_CSV = "portable_efficiency_gemm_best_by_size.csv";

    private static final String GEMM_MAX_N_PROPERTY = "jlc.roofline.gemm_max_n";
    private static final String GEMM_THREADS_PROPERTY = "jlc.roofline.gemm_threads";
    private static final String GEMM_MR_VALUES_PROPERTY = "jlc.roofline.gemm_mr_values";
    private static final String GEMM_NR_VALUES_PROPERTY = "jlc.roofline.gemm_nr_values";
    private static final String SIZES_PROPERTY = "jlc.roofline.sizes";

    private static final int DEFAULT_GEMM_MAX_N = 2048;
    // Sweep sizes from 128 up to 2048: powers-of-two and midpoints
    // (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
    private static final int[] SWEEP_SIZES = {
        128, 192, 256, 384, 512, 768, 1024, 1536, 2048
    };

    private static volatile double sink;

    @Test
    public void runPortableEfficiencyBenchmarks() throws IOException {
        BenchmarkOutputs outputs = BenchmarkOutputs.fromSystemProperties();
        RooflineSession roofline = RooflineSession.get();
        int[] sweepSizes = sweepSizes();
        List<PesResult> results = new ArrayList<>();
        List<KernelSweepSummary> maxima = new ArrayList<>();

        GemmSweepArtifacts gemmArtifacts = sweepBestGemmBySize(sweepSizes, gemmMaxEnabled(), roofline);
        results.addAll(gemmArtifacts.bestResults);
        maxima.add(summarizeKernel("GEMM", gemmArtifacts.bestResults, gemmArtifacts.testedSizes));

        if (!gemmOnlyMode()) {
            maxima.add(sweepKernel("QR", sweepSizes, 2048, roofline, PortableEfficiencyBenchmarkTest::runQr, results, new ArrayList<>()));
            maxima.add(sweepKernel("LU", sweepSizes, 2048, roofline, PortableEfficiencyBenchmarkTest::runLu, results, new ArrayList<>()));
            maxima.add(sweepKernel("Hessenberg", sweepSizes, 1024, roofline, PortableEfficiencyBenchmarkTest::runHessenberg, results, new ArrayList<>()));
            maxima.add(sweepKernel("Bidiagonal", sweepSizes, 1024, roofline, PortableEfficiencyBenchmarkTest::runBidiagonal, results, new ArrayList<>()));
            maxima.add(sweepKernel("Cholesky", sweepSizes, 1024, roofline, PortableEfficiencyBenchmarkTest::runCholesky, results, new ArrayList<>()));
            maxima.add(sweepKernel("Schur", sweepSizes, 512, roofline, PortableEfficiencyBenchmarkTest::runSchur, results, new ArrayList<>()));
            maxima.add(sweepKernel("SVD", sweepSizes, 512, roofline, PortableEfficiencyBenchmarkTest::runSvd, results, new ArrayList<>()));
            maxima.add(sweepKernel("Polar", sweepSizes, 512, roofline, PortableEfficiencyBenchmarkTest::runPolar, results, new ArrayList<>()));
        }

        Files.createDirectories(outputs.outputDir);
        writeCsv(outputs.csvOutput, results);
        writeMaxCsv(outputs.maxCsvOutput, maxima);
        writeGemmRunsCsv(outputs.gemmRunsCsvOutput, gemmArtifacts.allRuns);
        writeGemmBestBySizeCsv(outputs.gemmBestBySizeOutput, gemmArtifacts.bestRuns);
        writeJson(outputs.jsonOutput, roofline, results, maxima, gemmArtifacts);
        printSummary(outputs, roofline, results, maxima, gemmArtifacts.bestRuns);
    }

    private static GemmSweepArtifacts sweepBestGemmBySize(int[] sizes, int maxEnabledSize, RooflineSession roofline) {
        GemmDispatch.BlockSizes defaults = GemmDispatch.computeBlockSizes();
        GemmBenchmarkConfig config = GemmBenchmarkConfig.fromSystemProperties(defaults);

        List<GemmRunResult> allRuns = new ArrayList<>();
        List<GemmRunResult> bestRuns = new ArrayList<>();
        List<PesResult> bestResults = new ArrayList<>();
        List<Integer> testedSizes = new ArrayList<>();

        for (int n : sizes) {
            if (n > maxEnabledSize) {
                continue;
            }

            Matrix a = randomSquareMatrix(n, 1001L + n);
            Matrix b = randomSquareMatrix(n, 2001L + n);
            Matrix c = new Matrix(n, n);

            GemmRunResult best = null;
            for (int threads : config.threadCandidates) {
                for (int mr : config.mrCandidates) {
                    for (int nr : config.nrCandidates) {
                        GemmRunResult run = runGemmCandidate(n, a, b, c, threads, mr, nr, roofline);
                        allRuns.add(run);
                        if (isBetterGemmRun(run, best)) {
                            best = run;
                        }
                    }
                }
            }

            if (best != null) {
                testedSizes.add(n);
                bestRuns.add(best);
                bestResults.add(best.result);
            }
        }

        return new GemmSweepArtifacts(config, allRuns, bestRuns, bestResults, testedSizes);
    }

    private static GemmRunResult runGemmCandidate(int n,
                                                  Matrix a,
                                                  Matrix b,
                                                  Matrix c,
                                                  int threads,
                                                  int mr,
                                                  int nr,
                                                  RooflineSession roofline) {
        String oldMr = System.getProperty("la.gemm.mr");
        String oldNr = System.getProperty("la.gemm.nr");

        try {
            System.setProperty("la.gemm.mr", Integer.toString(mr));
            System.setProperty("la.gemm.nr", Integer.toString(nr));

            DispatchPolicy policy = DispatchPolicy.builder()
                .enableCuda(false)
                .enableParallel(threads > 1)
                .parallelism(threads)
                .enableStrassen(false)
                .build();

            long bestNanos = Long.MAX_VALUE;
            NativeGemmProfile bestNativeProfile = null;
            boolean profilingEnabled = NativeProfiling.setEnabled(true);
            try {
                for (int i = 0; i < warmupForSize(n); i++) {
                    Gemm.gemm(a, b, c, 1.0, 0.0, policy);
                    sink += c.get(0, 0);
                }
                for (int i = 0; i < measuredRunsForSize(n); i++) {
                    if (profilingEnabled) {
                        NativeProfiling.reset();
                    }
                    long start = System.nanoTime();
                    Gemm.gemm(a, b, c, 1.0, 0.0, policy);
                    long elapsed = System.nanoTime() - start;
                    sink += c.get(0, 0);
                    if (elapsed < bestNanos) {
                        bestNanos = elapsed;
                        bestNativeProfile = profilingEnabled ? NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY) : null;
                    }
                }
            } finally {
                if (profilingEnabled) {
                    NativeProfiling.setEnabled(false);
                    NativeProfiling.reset();
                }
            }

            double bestSeconds = bestNanos / 1e9;

            GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
            KernelProfile profile = KernelModel.gemm(n, n, n, blocks.mc, blocks.nc, blocks.kc);
            PesResult result = PesScorer.score(profile, bestSeconds, roofline);
            PesResult nativeCoreResult = null;
            double nativeCoreCoverage = 0.0;
            if (bestNativeProfile != null && bestNativeProfile.hasTimingData()) {
                double nativeCoreSeconds = bestNativeProfile.wallSeconds();
                if (nativeCoreSeconds > 0.0) {
                    nativeCoreResult = PesScorer.score(profile, nativeCoreSeconds, roofline);
                    nativeCoreCoverage = nativeCoreSeconds / Math.max(1e-12, bestSeconds);
                }
            }
            return new GemmRunResult(
                result, threads, blocks.mr, blocks.nr, blocks.mc, blocks.nc, blocks.kc,
                bestNativeProfile, nativeCoreResult, nativeCoreCoverage
            );
        } finally {
            restoreProperty("la.gemm.mr", oldMr);
            restoreProperty("la.gemm.nr", oldNr);
        }
    }

    private static boolean isBetterGemmRun(GemmRunResult candidate, GemmRunResult incumbent) {
        if (candidate == null) {
            return false;
        }
        if (incumbent == null) {
            return true;
        }

        int cmp = Double.compare(candidate.result.measuredGflops, incumbent.result.measuredGflops);
        if (cmp != 0) {
            return cmp > 0;
        }

        cmp = Double.compare(candidate.result.portableEfficiencyScore, incumbent.result.portableEfficiencyScore);
        if (cmp != 0) {
            return cmp > 0;
        }

        cmp = Double.compare(incumbent.result.elapsedSeconds, candidate.result.elapsedSeconds);
        if (cmp != 0) {
            return cmp > 0;
        }

        return candidate.threads < incumbent.threads;
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

    private static PesResult runBidiagonal(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 4501L);
        Bidiagonalization bidiagonalization = new Bidiagonalization();
        double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
            sink += bidiagonalization.decompose(a).getB().get(0, 0);
        });
        return PesScorer.score(KernelModel.bidiagonal(n), bestSeconds, roofline);
    }

    private static PesResult runCholesky(int n, RooflineSession roofline) {
        Matrix a = randomSpdMatrix(n, 4751L);
        CholeskyDecomposition cholesky = new CholeskyDecomposition();
        double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
            sink += cholesky.decompose(a).getL().get(0, 0);
        });
        return PesScorer.score(KernelModel.cholesky(n), bestSeconds, roofline);
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

    private static PesResult runPolar(int n, RooflineSession roofline) {
        Matrix a = randomSquareMatrix(n, 6501L);
        PolarDecomposition polar = new PolarDecomposition();
        double bestSeconds = bestOf(Math.min(1, warmupForSize(n)), Math.max(1, measuredRunsForSize(n) - 1), () -> {
            sink += polar.decompose(a).getP().get(0, 0);
        });
        return PesScorer.score(KernelModel.polar(n), bestSeconds, roofline);
    }

    private static Matrix randomSquareMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[n * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, n, n);
    }

    private static Matrix randomSpdMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[n * n];
        double[] rowSums = new double[n];
        for (int row = 0; row < n; row++) {
            for (int col = row + 1; col < n; col++) {
                double value = rnd.nextDouble() - 0.5;
                data[row * n + col] = value;
                data[col * n + row] = value;
                double abs = Math.abs(value);
                rowSums[row] += abs;
                rowSums[col] += abs;
            }
        }
        for (int i = 0; i < n; i++) {
            data[i * n + i] = rowSums[i] + 1.0;
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

    private static void writeCsv(Path output, List<PesResult> results) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("kernel,measurement_scope,m,n,k,arithmetic_intensity,traffic_model,bound_type,memory_level,")
            .append("compute_utilization,memory_utilization,algorithmic_efficiency,")
            .append("portable_efficiency_score,pes_l1,pes_l2,pes_l3,pes_dram,")
            .append("measured_gflops,compute_roof_gflops,roof_gflops,memory_roof_gbps,")
            .append("elapsed_seconds,confidence,flag");
        sb.append('\n');
        for (PesResult r : results) {
            sb.append(r.kernel).append(',')
                .append("end_to_end").append(',')
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
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeMaxCsv(Path output, List<KernelSweepSummary> maxima) throws IOException {
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
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeGemmRunsCsv(Path output, List<GemmRunResult> runs) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("m,n,k,threads,mr,nr,mc,nc,kc,measured_gflops,portable_efficiency_score,")
            .append("compute_roof_gflops,roof_gflops,bound_type,memory_level,elapsed_seconds,")
            .append("arithmetic_intensity,traffic_model,confidence,flag,measurement_scope,")
            .append("native_core_elapsed_seconds,native_core_measured_gflops,native_core_portable_efficiency_score,")
            .append("native_core_coverage_fraction,native_profile_calls,native_vendor_seconds,native_scale_c_seconds,")
            .append("native_pack_a_seconds,native_pack_b_seconds,native_kernel_seconds,native_thread_seconds");
        sb.append('\n');
        for (GemmRunResult run : runs) {
            appendGemmRunCsvLine(sb, run);
        }
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeGemmBestBySizeCsv(Path output, List<GemmRunResult> bestRuns) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("m,n,k,threads,mr,nr,mc,nc,kc,measured_gflops,portable_efficiency_score,")
            .append("compute_roof_gflops,roof_gflops,bound_type,memory_level,elapsed_seconds,")
            .append("arithmetic_intensity,traffic_model,confidence,flag,measurement_scope,")
            .append("native_core_elapsed_seconds,native_core_measured_gflops,native_core_portable_efficiency_score,")
            .append("native_core_coverage_fraction,native_profile_calls,native_vendor_seconds,native_scale_c_seconds,")
            .append("native_pack_a_seconds,native_pack_b_seconds,native_kernel_seconds,native_thread_seconds");
        sb.append('\n');
        for (GemmRunResult run : bestRuns) {
            appendGemmRunCsvLine(sb, run);
        }
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void appendGemmRunCsvLine(StringBuilder sb, GemmRunResult run) {
        PesResult r = run.result;
        sb.append(r.m).append(',')
            .append(r.n).append(',')
            .append(r.k).append(',')
            .append(run.threads).append(',')
            .append(run.mr).append(',')
            .append(run.nr).append(',')
            .append(run.mc).append(',')
            .append(run.nc).append(',')
            .append(run.kc).append(',')
            .append(format(r.measuredGflops)).append(',')
            .append(format(r.portableEfficiencyScore)).append(',')
            .append(format(r.computeRoofGflops)).append(',')
            .append(format(r.roofGflops)).append(',')
            .append(r.boundType).append(',')
            .append(r.memoryLevel).append(',')
            .append(format(r.elapsedSeconds)).append(',')
            .append(format(r.arithmeticIntensity)).append(',')
            .append(r.trafficModel).append(',')
            .append(r.confidence).append(',')
            .append(r.flag).append(',')
            .append("end_to_end").append(',')
            .append(format(run.nativeCoreSeconds())).append(',')
            .append(format(run.nativeCoreMeasuredGflops())).append(',')
            .append(format(run.nativeCorePes())).append(',')
            .append(format(run.nativeCoreCoverage)).append(',')
            .append(run.nativeProfileCallCount()).append(',')
            .append(format(run.nativeVendorSeconds())).append(',')
            .append(format(run.nativeScaleCSeconds())).append(',')
            .append(format(run.nativePackASeconds())).append(',')
            .append(format(run.nativePackBSeconds())).append(',')
            .append(format(run.nativeKernelSeconds())).append(',')
            .append(format(run.nativeThreadSeconds()))
            .append('\n');
    }

    private static void writeJson(Path output,
                                  RooflineSession roofline,
                                  List<PesResult> results,
                                  List<KernelSweepSummary> maxima,
                                  GemmSweepArtifacts gemmArtifacts) throws IOException {
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
        sb.append("  \"gemm_config\": {\n");
        sb.append("    \"thread_candidates\": [").append(intArrayJson(gemmArtifacts.config.threadCandidates)).append("],\n");
        sb.append("    \"mr_candidates\": [").append(intArrayJson(gemmArtifacts.config.mrCandidates)).append("],\n");
        sb.append("    \"nr_candidates\": [").append(intArrayJson(gemmArtifacts.config.nrCandidates)).append("]\n");
        sb.append("  },\n");
        sb.append("  \"gemm_best_by_size\": [\n");
        for (int i = 0; i < gemmArtifacts.bestRuns.size(); i++) {
            GemmRunResult run = gemmArtifacts.bestRuns.get(i);
            PesResult r = run.result;
            sb.append("    {\n");
            sb.append("      \"m\": ").append(r.m).append(",\n");
            sb.append("      \"n\": ").append(r.n).append(",\n");
            sb.append("      \"k\": ").append(r.k).append(",\n");
            sb.append("      \"threads\": ").append(run.threads).append(",\n");
            sb.append("      \"mr\": ").append(run.mr).append(",\n");
            sb.append("      \"nr\": ").append(run.nr).append(",\n");
            sb.append("      \"mc\": ").append(run.mc).append(",\n");
            sb.append("      \"nc\": ").append(run.nc).append(",\n");
            sb.append("      \"kc\": ").append(run.kc).append(",\n");
            sb.append("      \"measurement_scope\": \"end_to_end\",\n");
            sb.append("      \"measured_gflops\": ").append(format(r.measuredGflops)).append(",\n");
            sb.append("      \"portable_efficiency_score\": ").append(format(r.portableEfficiencyScore)).append(",\n");
            sb.append("      \"native_core_elapsed_seconds\": ").append(format(run.nativeCoreSeconds())).append(",\n");
            sb.append("      \"native_core_measured_gflops\": ").append(format(run.nativeCoreMeasuredGflops())).append(",\n");
            sb.append("      \"native_core_portable_efficiency_score\": ").append(format(run.nativeCorePes())).append(",\n");
            sb.append("      \"native_core_coverage_fraction\": ").append(format(run.nativeCoreCoverage)).append(",\n");
            sb.append("      \"roof_gflops\": ").append(format(r.roofGflops)).append(",\n");
            sb.append("      \"bound_type\": \"").append(escape(r.boundType)).append("\",\n");
            sb.append("      \"memory_level\": \"").append(escape(r.memoryLevel)).append("\"\n");
            sb.append("    }");
            if (i < gemmArtifacts.bestRuns.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ],\n");
        sb.append("  \"results\": [\n");
        for (int i = 0; i < results.size(); i++) {
            PesResult r = results.get(i);
            sb.append("    {\n");
            sb.append("      \"kernel\": \"").append(escape(r.kernel)).append("\",\n");
            sb.append("      \"measurement_scope\": \"end_to_end\",\n");
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
        Files.writeString(output, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void printSummary(BenchmarkOutputs outputs,
                                     RooflineSession roofline,
                                     List<PesResult> results,
                                     List<KernelSweepSummary> maxima,
                                     List<GemmRunResult> gemmBestRuns) {
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
        System.out.println("--- GEMM best-by-size ---");
        for (GemmRunResult run : gemmBestRuns) {
            PesResult r = run.result;
            if (run.nativeCoreResult != null) {
                System.out.printf(Locale.ROOT,
                    "GEMM n=%d measured_gflops=%.4f pes=%.4f core_gflops=%.4f core_pes=%.4f coverage=%.4f threads=%d mr=%d nr=%d bound=%s level=%s%n",
                    r.n, r.measuredGflops, r.portableEfficiencyScore,
                    run.nativeCoreResult.measuredGflops, run.nativeCoreResult.portableEfficiencyScore, run.nativeCoreCoverage,
                    run.threads, run.mr, run.nr, r.boundType, r.memoryLevel);
            } else {
                System.out.printf(Locale.ROOT,
                    "GEMM n=%d measured_gflops=%.4f pes=%.4f threads=%d mr=%d nr=%d bound=%s level=%s%n",
                    r.n, r.measuredGflops, r.portableEfficiencyScore, run.threads, run.mr, run.nr, r.boundType, r.memoryLevel);
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
        System.out.println("CSV=" + outputs.csvOutput);
        System.out.println("MAX_CSV=" + outputs.maxCsvOutput);
        System.out.println("JSON=" + outputs.jsonOutput);
        System.out.println("GEMM_RUNS_CSV=" + outputs.gemmRunsCsvOutput);
        System.out.println("GEMM_BEST_BY_SIZE_CSV=" + outputs.gemmBestBySizeOutput);
    }

    private static int gemmMaxEnabled() {
        return parsePositiveInt(System.getProperty(GEMM_MAX_N_PROPERTY), DEFAULT_GEMM_MAX_N);
    }

    private static boolean gemmOnlyMode() {
        return Boolean.parseBoolean(System.getProperty("jlc.roofline.gemm_only", "false"));
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static int[] defaultThreadCandidates() {
        int maxThreads = Math.max(1, Runtime.getRuntime().availableProcessors());
        TreeSet<Integer> values = new TreeSet<>();
        int threads = 1;
        while (threads < maxThreads) {
            values.add(threads);
            threads <<= 1;
        }
        values.add(maxThreads);
        return values.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] defaultMrCandidates(int defaultMr) {
        TreeSet<Integer> values = new TreeSet<>();
        values.add(Math.max(1, defaultMr - 1));
        values.add(Math.max(1, defaultMr));
        values.add(Math.max(1, defaultMr + 1));
        return values.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] defaultNrCandidates(int defaultNr) {
        return new int[]{Math.max(1, defaultNr)};
    }

    private static int[] parsePositiveIntList(String raw, int[] fallback) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        TreeSet<Integer> values = new TreeSet<>();
        for (String token : raw.split(",")) {
            String trimmed = token.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            try {
                int parsed = Integer.parseInt(trimmed);
                if (parsed > 0) {
                    values.add(parsed);
                }
            } catch (NumberFormatException ignored) {
            }
        }
        if (values.isEmpty()) {
            return fallback;
        }
        return values.stream().mapToInt(Integer::intValue).toArray();
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

    private static int[] sweepSizes() {
        return parsePositiveIntList(System.getProperty(SIZES_PROPERTY), SWEEP_SIZES);
    }

    private static String readPropertyOrEnv(String propertyKey, String alternatePropertyKey, String envKey, String fallback) {
        String byProperty = System.getProperty(propertyKey);
        if (byProperty != null && !byProperty.isBlank()) {
            return byProperty;
        }
        if (alternatePropertyKey != null) {
            String byAlternateProperty = System.getProperty(alternatePropertyKey);
            if (byAlternateProperty != null && !byAlternateProperty.isBlank()) {
                return byAlternateProperty;
            }
        }
        String byEnv = System.getenv(envKey);
        if (byEnv != null && !byEnv.isBlank()) {
            return byEnv;
        }
        return fallback;
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

    private static String intArrayJson(int[] values) {
        return java.util.Arrays.stream(values).mapToObj(String::valueOf).collect(Collectors.joining(","));
    }

    @FunctionalInterface
    private interface KernelRunner {
        PesResult run(int n, RooflineSession roofline);
    }

    private static final class BenchmarkOutputs {
        final Path outputDir;
        final Path csvOutput;
        final Path maxCsvOutput;
        final Path jsonOutput;
        final Path gemmRunsCsvOutput;
        final Path gemmBestBySizeOutput;

        BenchmarkOutputs(Path outputDir) {
            this.outputDir = outputDir;
            this.csvOutput = outputDir.resolve(RESULTS_CSV);
            this.maxCsvOutput = outputDir.resolve(MAXIMA_CSV);
            this.jsonOutput = outputDir.resolve(RESULTS_JSON);
            this.gemmRunsCsvOutput = outputDir.resolve(GEMM_RUNS_CSV);
            this.gemmBestBySizeOutput = outputDir.resolve(GEMM_BEST_BY_SIZE_CSV);
        }

        static BenchmarkOutputs fromSystemProperties() {
            String rawPath = readPropertyOrEnv(
                OUTPUT_DIR_PROPERTY,
                OUTPUT_DIR_PROPERTY_ALT,
                OUTPUT_DIR_ENV,
                DEFAULT_OUTPUT_DIR.toString()
            );
            return new BenchmarkOutputs(Paths.get(rawPath));
        }
    }

    private static final class GemmBenchmarkConfig {
        final int[] threadCandidates;
        final int[] mrCandidates;
        final int[] nrCandidates;

        GemmBenchmarkConfig(int[] threadCandidates, int[] mrCandidates, int[] nrCandidates) {
            this.threadCandidates = threadCandidates;
            this.mrCandidates = mrCandidates;
            this.nrCandidates = nrCandidates;
        }

        static GemmBenchmarkConfig fromSystemProperties(GemmDispatch.BlockSizes defaults) {
            int[] threadCandidates = parsePositiveIntList(
                System.getProperty(GEMM_THREADS_PROPERTY),
                defaultThreadCandidates()
            );
            int[] mrCandidates = parsePositiveIntList(
                System.getProperty(GEMM_MR_VALUES_PROPERTY),
                defaultMrCandidates(defaults.mr)
            );
            int[] nrCandidates = parsePositiveIntList(
                System.getProperty(GEMM_NR_VALUES_PROPERTY),
                defaultNrCandidates(defaults.nr)
            );
            return new GemmBenchmarkConfig(threadCandidates, mrCandidates, nrCandidates);
        }
    }

    private static final class GemmRunResult {
        final PesResult result;
        final int threads;
        final int mr;
        final int nr;
        final int mc;
        final int nc;
        final int kc;
        final NativeGemmProfile nativeProfile;
        final PesResult nativeCoreResult;
        final double nativeCoreCoverage;

        GemmRunResult(PesResult result, int threads, int mr, int nr, int mc, int nc, int kc,
                      NativeGemmProfile nativeProfile, PesResult nativeCoreResult, double nativeCoreCoverage) {
            this.result = result;
            this.threads = threads;
            this.mr = mr;
            this.nr = nr;
            this.mc = mc;
            this.nc = nc;
            this.kc = kc;
            this.nativeProfile = nativeProfile == null ? NativeGemmProfile.EMPTY : nativeProfile;
            this.nativeCoreResult = nativeCoreResult;
            this.nativeCoreCoverage = nativeCoreCoverage;
        }

        double nativeCoreSeconds() {
            return nativeCoreResult == null ? 0.0 : nativeCoreResult.elapsedSeconds;
        }

        double nativeCoreMeasuredGflops() {
            return nativeCoreResult == null ? 0.0 : nativeCoreResult.measuredGflops;
        }

        double nativeCorePes() {
            return nativeCoreResult == null ? 0.0 : nativeCoreResult.portableEfficiencyScore;
        }

        long nativeProfileCallCount() {
            return nativeProfile.calls();
        }

        double nativeVendorSeconds() {
            return nativeProfile.vendorNanos() / 1e9;
        }

        double nativeScaleCSeconds() {
            return nativeProfile.scaleCNanos() / 1e9;
        }

        double nativePackASeconds() {
            return nativeProfile.packANanos() / 1e9;
        }

        double nativePackBSeconds() {
            return nativeProfile.packBNanos() / 1e9;
        }

        double nativeKernelSeconds() {
            return nativeProfile.nativeKernelNanos() / 1e9;
        }

        double nativeThreadSeconds() {
            return nativeProfile.nativeThreadingNanos() / 1e9;
        }
    }

    private static final class GemmSweepArtifacts {
        final GemmBenchmarkConfig config;
        final List<GemmRunResult> allRuns;
        final List<GemmRunResult> bestRuns;
        final List<PesResult> bestResults;
        final List<Integer> testedSizes;

        GemmSweepArtifacts(GemmBenchmarkConfig config,
                           List<GemmRunResult> allRuns,
                           List<GemmRunResult> bestRuns,
                           List<PesResult> bestResults,
                           List<Integer> testedSizes) {
            this.config = config;
            this.allRuns = allRuns;
            this.bestRuns = bestRuns;
            this.bestResults = bestResults;
            this.testedSizes = testedSizes;
        }
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
