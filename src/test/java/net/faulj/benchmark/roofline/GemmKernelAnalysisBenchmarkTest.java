package net.faulj.benchmark.roofline;

import net.faulj.compute.DispatchPolicy;
import net.faulj.compute.GemmDispatch;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * GEMM-specific, black-box performance analysis harness.
 */
public class GemmKernelAnalysisBenchmarkTest {
    private static final Path DEFAULT_OUTPUT_DIR = Paths.get("build", "reports", "gemm-analysis");
    private static final String RUNS_FILE = "gemm_analysis_runs.csv";
    private static final String SUMMARY_FILE = "gemm_analysis_summary.json";
    private static final String SUITE_FILE = "gemm_analysis_suite.json";
    private static final String SCHEMA_FILE = "gemm_analysis_schema.json";
    private static final String OPPORTUNITIES_FILE = "opportunities.md";
    private static final String PERFORMANCE_FILE = "PERFORMANCE.md";

    private static volatile double sink;

    @Test
    public void runGemmKernelAnalysis() throws IOException {
        Assume.assumeTrue(
            "Benchmark mode is disabled. Enable with -D" + BenchmarkMode.MODE_PROPERTY + "=true",
            BenchmarkMode.isEnabled()
        );

        AnalysisConfig config = AnalysisConfig.fromSystemProperties();
        RooflineSession roofline = RooflineSession.get();
        GemmDispatch.BlockSizes blockSizes = GemmDispatch.computeBlockSizes();
        int socketCores = Math.max(1, roofline.hardware.cores);
        int[] primaryThreads = primaryThreadSet(socketCores);
        int[] scalingThreads = scalingThreadSet(socketCores);

        Files.createDirectories(config.outputDir);
        Path runsCsv = config.outputDir.resolve(RUNS_FILE);
        Map<String, Double> priorBest = loadPriorBestByKey(runsCsv);

        List<GemmCase> geometryCases = buildGeometryCases(config);
        List<GemmCase> scalingCases = buildScalingCases(config);
        List<GemmCase> simdProbeCases = buildSimdProbeCases(config);
        writeSuiteManifest(config, roofline, geometryCases, scalingCases, simdProbeCases, primaryThreads, scalingThreads);
        writeSchema(config.outputDir.resolve(SCHEMA_FILE));

        List<Observation> observations = new ArrayList<>();

        for (GemmCase gemmCase : geometryCases) {
            for (int threads : primaryThreads) {
                observations.add(runCase(
                    gemmCase, "geometry", "default", threads, true, roofline, blockSizes, config
                ));
            }
        }

        for (GemmCase gemmCase : scalingCases) {
            for (int threads : scalingThreads) {
                observations.add(runCase(
                    gemmCase, "scaling", "default", threads, true, roofline, blockSizes, config
                ));
            }
        }

        for (GemmCase gemmCase : simdProbeCases) {
            observations.add(runCase(
                gemmCase, "simd_probe", "simd_on", 1, true, roofline, blockSizes, config
            ));
            observations.add(runCase(
                gemmCase, "simd_probe", "simd_off", 1, false, roofline, blockSizes, config
            ));
        }

        annotateScaling(observations);
        annotateSimd(observations, blockSizes.nr);
        annotateRegression(observations, priorBest, config.regressionThresholdFraction);

        List<CacheTransition> transitions = detectCacheTransitions(observations, 1);
        List<Opportunity> opportunities = deriveOpportunities(observations, transitions, socketCores);

        writeRunsCsv(runsCsv, observations);
        writeSummaryJson(config.outputDir.resolve(SUMMARY_FILE), config, roofline, blockSizes, observations, transitions, opportunities);
        writeOpportunitiesMarkdown(config.outputDir.resolve(OPPORTUNITIES_FILE), opportunities);
        writePerformanceMarkdown(config.outputDir.resolve(PERFORMANCE_FILE), config, roofline, observations, transitions, opportunities);

        printSummary(config, roofline, observations, transitions, opportunities);
    }

    private static Observation runCase(GemmCase gemmCase,
                                       String scenario,
                                       String variant,
                                       int threads,
                                       boolean simdEnabled,
                                       RooflineSession roofline,
                                       GemmDispatch.BlockSizes blockSizes,
                                       AnalysisConfig config) {
        Matrix a = randomMatrix(gemmCase.m, gemmCase.k, gemmCase.seedA);
        Matrix b = randomMatrix(gemmCase.k, gemmCase.n, gemmCase.seedB);
        Matrix c = new Matrix(gemmCase.m, gemmCase.n);
        DispatchPolicy policy = DispatchPolicy.builder()
            .enableCuda(false)
            .enableParallel(threads > 1)
            .parallelism(threads)
            .enableStrassen(false)
            .enableSimd(simdEnabled)
            .build();

        IterationPlan plan = IterationPlan.forFlops(gemmCase.flops(), config);
        for (int i = 0; i < plan.warmupRuns; i++) {
            Arrays.fill(c.getRawData(), 0.0);
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            sink += c.get(0, 0);
        }

        double[] elapsed = new double[plan.measurementRuns];
        for (int i = 0; i < plan.measurementRuns; i++) {
            Arrays.fill(c.getRawData(), 0.0);
            long start = System.nanoTime();
            Gemm.gemm(a, b, c, 1.0, 0.0, policy);
            long nanos = System.nanoTime() - start;
            elapsed[i] = nanos / 1e9;
            sink += c.get(0, 0);
        }

        RunStats stats = RunStats.of(elapsed);
        double flops = gemmCase.flops();
        double bytesMin = gemmCase.minBytes();
        double bytesModeled = gemmCase.modeledBytes();
        double aiMin = safeDivide(flops, bytesMin);
        double aiModeled = safeDivide(flops, bytesModeled);
        double workingSet = gemmCase.workingSetBytes();
        MemoryBandwidthProbe.BandwidthSelection bw = roofline.memoryBandwidths.forWorkingSet(workingSet);

        double computeRoof = threadScaledComputeRoof(roofline, threads);
        double memoryRoofFlops = aiModeled * bw.bytesPerSecond;
        double effectiveRoofFlops = Math.min(computeRoof, memoryRoofFlops);
        String bound = computeRoof <= memoryRoofFlops ? "compute" : "memory";

        double measuredFlops = safeDivide(flops, stats.bestSeconds);
        double measuredBytes = safeDivide(bytesModeled, stats.bestSeconds);
        double measuredGflopsBest = measuredFlops / 1e9;
        double measuredGflopsMedian = safeDivide(flops, stats.medianSeconds) / 1e9;
        double measuredGflopsMean = safeDivide(flops, stats.meanSeconds) / 1e9;
        double measuredBandwidthGbps = measuredBytes / 1e9;

        double computeUtil = clamp01(safeDivide(measuredFlops, computeRoof));
        double memoryUtil = clamp01(safeDivide(measuredBytes, bw.bytesPerSecond));
        double rooflineUtil = clamp01(safeDivide(measuredFlops, Math.max(1e-9, effectiveRoofFlops)));

        double cycles = estimatedCycles(stats.bestSeconds, roofline.hardware.clockGhz, threads);
        double flopsPerCycle = safeDivide(flops, cycles);
        double peakFlopsPerCycleCore = roofline.hardware.simdLanesDouble
            * (roofline.hardware.fmaEnabled ? 2.0 : 1.0)
            * Math.max(1, roofline.hardware.vectorIssueWidth);
        double perCoreFlopsPerCycle = safeDivide(flopsPerCycle, threads);
        double fmaUtilProxy = clamp01(safeDivide(perCoreFlopsPerCycle, peakFlopsPerCycleCore));

        int roundUpN = roundUp(gemmCase.n, Math.max(1, blockSizes.nr));
        double laneOccupancy = gemmCase.n <= 0 ? 0.0 : safeDivide(gemmCase.n, roundUpN);
        double vectorWidthUtil = safeDivide(blockSizes.nr, Math.max(1.0, roofline.hardware.simdLanesDouble));

        Observation o = new Observation();
        o.timestampUtc = Instant.now().toString();
        o.caseId = gemmCase.id;
        o.family = gemmCase.family;
        o.scenario = scenario;
        o.variant = variant;
        o.kernel = "GEMM";
        o.m = gemmCase.m;
        o.n = gemmCase.n;
        o.k = gemmCase.k;
        o.shape = classifyShape(gemmCase.m, gemmCase.n, gemmCase.k);
        o.threads = threads;
        o.simdEnabled = simdEnabled;
        o.dispatchAlgorithm = policy.selectCpuAlgorithm(gemmCase.m, gemmCase.n, gemmCase.k).name();
        o.warmupRuns = plan.warmupRuns;
        o.measurementRuns = plan.measurementRuns;

        o.flops = flops;
        o.bytesMin = bytesMin;
        o.bytesModeled = bytesModeled;
        o.arithmeticIntensityMin = aiMin;
        o.arithmeticIntensityModeled = aiModeled;
        o.workingSetBytes = workingSet;
        o.cacheRegime = classifyCacheRegime(workingSet, roofline.memoryBandwidths);

        o.bestSeconds = stats.bestSeconds;
        o.meanSeconds = stats.meanSeconds;
        o.medianSeconds = stats.medianSeconds;
        o.p95Seconds = stats.p95Seconds;
        o.stdevSeconds = stats.stdevSeconds;
        o.cv = stats.cv;

        o.measuredGflopsBest = measuredGflopsBest;
        o.measuredGflopsMedian = measuredGflopsMedian;
        o.measuredGflopsMean = measuredGflopsMean;
        o.measuredBandwidthGbps = measuredBandwidthGbps;

        o.computeRoofGflops = computeRoof / 1e9;
        o.memoryRoofGflops = memoryRoofFlops / 1e9;
        o.effectiveRoofGflops = effectiveRoofFlops / 1e9;
        o.boundType = bound;
        o.computeUtilization = computeUtil;
        o.memoryUtilization = memoryUtil;
        o.rooflineUtilization = rooflineUtil;
        o.rooflineGapGflops = Math.max(0.0, o.effectiveRoofGflops - measuredGflopsBest);
        o.selectedMemoryRoofGbps = bw.bytesPerSecond / 1e9;
        o.selectedMemoryLevel = bw.level.name().toLowerCase(Locale.ROOT);

        o.estimatedCycles = cycles;
        o.flopsPerCycle = flopsPerCycle;
        o.perCoreFlopsPerCycle = perCoreFlopsPerCycle;
        o.fmaUtilizationProxy = fmaUtilProxy;
        o.loadStoreBytesPerFlop = safeDivide(bytesModeled, flops);

        o.simdLanesDouble = roofline.hardware.simdLanesDouble;
        o.mr = blockSizes.mr;
        o.nr = blockSizes.nr;
        o.kc = blockSizes.kc;
        o.mc = blockSizes.mc;
        o.nc = blockSizes.nc;
        o.remainderM = positiveRemainder(gemmCase.m, blockSizes.mr);
        o.remainderN = positiveRemainder(gemmCase.n, blockSizes.nr);
        o.remainderK = positiveRemainder(gemmCase.k, Math.max(1, blockSizes.kc));
        o.simdLaneOccupancy = laneOccupancy;
        o.vectorWidthUtilization = clamp01(vectorWidthUtil);
        o.scalarFallbackSuspected = false;

        o.l1MissRate = Double.NaN;
        o.l2MissRate = Double.NaN;
        o.l3MissRate = Double.NaN;
        o.instructionsPerCycle = Double.NaN;
        o.simdInstructionRatio = Double.NaN;
        o.counterSource = "not-collected";

        o.speedupVsOneThread = Double.NaN;
        o.parallelEfficiency = Double.NaN;
        o.simdSpeedupVsScalar = Double.NaN;
        o.simdEfficiencyVsLanes = Double.NaN;
        o.regressionVsPreviousFraction = Double.NaN;
        o.regressionFlag = false;

        return o;
    }

    private static void annotateScaling(List<Observation> observations) {
        Map<String, Double> oneThreadByCase = new HashMap<>();
        for (Observation o : observations) {
            if (!"scaling".equals(o.scenario)) {
                continue;
            }
            if (o.threads == 1 && o.simdEnabled) {
                oneThreadByCase.put(o.caseId, o.measuredGflopsBest);
            }
        }
        for (Observation o : observations) {
            if (!"scaling".equals(o.scenario)) {
                continue;
            }
            Double base = oneThreadByCase.get(o.caseId);
            if (base == null || base <= 0.0) {
                continue;
            }
            o.speedupVsOneThread = safeDivide(o.measuredGflopsBest, base);
            o.parallelEfficiency = safeDivide(o.speedupVsOneThread, Math.max(1.0, o.threads));
        }
    }

    private static void annotateSimd(List<Observation> observations, int nr) {
        Map<String, Double> scalarByCase = new HashMap<>();
        for (Observation o : observations) {
            if (!"simd_probe".equals(o.scenario)) {
                continue;
            }
            if (!o.simdEnabled) {
                scalarByCase.put(o.caseId, o.measuredGflopsBest);
            }
        }
        for (Observation o : observations) {
            if (!"simd_probe".equals(o.scenario) || !o.simdEnabled) {
                continue;
            }
            Double scalar = scalarByCase.get(o.caseId);
            if (scalar == null || scalar <= 0.0) {
                continue;
            }
            o.simdSpeedupVsScalar = safeDivide(o.measuredGflopsBest, scalar);
            o.simdEfficiencyVsLanes = clamp01(safeDivide(o.simdSpeedupVsScalar, Math.max(1.0, o.simdLanesDouble)));
            boolean vectorFriendly = o.n >= nr * 8 && o.remainderN == 0;
            o.scalarFallbackSuspected = vectorFriendly && o.simdSpeedupVsScalar < 1.10;
        }
    }

    private static void annotateRegression(List<Observation> observations,
                                           Map<String, Double> priorBestByKey,
                                           double regressionThresholdFraction) {
        for (Observation o : observations) {
            String key = observationKey(o);
            Double prior = priorBestByKey.get(key);
            if (prior == null || prior <= 0.0) {
                continue;
            }
            o.regressionVsPreviousFraction = safeDivide(o.measuredGflopsBest - prior, prior);
            o.regressionFlag = o.regressionVsPreviousFraction < -Math.abs(regressionThresholdFraction);
        }
    }
    private static List<CacheTransition> detectCacheTransitions(List<Observation> observations, int socketCores) {
        List<Observation> square = observations.stream()
            .filter(o -> "geometry".equals(o.scenario))
            .filter(o -> o.threads == socketCores)
            .filter(o -> o.simdEnabled)
            .filter(o -> "square".equals(o.shape))
            .sorted(Comparator.comparingDouble(o -> o.workingSetBytes))
            .collect(Collectors.toList());

        if (square.size() < 2) {
            return Collections.emptyList();
        }

        List<CacheTransition> transitions = new ArrayList<>();
        Observation prev = square.get(0);
        for (int i = 1; i < square.size(); i++) {
            Observation curr = square.get(i);
            boolean regimeChange = !prev.cacheRegime.equals(curr.cacheRegime);
            double perfDropFraction = safeDivide(prev.measuredGflopsBest - curr.measuredGflopsBest,
                Math.max(1e-9, prev.measuredGflopsBest));
            boolean largeDrop = perfDropFraction > 0.15;
            if (regimeChange || largeDrop) {
                CacheTransition t = new CacheTransition();
                t.fromCase = prev.caseId;
                t.toCase = curr.caseId;
                t.fromRegime = prev.cacheRegime;
                t.toRegime = curr.cacheRegime;
                t.fromWorkingSetBytes = prev.workingSetBytes;
                t.toWorkingSetBytes = curr.workingSetBytes;
                t.fromGflops = prev.measuredGflopsBest;
                t.toGflops = curr.measuredGflopsBest;
                t.dropFraction = Math.max(0.0, perfDropFraction);
                t.reason = regimeChange ? "cache-regime-change" : "performance-drop";
                transitions.add(t);
            }
            prev = curr;
        }
        return transitions;
    }

    private static List<Opportunity> deriveOpportunities(List<Observation> observations,
                                                         List<CacheTransition> transitions,
                                                         int socketCores) {
        Set<String> unique = new LinkedHashSet<>();
        List<Opportunity> opportunities = new ArrayList<>();

        for (Observation o : observations) {
            if (!"geometry".equals(o.scenario) || o.threads != socketCores || !o.simdEnabled) {
                continue;
            }
            if ("memory".equals(o.boundType) && o.memoryUtilization > 0.85 && o.rooflineUtilization < 0.65) {
                addOpportunity(unique, opportunities, "high", o.caseId,
                    "Memory bandwidth is near saturation but roofline utilization is low; focus on reducing bytes moved (packing reuse, improved blocking reuse, write-allocate avoidance).");
            }
            if ("compute".equals(o.boundType) && o.computeUtilization < 0.65 && o.simdLaneOccupancy < 0.90) {
                addOpportunity(unique, opportunities, "medium", o.caseId,
                    "Compute-bound case underuses SIMD lanes; investigate remainder paths and NR/MR tuning for better lane occupancy.");
            }
            if (o.cv > 0.05 && o.bestSeconds > 0.001) {
                addOpportunity(unique, opportunities, "medium", o.caseId,
                    "Runtime variance is high (CV > 5%); evaluate CPU frequency pinning, affinity, and GC noise controls for stable benchmarking.");
            }
            if (o.regressionFlag) {
                addOpportunity(unique, opportunities, "high", o.caseId,
                    String.format(Locale.ROOT, "Detected regression vs prior run: %.1f%%", 100.0 * o.regressionVsPreviousFraction));
            }
        }

        Map<String, List<Observation>> scalingGroups = observations.stream()
            .filter(o -> "scaling".equals(o.scenario))
            .filter(o -> o.simdEnabled)
            .collect(Collectors.groupingBy(o -> o.caseId, LinkedHashMap::new, Collectors.toList()));
        for (Map.Entry<String, List<Observation>> entry : scalingGroups.entrySet()) {
            List<Observation> group = entry.getValue().stream()
                .sorted(Comparator.comparingInt(o -> o.threads))
                .collect(Collectors.toList());
            Observation max = group.get(group.size() - 1);
            if (Double.isFinite(max.parallelEfficiency) && max.parallelEfficiency < 0.60 && max.threads >= 4) {
                addOpportunity(unique, opportunities, "high", max.caseId,
                    String.format(Locale.ROOT,
                        "Parallel scaling is weak at %d threads (efficiency %.2f); inspect tile partitioning, work-stealing overhead, and false-sharing risk.",
                        max.threads, max.parallelEfficiency));
            }
        }

        for (Observation o : observations) {
            if (!"simd_probe".equals(o.scenario) || !o.simdEnabled) {
                continue;
            }
            if (o.scalarFallbackSuspected) {
                addOpportunity(unique, opportunities, "high", o.caseId,
                    String.format(Locale.ROOT,
                        "SIMD fallback suspected (speedup %.2fx vs scalar on vector-friendly dimensions). Validate vectorized microkernel dispatch and JIT vector lowering.",
                        o.simdSpeedupVsScalar));
            }
            if (Double.isFinite(o.simdEfficiencyVsLanes) && o.simdEfficiencyVsLanes < 0.35) {
                addOpportunity(unique, opportunities, "medium", o.caseId,
                    String.format(Locale.ROOT,
                        "Low SIMD efficiency vs lane count (%.2f). Consider improving FMA throughput and reducing scalar remainder handling.",
                        o.simdEfficiencyVsLanes));
            }
        }

        for (CacheTransition t : transitions) {
            if (t.dropFraction > 0.20) {
                addOpportunity(unique, opportunities, "medium", t.toCase,
                    String.format(Locale.ROOT,
                        "Significant throughput drop (%.1f%%) across %s->%s transition; tune panel/block sizes around this working-set boundary.",
                        100.0 * t.dropFraction, t.fromRegime, t.toRegime));
            }
        }

        return opportunities;
    }

    private static void addOpportunity(Set<String> unique, List<Opportunity> opportunities, String severity, String caseId, String text) {
        String key = severity + "|" + caseId + "|" + text;
        if (unique.add(key)) {
            Opportunity o = new Opportunity();
            o.severity = severity;
            o.caseId = caseId;
            o.text = text;
            opportunities.add(o);
        }
    }

    private static void writeRunsCsv(Path out, List<Observation> observations) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("timestamp_utc,case_id,family,scenario,variant,kernel,m,n,k,shape,threads,simd_enabled,dispatch_algorithm,warmup_runs,measurement_runs,");
        sb.append("flops,bytes_min,bytes_modeled,arithmetic_intensity_min,arithmetic_intensity_modeled,working_set_bytes,cache_regime,");
        sb.append("best_seconds,mean_seconds,median_seconds,p95_seconds,stdev_seconds,cv,");
        sb.append("measured_gflops_best,measured_gflops_median,measured_gflops_mean,measured_bandwidth_gbps,");
        sb.append("compute_roof_gflops,memory_roof_gflops,effective_roof_gflops,bound_type,compute_utilization,memory_utilization,roofline_utilization,roofline_gap_gflops,");
        sb.append("selected_memory_roof_gbps,selected_memory_level,estimated_cycles,flops_per_cycle,per_core_flops_per_cycle,fma_utilization_proxy,load_store_bytes_per_flop,");
        sb.append("simd_lanes_double,mr,nr,kc,mc,nc,remainder_m,remainder_n,remainder_k,simd_lane_occupancy,vector_width_utilization,scalar_fallback_suspected,");
        sb.append("l1_miss_rate,l2_miss_rate,l3_miss_rate,instructions_per_cycle,simd_instruction_ratio,counter_source,");
        sb.append("speedup_vs_1thread,parallel_efficiency,simd_speedup_vs_scalar,simd_efficiency_vs_lanes,");
        sb.append("regression_vs_previous_fraction,regression_flag\n");

        for (Observation o : observations) {
            csv(sb, o.timestampUtc);
            csv(sb, o.caseId);
            csv(sb, o.family);
            csv(sb, o.scenario);
            csv(sb, o.variant);
            csv(sb, o.kernel);
            csv(sb, o.m);
            csv(sb, o.n);
            csv(sb, o.k);
            csv(sb, o.shape);
            csv(sb, o.threads);
            csv(sb, o.simdEnabled);
            csv(sb, o.dispatchAlgorithm);
            csv(sb, o.warmupRuns);
            csv(sb, o.measurementRuns);
            csv(sb, o.flops);
            csv(sb, o.bytesMin);
            csv(sb, o.bytesModeled);
            csv(sb, o.arithmeticIntensityMin);
            csv(sb, o.arithmeticIntensityModeled);
            csv(sb, o.workingSetBytes);
            csv(sb, o.cacheRegime);
            csv(sb, o.bestSeconds);
            csv(sb, o.meanSeconds);
            csv(sb, o.medianSeconds);
            csv(sb, o.p95Seconds);
            csv(sb, o.stdevSeconds);
            csv(sb, o.cv);
            csv(sb, o.measuredGflopsBest);
            csv(sb, o.measuredGflopsMedian);
            csv(sb, o.measuredGflopsMean);
            csv(sb, o.measuredBandwidthGbps);
            csv(sb, o.computeRoofGflops);
            csv(sb, o.memoryRoofGflops);
            csv(sb, o.effectiveRoofGflops);
            csv(sb, o.boundType);
            csv(sb, o.computeUtilization);
            csv(sb, o.memoryUtilization);
            csv(sb, o.rooflineUtilization);
            csv(sb, o.rooflineGapGflops);
            csv(sb, o.selectedMemoryRoofGbps);
            csv(sb, o.selectedMemoryLevel);
            csv(sb, o.estimatedCycles);
            csv(sb, o.flopsPerCycle);
            csv(sb, o.perCoreFlopsPerCycle);
            csv(sb, o.fmaUtilizationProxy);
            csv(sb, o.loadStoreBytesPerFlop);
            csv(sb, o.simdLanesDouble);
            csv(sb, o.mr);
            csv(sb, o.nr);
            csv(sb, o.kc);
            csv(sb, o.mc);
            csv(sb, o.nc);
            csv(sb, o.remainderM);
            csv(sb, o.remainderN);
            csv(sb, o.remainderK);
            csv(sb, o.simdLaneOccupancy);
            csv(sb, o.vectorWidthUtilization);
            csv(sb, o.scalarFallbackSuspected);
            csv(sb, o.l1MissRate);
            csv(sb, o.l2MissRate);
            csv(sb, o.l3MissRate);
            csv(sb, o.instructionsPerCycle);
            csv(sb, o.simdInstructionRatio);
            csv(sb, o.counterSource);
            csv(sb, o.speedupVsOneThread);
            csv(sb, o.parallelEfficiency);
            csv(sb, o.simdSpeedupVsScalar);
            csv(sb, o.simdEfficiencyVsLanes);
            csv(sb, o.regressionVsPreviousFraction);
            csv(sb, o.regressionFlag);
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) == ',') {
                sb.setLength(sb.length() - 1);
            }
            sb.append('\n');
        }

        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }
    private static void writeSummaryJson(Path out,
                                         AnalysisConfig config,
                                         RooflineSession roofline,
                                         GemmDispatch.BlockSizes blockSizes,
                                         List<Observation> observations,
                                         List<CacheTransition> transitions,
                                         List<Opportunity> opportunities) throws IOException {
        List<Observation> sorted = observations.stream()
            .sorted(Comparator.comparingDouble((Observation o) -> o.measuredGflopsBest).reversed())
            .collect(Collectors.toList());
        Observation best = sorted.isEmpty() ? null : sorted.get(0);
        double maxGeometryCv = observations.stream()
            .filter(o -> "geometry".equals(o.scenario))
            .mapToDouble(o -> o.cv)
            .max()
            .orElse(0.0);
        double worstScalingEff = observations.stream()
            .filter(o -> "scaling".equals(o.scenario))
            .filter(o -> Double.isFinite(o.parallelEfficiency))
            .mapToDouble(o -> o.parallelEfficiency)
            .min()
            .orElse(Double.NaN);

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        appendJsonField(sb, "generated_utc", Instant.now().toString(), true);
        sb.append("  \"config\": {\n");
        appendJsonField(sb, "profile", config.profile, true, 4);
        appendJsonField(sb, "output_dir", config.outputDir.toString(), true, 4);
        appendJsonField(sb, "forced_warmup_runs", config.forcedWarmupRuns, true, 4);
        appendJsonField(sb, "forced_measurement_runs", config.forcedMeasurementRuns, true, 4);
        appendJsonField(sb, "regression_threshold_fraction", config.regressionThresholdFraction, false, 4);
        sb.append("  },\n");
        sb.append("  \"hardware\": {\n");
        appendJsonField(sb, "cores", roofline.hardware.cores, true, 4);
        appendJsonField(sb, "clock_ghz", roofline.hardware.clockGhz, true, 4);
        appendJsonField(sb, "simd_lanes_double", roofline.hardware.simdLanesDouble, true, 4);
        appendJsonField(sb, "fma_enabled", roofline.hardware.fmaEnabled, true, 4);
        appendJsonField(sb, "vector_issue_width", roofline.hardware.vectorIssueWidth, true, 4);
        appendJsonField(sb, "clock_source", roofline.hardware.clockSource, false, 4);
        sb.append("  },\n");
        sb.append("  \"roofline\": {\n");
        appendJsonField(sb, "compute_roof_gflops", roofline.peakFlopsPerSecond / 1e9, true, 4);
        appendJsonField(sb, "compute_theoretical_peak_gflops", roofline.rawTheoreticalPeakFlopsPerSecond / 1e9, true, 4);
        appendJsonField(sb, "compute_gemm_anchor_gflops", roofline.measuredGemmAnchorGflops, true, 4);
        appendJsonField(sb, "memory_l1_gbps", roofline.memoryBandwidths.l1BytesPerSecond / 1e9, true, 4);
        appendJsonField(sb, "memory_l2_gbps", roofline.memoryBandwidths.l2BytesPerSecond / 1e9, true, 4);
        appendJsonField(sb, "memory_l3_gbps", roofline.memoryBandwidths.l3BytesPerSecond / 1e9, true, 4);
        appendJsonField(sb, "memory_dram_gbps", roofline.memoryBandwidths.dramBytesPerSecond / 1e9, true, 4);
        appendJsonField(sb, "compute_source", roofline.computeRoofSource, true, 4);
        appendJsonField(sb, "memory_source", roofline.memoryRoofSource, false, 4);
        sb.append("  },\n");
        sb.append("  \"block_sizes\": {\n");
        appendJsonField(sb, "mr", blockSizes.mr, true, 4);
        appendJsonField(sb, "nr", blockSizes.nr, true, 4);
        appendJsonField(sb, "kc", blockSizes.kc, true, 4);
        appendJsonField(sb, "mc", blockSizes.mc, true, 4);
        appendJsonField(sb, "nc", blockSizes.nc, false, 4);
        sb.append("  },\n");
        sb.append("  \"aggregate\": {\n");
        appendJsonField(sb, "observations", observations.size(), true, 4);
        appendJsonField(sb, "best_gflops", best == null ? Double.NaN : best.measuredGflopsBest, true, 4);
        appendJsonField(sb, "best_case_id", best == null ? "" : best.caseId, true, 4);
        appendJsonField(sb, "best_case_threads", best == null ? 0 : best.threads, true, 4);
        appendJsonField(sb, "best_case_bound_type", best == null ? "" : best.boundType, true, 4);
        appendJsonField(sb, "max_geometry_cv", maxGeometryCv, true, 4);
        appendJsonField(sb, "worst_scaling_efficiency", worstScalingEff, true, 4);
        appendJsonField(sb, "transition_count", transitions.size(), true, 4);
        appendJsonField(sb, "opportunity_count", opportunities.size(), false, 4);
        sb.append("  },\n");
        sb.append("  \"transitions\": [\n");
        for (int i = 0; i < transitions.size(); i++) {
            CacheTransition t = transitions.get(i);
            sb.append("    {\n");
            appendJsonField(sb, "from_case", t.fromCase, true, 6);
            appendJsonField(sb, "to_case", t.toCase, true, 6);
            appendJsonField(sb, "from_regime", t.fromRegime, true, 6);
            appendJsonField(sb, "to_regime", t.toRegime, true, 6);
            appendJsonField(sb, "from_working_set_bytes", t.fromWorkingSetBytes, true, 6);
            appendJsonField(sb, "to_working_set_bytes", t.toWorkingSetBytes, true, 6);
            appendJsonField(sb, "from_gflops", t.fromGflops, true, 6);
            appendJsonField(sb, "to_gflops", t.toGflops, true, 6);
            appendJsonField(sb, "drop_fraction", t.dropFraction, true, 6);
            appendJsonField(sb, "reason", t.reason, false, 6);
            sb.append("    }");
            if (i < transitions.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ],\n");
        sb.append("  \"opportunities\": [\n");
        for (int i = 0; i < opportunities.size(); i++) {
            Opportunity o = opportunities.get(i);
            sb.append("    {\n");
            appendJsonField(sb, "severity", o.severity, true, 6);
            appendJsonField(sb, "case_id", o.caseId, true, 6);
            appendJsonField(sb, "text", o.text, false, 6);
            sb.append("    }");
            if (i < opportunities.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ]\n");
        sb.append("}\n");
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeSuiteManifest(AnalysisConfig config,
                                           RooflineSession roofline,
                                           List<GemmCase> geometryCases,
                                           List<GemmCase> scalingCases,
                                           List<GemmCase> simdCases,
                                           int[] primaryThreads,
                                           int[] scalingThreads) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        appendJsonField(sb, "generated_utc", Instant.now().toString(), true);
        sb.append("  \"profile\": \"").append(escapeJson(config.profile)).append("\",\n");
        sb.append("  \"primary_threads\": ").append(intArrayJson(primaryThreads)).append(",\n");
        sb.append("  \"scaling_threads\": ").append(intArrayJson(scalingThreads)).append(",\n");
        sb.append("  \"cache_bytes\": {\n");
        appendJsonField(sb, "l1", roofline.memoryBandwidths.l1SizeBytes, true, 4);
        appendJsonField(sb, "l2", roofline.memoryBandwidths.l2SizeBytes, true, 4);
        appendJsonField(sb, "l3", roofline.memoryBandwidths.l3SizeBytes, false, 4);
        sb.append("  },\n");
        sb.append("  \"geometry_cases\": ").append(caseListJson(geometryCases)).append(",\n");
        sb.append("  \"scaling_cases\": ").append(caseListJson(scalingCases)).append(",\n");
        sb.append("  \"simd_probe_cases\": ").append(caseListJson(simdCases)).append('\n');
        sb.append("}\n");
        Files.writeString(config.outputDir.resolve(SUITE_FILE), sb.toString(), StandardCharsets.UTF_8);
    }

    private static String caseListJson(List<GemmCase> cases) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        for (int i = 0; i < cases.size(); i++) {
            GemmCase c = cases.get(i);
            sb.append("    {\"id\":\"").append(escapeJson(c.id)).append("\",\"family\":\"").append(escapeJson(c.family))
                .append("\",\"m\":").append(c.m).append(",\"n\":").append(c.n).append(",\"k\":").append(c.k).append("}");
            if (i < cases.size() - 1) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("  ]");
        return sb.toString();
    }

    private static void writeSchema(Path out) throws IOException {
        String schema = "{\n" +
            "  \"version\": \"1.0\",\n" +
            "  \"description\": \"Per-run GEMM analysis schema\",\n" +
            "  \"key_fields\": [\"case_id\", \"scenario\", \"threads\", \"simd_enabled\"],\n" +
            "  \"dimensions\": {\n" +
            "    \"geometry\": [\"m\", \"n\", \"k\", \"shape\"],\n" +
            "    \"execution\": [\"threads\", \"simd_enabled\", \"dispatch_algorithm\", \"warmup_runs\", \"measurement_runs\"],\n" +
            "    \"time_stats\": [\"best_seconds\", \"mean_seconds\", \"median_seconds\", \"p95_seconds\", \"stdev_seconds\", \"cv\"],\n" +
            "    \"roofline\": [\"arithmetic_intensity_modeled\", \"compute_roof_gflops\", \"memory_roof_gflops\", \"effective_roof_gflops\", \"bound_type\", \"roofline_utilization\"],\n" +
            "    \"memory\": [\"bytes_modeled\", \"working_set_bytes\", \"cache_regime\", \"measured_bandwidth_gbps\", \"selected_memory_roof_gbps\"],\n" +
            "    \"simd\": [\"simd_lanes_double\", \"mr\", \"nr\", \"remainder_n\", \"simd_lane_occupancy\", \"simd_speedup_vs_scalar\"],\n" +
            "    \"instruction_proxies\": [\"estimated_cycles\", \"flops_per_cycle\", \"per_core_flops_per_cycle\", \"fma_utilization_proxy\"],\n" +
            "    \"scaling\": [\"speedup_vs_1thread\", \"parallel_efficiency\"],\n" +
            "    \"stability\": [\"cv\", \"regression_vs_previous_fraction\", \"regression_flag\"]\n" +
            "  }\n" +
            "}\n";
        Files.writeString(out, schema, StandardCharsets.UTF_8);
    }

    private static void writeOpportunitiesMarkdown(Path out, List<Opportunity> opportunities) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# GEMM Optimization Opportunities\n\n");
        if (opportunities.isEmpty()) {
            sb.append("No actionable opportunities detected from current heuristics.\n");
        } else {
            for (Opportunity o : opportunities) {
                sb.append("- [").append(o.severity.toUpperCase(Locale.ROOT)).append("] ")
                    .append(o.caseId).append(": ").append(o.text).append('\n');
            }
        }
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }
    private static void writePerformanceMarkdown(Path out,
                                                 AnalysisConfig config,
                                                 RooflineSession roofline,
                                                 List<Observation> observations,
                                                 List<CacheTransition> transitions,
                                                 List<Opportunity> opportunities) throws IOException {
        Observation best = observations.stream()
            .max(Comparator.comparingDouble(o -> o.measuredGflopsBest))
            .orElse(null);
        List<Double> geometryUtil = observations.stream()
            .filter(o -> "geometry".equals(o.scenario))
            .map(o -> o.rooflineUtilization)
            .sorted()
            .collect(Collectors.toList());
        double medianGeometryRoofline = geometryUtil.isEmpty()
            ? Double.NaN
            : geometryUtil.get(geometryUtil.size() / 2);

        StringBuilder sb = new StringBuilder();
        sb.append("# GEMM PERFORMANCE ANALYSIS\n\n");
        sb.append("## Scope\n\n");
        sb.append("- Profile: `").append(config.profile).append("`\n");
        sb.append("- GEMM treated as black-box through `net.faulj.kernels.gemm.Gemm`\n");
        sb.append("- Output dataset: `").append(DEFAULT_OUTPUT_DIR.resolve(RUNS_FILE)).append("`\n\n");

        sb.append("## Hardware Roofline\n\n");
        sb.append(String.format(Locale.ROOT, "- Compute roof: %.2f GFLOP/s (`%s`)%n", roofline.peakFlopsPerSecond / 1e9, roofline.computeRoofSource));
        sb.append(String.format(Locale.ROOT, "- Memory roofs (GB/s): L1=%.2f, L2=%.2f, L3=%.2f, DRAM=%.2f%n",
            roofline.memoryBandwidths.l1BytesPerSecond / 1e9,
            roofline.memoryBandwidths.l2BytesPerSecond / 1e9,
            roofline.memoryBandwidths.l3BytesPerSecond / 1e9,
            roofline.memoryBandwidths.dramBytesPerSecond / 1e9));
        sb.append(String.format(Locale.ROOT, "- SIMD lanes (double): %d, FMA enabled: %s%n",
            roofline.hardware.simdLanesDouble, roofline.hardware.fmaEnabled));
        sb.append('\n');

        sb.append("## Key Results\n\n");
        if (best != null) {
            sb.append(String.format(Locale.ROOT,
                "- Peak observed: %.2f GFLOP/s (`%s`, m=%d n=%d k=%d, threads=%d, bound=%s)%n",
                best.measuredGflopsBest, best.caseId, best.m, best.n, best.k, best.threads, best.boundType));
        }
        sb.append(String.format(Locale.ROOT, "- Median geometry roofline utilization: %.2f%n", medianGeometryRoofline));
        sb.append(String.format(Locale.ROOT, "- Detected cache transitions: %d%n", transitions.size()));
        sb.append(String.format(Locale.ROOT, "- Optimization opportunities flagged: %d%n", opportunities.size()));
        sb.append('\n');

        sb.append("## Transition Evidence\n\n");
        if (transitions.isEmpty()) {
            sb.append("- No major cache-transition discontinuities exceeded configured thresholds.\n");
        } else {
            for (CacheTransition t : transitions) {
                sb.append(String.format(Locale.ROOT,
                    "- `%s -> %s`: regime `%s -> %s`, drop %.1f%%%n",
                    t.fromCase, t.toCase, t.fromRegime, t.toRegime, 100.0 * t.dropFraction));
            }
        }
        sb.append('\n');

        sb.append("## SIMD And Scaling Notes\n\n");
        long simdProbeCount = observations.stream().filter(o -> "simd_probe".equals(o.scenario) && o.simdEnabled).count();
        long fallbackCount = observations.stream().filter(o -> "simd_probe".equals(o.scenario) && o.simdEnabled && o.scalarFallbackSuspected).count();
        sb.append(String.format(Locale.ROOT, "- SIMD probes executed: %d%n", simdProbeCount));
        sb.append(String.format(Locale.ROOT, "- SIMD fallback-suspected probes: %d%n", fallbackCount));
        sb.append("- Parallel scaling metrics are available in `speedup_vs_1thread` and `parallel_efficiency` columns.\n\n");

        sb.append("## Next Actions\n\n");
        if (opportunities.isEmpty()) {
            sb.append("- No high-confidence issues detected from current heuristics.\n");
        } else {
            for (Opportunity o : opportunities.stream().limit(8).collect(Collectors.toList())) {
                sb.append("- [").append(o.severity.toUpperCase(Locale.ROOT)).append("] ").append(o.text).append('\n');
            }
        }
        sb.append("\n## Reproducibility\n\n");
        sb.append("- Run with `-Djlc.benchmark.mode=true`.\n");
        sb.append("- Use `scripts/run_gemm_kernel_analysis.ps1` to execute benchmark + plot pipeline.\n");
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void printSummary(AnalysisConfig config,
                                     RooflineSession roofline,
                                     List<Observation> observations,
                                     List<CacheTransition> transitions,
                                     List<Opportunity> opportunities) {
        Observation best = observations.stream()
            .max(Comparator.comparingDouble(o -> o.measuredGflopsBest))
            .orElse(null);
        System.out.println("=== GEMM Kernel Analysis ===");
        System.out.println("profile=" + config.profile);
        System.out.printf(Locale.ROOT, "compute_roof_gflops=%.3f%n", roofline.peakFlopsPerSecond / 1e9);
        System.out.printf(Locale.ROOT, "memory_roof_dram_gbps=%.3f%n", roofline.memoryBandwidths.dramBytesPerSecond / 1e9);
        if (best != null) {
            System.out.printf(Locale.ROOT,
                "best_case=%s m=%d n=%d k=%d threads=%d gflops=%.3f bound=%s%n",
                best.caseId, best.m, best.n, best.k, best.threads, best.measuredGflopsBest, best.boundType);
        }
        System.out.println("transitions=" + transitions.size());
        System.out.println("opportunities=" + opportunities.size());
        System.out.println("runs_csv=" + config.outputDir.resolve(RUNS_FILE));
        System.out.println("summary_json=" + config.outputDir.resolve(SUMMARY_FILE));
    }

    private static Map<String, Double> loadPriorBestByKey(Path runsCsv) {
        if (!Files.exists(runsCsv)) {
            return Collections.emptyMap();
        }
        try {
            List<String> lines = Files.readAllLines(runsCsv, StandardCharsets.UTF_8);
            if (lines.size() < 2) {
                return Collections.emptyMap();
            }
            String[] header = lines.get(0).split(",", -1);
            Map<String, Integer> idx = new HashMap<>();
            for (int i = 0; i < header.length; i++) {
                idx.put(header[i], i);
            }
            Integer caseIdIdx = idx.get("case_id");
            Integer scenarioIdx = idx.get("scenario");
            Integer variantIdx = idx.get("variant");
            Integer threadsIdx = idx.get("threads");
            Integer simdIdx = idx.get("simd_enabled");
            Integer gflopsIdx = idx.get("measured_gflops_best");
            if (caseIdIdx == null || scenarioIdx == null || variantIdx == null
                || threadsIdx == null || simdIdx == null || gflopsIdx == null) {
                return Collections.emptyMap();
            }
            Map<String, Double> out = new HashMap<>();
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i);
                if (line == null || line.isBlank()) {
                    continue;
                }
                String[] parts = line.split(",", -1);
                if (parts.length <= gflopsIdx) {
                    continue;
                }
                String key = safe(parts, caseIdIdx) + "|" + safe(parts, scenarioIdx) + "|"
                    + safe(parts, variantIdx) + "|" + safe(parts, threadsIdx) + "|"
                    + safe(parts, simdIdx);
                double gflops = parseDoubleSafe(safe(parts, gflopsIdx), Double.NaN);
                if (Double.isFinite(gflops) && gflops > 0.0) {
                    out.put(key, gflops);
                }
            }
            return out;
        } catch (Exception ignored) {
            return Collections.emptyMap();
        }
    }

    private static String safe(String[] arr, int idx) {
        if (idx < 0 || idx >= arr.length) {
            return "";
        }
        return arr[idx];
    }

    private static String observationKey(Observation o) {
        return o.caseId + "|" + o.scenario + "|" + o.variant + "|" + o.threads + "|" + o.simdEnabled;
    }

    private static int[] primaryThreadSet(int cores) {
        if (cores <= 1) {
            return new int[]{1};
        }
        return new int[]{1, cores};
    }

    private static int[] scalingThreadSet(int cores) {
        List<Integer> threads = new ArrayList<>();
        int t = 1;
        while (t < cores) {
            threads.add(t);
            t <<= 1;
        }
        if (threads.isEmpty() || threads.get(threads.size() - 1) != cores) {
            threads.add(cores);
        }
        return threads.stream().mapToInt(Integer::intValue).toArray();
    }

    private static List<GemmCase> buildGeometryCases(AnalysisConfig config) {
        List<GemmCase> out = new ArrayList<>();
        int[] square;
        if ("full".equals(config.profile)) {
            square = new int[]{16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536};
        } else if ("quick".equals(config.profile)) {
            square = new int[]{32, 64, 96, 128, 192, 256, 384, 512};
        } else {
            square = new int[]{24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
        }
        for (int n : square) {
            out.add(new GemmCase("square_n" + n, "square", n, n, n, 1000L + n, 2000L + n));
        }

        int[] axisSizes = "quick".equals(config.profile)
            ? new int[]{64, 128, 256, 512}
            : new int[]{64, 128, 256, 512, 1024};
        for (int m : axisSizes) {
            out.add(new GemmCase("m_sweep_m" + m, "m_sweep", m, 256, 256, 3000L + m, 4000L + m));
        }
        for (int n : axisSizes) {
            out.add(new GemmCase("n_sweep_n" + n, "n_sweep", 256, n, 256, 5000L + n, 6000L + n));
        }
        for (int k : axisSizes) {
            out.add(new GemmCase("k_sweep_k" + k, "k_sweep", 256, 256, k, 7000L + k, 8000L + k));
        }

        out.add(new GemmCase("tall_skinny_2048x64x256", "tall_skinny", 2048, 64, 256, 9001L, 9002L));
        out.add(new GemmCase("tall_skinny_4096x64x128", "tall_skinny", 4096, 64, 128, 9003L, 9004L));
        out.add(new GemmCase("wide_short_64x2048x256", "wide_short", 64, 2048, 256, 9011L, 9012L));
        out.add(new GemmCase("wide_short_64x4096x128", "wide_short", 64, 4096, 128, 9013L, 9014L));
        out.add(new GemmCase("k_heavy_256x256x2048", "k_heavy", 256, 256, 2048, 9021L, 9022L));
        out.add(new GemmCase("k_light_256x256x32", "k_light", 256, 256, 32, 9023L, 9024L));

        return out;
    }
    private static List<GemmCase> buildScalingCases(AnalysisConfig config) {
        List<GemmCase> out = new ArrayList<>();
        out.add(new GemmCase("scale_square_512", "scaling_square", 512, 512, 512, 10001L, 10002L));
        out.add(new GemmCase("scale_square_1024", "scaling_square", 1024, 1024, 1024, 10003L, 10004L));
        out.add(new GemmCase("scale_tall_2048x64x256", "scaling_tall", 2048, 64, 256, 10005L, 10006L));
        out.add(new GemmCase("scale_k_heavy_256x256x2048", "scaling_k_heavy", 256, 256, 2048, 10007L, 10008L));
        if ("full".equals(config.profile)) {
            out.add(new GemmCase("scale_square_1536", "scaling_square", 1536, 1536, 1536, 10009L, 10010L));
        }
        return out;
    }

    private static List<GemmCase> buildSimdProbeCases(AnalysisConfig config) {
        List<GemmCase> out = new ArrayList<>();
        int[] sizes = "quick".equals(config.profile)
            ? new int[]{127, 128, 129, 255, 256, 257}
            : new int[]{127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513};
        for (int n : sizes) {
            out.add(new GemmCase("simd_square_n" + n, "simd_probe", n, n, n, 11000L + n, 12000L + n));
        }
        return out;
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextDouble() - 0.5;
        }
        return Matrix.wrap(data, rows, cols);
    }

    private static double threadScaledComputeRoof(RooflineSession roofline, int threads) {
        int cores = Math.max(1, roofline.hardware.cores);
        double perCore = roofline.rawTheoreticalPeakFlopsPerSecond / cores;
        return perCore * Math.min(threads, cores);
    }

    private static String classifyShape(int m, int n, int k) {
        if (m == n && n == k) {
            return "square";
        }
        if (m > n * 2) {
            return "tall";
        }
        if (n > m * 2) {
            return "wide";
        }
        if (k > Math.max(m, n) * 2) {
            return "k_heavy";
        }
        return "rectangular";
    }

    private static String classifyCacheRegime(double workingSetBytes, MemoryBandwidthProbe.BandwidthHierarchy h) {
        if (workingSetBytes <= h.l1SizeBytes) {
            return "l1";
        }
        if (workingSetBytes <= h.l2SizeBytes) {
            return "l2";
        }
        if (workingSetBytes <= h.l3SizeBytes) {
            return "l3";
        }
        return "dram";
    }

    private static double estimatedCycles(double seconds, double ghz, int threads) {
        return seconds * ghz * 1e9 * Math.max(1, threads);
    }

    private static int positiveRemainder(int value, int base) {
        int b = Math.max(1, base);
        int r = value % b;
        return r < 0 ? r + b : r;
    }

    private static int roundUp(int value, int block) {
        int b = Math.max(1, block);
        return ((value + b - 1) / b) * b;
    }

    private static double safeDivide(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b) || b == 0.0) {
            return 0.0;
        }
        return a / b;
    }

    private static double clamp01(double v) {
        if (!Double.isFinite(v) || v < 0.0) {
            return 0.0;
        }
        return Math.min(1.0, v);
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String intArrayJson(int[] values) {
        return Arrays.stream(values).mapToObj(String::valueOf).collect(Collectors.joining(",", "[", "]"));
    }

    private static void appendJsonField(StringBuilder sb, String key, String value, boolean comma) {
        appendJsonField(sb, key, value, comma, 2);
    }

    private static void appendJsonField(StringBuilder sb, String key, String value, boolean comma, int indentSpaces) {
        indent(sb, indentSpaces).append('"').append(escapeJson(key)).append('"').append(": \"")
            .append(escapeJson(value)).append('"');
        if (comma) {
            sb.append(',');
        }
        sb.append('\n');
    }

    private static void appendJsonField(StringBuilder sb, String key, double value, boolean comma, int indentSpaces) {
        indent(sb, indentSpaces).append('"').append(escapeJson(key)).append('"').append(": ")
            .append(formatNumber(value));
        if (comma) {
            sb.append(',');
        }
        sb.append('\n');
    }

    private static void appendJsonField(StringBuilder sb, String key, int value, boolean comma, int indentSpaces) {
        indent(sb, indentSpaces).append('"').append(escapeJson(key)).append('"').append(": ").append(value);
        if (comma) {
            sb.append(',');
        }
        sb.append('\n');
    }

    private static void appendJsonField(StringBuilder sb, String key, boolean value, boolean comma, int indentSpaces) {
        indent(sb, indentSpaces).append('"').append(escapeJson(key)).append('"').append(": ").append(value);
        if (comma) {
            sb.append(',');
        }
        sb.append('\n');
    }

    private static StringBuilder indent(StringBuilder sb, int spaces) {
        for (int i = 0; i < spaces; i++) {
            sb.append(' ');
        }
        return sb;
    }

    private static void csv(StringBuilder sb, Object value) {
        if (value == null) {
            sb.append(',');
            return;
        }
        if (value instanceof Number) {
            sb.append(formatNumber(((Number) value).doubleValue())).append(',');
            return;
        }
        if (value instanceof Boolean) {
            sb.append(value).append(',');
            return;
        }
        String s = String.valueOf(value);
        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0) {
            sb.append('"').append(s.replace("\"", "\"\"")).append('"').append(',');
        } else {
            sb.append(s).append(',');
        }
    }

    private static String formatNumber(double value) {
        if (!Double.isFinite(value)) {
            return "NaN";
        }
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static double parseDoubleSafe(String raw, double fallback) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static String readPropertyOrEnv(String propertyKey, String envKey, String fallback) {
        String byProperty = System.getProperty(propertyKey);
        if (byProperty != null && !byProperty.isBlank()) {
            return byProperty;
        }
        String byEnv = System.getenv(envKey);
        if (byEnv != null && !byEnv.isBlank()) {
            return byEnv;
        }
        return fallback;
    }

    private static final class AnalysisConfig {
        final String profile;
        final Path outputDir;
        final int forcedWarmupRuns;
        final int forcedMeasurementRuns;
        final double regressionThresholdFraction;

        AnalysisConfig(String profile,
                       Path outputDir,
                       int forcedWarmupRuns,
                       int forcedMeasurementRuns,
                       double regressionThresholdFraction) {
            this.profile = profile;
            this.outputDir = outputDir;
            this.forcedWarmupRuns = forcedWarmupRuns;
            this.forcedMeasurementRuns = forcedMeasurementRuns;
            this.regressionThresholdFraction = regressionThresholdFraction;
        }

        static AnalysisConfig fromSystemProperties() {
            String profileRaw = readPropertyOrEnv("jlc.gemm.analysis.profile", "JLC_GEMM_ANALYSIS_PROFILE", "standard")
                .trim().toLowerCase(Locale.ROOT);
            String profile = switch (profileRaw) {
                case "quick", "standard", "full" -> profileRaw;
                default -> "standard";
            };
            Path outputDir = Paths.get(readPropertyOrEnv("jlc.gemm.analysis.output_dir", "JLC_GEMM_ANALYSIS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR.toString()));
            int forcedWarmup = parseNonNegativeInt(readPropertyOrEnv("jlc.gemm.analysis.warmup_runs", "JLC_GEMM_ANALYSIS_WARMUP_RUNS", null), -1);
            int forcedRuns = parsePositiveInt(readPropertyOrEnv("jlc.gemm.analysis.measurement_runs", "JLC_GEMM_ANALYSIS_MEASUREMENT_RUNS", null), -1);
            double regression = parsePositiveDouble(readPropertyOrEnv("jlc.gemm.analysis.regression_threshold", "JLC_GEMM_ANALYSIS_REGRESSION_THRESHOLD", null), 0.05);
            return new AnalysisConfig(profile, outputDir, forcedWarmup, forcedRuns, regression);
        }
    }

    private static final class IterationPlan {
        final int warmupRuns;
        final int measurementRuns;

        IterationPlan(int warmupRuns, int measurementRuns) {
            this.warmupRuns = warmupRuns;
            this.measurementRuns = measurementRuns;
        }

        static IterationPlan forFlops(double flops, AnalysisConfig config) {
            if (config.forcedWarmupRuns >= 0 && config.forcedMeasurementRuns > 0) {
                return new IterationPlan(config.forcedWarmupRuns, config.forcedMeasurementRuns);
            }
            if (flops >= 8e9) {
                return new IterationPlan(1, 2);
            }
            if (flops >= 1e9) {
                return new IterationPlan(1, 3);
            }
            if (flops >= 1e8) {
                return new IterationPlan(2, 4);
            }
            return new IterationPlan(3, 6);
        }
    }

    private static final class GemmCase {
        final String id;
        final String family;
        final int m;
        final int n;
        final int k;
        final long seedA;
        final long seedB;

        GemmCase(String id, String family, int m, int n, int k, long seedA, long seedB) {
            this.id = id;
            this.family = family;
            this.m = m;
            this.n = n;
            this.k = k;
            this.seedA = seedA;
            this.seedB = seedB;
        }

        double flops() {
            return 2.0 * m * (double) n * k;
        }

        double workingSetBytes() {
            return Double.BYTES * (m * (double) k + k * (double) n + m * (double) n);
        }

        double minBytes() {
            return workingSetBytes();
        }

        double modeledBytes() {
            return workingSetBytes() + Double.BYTES * (m * (double) k + k * (double) n);
        }
    }

    private static final class RunStats {
        final double bestSeconds;
        final double meanSeconds;
        final double medianSeconds;
        final double p95Seconds;
        final double stdevSeconds;
        final double cv;

        RunStats(double bestSeconds,
                 double meanSeconds,
                 double medianSeconds,
                 double p95Seconds,
                 double stdevSeconds,
                 double cv) {
            this.bestSeconds = bestSeconds;
            this.meanSeconds = meanSeconds;
            this.medianSeconds = medianSeconds;
            this.p95Seconds = p95Seconds;
            this.stdevSeconds = stdevSeconds;
            this.cv = cv;
        }

        static RunStats of(double[] seconds) {
            double[] copy = Arrays.copyOf(seconds, seconds.length);
            Arrays.sort(copy);
            double best = copy[0];
            double mean = Arrays.stream(copy).average().orElse(0.0);
            double median = percentile(copy, 50.0);
            double p95 = percentile(copy, 95.0);
            double variance = 0.0;
            for (double s : copy) {
                double d = s - mean;
                variance += d * d;
            }
            variance = variance / Math.max(1, copy.length - 1);
            double stdev = Math.sqrt(Math.max(0.0, variance));
            double cv = safeDivide(stdev, Math.max(1e-12, mean));
            return new RunStats(best, mean, median, p95, stdev, cv);
        }

        private static double percentile(double[] sorted, double p) {
            if (sorted.length == 0) {
                return 0.0;
            }
            double rank = (p / 100.0) * (sorted.length - 1);
            int low = (int) Math.floor(rank);
            int high = (int) Math.ceil(rank);
            if (low == high) {
                return sorted[low];
            }
            double t = rank - low;
            return sorted[low] * (1.0 - t) + sorted[high] * t;
        }
    }

    private static final class Observation {
        String timestampUtc;
        String caseId;
        String family;
        String scenario;
        String variant;
        String kernel;
        int m;
        int n;
        int k;
        String shape;
        int threads;
        boolean simdEnabled;
        String dispatchAlgorithm;
        int warmupRuns;
        int measurementRuns;

        double flops;
        double bytesMin;
        double bytesModeled;
        double arithmeticIntensityMin;
        double arithmeticIntensityModeled;
        double workingSetBytes;
        String cacheRegime;

        double bestSeconds;
        double meanSeconds;
        double medianSeconds;
        double p95Seconds;
        double stdevSeconds;
        double cv;

        double measuredGflopsBest;
        double measuredGflopsMedian;
        double measuredGflopsMean;
        double measuredBandwidthGbps;

        double computeRoofGflops;
        double memoryRoofGflops;
        double effectiveRoofGflops;
        String boundType;
        double computeUtilization;
        double memoryUtilization;
        double rooflineUtilization;
        double rooflineGapGflops;
        double selectedMemoryRoofGbps;
        String selectedMemoryLevel;

        double estimatedCycles;
        double flopsPerCycle;
        double perCoreFlopsPerCycle;
        double fmaUtilizationProxy;
        double loadStoreBytesPerFlop;

        int simdLanesDouble;
        int mr;
        int nr;
        int kc;
        int mc;
        int nc;
        int remainderM;
        int remainderN;
        int remainderK;
        double simdLaneOccupancy;
        double vectorWidthUtilization;
        boolean scalarFallbackSuspected;

        double l1MissRate;
        double l2MissRate;
        double l3MissRate;
        double instructionsPerCycle;
        double simdInstructionRatio;
        String counterSource;

        double speedupVsOneThread;
        double parallelEfficiency;
        double simdSpeedupVsScalar;
        double simdEfficiencyVsLanes;

        double regressionVsPreviousFraction;
        boolean regressionFlag;
    }

    private static final class CacheTransition {
        String fromCase;
        String toCase;
        String fromRegime;
        String toRegime;
        double fromWorkingSetBytes;
        double toWorkingSetBytes;
        double fromGflops;
        double toGflops;
        double dropFraction;
        String reason;
    }

    private static final class Opportunity {
        String severity;
        String caseId;
        String text;
    }

    private static int parsePositiveInt(String value, int fallback) {
        if (value == null) {
            return fallback;
        }
        try {
            int v = Integer.parseInt(value.trim());
            return v > 0 ? v : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static int parseNonNegativeInt(String value, int fallback) {
        if (value == null) {
            return fallback;
        }
        try {
            int v = Integer.parseInt(value.trim());
            return v >= 0 ? v : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static double parsePositiveDouble(String value, double fallback) {
        if (value == null) {
            return fallback;
        }
        try {
            double v = Double.parseDouble(value.trim());
            return v > 0.0 ? v : fallback;
        } catch (Exception ignored) {
            return fallback;
        }
    }
}
