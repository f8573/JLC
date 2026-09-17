package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.kernel.KernelBinding;

/**
 * Correctness-first bounded selector for generated kernel variants.
 *
 * <p>This class is explicitly invoked by a calibration harness. It does not
 * discover compilers, write profiles, or run from an application execution
 * thread.</p>
 */
public final class KernelVariantTuner {
    private KernelVariantTuner() {
    }

    public static KernelTuningResult tune(PseudokernelPlan plan,
                                          KernelVariantCompiler compiler,
                                          List<KernelBinding> validationCorpus,
                                          KernelBinding benchmarkBinding,
                                          KernelTuningConfig config) {
        if (plan == null || compiler == null || config == null) {
            throw new IllegalArgumentException("Tuning plan, compiler, and config are required");
        }
        return tune(plan, compiler, validationCorpus, benchmarkBinding, config,
            (variant, binding, tuningConfig) -> KernelBenchmark.measure(
                variant.invoker(), binding, tuningConfig), List.of());
    }

    public static KernelTuningResult tune(PseudokernelPlan plan,
                                          KernelVariantCompiler compiler,
                                          List<KernelBinding> validationCorpus,
                                          KernelBinding benchmarkBinding,
                                          KernelTuningConfig config,
                                          KernelBenchmarkRunner benchmarkRunner) {
        return tune(plan, compiler, validationCorpus, benchmarkBinding, config,
            benchmarkRunner, List.of());
    }

    public static KernelTuningResult tune(PseudokernelPlan plan,
                                          KernelVariantCompiler compiler,
                                          List<KernelBinding> validationCorpus,
                                          KernelBinding benchmarkBinding,
                                          KernelTuningConfig config,
                                          KernelBenchmarkRunner benchmarkRunner,
                                          List<KernelTrustedBaseline> trustedBaselines) {
        if (plan == null || compiler == null || config == null || benchmarkRunner == null) {
            throw new IllegalArgumentException("Tuning plan, compiler, config, and benchmark runner are required");
        }
        List<KernelBinding> corpus = validationCorpus == null
            ? KernelValidationCorpus.forPlan(plan) : List.copyOf(validationCorpus);
        if (corpus.isEmpty()) {
            throw new IllegalArgumentException("At least one correctness fixture is required");
        }
        KernelBinding benchmark = benchmarkBinding == null ? corpus.get(0) : benchmarkBinding;
        List<KernelVariantCandidate> candidates = KernelVariantGenerator.enumerate(
            plan, OptimizationSemantics.STRICT, config.maxCandidates());
        List<KernelCandidateEvidence> evidence = new ArrayList<>();
        long sessionStart = System.nanoTime();
        KernelVariantSignature expectedBaseline = KernelVariantSignature.baselineAvx2(
            plan.signature());
        for (KernelVariantCandidate candidate : candidates) {
            if (candidate.isPruned()) {
                evidence.add(new KernelCandidateEvidence(candidate,
                    KernelCandidateOutcome.PRUNED, null, null, 0L, 0L, 0L, 0L,
                    candidate.reason()));
                continue;
            }
            CompiledKernelVariant compiled;
            try {
                compiled = compiler.compile(plan, candidate);
                if (compiled == null) {
                    throw new IllegalStateException("compiler returned null");
                }
            } catch (Exception | LinkageError failure) {
                evidence.add(new KernelCandidateEvidence(candidate,
                    KernelCandidateOutcome.COMPILATION_FAILED, null, null, 0L, 0L, 0L, 0L,
                    message(failure)));
                continue;
            }

            KernelCorrectnessResult correctness = KernelCorrectnessGate.validate(
                plan, compiled.invoker(), corpus);
            if (!correctness.passed()) {
                evidence.add(new KernelCandidateEvidence(candidate,
                    KernelCandidateOutcome.CORRECTNESS_FAILED, correctness, null,
                    compiled.sourceGenerationNanos(), compiled.compilationNanos(), 0L,
                    compiled.compiledTextBytes(), correctness.diagnostic()));
                continue;
            }

            long benchmarkStart = System.nanoTime();
            try {
                KernelBenchmarkStatistics statistics = benchmarkRunner.measure(
                    compiled, benchmark, config);
                long benchmarkNanos = System.nanoTime() - benchmarkStart;
                boolean stable = statistics.stable(config.relativeMadThreshold());
                evidence.add(new KernelCandidateEvidence(candidate,
                    stable ? KernelCandidateOutcome.PASS : KernelCandidateOutcome.NOISY,
                    correctness, statistics, compiled.sourceGenerationNanos(),
                    compiled.compilationNanos(), benchmarkNanos, compiled.compiledTextBytes(),
                    stable ? "stable" : "relative MAD exceeds noise threshold"));
            } catch (Exception | LinkageError failure) {
                evidence.add(new KernelCandidateEvidence(candidate,
                    KernelCandidateOutcome.BENCHMARK_FAILED, correctness, null,
                    compiled.sourceGenerationNanos(), compiled.compilationNanos(),
                    System.nanoTime() - benchmarkStart, compiled.compiledTextBytes(),
                    message(failure)));
            }
            if (config.maxBenchmarkMillis() > 0L
                && System.nanoTime() - sessionStart >= config.maxBenchmarkMillis() * 1_000_000L) {
                // Always give the trusted generated baseline a chance to establish
                // the comparison point before honoring the session deadline.
                if (candidate.signature().equals(expectedBaseline)
                    || evidence.stream().anyMatch(item ->
                        item.variantSignature().equals(expectedBaseline))) {
                    break;
                }
            }
        }

        KernelVariantSignature baseline = chooseBaseline(plan, evidence);
        KernelCandidateEvidence baselineEvidence = find(evidence, baseline);
        List<KernelCandidateEvidence> stable = evidence.stream()
            .filter(item -> item.outcome() == KernelCandidateOutcome.PASS
                && item.statistics() != null)
            .toList();
        KernelCandidateEvidence fastest = stable.stream()
            .min(Comparator.comparingDouble((KernelCandidateEvidence item)
                    -> item.statistics().medianNanos())
                .thenComparing((KernelCandidateEvidence item)
                    -> item.variantSignature().variantId()))
            .orElse(null);

        KernelVariantSignature winner = null;
        double speedup = 0.0;
        String reason;
        if (fastest == null) {
            reason = "no correctness-passing stable candidate; use fallback baseline";
        } else if (baselineEvidence == null
            || baselineEvidence.outcome() != KernelCandidateOutcome.PASS
            || baselineEvidence.statistics() == null) {
            reason = "trusted baseline was not stable; use fallback baseline";
        } else {
            double baselineMedian = baselineEvidence.statistics().medianNanos();
            speedup = baselineMedian / fastest.statistics().medianNanos();
            if (!Double.isFinite(speedup) || speedup < 1.0 + config.promotionThreshold()) {
                winner = baseline;
                speedup = 1.0;
                reason = "baseline retained after stable end-to-end calibration";
            } else {
                winner = fastest.variantSignature();
                reason = "lowest stable median exceeded promotion threshold";
            }
        }
        return new KernelTuningResult(plan.signature(), baseline, evidence, winner, speedup,
            reason, retainTrustedBaselines(trustedBaselines, evidence, baseline));
    }

    private static KernelVariantSignature chooseBaseline(PseudokernelPlan plan,
                                                          List<KernelCandidateEvidence> evidence) {
        KernelVariantSignature avx2 = KernelVariantSignature.baselineAvx2(plan.signature());
        if (plan.avx2Eligible() && evidence.stream().anyMatch(item ->
            item.variantSignature().equals(avx2) && item.candidate().eligible())) {
            return avx2;
        }
        return evidence.stream()
            .map(KernelCandidateEvidence::variantSignature)
            .filter(item -> item.backend() == CodegenBackend.SCALAR_CPP)
            .findFirst()
            .orElse(avx2);
    }

    private static KernelCandidateEvidence find(List<KernelCandidateEvidence> evidence,
                                                KernelVariantSignature signature) {
        return evidence.stream()
            .filter(item -> item.variantSignature().equals(signature))
            .findFirst().orElse(null);
    }

    private static List<KernelTrustedBaseline> retainTrustedBaselines(
            List<KernelTrustedBaseline> supplied,
            List<KernelCandidateEvidence> evidence,
            KernelVariantSignature baseline) {
        List<KernelTrustedBaseline> result = new ArrayList<>(
            supplied == null ? List.of() : supplied);
        Set<String> names = result.stream().map(KernelTrustedBaseline::name)
            .collect(Collectors.toSet());
        if (!names.contains("R2_JAVA_FUSED")) {
            result.add(new KernelTrustedBaseline("R2_JAVA_FUSED", "java", true, null,
                "trusted semantic baseline; end-to-end timing is supplied by the crossover harness"));
        }
        addGeneratedBaseline(result, names, evidence, CodegenBackend.SCALAR_CPP,
            "R4_GENERATED_SCALAR_CPP", "scalar_cpp",
            "fixed generated scalar C++ baseline");
        addGeneratedBaseline(result, names, evidence, CodegenBackend.AVX2,
            "R4_BASELINE_AVX2", "avx2",
            "fixed R4 AVX2 u1 flat reference variant", baseline);
        return List.copyOf(result);
    }

    private static void addGeneratedBaseline(List<KernelTrustedBaseline> result,
                                             Set<String> names,
                                             List<KernelCandidateEvidence> evidence,
                                             CodegenBackend backend,
                                             String name,
                                             String backendName,
                                             String note) {
        addGeneratedBaseline(result, names, evidence, backend, name, backendName, note, null);
    }

    private static void addGeneratedBaseline(List<KernelTrustedBaseline> result,
                                             Set<String> names,
                                             List<KernelCandidateEvidence> evidence,
                                             CodegenBackend backend,
                                             String name,
                                             String backendName,
                                             String note,
                                             KernelVariantSignature preferred) {
        if (names.contains(name)) return;
        KernelCandidateEvidence selected = preferred == null ? evidence.stream()
            .filter(item -> item.variantSignature().backend() == backend)
            .findFirst().orElse(null) : find(evidence, preferred);
        result.add(new KernelTrustedBaseline(name, backendName,
            selected != null && selected.correctnessPassed(),
            selected == null ? null : selected.statistics(),
            selected == null ? note + "; unavailable in this session" : note));
        names.add(name);
    }

    private static String message(Throwable failure) {
        return failure.getMessage() == null
            ? failure.getClass().getSimpleName() : failure.getMessage();
    }
}
