package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Correctness-first, end-to-end selector for typed backend choices.
 *
 * <p>Correctness is supplied by the caller because Java and generated native
 * choices often need different fixtures.  Invalid choices are retained as
 * evidence but are never timed.</p>
 */
public final class BackendCalibrationRunner {
    private BackendCalibrationRunner() {
    }

    public static BackendCalibrationResult calibrate(
        KernelSignature signature,
        String workloadBucket,
        BackendChoice baseline,
        List<BackendCalibrationCandidate> candidates,
        BackendBenchmarkRunner benchmarkRunner,
        KernelTuningConfig config) {
        Objects.requireNonNull(signature, "Kernel signature");
        Objects.requireNonNull(baseline, "Calibration baseline");
        Objects.requireNonNull(benchmarkRunner, "Backend benchmark runner");
        Objects.requireNonNull(config, "Tuning config");
        if (candidates == null || candidates.isEmpty()) {
            throw new IllegalArgumentException("At least one backend candidate is required");
        }

        List<BackendCandidateEvidence> evidence = new ArrayList<>();
        for (BackendCalibrationCandidate candidate : deduplicate(candidates)) {
            long started = System.nanoTime();
            if (!candidate.correctnessPassed()) {
                evidence.add(new BackendCandidateEvidence(candidate.choice(),
                    KernelCandidateOutcome.CORRECTNESS_FAILED, false, false, null,
                    0L, candidate.reason()));
                continue;
            }
            try {
                KernelBenchmarkStatistics statistics = benchmarkRunner.measure(
                    candidate.choice(), config);
                boolean stable = statistics != null
                    && statistics.stable(config.relativeMadThreshold());
                evidence.add(new BackendCandidateEvidence(candidate.choice(),
                    stable ? KernelCandidateOutcome.PASS : KernelCandidateOutcome.NOISY,
                    true, stable, statistics, elapsed(started),
                    stable ? candidate.reason() : "timing dispersion exceeded stability threshold"));
            } catch (Exception | LinkageError failure) {
                evidence.add(new BackendCandidateEvidence(candidate.choice(),
                    KernelCandidateOutcome.BENCHMARK_FAILED, true, false, null,
                    elapsed(started), "benchmark failed: " + message(failure)));
            }
        }

        List<BackendCandidateEvidence> stable = evidence.stream()
            .filter(BackendCandidateEvidence::valid)
            .sorted(Comparator
                .<BackendCandidateEvidence>comparingDouble(item -> item.statistics().medianNanos())
                .thenComparing(item -> item.choice().tag()))
            .toList();
        BackendCandidateEvidence baselineEvidence = evidence.stream()
            .filter(item -> item.choice().equals(baseline))
            .findFirst().orElse(null);
        BackendCandidateEvidence fastest = stable.isEmpty() ? null : stable.get(0);
        BackendCandidateEvidence selected = selectWithPromotion(
            fastest, baselineEvidence, config.promotionThreshold());
        BackendSelectionStatus status = selected == null
            ? BackendSelectionStatus.NO_STABLE_WINNER : BackendSelectionStatus.CALIBRATED;
        BackendChoice winner = selected == null ? null : selected.choice();
        double speedup = speedup(baselineEvidence, selected);
        String reason;
        if (selected == null) {
            reason = "no stable correctness-gated backend; use fallback baseline";
        } else if (selected.choice().equals(baseline)) {
            reason = "baseline retained after end-to-end calibration";
        } else {
            reason = "selected fastest stable correctness-gated backend";
        }
        return new BackendCalibrationResult(signature, workloadBucket, baseline, evidence,
            status, winner, speedup, selected != null && selected.valid(), reason);
    }

    /** Convenience seam for tests and callers with pre-built invocations. */
    public static BackendCalibrationResult calibrate(
        KernelSignature signature,
        String workloadBucket,
        BackendChoice baseline,
        List<BackendCalibrationCandidate> candidates,
        Map<BackendChoice, BackendInvocation> invocations,
        KernelTuningConfig config) {
        Map<BackendChoice, BackendInvocation> safe = invocations == null
            ? Map.of() : Map.copyOf(invocations);
        return calibrate(signature, workloadBucket, baseline, candidates,
            (choice, tuningConfig) -> {
                BackendInvocation invocation = safe.get(choice);
                if (invocation == null) {
                    throw new IllegalStateException("no invocation for " + choice.tag());
                }
                return KernelBenchmark.measure(invocation, tuningConfig);
            }, config);
    }

    private static List<BackendCalibrationCandidate> deduplicate(
        List<BackendCalibrationCandidate> candidates) {
        Map<BackendChoice, BackendCalibrationCandidate> unique = new LinkedHashMap<>();
        for (BackendCalibrationCandidate candidate : candidates) {
            if (candidate == null) {
                throw new IllegalArgumentException("Backend candidate must not be null");
            }
            unique.putIfAbsent(candidate.choice(), candidate);
        }
        return List.copyOf(unique.values());
    }

    private static BackendCandidateEvidence selectWithPromotion(
        BackendCandidateEvidence fastest,
        BackendCandidateEvidence baseline,
        double promotionThreshold) {
        if (fastest == null) return null;
        if (baseline == null || !baseline.valid()
            || fastest.choice().equals(baseline.choice())) {
            return baseline == null || !baseline.valid() ? null : fastest;
        }
        double speedup = baseline.statistics().medianNanos()
            / fastest.statistics().medianNanos();
        return speedup >= 1.0 + promotionThreshold ? fastest : baseline;
    }

    private static double speedup(BackendCandidateEvidence baseline,
                                  BackendCandidateEvidence selected) {
        if (baseline == null || selected == null || baseline.statistics() == null
            || selected.statistics() == null || selected.statistics().medianNanos() <= 0.0) {
            return 0.0;
        }
        return baseline.statistics().medianNanos() / selected.statistics().medianNanos();
    }

    private static long elapsed(long started) {
        return Math.max(0L, System.nanoTime() - started);
    }

    private static String message(Throwable failure) {
        return failure.getMessage() == null ? failure.getClass().getSimpleName()
            : failure.getMessage();
    }
}
