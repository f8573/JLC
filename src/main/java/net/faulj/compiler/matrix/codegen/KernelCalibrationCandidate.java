package net.faulj.compiler.matrix.codegen;

import java.util.List;

/** JSON-friendly evidence projection for one calibrated candidate. */
public record KernelCalibrationCandidate(
    String variantId,
    String variantSignature,
    String backend,
    int unroll,
    String loopForm,
    int estimatedLiveVectorRegisters,
    String outcome,
    boolean correctnessPassed,
    boolean stable,
    String reason,
    Double medianNanos,
    Double minNanos,
    Double maxNanos,
    Double madNanos,
    List<Long> rawSamplesNanos,
    long sourceGenerationNanos,
    long compilationNanos,
    long benchmarkNanos,
    long compiledTextBytes
) {
    public KernelCalibrationCandidate {
        rawSamplesNanos = rawSamplesNanos == null ? List.of() : List.copyOf(rawSamplesNanos);
        reason = reason == null ? "" : reason;
    }

    public static KernelCalibrationCandidate from(KernelCandidateEvidence evidence) {
        KernelBenchmarkStatistics stats = evidence.statistics();
        return new KernelCalibrationCandidate(
            evidence.variantSignature().variantId(),
            evidence.variantSignature().canonicalText(),
            evidence.variantSignature().backend().name(),
            evidence.variantSignature().unroll(),
            evidence.variantSignature().loopForm().name(),
            evidence.candidate().estimatedLiveVectorRegisters(),
            evidence.outcome().name(),
            evidence.correctnessPassed(), evidence.stable(), evidence.reason(),
            stats == null ? null : stats.medianNanos(),
            stats == null ? null : stats.minNanos(),
            stats == null ? null : stats.maxNanos(),
            stats == null ? null : stats.madNanos(),
            stats == null ? List.of() : stats.rawSamplesNanos(),
            evidence.sourceGenerationNanos(), evidence.compilationNanos(),
            evidence.benchmarkNanos(), evidence.compiledTextBytes());
    }
}
