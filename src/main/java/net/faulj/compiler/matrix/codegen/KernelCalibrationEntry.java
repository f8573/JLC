package net.faulj.compiler.matrix.codegen;

import java.util.List;

/** One exact-shape calibration decision and all candidate evidence. */
public record KernelCalibrationEntry(
    String kernelSha256,
    String kernelSignature,
    String baselineVariant,
    String winnerVariant,
    double winnerSpeedup,
    String decisionReason,
    List<KernelCalibrationCandidate> candidates,
    List<KernelTrustedBaseline> trustedBaselines
) {
    public KernelCalibrationEntry(String kernelSha256,
                                  String kernelSignature,
                                  String baselineVariant,
                                  String winnerVariant,
                                  double winnerSpeedup,
                                  String decisionReason,
                                  List<KernelCalibrationCandidate> candidates) {
        this(kernelSha256, kernelSignature, baselineVariant, winnerVariant, winnerSpeedup,
            decisionReason, candidates, List.of());
    }

    public KernelCalibrationEntry {
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        trustedBaselines = trustedBaselines == null ? List.of() : List.copyOf(trustedBaselines);
        decisionReason = decisionReason == null ? "" : decisionReason;
    }

    public static KernelCalibrationEntry from(KernelTuningResult result) {
        return new KernelCalibrationEntry(
            result.kernelSignature().sha256(),
            result.kernelSignature().canonicalText(),
            result.baseline().canonicalText(),
            result.winner() == null ? null : result.winner().canonicalText(),
            result.winnerSpeedup(), result.decisionReason(),
            result.candidates().stream().map(KernelCalibrationCandidate::from).toList(),
            result.trustedBaselines());
    }
}
