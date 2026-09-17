package net.faulj.compiler.matrix.codegen;

import java.util.List;
import java.util.Objects;

/** One exact semantic-kernel calibration decision and all candidate evidence. */
public final class KernelCalibrationEntry {
    private final String kernelSha256;
    private final String kernelSignature;
    private final String workloadBucket;
    private final String baselineVariant;
    private final String winnerVariant;
    private final double winnerSpeedup;
    private final String decisionReason;
    private final List<KernelCalibrationCandidate> candidates;
    private final List<KernelTrustedBaseline> trustedBaselines;
    private final BackendChoice baselineChoice;
    private final BackendChoice winnerChoice;
    private final List<BackendCandidateEvidence> backendCandidates;
    private final BackendSelectionStatus selectionStatus;
    private final boolean stable;

    /** Existing R5 constructor retained for profile/API compatibility. */
    public KernelCalibrationEntry(String kernelSha256,
                                  String kernelSignature,
                                  String baselineVariant,
                                  String winnerVariant,
                                  double winnerSpeedup,
                                  String decisionReason,
                                  List<KernelCalibrationCandidate> candidates,
                                  List<KernelTrustedBaseline> trustedBaselines) {
        this(kernelSha256, kernelSignature, "exact", baselineVariant, winnerVariant,
            legacyWinnerSpeedup(winnerVariant, winnerSpeedup), decisionReason, candidates,
            trustedBaselines,
            inferChoice(kernelSignature, baselineVariant),
            inferChoice(kernelSignature, winnerVariant), List.of(),
            legacyStatus(winnerVariant), winnerVariant != null);
    }

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

    /** Typed profile entry used by the production backend calibration pass. */
    public KernelCalibrationEntry(String kernelSha256,
                                  String kernelSignature,
                                  String workloadBucket,
                                  BackendChoice baselineChoice,
                                  BackendChoice winnerChoice,
                                  double winnerSpeedup,
                                  String decisionReason,
                                  List<BackendCandidateEvidence> backendCandidates) {
        this(kernelSha256, kernelSignature, workloadBucket,
            legacyVariant(baselineChoice), legacyVariant(winnerChoice), winnerSpeedup,
            decisionReason, List.of(), List.of(), baselineChoice, winnerChoice,
            backendCandidates, winnerChoice == null
                ? BackendSelectionStatus.NO_STABLE_WINNER : BackendSelectionStatus.CALIBRATED,
            winnerChoice != null);
    }

    public KernelCalibrationEntry(KernelSignature signature,
                                  String workloadBucket,
                                  BackendChoice baselineChoice,
                                  BackendChoice winnerChoice,
                                  double winnerSpeedup,
                                  String decisionReason,
                                  List<BackendCandidateEvidence> backendCandidates) {
        this(Objects.requireNonNull(signature, "Kernel signature").sha256(),
            signature.canonicalText(), workloadBucket, baselineChoice, winnerChoice,
            winnerSpeedup, decisionReason, backendCandidates);
    }

    private KernelCalibrationEntry(String kernelSha256,
                                   String kernelSignature,
                                   String workloadBucket,
                                   String baselineVariant,
                                   String winnerVariant,
                                   double winnerSpeedup,
                                   String decisionReason,
                                   List<KernelCalibrationCandidate> candidates,
                                   List<KernelTrustedBaseline> trustedBaselines,
                                   BackendChoice baselineChoice,
                                   BackendChoice winnerChoice,
                                   List<BackendCandidateEvidence> backendCandidates,
                                   BackendSelectionStatus selectionStatus,
                                   boolean stable) {
        this.kernelSha256 = text(kernelSha256);
        this.kernelSignature = text(kernelSignature);
        this.workloadBucket = workloadBucket == null || workloadBucket.isBlank()
            ? "exact" : workloadBucket.trim();
        this.baselineVariant = baselineVariant;
        this.winnerVariant = winnerVariant;
        this.winnerSpeedup = winnerSpeedup;
        this.decisionReason = decisionReason == null ? "" : decisionReason;
        this.candidates = candidates == null ? List.of() : List.copyOf(candidates);
        this.trustedBaselines = trustedBaselines == null ? List.of() : List.copyOf(trustedBaselines);
        this.baselineChoice = baselineChoice;
        this.winnerChoice = winnerChoice;
        this.backendCandidates = backendCandidates == null
            ? List.of() : List.copyOf(backendCandidates);
        this.selectionStatus = Objects.requireNonNull(selectionStatus, "Selection status");
        this.stable = stable;
        validateSelection();
        validateChoices();
    }

    public String kernelSha256() { return kernelSha256; }
    public String kernelSignature() { return kernelSignature; }
    public String workloadBucket() { return workloadBucket; }
    public String baselineVariant() { return baselineVariant; }
    public String winnerVariant() { return winnerVariant; }
    public double winnerSpeedup() { return winnerSpeedup; }
    public String decisionReason() { return decisionReason; }
    public List<KernelCalibrationCandidate> candidates() { return candidates; }
    public List<KernelTrustedBaseline> trustedBaselines() { return trustedBaselines; }
    public BackendChoice baselineChoice() { return baselineChoice; }
    public BackendChoice winnerChoice() { return winnerChoice; }
    public List<BackendCandidateEvidence> backendCandidates() { return backendCandidates; }
    public BackendSelectionStatus selectionStatus() { return selectionStatus; }
    public boolean stable() { return stable; }

    public BackendCandidateEvidence evidence(BackendChoice choice) {
        if (choice == null) return null;
        return backendCandidates.stream()
            .filter(candidate -> candidate.choice().equals(choice))
            .findFirst().orElse(null);
    }

    public static KernelCalibrationEntry from(KernelTuningResult result) {
        Objects.requireNonNull(result, "Tuning result");
        BackendChoice baseline = generatedChoice(result.baseline());
        BackendChoice winner = generatedChoice(result.winner());
        BackendSelectionStatus status = result.winner() == null
            ? BackendSelectionStatus.NO_STABLE_WINNER : BackendSelectionStatus.CALIBRATED;
        List<BackendCandidateEvidence> backendEvidence = result.candidates().stream()
            .map(KernelCalibrationEntry::backendEvidence)
            .toList();
        return new KernelCalibrationEntry(
            result.kernelSignature().sha256(), result.kernelSignature().canonicalText(), "exact",
            result.baseline().canonicalText(),
            result.winner() == null ? null : result.winner().canonicalText(),
            result.winnerSpeedup(), result.decisionReason(),
            result.candidates().stream().map(KernelCalibrationCandidate::from).toList(),
            result.trustedBaselines(), baseline, winner, backendEvidence, status,
            result.winner() != null);
    }

    public static KernelCalibrationEntry from(BackendCalibrationResult result) {
        Objects.requireNonNull(result, "Backend calibration result");
        return new KernelCalibrationEntry(result.kernelSignature().sha256(),
            result.kernelSignature().canonicalText(), result.workloadBucket(),
            result.baseline().variantOptional().map(KernelVariantSignature::canonicalText).orElse(null),
            result.winner() == null ? null : result.winner().variantOptional()
                .map(KernelVariantSignature::canonicalText).orElse(null),
            result.winnerSpeedup(), result.decisionReason(), List.of(), List.of(),
            result.baseline(), result.winner(), result.candidates(), result.selectionStatus(),
            result.stable());
    }

    static KernelCalibrationEntry persisted(String kernelSha256,
                                            String kernelSignature,
                                            String workloadBucket,
                                            String baselineVariant,
                                            String winnerVariant,
                                            double winnerSpeedup,
                                            String decisionReason,
                                            List<KernelCalibrationCandidate> candidates,
                                            List<KernelTrustedBaseline> trustedBaselines,
                                            BackendChoice baselineChoice,
                                            BackendChoice winnerChoice,
                                            List<BackendCandidateEvidence> backendCandidates,
                                            BackendSelectionStatus selectionStatus,
                                            boolean stable) {
        return new KernelCalibrationEntry(kernelSha256, kernelSignature, workloadBucket,
            baselineVariant, winnerVariant, winnerSpeedup, decisionReason, candidates,
            trustedBaselines, baselineChoice, winnerChoice, backendCandidates,
            selectionStatus, stable);
    }

    private void validateSelection() {
        if (selectionStatus == BackendSelectionStatus.NO_STABLE_WINNER) {
            if (winnerChoice != null || winnerVariant != null || stable
                || winnerSpeedup != 0.0) {
                throw new IllegalArgumentException(
                    "No-stable-winner entries must not retain a winner");
            }
            return;
        }
        if (winnerChoice == null || !stable || !Double.isFinite(winnerSpeedup)
            || winnerSpeedup <= 0.0) {
            throw new IllegalArgumentException(
                "Calibrated entries require a stable finite positive winner speedup");
        }
    }

    private void validateChoices() {
        if (baselineChoice == null) {
            throw new IllegalArgumentException("Profile entry must retain a baseline choice");
        }
        if (!matchesKernel(baselineChoice)) {
            throw new IllegalArgumentException("Baseline choice does not match profile kernel");
        }
        if (winnerChoice != null && !matchesKernel(winnerChoice)) {
            throw new IllegalArgumentException("Winner choice does not match profile kernel");
        }
        for (BackendCandidateEvidence evidence : backendCandidates) {
            if (!matchesKernel(evidence.choice())) {
                throw new IllegalArgumentException("Candidate choice does not match profile kernel");
            }
        }
    }

    private boolean matchesKernel(BackendChoice choice) {
        return choice instanceof R2JavaBackend
            || choice.variantOptional().map(variant -> kernelSha256.equals(variant.kernelSignature().sha256())
                && kernelSignature.equals(variant.kernelSignature().canonicalText())).orElse(false);
    }

    private static BackendCandidateEvidence backendEvidence(KernelCandidateEvidence evidence) {
        BackendChoice choice = generatedChoice(evidence.variantSignature());
        return new BackendCandidateEvidence(choice, evidence.outcome(),
            evidence.correctnessPassed(), evidence.stable(), evidence.statistics(),
            evidence.benchmarkNanos(), evidence.reason());
    }

    private static BackendChoice generatedChoice(KernelVariantSignature variant) {
        if (variant == null) return null;
        return variant.backend() == CodegenBackend.AVX2
            ? new GeneratedAvx2Backend(variant) : new ScalarNativeBackend(variant);
    }

    private static String legacyVariant(BackendChoice choice) {
        return choice == null ? null : choice.variantOptional()
            .map(KernelVariantSignature::canonicalText).orElse(null);
    }

    private static BackendChoice inferChoice(String signatureText, String variantText) {
        if (variantText == null || variantText.isBlank() || signatureText == null || signatureText.isBlank()) {
            return null;
        }
        try {
            KernelSignature signature = KernelSignature.fromCanonicalText(signatureText);
            KernelVariantSignature variant = KernelVariantSignature.fromCanonicalText(signature, variantText);
            return variant.backend() == CodegenBackend.AVX2
                ? new GeneratedAvx2Backend(variant) : new ScalarNativeBackend(variant);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    private static BackendSelectionStatus legacyStatus(String winnerVariant) {
        return winnerVariant == null || winnerVariant.isBlank()
            ? BackendSelectionStatus.NO_STABLE_WINNER : BackendSelectionStatus.CALIBRATED;
    }

    private static double legacyWinnerSpeedup(String winnerVariant, double speedup) {
        return winnerVariant == null || winnerVariant.isBlank() ? 0.0 : speedup;
    }

    private static String text(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Profile kernel identity must not be blank");
        }
        return value;
    }
}
