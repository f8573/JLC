package net.faulj.autotune.inference;

import net.faulj.autotune.probe.ProbeResult;

import java.util.Locale;

/**
 * Diagnostic report for Phase 2 hardware limit inference.
 *
 * <p>Provides transparency into the inference process, showing:
 * <ul>
 *   <li>Input probe measurements</li>
 *   <li>Output conservative limits</li>
 *   <li>Safety factors applied</li>
 *   <li>Confidence assessment rationale</li>
 * </ul>
 *
 * <p>Useful for debugging, logging, and understanding why certain limits were chosen.</p>
 */
public final class InferenceReport {

    /** Input probe measurements. */
    public final ProbeResult inputProbes;

    /** Output hardware limits. */
    public final HardwareLimits outputLimits;

    /** Overall inference confidence. */
    public final InferenceConfidence confidence;

    /** Summary of applied safety factors. */
    public final String safetyFactorSummary;

    // ────────────────────────────────────────────────────────────────────

    public InferenceReport(ProbeResult inputProbes, HardwareLimits outputLimits,
                           InferenceConfidence confidence, String safetyFactorSummary) {
        this.inputProbes = inputProbes;
        this.outputLimits = outputLimits;
        this.confidence = confidence;
        this.safetyFactorSummary = safetyFactorSummary;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "InferenceReport {%n" +
            "  Confidence:     %s%n" +
            "  Reason:         %s%n" +
            "  Safety Factors: %s%n" +
            "%n" +
            "  ── Input Probes ──────────────────────────────────────────────%n" +
            "%s%n" +
            "  ── Output Limits ─────────────────────────────────────────────%n" +
            "%s" +
            "}",
            confidence,
            outputLimits.confidenceReason,
            safetyFactorSummary,
            indent(inputProbes.toString(), "  "),
            indent(outputLimits.toString(), "  "));
    }

    /**
     * Returns a concise summary suitable for logging.
     */
    public String toSummary() {
        return String.format(Locale.ROOT,
            "Inference: %s confidence, L1=%d KB, L2=%d KB, L3=%d KB, " +
            "BW(L1)=%.1f GB/s, BW(DRAM)=%.1f GB/s, IssueWidth=%d, MaxAccum=%d",
            confidence,
            outputLimits.maxL1WorkingSetBytes / 1024,
            outputLimits.maxL2WorkingSetBytes / 1024,
            outputLimits.maxL3WorkingSetBytes / 1024,
            outputLimits.maxL1BandwidthBytesPerSec / 1e9,
            outputLimits.maxDramBandwidthBytesPerSec / 1e9,
            outputLimits.conservativeFmaIssueWidth,
            outputLimits.maxVectorAccumulators);
    }

    /**
     * Returns detailed JSON-like structured output.
     */
    public String toStructuredOutput() {
        return String.format(Locale.ROOT,
            "{\n" +
            "  \"confidence\": \"%s\",\n" +
            "  \"confidence_reason\": \"%s\",\n" +
            "  \"safety_factors\": \"%s\",\n" +
            "  \"limits\": {\n" +
            "    \"cache\": {\n" +
            "      \"l1_bytes\": %d,\n" +
            "      \"l2_bytes\": %d,\n" +
            "      \"l3_bytes\": %d\n" +
            "    },\n" +
            "    \"bandwidth_bytes_per_sec\": {\n" +
            "      \"l1\": %.0f,\n" +
            "      \"l2\": %.0f,\n" +
            "      \"l3\": %.0f,\n" +
            "      \"dram\": %.0f\n" +
            "    },\n" +
            "    \"compute\": {\n" +
            "      \"fma_issue_width\": %d,\n" +
            "      \"vector_length_doubles\": %d,\n" +
            "      \"fma_available\": %s,\n" +
            "      \"simd_quality\": %.4f,\n" +
            "      \"max_vector_accumulators\": %d,\n" +
            "      \"fma_latency_cycles\": %.2f\n" +
            "    }\n" +
            "  }\n" +
            "}",
            confidence,
            outputLimits.confidenceReason.replace("\"", "\\\""),
            safetyFactorSummary.replace("\"", "\\\""),
            outputLimits.maxL1WorkingSetBytes,
            outputLimits.maxL2WorkingSetBytes,
            outputLimits.maxL3WorkingSetBytes,
            outputLimits.maxL1BandwidthBytesPerSec,
            outputLimits.maxL2BandwidthBytesPerSec,
            outputLimits.maxL3BandwidthBytesPerSec,
            outputLimits.maxDramBandwidthBytesPerSec,
            outputLimits.conservativeFmaIssueWidth,
            outputLimits.vectorLengthDoubles,
            outputLimits.fmaAvailable,
            outputLimits.simdLoweringQuality,
            outputLimits.maxVectorAccumulators,
            outputLimits.fmaPipelineLatencyCycles);
    }

    /**
     * Indent each line of a multi-line string.
     */
    private static String indent(String text, String prefix) {
        return text.lines()
                .map(line -> prefix + line)
                .reduce((a, b) -> a + "\n" + b)
                .orElse("");
    }
}
