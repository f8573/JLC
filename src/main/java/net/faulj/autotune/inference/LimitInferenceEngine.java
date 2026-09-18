package net.faulj.autotune.inference;

import net.faulj.autotune.probe.ProbeConfidence;
import net.faulj.autotune.probe.ProbeResult;

/**
 * Transforms Phase 1 probe measurements into conservative hardware limits
 * suitable for Phase 3 search space reduction and Phase 4 benchmarking.
 *
 * <p>Applies safety factors to all measurements to ensure kernel execution
 * stays within proven hardware capabilities.</p>
 *
 * <p>This class is stateless and thread-safe.</p>
 */
public final class LimitInferenceEngine {

    // ── Safety Factors ──────────────────────────────────────────────────
    //
    // These factors ensure we operate conservatively within measured bounds.

    /** Cache working set safety factor (use 75% of detected size). */
    private static final double CACHE_SAFETY_FACTOR = 0.75;

    /** Bandwidth safety factor (assume 85% sustainable throughput). */
    private static final double BANDWIDTH_SAFETY_FACTOR = 0.85;

    /** Issue width safety factor (reduce by 1 if SIMD quality < 0.75). */
    private static final double SIMD_QUALITY_THRESHOLD = 0.75;

    /** Accumulator safety factor (use 75% of spill threshold). */
    private static final double ACCUMULATOR_SAFETY_FACTOR = 0.75;

    /** Minimum accumulator count even if probe fails. */
    private static final int MIN_ACCUMULATORS = 4;

    /** Maximum accumulator count cap. */
    private static final int MAX_ACCUMULATORS = 16;

    /** FMA latency safety factor (add 10% to measured latency). */
    private static final double LATENCY_SAFETY_FACTOR = 1.10;

    // ────────────────────────────────────────────────────────────────────

    private LimitInferenceEngine() {}

    /**
     * Infer conservative hardware limits from probe results.
     *
     * @param probeResult Phase 1 probe measurements
     * @return conservative hardware limits with confidence assessment
     */
    public static InferenceResult infer(ProbeResult probeResult) {
        // ── 1. Cache Limits ─────────────────────────────────────────────
        long maxL1 = (long) (probeResult.effectiveL1Bytes * CACHE_SAFETY_FACTOR);
        long maxL2 = (long) (probeResult.effectiveL2Bytes * CACHE_SAFETY_FACTOR);
        long maxL3 = (long) (probeResult.effectiveL3Bytes * CACHE_SAFETY_FACTOR);

        // Ensure minimum sizes (8KB L1, 64KB L2, 512KB L3)
        maxL1 = Math.max(maxL1, 8 * 1024L);
        maxL2 = Math.max(maxL2, 64 * 1024L);
        maxL3 = Math.max(maxL3, 512 * 1024L);

        // ── 2. Bandwidth Limits ─────────────────────────────────────────
        double maxBwL1 = probeResult.bandwidthL1 * BANDWIDTH_SAFETY_FACTOR;
        double maxBwL2 = probeResult.bandwidthL2 * BANDWIDTH_SAFETY_FACTOR;
        double maxBwL3 = probeResult.bandwidthL3 * BANDWIDTH_SAFETY_FACTOR;
        double maxBwDram = probeResult.bandwidthDram * BANDWIDTH_SAFETY_FACTOR;

        // Ensure minimum bandwidth (1 GB/s per level)
        maxBwL1 = Math.max(maxBwL1, 1e9);
        maxBwL2 = Math.max(maxBwL2, 1e9);
        maxBwL3 = Math.max(maxBwL3, 1e9);
        maxBwDram = Math.max(maxBwDram, 1e9);

        // ── 3. Conservative FMA Issue Width ─────────────────────────────
        int conservativeIssueWidth = probeResult.effectiveIssueWidth;
        if (probeResult.simdLoweringQuality < SIMD_QUALITY_THRESHOLD && conservativeIssueWidth > 1) {
            conservativeIssueWidth = 1; // Reduce to 1 if SIMD quality is poor
        }

        // ── 4. Vector Accumulator Limit ─────────────────────────────────
        int maxAccumulators = (int) (probeResult.registerSpillThreshold * ACCUMULATOR_SAFETY_FACTOR);
        maxAccumulators = Math.max(MIN_ACCUMULATORS, Math.min(MAX_ACCUMULATORS, maxAccumulators));

        // ── 5. FMA Latency (Conservative) ───────────────────────────────
        double conservativeLatency = probeResult.fmaLatencyCycles * LATENCY_SAFETY_FACTOR;

        // ── 6. Confidence Assessment ────────────────────────────────────
        InferenceConfidence confidence = assessConfidence(probeResult);
        String reason = buildConfidenceReason(probeResult);

        // ── 7. Construct Limits ─────────────────────────────────────────
        HardwareLimits limits = new HardwareLimits(
                maxL1, maxL2, maxL3,
                maxBwL1, maxBwL2, maxBwL3, maxBwDram,
                conservativeIssueWidth,
                probeResult.vectorLength,
                probeResult.fmaSupported,
                probeResult.simdLoweringQuality,
                maxAccumulators,
                conservativeLatency,
                confidence,
                reason);

        // ── 8. Build Report ─────────────────────────────────────────────
        InferenceReport report = new InferenceReport(
                probeResult, limits, confidence,
                computeSafetyFactorSummary(probeResult, limits));

        return new InferenceResult(limits, report);
    }

    /**
     * Assess overall confidence based on individual probe confidences.
     */
    private static InferenceConfidence assessConfidence(ProbeResult pr) {
        int measured = 0;
        int estimated = 0;
        int failed = 0;

        ProbeConfidence[] confidences = {
            pr.cacheSizeConfidence, pr.bandwidthConfidence,
            pr.issueWidthConfidence, pr.simdQualityConfidence,
            pr.spillThresholdConfidence, pr.fmaLatencyConfidence
        };

        for (ProbeConfidence c : confidences) {
            switch (c) {
                case MEASURED: measured++; break;
                case ESTIMATED: estimated++; break;
                case FAILED: failed++; break;
            }
        }

        // Confidence rules:
        // HIGH: all measured
        // MEDIUM: 1-2 estimated or 1 non-critical failed
        // LOW: 2+ failed or critical probes estimated
        // FAILED: 3+ failed

        if (measured == confidences.length) return InferenceConfidence.HIGH;
        if (failed >= 3) return InferenceConfidence.FAILED;
        if (failed >= 2 || isCriticalProbeFailed(pr)) return InferenceConfidence.LOW;
        if (estimated <= 2 && failed <= 1) return InferenceConfidence.MEDIUM;
        return InferenceConfidence.LOW;
    }

    private static boolean isCriticalProbeFailed(ProbeResult pr) {
        // Critical probes: cache size, bandwidth, issue width
        return pr.cacheSizeConfidence == ProbeConfidence.FAILED
            || pr.bandwidthConfidence == ProbeConfidence.FAILED
            || pr.issueWidthConfidence == ProbeConfidence.FAILED;
    }

    /**
     * Build human-readable confidence reason.
     */
    private static String buildConfidenceReason(ProbeResult pr) {
        StringBuilder sb = new StringBuilder();
        int failed = 0;
        int estimated = 0;

        if (pr.cacheSizeConfidence != ProbeConfidence.MEASURED) {
            sb.append("cache_size=").append(pr.cacheSizeConfidence).append(" ");
            if (pr.cacheSizeConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }
        if (pr.bandwidthConfidence != ProbeConfidence.MEASURED) {
            sb.append("bandwidth=").append(pr.bandwidthConfidence).append(" ");
            if (pr.bandwidthConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }
        if (pr.issueWidthConfidence != ProbeConfidence.MEASURED) {
            sb.append("issue_width=").append(pr.issueWidthConfidence).append(" ");
            if (pr.issueWidthConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }
        if (pr.simdQualityConfidence != ProbeConfidence.MEASURED) {
            sb.append("simd_quality=").append(pr.simdQualityConfidence).append(" ");
            if (pr.simdQualityConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }
        if (pr.spillThresholdConfidence != ProbeConfidence.MEASURED) {
            sb.append("spill_threshold=").append(pr.spillThresholdConfidence).append(" ");
            if (pr.spillThresholdConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }
        if (pr.fmaLatencyConfidence != ProbeConfidence.MEASURED) {
            sb.append("fma_latency=").append(pr.fmaLatencyConfidence).append(" ");
            if (pr.fmaLatencyConfidence == ProbeConfidence.FAILED) failed++;
            else estimated++;
        }

        if (sb.length() == 0) return "all probes measured";
        return String.format("%d failed, %d estimated: %s", failed, estimated, sb.toString().trim());
    }

    /**
     * Compute summary of applied safety factors.
     */
    private static String computeSafetyFactorSummary(ProbeResult pr, HardwareLimits limits) {
        return String.format(
            "Cache: %.0f%%, Bandwidth: %.0f%%, Accumulators: %.0f%%, Latency: +%.0f%%",
            CACHE_SAFETY_FACTOR * 100,
            BANDWIDTH_SAFETY_FACTOR * 100,
            ACCUMULATOR_SAFETY_FACTOR * 100,
            (LATENCY_SAFETY_FACTOR - 1.0) * 100);
    }

    /**
     * Result of inference: limits + diagnostic report.
     */
    public static final class InferenceResult {
        public final HardwareLimits limits;
        public final InferenceReport report;

        public InferenceResult(HardwareLimits limits, InferenceReport report) {
            this.limits = limits;
            this.report = report;
        }
    }
}
