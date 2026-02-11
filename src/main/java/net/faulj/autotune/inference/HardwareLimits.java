package net.faulj.autotune.inference;

import java.util.Locale;

/**
 * Conservative hardware execution limits derived from Phase 1 probe measurements.
 *
 * <p>All values include safety factors and represent <em>conservative bounds</em>,
 * not optimal operating points. Downstream phases (search space reduction,
 * benchmarking) use these limits to constrain parameter selection.</p>
 *
 * <p>Fields are immutable and non-nullable.</p>
 */
public final class HardwareLimits {

    // ── Cache & Memory ──────────────────────────────────────────────────

    /** Maximum bytes for L1-resident working set (per core). */
    public final long maxL1WorkingSetBytes;

    /** Maximum bytes for L2-resident working set (per core). */
    public final long maxL2WorkingSetBytes;

    /** Maximum bytes for L3-resident working set (shared). */
    public final long maxL3WorkingSetBytes;

    /** Sustained L1 bandwidth limit (bytes/sec, per core). */
    public final double maxL1BandwidthBytesPerSec;

    /** Sustained L2 bandwidth limit (bytes/sec, per core). */
    public final double maxL2BandwidthBytesPerSec;

    /** Sustained L3 bandwidth limit (bytes/sec, shared). */
    public final double maxL3BandwidthBytesPerSec;

    /** Sustained DRAM bandwidth limit (bytes/sec, system-wide). */
    public final double maxDramBandwidthBytesPerSec;

    // ── Computation ─────────────────────────────────────────────────────

    /** Conservative FMA issue width (1 or 2). */
    public final int conservativeFmaIssueWidth;

    /** Number of doubles per vector register. */
    public final int vectorLengthDoubles;

    /** True if hardware FMA is available and usable. */
    public final boolean fmaAvailable;

    /** SIMD lowering quality: 0.0 (poor) to 1.0+ (excellent). */
    public final double simdLoweringQuality;

    // ── Register Pressure ───────────────────────────────────────────────

    /** Maximum safe vector accumulator count before spill. */
    public final int maxVectorAccumulators;

    /** FMA pipeline latency in cycles (for dependency chain estimation). */
    public final double fmaPipelineLatencyCycles;

    // ── Confidence & Metadata ───────────────────────────────────────────

    /** Overall confidence in these limits. */
    public final InferenceConfidence confidence;

    /** Human-readable summary of what drove the confidence level. */
    public final String confidenceReason;

    // ────────────────────────────────────────────────────────────────────

    public HardwareLimits(
            long maxL1WorkingSetBytes, long maxL2WorkingSetBytes, long maxL3WorkingSetBytes,
            double maxL1BandwidthBytesPerSec, double maxL2BandwidthBytesPerSec,
            double maxL3BandwidthBytesPerSec, double maxDramBandwidthBytesPerSec,
            int conservativeFmaIssueWidth, int vectorLengthDoubles, boolean fmaAvailable,
            double simdLoweringQuality, int maxVectorAccumulators,
            double fmaPipelineLatencyCycles,
            InferenceConfidence confidence, String confidenceReason) {
        this.maxL1WorkingSetBytes = maxL1WorkingSetBytes;
        this.maxL2WorkingSetBytes = maxL2WorkingSetBytes;
        this.maxL3WorkingSetBytes = maxL3WorkingSetBytes;
        this.maxL1BandwidthBytesPerSec = maxL1BandwidthBytesPerSec;
        this.maxL2BandwidthBytesPerSec = maxL2BandwidthBytesPerSec;
        this.maxL3BandwidthBytesPerSec = maxL3BandwidthBytesPerSec;
        this.maxDramBandwidthBytesPerSec = maxDramBandwidthBytesPerSec;
        this.conservativeFmaIssueWidth = conservativeFmaIssueWidth;
        this.vectorLengthDoubles = vectorLengthDoubles;
        this.fmaAvailable = fmaAvailable;
        this.simdLoweringQuality = simdLoweringQuality;
        this.maxVectorAccumulators = maxVectorAccumulators;
        this.fmaPipelineLatencyCycles = fmaPipelineLatencyCycles;
        this.confidence = confidence;
        this.confidenceReason = confidenceReason;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "HardwareLimits {%n" +
            "  Working sets:     L1=%d KB, L2=%d KB, L3=%d KB%n" +
            "  Bandwidth:        L1=%.1f GB/s, L2=%.1f GB/s, L3=%.1f GB/s, DRAM=%.1f GB/s%n" +
            "  FMA issue width:  %d (conservative)%n" +
            "  Vector length:    %d doubles%n" +
            "  FMA available:    %s%n" +
            "  SIMD quality:     %.3f%n" +
            "  Max accumulators: %d%n" +
            "  FMA latency:      %.3f cycles%n" +
            "  Confidence:       %s (%s)%n" +
            "}",
            maxL1WorkingSetBytes / 1024, maxL2WorkingSetBytes / 1024, maxL3WorkingSetBytes / 1024,
            maxL1BandwidthBytesPerSec / 1e9, maxL2BandwidthBytesPerSec / 1e9,
            maxL3BandwidthBytesPerSec / 1e9, maxDramBandwidthBytesPerSec / 1e9,
            conservativeFmaIssueWidth, vectorLengthDoubles, fmaAvailable,
            simdLoweringQuality, maxVectorAccumulators, fmaPipelineLatencyCycles,
            confidence, confidenceReason);
    }
}
