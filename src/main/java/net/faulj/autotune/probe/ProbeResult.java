package net.faulj.autotune.probe;

import java.util.Locale;

/**
 * Aggregate result of all Phase 1 hardware probes.
 * Every field carries an explicit {@link ProbeConfidence}.
 */
public final class ProbeResult {

    public final long effectiveL1Bytes;
    public final long effectiveL2Bytes;
    public final long effectiveL3Bytes;

    public final double bandwidthL1;
    public final double bandwidthL2;
    public final double bandwidthL3;
    public final double bandwidthDram;

    public final int effectiveIssueWidth;
    public final double simdLoweringQuality;
    public final int registerSpillThreshold;
    public final double fmaLatencyCycles;
    public final int vectorLength;
    public final boolean fmaSupported;

    public final ProbeConfidence cacheSizeConfidence;
    public final ProbeConfidence bandwidthConfidence;
    public final ProbeConfidence issueWidthConfidence;
    public final ProbeConfidence simdQualityConfidence;
    public final ProbeConfidence spillThresholdConfidence;
    public final ProbeConfidence fmaLatencyConfidence;

    public ProbeResult(
            long effectiveL1Bytes, long effectiveL2Bytes, long effectiveL3Bytes,
            double bandwidthL1, double bandwidthL2, double bandwidthL3, double bandwidthDram,
            int effectiveIssueWidth, double simdLoweringQuality,
            int registerSpillThreshold, double fmaLatencyCycles,
            int vectorLength, boolean fmaSupported,
            ProbeConfidence cacheSizeConfidence, ProbeConfidence bandwidthConfidence,
            ProbeConfidence issueWidthConfidence, ProbeConfidence simdQualityConfidence,
            ProbeConfidence spillThresholdConfidence, ProbeConfidence fmaLatencyConfidence) {
        this.effectiveL1Bytes = effectiveL1Bytes;
        this.effectiveL2Bytes = effectiveL2Bytes;
        this.effectiveL3Bytes = effectiveL3Bytes;
        this.bandwidthL1 = bandwidthL1;
        this.bandwidthL2 = bandwidthL2;
        this.bandwidthL3 = bandwidthL3;
        this.bandwidthDram = bandwidthDram;
        this.effectiveIssueWidth = effectiveIssueWidth;
        this.simdLoweringQuality = simdLoweringQuality;
        this.registerSpillThreshold = registerSpillThreshold;
        this.fmaLatencyCycles = fmaLatencyCycles;
        this.vectorLength = vectorLength;
        this.fmaSupported = fmaSupported;
        this.cacheSizeConfidence = cacheSizeConfidence;
        this.bandwidthConfidence = bandwidthConfidence;
        this.issueWidthConfidence = issueWidthConfidence;
        this.simdQualityConfidence = simdQualityConfidence;
        this.spillThresholdConfidence = spillThresholdConfidence;
        this.fmaLatencyConfidence = fmaLatencyConfidence;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "ProbeResult {%n" +
            "  Cache sizes:      L1=%d KB, L2=%d KB, L3=%d KB [%s]%n" +
            "  Bandwidth:        L1=%.1f GB/s, L2=%.1f GB/s, L3=%.1f GB/s, DRAM=%.1f GB/s [%s]%n" +
            "  Issue width:      %d [%s]%n" +
            "  SIMD quality:     %.3f [%s]%n" +
            "  Spill threshold:  %d accumulators [%s]%n" +
            "  FMA latency:      %.3f cycles [%s]%n" +
            "  Vector length:    %d doubles%n" +
            "  FMA supported:    %s%n" +
            "}",
            effectiveL1Bytes / 1024, effectiveL2Bytes / 1024, effectiveL3Bytes / 1024,
            cacheSizeConfidence,
            bandwidthL1 / 1e9, bandwidthL2 / 1e9, bandwidthL3 / 1e9, bandwidthDram / 1e9,
            bandwidthConfidence,
            effectiveIssueWidth, issueWidthConfidence,
            simdLoweringQuality, simdQualityConfidence,
            registerSpillThreshold, spillThresholdConfidence,
            fmaLatencyCycles, fmaLatencyConfidence,
            vectorLength,
            fmaSupported);
    }
}
