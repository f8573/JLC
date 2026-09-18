package net.faulj.benchmark.roofline;

/**
 * Computes the Portable Efficiency Score (PES) from a kernel profile and
 * roofline session.
 *
 * <p><b>PES = measured_GFLOPs / min(compute_roof, bandwidth × AI)</b></p>
 *
 * <p>Key design decisions:
 * <ul>
 *   <li>PES is <b>never clamped</b>.  Values &gt; 1.0 are flagged, not hidden.</li>
 *   <li>Regime-aware PES variants (PES_L1 through PES_DRAM) are computed
 *       for every result as diagnostics.</li>
 *   <li>Confidence level reflects which parts of the roofline model were
 *       measured vs assumed.</li>
 * </ul>
 */
final class PesScorer {

    private PesScorer() {
    }

    /**
     * Score a kernel run.
     *
     * @param profile   kernel FLOP count and traffic model
     * @param elapsedSeconds  best-of-N wall-clock time
     * @param roofline  hardware roofline (compute + bandwidth hierarchy)
     * @return full diagnostic PES result
     */
    static PesResult score(KernelProfile profile, double elapsedSeconds, RooflineSession roofline) {
        double seconds = Math.max(1e-12, elapsedSeconds);

        // ── Measured throughput ─────────────────────────────────────────
        double measuredFlopsPerSecond = profile.theoreticalFlops / seconds;
        double measuredBytesPerSecond = profile.estimatedBytesMoved / seconds;
        double measuredGflops = measuredFlopsPerSecond / 1e9;

        // ── Arithmetic intensity ───────────────────────────────────────
        double ai = profile.estimatedBytesMoved > 0.0
            ? profile.theoreticalFlops / profile.estimatedBytesMoved
            : 0.0;

        // ── Roofline ceilings ──────────────────────────────────────────
        double computeRoof = roofline.peakFlopsPerSecond;
        double computeRoofGflops = computeRoof / 1e9;

        MemoryBandwidthProbe.BandwidthSelection selectedMemory =
            roofline.memoryBandwidths.forWorkingSet(profile.workingSetBytes);
        double selectedBandwidth = selectedMemory.bytesPerSecond;
        double memoryRoof = selectedBandwidth * ai;
        double effectiveRoof = Math.min(computeRoof, memoryRoof);
        double boundedRoof = Math.max(1e-9, effectiveRoof);

        String boundType = computeRoof <= memoryRoof ? "compute" : "memory";
        String memoryLevel = selectedMemory.level.name().toLowerCase();

        // ── Utilization fractions ──────────────────────────────────────
        double computeUtil = safeDivide(measuredFlopsPerSecond, computeRoof);
        double memoryUtil = safeDivide(measuredBytesPerSecond, selectedBandwidth);

        double algorithmicEfficiency;
        if (profile.actualFlops <= 0.0) {
            algorithmicEfficiency = 1.0;
        } else {
            algorithmicEfficiency = Math.min(1.0, profile.theoreticalFlops / profile.actualFlops);
        }

        // ── Canonical PES — NO CLAMPING ────────────────────────────────
        double rawPes = safeDivide(measuredFlopsPerSecond, boundedRoof);
        double pes = rawPes * algorithmicEfficiency;

        // ── Regime-aware PES variants ──────────────────────────────────
        double pesL1 = regimePes(measuredFlopsPerSecond, computeRoof,
            roofline.memoryBandwidths.l1BytesPerSecond, ai);
        double pesL2 = regimePes(measuredFlopsPerSecond, computeRoof,
            roofline.memoryBandwidths.l2BytesPerSecond, ai);
        double pesL3 = regimePes(measuredFlopsPerSecond, computeRoof,
            roofline.memoryBandwidths.l3BytesPerSecond, ai);
        double pesDram = regimePes(measuredFlopsPerSecond, computeRoof,
            roofline.memoryBandwidths.dramBytesPerSecond, ai);

        // ── Confidence ─────────────────────────────────────────────────
        String confidence;
        if (roofline.hardware.tier == CapabilityTier.SAFE_MODE) {
            confidence = "UNDEFINED";
        } else if (roofline.hardware.tier == CapabilityTier.FULLY_MEASURED) {
            confidence = "MEASURED";
        } else if (roofline.hardware.tier == CapabilityTier.MINIMAL) {
            confidence = "SAFE_MODE";
        } else {
            confidence = "ESTIMATED";
        }

        // ── Flags — PES > 1.0 is a model or measurement error ─────────
        String flag;
        if (roofline.hardware.tier == CapabilityTier.SAFE_MODE) {
            flag = "UNDEFINED";
        } else if (!Double.isFinite(pes) || pes < 0.0) {
            flag = "MEASUREMENT_ERROR";
        } else if (pes > 1.05) {
            flag = "MEASUREMENT_ERROR";
        } else if (pes > 1.0) {
            flag = "MODEL_ERROR";
        } else {
            flag = "OK";
        }

        return new PesResult(
            profile.kernel, profile.m, profile.n, profile.k,
            ai, profile.trafficModel,
            boundType, memoryLevel,
            computeUtil, memoryUtil, algorithmicEfficiency,
            pes,
            pesL1, pesL2, pesL3, pesDram,
            measuredGflops, computeRoofGflops, boundedRoof / 1e9,
            selectedBandwidth / 1e9,
            seconds,
            confidence, flag
        );
    }

    /**
     * Compute PES for a specific memory regime (diagnostic).
     */
    private static double regimePes(double measuredFlops, double computeRoof,
                                     double regimeBandwidth, double ai) {
        double memRoof = regimeBandwidth * ai;
        double roof = Math.min(computeRoof, memRoof);
        return safeDivide(measuredFlops, Math.max(1e-9, roof));
    }

    private static double safeDivide(double a, double b) {
        if (!Double.isFinite(a) || !Double.isFinite(b) || b <= 0.0) {
            return 0.0;
        }
        return a / b;
    }
}
