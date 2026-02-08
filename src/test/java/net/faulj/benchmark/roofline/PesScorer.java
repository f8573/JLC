package net.faulj.benchmark.roofline;

final class PesScorer {
    private static final double MAX_PES = 0.95;

    private PesScorer() {
    }

    static PesResult score(KernelProfile profile, double elapsedSeconds, RooflineSession roofline) {
        double seconds = Math.max(1e-9, elapsedSeconds);
        double ai = profile.estimatedBytesMoved > 0.0 ? profile.theoreticalFlops / profile.estimatedBytesMoved : 0.0;
        double measuredFlopsPerSecond = profile.theoreticalFlops / seconds;
        double measuredBytesPerSecond = profile.estimatedBytesMoved / seconds;

        double computeRoof = roofline.peakFlopsPerSecond;
        MemoryBandwidthProbe.BandwidthSelection selectedMemory = roofline.memoryBandwidths.forWorkingSet(profile.workingSetBytes);
        double selectedBandwidth = selectedMemory.bytesPerSecond;
        double memoryRoof = selectedBandwidth * ai;
        double effectiveRoof = Math.min(computeRoof, memoryRoof);
        double boundedRoof = Math.max(1e-9, effectiveRoof);

        String boundType = computeRoof <= memoryRoof ? "compute" : "memory";
        double computeUtil = clamp01(measuredFlopsPerSecond / Math.max(1e-9, computeRoof));
        double memoryUtil = clamp01(measuredBytesPerSecond / Math.max(1e-9, selectedBandwidth));
        double roofUtil = clamp01(measuredFlopsPerSecond / boundedRoof);

        double algorithmicEfficiency;
        if (profile.actualFlops <= 0.0) {
            algorithmicEfficiency = 1.0;
        } else {
            algorithmicEfficiency = clamp01(profile.theoreticalFlops / profile.actualFlops);
        }

        double pes = clampPes(roofUtil * algorithmicEfficiency);
        return new PesResult(
            profile.kernel,
            profile.n,
            ai,
            boundType,
            selectedMemory.level.name().toLowerCase(),
            computeUtil,
            memoryUtil,
            algorithmicEfficiency,
            pes,
            measuredFlopsPerSecond / 1e9,
            boundedRoof / 1e9,
            selectedBandwidth / 1e9,
            seconds
        );
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }
        if (value < 0.0) {
            return 0.0;
        }
        return Math.min(1.0, value);
    }

    private static double clampPes(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value) || value < 0.0) {
            return 0.0;
        }
        return Math.min(MAX_PES, value);
    }
}
