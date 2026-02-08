package net.faulj.benchmark.roofline;

final class RooflineSession {
    private static volatile RooflineSession cached;

    final HardwareInfo hardware;
    final double peakFlopsPerSecond;
    final double rawTheoreticalPeakFlopsPerSecond;
    final double measuredGemmAnchorGflops;
    final MemoryBandwidthProbe.BandwidthHierarchy memoryBandwidths;
    final String computeRoofSource;
    final String memoryRoofSource;

    private RooflineSession(HardwareInfo hardware,
                            double peakFlopsPerSecond,
                            double rawTheoreticalPeakFlopsPerSecond,
                            double measuredGemmAnchorGflops,
                            MemoryBandwidthProbe.BandwidthHierarchy memoryBandwidths,
                            String computeRoofSource,
                            String memoryRoofSource) {
        this.hardware = hardware;
        this.peakFlopsPerSecond = peakFlopsPerSecond;
        this.rawTheoreticalPeakFlopsPerSecond = rawTheoreticalPeakFlopsPerSecond;
        this.measuredGemmAnchorGflops = measuredGemmAnchorGflops;
        this.memoryBandwidths = memoryBandwidths;
        this.computeRoofSource = computeRoofSource;
        this.memoryRoofSource = memoryRoofSource;
    }

    static RooflineSession get() {
        RooflineSession local = cached;
        if (local != null) {
            return local;
        }
        synchronized (RooflineSession.class) {
            local = cached;
            if (local == null) {
                HardwareInfo hardware = HardwareProbe.probe();
                ComputeRoofProbe.ComputeEstimate compute = ComputeRoofProbe.probe(hardware);
                MemoryBandwidthProbe.BandwidthHierarchy bw = MemoryBandwidthProbe.probe();
                local = new RooflineSession(
                    hardware,
                    compute.bytesFlopsPerSecond,
                    compute.rawTheoreticalPeak,
                    compute.measuredGemmAnchorGflops,
                    bw,
                    compute.source,
                    bw.source
                );
                cached = local;
            }
        }
        return local;
    }

    RooflineSession withComputeRoof(double computeRoofFlopsPerSecond, String source) {
        double roof = Math.max(1e9, computeRoofFlopsPerSecond);
        return new RooflineSession(
            hardware,
            roof,
            rawTheoreticalPeakFlopsPerSecond,
            measuredGemmAnchorGflops,
            memoryBandwidths,
            source,
            memoryRoofSource
        );
    }
}
