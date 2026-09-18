package net.faulj.benchmark.roofline;

import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Top-level API for the portable, regime-aware roofline engine.
 *
 * <p>Usage:
 * <pre>{@code
 *   RooflineModel model = RooflineModel.detect();
 *
 *   // Score a GEMM run:
 *   KernelProfile profile = KernelModel.gemm(m, n, k, mc, nc, kc);
 *   PesResult result = model.score(profile, elapsedSeconds);
 *
 *   // Inspect:
 *   System.out.println("PES = " + result.portableEfficiencyScore);
 *   System.out.println("Confidence = " + result.confidence);
 *   System.out.println("Flag = " + result.flag);
 * }</pre>
 *
 * <p>The model automatically selects the highest available capability tier
 * (Tier 0 through Tier 3) based on which hardware probes succeed.  Every
 * assumption is recorded and exposed via {@link #assumptions()}.</p>
 */
final class RooflineModel {

    private final RooflineSession session;

    private RooflineModel(RooflineSession session) {
        this.session = session;
    }

    // ── Factory methods ────────────────────────────────────────────────

    /**
     * Detect hardware, measure bandwidths, and construct the roofline model.
     * Automatically selects the highest achievable capability tier.
     */
    static RooflineModel detect() {
        return new RooflineModel(RooflineSession.get());
    }

    /**
     * Wrap an existing roofline session (for testing or reuse).
     */
    static RooflineModel wrap(RooflineSession session) {
        return new RooflineModel(session);
    }

    // ── Scoring ────────────────────────────────────────────────────────

    /**
     * Compute PES for a kernel run.
     *
     * @param profile   kernel computational profile (FLOP count, traffic model)
     * @param elapsedSeconds  best-of-N wall-clock time
     * @return full diagnostic PES result with confidence and flags
     */
    PesResult score(KernelProfile profile, double elapsedSeconds) {
        return PesScorer.score(profile, elapsedSeconds, session);
    }

    // ── Accessors ──────────────────────────────────────────────────────

    CapabilityTier tier() {
        return session.hardware.tier;
    }

    HardwareInfo hardware() {
        return session.hardware;
    }

    double computeRoofGflops() {
        return session.peakFlopsPerSecond / 1e9;
    }

    double theoreticalPeakGflops() {
        return session.rawTheoreticalPeakFlopsPerSecond / 1e9;
    }

    double measuredGemmAnchorGflops() {
        return session.measuredGemmAnchorGflops;
    }

    MemoryBandwidthProbe.BandwidthHierarchy bandwidths() {
        return session.memoryBandwidths;
    }

    String computeRoofSource() {
        return session.computeRoofSource;
    }

    String memoryRoofSource() {
        return session.memoryRoofSource;
    }

    List<String> assumptions() {
        return session.hardware.assumptions;
    }

    RooflineSession session() {
        return session;
    }

    // ── Diagnostics ────────────────────────────────────────────────────

    /**
     * Produce a human-readable summary of the roofline model for logging.
     */
    String summary() {
        HardwareInfo hw = session.hardware;
        MemoryBandwidthProbe.BandwidthHierarchy bw = session.memoryBandwidths;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT,
            "RooflineModel [%s]%n", hw.tier.description));
        sb.append(String.format(Locale.ROOT,
            "  Compute roof:  %.2f GFLOP/s (%s)%n",
            session.peakFlopsPerSecond / 1e9, session.computeRoofSource));
        sb.append(String.format(Locale.ROOT,
            "  Theoretical:   %.2f GFLOP/s%n",
            session.rawTheoreticalPeakFlopsPerSecond / 1e9));
        if (Double.isFinite(session.measuredGemmAnchorGflops)) {
            sb.append(String.format(Locale.ROOT,
                "  GEMM anchor:   %.2f GFLOP/s%n", session.measuredGemmAnchorGflops));
        }
        sb.append(String.format(Locale.ROOT,
            "  Hardware:      %d cores, %.2f GHz, %d SIMD lanes, FMA=%s, issue_width=%d%n",
            hw.cores, hw.clockGhz, hw.simdLanesDouble, hw.fmaEnabled, hw.vectorIssueWidth));
        sb.append(String.format(Locale.ROOT,
            "  Bandwidth:     L1=%.1f, L2=%.1f, L3=%.1f, DRAM=%.1f GB/s (%s)%n",
            bw.l1BytesPerSecond / 1e9, bw.l2BytesPerSecond / 1e9,
            bw.l3BytesPerSecond / 1e9, bw.dramBytesPerSecond / 1e9,
            bw.source));
        if (!hw.assumptions.isEmpty()) {
            sb.append("  Assumptions:\n");
            for (String assumption : hw.assumptions) {
                sb.append("    - ").append(assumption).append('\n');
            }
        }
        return sb.toString();
    }

    /**
     * Return a RooflineModel with an adjusted compute roof (e.g. derived
     * from a GEMM sweep maximum).
     */
    RooflineModel withComputeRoof(double computeRoofFlopsPerSecond, String source) {
        return new RooflineModel(session.withComputeRoof(computeRoofFlopsPerSecond, source));
    }
}
