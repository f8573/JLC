package net.faulj.autotune.probe;

import net.faulj.autotune.probe.compute.FmaLatencyProbe;
import net.faulj.autotune.probe.compute.IssueWidthProbe;
import net.faulj.autotune.probe.compute.RegisterSpillProbe;
import net.faulj.autotune.probe.memory.CacheSizeProbe;
import net.faulj.autotune.probe.memory.BandwidthProbe;
import net.faulj.autotune.probe.simd.SimdQualityProbe;

import java.util.Locale;

/**
 * Runs all Phase 1 hardware probes and aggregates results into a
 * {@link ProbeResult}. Can be executed standalone from the command line.
 *
 * <p>Each probe is run independently. A failure in one probe does not
 * prevent others from running. The aggregate result carries per-field
 * confidence so downstream phases know which values to trust.</p>
 *
 * <p>Usage: {@code java --add-modules jdk.incubator.vector --enable-preview
 * -cp <classpath> net.faulj.autotune.probe.ProbeRunner}</p>
 */
public final class ProbeRunner {

    private ProbeRunner() {}

    /**
     * Run all probes and return an aggregate result.
     */
    public static ProbeResult runAll() {
        System.out.println("=== JLC Phase 1: Hardware Probes ===");
        System.out.println();

        // ── 1. Memory bandwidth ─────────────────────────────────────
        System.out.print("  [1/6] Bandwidth probe ... ");
        long t0 = System.nanoTime();
        BandwidthProbe.BandwidthResult bw = BandwidthProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, bw.confidence);

        // ── 2. Cache size inference ─────────────────────────────────
        System.out.print("  [2/6] Cache size probe ... ");
        t0 = System.nanoTime();
        CacheSizeProbe.CacheSizeResult cs = CacheSizeProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, cs.confidence);

        // ── 3. FMA issue width ──────────────────────────────────────
        System.out.print("  [3/6] Issue width probe ... ");
        t0 = System.nanoTime();
        IssueWidthProbe.IssueWidthResult iw = IssueWidthProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, iw.confidence);

        // ── 4. FMA latency ──────────────────────────────────────────
        System.out.print("  [4/6] FMA latency probe ... ");
        t0 = System.nanoTime();
        FmaLatencyProbe.FmaLatencyResult fl = FmaLatencyProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, fl.confidence);

        // ── 5. Register spill threshold ─────────────────────────────
        System.out.print("  [5/6] Register spill probe ... ");
        t0 = System.nanoTime();
        RegisterSpillProbe.SpillResult sp = RegisterSpillProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, sp.confidence);

        // ── 6. SIMD lowering quality ────────────────────────────────
        System.out.print("  [6/6] SIMD quality probe ... ");
        t0 = System.nanoTime();
        SimdQualityProbe.SimdQualityResult sq = SimdQualityProbe.run();
        System.out.printf(Locale.ROOT, "done (%.1fs) [%s]%n",
                (System.nanoTime() - t0) / 1e9, sq.confidence);

        System.out.println();

        // ── Aggregate ───────────────────────────────────────────────
        return new ProbeResult(
                cs.l1Bytes, cs.l2Bytes, cs.l3Bytes,
                bw.l1BytesPerSec, bw.l2BytesPerSec, bw.l3BytesPerSec, bw.dramBytesPerSec,
                iw.effectiveIssueWidth, sq.quality,
                sp.spillThreshold, (double) fl.latencyCycles,
                sq.vectorLength, iw.fmaSupported,
                cs.confidence, bw.confidence,
                iw.confidence, sq.confidence,
                sp.confidence, fl.confidence);
    }

    // ── Standalone entry point ──────────────────────────────────────

    public static void main(String[] args) {
        long wallStart = System.nanoTime();

        ProbeResult result = runAll();

        double wallSeconds = (System.nanoTime() - wallStart) / 1e9;

        System.out.println("=== Probe Results ===");
        System.out.println(result);
        System.out.printf(Locale.ROOT, "Total probe time: %.1fs%n", wallSeconds);
        System.out.println();

        // ── Structured output ───────────────────────────────────────
        System.out.println("=== Structured Output (machine-readable) ===");
        System.out.printf(Locale.ROOT,
            "{"
            + "%n  \"effective_l1_bytes\": %d,"
            + "%n  \"effective_l2_bytes\": %d,"
            + "%n  \"effective_l3_bytes\": %d,"
            + "%n  \"bandwidth_l1_gbps\": %.2f,"
            + "%n  \"bandwidth_l2_gbps\": %.2f,"
            + "%n  \"bandwidth_l3_gbps\": %.2f,"
            + "%n  \"bandwidth_dram_gbps\": %.2f,"
            + "%n  \"effective_issue_width\": %d,"
            + "%n  \"simd_lowering_quality\": %.4f,"
            + "%n  \"register_spill_threshold\": %d,"
            + "%n  \"fma_latency_cycles\": %.2f,"
            + "%n  \"vector_length\": %d,"
            + "%n  \"fma_supported\": %s,"
            + "%n  \"confidence\": {"
            + "%n    \"cache_sizes\": \"%s\","
            + "%n    \"bandwidth\": \"%s\","
            + "%n    \"issue_width\": \"%s\","
            + "%n    \"simd_quality\": \"%s\","
            + "%n    \"spill_threshold\": \"%s\","
            + "%n    \"fma_latency\": \"%s\""
            + "%n  },"
            + "%n  \"total_probe_seconds\": %.2f"
            + "%n}%n",
            result.effectiveL1Bytes,
            result.effectiveL2Bytes,
            result.effectiveL3Bytes,
            result.bandwidthL1 / 1e9,
            result.bandwidthL2 / 1e9,
            result.bandwidthL3 / 1e9,
            result.bandwidthDram / 1e9,
            result.effectiveIssueWidth,
            result.simdLoweringQuality,
            result.registerSpillThreshold,
            result.fmaLatencyCycles,
            result.vectorLength,
            result.fmaSupported,
            result.cacheSizeConfidence,
            result.bandwidthConfidence,
            result.issueWidthConfidence,
            result.simdQualityConfidence,
            result.spillThresholdConfidence,
            result.fmaLatencyConfidence,
            wallSeconds);
    }
}
