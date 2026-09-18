package net.faulj.benchmark.roofline;

/**
 * Complete diagnostic result of a single PES evaluation.
 *
 * <p>Every PES result includes:
 * <ul>
 *   <li>The canonical PES score (measured / min(compute_roof, memory_roof))</li>
 *   <li>Per-regime PES variants (PES_L1 through PES_DRAM) for diagnostics</li>
 *   <li>Confidence level based on how the roofline was constructed</li>
 *   <li>Flags if PES exceeds 1.0 (model error or measurement error)</li>
 *   <li>All intermediate values for auditability</li>
 * </ul>
 *
 * <p><b>PES is never clamped.</b>  If PES &gt; 1.0, the raw value is
 * preserved and a flag is set.  Silent clamping hides model errors.</p>
 */
final class PesResult {
    // ── Identity ───────────────────────────────────────────────────────
    final String kernel;
    final int m;
    final int n;
    final int k;

    // ── Roofline inputs ────────────────────────────────────────────────
    final double arithmeticIntensity;
    final String trafficModel;  // "blis-blocked" or "cold-start"
    final String boundType;     // "compute" or "memory"
    final String memoryLevel;   // "l1", "l2", "l3", "dram"

    // ── Utilization fractions ──────────────────────────────────────────
    final double computeUtilization;   // measured / compute_roof
    final double memoryUtilization;    // measured_bytes/s / selected_bandwidth
    final double algorithmicEfficiency; // theoretical_flops / actual_flops

    // ── The score ──────────────────────────────────────────────────────
    /** Canonical PES = measured / min(compute_roof, memory_roof).  NOT clamped. */
    final double portableEfficiencyScore;

    // ── Regime-aware PES variants (diagnostic) ─────────────────────────
    final double pesL1;
    final double pesL2;
    final double pesL3;
    final double pesDram;

    // ── Absolute values ────────────────────────────────────────────────
    final double measuredGflops;
    final double computeRoofGflops;
    final double roofGflops;   // min(compute, memory) — the effective ceiling
    final double selectedMemoryRoofGbps;
    final double elapsedSeconds;

    // ── Confidence and flags ───────────────────────────────────────────
    /** MEASURED | ESTIMATED | SAFE_MODE | UNDEFINED */
    final String confidence;
    /** OK | MODEL_ERROR | MEASUREMENT_ERROR | UNDEFINED */
    final String flag;

    PesResult(String kernel, int m, int n, int k,
              double arithmeticIntensity, String trafficModel,
              String boundType, String memoryLevel,
              double computeUtilization, double memoryUtilization,
              double algorithmicEfficiency,
              double portableEfficiencyScore,
              double pesL1, double pesL2, double pesL3, double pesDram,
              double measuredGflops, double computeRoofGflops,
              double roofGflops, double selectedMemoryRoofGbps,
              double elapsedSeconds,
              String confidence, String flag) {
        this.kernel = kernel;
        this.m = m;
        this.n = n;
        this.k = k;
        this.arithmeticIntensity = arithmeticIntensity;
        this.trafficModel = trafficModel;
        this.boundType = boundType;
        this.memoryLevel = memoryLevel;
        this.computeUtilization = computeUtilization;
        this.memoryUtilization = memoryUtilization;
        this.algorithmicEfficiency = algorithmicEfficiency;
        this.portableEfficiencyScore = portableEfficiencyScore;
        this.pesL1 = pesL1;
        this.pesL2 = pesL2;
        this.pesL3 = pesL3;
        this.pesDram = pesDram;
        this.measuredGflops = measuredGflops;
        this.computeRoofGflops = computeRoofGflops;
        this.roofGflops = roofGflops;
        this.selectedMemoryRoofGbps = selectedMemoryRoofGbps;
        this.elapsedSeconds = elapsedSeconds;
        this.confidence = confidence;
        this.flag = flag;
    }
}
