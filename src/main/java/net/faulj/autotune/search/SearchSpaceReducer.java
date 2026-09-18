package net.faulj.autotune.search;

import net.faulj.autotune.inference.HardwareLimits;

/**
 * Derives a reduced GEMM parameter search domain from hardware limits.
 *
 * <p>This phase does <b>not</b> run benchmarks or enumerate combinations.
 * It uses cache geometry, register pressure, and FMA pipeline characteristics
 * to produce independent parameter ranges plus formal coupling constraints.</p>
 *
 * <h3>Hierarchy (coarse to fine)</h3>
 * <ol>
 *   <li>MR/NR — microkernel register tile (fixed or very small range)</li>
 *   <li>KC — inner K blocking, sized for L1</li>
 *   <li>NC — panel blocking, sized for L2</li>
 *   <li>MC — outer blocking, sized for L3</li>
 *   <li>kUnroll — pipeline latency hiding</li>
 * </ol>
 *
 * <p>This class is stateless and thread-safe.</p>
 */
public final class SearchSpaceReducer {

    private static final int BYTES_PER_DOUBLE = 8;

    /** Minimum values (absolute floor). */
    private static final int MIN_MR = 2;
    private static final int MIN_KC = 32;
    private static final int MIN_NC = 16;
    private static final int MIN_MC = 16;
    private static final int MIN_KUNROLL = 2;

    /** Maximum values (absolute ceiling). */
    private static final int MAX_MR = 8;
    private static final int MAX_KC = 512;
    private static final int MAX_NC = 4096;
    private static final int MAX_MC = 2048;
    private static final int MAX_KUNROLL = 12;

    /** KC alignment (multiple of 8 for SIMD-friendly packing). */
    private static final int KC_ALIGN = 8;

    private SearchSpaceReducer() {}

    /**
     * Reduce the GEMM search domain using hardware limits.
     *
     * @param hw conservative hardware limits from Phase 2
     * @return search domain with ranges, constraints, and diagnostic report
     */
    public static ReductionResult reduce(HardwareLimits hw) {
        // ── 1. NR: fixed to vector length ───────────────────────────────
        int nrFixed = Math.max(2, hw.vectorLengthDoubles);
        ParameterRange nr = new ParameterRange("NR", nrFixed, nrFixed, 1);

        // ── 2. MR: constrained by register file ────────────────────────
        //
        // MR accumulators + overhead (A broadcast, B load) must not
        // exceed maxVectorAccumulators. Reserve 2 for overhead.
        int mrMax = Math.max(MIN_MR, hw.maxVectorAccumulators - 2);
        mrMax = Math.min(mrMax, MAX_MR);
        int mrMin = MIN_MR;
        if (hw.vectorLengthDoubles >= 8) {
            mrMin = Math.max(4, mrMin);   // AVX-512 sweet spot: 4-6
        } else if (hw.vectorLengthDoubles >= 4) {
            mrMin = Math.max(3, mrMin);   // AVX2 sweet spot: 3-5
        }
        mrMin = Math.min(mrMin, mrMax);
        ParameterRange mr = new ParameterRange("MR", mrMin, mrMax, 1);

        // ── 3. kUnroll: hide FMA pipeline latency ──────────────────────
        //
        // Minimum unroll = ceil(latency / issueWidth) to fill pipeline.
        // Allow exploration up to +4 beyond minimum.
        int issueWidth = Math.max(1, hw.conservativeFmaIssueWidth);
        int minUnroll = (int) Math.ceil(hw.fmaPipelineLatencyCycles / issueWidth);
        minUnroll = Math.max(MIN_KUNROLL, minUnroll);
        int maxUnroll = Math.min(MAX_KUNROLL, minUnroll + 4);
        minUnroll = Math.min(minUnroll, maxUnroll);
        ParameterRange kUnroll = new ParameterRange("kUnroll", minUnroll, maxUnroll, 1);

        // ── 4. KC: A panel [MR × KC] fits in half L1 ──────────────────
        //
        // Conservative: use max MR for the upper bound.
        long halfL1 = hw.maxL1WorkingSetBytes / 2;
        int kcMaxByL1 = (int) (halfL1 / ((long) BYTES_PER_DOUBLE * mrMax));
        int kcMaxByL1Nr = (int) (halfL1 / (2L * BYTES_PER_DOUBLE * nrFixed));
        int kcMax = Math.min(kcMaxByL1, kcMaxByL1Nr);
        kcMax = roundDown(kcMax, KC_ALIGN);
        kcMax = Math.min(kcMax, MAX_KC);
        kcMax = Math.max(kcMax, MIN_KC);
        ParameterRange kc = new ParameterRange("KC", MIN_KC, kcMax, KC_ALIGN);

        // ── 5. NC: B panel [KC × NC] fits in half L2 ──────────────────
        //
        // Use minimum KC for the widest NC upper bound.
        long halfL2 = hw.maxL2WorkingSetBytes / 2;
        int ncMax = (int) (halfL2 / ((long) BYTES_PER_DOUBLE * MIN_KC));
        ncMax = roundDown(ncMax, nrFixed);
        ncMax = Math.min(ncMax, MAX_NC);
        int ncMin = Math.max(MIN_NC, nrFixed * 4);
        ncMin = roundUp(ncMin, nrFixed);
        ncMax = Math.max(ncMin, ncMax);
        ParameterRange nc = new ParameterRange("NC", ncMin, ncMax, nrFixed);

        // ── 6. MC: working set fits in L3 ──────────────────────────────
        //
        // MC × (KC + NC) × 8 ≤ L3. Use min KC + min NC for widest bound.
        long l3Bytes = hw.maxL3WorkingSetBytes;
        int mcMax = (int) (l3Bytes / ((long) BYTES_PER_DOUBLE * (MIN_KC + ncMin)));
        mcMax = roundDown(mcMax, mrMax);
        mcMax = Math.min(mcMax, MAX_MC);
        int mcMin = Math.max(MIN_MC, mrMax * 4);
        mcMin = roundUp(mcMin, mrMax);
        mcMax = Math.max(mcMin, mcMax);
        ParameterRange mc = new ParameterRange("MC", mcMin, mcMax, mrMin);

        // ── 7. Coupling constraints ─────────────────────────────────────
        ConstraintSet constraints = new ConstraintSet(
                hw.maxVectorAccumulators,
                hw.vectorLengthDoubles,
                hw.maxL1WorkingSetBytes,
                hw.maxL2WorkingSetBytes,
                hw.maxL3WorkingSetBytes);

        // ── Build result ────────────────────────────────────────────────
        SearchSpace space = new SearchSpace(mr, nr, kc, nc, mc, kUnroll, constraints);

        ReductionReport report = new ReductionReport(hw, space);

        return new ReductionResult(space, report);
    }

    // ── Utility ─────────────────────────────────────────────────────────

    private static int roundDown(int value, int multiple) {
        return Math.max(multiple, (value / multiple) * multiple);
    }

    private static int roundUp(int value, int multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    /**
     * Result of search space reduction: domain + diagnostic report.
     */
    public static final class ReductionResult {
        public final SearchSpace space;
        public final ReductionReport report;

        public ReductionResult(SearchSpace space, ReductionReport report) {
            this.space = space;
            this.report = report;
        }
    }
}
