package net.faulj.autotune.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Formal coupling constraints between GEMM blocking parameters.
 *
 * <p>These predicates define <b>admissibility</b> — which parameter
 * combinations are structurally valid given hardware limits. They
 * encode the hierarchical dependencies:</p>
 *
 * <ol>
 *   <li>MR/NR define register pressure (microkernel tier)</li>
 *   <li>KC defines inner-loop shape (L1 tier)</li>
 *   <li>NC defines L2 panel reuse (L2 tier)</li>
 *   <li>MC defines thread-level blocking (L3 tier)</li>
 *   <li>kUnroll refines latency hiding (pipeline tier)</li>
 * </ol>
 *
 * <p>Phase 4 queries these constraints to prune during hierarchical
 * search, not by enumerating the full cartesian product.</p>
 */
public final class ConstraintSet {

    private final int maxAccumulators;
    private final int vectorLength;
    private final long maxL1WorkingSetBytes;
    private final long maxL2WorkingSetBytes;
    private final long maxL3WorkingSetBytes;

    private static final int BYTES_PER_DOUBLE = 8;

    public ConstraintSet(int maxAccumulators, int vectorLength,
                         long maxL1WorkingSetBytes, long maxL2WorkingSetBytes,
                         long maxL3WorkingSetBytes) {
        this.maxAccumulators = maxAccumulators;
        this.vectorLength = vectorLength;
        this.maxL1WorkingSetBytes = maxL1WorkingSetBytes;
        this.maxL2WorkingSetBytes = maxL2WorkingSetBytes;
        this.maxL3WorkingSetBytes = maxL3WorkingSetBytes;
    }

    // ── Microkernel tier ────────────────────────────────────────────────

    /**
     * MR × ceil(NR / vectorLength) must not exceed the register file.
     * Each MR row needs one accumulator vector; NR/vecLen vectors per row.
     */
    public boolean isValidMrNr(int mr, int nr) {
        int vectorsPerRow = (nr + vectorLength - 1) / vectorLength;
        return mr * vectorsPerRow <= maxAccumulators;
    }

    // ── Alignment tier ──────────────────────────────────────────────────

    /** MC must be a multiple of MR for clean microkernel tiling. */
    public boolean isValidMcMr(int mc, int mr) {
        return mc % mr == 0;
    }

    /** NC must be a multiple of NR for SIMD-aligned column blocking. */
    public boolean isValidNcNr(int nc, int nr) {
        return nc % nr == 0;
    }

    /** KC must be a multiple of kUnroll for clean inner-loop unrolling. */
    public boolean isValidKcKUnroll(int kc, int kUnroll) {
        return kc % kUnroll == 0;
    }

    // ── Cache tier ──────────────────────────────────────────────────────

    /**
     * A panel [MR × KC] must fit in half of L1.
     * Leaves room for B slice and C tile in the other half.
     */
    public boolean isValidKcMrForL1(int kc, int mr) {
        long aPanelBytes = (long) mr * kc * BYTES_PER_DOUBLE;
        return aPanelBytes <= maxL1WorkingSetBytes / 2;
    }

    /**
     * B panel [KC × NC] must fit in half of L2.
     * Leaves room for A streaming and C writeback.
     */
    public boolean isValidKcNcForL2(int kc, int nc) {
        long bPanelBytes = (long) kc * nc * BYTES_PER_DOUBLE;
        return bPanelBytes <= maxL2WorkingSetBytes / 2;
    }

    /**
     * Combined working set MC×KC (A) + KC×NC (B) must fit in L3.
     */
    public boolean isValidMcKcNcForL3(int mc, int kc, int nc) {
        long aBytes = (long) mc * kc * BYTES_PER_DOUBLE;
        long bBytes = (long) kc * nc * BYTES_PER_DOUBLE;
        return (aBytes + bBytes) <= maxL3WorkingSetBytes;
    }

    // ── Composite validation ────────────────────────────────────────────

    /**
     * Check all constraints for a complete parameter set.
     * Used by Phase 4 to validate a candidate before benchmarking.
     */
    public boolean isAdmissible(int mr, int nr, int kc, int nc, int mc, int kUnroll) {
        return isValidMrNr(mr, nr)
            && isValidMcMr(mc, mr)
            && isValidNcNr(nc, nr)
            && isValidKcKUnroll(kc, kUnroll)
            && isValidKcMrForL1(kc, mr)
            && isValidKcNcForL2(kc, nc)
            && isValidMcKcNcForL3(mc, kc, nc);
    }

    /**
     * Returns all constraints as human-readable descriptions.
     */
    public List<String> describe() {
        List<String> out = new ArrayList<>();
        out.add(String.format(Locale.ROOT,
                "MR * ceil(NR / %d) <= %d accumulators", vectorLength, maxAccumulators));
        out.add("MC %% MR == 0");
        out.add("NC %% NR == 0");
        out.add("KC %% kUnroll == 0");
        out.add(String.format(Locale.ROOT,
                "MR * KC * %d <= %d bytes (L1/2)", BYTES_PER_DOUBLE, maxL1WorkingSetBytes / 2));
        out.add(String.format(Locale.ROOT,
                "KC * NC * %d <= %d bytes (L2/2)", BYTES_PER_DOUBLE, maxL2WorkingSetBytes / 2));
        out.add(String.format(Locale.ROOT,
                "(MC*KC + KC*NC) * %d <= %d bytes (L3)", BYTES_PER_DOUBLE, maxL3WorkingSetBytes));
        return Collections.unmodifiableList(out);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Constraints:\n");
        for (String c : describe()) {
            sb.append("    ").append(c).append('\n');
        }
        return sb.toString();
    }
}
