package net.faulj.autotune.search;

import net.faulj.autotune.inference.HardwareLimits;

import java.util.Locale;

/**
 * Diagnostic report for Phase 3 search space reduction.
 *
 * <p>Describes the derived parameter ranges, the constraints that
 * couple them, and the hardware limits that drove each bound.
 * Does not count or enumerate combinations.</p>
 */
public final class ReductionReport {

    /** Hardware limits that drove the reduction. */
    public final HardwareLimits inputLimits;

    /** Resulting search domain. */
    public final SearchSpace outputSpace;

    public ReductionReport(HardwareLimits inputLimits, SearchSpace outputSpace) {
        this.inputLimits = inputLimits;
        this.outputSpace = outputSpace;
    }

    @Override
    public String toString() {
        SearchSpace s = outputSpace;
        HardwareLimits hw = inputLimits;

        StringBuilder sb = new StringBuilder();
        sb.append("Phase 3 — Search Domain\n\n");

        sb.append("Microkernel:\n");
        sb.append(String.format(Locale.ROOT, "  %s\n", s.nr));
        sb.append(String.format(Locale.ROOT, "  %s\n", s.mr));
        sb.append(String.format(Locale.ROOT,
                "  Constraint: MR * ceil(NR / %d) <= %d accumulators\n",
                hw.vectorLengthDoubles, hw.maxVectorAccumulators));

        sb.append("\nInner K blocking:\n");
        sb.append(String.format(Locale.ROOT, "  %s\n", s.kc));
        sb.append(String.format(Locale.ROOT,
                "  Constraint: KC %% kUnroll == 0\n"));
        sb.append(String.format(Locale.ROOT,
                "  Bound: MR * KC * 8 <= %d bytes (L1/2)\n",
                hw.maxL1WorkingSetBytes / 2));

        sb.append("\nPanel blocking:\n");
        sb.append(String.format(Locale.ROOT, "  %s\n", s.nc));
        sb.append(String.format(Locale.ROOT,
                "  Constraint: NC %% NR == 0\n"));
        sb.append(String.format(Locale.ROOT,
                "  Bound: KC * NC * 8 <= %d bytes (L2/2)\n",
                hw.maxL2WorkingSetBytes / 2));

        sb.append("\nOuter blocking:\n");
        sb.append(String.format(Locale.ROOT, "  %s\n", s.mc));
        sb.append(String.format(Locale.ROOT,
                "  Constraint: MC %% MR == 0\n"));
        sb.append(String.format(Locale.ROOT,
                "  Bound: (MC*KC + KC*NC) * 8 <= %d bytes (L3)\n",
                hw.maxL3WorkingSetBytes));

        sb.append("\nUnroll:\n");
        sb.append(String.format(Locale.ROOT, "  %s\n", s.kUnroll));
        sb.append(String.format(Locale.ROOT,
                "  Derived from: ceil(%.1f latency / %d issue width) = %d minimum\n",
                hw.fmaPipelineLatencyCycles,
                hw.conservativeFmaIssueWidth,
                s.kUnroll.min));

        return sb.toString();
    }
}
