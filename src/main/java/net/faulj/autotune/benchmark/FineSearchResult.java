package net.faulj.autotune.benchmark;

import java.util.Locale;

/**
 * Result of Phase 4B fine search for one microkernel shape.
 *
 * <p>Contains the finalized KC and kUnroll values along with
 * achieved performance, stability metrics, and termination metadata
 * for Phase 5 convergence analysis.</p>
 */
public final class FineSearchResult {

    public final int mr;
    public final int nr;
    public final int bestKc;
    public final int bestKUnroll;
    public final int mc;
    public final int nc;
    public final double bestGflops;
    public final double medianGflops;
    public final double cv;

    /** Why the KC refinement loop stopped. */
    public final FineSearchTermination termination;

    /** Relative improvement in the last refinement iteration: (prev - curr) / prev. */
    public final double lastImprovementRatio;

    public FineSearchResult(int mr, int nr, int bestKc, int bestKUnroll,
                            int mc, int nc,
                            double bestGflops, double medianGflops, double cv,
                            FineSearchTermination termination,
                            double lastImprovementRatio) {
        this.mr = mr;
        this.nr = nr;
        this.bestKc = bestKc;
        this.bestKUnroll = bestKUnroll;
        this.mc = mc;
        this.nc = nc;
        this.bestGflops = bestGflops;
        this.medianGflops = medianGflops;
        this.cv = cv;
        this.termination = termination;
        this.lastImprovementRatio = lastImprovementRatio;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT,
            "FineSearchResult { MR=%d, NR=%d, KC=%d, kUnroll=%d, MC=%d, NC=%d, " +
            "%.1f GFLOP/s (median %.1f, CV=%.3f), termination=%s, lastImprovement=%.4f }",
            mr, nr, bestKc, bestKUnroll, mc, nc,
            bestGflops, medianGflops, cv, termination, lastImprovementRatio);
    }
}
