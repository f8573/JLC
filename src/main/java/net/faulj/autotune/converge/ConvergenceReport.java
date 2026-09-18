package net.faulj.autotune.converge;

import net.faulj.autotune.benchmark.FineSearchResult;
import net.faulj.autotune.benchmark.FineSearchTermination;

import java.util.Locale;

/**
 * Report produced by the ConvergenceAnalyzer.
 */
public final class ConvergenceReport {

    public final boolean converged;
    public final String reason;
    public final FineSearchResult selected;

    /** Diagnostics */
    public final double perfDelta;
    public final double cv;
    public final boolean plateauMet;
    public final boolean stableMet;
    public final boolean locallyDominant;
    public final boolean exhaustionMet;
    public final FineSearchTermination termination;
    public final double lastImprovementRatio;
    public final int configurationsEvaluated;

    public ConvergenceReport(boolean converged, String reason, FineSearchResult selected,
                             double perfDelta, double cv,
                             boolean plateauMet, boolean stableMet,
                             boolean locallyDominant, boolean exhaustionMet,
                             FineSearchTermination termination, double lastImprovementRatio,
                             int configurationsEvaluated) {
        this.converged = converged;
        this.reason = reason;
        this.selected = selected;
        this.perfDelta = perfDelta;
        this.cv = cv;
        this.plateauMet = plateauMet;
        this.stableMet = stableMet;
        this.locallyDominant = locallyDominant;
        this.exhaustionMet = exhaustionMet;
        this.termination = termination;
        this.lastImprovementRatio = lastImprovementRatio;
        this.configurationsEvaluated = configurationsEvaluated;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(converged ? "CONVERGED" : "NOT CONVERGED").append("\n\n");

        sb.append("Reason:\n");
        sb.append("  ").append(reason).append("\n\n");

        if (selected != null) {
            sb.append("Selected:\n  ").append(selected).append("\n\n");
        }

        sb.append("Criteria:\n");
        sb.append(String.format(Locale.ROOT,
                "  [%s] Plateau        — lastImprovement=%.4f (threshold=epsPerf)%n",
                plateauMet ? "PASS" : "FAIL", lastImprovementRatio));
        sb.append(String.format(Locale.ROOT,
                "  [%s] Stability      — CV=%.4f%n",
                stableMet ? "PASS" : "FAIL", cv));
        sb.append(String.format(Locale.ROOT,
                "  [%s] Dominance      — perfDelta=%.4f vs next best%n",
                locallyDominant ? "PASS" : "FAIL", perfDelta));
        sb.append(String.format(Locale.ROOT,
                "  [%s] Exhaustion     — termination=%s, configs=%d%n",
                exhaustionMet ? "PASS" : "FAIL",
                termination != null ? termination : "N/A",
                configurationsEvaluated));

        return sb.toString();
    }
}
