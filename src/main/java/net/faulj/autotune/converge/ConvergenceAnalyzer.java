package net.faulj.autotune.converge;

import net.faulj.autotune.benchmark.CoarseSearchResult;
import net.faulj.autotune.benchmark.FineSearchResult;
import net.faulj.autotune.benchmark.FineSearchTermination;

import java.util.Comparator;
import java.util.List;

/**
 * Pure analyzer that decides whether further tuning is likely to yield
 * material improvements based on Phase 4A/4B results and termination metadata.
 *
 * <p>Four strict criteria (logical AND):</p>
 * <ol>
 *   <li><b>Plateau</b> — fine search terminated via EPSILON_PLATEAU
 *       AND lastImprovementRatio &lt; epsPerf</li>
 *   <li><b>Stability</b> — selected result CV &lt; epsCv</li>
 *   <li><b>Local dominance</b> — no other fine result beats the selected
 *       by more than epsPerf</li>
 *   <li><b>Exhaustion</b> — termination is RANGE_EXHAUSTED or
 *       BUDGET_EXHAUSTED, OR plateau was already reached</li>
 * </ol>
 */
public final class ConvergenceAnalyzer {

    private ConvergenceAnalyzer() {}

    /**
     * Analyze fine search results against the coarse search baseline.
     */
    public static ConvergenceReport analyze(List<FineSearchResult> fineResults,
                                            CoarseSearchResult coarseResult,
                                            ConvergenceCriteria criteria) {
        if (fineResults == null || fineResults.isEmpty()) {
            return notConverged("No fine-search results available.", null,
                    0, 0.0, null, 0.0, 0);
        }

        int configurationsEvaluated = fineResults.size();

        // Select best fine result by bestGflops
        FineSearchResult selected = fineResults.stream()
                .max(Comparator.comparingDouble(r -> r.bestGflops))
                .orElse(null);

        if (selected == null) {
            return notConverged("Could not pick a best fine-search result.", null,
                    0, 0.0, null, 0.0, configurationsEvaluated);
        }

        FineSearchTermination termination = selected.termination;
        double lastImprovement = selected.lastImprovementRatio;

        // ── Criterion 1: Plateau ────────────────────────────────────────
        // The fine search must have stopped because improvement fell below
        // epsilon, AND the recorded last improvement must be < epsPerf.
        boolean plateauMet = (termination == FineSearchTermination.EPSILON_PLATEAU)
                && (Math.abs(lastImprovement) < criteria.epsPerf);

        // ── Criterion 2: Stability ──────────────────────────────────────
        boolean stableMet = selected.cv < criteria.epsCv;

        // ── Criterion 3: Local Dominance ────────────────────────────────
        // No other fine result beats the selected by more than epsPerf.
        boolean locallyDominant = true;
        double perfDelta = 0.0;
        for (FineSearchResult other : fineResults) {
            if (other == selected) continue;
            double delta = (other.bestGflops - selected.bestGflops) / Math.max(1e-9, selected.bestGflops);
            if (delta > perfDelta) perfDelta = delta;
            if (delta > criteria.epsPerf) {
                locallyDominant = false;
            }
        }

        // ── Criterion 4: Exhaustion ─────────────────────────────────────
        // The local search space was exhausted: search terminated via
        // range/budget exhaustion, OR an epsilon plateau was reached
        // (meaning further refinement would be pointless).
        boolean exhaustionMet = (termination == FineSearchTermination.RANGE_EXHAUSTED)
                || (termination == FineSearchTermination.BUDGET_EXHAUSTED)
                || (termination == FineSearchTermination.EPSILON_PLATEAU);

        // Also require minimum configurations evaluated
        if (configurationsEvaluated < criteria.minEvaluatedConfigs) {
            exhaustionMet = false;
        }

        // ── Strict conjunction ───────────────────────────────────────────
        boolean converged = plateauMet && stableMet && locallyDominant && exhaustionMet;

        // Build reason string
        StringBuilder reason = new StringBuilder();
        if (converged) {
            reason.append("All four convergence criteria satisfied. ");
            reason.append(String.format("KC refinement plateaued (improvement=%.4f < eps=%.4f), ",
                    lastImprovement, criteria.epsPerf));
            reason.append(String.format("measurements stable (CV=%.4f < %.4f), ",
                    selected.cv, criteria.epsCv));
            reason.append("selected config dominates alternatives, ");
            reason.append(String.format("search space exhausted (%s).", termination));
        } else {
            if (!plateauMet) {
                reason.append(String.format("Plateau NOT met: termination=%s, lastImprovement=%.4f (need EPSILON_PLATEAU and <%.4f). ",
                        termination, lastImprovement, criteria.epsPerf));
            }
            if (!stableMet) {
                reason.append(String.format("Stability NOT met: CV=%.4f >= %.4f. ",
                        selected.cv, criteria.epsCv));
            }
            if (!locallyDominant) {
                reason.append(String.format("Dominance NOT met: another config beats selected by %.4f > %.4f. ",
                        perfDelta, criteria.epsPerf));
            }
            if (!exhaustionMet) {
                reason.append(String.format("Exhaustion NOT met: termination=%s, configs=%d (need >=%d). ",
                        termination, configurationsEvaluated, criteria.minEvaluatedConfigs));
            }
        }

        return new ConvergenceReport(converged, reason.toString().trim(), selected,
                perfDelta, selected.cv,
                plateauMet, stableMet, locallyDominant, exhaustionMet,
                termination, lastImprovement, configurationsEvaluated);
    }

    private static ConvergenceReport notConverged(String reason, FineSearchResult selected,
                                                   double perfDelta, double cv,
                                                   FineSearchTermination termination,
                                                   double lastImprovement, int configs) {
        return new ConvergenceReport(false, reason, selected,
                perfDelta, cv, false, false, false, false,
                termination, lastImprovement, configs);
    }
}
