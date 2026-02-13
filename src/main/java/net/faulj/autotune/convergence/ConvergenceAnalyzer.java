package net.faulj.autotune.convergence;

import net.faulj.autotune.benchmark.FineSearchResult;
import net.faulj.autotune.benchmark.FineSearchTermination;
import net.faulj.autotune.converge.ConvergenceReport;
import net.faulj.autotune.converge.ConvergenceCriteria;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Phase 5: Convergence analysis over Phase 4B fine search results.
 *
 * <p>This class is <b>pure</b>: no side effects, no benchmarks, no I/O.
 * Same inputs always produce the same output.</p>
 *
 * <p>Convergence requires <b>all four</b> criteria to hold simultaneously:</p>
 * <ol>
 *   <li><b>Plateau:</b> relative improvement between best and runner-up &lt; ε_perf</li>
 *   <li><b>Stability:</b> selected configuration's CV &lt; ε_cv</li>
 *   <li><b>Dominance:</b> no adjacent configuration beats the selected by &gt; ε_perf</li>
 *   <li><b>Exhaustion:</b> sufficient configurations evaluated</li>
 * </ol>
 *
 * <p>No voting. No weighting. No fuzzy logic. Strict conjunction.</p>
 */
public final class ConvergenceAnalyzer {

    private ConvergenceAnalyzer() {}

    /**
     * Analyze convergence of Phase 4B results.
     *
     * @param results fine search results from Phase 4B (one per shape)
     * @param criteria convergence thresholds
     * @return convergence report with decision and diagnostics
     */
    public static ConvergenceReport analyze(List<FineSearchResult> results,
                                             ConvergenceCriteria criteria) {
        // ── Guard: insufficient data ────────────────────────────────────
        if (results == null || results.isEmpty()) {
            return insufficientData(criteria, "No fine search results provided");
        }

        // ── Sort by best GFLOP/s descending ─────────────────────────────
        List<FineSearchResult> ranked = new ArrayList<>(results);
        ranked.sort(Comparator.comparingDouble((FineSearchResult r) -> r.bestGflops).reversed());

        FineSearchResult best = ranked.get(0);
        int totalConfigs = ranked.size();

        // ── Criterion 1: Performance Plateau ────────────────────────────
        double performanceSlope;
        boolean plateauMet;
        String plateauReason;

        if (ranked.size() >= 2) {
            FineSearchResult runnerUp = ranked.get(1);
            double runnerUpGflops = runnerUp.bestGflops;
                performanceSlope = (best.bestGflops - runnerUpGflops)
                    / Math.max(1e-9, runnerUpGflops);
                plateauMet = performanceSlope < criteria.epsPerf;
                plateauReason = String.format(
                "Performance delta=%.4f (%.1f vs %.1f GFLOP/s), threshold=%.4f -> %s",
                performanceSlope, best.bestGflops, runnerUpGflops,
                criteria.epsPerf, plateauMet ? "PLATEAU" : "still improving");
        } else {
            // Single configuration: plateau is trivially met (no competitor)
            performanceSlope = 0.0;
            plateauMet = true;
            plateauReason = "Single configuration evaluated, plateau trivially met";
        }

        // ── Criterion 2: Stability ──────────────────────────────────────
        double selectedCv = best.cv;
        boolean stableMet = selectedCv < criteria.epsCv;
        String stabilityReason = String.format(
            "CV=%.4f, threshold=%.4f -> %s",
            selectedCv, criteria.epsCv, stableMet ? "STABLE" : "unstable");

        // ── Criterion 3: Local Dominance ────────────────────────────────
        //
        // The selected configuration must not be beaten by any other
        // evaluated configuration by more than ε_perf (relative).
        // "Adjacent" here means any configuration tested in the same
        // Phase 4B refinement — all results in the list share the
        // same parameter neighborhood from hierarchical search.
        boolean dominanceMet = true;
        String dominanceReason = "No configuration exceeds selected by > epsilon";

        for (FineSearchResult other : ranked) {
            if (other == best) continue;
            double relativeAdvantage = (other.bestGflops - best.bestGflops)
                    / Math.max(1e-9, best.bestGflops);
            if (relativeAdvantage > criteria.epsPerf) {
                dominanceMet = false;
                dominanceReason = String.format(
                    "MR=%d,NR=%d,KC=%d beats selected by %.4f (> %.4f)",
                    other.mr, other.nr, other.bestKc,
                    relativeAdvantage, criteria.epsPerf);
                break;
            }
        }

        // ── Criterion 4: Search Exhaustion ──────────────────────────────
        boolean exhaustionMet = totalConfigs >= criteria.minEvaluatedConfigs;
        String exhaustionReason = String.format(
            "%d configurations evaluated (minimum=%d) -> %s",
            totalConfigs, criteria.minEvaluatedConfigs,
            exhaustionMet ? "EXHAUSTED" : "insufficient exploration");

        // ── Decision: strict conjunction ─────────────────────────────────
        boolean converged = plateauMet && stableMet && dominanceMet && exhaustionMet;

        // ── Build reason string ─────────────────────────────────────────
        StringBuilder reason = new StringBuilder();
        reason.append(plateauReason).append('\n');
        reason.append(stabilityReason).append('\n');
        reason.append(dominanceReason).append('\n');
        reason.append(exhaustionReason);

        FineSearchTermination termination = best.termination;
        double lastImprovement = best.lastImprovementRatio;

        return new ConvergenceReport(
            converged, reason.toString(), best,
            performanceSlope, selectedCv,
            plateauMet, stableMet,
            dominanceMet, exhaustionMet,
            termination, lastImprovement,
            totalConfigs);
    }

    /**
     * Analyze with default criteria.
     */
    public static ConvergenceReport analyze(List<FineSearchResult> results) {
        return analyze(results, new ConvergenceCriteria());
    }

    private static ConvergenceReport insufficientData(ConvergenceCriteria criteria,
                                                       String message) {
        // Create a dummy result for the report
        FineSearchResult dummy = new FineSearchResult(
            0, 0, 0, 0, 0, 0,
            0.0, 0.0, 1.0,
            FineSearchTermination.RANGE_EXHAUSTED,
            0.0);
        return new ConvergenceReport(
            false, message, dummy,
            0.0, dummy.cv,
            false, false,
            false, false,
            dummy.termination, dummy.lastImprovementRatio,
            0);
    }
}
