package net.faulj.autotune.benchmark;

import net.faulj.autotune.search.ConstraintSet;
import net.faulj.autotune.search.ParameterRange;
import net.faulj.autotune.search.SearchSpace;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Phase 4B: Fine search over KC and kUnroll for a fixed (MR, NR) shape.
 *
 * <p>Performs 1-D refinement of KC using progressively smaller steps,
 * then sweeps kUnroll with KC fixed. NC and MC are held at minimum
 * legal values throughout.</p>
 *
 * <p>Does not sweep NC or MC. Does not enumerate cartesian products.</p>
 */
public final class FineSearch {

    /** Square matrix dimension for fine benchmarks. */
    private static final int BENCH_N = 512;

    /** Stop KC refinement when improvement drops below this threshold. */
    private static final double IMPROVEMENT_THRESHOLD = 0.02;

    /** Minimum KC step size before stopping refinement. */
    private static final int MIN_KC_STEP = 2;

    private FineSearch() {}

    /**
     * Run fine search for each selected shape from Phase 4A.
     *
     * @param coarseResult Phase 4A results (selected shapes)
     * @param space        search domain from Phase 3
     * @return fine search results for each shape
     */
    public static List<FineSearchResult> run(CoarseSearchResult coarseResult,
                                              SearchSpace space) {
        List<FineSearchResult> results = new ArrayList<>();

        for (CoarseSearchResult.ShapeResult shape : coarseResult.selectedShapes) {
            FineSearchResult result = refineShape(shape, space);
            results.add(result);
        }

        return results;
    }

    /**
     * Refine KC and kUnroll for one (MR, NR) shape.
     */
    private static FineSearchResult refineShape(CoarseSearchResult.ShapeResult shape,
                                                 SearchSpace space) {
        int mr = shape.mr;
        int nr = shape.nr;
        ConstraintSet cs = space.constraints;
        ParameterRange kcRange = space.kc;
        ParameterRange kuRange = space.kUnroll;

        System.out.printf(Locale.ROOT, "Phase 4B — Fine Search (MR=%d, NR=%d)%n%n", mr, nr);

        // ── Fixed minimal NC and MC ─────────────────────────────────────
        int ncMin = findLegalNc(space.nc.min, space.nc, nr, kcRange.min, cs);
        int kuMin = findLegalKUnroll(kuRange.min, kuRange, kcRange.min, cs);

        // ── Phase 1: KC refinement ──────────────────────────────────────
        System.out.println("KC sweep:");

        // Start from the coarse best KC (parse from kcTested string)
        int coarseBestKc = findCoarseBestKc(shape, kcRange);
        int currentStep = kcRange.step;
        int bestKc = coarseBestKc;
        double bestGflops = 0.0;
        double bestMedian = 0.0;
        double bestCv = 1.0;

        // Termination tracking
        FineSearchTermination termination = FineSearchTermination.BUDGET_EXHAUSTED;
        double lastImprovementRatio = 0.0;

        // Initial sweep at coarse step across the full range
        KcResult initial = sweepKc(mr, nr, kcRange.min, kcRange.max, currentStep,
                kcRange, kuRange, space, cs);
        if (initial != null) {
            bestKc = initial.kc;
            bestGflops = initial.gflops;
            bestMedian = initial.median;
            bestCv = initial.cv;
        }

        // Check if best KC is at a range boundary after initial sweep
        boolean atBound = (bestKc == kcRange.min || bestKc == kcRange.max);

        // Progressive refinement: halve step, search around best KC
        boolean refinementRan = false;
        while (currentStep > MIN_KC_STEP) {
            int finerStep = Math.max(MIN_KC_STEP, currentStep / 2);
            if (finerStep < MIN_KC_STEP) break;

            int lo = Math.max(kcRange.min, bestKc - currentStep);
            int hi = Math.min(kcRange.max, bestKc + currentStep);

            KcResult refined = sweepKc(mr, nr, lo, hi, finerStep,
                    kcRange, kuRange, space, cs);

            if (refined == null) {
                termination = FineSearchTermination.RANGE_EXHAUSTED;
                break;
            }

            refinementRan = true;
            double prevBest = bestGflops;
            double improvement = (refined.gflops - bestGflops) / Math.max(1e-9, bestGflops);
            lastImprovementRatio = improvement;

            if (refined.gflops > bestGflops) {
                bestKc = refined.kc;
                bestGflops = refined.gflops;
                bestMedian = refined.median;
                bestCv = refined.cv;
            }

            if (improvement < IMPROVEMENT_THRESHOLD) {
                termination = FineSearchTermination.EPSILON_PLATEAU;
                break;
            }
            currentStep = finerStep;
        }

        // If the loop exited normally (step reached MIN_KC_STEP), classify
        if (termination == FineSearchTermination.BUDGET_EXHAUSTED && refinementRan) {
            // Step size bottomed out — budget/resolution exhausted
            if (bestKc == kcRange.min || bestKc == kcRange.max) {
                termination = FineSearchTermination.RANGE_EXHAUSTED;
            }
        }

        // Check stability of the best KC result
        if (bestCv > IMPROVEMENT_THRESHOLD) {
            termination = FineSearchTermination.UNSTABLE;
        }

        System.out.printf(Locale.ROOT, "%n  Selected KC=%d (%.1f GFLOP/s, termination=%s)%n%n",
                bestKc, bestGflops, termination);

        // ── Phase 2: kUnroll sweep (KC now fixed) ───────────────────────
        System.out.println("kUnroll sweep:");

        int bestKu = kuMin;
        double bestKuGflops = 0.0;
        double bestKuMedian = 0.0;
        double bestKuCv = 1.0;

        int nc = findLegalNc(space.nc.min, space.nc, nr, bestKc, cs);
        int mc = findLegalMc(space.mc.min, space.mc, mr, bestKc, nc, cs);

        if (nc > 0 && mc > 0) {
            for (int ku = kuRange.min; ku <= kuRange.max; ku += kuRange.step) {
                if (!cs.isValidKcKUnroll(bestKc, ku)) continue;
                if (!cs.isAdmissible(mr, nr, bestKc, nc, mc, ku)) continue;

                BenchmarkHarness.Result result =
                        BenchmarkHarness.run(BENCH_N, mc, bestKc, nc, mr, nr);

                System.out.printf(Locale.ROOT, "  kUnroll=%d -> %.1f GFLOP/s%n",
                        ku, result.bestGflops);

                if (result.bestGflops > bestKuGflops) {
                    bestKu = ku;
                    bestKuGflops = result.bestGflops;
                    bestKuMedian = result.medianGflops;
                    bestKuCv = result.cv;
                }
            }
        }

        // Use kUnroll result if it improved over KC-only result
        double finalGflops = Math.max(bestGflops, bestKuGflops);
        double finalMedian = bestKuGflops >= bestGflops ? bestKuMedian : bestMedian;
        double finalCv = bestKuGflops >= bestGflops ? bestKuCv : bestCv;

        System.out.printf(Locale.ROOT, "%n  Selected kUnroll=%d%n%n", bestKu);

        int finalNc = findLegalNc(space.nc.min, space.nc, nr, bestKc, cs);
        int finalMc = findLegalMc(space.mc.min, space.mc, mr, bestKc, finalNc, cs);

        return new FineSearchResult(mr, nr, bestKc, bestKu, finalMc, finalNc,
                finalGflops, finalMedian, finalCv, termination, lastImprovementRatio);
    }

    // ── KC sweep helper ─────────────────────────────────────────────────

    private static KcResult sweepKc(int mr, int nr, int lo, int hi, int step,
                                     ParameterRange kcRange, ParameterRange kuRange,
                                     SearchSpace space, ConstraintSet cs) {
        int bestKc = -1;
        double bestGflops = 0.0;
        double bestMedian = 0.0;
        double bestCv = 1.0;

        // Align lo to step
        int start = ((lo + step - 1) / step) * step;
        start = Math.max(start, kcRange.min);

        for (int kc = start; kc <= Math.min(hi, kcRange.max); kc += step) {
            if (kc < kcRange.min || kc > kcRange.max) continue;

            int nc = findLegalNc(space.nc.min, space.nc, nr, kc, cs);
            int mc = findLegalMc(space.mc.min, space.mc, mr, kc, nc, cs);
            int ku = findLegalKUnroll(kuRange.min, kuRange, kc, cs);

            if (nc < 0 || mc < 0 || ku < 0) continue;
            if (!cs.isAdmissible(mr, nr, kc, nc, mc, ku)) continue;

            BenchmarkHarness.Result result =
                    BenchmarkHarness.run(BENCH_N, mc, kc, nc, mr, nr);

            System.out.printf(Locale.ROOT, "  KC=%d -> %.1f GFLOP/s%n",
                    kc, result.bestGflops);

            if (result.bestGflops > bestGflops) {
                bestKc = kc;
                bestGflops = result.bestGflops;
                bestMedian = result.medianGflops;
                bestCv = result.cv;
            }
        }

        if (bestKc < 0) return null;
        return new KcResult(bestKc, bestGflops, bestMedian, bestCv);
    }

    private static final class KcResult {
        final int kc;
        final double gflops;
        final double median;
        final double cv;

        KcResult(int kc, double gflops, double median, double cv) {
            this.kc = kc;
            this.gflops = gflops;
            this.median = median;
            this.cv = cv;
        }
    }

    // ── Constraint helpers (same logic as CoarseSearch) ─────────────────

    private static int findLegalNc(int ncStart, ParameterRange ncRange,
                                    int nr, int kc, ConstraintSet cs) {
        int ncFirst = ((ncStart + nr - 1) / nr) * nr;
        for (int nc = ncFirst; nc <= ncRange.max; nc += nr) {
            if (cs.isValidNcNr(nc, nr) && cs.isValidKcNcForL2(kc, nc)) {
                return nc;
            }
        }
        return -1;
    }

    private static int findLegalMc(int mcStart, ParameterRange mcRange,
                                    int mr, int kc, int nc, ConstraintSet cs) {
        if (nc < 0) return -1;
        int mcFirst = ((mcStart + mr - 1) / mr) * mr;
        for (int mc = mcFirst; mc <= mcRange.max; mc += mr) {
            if (cs.isValidMcMr(mc, mr) && cs.isValidMcKcNcForL3(mc, kc, nc)) {
                return mc;
            }
        }
        return -1;
    }

    private static int findLegalKUnroll(int kuStart, ParameterRange kuRange,
                                         int kc, ConstraintSet cs) {
        for (int ku = kuStart; ku <= kuRange.max; ku += kuRange.step) {
            if (cs.isValidKcKUnroll(kc, ku)) {
                return ku;
            }
        }
        return -1;
    }

    /**
     * Extract the best KC from the coarse search kcTested string.
     * Format: "32:5.1, 112:8.7, 192:9.7"
     */
    private static int findCoarseBestKc(CoarseSearchResult.ShapeResult shape,
                                         ParameterRange kcRange) {
        String kcStr = shape.kcTested;
        if (kcStr == null || kcStr.isEmpty()) return kcRange.min;

        int bestKc = kcRange.min;
        double bestVal = 0.0;
        for (String part : kcStr.split(",")) {
            part = part.trim();
            int colon = part.indexOf(':');
            if (colon < 0) continue;
            try {
                int kc = Integer.parseInt(part.substring(0, colon).trim());
                double gf = Double.parseDouble(part.substring(colon + 1).trim());
                if (gf > bestVal) {
                    bestVal = gf;
                    bestKc = kc;
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return bestKc;
    }
}
