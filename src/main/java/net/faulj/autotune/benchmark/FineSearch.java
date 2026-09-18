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
        MeasurementTier kuTierUsed = MeasurementTier.BASE;

        int nc = findLegalNc(space.nc.min, space.nc, nr, bestKc, cs);
        int mc = findLegalMc(space.mc.min, space.mc, mr, bestKc, nc, cs);

        if (nc > 0 && mc > 0) {
            // First pass BASE_REPS for all kUnroll candidates
            java.util.List<BenchmarkHarness.Result> kuCandidates = new java.util.ArrayList<>();
            java.util.List<Integer> kuVals = new java.util.ArrayList<>();
            for (int ku = kuRange.min; ku <= kuRange.max; ku += kuRange.step) {
                if (!cs.isValidKcKUnroll(bestKc, ku)) continue;
                if (!cs.isAdmissible(mr, nr, bestKc, nc, mc, ku)) continue;

                BenchmarkHarness.Result result = BenchmarkHarness.run(BENCH_N, mc, bestKc, nc, mr, nr, 3, MeasurementTier.BASE_REPS);
                kuCandidates.add(result);
                kuVals.add(ku);

                System.out.printf(Locale.ROOT, "  kUnroll=%d -> %.1f GFLOP/s (median=%.1f, CV=%.3f) [BASE]%n",
                        ku, result.bestGflops, result.medianGflops, result.cv);
            }

            if (!kuCandidates.isEmpty()) {
                int bestKIdx = 0;
                int secondKIdx = -1;
                for (int i = 1; i < kuCandidates.size(); i++) {
                    if (kuCandidates.get(i).medianGflops > kuCandidates.get(bestKIdx).medianGflops) {
                        secondKIdx = bestKIdx;
                        bestKIdx = i;
                    } else if (secondKIdx < 0 || kuCandidates.get(i).medianGflops > kuCandidates.get(secondKIdx).medianGflops) {
                        secondKIdx = i;
                    }
                }

                BenchmarkHarness.Result bestKuResult = kuCandidates.get(bestKIdx);
                double improvementOverPrev = bestKuResult.medianGflops > 0 ? (bestKuResult.medianGflops - bestGflops) / Math.max(1e-9, bestGflops) : 0.0;
                double gapToSecond = (secondKIdx >= 0) ? Math.abs(bestKuResult.medianGflops - kuCandidates.get(secondKIdx).medianGflops) / Math.max(1e-9, bestKuResult.medianGflops) : Double.POSITIVE_INFINITY;

                boolean escalate = (bestKuResult.cv > 0.02) || (improvementOverPrev > 0.05) || (gapToSecond < 0.03);
                if (escalate) {
                    System.out.printf(Locale.ROOT, "  Escalating kUnroll=%d to MID_REPS (%d)\n", kuVals.get(bestKIdx), MeasurementTier.MID_REPS);
                    int extra = MeasurementTier.MID_REPS - bestKuResult.times.length;
                    if (extra > 0) bestKuResult = BenchmarkHarness.appendMeasurements(bestKuResult, extra);
                    kuTierUsed = MeasurementTier.MID;
                    System.out.printf(Locale.ROOT, "    MID -> median=%.3f, CV=%.4f%n", bestKuResult.medianGflops, bestKuResult.cv);
                    if (bestKuResult.cv > 0.02) {
                        System.out.printf(Locale.ROOT, "  Escalating kUnroll=%d to HIGH_REPS (%d)\n", kuVals.get(bestKIdx), MeasurementTier.HIGH_REPS);
                        int extra2 = MeasurementTier.HIGH_REPS - bestKuResult.times.length;
                        if (extra2 > 0) bestKuResult = BenchmarkHarness.appendMeasurements(bestKuResult, extra2);
                        kuTierUsed = MeasurementTier.HIGH;
                        System.out.printf(Locale.ROOT, "    HIGH -> median=%.3f, CV=%.4f%n", bestKuResult.medianGflops, bestKuResult.cv);
                    }
                }

                bestKu = kuVals.get(bestKIdx);
                bestKuGflops = bestKuResult.medianGflops;
                bestKuMedian = bestKuResult.medianGflops;
                bestKuCv = bestKuResult.cv;
            }
        }

        // Use kUnroll result if it improved over KC-only result
        double finalGflops = Math.max(bestGflops, bestKuGflops);
        double finalMedian = bestKuGflops >= bestGflops ? bestKuMedian : bestMedian;
        double finalCv = bestKuGflops >= bestGflops ? bestKuCv : bestCv;

        System.out.printf(Locale.ROOT, "%n  Selected kUnroll=%d (tier=%s)%n%n", bestKu, kuTierUsed);

        int finalNc = findLegalNc(space.nc.min, space.nc, nr, bestKc, cs);
        int finalMc = findLegalMc(space.mc.min, space.mc, mr, bestKc, finalNc, cs);

        // ── Final Revalidation Pass (mandatory) ─────────────────────────
        System.out.printf(Locale.ROOT, "Final revalidation: warmup=20, reps=25 for MR=%d NR=%d KC=%d kUnroll=%d\n",
            mr, nr, bestKc, bestKu);
        BenchmarkHarness.Result validation = BenchmarkHarness.run(BENCH_N, finalMc, bestKc, finalNc, mr, nr, 20, MeasurementTier.HIGH_REPS);
        System.out.printf(Locale.ROOT, "  Validation median=%.3f GFLOP/s, CV=%.4f\n", validation.medianGflops, validation.cv);

        // Replace metrics with hardened measurement
        double hardenedMedian = validation.medianGflops;
        double hardenedCv = validation.cv;
        double hardenedGflops = validation.medianGflops; // use median as canonical performance value

        System.out.printf(Locale.ROOT, "  Final selection: MR=%d NR=%d KC=%d kUnroll=%d, median=%.3f, CV=%.4f, repetition_tier=%s, convergence_pending...\n",
            mr, nr, bestKc, bestKu, hardenedMedian, hardenedCv, kuTierUsed);

        return new FineSearchResult(mr, nr, bestKc, bestKu, finalMc, finalNc,
            hardenedGflops, hardenedMedian, hardenedCv, termination, lastImprovementRatio);
    }

    // ── KC sweep helper ─────────────────────────────────────────────────

    private static KcResult sweepKc(int mr, int nr, int lo, int hi, int step,
                                     ParameterRange kcRange, ParameterRange kuRange,
                                     SearchSpace space, ConstraintSet cs) {
        int bestKc = -1;
        double bestGflops = 0.0;
        double bestMedian = 0.0;
        double bestCv = 1.0;
        MeasurementTier tierUsed = MeasurementTier.BASE;

        // Align lo to step
        int start = ((lo + step - 1) / step) * step;
        start = Math.max(start, kcRange.min);

        // First pass: run BASE_REPS for all candidates
        java.util.List<BenchmarkHarness.Result> candidates = new java.util.ArrayList<>();
        java.util.List<Integer> candidateKcs = new java.util.ArrayList<>();

        for (int kc = start; kc <= Math.min(hi, kcRange.max); kc += step) {
            if (kc < kcRange.min || kc > kcRange.max) continue;

            int nc = findLegalNc(space.nc.min, space.nc, nr, kc, cs);
            int mc = findLegalMc(space.mc.min, space.mc, mr, kc, nc, cs);
            int ku = findLegalKUnroll(kuRange.min, kuRange, kc, cs);

            if (nc < 0 || mc < 0 || ku < 0) continue;
            if (!cs.isAdmissible(mr, nr, kc, nc, mc, ku)) continue;

            BenchmarkHarness.Result result = BenchmarkHarness.run(BENCH_N, mc, kc, nc, mr, nr, 3, MeasurementTier.BASE_REPS);
            candidates.add(result);
            candidateKcs.add(kc);

            System.out.printf(Locale.ROOT, "  KC=%d -> %.1f GFLOP/s (median=%.1f, CV=%.3f) [BASE]%n",
                    kc, result.bestGflops, result.medianGflops, result.cv);
        }

        if (candidates.isEmpty()) return null;

        // Identify best and second best by median GFLOP/s
        int bestIdx = 0;
        int secondIdx = -1;
        for (int i = 1; i < candidates.size(); i++) {
            if (candidates.get(i).medianGflops > candidates.get(bestIdx).medianGflops) {
                secondIdx = bestIdx;
                bestIdx = i;
            } else if (secondIdx < 0 || candidates.get(i).medianGflops > candidates.get(secondIdx).medianGflops) {
                secondIdx = i;
            }
        }

        // Apply escalation to best candidate if needed
        BenchmarkHarness.Result bestResult = candidates.get(bestIdx);
        double prevBest = bestGflops; // caller may pass in prev via outer scope
        double improvementOverPrev = prevBest > 0 ? (bestResult.medianGflops - prevBest) / Math.max(1e-9, prevBest) : 0.0;
        double gapToSecond = (secondIdx >= 0) ? Math.abs(bestResult.medianGflops - candidates.get(secondIdx).medianGflops) / Math.max(1e-9, bestResult.medianGflops) : Double.POSITIVE_INFINITY;

        boolean escalateToMid = (bestResult.cv > 0.02) || (improvementOverPrev > 0.05) || (gapToSecond < 0.03);

        if (escalateToMid) {
            // escalate to MID_REPS
            System.out.printf(Locale.ROOT, "  Escalating KC=%d to MID_REPS (%d)\n", candidateKcs.get(bestIdx), MeasurementTier.MID_REPS);
            int extra = MeasurementTier.MID_REPS - bestResult.times.length;
            if (extra > 0) {
                bestResult = BenchmarkHarness.appendMeasurements(bestResult, extra);
            }
            tierUsed = MeasurementTier.MID;
            System.out.printf(Locale.ROOT, "    MID -> median=%.3f, CV=%.4f%n", bestResult.medianGflops, bestResult.cv);

            // if still unstable, escalate to HIGH
            if (bestResult.cv > 0.02) {
                System.out.printf(Locale.ROOT, "  Escalating KC=%d to HIGH_REPS (%d)\n", candidateKcs.get(bestIdx), MeasurementTier.HIGH_REPS);
                int extra2 = MeasurementTier.HIGH_REPS - bestResult.times.length;
                if (extra2 > 0) bestResult = BenchmarkHarness.appendMeasurements(bestResult, extra2);
                tierUsed = MeasurementTier.HIGH;
                System.out.printf(Locale.ROOT, "    HIGH -> median=%.3f, CV=%.4f%n", bestResult.medianGflops, bestResult.cv);
            }
        }

        // Finalize best
        bestKc = candidateKcs.get(bestIdx);
        bestGflops = bestResult.medianGflops;
        bestMedian = bestResult.medianGflops;
        bestCv = bestResult.cv;

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
