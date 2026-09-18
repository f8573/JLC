package net.faulj.autotune.benchmark;

import net.faulj.autotune.search.ConstraintSet;
import net.faulj.autotune.search.ParameterRange;
import net.faulj.autotune.search.SearchSpace;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Phase 4A: Coarse search over microkernel shapes.
 *
 * <p>Iterates only over (MR, NR) pairs from the search domain.
 * For each pair, benchmarks three representative KC values with
 * minimal NC, MC, and kUnroll. Produces a ranked list of
 * microkernel shapes for Phase 4B fine search.</p>
 *
 * <p>Does not enumerate the full domain. Does not modify GEMM kernels.</p>
 */
public final class CoarseSearch {

    /** Square matrix dimension for coarse benchmarks. */
    private static final int BENCH_N = 512;

    /** Maximum shapes to retain for fine search. */
    private static final int MAX_SELECTED = 3;

    private CoarseSearch() {}

    /**
     * Run the coarse search.
     *
     * @param space search domain from Phase 3
     * @return ranked microkernel shapes with GFLOP/s measurements
     */
    public static CoarseSearchResult run(SearchSpace space) {
        System.out.println("Phase 4A — Coarse Search (N=" + BENCH_N + ")");
        System.out.println();

        ConstraintSet constraints = space.constraints;
        ParameterRange mrRange = space.mr;
        ParameterRange nrRange = space.nr;
        ParameterRange kcRange = space.kc;

        // Minimal blocking: smallest legal NC, MC, kUnroll
        int ncMin = space.nc.min;
        int mcMin = space.mc.min;
        int kuMin = space.kUnroll.min;

        // Select three representative KC values: min, mid, max
        int kcMin = kcRange.min;
        int kcMax = kcRange.max;
        int kcMid = roundToStep(kcMin + (kcMax - kcMin) / 2, kcRange.step);
        kcMid = Math.max(kcMin, Math.min(kcMax, kcMid));
        int[] kcPoints = distinctSorted(kcMin, kcMid, kcMax);

        List<CoarseSearchResult.ShapeResult> shapes = new ArrayList<>();

        // ── Iterate (MR, NR) pairs ──────────────────────────────────────
        for (int mr = mrRange.min; mr <= mrRange.max; mr += mrRange.step) {
            for (int nr = nrRange.min; nr <= nrRange.max; nr += nrRange.step) {

                // Check microkernel constraint
                if (!constraints.isValidMrNr(mr, nr)) continue;

                System.out.printf(Locale.ROOT, "  Testing MR=%d, NR=%d ... ", mr, nr);

                double bestGflops = 0.0;
                double bestMedian = 0.0;
                double bestCv = 1.0;
                StringBuilder kcDesc = new StringBuilder();

                // ── Three KC probe points ───────────────────────────────
                for (int kc : kcPoints) {
                    // Adjust NC, MC, kUnroll to satisfy all constraints
                    int nc = findLegalNc(ncMin, space.nc, nr, kc, constraints);
                    int mc = findLegalMc(mcMin, space.mc, mr, kc, nc, constraints);
                    int ku = findLegalKUnroll(kuMin, space.kUnroll, kc, constraints);

                    if (nc < 0 || mc < 0 || ku < 0) continue;

                    // Final admissibility check
                    if (!constraints.isAdmissible(mr, nr, kc, nc, mc, ku)) continue;

                    BenchmarkHarness.Result result =
                            BenchmarkHarness.run(BENCH_N, mc, kc, nc, mr, nr);

                    if (kcDesc.length() > 0) kcDesc.append(", ");
                    kcDesc.append(String.format(Locale.ROOT, "%d:%.1f", kc, result.bestGflops));

                    if (result.bestGflops > bestGflops) {
                        bestGflops = result.bestGflops;
                        bestMedian = result.medianGflops;
                        bestCv = result.cv;
                    }
                }

                if (bestGflops > 0) {
                    shapes.add(new CoarseSearchResult.ShapeResult(
                            mr, nr, bestGflops, bestMedian, bestCv, kcDesc.toString()));
                    System.out.printf(Locale.ROOT, "%.1f GFLOP/s%n", bestGflops);
                } else {
                    System.out.println("skipped (no admissible configuration)");
                }
            }
        }

        return CoarseSearchResult.build(shapes, MAX_SELECTED);
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /**
     * Find the smallest legal NC >= ncStart that satisfies constraints.
     * Steps by NR to guarantee NC % NR == 0, regardless of range step.
     */
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

    /**
     * Find the smallest legal MC >= mcStart that satisfies constraints.
     * Steps by MR to guarantee MC % MR == 0, regardless of range step.
     */
    private static int findLegalMc(int mcStart, ParameterRange mcRange,
                                    int mr, int kc, int nc, ConstraintSet cs) {
        // Start from the first multiple of MR >= mcStart
        int mcFirst = ((mcStart + mr - 1) / mr) * mr;
        for (int mc = mcFirst; mc <= mcRange.max; mc += mr) {
            if (cs.isValidMcMr(mc, mr) && cs.isValidMcKcNcForL3(mc, kc, nc)) {
                return mc;
            }
        }
        return -1;
    }

    /**
     * Find the smallest legal kUnroll >= kuStart that satisfies KC divisibility.
     */
    private static int findLegalKUnroll(int kuStart, ParameterRange kuRange,
                                         int kc, ConstraintSet cs) {
        for (int ku = kuStart; ku <= kuRange.max; ku += kuRange.step) {
            if (cs.isValidKcKUnroll(kc, ku)) {
                return ku;
            }
        }
        return -1;
    }

    private static int roundToStep(int value, int step) {
        return ((value + step / 2) / step) * step;
    }

    /** Return distinct sorted values (removes duplicates). */
    private static int[] distinctSorted(int a, int b, int c) {
        java.util.TreeSet<Integer> set = new java.util.TreeSet<>();
        set.add(a);
        set.add(b);
        set.add(c);
        return set.stream().mapToInt(Integer::intValue).toArray();
    }
}
