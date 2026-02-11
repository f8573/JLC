package net.faulj.autotune.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Result of Phase 4A coarse search over (MR, NR) microkernel shapes.
 *
 * <p>Contains per-shape benchmark results, ranked by best GFLOP/s.
 * The top shapes are retained for Phase 4B fine search.</p>
 */
public final class CoarseSearchResult {

    /** All tested microkernel shapes, ranked by best GFLOP/s descending. */
    public final List<ShapeResult> rankedShapes;

    /** Top shapes selected for fine search (best 2-3). */
    public final List<ShapeResult> selectedShapes;

    public CoarseSearchResult(List<ShapeResult> rankedShapes,
                              List<ShapeResult> selectedShapes) {
        this.rankedShapes = Collections.unmodifiableList(rankedShapes);
        this.selectedShapes = Collections.unmodifiableList(selectedShapes);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Phase 4A — Coarse Search\n\n");

        sb.append("Tested microkernels:\n");
        for (ShapeResult s : rankedShapes) {
            sb.append(String.format(Locale.ROOT,
                    "  MR=%d, NR=%d  ->  %.1f GFLOP/s (median %.1f, CV=%.3f, KC={%s})\n",
                    s.mr, s.nr, s.bestGflops, s.medianGflops, s.cv, s.kcTested));
        }

        sb.append("\nSelected for fine search:\n");
        for (ShapeResult s : selectedShapes) {
            sb.append(String.format(Locale.ROOT,
                    "  (MR=%d, NR=%d)\n", s.mr, s.nr));
        }

        return sb.toString();
    }

    /**
     * Aggregate result for one (MR, NR) microkernel shape.
     * The best GFLOP/s across all KC test points is reported.
     */
    public static final class ShapeResult {
        public final int mr;
        public final int nr;
        public final double bestGflops;
        public final double medianGflops;
        public final double cv;
        public final String kcTested;

        public ShapeResult(int mr, int nr, double bestGflops,
                           double medianGflops, double cv, String kcTested) {
            this.mr = mr;
            this.nr = nr;
            this.bestGflops = bestGflops;
            this.medianGflops = medianGflops;
            this.cv = cv;
            this.kcTested = kcTested;
        }
    }

    /**
     * Build a CoarseSearchResult from a list of shape results.
     * Ranks by best GFLOP/s and selects top shapes.
     */
    public static CoarseSearchResult build(List<ShapeResult> shapes, int maxSelected) {
        List<ShapeResult> ranked = new ArrayList<>(shapes);
        ranked.sort(Comparator.comparingDouble((ShapeResult s) -> s.bestGflops).reversed());

        int count = Math.min(maxSelected, ranked.size());
        List<ShapeResult> selected = new ArrayList<>(ranked.subList(0, count));

        return new CoarseSearchResult(ranked, selected);
    }
}
