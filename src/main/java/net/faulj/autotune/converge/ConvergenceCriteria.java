package net.faulj.autotune.converge;

import java.io.Serializable;

/**
 * Convergence thresholds for Phase 5 analysis.
 *
 * All fields are configurable to allow reproducible decisions.
 */
public final class ConvergenceCriteria implements Serializable {

    /** Performance plateau threshold (fraction). Default: 1.5% */
    public final double epsPerf;

    /** Stability threshold (CV). Default: 3% */
    public final double epsCv;

    /** Minimum number of evaluated configurations before exhaustion can be declared. Default: 2 */
    public final int minEvaluatedConfigs;

    /** Maximum allowed refinement depth (optional). Default: 8 */
    public final int maxRefinementDepth;

    /** Maximum allowed refinement iterations (optional). Default: 64 */
    public final int maxRefinementIterations;

    public ConvergenceCriteria(double epsPerf, double epsCv, int minEvaluatedConfigs,
                               int maxRefinementDepth, int maxRefinementIterations) {
        this.epsPerf = epsPerf;
        this.epsCv = epsCv;
        this.minEvaluatedConfigs = minEvaluatedConfigs;
        this.maxRefinementDepth = maxRefinementDepth;
        this.maxRefinementIterations = maxRefinementIterations;
    }

    public ConvergenceCriteria() {
        this(0.015, 0.03, 2, 8, 64);
    }

    @Override
    public String toString() {
        return String.format("ConvergenceCriteria{epsPerf=%.4f, epsCv=%.4f, minConfigs=%d, maxDepth=%d, maxIters=%d}",
                epsPerf, epsCv, minEvaluatedConfigs, maxRefinementDepth, maxRefinementIterations);
    }
}
