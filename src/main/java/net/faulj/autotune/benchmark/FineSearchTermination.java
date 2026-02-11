package net.faulj.autotune.benchmark;

/**
 * Reason why Phase 4B fine search stopped refining KC.
 *
 * <p>Recorded at the moment the search loop terminates, not inferred later.</p>
 */
public enum FineSearchTermination {

    /** KC refinement improvement fell below epsilon. */
    EPSILON_PLATEAU,

    /** KC hit the range bounds (min or max of the search space). */
    RANGE_EXHAUSTED,

    /** Best result had CV exceeding stability threshold. */
    UNSTABLE,

    /** Iteration or step-size budget was exhausted. */
    BUDGET_EXHAUSTED
}
