package net.faulj.compiler.matrix.kernel;

/** Eligibility of one R3 kernel lowering. */
public enum KernelEligibility {
    /** The region has a verified representation in the current Kernel IR. */
    ELIGIBLE,
    /** The region is known not to fit the current Kernel IR/backend subset. */
    INELIGIBLE,
    /** Eligibility depends on facts that were not available to the lowerer. */
    UNKNOWN
}
