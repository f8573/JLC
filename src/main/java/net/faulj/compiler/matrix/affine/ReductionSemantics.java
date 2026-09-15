package net.faulj.compiler.matrix.affine;

/**
 * M2 metadata describing whether a reduction may be reassociated by a future
 * scheduler. M2 records the fact but performs no transformation.
 */
public enum ReductionSemantics {
    STRICT_ORDERED,
    REASSOCIATION_ELIGIBLE;

    public boolean allowsReassociation() {
        return this == REASSOCIATION_ELIGIBLE;
    }
}
